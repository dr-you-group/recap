"""Read/reuse only the recorded RECAP Pod; no provisioning or environment changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import httpx

from recap.h0 import now
from recap.io import write_json
from recap.provider import load_credentials

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "serve-llm"
sys.path.insert(0, str(INFRA))
from runpod.common import read_env


def context():
    deployment = json.loads((INFRA / ".secrets/deployment.json").read_text())
    recorded = json.loads((INFRA / "runpod/deployment.json").read_text())
    if any(deployment.get(k) != recorded.get(k) for k in ["pod_id", "volume_id"]):
        raise ValueError("private/public dedicated Pod record differs")
    env = {**os.environ, **read_env(INFRA / ".secrets/runpod.env"), **read_env(INFRA / ".secrets/llm.env")}
    return deployment, env


def cli(deployment, env, *args):
    p = subprocess.run([str(INFRA / ".tools/runpodctl"), *args], env=env, capture_output=True, text=True, timeout=90)
    if p.returncode:
        try:
            error = json.loads(p.stderr.strip())
        except ValueError:
            error = {}
        raise RuntimeError("Runpod read/control failed: " + str(error.get("code", "unparsed_error")))
    return json.loads(p.stdout)


def pod_state(deployment, env):
    p = cli(deployment, env, "pod", "get", deployment["pod_id"])
    if p.get("id") != deployment["pod_id"] or p.get("networkVolumeId") != deployment["volume_id"] or p.get("name") != deployment.get("name", "recap-gemma4-31b"):
        raise ValueError("dedicated Pod identity/volume/name mismatch")
    return {k: p.get(k) for k in ["id", "name", "desiredStatus", "runtimeStatus", "runtimeStatusReason", "gpuCount", "lastStatusChange"]}


def public_state(state):
    return {k: v for k, v in state.items() if k != "id"}


def safe_control_output(text, deployment, env):
    for key,value in env.items():
        if value and any(word in key for word in ['KEY','TOKEN','PASSWORD','URL','DOMAIN']):
            text=text.replace(value,'[REDACTED]')
    for key in ['pod_id','volume_id']:
        text=text.replace(deployment[key],'['+key.upper()+']')
    return text


def endpoint_status():
    c = load_credentials(INFRA / ".secrets/llm.env")
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(c.base_url.removesuffix("/v1").rstrip("/") + "/v1/models", headers={"Authorization": "Bearer " + c.api_key})
        data = response.json() if response.status_code == 200 else {}
        models = [{k: m.get(k) for k in ["id", "root", "max_model_len"]} for m in data.get("data", [])]
        return {"http_status": response.status_code, "models": models, "ready": any(m['id']==c.served_model for m in models)}
    except (httpx.HTTPError, ValueError) as error:
        return {"ready": False, "error_type": type(error).__name__}


def remote_observation(deployment, env, *, known_hosts_path=None, strict_host_checking="accept-new"):
    info = cli(deployment, env, "ssh", "info", deployment["pod_id"])
    known = known_hosts_path or ROOT / "artifacts/private/m1/known_hosts"
    if known_hosts_path is not None and not known.is_file():
        raise ValueError("explicit known-hosts file must already exist")
    if not known.exists():
        shutil.copyfile(INFRA / ".secrets/runpod_known_hosts", known)
    script = '''import json,hashlib,pathlib,subprocess,importlib.metadata,re
r=pathlib.Path('/workspace/serve')
values={}
for line in (r/'settings.env').read_text().splitlines():
 if '=' in line and not line.startswith('#'):
  k,v=line.split('=',1); values[k]=v.strip().strip('"').strip("'")
keep=['MODEL_ID','MODEL_REVISION','SERVED_MODEL_NAME','MAX_MODEL_LEN','MAX_NUM_SEQS','MAX_IMAGES_PER_PROMPT','VISION_MAX_SOFT_TOKENS','JSON_DISABLE_ANY_WHITESPACE','ENFORCE_EAGER','GPU_MEMORY_UTILIZATION']
settings={k:values.get(k) for k in keep}
model=values['MODEL_ID']; revision=values['MODEL_REVISION']
snapshot=pathlib.Path('/workspace/hf-cache/hub')/('models--'+model.replace('/','--'))/'snapshots'/revision
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={p.name:{'sha256':h(p),'bytes':p.stat().st_size} for p in snapshot.iterdir() if p.is_file() and p.suffix in ['.json','.jinja']}
index=json.loads((snapshot/'model.safetensors.index.json').read_text())
weights=sorted(set(index['weight_map'].values()))
inventory=[{'name':n,'exists':(snapshot/n).is_file(),'bytes':(snapshot/n).stat().st_size if (snapshot/n).is_file() else None,'cache_blob_identity':(snapshot/n).resolve().name} for n in weights]
template_path=snapshot/'chat_template.jinja'
template=template_path.read_text() if template_path.exists() else json.loads((snapshot/'tokenizer_config.json').read_text()).get('chat_template','')
if not isinstance(template,str): template=json.dumps(template,sort_keys=True)
related=[line.strip() for line in template.splitlines() if any(k in line for k in ['enable_thinking','thought','thinking'])]
processes=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().decode().split('\\0')
  if any('/vllm' in a for a in args[:2]) and 'serve' in args:
   allowed=['--revision','--served-model-name','--dtype','--max-model-len','--max-num-seqs','--limit-mm-per-prompt','--mm-processor-kwargs','--structured-outputs-config','--reasoning-parser','--kv-cache-dtype','--quantization']
   processes.append({'pid':p.parent.name,'options':{key:args[args.index(key)+1] for key in allowed if key in args}})
 except (OSError,UnicodeError,IndexError):pass
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True)
print(json.dumps({'settings':settings,'files':files,'weights':inventory,'runtime_ready':(r/'.runtime-ready').is_file(),'engine_version':importlib.metadata.version('vllm'),'torch_version':importlib.metadata.version('torch'),'template_sha256':hashlib.sha256(template.encode()).hexdigest(),'template_thinking_lines':related,'vllm_processes':processes,'gpu_sample':gpu.stdout.strip(),'sample_note':'single polling sample, not peak','script_hashes':{n:h(r/n) for n in ['boot.sh','run-vllm.sh','settings.env']}}))
'''
    command = "/workspace/vllm-env/bin/python - <<'RECAP_M1_READ'\n" + script + "\nRECAP_M1_READ"
    arguments = ["ssh", "-i", str(INFRA / ".secrets/runpod_ssh"), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                 "-o", "StrictHostKeyChecking=" + strict_host_checking, "-o", "UserKnownHostsFile=" + str(known),
                 "-o", "ConnectTimeout=20", "-p", str(info["port"]), "root@" + info["ip"], command]
    p = subprocess.run(arguments, capture_output=True, text=True, timeout=90)
    if p.returncode:
        raise RuntimeError("remote observation failed; exit=" + str(p.returncode))
    return json.loads(p.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','start','observe','stop'])
    parser.add_argument('--target',choices=['original','us'],default='original')
    args=parser.parse_args()
    if args.target=='us' and args.action not in {'status','observe'}:
        parser.error('US target supports read-only status/observe here; original control scripts must never operate on the US target')
    deployment,env=context();path=ROOT/'artifacts/m1/resource_ledger.json'
    if args.target=='us':
        authorization=json.loads(path.read_text()).get('migration_authorization',{})
        target=json.loads((ROOT/'artifacts/private/m1/us_deployment.json').read_text())
        if not authorization.get('new_pod_and_volume') or target['region'] not in authorization.get('target_regions',[]):
            raise ValueError('US migration target is not authorized')
        deployment={**target,'name':'recap-gemma4-31b-m1-us'}
        path=ROOT/'artifacts/m1/us_resource_ledger.json'
    ledger=json.loads(path.read_text());state=pod_state(deployment,env)
    if args.target=='us':
        if hashlib.sha256(deployment['volume_id'].encode()).hexdigest()!=ledger['volume_fingerprint']:
            raise ValueError('US volume target changed')
        ledger['target_fingerprint']=ledger['pod_fingerprint']
        ledger['started_by_m1']=True
    if hashlib.sha256(deployment['pod_id'].encode()).hexdigest()!=ledger['target_fingerprint']:
        raise ValueError('resource target changed')
    ledger.setdefault('events',[])
    control_exit=0
    if args.action in {'start','stop'}:
        if args.action=='start' and state['runtimeStatus']=='stopped':
            ledger['events'].append({'at':now(),'action':'start_requested','state':public_state(state)})
            ledger['start_attempted_by_m1']=True
            ledger['cleanup_obligation']='pending_state_resolution'
            write_json(path,ledger)
            p=subprocess.run([str(INFRA/'start.sh'),'--no-wait','--start-attempts','1'],capture_output=True,text=True,timeout=120)
            control_exit=p.returncode
            after=pod_state(deployment,env)
            ledger['started_by_m1']=after['desiredStatus']=='RUNNING' and 'already running' not in p.stdout
            if 'already running' in p.stdout:ledger['start_attempted_by_m1']=False
            if ledger.get('shutdown_policy')=='keep_running_user_request':
                ledger['cleanup_obligation']='leave_running_per_user_instruction' if after['desiredStatus']=='RUNNING' else 'none_confirmed_stopped' if after['runtimeStatus']=='stopped' else 'pending_state_resolution'
                ledger['cleanup_verified']=None
            else:
                ledger['cleanup_obligation']='stop_same_pod' if ledger['started_by_m1'] else 'none_confirmed_stopped' if after['runtimeStatus']=='stopped' else 'pending_state_resolution'
            ledger['control_actions'].append({'at':now(),'action':'start','exit_code':p.returncode,'after':public_state(after),
                'stdout':safe_control_output(p.stdout,deployment,env),'stderr':safe_control_output(p.stderr,deployment,env)})
            ledger['events'].append({'at':now(),'action':'start_result','exit_code':p.returncode,'state':public_state(after)})
        elif args.action=='stop' and ledger.get('shutdown_policy')=='keep_running_user_request':
            ledger['events'].append({'at':now(),'action':'automatic_stop_suppressed','reason':'latest user instruction is to keep Pod running'})
            ledger['cleanup_verified']=None
        elif args.action=='stop' and (ledger.get('started_by_m1') or ledger.get('cleanup_obligation')=='pending_state_resolution'):
            p=subprocess.run([str(INFRA/'stop.sh')],capture_output=True,text=True,timeout=120)
            control_exit=p.returncode
            after=pod_state(deployment,env)
            ledger['control_actions'].append({'at':now(),'action':'stop','exit_code':p.returncode,'after':public_state(after)})
            ledger['cleanup_verified']=after['runtimeStatus']=='stopped'
        else:
            ledger['events'].append({'at':now(),'action':args.action+'_not_needed','state':public_state(state)})
    if args.action=='observe':
        if state['runtimeStatus']!='running':raise RuntimeError('remote observation requires running container')
        observation=remote_observation(deployment,env)
        observation_path=ROOT/'artifacts/m1/server_observations'/(now().replace(':','_')+'.json')
        write_json(observation_path,observation,immutable=True)
        from recap.h0_admin import derive_binding
        endpoint=endpoint_status()
        binding=derive_binding(observation,endpoint.get('models',[]))
        write_json(ROOT/'artifacts/m1/model_binding.json',binding,immutable=True)
        write_json(ROOT/'artifacts/m1/server_observation.json',observation)
        ledger['events'].append({'at':now(),'action':'server_observed','artifact':str(observation_path.relative_to(ROOT))})
        print(json.dumps(observation,indent=2))
    state=pod_state(deployment,env)
    endpoint=endpoint_status()
    ledger['latest']={'at':now(),'pod':public_state(state),'endpoint':endpoint}
    if args.action=='stop' and ledger.get('started_by_m1') and ledger.get('shutdown_policy')!='keep_running_user_request':
        ledger['cleanup_verified']=state['runtimeStatus']=='stopped'
    if ledger.get('shutdown_policy')=='keep_running_user_request':
        ledger['latest']['keep_running_verified']=state['runtimeStatus']=='running'
    write_json(path,ledger)
    print(json.dumps({'latest':ledger['latest'],'started_by_m1':ledger['started_by_m1'],'cleanup_verified':ledger.get('cleanup_verified')}))
    if control_exit:raise SystemExit(control_exit)


if __name__=='__main__':main()
