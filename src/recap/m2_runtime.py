"""Label-blind, host-orchestrated M2 runner. Never reads targets or pair metadata."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import httpx
from pathlib import Path

from .boundary import EvidenceBundle
from .contracts import CapabilityProfile, CaseInput, GenerationOptions, Prediction
from .h0 import AuditedHTTP, append, digest, now, request_for, prediction_from_response
from .io import canonical, read_jsonl, sha256, write_json
from .m2_contracts import VisualRecord, LinkedRecord, ReviewRecord, FinalRecord, validate_links, execute_calculations, record_ids
from .m2_prompts import PROMPTS
from .provider import ChatCompletionsAdapter, Message, ModelRequest
from .scoring import freeze_predictions

ARMS=('H0','H_repeat','H1')
LIMITS={'fixture':20,'development':480,'validation':700}

class TokenizerPreflightFailure(RuntimeError):
    """An exhausted/invalid tokenizer request, with safe non-content evidence."""
    def __init__(self, evidence):
        super().__init__('tokenizer preflight unavailable; no generation sent')
        self.evidence=evidence

class M2Budget:
    def __init__(self,path):
        self.path=path;path.parent.mkdir(parents=True,exist_ok=True)
    def reserve(self,stage,key):
        if stage not in LIMITS:raise ValueError('invalid budget phase')
        with self.path.open('a+') as h:
            fcntl.flock(h,fcntl.LOCK_EX);h.seek(0);text=h.read()
            state=json.loads(text) if text else {'attempts':0,'requests':{}}
            requests=state['requests'];entry=requests.get(key)
            if entry and entry['stage']!=stage:raise ValueError('request phase collision')
            if state['attempts']>=1450 or (entry and entry['attempts']>=2):raise RuntimeError('M2 attempt limit')
            if entry is None:
                if len(requests)>=1200 or sum(x['stage']==stage for x in requests.values())>=LIMITS[stage]:raise RuntimeError('M2 unique request limit')
                entry={'stage':stage,'attempts':0};requests[key]=entry
            entry['attempts']+=1;state['attempts']+=1
            h.seek(0);h.truncate();h.write(json.dumps(state,sort_keys=True));h.flush();os.fsync(h.fileno())
            return state['attempts'],entry['attempts']


def schema_for(arm,stage,base_schema,*,sources=None,parents=()):
    if arm=='H0' or (arm=='H_repeat' and stage==1):return base_schema
    cls=VisualRecord if stage==1 else LinkedRecord if arm=='H1' and stage==2 else ReviewRecord if stage==2 else FinalRecord
    schema=cls.model_json_schema();defs=schema.get('$defs',{})
    if sources is not None and 'SourceRef' in defs:
        defs['SourceRef']['properties']['source_id']['enum']=sorted(sources)
    fields={'Observation':('obs',16 if stage==1 else 12),'Binding':('binding',8),
            'Conflict':('conflict',6),'Revision':('revision',6),'CalculationRequest':('calc',3)}
    for name,(prefix,count) in fields.items():
        if name in defs:defs[name]['properties']['record_id']['enum']=[f's{stage}_{prefix}_{i}' for i in range(1,count+1)]
    prior={i for p in parents for i in record_ids(p['parsed'])}
    if 'Binding' in defs:
        obs={o['record_id'] for p in parents for o in p['parsed'].get('observations',[])}
        defs['Binding']['properties']['observation_ids']['items']['enum']=sorted(obs|{f's2_obs_{i}' for i in range(1,13)})
    if 'Revision' in defs and prior:defs['Revision']['properties']['replaces_record_id']['enum']=sorted(prior)
    if cls is ReviewRecord:schema['properties']['record_id']['const']='s2_review'
    if cls is FinalRecord:
        schema['properties']['evidence_refs']['items']=base_schema['properties']['evidence_refs']['items']
        if prior:schema['properties']['used_record_ids']['items']['enum']=sorted(prior)
    return schema


def stage_request(case,bundle,options,arm,stage,parents,*,protocol_sha256,candidate):
    if arm not in ARMS or stage not in range(1, (2 if arm=='H0' else 4)):raise ValueError('invalid stage')
    if len(parents)!=stage-1:raise ValueError('missing stage parents')
    for i,p in enumerate(parents,1):
        if any(p.get(k)!=v for k,v in {'input_id':case.input_id,'arm':arm,'stage':i,
                'protocol_sha256':protocol_sha256,'candidate':candidate,'status':'ok'}.items()):
            raise ValueError('parent outside case/arm/candidate/protocol or failed')
    base,info=request_for(case,bundle,options)
    images=base.messages[1].images
    sources={x['asset_id']:x['sha256'] for x in info['images']}
    if not (arm=='H1' and stage==1):
        for e in case.evidence:sources[e.evidence_id]=sources[e.asset_id]
        for k in ['claim','caption','context']:
            if getattr(case,k) is not None:sources[k]=hashlib.sha256(getattr(case,k).encode()).hexdigest()
    prompt=PROMPTS[arm][stage];schema=schema_for(arm,stage,base.output_schema,sources=sources,parents=parents)
    if arm=='H0' or (arm=='H_repeat' and stage==1):request=base
    else:
        # The visual stage includes only image references: no CaseInput serialization.
        data={'original':{'image_sources':list(sources)} if stage==1 else bundle.observation(case),
              'source_hashes':sources}
        if parents:
            data['previous_records']=[{'stage':p['stage'],'record':p['parsed'],
                'calculations':p['calculations']} for p in parents]
        request=ModelRequest(case.input_id,(Message('system',prompt),
            Message('user',canonical(data).decode(),images)),options,schema)
    metadata={**info,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'schema_sha256':digest(schema),'sources':sources,'arm':arm,'stage':stage,'candidate':candidate,
        'protocol_sha256':protocol_sha256,'parent_output_sha256':[digest(p) for p in parents]}
    return request,metadata


def preflight(request,max_context,tokenizer=None):
    if tokenizer is not None:
        tokens=tokenizer(request)
        if tokens["max_model_len"]!=max_context:raise RuntimeError("tokenizer context binding drift")
        total=tokens["count"]+request.options.output_budget+64
        if total>max_context:raise ValueError("preflight_context_limit")
        return {"method":"serving_tokenizer_original_messages",**tokens,"output_budget":request.options.output_budget,"margin_tokens":64,"projected_total":total,"context":max_context,"truncation":False,"scope":"same existing model/template/processor; fixture count matched generation usage"}

    # UTF-8 bytes upper-bound text tokens under the existing byte-backed tokenizer;
    # extra image/template allowances are an engineering guard, not measured tokenization.
    text_bytes=sum(len(m.text.encode()) for m in request.messages)
    schema_bytes=len(canonical(request.output_schema))
    images=sum(len(m.images) for m in request.messages)
    estimate=text_bytes+schema_bytes+images*1120+1024+request.options.output_budget
    result={'method':'utf8_bytes_plus_schema_image_template_allowance','text_bytes':text_bytes,
        'schema_bytes':schema_bytes,'image_allowance':images*1120,'template_allowance':1024,
        'output_budget':request.options.output_budget,'estimated_upper_bound':estimate,
        'context':max_context,'exact_tokenization_verified':False,'truncation':False}
    if estimate>max_context:raise ValueError('preflight_context_limit')
    return result


def stage_key(metadata,*,profile,model,code):
    return digest({'metadata':metadata,'profile_sha256':digest(profile),'model_sha256':digest(model),'code_sha256':code})


def validate_output(case,arm,stage,response,attempts,sources,parents):
    if arm=='H0' or (arm=='H_repeat' and stage==1):
        prediction=prediction_from_response(case,response,attempts)
        value=json.loads(response.text) if prediction.status=='ok' else None
        # Draft record is host-labelled; keep the exact returned final JSON separately.
        if value is not None and arm=='H_repeat':value={'record_id':'draft',**value}
        return prediction.status,prediction.diagnostic,value,prediction
    if any(x.get('binding_error') for x in attempts):return 'failed','served_model_alias_changed',None,None
    if response.status!='ok':return response.status,'output_'+response.status,None,None
    cls=VisualRecord if stage==1 else LinkedRecord if arm=='H1' and stage==2 else ReviewRecord if stage==2 else FinalRecord
    try:
        value=cls.model_validate_json(response.text,strict=True).model_dump()
        validate_links(value,sources,[p['parsed'] for p in parents],visual=stage==1)
        if stage==3 and not set(value['evidence_refs'])<={e.evidence_id for e in case.evidence}:raise ValueError('invalid evidence reference')
    except (ValueError,TypeError,KeyError):return 'invalid','invalid_stage_schema_or_reference',None,None
    prediction=Prediction(input_id=case.input_id,task_kind=case.task_kind,answer=value['answer'],
        evidence_refs=value['evidence_refs'],status='ok') if stage==3 else None
    return 'ok',None,value,prediction


def read_sealed(path):
    seal=json.loads(path.with_name(path.name+'.seal.json').read_text())
    if sha256(path)!=seal['sha256']:raise ValueError('stage seal mismatch')
    return json.loads(path.read_text())


def run_stage(*,case,bundle,arm,stage,parents,protocol,protocol_hash,output,credentials,budget_path,
              identity_guard=None,client=None,execution_kind='live',tokenizer=None):
    spec=protocol['profile'];candidate=protocol['candidates'][arm]
    request,info=stage_request(case,bundle,GenerationOptions.model_validate(spec['options']),arm,stage,parents,
                             protocol_sha256=protocol_hash,candidate=candidate)
    key=stage_key(info,profile=spec,model=protocol['model_binding'],code=protocol['source_tree_sha256'])
    base=output/arm/'cases'/case.input_id;result_path=base/f'stage_{stage}.json';attempt_path=base/f'attempts_{stage}.jsonl'
    if result_path.exists():
        result=read_sealed(result_path)
        if result['cache_key']!=key:raise ValueError('stage cache identity drift')
        if result.get('attempts_sha256') and sha256(attempt_path)!=result['attempts_sha256']:raise ValueError('attempt log changed')
        return result
    if attempt_path.exists():raise RuntimeError('unfinished attempted stage; inspect before resume, no automatic repair')
    prediction=None;parsed=None;calculations=[];response=None;recorder=None;guard_result=None
    if identity_guard:identity_guard()
    try:guard_result=preflight(request,protocol['model_binding']['max_context'],tokenizer)
    except TokenizerPreflightFailure as exc:
        status='failed';diagnostic='preflight_tokenizer_unavailable';guard_result=exc.evidence
    except ValueError:
        status='failed';diagnostic='preflight_context_limit'
    else:
        recorder=AuditedHTTP(credentials,attempt_path,M2Budget(budget_path),stage=protocol['phase'],
            request_key=key,expected_model=credentials.served_model,timeout=protocol['timeout_seconds'],client=client)
        try:
            adapter=ChatCompletionsAdapter(CapabilityProfile.model_validate(spec['capability']),credentials,
                client=recorder,allow_live=True,max_retries=1)
            response=adapter.generate(request)
            status,diagnostic,parsed,prediction=validate_output(case,arm,stage,response,recorder.records,info['sources'],parents)
            if status=='ok' and stage==2:calculations=execute_calculations(parsed['calculations'],info['sources'])
        finally:
            if client is None:recorder.client.close()
    result={'input_id':case.input_id,**info,'cache_key':key,'status':status,'diagnostic':diagnostic,
        'parsed':parsed,'calculations':calculations,'prediction':prediction.model_dump() if prediction else None,
        'preflight':guard_result,'execution_kind':execution_kind,'source_tree_sha256':protocol['source_tree_sha256'],
        'profile_sha256':digest(spec),'model_sha256':digest(protocol['model_binding']),
        'final_content':recorder.redact(response.text) if response else None,
        'requested':request.options.model_dump(),'transmitted':response.transmitted if response else None,
        'observed':response.observed if response else None,'finish_reason':response.finish_reason if response else None,
        'attempts_sha256':sha256(attempt_path) if attempt_path.exists() else None,
        'attempt_count':len(recorder.records) if recorder else 0,'retries':response.retries if response else 0,
        'first_attempt_valid':bool(status=='ok' and recorder and len(recorder.records)==1),'at':now()}
    write_json(result_path,result,immutable=True)
    write_json(result_path.with_name(result_path.name+'.seal.json'),{'sha256':sha256(result_path)},immutable=True)
    return result


def verify_protocol(path,root):
    protocol=read_sealed(path)
    if digest(protocol['source_files'])!=protocol['source_tree_sha256']:raise ValueError('bad source aggregate')
    for name,expected in protocol['source_files'].items():
        if sha256(root/name)!=expected:raise ValueError('source drift: '+name)
    if digest({a:{str(k):v for k,v in stages.items()} for a,stages in PROMPTS.items()})!=protocol['prompts_sha256']:
        raise ValueError('prompt drift')
    return protocol


def seal_run(output,protocol,protocol_hash,*,execution_kind):
    for arm in protocol['arms']:
        predictions=[]
        for case_id in protocol['case_order']:
            path=output/arm/'cases'/case_id/'result.json'
            if not path.exists():raise ValueError('cannot seal unfinished arm')
            result=read_sealed(path)
            predictions.append(Prediction.model_validate(result['prediction']))
        dest=output/arm/'predictions.jsonl'
        if not dest.exists():freeze_predictions(dest,predictions,execution_kind=execution_kind)
        elif read_jsonl(dest)!=[p.model_dump() for p in predictions]:raise ValueError('prediction differs')
    files={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='all_arms.seal.json'}
    seal={'protocol_sha256':protocol_hash,'files':files,'sha256':digest(files),'arms':protocol['arms'],
          'case_count':len(protocol['case_order']),'execution_kind':execution_kind}
    write_json(output/'all_arms.seal.json',seal,immutable=True)
    return seal


def run_experiment(*,root,protocol_path,bundle_path,output,credentials,budget_path,current_binding,
                   allow_live=False,identity_guard=None,client=None,execution_kind='live',tokenizer=None):
    if not allow_live:return {'status':'NOT_RUN','reason':'explicit opt-in required'}
    if execution_kind not in {'live','mock_contract'}:raise ValueError('invalid execution kind')
    protocol=verify_protocol(protocol_path,root);ph=sha256(protocol_path)
    if current_binding!=protocol['model_binding'] or credentials.served_model!=current_binding['served_alias']:raise ValueError('binding drift')
    if sha256(bundle_path/'cases.jsonl')!=protocol['cases_sha256'] or sha256(bundle_path/'assets.jsonl')!=protocol['assets_sha256']:raise ValueError('bundle drift')
    bundle=EvidenceBundle(bundle_path)
    cases={c.input_id:c for c in (CaseInput.model_validate(x) for x in read_jsonl(bundle_path/'cases.jsonl'))}
    if set(cases)!=set(protocol['case_order']) or len(protocol['case_order'])!=len(cases):raise ValueError('case schedule mismatch')
    expected={(i,a) for i in cases for a in protocol['arms']}
    schedule=[(x['input_id'],x['arm']) for x in protocol['schedule']]
    if len(schedule)!=len(expected) or set(schedule)!=expected:raise ValueError('arm schedule mismatch')
    write_json(output/'run.json',{'protocol_sha256':ph,'execution_kind':execution_kind,'phase':protocol['phase']},immutable=True)
    count=0
    for cid,arm in schedule:
        case=cases[cid];parents=[];prediction=None
        for stage in range(1,2 if arm=='H0' else 4):
            result=run_stage(case=case,bundle=bundle,arm=arm,stage=stage,parents=parents,protocol=protocol,
                protocol_hash=ph,output=output,credentials=credentials,budget_path=budget_path,
                identity_guard=identity_guard,client=client,execution_kind=execution_kind,tokenizer=tokenizer)
            parents.append(result)
            if result['status']!='ok':
                prediction=Prediction(input_id=cid,task_kind=case.task_kind,status=result['status'],diagnostic=f'stage_{stage}:'+str(result['diagnostic']))
                break
            if result['prediction']:prediction=Prediction.model_validate(result['prediction'])
        assert prediction is not None
        # An earlier draft is never substituted for a failed later stage.
        summary={'input_id':cid,'arm':arm,'protocol_sha256':ph,'stage_hashes':[digest(x) for x in parents],
            'prediction':prediction.model_dump(),'first_attempt_valid':all(x['first_attempt_valid'] for x in parents)}
        dest=output/arm/'cases'/cid/'result.json';write_json(dest,summary,immutable=True)
        write_json(dest.with_name(dest.name+'.seal.json'),{'sha256':sha256(dest)},immutable=True)
        count+=1
        print(json.dumps({'phase':protocol['phase'],'completed_case_arms':count,'scheduled_case_arms':len(schedule),
                         'arm':arm,'input_id':cid,'status':prediction.status}),flush=True)
        if any(p.get('diagnostic')=='served_model_alias_changed' for p in parents):raise RuntimeError('binding drift stop')
    seal=seal_run(output,protocol,ph,execution_kind=execution_kind)
    return {'status':'SEALED','case_arms':len(schedule),'all_arms_seal_sha256':sha256(output/'all_arms.seal.json')}


def tokenize_request(request,profile,credentials,client=None):
    """Existing serving tokenizer only. No completion, tool, or setting mutation."""
    payload=ChatCompletionsAdapter(profile,credentials).build_payload(request)
    body={k:payload[k] for k in ['model','messages','chat_template_kwargs'] if k in payload}
    body.update(add_generation_prompt=True,return_token_strs=False)
    owned=client is None;client=client or httpx.Client(timeout=30)
    events=[]
    def unavailable():
        return TokenizerPreflightFailure({'method':'serving_tokenizer_original_messages',
            'status':'unavailable','count':None,'count_not_observed':True,
            'tokenization_attempts':len(events),'tokenization_request_sha256':digest(body),
            'attempt_outcomes':events,'generation_requests':0,'truncation':False,
            'failure_record_origin':'runtime_tokenizer_exception'})
    try:
        for attempt in range(2):
            try:
                response=client.post(credentials.base_url.removesuffix('/v1').rstrip('/')+'/tokenize',json=body,
                    headers={'Authorization':'Bearer '+credentials.api_key})
                events.append({'attempt':attempt+1,'http_status':response.status_code})
                if response.status_code in {429,500,502,503,504} and attempt==0:continue
                response.raise_for_status();value=response.json()
                if type(value.get('count')) is not int or value['count']<0 or type(value.get('max_model_len')) is not int:raise ValueError('invalid tokenizer result')
                return {'count':value['count'],'max_model_len':value['max_model_len'],
                    'tokenization_attempts':attempt+1,'tokenization_request_sha256':digest(body),'generation_requests':0}
            except httpx.TransportError as exc:
                events.append({'attempt':attempt+1,'transport_error':type(exc).__name__})
                if attempt==0:continue
                raise unavailable() from None
            except (httpx.HTTPStatusError,ValueError,KeyError,TypeError) as exc:
                events[-1]['validation_error']=type(exc).__name__
                raise unavailable() from None
    finally:
        if owned:client.close()
