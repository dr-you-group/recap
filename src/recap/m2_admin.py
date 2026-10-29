"""Preparation/freeze and post-seal scoring. This module is not a predictor dependency."""
import itertools
import json
from pathlib import Path
import random
from collections import Counter,defaultdict
import numpy as np

from .contracts import EvaluationTarget
from .h0 import digest, now
from .h0_admin import source_snapshot
from .io import read_jsonl,sha256,write_json,write_bytes
from .m2_prompts import PROMPTS
from .m2_runtime import read_sealed,ARMS,LIMITS
from .scoring import score_files

SEED=20261014

def freeze(root,dest,*,phase,bundle_path,arms,candidates,experiment=None,candidate_history=None):
    if phase not in LIMITS or not arms or not set(arms)<=set(ARMS):raise ValueError('invalid phase/arms')
    if phase=='validation' and set(arms)!=set(ARMS):raise ValueError('all validation arms required')
    cases=read_jsonl(bundle_path/'cases.jsonl');order=sorted(c['input_id'] for c in cases)
    if len(order)!=len(set(order)):raise ValueError('duplicate inputs')
    if len(order)>({'fixture':2,'development':40,'validation':100}[phase]):raise ValueError('case cap')
    rng=random.Random(SEED);rng.shuffle(order)
    perms=list(itertools.permutations(arms));rng.shuffle(perms)
    schedule=[{'input_id':cid,'arm':arm} for i,cid in enumerate(order) for arm in perms[i%len(perms)]]
    snapshot=source_snapshot(root)
    binding=json.loads((root/'artifacts/m2/model_binding.json').read_text())
    profile=json.loads((root/'artifacts/m1/verified_profiles.json').read_text())['profiles']['primary']
    p={'phase':phase,'arms':list(arms),'candidates':candidates,'case_order':order,'schedule':schedule,
       'profile':profile,'model_binding':binding,'timeout_seconds':300,'transport_retries':1,
       'image_policy':'original_bytes_no_resize','server_processor_max_soft_tokens':560,
       'case_count':len(cases),'cases_sha256':sha256(bundle_path/'cases.jsonl'),
       'assets_sha256':sha256(bundle_path/'assets.jsonl'),'dataset_source':json.loads((bundle_path/'source.json').read_text()),
       'source_files':snapshot['files'],'source_tree_sha256':snapshot['sha256'],'tested_commit':None,
       'prompts_sha256':digest({a:{str(k):v for k,v in s.items()} for a,s in PROMPTS.items()}),
       'prompts':{a:{str(k):v for k,v in s.items()} for a,s in PROMPTS.items()},
       'limits':{'unique_by_phase':LIMITS,'unique_total':1200,'attempts_total':1450},
       'selection_rule':['development_pair_accuracy','development_accuracy','earlier_stable_candidate'],
       'failure_policy':'no content repair, no invalid/truncated retry; failed stage ends case-arm; full planned denominators',
       'context_policy':'Existing /tokenize on exact original messages/template kwargs + output + 64-token margin before generation; model context must match. GPU-free tests use conservative byte bound. No silent truncation.',
       'cache_scope':'case/arm/candidate/stage/parents/input/image/prompt/schema/profile/model/source/protocol hashes',
       'calculation_policy':'shared check_reported_percentage only, source-scoped model premises, conditional results; no native tools',
       'target_access':'only after every arm and all traces/attempts/calculations are sealed',
       'bootstrap':{'unit':'work_family','resamples':10000,'seed':SEED,'interpretation':'exploratory'},
       'primary':'pair_accuracy(H1)-pair_accuracy(H_repeat)',
       'secondary':['H1-H0','accuracy','macro-F1','confusions','case_pair_paired_tables','coverage_first_attempt_valid'],
       'posthoc_selection':{'seed':SEED,'cap_cases':20,'rule':'reserve up to 2 persistent errors and 2 persistent successes; include all discordant cases if remaining capacity allows, otherwise seeded stratified improvement/regression sample; total cap20' },
       'candidate_history':candidate_history,'frozen_at':now()}
    if phase!='fixture':
        data_dir=bundle_path.parent;p['targets_sha256']=sha256(data_dir/'scoring/targets.jsonl')
        p['manifest_sha256']=sha256(data_dir/'manifest.jsonl')
    if phase=='validation':
        if experiment is None or candidate_history is None:raise ValueError('validation requires prior declaration and selection history')
        p['experiment_sha256']=sha256(experiment)
        p['split_audit_sha256']=sha256(root/'artifacts/private/m2/validation_selection.json')
        p['exposure_sha256']=sha256(root/'artifacts/private/m2/exposure_ledger.json')
        p['rights_provenance_sha256']=sha256(bundle_path.parent/'assets.jsonl')
        p['sample_summary']=json.loads((root/'artifacts/m2/validation_data_summary.json').read_text())
        if p['sample_summary']['mode']=='insufficient':raise ValueError('independent data below minimum')
    write_json(dest,p,immutable=True);write_json(dest.with_name(dest.name+'.seal.json'),{'sha256':sha256(dest)},immutable=True)
    snapshot_dir=dest.parent/(dest.stem+'_source')
    for rel in snapshot['files']:write_bytes(snapshot_dir/rel,(root/rel).read_bytes(),immutable=True)
    return p


def verify_all_arms(output,protocol_path):
    p=read_sealed(protocol_path)
    seal=json.loads((output/'all_arms.seal.json').read_text())
    if seal['protocol_sha256']!=sha256(protocol_path) or seal['arms']!=p['arms'] or seal['case_count']!=p['case_count']:raise ValueError('seal/protocol mismatch')
    if digest(seal['files'])!=seal['sha256']:raise ValueError('bad seal aggregate')
    actual={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file() and f.name!='all_arms.seal.json'}
    if actual!=set(seal['files']):raise ValueError('unsealed or missing run files')
    for rel,expected in seal['files'].items():
        if sha256(output/rel)!=expected:raise ValueError('sealed run modified')
    for arm in p['arms']:
        if f'{arm}/predictions.jsonl' not in seal['files']:raise ValueError('arm missing from seal')
    return p,seal


def score(output,protocol_path,target_path,dest):
    # Do not even open target_path until all arms and traces have passed the gate.
    p,seal=verify_all_arms(output,protocol_path)
    if sha256(target_path)!=p['targets_sha256']:raise ValueError('target drift')
    targets=[EvaluationTarget.model_validate(x) for x in read_jsonl(target_path)]
    if {t.input_id for t in targets}!=set(p['case_order']):raise ValueError('target denominator mismatch')
    rows={a:{r['input_id']:r for r in read_jsonl(output/a/'predictions.jsonl')} for a in p['arms']}
    arms={};correct={};pairs=defaultdict(list)
    for t in targets:pairs[t.pair_id].append(t)
    for a in p['arms']:
        arms[a]=score_files(output/a/'predictions.jsonl',target_path)
        arms[a]['official_score_valid']=arms[a]['official_status']=='COMPUTED'
        correct[a]={t.input_id:rows[a].get(t.input_id,{}).get('status')=='ok' and rows[a].get(t.input_id,{}).get('answer')==t.gold_label for t in targets}
        matrix=[[0,0],[0,0]];labels=['Refuted','Supported'];invalid=Counter();first=0;attempts=0;tokens=Counter();calcs=Counter()
        for t in targets:
            r=rows[a].get(t.input_id,{})
            if r.get('status')=='ok' and r.get('answer') in labels:matrix[labels.index(t.gold_label)][labels.index(r['answer'])]+=1
            else:invalid[t.gold_label]+=1
            result=read_sealed(output/a/'cases'/t.input_id/'result.json');first+=result['first_attempt_valid']
            for stage in (output/a/'cases'/t.input_id).glob('stage_*.json'):
                if stage.name.endswith('.seal.json'):continue
                x=read_sealed(stage);attempts+=x['attempt_count']
                for k,v in (x.get('observed') or {}).get('usage',{}).items():
                    if isinstance(v,int):tokens[k]+=v
                for c in x['calculations']:calcs[c['result']['status']]+=1
        arms[a].update({'confusion_matrix':{'labels':labels,'rows':'gold','columns':'prediction','values':matrix},
            'invalid_by_gold_label':dict(invalid),'label_errors':{'Refuted_to_Supported':matrix[0][1],'Supported_to_Refuted':matrix[1][0]},
            'first_attempt_valid_case_arms':first,'final_valid':sum(r.get('status')=='ok' for r in rows[a].values()),
            'attempts':attempts,'observed_usage':dict(tokens),'calculations':dict(calcs)})
    comparisons={};case_table=[];pair_table=[]
    for t in targets:case_table.append({'input_id':t.input_id,'work_family':t.work_family,'pair_id':t.pair_id,'gold':t.gold_label,'correct':{a:correct[a][t.input_id] for a in p['arms']}})
    for pid,ts in pairs.items():
        if len({t.work_family for t in ts})!=1:raise ValueError('pair spans work families')
        pair_table.append({'pair_id':pid,'work_family':ts[0].work_family,'correct':{a:all(correct[a][t.input_id] for t in ts) for a in p['arms']}})
    def paired(table,control):
        counts=Counter(('both_correct' if r['correct']['H1'] and r['correct'][control] else 'H1_only' if r['correct']['H1'] else 'control_only' if r['correct'][control] else 'both_wrong') for r in table)
        return {k:counts[k] for k in ['both_correct','H1_only','control_only','both_wrong']}
    for control in ['H_repeat','H0']:
        if not {'H1',control}<=set(p['arms']):continue
        cmp={'case_table':paired(case_table,control),'pair_table':paired(pair_table,control),
             'official_comparison_valid':arms['H1']['official_score_valid'] and arms[control]['official_score_valid']}
        for name,table in [('accuracy',case_table),('pair_accuracy',pair_table)]:
            families=sorted({r['work_family'] for r in table})
            if None in families:raise ValueError('missing work family')
            numer=np.array([sum(int(r['correct']['H1'])-int(r['correct'][control]) for r in table if r['work_family']==f) for f in families])
            denom=np.array([sum(r['work_family']==f for r in table) for f in families])
            rng=np.random.default_rng(SEED);indices=rng.integers(0,len(families),size=(10000,len(families)))
            estimates=numer[indices].sum(axis=1)/denom[indices].sum(axis=1)
            cmp[name]={'difference':float(numer.sum()/denom.sum()),'exploratory_cluster_bootstrap_95_percentile':np.quantile(estimates,[.025,.975]).tolist(),'clusters':len(families),'resamples':10000,'seed':SEED,'unit':'work_family','all_planned_denominator':True}
        comparisons['H1-'+control]=cmp
    result={'phase':p['phase'],'at':now(),'protocol_sha256':sha256(protocol_path),
            'all_arms_seal_sha256':sha256(output/'all_arms.seal.json'),'target_sha256':sha256(target_path),
            'cases':len(targets),'pairs':len(pairs),'families':len({t.work_family for t in targets}),
            'arms':arms,'comparisons':comparisons,'case_table':case_table,'pair_table':pair_table,
            'status':'DONE' if all(a['official_score_valid'] for a in arms.values()) else 'PARTIAL',
            'judge_status':'NOT_RUN','gold_changed':False,'development_feedback_allowed':p['phase']=='development'}
    write_json(dest,result,immutable=True);return result


def create_fixtures(root):
    """Two self-authored count tables, not scientific benchmark inputs or scores."""
    import io
    from PIL import Image,ImageDraw,ImageFont
    from .contracts import CaseInput,Evidence
    from .io import write_jsonl
    cases=[];assets=[]
    for i,n in enumerate([1,2],1):
        aid='a_'+digest({'fixture':'M2_table','i':i})[:32]
        eid='e_'+digest({'fixture':'M2_evidence','i':i})[:32]
        cid='c_'+digest({'fixture':'M2_case','i':i})[:32]
        image=Image.new('RGB',(1000,400),'white');draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=38)
        for y,text in [(35,'Baseline: simple counts in the same population'),(110,'Group       Count       Total       Reported %'),(185,f'A                {n}                 4                 {25*n}%'),(260,f'B                {4-n}                 4                 {25*(4-n)}%')]:draw.text((35,y),text,font=font,fill='black')
        buf=io.BytesIO();image.save(buf,format='PNG');path=root/'assets'/(aid+'.png');write_bytes(path,buf.getvalue(),immutable=True)
        assets.append({'asset_id':aid,'path':'assets/'+path.name,'media_type':'image/png','sha256':sha256(path)})
        cases.append(CaseInput(input_id=cid,task_kind='claim_verification',condition='benchmark_native',
            claim='Group A accounts for 25% of the total at baseline.',caption='Self-authored count table. All counts and percentages refer to the same population at baseline.',
            answer_domain=['Supported','Refuted'],evidence=[Evidence(evidence_id=eid,asset_id=aid,location_kind='image',location='entire image',extraction_method='self_authored_fixture')]).model_dump())
    write_jsonl(root/'cases.jsonl',cases,immutable=True);write_jsonl(root/'assets.jsonl',assets,immutable=True)
    write_json(root/'source.json',{'dataset_id':'RECAP_self_authored_M2_fixture','dataset_revision':'1'},immutable=True)
    return {'cases':2,'expected_by_input_id':{c['input_id']:('Supported' if i==0 else 'Refuted') for i,c in enumerate(cases)}}
