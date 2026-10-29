"""M4 opt-in execution boundary; one stream at a time, durable cap, no retries."""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime,timezone
import fcntl,json,os,sys
import httpx
from pathlib import Path
from .contracts import CapabilityProfile,GenerationOptions
from .provider import load_credentials
from .streaming import ChatCompletionsSSEAdapter
from .m3 import tokenize_once
from .m2_runtime import preflight
from .io import write_json,sha256,now
from .h0 import digest
from .result_durability import persist_then_postcheck, ResultPersistenceError, IdentityMismatch

ROOT=Path(__file__).resolve().parents[2]

class LiveSession:
    def __init__(self,profile_path,attempt_dir,*,allow_live=False,credentials_path=None,control_path=None,milestone='m4',source_paths=None,work_path=None,run_namespace=None):
        if not allow_live:raise PermissionError('explicit --allow-live required')
        if milestone not in {'m4','m5','m6','r1','r2','r3','r4','r5','r6','final'}:raise ValueError('unsupported execution milestone')
        self.root=ROOT;self.work=Path(work_path) if work_path is not None else ROOT/'artifacts/private'/milestone
        self.run_namespace=run_namespace
        self.milestone=milestone
        self.start=json.loads(Path(control_path or self.work/'start_state.json').read_text())
        self.cutoff=datetime.fromisoformat(self.start['no_new_live_after']);self.deadline=datetime.fromisoformat(self.start['report_only_after'])
        self.is_recovery=milestone in {'r1','r2','r3','r4','r5','r6','final'}
        self.request_total=self.start.get('request_total_seconds',300) if self.is_recovery else 300
        self.max_generations=self.start.get('max_generations',6) if self.is_recovery else 15
        cap_limit={'r1':6,'r2':12,'r3':12,'r4':4,'r5':1,'r6':12,'final':15}.get(milestone,15)
        if not 0<self.request_total<=900 or not isinstance(self.max_generations,int) or not 0<self.max_generations<=cap_limit:raise ValueError('invalid recovery execution limits')
        if milestone in {'r2','r3'} and (self.request_total!=240 or self.max_generations!=12):raise ValueError('R2/R3 requires fixed 240 seconds and 12 generations')
        if milestone in {'r4','r5','r6','final'} and (self.request_total!=240 or self.max_generations!=({'r4':4,'r5':1,'r6':12,'final':15}[milestone])):raise ValueError('fixed recovery total/cap required')
        if datetime.now(timezone.utc)>=self.cutoff:raise TimeoutError('no_new_live_cutoff')
        self._require_request_time()
        self.attempt_dir=Path(attempt_dir)
        if self.attempt_dir.exists():raise FileExistsError('attempt directory exists; automatic resume prohibited')
        self.attempt_dir.mkdir(parents=True)
        self.lock=(self.work/'live.lock').open('a+')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.lock.close();raise
        try:
            if (self.work/'live_halt.json').exists():raise RuntimeError('live previously halted; no automatic resume')
            self.spec=json.loads(Path(profile_path).read_text())
            self.transport_kwargs={}
            if milestone in {'r4','r5','r6','final'}:
                expected={'enforce_idle_deadlines':False,'client_read_seconds':240,'write_seconds':240 if milestone in {'r5','r6','final'} else 60,'connect_seconds':20}
                if self.spec.get(milestone+'_control_transport')!=expected:raise ValueError('R4 transport profile mismatch')
                self.transport_kwargs=dict(expected)
            self.thinking_token_budget=self.spec.get('completion_recovery',{}).get('thinking_token_budget') if self.is_recovery else None
            self.profile=CapabilityProfile.model_validate(self.spec['capability'])
            self.options=GenerationOptions.model_validate(self.spec['options'])
            self.credentials=load_credentials(Path(credentials_path or ROOT/'serve-llm/.secrets/llm.env'))
            sys.path.insert(0,str(ROOT/'scripts'))
            import importlib
            if milestone=='final':
                from .final_runtime import SnapshotInfra
                self.binding=self.spec['model_binding'];self.infra=SnapshotInfra(self.work,self.binding)
            else:
                self.infra=importlib.import_module(milestone+'_infra')
                self.binding=json.loads((ROOT/'artifacts'/milestone/'model_binding.json').read_text())
            if self.credentials.served_model!=self.binding['served_alias'] or self.profile.model_id!=self.binding['model_id'] or self.profile.model_revision!=self.binding['model_revision']:
                raise ValueError('approved profile/model binding mismatch')
            initial=self.infra.precheck(self.binding,self.credentials) if milestone in {'r5','r6','final'} else self.infra.metrics_idle(credentials=self.credentials)
            write_json(self.attempt_dir/'initial_metrics.json',initial,immutable=True)
            if initial['idle'] is not True:raise RuntimeError('initial_server_idle_unverified')
            if milestone not in {'r5','r6','final'}:self.infra.identity_guard(self.binding,self.credentials)
            self.preverified=True
            files=([ROOT/p for p in source_paths] if source_paths is not None else [p for p in (ROOT/'src/recap').glob('*.py')])+([ROOT/'scripts'/(milestone+'_infra.py')] if milestone!='final' else [ROOT/'scripts/r5_infra.py',ROOT/'scripts/r2_infra.py',ROOT/'scripts/m1_infra.py'])
            self.source={str(p.relative_to(ROOT)):sha256(p) for p in files}
            write_json(self.attempt_dir/'protocol.json',{'at':now(),'source_files':self.source,'source_sha256':digest(self.source),'profile':self.spec,'model_binding':self.binding,'transport':'chat_completions_SSE','transport_overrides':self.transport_kwargs,'read_idle_seconds':60,'progress_idle_seconds':90,'request_total_seconds':self.request_total,'milestone_deadline':self.deadline.isoformat(),'max_generations':self.max_generations,'retry':0,'concurrency':1,'control_sha256':sha256(Path(control_path or self.work/'start_state.json'))},immutable=True)
        except BaseException:self.lock.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.lock.close()
    def _require_request_time(self):
        if self.is_recovery and (self.deadline-datetime.now(timezone.utc)).total_seconds()<self.request_total:
            raise TimeoutError('full_request_deadline_does_not_fit')
    def _tokenize(self, request):
        if self.milestone in {'r3','r4','r5','r6','final'}:
            # The earlier 30s CPU observation was incomplete (cause unresolved).
            # Match the existing inference read-idle limit for original PNGs.
            with httpx.Client(timeout=httpx.Timeout(60, connect=20)) as client:
                return tokenize_once(request,self.profile,self.credentials,client=client)
        return tokenize_once(request,self.profile,self.credentials)
    def generate(self,request,validator,stage,*,commit_prediction=None,expected_request_sha256=None):
        if request.options!=self.options:raise ValueError('request options differ from frozen protocol profile')
        if (self.work/'live_halt.json').exists():raise RuntimeError('live_halted_no_resend')
        if datetime.now(timezone.utc)>=self.cutoff:raise TimeoutError('no_new_live_cutoff')
        self._require_request_time()
        for p,h in self.source.items():
            if sha256(ROOT/p)!=h:raise RuntimeError('source_changed_during_live_run')
        if self.milestone not in {'r5','final'}:self.infra.identity_guard(self.binding,self.credentials)
        if self.milestone=='final':
            request=replace(request,mm_processor_kwargs={'max_soft_tokens':560} if any(m.images for m in request.messages) else None)
            if expected_request_sha256 is None:
                payload=ChatCompletionsSSEAdapter(self.profile,self.credentials,thinking_token_budget=self.thinking_token_budget).build_payload(request)
                expected_request_sha256=__import__('hashlib').sha256(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        check=preflight(request,self.binding['max_context'],self._tokenize)
        path=self.attempt_dir/(stage+'.events.jsonl')
        if path.exists():raise FileExistsError('stage already attempted; no resend')
        write_json(self.attempt_dir/(stage+'.preflight.json'),check,immutable=True)
        def reserve(initial):
            if expected_request_sha256 is not None and initial['request_sha256']!=expected_request_sha256:
                raise IdentityMismatch('actual request hash mismatch')
            if datetime.now(timezone.utc)>=self.cutoff:raise TimeoutError('no_new_live_cutoff')
            self._require_request_time()
            p=self.work/'budget.json'
            with p.open('a+') as f:
                fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);raw=f.read();state=json.loads(raw) if raw else {'attempts':0,'requests':{}}
                key=(self.run_namespace+':' if self.run_namespace else '')+request.input_id
                if state['attempts']>=self.max_generations or key in state['requests']:raise RuntimeError('generation_cap_or_duplicate_input')
                state['attempts']+=1;state['requests'][key]={'stage':stage,'client_request_id':initial['client_request_id'],'request_sha256':initial['request_sha256'],'at':now()}
                f.seek(0);f.truncate();json.dump(state,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        def record(event):
            with path.open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush()
        provider=ChatCompletionsSSEAdapter(self.profile,self.credentials,allow_live=True,record_event=record,before_send=reserve,validate_final=validator,milestone_deadline=self.deadline,request_total_seconds=self.request_total,thinking_token_budget=self.thinking_token_budget,**self.transport_kwargs)
        response=provider.generate(request)
        try:
            state=persist_then_postcheck(response,self.attempt_dir/(stage+'.response.json'),
                {'at':now(),**asdict(response),'events_sha256':sha256(path),'input_id':request.input_id},
                commit_prediction=commit_prediction,
                identity=lambda **kwargs:self.infra.identity_guard(self.binding,self.credentials,**kwargs),
                metrics=lambda **kwargs:self.infra.metrics_idle(credentials=self.credentials,**kwargs),
                preverified=self.preverified,expected_model=self.binding['served_alias'],soft_observation_policy=self.milestone=='final')
        except ResultPersistenceError:
            write_json(self.work/'live_halt.json',{'at':now(),'reason':'result_persistence_failed','automatic_resume_allowed':False},immutable=True)
            raise
        if not state['continue_allowed']:
            write_json(self.work/'live_halt.json',{'at':now(),'reason':'postcheck_or_provenance_unresolved','automatic_resume_allowed':False,'state_file':str(self.attempt_dir/(stage+'.state.json'))},immutable=True)
        return replace(response,observed={**response.observed,'execution_state':state})


def review_cli(args):
    from .document_sources import index_document,select_sources
    from .document_review import review_document
    if not args.allow_live:return {'status':'NOT_RUN','reason':'explicit live opt-in required'}
    if args.source_index:
        index=json.loads(args.source_index.read_text())
        if index['condition']!=args.condition or index['source_asset']['sha256']!=sha256(args.input):raise ValueError('source index does not match input/condition')
        actual=index_document(args.input,args.condition,provenance=index['source_asset'])
        if actual['sources']!=index['sources'] or actual['inventory']!=index['inventory']:
            raise ValueError('source index differs from actual native input; assisted or fabricated evidence rejected')
        index=actual
    else:index=index_document(args.input,args.condition)
    selected=select_sources(index,max_chars=args.max_chars,max_sources=40)
    with LiveSession(args.profile,ROOT/'artifacts/private/m4'/('review_'+args.output.name+'.attempts'),allow_live=True,credentials_path=args.credentials,control_path=args.control) as live:
        return review_document(index,selected,live.generate,live.options,args.output,max_checks=args.max_checks)
