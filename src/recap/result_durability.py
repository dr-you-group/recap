"""Commit responses before independent postchecks; no gold or model calls here."""
from pathlib import Path
import time
import httpx
from .io import now, sha256, write_json

RULE_VERSION = 'result_durability_v1'


class IdentityMismatch(ValueError):
    reason_code = 'confirmed_identity_or_input_mismatch'


class ObservationUnavailable(RuntimeError):
    reason_code = 'observation_unavailable'
    def __init__(self, *, http_status=None):
        super().__init__(self.reason_code)
        self.http_status = http_status


class ResultPersistenceError(RuntimeError):
    def __init__(self, state):
        super().__init__('result_persistence_failed')
        self.state = state


def safe_error(error, stage):
    """Types and observed status only; never stringify exception messages or URLs."""
    chain=[];seen=set();current=error
    while current is not None and id(current) not in seen and len(chain)<8:
        seen.add(id(current))
        status=getattr(current,'http_status',None)
        if isinstance(current,httpx.HTTPStatusError):status=current.response.status_code
        chain.append({'type':type(current).__name__,'http_status':status})
        current=current.__cause__ or current.__context__
    return {'at':now(),'stage':stage,'exception_type':type(error).__name__,'cause_chain':chain,
            'safe_cause':getattr(error,'reason_code','unclassified_exception_message_omitted')}


def response_status(response):
    if response.status=='ok' and response.finish_reason=='stop' and response.observed.get('done_received') is True:
        return 'valid_final'
    if response.status=='failed':return 'transport_error'
    if response.status=='invalid' and response.observed.get('done_received'):return 'invalid_schema'
    return 'incomplete'


def eligibility(validity, persistence, provenance):
    if validity!='valid_final' or persistence!='committed':return 'not_eligible'
    return 'eligible' if provenance=='verified' else 'quarantine'


def persist_then_postcheck(response, raw_path, raw_record, *, commit_prediction=None,
                          identity, metrics, preverified, expected_model, soft_observation_policy=False):
    raw_path=Path(raw_path);state_path=raw_path.with_name(raw_path.name.replace('.response.json','.state.json'))
    provenance='verified' if preverified and response.observed.get('model')==expected_model else 'unknown'
    if response.observed.get('model') not in (None,expected_model) or response.observed.get('diagnostic')=='served_model_alias_changed':provenance='mismatch'
    state={'at':now(),'rule_version':RULE_VERSION,'response_status':response_status(response),
           'response_persistence':'absent','provenance_status':provenance,'postcheck_status':'not_run',
           'continue_allowed':False,'pre_request_evidence_verified':bool(preverified),'observations':[]}
    phase='raw_response_commit'
    try:
        write_json(raw_path,raw_record,immutable=True)
        state['raw_response']={'path':str(raw_path),'sha256':sha256(raw_path)}
        phase='prediction_commit'
        if commit_prediction is not None:
            state['prediction_artifact']=commit_prediction(response,state['raw_response'])
        state['response_persistence']='committed'
        state['committed_at']=now()
    except Exception as error:
        state.update(response_persistence='failed',error=safe_error(error,phase),evaluation_eligibility='not_eligible')
        try:write_json(state_path,state,immutable=True)
        except OSError:pass
        raise ResultPersistenceError(state) from error
    state['per_request_remote_identity_verified']=False
    if soft_observation_policy:
        state['identity_basis']='fixed run snapshot + actual request binding + returned model; remote postcheck separately recorded'
    if provenance=='mismatch':
        state.update(postcheck_status='mismatch',error={'stage':'response_identity','safe_cause':'observed_model_mismatch'})
    else:
        phase='postcheck_identity'
        retries=[0];retry_deadline=[None]
        def observe(call):
            while True:
                try:
                    if not soft_observation_policy:return call()
                    remaining=60 if retry_deadline[0] is None else retry_deadline[0]-time.monotonic()
                    if remaining<=0:raise ObservationUnavailable()
                    return call(timeout_seconds=min(20,remaining))
                except (ObservationUnavailable,httpx.HTTPError) as error:
                    if not soft_observation_policy or retries[0]>=2:raise
                    if retry_deadline[0] is None:retry_deadline[0]=time.monotonic()+60
                    if time.monotonic()>=retry_deadline[0]:raise
                    retries[0]+=1
                    state['observations'].append({'operation':phase,'recheck':retries[0],'error':safe_error(error,phase)})
        try:
            observe(identity)
            state['per_request_remote_identity_verified']=True
            state['observations'].append({'at':now(),'operation':'identity','verified':True})
            for i in range(2):
                phase='postcheck_idle_'+str(i+1)
                result=observe(metrics)
                state['observations'].append({'operation':'idle','result':result})
            ready=all(x['result'].get('idle') is True for x in state['observations'] if x['operation']=='idle')
            state.update(postcheck_status='ok' if ready else 'unavailable',continue_allowed=ready)
        except Exception as error:
            state['error']=safe_error(error,phase)
            if isinstance(error,IdentityMismatch):
                state.update(provenance_status='mismatch',postcheck_status='mismatch')
            elif isinstance(error,(ObservationUnavailable,httpx.HTTPError)):
                state['postcheck_status']='unavailable'
                if soft_observation_policy and state['response_status']=='valid_final' and state['provenance_status']=='verified':
                    state['continue_allowed']=True
                    state['continue_reason']='normal terminal and committed final; fixed run binding and request/model verified; optional remote observation unavailable'
            else:
                state.update(provenance_status='unknown',postcheck_status='error')
    # Aggregate idle never certifies cancellation of an incomplete generation.
    if soft_observation_policy and not (response.finish_reason=='stop' and response.observed.get('done_received') is True):
        state['continue_allowed']=False
        state['continue_reason']='previous request lacks normal terminal; aggregate idle is not cancellation proof'
    if soft_observation_policy and state['provenance_status']!='verified':
        state['continue_allowed']=False
    state['evaluation_eligibility']=eligibility(state['response_status'],state['response_persistence'],state['provenance_status'])
    state['evaluation_reason']='versioned validity/persistence/provenance rule; correctness not inspected'
    state['completed_at']=now()
    write_json(state_path,state,immutable=True)
    return state
