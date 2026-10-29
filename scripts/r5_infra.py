"""One bounded precheck, plus independent postchecks on the existing US endpoint."""
from pathlib import Path
import json
import r2_infra as previous
from recap.io import now, write_json
from recap.result_durability import IdentityMismatch, ObservationUnavailable, safe_error

ROOT=Path(__file__).resolve().parents[1]


def identity_guard(binding,credentials,*,client=None,milestone="r5"):
    if json.loads((ROOT/'artifacts'/milestone/'model_binding.json').read_text())!=binding:
        raise IdentityMismatch('R5 local binding differs')
    return previous.identity_guard(binding,credentials,client=client)


def metrics_idle(**kwargs):return previous.metrics_idle(**kwargs)


def precheck(binding,credentials,*,milestone="r5",output_root=None):
    destination=Path(output_root) if output_root is not None else ROOT/'artifacts'/milestone
    output=destination/'precheck.json'
    if output.exists():raise RuntimeError('one R5 precheck only; no polling resume')
    writer=previous.write_json
    def redirect(path,value,**kwargs):
        if Path(path).name=='resource_ledger.json':
            value={'latest':value['latest'],'lifecycle_actions':[],'shutdown_policy':'keep_running_user_request'}
        return write_json(destination/Path(path).name,value,immutable=True)
    state={'at':now(),'idle':False,'identity_verified':False,'observations':[]}
    previous.write_json=redirect
    try:
        observed=previous.observe(server=True)
        if not observed.get('serving_now'):raise ObservationUnavailable()
        if not observed.get('live_binding_allowed'):raise IdentityMismatch('current runtime binding differs')
        if json.loads((destination/'model_binding.json').read_text())!=binding:
            raise IdentityMismatch('current binding differs')
        state['identity_verified']=True
        state['observations']=[metrics_idle(credentials=credentials) for _ in range(2)]
        state['idle']=all(x.get('idle') is True for x in state['observations'])
    except Exception as error:
        state['error']=safe_error(error,'precheck_identity_and_idle')
    finally:
        previous.write_json=writer
    write_json(output,state,immutable=True)
    return state
