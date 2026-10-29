"""Final profile wiring over the existing LiveSession and approved read-only checks."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import asyncio,json,sys
import httpx
from .io import now,sha256,write_json
from .h0 import digest
from .result_durability import ObservationUnavailable,IdentityMismatch

ROOT=Path(__file__).resolve().parents[2]


class BoundedReadClient:
    """Existing GET observers with one total timeout, including response body."""
    def __init__(self,seconds):self.seconds=seconds
    def get(self,*args,**kwargs):
        async def read():
            async with httpx.AsyncClient(timeout=self.seconds) as client:
                return await asyncio.wait_for(client.get(*args,**kwargs),timeout=self.seconds)
        try:return asyncio.run(read())
        except TimeoutError as error:raise ObservationUnavailable() from error


class SnapshotInfra:
    def __init__(self,work,binding):
        sys.path.insert(0,str(ROOT/'scripts'))
        import r5_infra
        self.previous=r5_infra;self.work=Path(work);self.binding=binding

    def precheck(self,binding,credentials):
        folder=self.work/'initial'
        if (folder/'precheck.json').exists():
            snapshot=json.loads((folder/'precheck.json').read_text())
            if json.loads((folder/'model_binding.json').read_text())!=binding:raise IdentityMismatch('run binding changed')
            return {**snapshot,'reused_run_snapshot':True,'observed_now':False}
        write_json(folder/'model_binding.json',binding,immutable=True)
        return self.previous.precheck(binding,credentials,milestone='final',output_root=folder)

    def identity_guard(self,binding,credentials,*,timeout_seconds=20):
        if binding!=self.binding:raise IdentityMismatch('run binding changed')
        return self.previous.previous.identity_guard(binding,credentials,client=BoundedReadClient(timeout_seconds))

    def metrics_idle(self,*,credentials,timeout_seconds=20):
        value=self.previous.metrics_idle(credentials=credentials,client=BoundedReadClient(timeout_seconds))
        if value.get('running') is None or value.get('waiting') is None:
            raise ObservationUnavailable(http_status=value.get('http_status'))
        return value


def session_for(profile_path,output,*,control=None,credentials=None):
    from .m4_execution import LiveSession
    output=Path(output);runtime=output.with_name(output.name+'.runtime')
    control=Path(control) if control else runtime/'start_state.json'
    if not control.exists():
        start=datetime.now(timezone.utc)
        write_json(control,{'started_at':start.isoformat(),'deadline':(start+timedelta(seconds=3600)).isoformat(),
            'report_only_after':(start+timedelta(seconds=3300)).isoformat(),'no_new_live_after':(start+timedelta(seconds=3060)).isoformat(),
            'request_total_seconds':240,'max_generations':15,'retry':0,'concurrency':1},immutable=True)
    return LiveSession(profile_path,runtime/'attempts',allow_live=True,control_path=control,
        credentials_path=credentials,milestone='final',work_path=control.parent,run_namespace=digest(str(output.resolve()))[:16])
