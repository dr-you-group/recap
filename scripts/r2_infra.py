"""R2 US-only read-only observation. No lifecycle commands are implemented."""
import argparse
import hashlib
import json
from pathlib import Path

import httpx
from recap.provider import load_credentials
from recap.result_durability import IdentityMismatch, ObservationUnavailable

import m1_infra as old
from recap.h0 import now
from recap.h0_admin import derive_binding
from recap.io import write_json

ROOT = Path(__file__).resolve().parents[1]

def observe(*, server=False):
    _, env = old.context()
    deployment = json.loads((ROOT / 'artifacts/private/m1/us_deployment.json').read_text())
    original = json.loads((ROOT / 'serve-llm/.secrets/deployment.json').read_text())
    previous = json.loads((ROOT / 'artifacts/m1/us_resource_ledger.json').read_text())
    if deployment['pod_id'] == original['pod_id'] or deployment['volume_id'] == original['volume_id']:
        raise ValueError('R2 must never address original resources')
    if deployment['region'] != 'US-MD-1':
        raise ValueError('unapproved region')
    for key, expected in [('pod_id', previous['pod_fingerprint']), ('volume_id', previous['volume_fingerprint'])]:
        if hashlib.sha256(deployment[key].encode()).hexdigest() != expected:
            raise ValueError('US target identity drift')
    deployment = {**deployment, 'name': 'recap-gemma4-31b-m1-us'}
    state = old.pod_state(deployment, env)
    endpoint = old.endpoint_status()
    record = {'at': now(), 'pod': old.public_state(state), 'endpoint': endpoint,
              'region': deployment['region'], 'lifecycle_actions': [], 'policy': 'keep_running',
              'identity_verified': True, 'serving_now': state['runtimeStatus'] == 'running' and endpoint.get('ready', False)}
    if server and record['serving_now']:
        # Reuse the already verified host-key file without changing any private state.
        known = ROOT / 'artifacts/private/m5/known_hosts'
        if not known.is_file():
            raise ValueError('existing verified US host-key file unavailable')
        observation = old.remote_observation(deployment, env, known_hosts_path=known, strict_host_checking='yes')
        binding = derive_binding(observation, endpoint.get('models', []))
        expected = json.loads((ROOT / 'artifacts/m1/model_binding.json').read_text())
        record['binding_equal_to_m1'] = binding == expected
        old_observation = json.loads((ROOT / 'artifacts/m1/server_observation.json').read_text())
        record['runtime_config_equal_to_m1'] = all(observation[k] == old_observation[k]
            for k in ['settings', 'engine_version', 'torch_version', 'files', 'template_sha256', 'script_hashes'])
        write_json(ROOT / 'artifacts/r2/server_observation.json', observation)
        write_json(ROOT / 'artifacts/r2/model_binding.json', binding, immutable=True)
        record['live_binding_allowed'] = record['binding_equal_to_m1'] and record['runtime_config_equal_to_m1']
    path = ROOT / 'artifacts/r2/resource_ledger.json'
    ledger = json.loads(path.read_text()) if path.exists() else {'events': [], 'lifecycle_actions': [], 'shutdown_policy': 'keep_running_user_request'}
    ledger['events'].append(record); ledger['latest'] = record
    write_json(path, ledger)
    return record

def identity_guard(binding, credentials, *, client=None):
    """Read /v1/models only; no generation and no endpoint settings changes."""
    path = ROOT / 'artifacts/r2/model_binding.json'
    if json.loads(path.read_text()) != binding:
        raise IdentityMismatch('local binding drift')
    owned = client is None
    client = client or httpx.Client(timeout=20)
    try:
        response = client.get(credentials.base_url.removesuffix('/v1').rstrip('/') + '/v1/models',
                              headers={'Authorization': 'Bearer ' + credentials.api_key})
        if response.status_code != 200:
            raise ObservationUnavailable(http_status=response.status_code)
        matches = [m for m in response.json().get('data', []) if m.get('id') == binding['served_alias']]
        if (len(matches) != 1 or matches[0].get('root') != binding['model_id'] or
                matches[0].get('max_model_len') != binding['max_context']):
            raise IdentityMismatch('endpoint binding drift')
        return True
    except httpx.HTTPError as error:
        raise ObservationUnavailable() from error
    finally:
        if owned:
            client.close()


def metrics_idle(*, credentials=None, client=None):
    """Aggregate read only; unknown/missing metrics never count as idle."""
    credentials = credentials or load_credentials(ROOT / 'serve-llm/.secrets/llm.env')
    record = {'at': now(), 'scope': 'aggregate only; no request ownership inference',
              'idle': False, 'running': None, 'waiting': None}
    owned = client is None
    client = client or httpx.Client(timeout=20)
    try:
        response = client.get(credentials.base_url.removesuffix('/v1').rstrip('/') + '/metrics',
                              headers={'Authorization': 'Bearer ' + credentials.api_key})
        record['http_status'] = response.status_code
        if response.status_code == 200:
            for field, name in [('running', 'vllm:num_requests_running'), ('waiting', 'vllm:num_requests_waiting')]:
                values = []
                for line in response.text.splitlines():
                    if line.startswith(name + '{') or line.startswith(name + ' '):
                        values.append(float(line.rsplit(' ', 1)[-1]))
                if values:
                    record[field] = sum(values)
            record['idle'] = record['running'] == 0 and record['waiting'] == 0
    except (httpx.HTTPError, ValueError) as error:
        record['error_type'] = type(error).__name__
    finally:
        if owned:
            client.close()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe', action='store_true', help='Read existing server/cache/process metadata over SSH')
    args = parser.parse_args()
    print(json.dumps(observe(server=args.observe)))

if __name__ == '__main__':
    main()
