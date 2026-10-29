"""Bounded Chat Completions SSE transport; existing non-streaming adapter is unchanged.

Callbacks are synchronous and must persist to private attempt storage. They must not
print partial content/reasoning. No request is retried, cancelled remotely, or detached.
"""
from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

import httpx

from .h0 import safe_request
from .provider import ChatCompletionsAdapter, ModelResponse
from .result_durability import safe_error, IdentityMismatch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SSEDecoder:
    """Incremental UTF-8/line/frame decoding, including split CRLF and multiline data."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.buffer = ''; self.lines = []

    def feed(self, chunk):
        self.buffer += self.decoder.decode(chunk)
        frames = []
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.removesuffix('\r')
            if line:
                self.lines.append(line)
            else:
                event = 'message'; data = []
                for item in self.lines:
                    if item.startswith(':'):
                        continue
                    key, _, value = item.partition(':')
                    value = value[1:] if value.startswith(' ') else value
                    if key == 'event': event = value
                    elif key == 'data': data.append(value)
                frames.append((event, '\n'.join(data)))
                self.lines = []
        return frames

    def finish(self):
        self.buffer += self.decoder.decode(b'', final=True)
        if self.buffer.strip() or self.lines:
            raise ValueError('incomplete_sse_frame')


class StreamProtocolError(ValueError):
    pass


class StreamDeadline(TimeoutError):
    pass


class ChatCompletionsSSEAdapter:
    execution_kind = 'live'

    def __init__(self, profile, credentials, *, allow_live=False, record_event=None,
                 before_send=None, validate_final=None, client=None, request_total_seconds=300,
                 read_idle_seconds=60, progress_idle_seconds=90, connect_seconds=20,
                 milestone_deadline=None, request_id_header_supported=False, thinking_token_budget=None,
                 enforce_idle_deadlines=True, client_read_seconds=None, write_seconds=None):
        self.profile = profile; self.credentials = credentials
        self.allow_live = allow_live; self.record_event = record_event
        self.before_send = before_send; self.validate_final = validate_final; self.client = client
        if not isinstance(request_total_seconds, (int, float)) or not 0 < request_total_seconds <= 900:
            raise ValueError('request total must be positive and no greater than 900 seconds')
        limits = [read_idle_seconds, progress_idle_seconds, connect_seconds]
        if any(not isinstance(x, (int, float)) or not 0 < x <= 300 for x in limits):
            raise ValueError('timeouts must be positive and no greater than 300 seconds')
        if milestone_deadline is not None and (not isinstance(milestone_deadline, datetime) or milestone_deadline.tzinfo is None):
            raise ValueError('milestone deadline requires an aware datetime')
        self.total = request_total_seconds; self.read_idle = read_idle_seconds
        self.progress_idle = progress_idle_seconds; self.connect = connect_seconds
        self.enforce_idle_deadlines = enforce_idle_deadlines
        self.client_read = read_idle_seconds if client_read_seconds is None else client_read_seconds
        self.write = read_idle_seconds if write_seconds is None else write_seconds
        if type(enforce_idle_deadlines) is not bool or any(type(v) not in (int, float) or not 0 < v <= 900 for v in (self.client_read, self.write)):
            raise ValueError('invalid transport override')
        self.deadline = milestone_deadline; self.header_supported = request_id_header_supported
        if thinking_token_budget is not None and (type(thinking_token_budget) is not int or thinking_token_budget <= 0):
            raise ValueError('thinking token budget must be a positive integer')
        self.thinking_token_budget = thinking_token_budget
        self._used_keys = set()

    def build_payload(self, request):
        payload = ChatCompletionsAdapter(self.profile, self.credentials).build_payload(request)
        if request.tools:
            raise ValueError('SSE native tool execution is not implemented; use host structured records')
        if self.thinking_token_budget is not None:
            if request.options.reasoning != 'enabled' or self.thinking_token_budget >= request.options.output_budget:
                raise ValueError('thinking budget requires enabled reasoning and remaining final output budget')
            payload['thinking_token_budget'] = self.thinking_token_budget
        payload['stream'] = True
        payload['stream_options'] = {'include_usage': True}
        return payload

    def generate(self, request):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_generate(request))
        raise RuntimeError('use async_generate inside an existing event loop')

    async def async_generate(self, request):
        payload = self.build_payload(request)
        if not self.allow_live:
            raise PermissionError('live streaming requires explicit opt-in')
        if self.record_event is None or self.before_send is None:
            raise ValueError('private event persistence and durable budget callbacks are required')
        if request.options.structured_output and self.validate_final is None:
            raise ValueError('structured output requires a final schema validator callback')
        if self.deadline is not None and datetime.now(timezone.utc) >= self.deadline:
            raise TimeoutError('milestone deadline reached before sending')
        canonical = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        key = hashlib.sha256(request.input_id.encode() + b'\0' + canonical).hexdigest()
        if key in self._used_keys:
            raise RuntimeError('same input/payload was already attempted; no resend')
        self._used_keys.add(key)
        request_id = 'recap-' + uuid.uuid4().hex
        started = time.monotonic(); last_progress = started; last_received = started
        total_deadline = started + self.total
        milestone_monotonic = None if self.deadline is None else started + (self.deadline - datetime.now(timezone.utc)).total_seconds()
        times = {'transmitted_at': utc_now(), 'headers_received_at': None, 'first_received_at': None,
                 'last_received_at': None, 'first_progress_at': None, 'last_progress_at': None, 'finish_reason_received_at': None,
                 'terminal_at': None}
        text = []; reasoning = []; tool_deltas = []; observed = {}
        reason = None; status = 'failed'; finish = None; done = False; body_bytes = 0; frame_count = 0
        headers = {'Authorization': 'Bearer ' + self.credentials.api_key}
        if self.header_supported:
            headers['X-Request-Id'] = request_id
        transmitted = {k: v for k, v in payload.items() if k != 'messages'}

        def redact(value):
            raw = json.dumps(value, ensure_ascii=False)
            for secret in (self.credentials.api_key, self.credentials.base_url,
                           self.credentials.base_url.removesuffix('/v1').rstrip('/')):
                if secret: raw = raw.replace(secret, '[REDACTED]')
            return json.loads(raw)

        def emit(event, **data):
            self.record_event(redact({'event': event, 'at': utc_now(), 'client_request_id': request_id,
                                      'input_id': request.input_id, **data}))

        def remaining():
            current = time.monotonic()
            options = [(total_deadline - current, 'request_total_deadline')]
            if self.enforce_idle_deadlines:
                options += [(last_received + self.read_idle - current, 'read_idle_deadline'),
                            (last_progress + self.progress_idle - current, 'real_progress_idle_deadline')]
            if milestone_monotonic is not None:
                options.append((milestone_monotonic - current, 'milestone_deadline'))
            seconds, label = min(options)
            if seconds <= 0: raise StreamDeadline(label)
            return seconds, label

        async def bounded(awaitable):
            try:
                seconds, label = remaining()
            except BaseException:
                if hasattr(awaitable, 'close'): awaitable.close()
                raise
            try:
                return await asyncio.wait_for(awaitable, timeout=seconds)
            except asyncio.TimeoutError:
                raise StreamDeadline(label) from None

        owned = self.client is None
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(self.client_read, connect=self.connect, write=self.write, pool=self.connect))
        response = None; decoder = None; phase = 'before_send'
        initial = {'client_request_id': request_id, 'input_id': request.input_id,
                   'request_sha256': hashlib.sha256(canonical).hexdigest(), 'request': safe_request(payload),
                   'timeouts': {'connect_seconds': self.connect, 'read_idle_seconds': self.read_idle,
                                'real_progress_idle_seconds': self.progress_idle, 'request_total_seconds': self.total,
                                'milestone_deadline': self.deadline.isoformat() if self.deadline else None,
                                'enforce_idle_deadlines': self.enforce_idle_deadlines,
                                'client_read_seconds': self.client_read, 'write_seconds': self.write,
                                'pool_seconds': self.connect},
                   'request_id_header_sent': self.header_supported,
                   'client_id_is_idempotency_or_cancel_handle': False}
        try:
            self.before_send(redact(initial))
            emit('started', **initial, **times)
            outgoing = client.build_request('POST', self.credentials.base_url.removesuffix('/v1').rstrip('/') + '/v1/chat/completions',
                                            json=payload, headers=headers,
                                            timeout=httpx.Timeout(self.client_read, connect=self.connect, write=self.write, pool=self.connect))
            observed['request_body_declared_bytes'] = len(outgoing.content)
            observed['actual_sent_bytes'] = None
            observed['fresh_client_for_request'] = owned
            observed['transport_type'] = type(client._transport).__name__
            observed['transport_retries'] = getattr(getattr(client._transport, '_pool', None), '_retries', None)
            if owned and observed['transport_retries'] != 0:
                raise RuntimeError('transport_retry_policy_unverified')
            phase = 'send_headers'
            response = await bounded(client.send(outgoing, stream=True))
            times['headers_received_at'] = utc_now()
            observed['http_status'] = response.status_code
            if self.header_supported and response.headers.get('x-request-id'):
                observed['server_request_id_header'] = response.headers['x-request-id']
            if response.status_code != 200:
                emit('http_error', http_status=response.status_code)
                raise StreamProtocolError('http_status_' + str(response.status_code))
            if 'text/event-stream' not in response.headers.get('content-type', '').lower():
                raise StreamProtocolError('expected_text_event_stream')
            decoder = SSEDecoder(); iterator = response.aiter_bytes().__aiter__(); phase = 'read_body'
            while not done:
                try:
                    chunk = await bounded(iterator.__anext__())
                except StopAsyncIteration:
                    decoder.finish()
                    raise StreamProtocolError('eof_before_done')
                received = utc_now(); last_received = time.monotonic()
                times['first_received_at'] = times['first_received_at'] or received
                times['last_received_at'] = received; body_bytes += len(chunk)
                if body_bytes > 8 * 1024 * 1024:
                    raise StreamProtocolError('stream_exceeds_8MiB_guard')
                emit('sse_chunk', bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest())
                frames = decoder.feed(chunk)
                for index, (event_type, data) in enumerate(frames):
                    remaining()
                    frame_count += 1
                    emit('sse_frame', event_type=event_type, data=data)
                    if not data or event_type in {'ping', 'heartbeat'}:
                        continue
                    if data == '[DONE]':
                        done = True; times['terminal_at'] = utc_now()
                        if index != len(frames) - 1 or decoder.buffer.strip() or decoder.lines:
                            raise StreamProtocolError('data_after_done')
                        break
                    if event_type == 'error': raise StreamProtocolError('sse_error_event')
                    value = json.loads(data)
                    if not isinstance(value, dict): raise StreamProtocolError('invalid_sse_object')
                    if value.get('error') is not None: raise StreamProtocolError('sse_error_payload')
                    if value.get('model') is not None:
                        if value['model'] != self.credentials.served_model:
                            raise StreamProtocolError('served_model_alias_changed')
                        observed['model'] = value['model']
                    for field in ['id', 'system_fingerprint', 'usage']:
                        if value.get(field) is not None: observed[field] = value[field]
                    choices = value.get('choices', [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise StreamProtocolError('unexpected_choice_count')
                    progress = False
                    for choice in choices:
                        if choice.get('index', 0) != 0: raise StreamProtocolError('unexpected_choice_index')
                        delta = choice.get('delta') or {}
                        if not isinstance(delta, dict): raise StreamProtocolError('invalid_delta')
                        if delta.get('refusal'): raise StreamProtocolError('refusal')
                        for field, target in [('content', text), ('reasoning', reasoning), ('reasoning_content', reasoning)]:
                            part = delta.get(field)
                            if part is not None and not isinstance(part, str): raise StreamProtocolError('nontext_delta')
                            if part:
                                if finish is not None: raise StreamProtocolError('delta_after_finish')
                                target.append(part); progress = True
                        if delta.get('tool_calls'):
                            tool_deltas.append(delta['tool_calls']); progress = True
                        new_finish = choice.get('finish_reason')
                        if new_finish is not None:
                            if finish is not None: raise StreamProtocolError('duplicate_finish')
                            finish = new_finish; times['finish_reason_received_at'] = utc_now()
                    if progress:
                        last_progress = time.monotonic(); times['last_progress_at'] = utc_now()
                        times['first_progress_at'] = times['first_progress_at'] or times['last_progress_at']
                    if tool_deltas: raise StreamProtocolError('unexpected_native_tool_delta')
            phase = 'validate_final'
            if finish == 'length':
                status, reason = 'truncated', 'length_finish'
            elif finish != 'stop' or not ''.join(text).strip():
                status, reason = 'invalid', 'missing_stop_or_final_content'
            elif observed.get('model') != self.credentials.served_model:
                status, reason = 'invalid', 'missing_observed_model_alias'
            else:
                if request.options.structured_output:
                    json.loads(''.join(text))
                    self.validate_final(''.join(text))
                status = 'ok'
        except IdentityMismatch:
            raise
        except StreamDeadline as error:
            reason = str(error)
            observed['exception'] = safe_error(error, 'generation_' + phase)
        except httpx.HTTPError as error:
            reason = type(error).__name__
            observed['exception'] = safe_error(error, 'generation_' + phase)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            status = 'failed' if isinstance(error, StreamProtocolError) and str(error).startswith(('http_status_', 'sse_error_')) else 'invalid'
            reason = str(error) if isinstance(error, StreamProtocolError) else 'invalid_stream_or_final_schema'
        finally:
            if response is not None:
                await response.aclose()
            if owned:
                await client.aclose()
            summary = {**times, 'client_finished_at': utc_now(), 'elapsed_seconds': round(time.monotonic() - started, 6),
                       'finish_reason': finish, 'done_received': done, 'frame_count': frame_count,
                       'received_bytes': body_bytes, 'status': status, 'diagnostic': reason,
                       'partial_content': ''.join(text), 'partial_reasoning': ''.join(reasoning),
                       'tool_deltas': tool_deltas, 'observed': observed,
                       'incomplete_sse_frame': ('\n'.join(decoder.lines) + '\n' + decoder.buffer).strip() if decoder else '',
                       'remote_cancellation_confirmed': False, 'retries': 0}
            emit('finished', **summary)
        metadata = {k: v for k, v in summary.items() if k not in {'partial_content', 'partial_reasoning', 'tool_deltas', 'observed', 'incomplete_sse_frame'}}
        metadata.update(observed)
        metadata['partial_content_characters'] = sum(map(len, text))
        metadata['partial_reasoning_characters'] = sum(map(len, reasoning))
        return ModelResponse(''.join(text) if status == 'ok' else '', finish, status, transmitted, metadata, retries=0)
