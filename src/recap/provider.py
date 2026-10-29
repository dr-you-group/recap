"""Provider-neutral types and one local Chat Completions transport adapter."""
from __future__ import annotations

import base64
import json
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from .contracts import CapabilityProfile, GenerationOptions


@dataclass(frozen=True)
class ImagePart:
    media_type: str
    data: bytes


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    images: tuple[ImagePart, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ModelRequest:
    input_id: str
    messages: tuple[Message, ...]
    options: GenerationOptions
    output_schema: dict | None = None
    tools: tuple[ToolSpec, ...] = ()
    mm_processor_kwargs: dict | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    finish_reason: str | None
    status: str
    transmitted: dict
    observed: dict
    retries: int = 0
    tool_calls: tuple[ToolCall, ...] = ()


class Provider(Protocol):
    profile: CapabilityProfile
    execution_kind: str
    def generate(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True, repr=False)
class Credentials:
    base_url: str
    api_key: str
    served_model: str


def load_credentials(path: Path | None = None) -> Credentials:
    keys = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")
    values = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            if key.strip() in keys:
                parts = shlex.split(value, comments=True)
                if len(parts) != 1:
                    raise ValueError("invalid credential file syntax")
                values[key.strip()] = parts[0]
    for key in keys:
        if os.environ.get(key):
            values[key] = os.environ[key]
    if not all(values.get(key) for key in keys):
        raise ValueError("missing endpoint credentials/model; values are not logged")
    return Credentials(values[keys[0]].rstrip("/"), values[keys[1]], values[keys[2]])


def validate_capabilities(profile: CapabilityProfile, request: ModelRequest) -> None:
    options = request.options
    images = sum(len(x.images) for x in request.messages)
    required = {"image": bool(images), "tools": options.use_tools,
                "structured_output": options.structured_output,
                "reasoning": options.reasoning is not None,
                "sampling": options.temperature is not None or options.top_p is not None or options.seed is not None,
                "logprobs": options.logprobs}
    for name, needed in required.items():
        if needed and getattr(profile, name) != "supported":
            raise ValueError(f"requested {name} capability is {getattr(profile, name)}")
    if images > profile.max_images:
        raise ValueError("image count exceeds declared profile")
    if options.use_tools != bool(request.tools):
        raise ValueError("tool request and declared tool specifications must agree")
    if options.structured_output and request.output_schema is None:
        raise ValueError("structured output requires an explicit schema")


class ChatCompletionsAdapter:
    execution_kind = "live"

    def __init__(self, profile: CapabilityProfile, credentials: Credentials, *, client=None,
                 allow_live: bool = False, sleep=time.sleep, max_retries: int = 2):
        if profile.transport != "chat_completions":
            raise ValueError("this adapter only supports chat_completions")
        self.profile, self.credentials = profile, credentials
        self.client = client
        self.allow_live = allow_live
        self.sleep = sleep
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError("max_retries must be 0, 1, or 2")
        self.max_retries = max_retries

    def build_payload(self, request: ModelRequest) -> dict:
        validate_capabilities(self.profile, request)
        messages = []
        for message in request.messages:
            if message.role not in {"system", "user", "assistant"}:
                raise ValueError("unsupported neutral message role")
            content = [{"type": "text", "text": message.text}]
            for image in message.images:
                if image.media_type not in {"image/png", "image/jpeg", "image/webp"}:
                    raise ValueError("unsupported image media type")
                encoded = base64.b64encode(image.data).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{encoded}"}})
            messages.append({"role": message.role, "content": content if message.images else message.text})
        payload = {"model": self.credentials.served_model, "messages": messages,
                   "max_tokens": request.options.output_budget, "stream": False}
        if request.mm_processor_kwargs is not None:
            kwargs = request.mm_processor_kwargs
            if (not isinstance(kwargs, dict) or set(kwargs) != {"max_soft_tokens"} or
                    type(kwargs["max_soft_tokens"]) is not int or kwargs["max_soft_tokens"] not in {560, 1120} or
                    not any(message.images for message in request.messages)):
                raise ValueError("R3 image processor override requires original images and max_soft_tokens 560 or 1120")
            payload["mm_processor_kwargs"] = dict(kwargs)
        for key in ("temperature", "top_p", "seed"):
            value = getattr(request.options, key)
            if value is not None:
                payload[key] = value
        if request.options.reasoning is not None:
            if self.profile.reasoning_wire_key != "chat_template_kwargs.enable_thinking":
                raise ValueError("no audited reasoning translation for this transport/model")
            payload["chat_template_kwargs"] = {"enable_thinking": request.options.reasoning == "enabled"}
        if request.options.structured_output:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "recap_answer", "strict": True, "schema": request.output_schema}}
        if request.options.logprobs:
            payload["logprobs"] = True
        if request.tools:
            if any(t.name not in {"get_evidence", "check_reported_percentage"} for t in request.tools):
                raise ValueError("tool specification not allowlisted")
            payload["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in request.tools]
        return payload

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self.build_payload(request)
        if not self.allow_live:
            raise PermissionError("live inference requires explicit opt-in")
        client = self.client or httpx.Client(timeout=120)
        origin = self.credentials.base_url.removesuffix("/v1").rstrip("/")
        transmitted = {k: v for k, v in payload.items() if k != "messages"}
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(origin + "/v1/chat/completions", json=payload,
                    headers={"Authorization": "Bearer " + self.credentials.api_key})
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    self.sleep(2**attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                finish = choice.get("finish_reason")
                output = choice.get("message", {}).get("content")
                calls = tuple(ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                              for c in choice.get("message", {}).get("tool_calls", []))
                if calls and (not request.options.use_tools or any(c.name not in {t.name for t in request.tools} or not isinstance(c.arguments, dict) for c in calls)):
                    raise ValueError("unexpected tool call")
                status = "truncated" if finish == "length" else "ok" if finish == "stop" and isinstance(output, str) and output.strip() else "invalid"
                if calls and finish == "tool_calls":
                    status = "tool_calls"
                return ModelResponse(text=output if isinstance(output, str) else "", finish_reason=finish,
                    status=status, transmitted=transmitted,
                    observed={k: data[k] for k in ("model", "usage", "system_fingerprint") if k in data},
                    retries=attempt, tool_calls=calls)
            except httpx.TransportError:
                if attempt < self.max_retries:
                    self.sleep(2**attempt)
                    continue
                return ModelResponse("", None, "failed", transmitted, {"error": "transport_error"}, attempt)
            except httpx.HTTPStatusError as error:
                return ModelResponse("", None, "failed", transmitted, {"http_status": error.response.status_code}, attempt)
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                return ModelResponse("", None, "invalid", transmitted, {"error": "invalid_server_response"}, attempt)
        raise AssertionError("bounded adapter loop exhausted unexpectedly")
