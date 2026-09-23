"""LLM provider selection. Anthropic is the default; Ollama and Earth Runtime are fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any
from uuid import uuid4

from orbweaver.config import settings
from orbweaver.open_models import (
    DEFAULT_OPENROUTER_BASE,
    OPEN_MODELS,
    is_claude_model,
    is_open_model,
    max_tokens_for,
)

log = logging.getLogger(__name__)

# WebSocket progress kinds. Not persisted; the UI may ignore them.
ASSISTANT_DELTA = "assistant_delta"
TOOL_USE_PROGRESS = "tool_use_progress"

# Raw Anthropic SSE types. The SDK also yields derived `text` / `input_json`
# events for the same deltas — ignore those so we do not double-count.
_STREAM_EVENT_TYPES = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    }
)

SONNET_MAX_TOKENS = 64_000
OPUS_MAX_TOKENS = 32_000
DEFAULT_MAX_TOKENS = 8_192

# Anthropic's client default is 10 minutes. Sonnet's 64k output budget can take
# longer than that even when streamed; the SDK also rejects non-streaming
# create() once estimated time exceeds 10 minutes (see streaming_required_for_max_tokens).
ANTHROPIC_TIMEOUT_S = 3600.0
_SDK_NONSTREAMING_LIMIT_S = 600.0
_SDK_TOKENS_PER_HOUR = 128_000


def max_tokens_for_model(model: str) -> int:
    """Output budget for a model id (Claw Code: Sonnet 64k, Opus 32k)."""
    name = (model or "").lower()
    if "opus" in name:
        return OPUS_MAX_TOKENS
    if "sonnet" in name:
        return SONNET_MAX_TOKENS
    return DEFAULT_MAX_TOKENS


def completion_max_tokens(model: str | None = None) -> int:
    """Tokens to request; stay at or under output_reserve so compact math holds."""
    raw = max_tokens_for_model(model or settings.orbweaver_model)
    return max(1, min(raw, settings.output_reserve))


def streaming_required_for_max_tokens(max_tokens: Any) -> bool:
    """True when Anthropic's SDK would reject non-streaming ``messages.create``.

    The client computes ``expected_time = 3600 * max_tokens / 128000`` and raises
    ``ValueError: Streaming is required...`` when that exceeds 10 minutes
    (~21334 tokens). Sonnet 64k and Opus 32k both trip it; Haiku 8k does not.
    """
    try:
        n = int(max_tokens)
    except (TypeError, ValueError):
        return False
    if n <= 0:
        return False
    return (60 * 60 * n / _SDK_TOKENS_PER_HOUR) > _SDK_NONSTREAMING_LIMIT_S


def is_streaming_required_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and "Streaming is required" in str(exc)


def _ollama_configured() -> bool:
    return bool(settings.ollama_base_url.strip() and settings.ollama_model.strip())


def openrouter_configured() -> bool:
    return bool(settings.openrouter_key)


def unknown_model(model: str | None) -> bool:
    """A non-empty id that is not Auto, a Claude name, a catalog name, or Ollama.

    Empty is "unspecified", not unknown, so the provider defaults still apply.
    """
    m = (model or "").strip()
    if not m or m.lower() == "auto" or is_open_model(m) or is_claude_model(m):
        return False
    ollama = settings.ollama_model.strip()
    return not (ollama and m == ollama)


def select_provider(model: str | None = None) -> str:
    """Claude names → Anthropic; catalog names → Earth Runtime; the Ollama model → Ollama.

    ``auto`` is the virtual router id: it is available when any concrete
    provider is configured. An id we do not recognise routes nowhere. It used
    to fall through to Anthropic, which answered ``404 not_found_error`` on
    every turn — that is what ``ORBWEAVER_TELEGRAM_MODEL=glm-5.3-flash`` looked
    like in production while that (real) Earth Runtime model was missing from
    ``OPEN_MODELS``.
    """
    model = (model or settings.orbweaver_model).strip()
    if model.lower() == "auto":
        if settings.anthropic_api_key.strip() or openrouter_configured() or _ollama_configured():
            return "auto"
        return "none"
    if is_open_model(model):
        return "openrouter" if openrouter_configured() else "none"
    if is_claude_model(model):
        if settings.anthropic_api_key.strip():
            return "anthropic"
        if _ollama_configured():
            return "ollama"
        return "none"
    if _ollama_configured() and model == settings.ollama_model.strip():
        return "ollama"
    if unknown_model(model):
        return "none"
    if settings.anthropic_api_key.strip():
        return "anthropic"
    if openrouter_configured():
        return "openrouter"
    if _ollama_configured():
        return "ollama"
    return "none"


def hosted_provider(model: str | None = None) -> str:
    """Anthropic or Earth Runtime only (classifier, probe, compact). No Ollama.

    ``auto`` is not a hosted id; callers must ``realize_turn_model`` first.
    """
    model = (model or "").strip()
    if model.lower() == "auto":
        return "none"
    if is_open_model(model):
        return "openrouter" if openrouter_configured() else "none"
    if unknown_model(model):
        return "none"
    if settings.anthropic_api_key.strip():
        return "anthropic"
    return "none"


def no_llm_echo(user_text: str, model: str | None = None) -> str:
    model = (model or settings.orbweaver_model).strip()
    echo = (user_text or "")[:500]
    if is_open_model(model) and not openrouter_configured():
        return (
            f"OPENROUTER_API_KEY is not set (needed for {model}). Echo: {echo}\n"
            "Get a key from earthruntime.com and set OPENROUTER_API_KEY "
            "(optional OPENROUTER_BASE_URL, default https://api.earthruntime.com/v1)."
        )
    if unknown_model(model):
        return (
            f"{model} is not a model Orbweaver can route, so no provider was picked. "
            f"Echo: {echo}\nUse a claude-* id, the configured OLLAMA_MODEL, or one of: "
            + ", ".join(sorted(OPEN_MODELS))
            + "."
        )
    return (
        "No LLM provider is configured. Echo: "
        + echo
        + "\nSet ANTHROPIC_API_KEY for Claude, OPENROUTER_API_KEY for Earth Runtime "
        "open models, or OLLAMA_BASE_URL and OLLAMA_MODEL for a local Ollama fallback."
    )


CACHE_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}
# Mark the tools array as a cache prefix only when it is big enough to matter.
CACHE_TOOLS_MIN = 8


def prompt_cache_supported(client: Any) -> bool:
    """Anthropic honours ``cache_control``; the Ollama and OpenAI-compatible shims
    rebuild messages without it."""
    return not isinstance(client, (OllamaMessagesClient, OpenAICompatClient))


def with_message_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy ``messages`` with ``cache_control`` on the last block of the final user message.

    Tool results are appended each round, so the previous breakpoint's prefix is
    a prefix of this round's request and Anthropic serves it as a cache read.
    String user content becomes a single text block on every user message so
    the message that carried last round's breakpoint is byte-identical this
    round. Never mutates the input (blocks may be shared with event payloads).
    """
    if not messages:
        return messages
    out: list[dict[str, Any]] = []
    for m in messages:
        msg = dict(m)
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, str) and content:
            msg["content"] = [{"type": "text", "text": content}]
        out.append(msg)
    for i in range(len(out) - 1, -1, -1):
        msg = out[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            blocks = list(content)
            blocks[-1] = {**blocks[-1], "cache_control": dict(CACHE_EPHEMERAL)}
            msg["content"] = blocks
        break
    return out


def with_tool_cache_breakpoint(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Copy ``tools`` with ``cache_control`` on the last definition when the list is large."""
    if not tools or len(tools) < CACHE_TOOLS_MIN or not isinstance(tools[-1], dict):
        return tools
    out = list(tools)
    out[-1] = {**out[-1], "cache_control": dict(CACHE_EPHEMERAL)}
    return out


def make_anthropic_client() -> Any | None:
    if not settings.anthropic_api_key.strip():
        return None
    import anthropic

    headers = {}
    if settings.anthropic_workspace_id.strip():
        headers["anthropic-workspace-id"] = settings.anthropic_workspace_id.strip()
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        timeout=ANTHROPIC_TIMEOUT_S,
        default_headers=headers or None,
    )


def make_agent_client(*, http: Any | None = None, model: str | None = None) -> Any | None:
    """Anthropic for Claude; Earth Runtime for open models; Ollama as last resort."""
    provider = select_provider(model)
    if provider == "anthropic":
        return make_anthropic_client()
    if provider == "openrouter":
        return OpenAICompatClient(
            settings.openrouter_base_url.strip() or DEFAULT_OPENROUTER_BASE,
            settings.openrouter_key,
            http=http,
        )
    if provider == "ollama":
        return OllamaMessagesClient(
            settings.ollama_base_url.strip(),
            settings.ollama_model.strip(),
            http=http,
        )
    return None


def make_hosted_client(model: str, *, http: Any | None = None) -> Any | None:
    """Client for classifier / probe / compact: Claude → Anthropic, catalog → Earth Runtime."""
    provider = hosted_provider(model)
    if provider == "anthropic":
        return make_anthropic_client()
    if provider == "openrouter":
        return OpenAICompatClient(
            settings.openrouter_base_url.strip() or DEFAULT_OPENROUTER_BASE,
            settings.openrouter_key,
            http=http,
        )
    return None


def compact_llm_client(agent_client: Any | None, *, http: Any | None = None) -> Any | None:
    """Client that can serve ORBWEAVER_COMPACT_MODEL (Claude or catalog)."""
    model = settings.orbweaver_compact_model.strip()
    provider = hosted_provider(model)
    if provider == "none":
        return None
    if provider == "openrouter":
        if isinstance(agent_client, OpenAICompatClient):
            return agent_client
        return make_hosted_client(model, http=http)
    if isinstance(agent_client, OpenAICompatClient):
        return make_anthropic_client()
    return agent_client if agent_client is not None else make_anthropic_client()


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, uid: str, name: str, inp: dict[str, Any]) -> None:
        self.id = uid
        self.name = name
        self.input = inp


class _Usage:
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cost_usd: float | None = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cost_usd = cost_usd


class _Response:
    def __init__(self, content: list[Any], usage: _Usage | None = None) -> None:
        self.content = content
        self.usage = usage or _Usage()


class OllamaMessagesClient:
    """Anthropic-shaped client that talks to Ollama /api/chat."""

    def __init__(self, base_url: str, model: str, *, http: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = http
        self.messages = self

    async def create(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        system: Any = "",
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]],
    ) -> _Response:
        del max_tokens
        payload = {
            "model": (model or self.model) or settings.ollama_model,
            "stream": False,
            "messages": _to_ollama_messages(system, messages),
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]
        data = await self._post(payload)
        return _from_ollama_message(data.get("message") or {})

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        if self._http is not None:
            resp = await self._http.post(url, json=payload)
            return _json_of(resp)
        import httpx

        resp = await httpx.AsyncClient(timeout=120.0).post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _json_of(resp: Any) -> dict[str, Any]:
    if hasattr(resp, "json"):
        data = resp.json()
        if callable(data):
            data = data()
        if hasattr(data, "__await__"):
            raise TypeError("fake client json() must be sync")
        return data
    if isinstance(resp, dict):
        return resp
    raise TypeError("ollama response is not JSON")


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "\n\n".join(p for p in parts if p)
    return str(system or "")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    bits: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            bits.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            src = block.get("source") or {}
            bits.append(f"[image {src.get('path') or src.get('media_type') or 'attached'}]")
    return "\n".join(b for b in bits if b)


def _to_ollama_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sys_text = _system_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or str(uuid4()),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
                elif block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
            row: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
            if tool_calls:
                row["tool_calls"] = tool_calls
            out.append(row)
            continue
        if role == "user" and isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for block in tool_results:
                    inner = block.get("content")
                    if isinstance(inner, list):
                        inner = _content_to_text(inner)
                    out.append(
                        {
                            "role": "tool",
                            "tool_name": str(block.get("tool_use_id") or ""),
                            "content": str(inner or ""),
                        }
                    )
                continue
        out.append({"role": "user" if role == "user" else role, "content": _content_to_text(content)})
    return out


def normalize_openai_tool_parameters(schema: Any) -> dict[str, Any]:
    """Make a tool ``parameters`` object acceptable to strict OpenAI-compatible APIs.

    GLM (and several other hosts) reject object schemas that omit ``required`` or
    send ``required: null`` — the error is often ``at '/required': got null, want
    array``. Parameter-free tools therefore need an explicit empty array. Also
    fill in ``type`` / ``properties`` when an MCP server hands us ``{}``.
    """
    if not isinstance(schema, dict):
        schema = {}
    out = dict(schema)
    out.setdefault("type", "object")
    props = out.get("properties")
    if not isinstance(props, dict):
        props = {}
        out["properties"] = props
    required = out.get("required")
    if isinstance(required, list):
        out["required"] = [r for r in required if isinstance(r, str) and r in props]
    else:
        out["required"] = []
    return out


def _to_ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description") or "",
            "parameters": normalize_openai_tool_parameters(tool.get("input_schema")),
        },
    }


def _from_ollama_message(message: dict[str, Any]) -> _Response:
    content: list[Any] = []
    text = str(message.get("content") or "")
    if text:
        content.append(_TextBlock(text))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or call
        raw_args = fn.get("arguments") or call.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                raw_args = {"raw": raw_args}
        content.append(
            _ToolUseBlock(
                str(call.get("id") or uuid4()),
                str(fn.get("name") or call.get("name") or "unknown"),
                raw_args if isinstance(raw_args, dict) else {},
            )
        )
    return _Response(content)


class OpenAICompatError(Exception):
    """OpenAI-compatible API error; overflow detection reads type/body/status_code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        body: dict[str, Any] | None = None,
        type: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.type = type
        # Response headers, lowercased, so retry logic can read Retry-After.
        self.headers = headers or {}


class OpenAICompatClient:
    """Anthropic-shaped client that talks to OpenAI-compatible /chat/completions."""

    def __init__(self, base_url: str, api_key: str, *, http: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = http
        self.messages = self

    async def create(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        system: Any = "",
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]],
        **_kw: Any,
    ) -> _Response:
        chosen = (model or settings.orbweaver_model).strip()
        payload: dict[str, Any] = {
            "model": chosen,
            "max_tokens": max_tokens_for(chosen, max_tokens),
            "stream": False,
            "messages": _to_openai_messages(system, messages),
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]
        stops = _kw.get("stop_sequences")
        if stops:
            payload["stop"] = stops
        data = await self._post(payload)
        return _from_openai_completion(data)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempts = max(0, int(settings.orbweaver_llm_retries)) + 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._post_once(url, headers, payload)
            except OpenAICompatError as e:
                delay = retry_delay_for(e, attempt=attempt)
                if delay is None or attempt >= attempts:
                    raise
                log.warning(
                    "llm %s failed (HTTP %s); retry %d/%d in %.2fs",
                    payload.get("model") or "?",
                    getattr(e, "status_code", "?"),
                    attempt,
                    attempts - 1,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _post_once(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        if self._http is not None:
            resp = await self._http.post(url, json=payload, headers=headers)
            data = _json_of(resp)
            status = int(getattr(resp, "status_code", 200) or 200)
            if status >= 400 or (isinstance(data, dict) and data.get("error")):
                raise _openai_error(data, max(status, 400), headers=_resp_headers(resp))
            return data
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as http:
            resp = await http.post(url, json=payload, headers=headers)
            try:
                data = resp.json()
            except Exception:
                data = {"error": {"message": resp.text}}
            if resp.status_code >= 400:
                raise _openai_error(
                    data if isinstance(data, dict) else {},
                    resp.status_code,
                    headers=_resp_headers(resp),
                )
            return data


# Transient same-model retries. 429 is the shared open-model pool shedding load;
# 500/504 are gateway-level blips. 502 is deliberately *not* retried here:
# Earth Runtime wraps context-overflow failures as 502, and compacting is the
# right response, so that judgement stays with
# compact.overflow.should_overflow_retry(). 503 (and Anthropic 529 overloaded)
# means the *model* is unavailable: the Auto router / turn loop switches to a
# different model, preferably another provider, instead of sleeping on this id.
_RETRY_STATUSES = frozenset({429, 500, 504})
_UNAVAILABLE_STATUSES = frozenset({503, 529})


def is_upstream_unavailable(exc: BaseException) -> bool:
    """True when the provider says this model cannot serve the request now.

    HTTP 503 from Earth Runtime ("The model is temporarily rate limited") and
    Anthropic 529 overloaded are the production cases. Same-model retry will
    not help; a different model (ideally a different provider) might. A daily
    spend cap on this provider is the same shape: skip it and try another.
    """
    from orbweaver.spend import SpendCapped

    if isinstance(exc, SpendCapped):
        return True
    status = int(getattr(exc, "status_code", 0) or 0)
    if status in _UNAVAILABLE_STATUSES:
        return True
    err_type = str(getattr(exc, "type", "") or "").lower()
    return err_type in {"overloaded_error", "overloaded"}


def _resp_headers(resp: Any) -> dict[str, str]:
    raw = getattr(resp, "headers", None)
    if not raw:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in dict(raw).items()}
    except Exception:
        return {}


def retry_after_seconds(exc: BaseException) -> float | None:
    """Honour a server-sent Retry-After (delta-seconds form) when present."""
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, dict):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        secs = float(str(raw).strip())
    except (TypeError, ValueError):
        return None  # HTTP-date form: fall back to our own backoff.
    return secs if secs >= 0 else None


def retry_delay_for(exc: BaseException, *, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if this error is not retryable.

    ``attempt`` is 1-based. Exponential backoff with full jitter, clamped to
    ``orbweaver_llm_retry_max_s``; a sane Retry-After wins over the backoff.
    """
    status = int(getattr(exc, "status_code", 0) or 0)
    if status not in _RETRY_STATUSES:
        return None
    ceiling = max(0.0, float(settings.orbweaver_llm_retry_max_s))
    after = retry_after_seconds(exc)
    if after is not None:
        return min(after, ceiling)
    base = max(0.0, float(settings.orbweaver_llm_retry_base_s))
    window = min(base * (2 ** (attempt - 1)), ceiling)
    # Full jitter: spread concurrent turns instead of retrying in lockstep.
    return random.uniform(0.0, window) if window > 0 else 0.0


def _openai_error(
    data: dict[str, Any], status: int, *, headers: dict[str, str] | None = None
) -> OpenAICompatError:
    err = data.get("error") if isinstance(data, dict) else None
    extra = ""
    if isinstance(err, dict):
        message = str(err.get("message") or data)
        err_type = str(err.get("code") or err.get("type") or "")
        meta = err.get("metadata")
        if isinstance(meta, dict):
            extra = str(meta.get("raw") or meta.get("provider_name") or "").strip()
            if extra and extra not in message:
                message = f"{message} ({extra[:500]})"
    else:
        message = str(err or data or f"HTTP {status}")
        err_type = ""
    return OpenAICompatError(
        message,
        status_code=status,
        body=data if isinstance(data, dict) else None,
        type=err_type,
        headers=headers,
    )


def _to_openai_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"type": "text", "text": str(block.get("text") or "")})
        elif block.get("type") == "image":
            src = block.get("source") or {}
            data = src.get("data")
            media = str(src.get("media_type") or "image/png")
            if src.get("type") == "base64" and data:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{data}"},
                    }
                )
            else:
                parts.append(
                    {
                        "type": "text",
                        "text": f"[image {src.get('path') or src.get('media_type') or 'attached'}]",
                    }
                )
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts or ""


def _to_openai_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sys_text = _system_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or str(uuid4()),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
                elif block.get("type") == "text":
                    texts.append(str(block.get("text") or ""))
            row: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if tool_calls:
                row["tool_calls"] = tool_calls
            out.append(row)
            continue
        if role == "user" and isinstance(content, list):
            tool_results = [
                b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            other = [
                b for b in content if isinstance(b, dict) and b.get("type") != "tool_result"
            ]
            if tool_results:
                for block in tool_results:
                    inner = block.get("content")
                    if isinstance(inner, list):
                        inner = _content_to_text(inner)
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id") or ""),
                            "content": str(inner or ""),
                        }
                    )
                if other:
                    mapped = _to_openai_content(other)
                    if mapped:
                        out.append({"role": "user", "content": mapped})
                continue
        out.append({"role": "user" if role == "user" else role, "content": _to_openai_content(content)})
    return out


def _from_openai_completion(data: dict[str, Any]) -> _Response:
    choices = data.get("choices") or []
    message = ((choices[0] if choices else {}) or {}).get("message") or {}
    content: list[Any] = []
    raw = message.get("content")
    if isinstance(raw, list):
        text = "\n".join(
            str(p.get("text") or "")
            for p in raw
            if isinstance(p, dict) and p.get("type") == "text"
        )
    else:
        text = str(raw or "")
    if text:
        content.append(_TextBlock(text))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or call
        raw_args = fn.get("arguments") or call.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                raw_args = {"raw": raw_args}
        content.append(
            _ToolUseBlock(
                str(call.get("id") or uuid4()),
                str(fn.get("name") or call.get("name") or "unknown"),
                raw_args if isinstance(raw_args, dict) else {},
            )
        )
    usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage_raw = usage_raw or {}
    cost = usage_raw.get("cost", data.get("cost"))
    try:
        cost_usd = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_usd = None
    usage = _Usage(
        input_tokens=int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0),
        output_tokens=int(
            usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0
        ),
        cost_usd=cost_usd,
    )
    return _Response(content, usage)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _event_type(event: Any) -> str:
    return str(_field(event, "type") or "")


def _try_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


class StreamAssembler:
    """Accumulate Anthropic SSE events into a create-shaped response."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self.usage = _Usage()
        self.stop_reason: str | None = None
        self.started = False

    def apply(self, event: Any) -> list[tuple[str, dict[str, Any]]]:
        typ = _event_type(event)
        if typ not in _STREAM_EVENT_TYPES:
            return []
        self.started = True
        if typ == "message_start":
            self._note_usage(_field(_field(event, "message"), "usage"))
            return []
        if typ == "content_block_start":
            return self._start_block(event)
        if typ == "content_block_delta":
            return self._delta_block(event)
        if typ == "content_block_stop":
            self._stop_block(event)
            return []
        if typ == "message_delta":
            self._note_usage(_field(event, "usage"))
            delta = _field(event, "delta")
            reason = _field(delta, "stop_reason")
            if reason:
                self.stop_reason = str(reason)
            return []
        return []

    def _note_usage(self, usage: Any) -> None:
        if usage is None:
            return
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            val = int(_field(usage, name, 0) or 0)
            if val:
                setattr(self.usage, name, val)
        cost = _field(usage, "cost", None)
        if cost is None:
            cost = _field(usage, "cost_usd", None)
        if cost is not None:
            try:
                self.usage.cost_usd = float(cost)
            except (TypeError, ValueError):
                pass

    def _index(self, event: Any) -> int:
        return int(_field(event, "index", 0) or 0)

    def _start_block(self, event: Any) -> list[tuple[str, dict[str, Any]]]:
        idx = self._index(event)
        block = _field(event, "content_block")
        btype = str(_field(block, "type") or "text")
        if btype == "tool_use":
            uid = str(_field(block, "id") or uuid4())
            name = str(_field(block, "name") or "unknown")
            raw_inp = _field(block, "input") or {}
            inp = raw_inp if isinstance(raw_inp, dict) else {}
            self._blocks[idx] = {
                "type": "tool_use",
                "id": uid,
                "name": name,
                "json_buf": "",
                "input": inp,
            }
            return [(TOOL_USE_PROGRESS, {"id": uid, "name": name, "input": dict(inp)})]
        text = str(_field(block, "text") or "")
        self._blocks[idx] = {"type": "text", "text": text}
        if text:
            return [(ASSISTANT_DELTA, {"text": text})]
        return []

    def _delta_block(self, event: Any) -> list[tuple[str, dict[str, Any]]]:
        idx = self._index(event)
        delta = _field(event, "delta")
        dtype = str(_field(delta, "type") or "")
        if dtype == "text_delta":
            chunk = str(_field(delta, "text") or "")
            slot = self._blocks.setdefault(idx, {"type": "text", "text": ""})
            if slot.get("type") != "text":
                slot = {"type": "text", "text": ""}
                self._blocks[idx] = slot
            slot["text"] = str(slot.get("text") or "") + chunk
            if chunk:
                return [(ASSISTANT_DELTA, {"text": chunk})]
            return []
        if dtype == "input_json_delta":
            partial = str(_field(delta, "partial_json") or "")
            slot = self._blocks.setdefault(
                idx,
                {
                    "type": "tool_use",
                    "id": str(uuid4()),
                    "name": "unknown",
                    "json_buf": "",
                    "input": {},
                },
            )
            if slot.get("type") != "tool_use":
                return []
            slot["json_buf"] = str(slot.get("json_buf") or "") + partial
            parsed = _try_json(slot["json_buf"])
            if parsed is not None:
                slot["input"] = parsed
            return [
                (
                    TOOL_USE_PROGRESS,
                    {
                        "id": slot["id"],
                        "name": slot["name"],
                        "input": dict(slot.get("input") or {}),
                    },
                )
            ]
        return []

    def _stop_block(self, event: Any) -> None:
        idx = self._index(event)
        slot = self._blocks.get(idx)
        if not slot or slot.get("type") != "tool_use":
            return
        parsed = _try_json(str(slot.get("json_buf") or ""))
        if parsed is not None:
            slot["input"] = parsed

    def text(self) -> str:
        parts = [
            str(b.get("text") or "")
            for i, b in sorted(self._blocks.items())
            if b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)

    def tool_use_blocks(self) -> list[_ToolUseBlock]:
        out: list[_ToolUseBlock] = []
        for _i, b in sorted(self._blocks.items()):
            if b.get("type") != "tool_use" or not b.get("id"):
                continue
            inp = b.get("input")
            if not isinstance(inp, dict):
                parsed = _try_json(str(b.get("json_buf") or ""))
                inp = parsed if isinstance(parsed, dict) else {}
            out.append(_ToolUseBlock(str(b["id"]), str(b.get("name") or "unknown"), inp))
        return out

    def response(self) -> _Response:
        content: list[Any] = []
        for _i, b in sorted(self._blocks.items()):
            if b.get("type") == "text":
                text = str(b.get("text") or "")
                if text:
                    content.append(_TextBlock(text))
            elif b.get("type") == "tool_use" and b.get("id"):
                inp = b.get("input")
                if not isinstance(inp, dict):
                    parsed = _try_json(str(b.get("json_buf") or ""))
                    inp = parsed if isinstance(parsed, dict) else {}
                content.append(
                    _ToolUseBlock(str(b["id"]), str(b.get("name") or "unknown"), inp)
                )
        return _Response(content, self.usage)


def client_can_stream(client: Any) -> bool:
    return callable(getattr(getattr(client, "messages", None), "stream", None))


def message_stream(client: Any, **kwargs: Any) -> Any | None:
    """Return a stream context manager, or None if the client cannot stream.

    Open failures propagate so a long request does not silently fall back to
    non-streaming ``create()``, which Anthropic rejects above ~21k max_tokens.
    """
    stream_fn = getattr(getattr(client, "messages", None), "stream", None)
    if not callable(stream_fn):
        return None
    return stream_fn(**kwargs)
