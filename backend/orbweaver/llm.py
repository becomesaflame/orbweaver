"""LLM provider selection. Anthropic is the default; Ollama and Earth Runtime are fallbacks."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from orbweaver.config import settings
from orbweaver.open_models import (
    DEFAULT_OPENROUTER_BASE,
    is_claude_model,
    is_open_model,
    max_tokens_for,
)


def _ollama_configured() -> bool:
    return bool(settings.ollama_base_url.strip() and settings.ollama_model.strip())


def openrouter_configured() -> bool:
    return bool(settings.openrouter_key)


def select_provider(model: str | None = None) -> str:
    """Route the agent model. Open catalog names use Earth Runtime even if Claude is set."""
    model = (model or settings.orbweaver_model).strip()
    if openrouter_configured() and not is_claude_model(model):
        return "openrouter"
    if settings.anthropic_api_key.strip():
        return "anthropic"
    if openrouter_configured():
        return "openrouter"
    if _ollama_configured():
        return "ollama"
    return "none"


def no_llm_echo(user_text: str) -> str:
    model = settings.orbweaver_model.strip()
    echo = (user_text or "")[:500]
    if is_open_model(model) and not openrouter_configured():
        return (
            f"OPENROUTER_API_KEY is not set (needed for {model}). Echo: {echo}\n"
            "Get a key from earthruntime.com and set OPENROUTER_API_KEY "
            "(optional OPENROUTER_BASE_URL, default https://api.earthruntime.com/v1)."
        )
    return (
        "No LLM provider is configured. Echo: "
        + echo
        + "\nSet ANTHROPIC_API_KEY for Claude, OPENROUTER_API_KEY for Earth Runtime "
        "open models, or OLLAMA_BASE_URL and OLLAMA_MODEL for a local Ollama fallback."
    )


def make_anthropic_client() -> Any | None:
    if not settings.anthropic_api_key.strip():
        return None
    import anthropic

    headers = {}
    if settings.anthropic_workspace_id.strip():
        headers["anthropic-workspace-id"] = settings.anthropic_workspace_id.strip()
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, default_headers=headers or None
    )


def make_agent_client(*, http: Any | None = None) -> Any | None:
    """Anthropic for Claude; Earth Runtime for open models; Ollama as last resort."""
    provider = select_provider()
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


def compact_llm_client(agent_client: Any | None) -> Any | None:
    """Keep compact/summarize on Anthropic when both providers are configured."""
    if isinstance(agent_client, OpenAICompatClient) and settings.anthropic_api_key.strip():
        anth = make_anthropic_client()
        if anth is not None:
            return anth
    return agent_client


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
    def __init__(self, input_tokens: int = 0) -> None:
        self.input_tokens = input_tokens


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


def _to_ollama_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name"),
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
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
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.type = type


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
        data = await self._post(payload)
        return _from_openai_completion(data)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self._http is not None:
            resp = await self._http.post(url, json=payload, headers=headers)
            data = _json_of(resp)
            status = int(getattr(resp, "status_code", 200) or 200)
            if status >= 400 or (isinstance(data, dict) and data.get("error")):
                raise _openai_error(data, max(status, 400))
            return data
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as http:
            resp = await http.post(url, json=payload, headers=headers)
            try:
                data = resp.json()
            except Exception:
                data = {"error": {"message": resp.text}}
            if resp.status_code >= 400:
                raise _openai_error(data if isinstance(data, dict) else {}, resp.status_code)
            return data


def _openai_error(data: dict[str, Any], status: int) -> OpenAICompatError:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        message = str(err.get("message") or data)
        err_type = str(err.get("code") or err.get("type") or "")
    else:
        message = str(err or data or f"HTTP {status}")
        err_type = ""
    return OpenAICompatError(
        message, status_code=status, body=data if isinstance(data, dict) else None, type=err_type
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
    usage_raw = data.get("usage") or {}
    usage = _Usage(
        input_tokens=int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0)
    )
    return _Response(content, usage)

