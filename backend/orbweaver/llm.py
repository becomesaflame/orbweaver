"""LLM provider selection. Anthropic is the default; Ollama is an optional fallback."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from orbweaver.config import settings

SONNET_MAX_TOKENS = 64_000
OPUS_MAX_TOKENS = 32_000
DEFAULT_MAX_TOKENS = 8_192


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


def select_provider() -> str:
    if settings.anthropic_api_key.strip():
        return "anthropic"
    if settings.ollama_base_url.strip() and settings.ollama_model.strip():
        return "ollama"
    return "none"


def no_llm_echo(user_text: str) -> str:
    return (
        "ANTHROPIC_API_KEY is not set. Echo: "
        + (user_text or "")[:500]
        + "\nSet the key to enable the Claude tool loop, or set OLLAMA_BASE_URL and "
        "OLLAMA_MODEL for a local Ollama fallback."
    )


def make_agent_client(*, http: Any | None = None) -> Any | None:
    """Anthropic when the key is set; otherwise Ollama if configured."""
    provider = select_provider()
    if provider == "anthropic":
        import anthropic

        headers = {}
        if settings.anthropic_workspace_id.strip():
            headers["anthropic-workspace-id"] = settings.anthropic_workspace_id.strip()
        return anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key, default_headers=headers or None
        )
    if provider == "ollama":
        return OllamaMessagesClient(
            settings.ollama_base_url.strip(),
            settings.ollama_model.strip(),
            http=http,
        )
    return None


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
