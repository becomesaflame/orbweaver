"""Telegram adapter: one operator session per user, routed through the session router.

Each allowlisted Telegram user owns one *operator* session. Plain messages run
on that session unless the chat is *attached* to another session
(``/attach``, or the operator's ``AttachSession`` tool), in which case they run
``agent_turn`` on the target: the same event stream the web UI or VS Code is
looking at. A chat sink registered on both sessions pushes approval keyboards
and finished-turn summaries here, so a turn started at the desk reports to the
phone. Turns run as background tasks; ``/stop`` cancels the current one.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from orbweaver import __version__
from orbweaver.agent import (
    TurnCancelled,
    agent_turn,
    pending_approvals,
    pending_ask_user,
    resolve_approval,
)
from orbweaver.auth import mint_token
from orbweaver.channels import router
from orbweaver.channels.router import (
    ATTACHED_TURN_HINT,
    OPERATOR_ROLE,
    OPERATOR_SYSTEM_EXTRA,
    AmbiguousSessionRef,
    RouterError,
)
from orbweaver.config import settings
from orbweaver.image import (
    PHOTO_MEDIA_TYPES,
    generate_image_bytes,
    is_image_path,
    save_inbound_image,
    sniff_media_type,
)
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Store,
    get_store,
    new_uuid,
    session_at_id,
)
from orbweaver.turns import GatewayDraining, RunningTurn
from orbweaver.turns import acquire as acquire_turn
from orbweaver.turns import get as get_running_turn
from orbweaver.turns import release as release_turn
from orbweaver.workspace import WORKSPACE_KIND_LOCAL, apply_local_workspace_kind, bind_workspace

log = logging.getLogger(__name__)

TELEGRAM_IMAGE_HINT = (
    "The user may send photos; those arrive as images you can see. "
    "To send a photo on Telegram, write an image file in the workspace and call SendPhoto. "
    "GenerateImage creates a file and, on Telegram sessions, sends it to the chat."
)
CHANNEL = "telegram"

# chat_id -> sessions whose router sink delivers to that chat (operator + attached target).
_bound: dict[int, set[UUID]] = {}
# Background turn and delivery tasks, kept alive until done.
_tasks: set[asyncio.Task[Any]] = set()


def user_allowed(user_id: int) -> bool:
    ids = settings.telegram_user_ids
    return not ids or user_id in ids


def texts_for_reply(events) -> str:
    """Telegram/cron get the turn's last user-visible text, not every thought."""
    last = ""
    for e in events:
        text = (e.payload or {}).get("text")
        if text and e.kind in {"assistant", "turn_aborted"}:
            last = str(text)
        elif e.kind == "ask_user":
            question = (e.payload or {}).get("question")
            if question:
                last = str(question)
    return last[:3500]


async def notify_telegram_chat(chat_id: int, text: str) -> None:
    token = settings.telegram_bot_token
    if not token or not text:
        return
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3500]},
        )


APPROVAL_CALLBACK_PREFIX = "apr"
_APPROVAL_BUTTONS: tuple[tuple[str, str, str], ...] = (
    ("Allow", "allow", "once"),
    ("Allow for session", "allow", "session"),
    ("Deny", "deny", "once"),
)


def approval_request_text(payload: dict) -> str:
    name = str(payload.get("name") or "tool")
    summary = str(payload.get("summary") or "")
    reason = str(payload.get("reason") or "")
    lines = [f"Approval needed: {name}"]
    if summary:
        lines.append(summary[:1500])
    if reason:
        lines.append(f"Why: {reason[:600]}")
    return "\n".join(lines)[:3500]


def approval_keyboard(tool_use_id: str) -> dict:
    """Telegram reply_markup with Allow / Allow for session / Deny buttons."""
    row = []
    for label, decision, scope in _APPROVAL_BUTTONS:
        data = f"{APPROVAL_CALLBACK_PREFIX}:{decision}:{scope}:{tool_use_id}"
        row.append({"text": label, "callback_data": data[:64]})
    return {"inline_keyboard": [row]}


def parse_approval_callback(data: str) -> tuple[str, str, str] | None:
    """(decision, scope, tool_use_id) from an inline-keyboard callback, or None."""
    parts = str(data or "").split(":", 3)
    if len(parts) != 4 or parts[0] != APPROVAL_CALLBACK_PREFIX:
        return None
    _prefix, decision, scope, tool_use_id = parts
    if decision not in {"allow", "deny"} or scope not in {"once", "session"} or not tool_use_id:
        return None
    return decision, scope, tool_use_id


async def notify_telegram_approval(chat_id: int, payload: dict) -> None:
    token = settings.telegram_bot_token
    tool_use_id = str(payload.get("tool_use_id") or "")
    if not token or not tool_use_id:
        return
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": approval_request_text(payload),
                "reply_markup": approval_keyboard(tool_use_id),
            },
        )


# ------------------------------------------------------------------ chat sinks


def _spawn(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)

    def _done(t: asyncio.Task[Any]) -> None:
        _tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("telegram task %s failed", name, exc_info=t.exception())

    task.add_done_callback(_done)
    return task


def _sink_key(chat_id: int) -> str:
    return f"telegram:{chat_id}"


def chats_for_session(session_id: UUID) -> list[int]:
    """Chats whose sink is bound to ``session_id`` (operator or attached)."""
    return sorted(chat for chat, sessions in _bound.items() if session_id in sessions)


def _turn_status_text(status: str) -> str:
    return {
        "stopped": "Turn stopped.",
        "discarded": "Turn discarded.",
        "error": "Turn failed.",
        "waiting_ask": "Waiting for your answer.",
    }.get(status, "Turn finished.")


async def deliver_turn_result(chat_id: int, session_id: UUID, done: dict[str, Any]) -> None:
    """A turn someone else started on a bound session finished: report it here."""
    store = get_store()
    events = await store.list_events(session_id)
    try:
        user_seq = int(done.get("user_seq") or 0)
    except (TypeError, ValueError):
        user_seq = 0
    tail = [e for e in events if e.seq > user_seq]
    text = texts_for_reply(tail) or _turn_status_text(str(done.get("status") or ""))
    sess = await store.get_entity(session_id)
    title = router.display_title(sess.jsonld or {}, events) if sess else str(session_id)[:8]
    await notify_telegram_chat(chat_id, f"[{title}]\n{text}")


def make_chat_sink(chat_id: int, session_id: UUID) -> router.Sink:
    """Router sink: approval keyboards for any turn; results for turns run elsewhere.

    ``turn_done`` frames from this adapter carry ``channel: telegram`` and are
    skipped because the handler already replied to the user's message.
    """

    def sink(msg: dict[str, Any]) -> None:
        kind = msg.get("kind")
        if kind == "permission_request":
            payload = dict(msg.get("payload") or {})
            _spawn(notify_telegram_approval(chat_id, payload), f"tg-approval-{chat_id}")
        elif kind == "turn_done" and msg.get("channel") != CHANNEL:
            _spawn(deliver_turn_result(chat_id, session_id, dict(msg)), f"tg-deliver-{chat_id}")

    return sink


def bind_chat(chat_id: int, sessions: set[UUID]) -> None:
    """Make ``chat_id``'s sink follow exactly ``sessions``."""
    key = _sink_key(chat_id)
    current = _bound.get(chat_id, set())
    for sid in current - sessions:
        router.remove_sink(sid, key)
    for sid in sessions - current:
        router.add_sink(sid, key, make_chat_sink(chat_id, sid))
    if sessions:
        _bound[chat_id] = set(sessions)
    else:
        _bound.pop(chat_id, None)


def _jsonld_chat_id(jsonld: dict[str, Any]) -> int | None:
    raw = jsonld.get("telegram_chat_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def sync_chat_sinks(operator: Entity) -> None:
    """Bind the operator's chat to its own session plus the attached target, if any."""
    chat_id = _jsonld_chat_id(operator.jsonld or {})
    if chat_id is None:
        return
    wanted = {operator.id}
    target = router.attached_id(operator)
    if target is not None:
        wanted.add(target)
    bind_chat(chat_id, wanted)


def _on_attach_changed(operator: Entity, _target: UUID | None) -> None:
    sync_chat_sinks(operator)


def install_router_hooks() -> None:
    router.add_attach_hook(_on_attach_changed)


async def rehydrate_chat_sinks(store: Store) -> int:
    """After a restart, rebuild sinks from persisted attach pointers."""
    count = 0
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") is None:
            continue
        sync_chat_sinks(ent)
        count += 1
    return count


def reset_for_tests() -> None:
    _bound.clear()
    _tasks.clear()


# -------------------------------------------------------------- photos / images


def _telegram_chat_id(sess: Entity | None) -> int | None:
    """Chat bound to ``sess``: its own Telegram binding, else a chat attached to it."""
    if not sess:
        return None
    own = _jsonld_chat_id(sess.jsonld or {})
    if own is not None:
        return own
    chats = chats_for_session(sess.id)
    return chats[0] if chats else None


async def send_session_photo(ctx: dict, inp: dict) -> str:
    path = str(inp.get("path") or "").strip()
    caption = str(inp.get("caption") or "")[:1024]
    if not path or not is_image_path(path):
        return "path must be a workspace image (.jpg, .jpeg, .png, .webp, .gif)"
    ws = ctx["workspace"]
    try:
        data = ws.read_bytes(path)
    except (OSError, PermissionError, FileNotFoundError, IsADirectoryError) as e:
        return f"read failed: {e}"
    store = ctx["store"]
    sess = await store.get_entity(ctx["session_id"])
    chat_id = _telegram_chat_id(sess)
    if chat_id is None:
        return f"no telegram chat bound to this session; image remains at {path}"
    media_type = sniff_media_type(data)
    if media_type == "application/octet-stream":
        media_type = "image/jpeg"
    result = await notify_telegram_photo(chat_id, data, caption, Path(path).name, media_type)
    if result != "ok":
        return result
    return json.dumps({"sent": True, "path": path})


async def notify_telegram_photo(
    chat_id: int,
    data: bytes,
    caption: str = "",
    filename: str = "photo.jpg",
    media_type: str = "image/jpeg",
) -> str:
    token = settings.telegram_bot_token
    if not token or not data:
        return "telegram bot token missing or empty image"
    import httpx

    field = "photo" if media_type in PHOTO_MEDIA_TYPES else "document"
    endpoint = "sendPhoto" if field == "photo" else "sendDocument"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{token}/{endpoint}",
            data={"chat_id": str(chat_id), "caption": (caption or "")[:1024]},
            files={field: (filename, data, media_type)},
        )
        if r.status_code >= 400:
            return f"telegram {endpoint} failed: {r.status_code} {r.text[:300]}"
    return "ok"


async def generate_and_maybe_send(ctx: dict, inp: dict) -> str:
    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        return "prompt is required"
    if not settings.orbweaver_image_api_key.strip():
        return (
            "ORBWEAVER_IMAGE_API_KEY is not set. Create an image file in attachments/ "
            "(for example with code) and call SendPhoto."
        )
    try:
        data, media_type = generate_image_bytes(prompt)
    except Exception as e:
        return f"image generation failed: {e}"
    default_ext = ".png" if media_type == "image/png" else ".jpg"
    rel = str(inp.get("path") or "").strip() or f"attachments/generated-{new_uuid()}{default_ext}"
    if not is_image_path(rel):
        rel = str(Path(rel).with_suffix(default_ext))
    ws = ctx["workspace"]
    ws.write_bytes(rel, data)
    payload: dict = {"path": rel, "media_type": media_type}
    store = ctx["store"]
    sess = await store.get_entity(ctx["session_id"])
    chat_id = _telegram_chat_id(sess)
    caption = str(inp.get("caption") or prompt)[:1024]
    if chat_id is not None:
        sent = await notify_telegram_photo(chat_id, data, caption, Path(rel).name, media_type)
        payload["sent"] = sent == "ok"
        if sent != "ok":
            payload["send_error"] = sent
    else:
        payload["sent"] = False
    return json.dumps(payload)


# ------------------------------------------------------------ operator session


def _normalize_operator(jsonld: dict[str, Any], chat_id: int | None) -> bool:
    """Bring an operator session's JSON-LD up to date. Returns True when it changed."""
    changed = apply_local_workspace_kind(jsonld)
    if chat_id is not None and jsonld.get("telegram_chat_id") != chat_id:
        jsonld["telegram_chat_id"] = chat_id
        changed = True
    if jsonld.get("channel") != CHANNEL:
        jsonld["channel"] = CHANNEL
        changed = True
    if jsonld.get("role") != OPERATOR_ROLE:
        jsonld["role"] = OPERATOR_ROLE
        changed = True
    return changed


async def session_for_telegram_user(
    store: Store, user_id: int, chat_id: int | None = None
) -> Entity:
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") == user_id:
            if _normalize_operator(ent.jsonld, chat_id):
                await store.put_entity(ent)
            return ent
    uid = new_uuid()
    ent = Entity(
        id=uid,
        at_id=session_at_id(uid),
        at_type=SESSION_TYPE,
        jsonld={
            "@id": session_at_id(uid),
            "@type": SESSION_TYPE,
            "workspace_uri": "workspace:default",
            "workspace_kind": WORKSPACE_KIND_LOCAL,
            "title": f"telegram:{user_id}",
            "channel": CHANNEL,
            "role": OPERATOR_ROLE,
            "telegram_user_id": user_id,
            "telegram_chat_id": chat_id if chat_id is not None else user_id,
            "status": "active",
        },
    )
    await store.put_entity(ent)
    return ent


@dataclass
class Bound:
    """Where a Telegram message runs: the operator's own session or its attached target."""

    store: Store
    operator: Entity
    target: Entity
    ws: Any
    kind: str
    chat_id: int | None

    @property
    def session_id(self) -> UUID:
        return self.target.id

    @property
    def attached(self) -> bool:
        return self.target.id != self.operator.id


def _user_data(context) -> dict[str, Any]:
    data = getattr(context, "user_data", None)
    return data if isinstance(data, dict) else {}


async def _operator_session(update, context) -> tuple[Store, Entity, int | None]:
    store = get_store()
    chat = getattr(update, "effective_chat", None)
    chat_id = chat.id if chat is not None else None
    data = _user_data(context)
    ent: Entity | None = None
    sid = data.get("session_id")
    if sid:
        try:
            ent = await store.get_entity(UUID(str(sid)))
        except ValueError:
            ent = None
        if ent is not None and _normalize_operator(ent.jsonld, chat_id):
            await store.put_entity(ent)
    if ent is None:
        ent = await session_for_telegram_user(store, update.effective_user.id, chat_id)
        data["session_id"] = str(ent.id)
    install_router_hooks()
    sync_chat_sinks(ent)
    return store, ent, chat_id


async def _session_workspace(update, context, *, operator_only: bool = False) -> Bound:
    store, operator, chat_id = await _operator_session(update, context)
    target = operator if operator_only else await router.resolve_target(store, operator)
    ws, kind, kind_changed = bind_workspace(
        target.jsonld, settings.workspace_root, session_key=str(target.id)
    )
    if kind_changed:
        await store.put_entity(target)
    return Bound(store=store, operator=operator, target=target, ws=ws, kind=kind, chat_id=chat_id)


def _busy_reply(state: RunningTurn) -> str:
    via = f" ({state.channel})" if state.channel else ""
    return (
        f"A turn is already running on this session{via}; your message was added to it. "
        "Send /stop to cancel it, or wait for it to finish."
    )


def _title_of(ent: Entity, events) -> str:
    return router.display_title(ent.jsonld or {}, events)


# ------------------------------------------------------------------- turns


async def _run_turn(
    update,
    context,
    text: str,
    images: list[dict[str, str]] | None = None,
    *,
    operator_only: bool = False,
) -> None:
    state: RunningTurn | None = None
    session_id: UUID | None = None
    status = "error"
    # Always set before the post-finally reply so CancelledError (BaseException)
    # during a gateway restart cannot leave ``reply`` unbound and swallow the turn.
    reply = "Turn interrupted."
    try:
        bound = await _session_workspace(update, context, operator_only=operator_only)
        store, session_id = bound.store, bound.session_id
        try:
            state = acquire_turn(session_id, channel=CHANNEL)
        except GatewayDraining:
            reply = "Orbweaver is updating; send that again in a minute."
            status = "stopped"
            return
        if state is None:
            # Same path as POST /turns/inject: the running turn picks the text up
            # on its next LLM call instead of a second agent_turn racing it.
            from orbweaver.app import inject_into_turn

            running = get_running_turn(session_id)
            if running is None:
                reply = "A turn is already running on this session; send that again in a moment."
            else:
                await inject_into_turn(
                    store, session_id, running, text, images, via=CHANNEL if bound.attached else None
                )
                reply = _busy_reply(running)
            log.info("telegram: session %s busy, message injected into running turn", session_id)
        else:
            extra = TELEGRAM_IMAGE_HINT
            tools = None
            via = None
            if bound.attached:
                extra += " " + ATTACHED_TURN_HINT
                via = CHANNEL
            else:
                extra += " " + OPERATOR_SYSTEM_EXTRA
                tools = await router.operator_tools(bound.ws, CHANNEL)
            events = await agent_turn(
                store,
                session_id,
                text,
                bound.ws,
                workspace_kind=bound.kind,
                emit=router.emitter(session_id),
                headless=True,
                interactive=True,
                images=images,
                system_extra=extra,
                channel=CHANNEL,
                cancel=state.cancel,
                turn_state=state,
                tools=tools,
                via=via,
            )
            reply = texts_for_reply(events) or "(no assistant text)"
            status = "waiting_ask" if pending_ask_user(await store.list_events(session_id)) else "ok"
    except TurnCancelled as e:
        # Stop from the web UI (or /stop) reaches Telegram turns through the shared registry.
        status = "stopped"
        if session_id is not None:
            marker = await store.append_event(session_id, "turn_interrupted", {"reason": "stop"})
            router.emit(
                session_id,
                {"kind": marker.kind, "payload": marker.payload, "id": str(marker.id), "seq": marker.seq},
            )
        reply = texts_for_reply(e.produced) or "Turn stopped."
    except asyncio.CancelledError:
        # Deploy restart / task cancel: still try to tell the chat what happened.
        status = "stopped"
        reply = "Turn interrupted (stop or gateway restart)."
        if session_id is not None:
            try:
                store = get_store()
                marker = await store.append_event(session_id, "turn_interrupted", {"reason": "cancelled"})
                router.emit(
                    session_id,
                    {
                        "kind": marker.kind,
                        "payload": marker.payload,
                        "id": str(marker.id),
                        "seq": marker.seq,
                    },
                )
            except Exception:
                log.exception("telegram: could not record cancelled turn")
        raise
    except Exception as e:
        log.exception("telegram turn failed")
        reply = f"Turn failed: {e}"[:3500]
    finally:
        if state is not None and session_id is not None:
            release_turn(session_id, state)
            router.emit(
                session_id,
                {"kind": "turn_done", "status": status, "user_seq": state.user_seq, "channel": CHANNEL},
            )
        if update.message:
            try:
                await update.message.reply_text(reply)
            except Exception:
                log.exception("telegram reply failed")


def start_turn(update, context, text: str, images: list[dict[str, str]] | None = None, **kw) -> asyncio.Task[Any]:
    """Run the turn off the handler so /stop, approvals, and new messages keep flowing."""
    return _spawn(_run_turn(update, context, text, images, **kw), "tg-turn")


# ---------------------------------------------------------------- command text


def _session_model_line(sess: Entity, *, channel: str = CHANNEL) -> str:
    from orbweaver.model_routing import (
        channel_default_model,
        format_model_id,
        resolve_turn_model,
    )

    stored = str(sess.jsonld.get("model") or "").strip()
    effective = resolve_turn_model(session=sess.jsonld, channel=channel)
    shown = format_model_id(effective)
    if stored:
        return f"model: {shown}"
    default = format_model_id(channel_default_model(channel)) or shown
    return f"model: default ({default})"


async def status_text(bound: Bound) -> str:
    store = bound.store
    lines = [f"Orbweaver {__version__}"]
    if bound.attached:
        events = await store.list_events(bound.target.id)
        lines.append(f"attached to: {_title_of(bound.target, events)} ({str(bound.target.id)[:8]})")
        lines.append(f"workspace: {bound.target.jsonld.get('workspace_uri') or 'workspace:default'}")
    else:
        lines.append(f"own session {str(bound.operator.id)[:8]} (workspace:default); not attached")
        events = await store.list_events(bound.operator.id)
    lines.append(_session_model_line(bound.target))
    state = get_running_turn(bound.session_id)
    if state is not None:
        lines.append(f"a turn is running now (via {state.channel or 'unknown'})")
    pending = pending_ask_user(events)
    if pending is not None:
        question = str(((pending.payload or {}).get("input") or {}).get("question") or "")
        lines.append(f"waiting for your answer: {question}" if question else "waiting for your answer")
    if bound.attached:
        lines.append("/detach to return to your own session")
    else:
        lines.append("/sessions to see what you can attach to")
    return "\n".join(lines)


async def sessions_text(bound: Bound) -> str:
    rows = await router.list_sessions(bound.store, exclude={bound.operator.id})
    body = router.format_session_list(rows, router.attached_id(bound.operator))
    return body + "\n\n/attach <id or title> to continue one here."


async def attach_text(bound: Bound, ref: str) -> str:
    store = bound.store
    ref = ref.strip()
    if not ref:
        return await sessions_text(bound)
    try:
        target = await router.find_session(store, ref, exclude={bound.operator.id})
    except AmbiguousSessionRef as e:
        rows = [await router.session_row(store, m) for m in e.matches]
        return "Several sessions match; pick one by id:\n" + router.format_session_list(rows)
    if target is None:
        return f"No session matches '{ref}'. /sessions lists them."
    try:
        await router.attach(store, bound.operator, target.id)
    except RouterError as e:
        return f"Cannot attach: {e}"
    digest = await router.session_digest(store, target.id)
    return (
        "Attached. Your messages now run on this session; /detach to come back.\n\n"
        + router.format_digest(digest)
    )[:3500]


async def detach_text(bound: Bound) -> str:
    previous = await router.detach(bound.store, bound.operator)
    if previous is None:
        return "Not attached; you are on your own session."
    return f"Detached from {str(previous)[:8]}. Back on your own session."


def _catalog_listing(catalog: list[dict], current: str) -> str:
    from orbweaver.model_routing import model_label

    lines: list[str] = []
    for row in catalog:
        mid = str(row.get("id") or "")
        if not mid:
            continue
        mark = ">" if mid == current else " "
        lab = str(row.get("label") or model_label(mid))
        suffix = " (no key)" if row.get("available") is False else ""
        if lab and lab != mid:
            lines.append(f"{mark} {mid} — {lab}{suffix}")
        else:
            lines.append(f"{mark} {mid}{suffix}")
    return "\n".join(lines)


async def model_text(bound: Bound, args: str) -> str:
    """``/model`` lists; ``/model <id>`` sets this chat; ``/model default`` clears."""
    from orbweaver.model_routing import (
        channel_default_model,
        format_model_id,
        resolve_model_pick,
        store_session_model,
        supported_models,
    )

    sess = bound.target
    channel = CHANNEL
    catalog = supported_models()
    ids = [str(row["id"]) for row in catalog if row.get("id")]
    stored = str(sess.jsonld.get("model") or "").strip()
    default = channel_default_model(channel)
    query = args.strip()
    if not query:
        header = [
            _session_model_line(sess),
            f"channel default: {format_model_id(default)}",
            "",
            "Available:",
            _catalog_listing(catalog, stored or default),
            "",
            "/model <id> to switch · /model default to use the channel default",
        ]
        if bound.attached:
            events = await bound.store.list_events(sess.id)
            header.insert(0, f"this chat: {_title_of(sess, events)}")
        return "\n".join(header)[:3500]

    chosen, matches = resolve_model_pick(query, ids)
    if chosen == "":
        await store_session_model(bound.store, sess, "")
        return f"This chat now uses the default: {format_model_id(default)}."
    if chosen is None:
        if not matches:
            return f"Unknown model '{query}'. /model lists the supported ids."
        listed = "\n".join(matches)
        return f"Several models match '{query}'; pick one by id:\n{listed}"
    row = next((r for r in catalog if r.get("id") == chosen), None)
    if row is not None and row.get("available") is False:
        return f"{format_model_id(chosen)} has no API key configured. /model lists what's available."
    await store_session_model(bound.store, sess, chosen)
    return f"This chat now uses {format_model_id(chosen)}."


def _approval_session(candidates: list[UUID], tool_use_id: str) -> UUID | None:
    for sid in candidates:
        if any(p.tool_use_id == tool_use_id for p in pending_approvals(sid)):
            return sid
    return None


# ----------------------------------------------------------------- application


async def start_telegram() -> None:
    from telegram import Update
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    install_router_hooks()
    try:
        bound_count = await rehydrate_chat_sinks(get_store())
        log.info("telegram: rebuilt chat sinks for %d operator session(s)", bound_count)
    except Exception:
        log.exception("telegram: rehydrating chat sinks failed")

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .build()
    )

    def _args_text(context) -> str:
        args = getattr(context, "args", None) or []
        return " ".join(str(a) for a in args).strip()

    async def _guard(update: Update) -> bool:
        if not update.effective_user or not update.message:
            return False
        if not user_allowed(update.effective_user.id):
            await update.message.reply_text("not allowlisted")
            return False
        return True

    async def on_start(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message and update.effective_user
        bound = await _session_workspace(update, context)
        mint_token(f"telegram:{update.effective_user.id}", extra={"channel": CHANNEL})
        await update.message.reply_text(f"session {bound.operator.id}\n" + await status_text(bound))
        log.info("telegram session %s jwt issued", bound.operator.id)

    async def on_version(update: Update, context) -> None:
        del context
        if not await _guard(update):
            return
        assert update.message
        await update.message.reply_text(f"Orbweaver {__version__}")

    async def on_status(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message
        bound = await _session_workspace(update, context)
        await update.message.reply_text(await status_text(bound))

    async def on_sessions(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message
        bound = await _session_workspace(update, context, operator_only=True)
        await update.message.reply_text(await sessions_text(bound))

    async def on_attach(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message
        bound = await _session_workspace(update, context, operator_only=True)
        await update.message.reply_text(await attach_text(bound, _args_text(context)))

    async def on_detach(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message
        bound = await _session_workspace(update, context, operator_only=True)
        await update.message.reply_text(await detach_text(bound))

    async def on_model(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message
        bound = await _session_workspace(update, context)
        await update.message.reply_text(await model_text(bound, _args_text(context)))

    async def on_stop(update: Update, context) -> None:
        if not await _guard(update):
            return
        assert update.message
        bound = await _session_workspace(update, context)
        state = get_running_turn(bound.session_id)
        if state is None:
            await update.message.reply_text("Nothing is running on this session.")
            return
        state.cancel.set()
        await update.message.reply_text("Stopping the running turn.")

    async def on_operator(update: Update, context) -> None:
        """/op <text>: talk to the operator even while attached."""
        if not await _guard(update):
            return
        assert update.message
        text = _args_text(context)
        if not text:
            await update.message.reply_text("Usage: /op <message for the operator>")
            return
        start_turn(update, context, text, operator_only=True)

    async def on_text(update: Update, context) -> None:
        if not update.effective_user or not update.message or not update.message.text:
            return
        if not user_allowed(update.effective_user.id):
            return
        start_turn(update, context, update.message.text)

    async def on_photo(update: Update, context) -> None:
        if not update.effective_user or not update.message or not update.message.photo:
            return
        if not user_allowed(update.effective_user.id):
            return
        largest = update.message.photo[-1]
        try:
            file = await largest.get_file()
            data = bytes(await file.download_as_bytearray())
        except Exception as e:
            await update.message.reply_text(f"photo download failed: {e}")
            return
        bound = await _session_workspace(update, context)
        try:
            img = save_inbound_image(bound.ws, data, f"photo_{update.message.message_id}")
        except Exception as e:
            await update.message.reply_text(f"photo processing failed: {e}")
            return
        caption = (update.message.caption or "").strip() or "[Photo]"
        start_turn(update, context, caption, images=[img])

    async def on_image_document(update: Update, context) -> None:
        if not update.effective_user or not update.message or not update.message.document:
            return
        if not user_allowed(update.effective_user.id):
            return
        doc = update.message.document
        try:
            file = await doc.get_file()
            data = bytes(await file.download_as_bytearray())
        except Exception as e:
            await update.message.reply_text(f"image download failed: {e}")
            return
        bound = await _session_workspace(update, context)
        stem = (doc.file_name or f"image_{update.message.message_id}").rsplit(".", 1)[0]
        try:
            img = save_inbound_image(bound.ws, data, stem)
        except Exception as e:
            await update.message.reply_text(f"image processing failed: {e}")
            return
        caption = (update.message.caption or "").strip() or f"[Image: {img['path']}]"
        start_turn(update, context, caption, images=[img])

    async def on_voice(update: Update, context) -> None:
        if not update.message or not update.message.voice:
            return
        if update.effective_user and not user_allowed(update.effective_user.id):
            return
        file = await update.message.voice.get_file()
        data = await file.download_as_bytearray()
        try:
            from orbweaver.stt import transcribe_bytes

            text = transcribe_bytes(bytes(data), "voice.ogg")
        except Exception as e:
            await update.message.reply_text(f"STT failed: {e}")
            return
        if not update.effective_user:
            return
        start_turn(update, context, text)

    async def on_approval(update: Update, context) -> None:
        query = update.callback_query
        if query is None or not update.effective_user:
            return
        if not user_allowed(update.effective_user.id):
            await query.answer("not allowlisted")
            return
        parsed = parse_approval_callback(query.data or "")
        if parsed is None:
            await query.answer()
            return
        decision, scope, tool_use_id = parsed
        _store, operator, _chat_id = await _operator_session(update, context)
        candidates = [operator.id]
        attached = router.attached_id(operator)
        if attached is not None:
            candidates.insert(0, attached)
        target_sid = _approval_session(candidates, tool_use_id)
        resolved = target_sid is not None and resolve_approval(target_sid, tool_use_id, decision, scope)
        if resolved:
            label = {
                ("allow", "once"): "Allowed",
                ("allow", "session"): "Allowed for this session",
                ("deny", "once"): "Denied",
            }.get((decision, scope), decision)
        else:
            label = "No longer pending (timed out, cancelled, or already answered)"
        await query.answer(label)
        try:
            original = str(getattr(query.message, "text", None) or "")
            await query.edit_message_text(f"{original}\n\n→ {label}"[:4000])
        except Exception:
            log.debug("telegram approval message edit failed", exc_info=True)

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("version", on_version))
    app.add_handler(CommandHandler("status", on_status))
    app.add_handler(CommandHandler("sessions", on_sessions))
    app.add_handler(CommandHandler("attach", on_attach))
    app.add_handler(CommandHandler("detach", on_detach))
    app.add_handler(CommandHandler(["model", "models"], on_model))
    app.add_handler(CommandHandler("stop", on_stop))
    app.add_handler(CommandHandler(["op", "operator"], on_operator))
    app.add_handler(
        CallbackQueryHandler(on_approval, pattern=rf"^{APPROVAL_CALLBACK_PREFIX}:")
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, on_image_document))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))

    async def on_error(update, context) -> None:
        log.exception("telegram handler error", exc_info=context.error)
        msg = getattr(update, "effective_message", None) if update is not None else None
        if msg is not None:
            await msg.reply_text(f"Turn failed: {context.error}"[:3500])

    app.add_error_handler(on_error)
    await app.initialize()
    await app.start()
    updater = app.updater
    if updater is None:
        raise RuntimeError("telegram updater missing after start")
    await updater.start_polling()
    log.info("telegram polling started")
