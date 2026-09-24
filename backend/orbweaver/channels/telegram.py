"""Telegram adapter: a dispatcher operator, not a cursor inside another chat.

Each allowlisted Telegram user owns one *operator* session. Plain messages
run ``agent_turn`` on that session, except a reply to a tagged report and the
next message after an AskUser ping, which answer that session. The operator
prompts other sessions with ``PromptSession`` (or the user replies to a tagged
report); those turns keep the target's home channel, tools, and model, and
user events carry ``via: telegram``. A global router sink pushes approval
keyboards and AskUser pings from any chat, plus completion summaries for
sessions Telegram recently prompted (and errors/aborts from anywhere). Turns
run as background tasks;
``/stop`` cancels the operator turn, the last prompted session, or a tagged
reply target.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from orbweaver import __version__
from orbweaver.agent import (
    TurnCancelled,
    agent_turn,
    find_pending_approval,
    pending_ask_user,
    resolve_approval,
)
from orbweaver.auth import mint_token
from orbweaver.channels import router
from orbweaver.channels.router import OPERATOR_ROLE, OPERATOR_SYSTEM_EXTRA
from orbweaver.config import settings
from orbweaver.extract import human_size
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
    is_deleted_session,
    new_uuid,
    session_at_id,
)
from orbweaver.turns import GatewayDraining, RunningTurn, acquire_when_idle
from orbweaver.turns import acquire as acquire_turn
from orbweaver.turns import get as get_running_turn
from orbweaver.turns import release as release_turn
from orbweaver.uploads import (
    TELEGRAM_MAX_BYTES,
    StoredUpload,
    UploadError,
    describe_for_turn,
    store_upload,
)
from orbweaver.workspace import WORKSPACE_KIND_LOCAL, apply_local_workspace_kind, bind_workspace

log = logging.getLogger(__name__)

TELEGRAM_IMAGE_HINT = (
    "The user may send photos; those arrive as images you can see. "
    "Other attachments (PDF, text, code) are stored under attachments/ with their "
    "text extracted into the message; Read the stored path for anything truncated. "
    "To send a photo on Telegram, write an image file in the workspace and call SendPhoto. "
    "GenerateImage creates a file and, on Telegram sessions, sends it to the chat."
)
CHANNEL = "telegram"
# Operator JSON-LD: session whose forwarded AskUser the next plain message answers.
OPEN_ASK_KEY = "telegram_open_ask"
# [title · a1b2c3d4] — durable so a reply-to still resolves after a restart.
_SESSION_TAG_RE = re.compile(r"\[(?P<title>.+?) · (?P<short>[0-9a-fA-F]{8})\]")

# chat_id -> operator session id (in-memory; rebuilt from store on rehydrate).
_operator_chats: dict[int, UUID] = {}
# prompted target session -> operator chat_id, for SendPhoto and fast watch checks.
_watched_chats: dict[UUID, int] = {}
# Background turn and delivery tasks, kept alive until done.
_tasks: set[asyncio.Task[Any]] = set()
_sink_installed = False


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


def session_tag(title: str, session_id: UUID) -> str:
    clean = str(title or "session").replace("]", "").replace("\n", " ").strip()[:80] or "session"
    return f"[{clean} · {str(session_id)[:8]}]"


def parse_session_tag(text: str) -> str | None:
    """8-char session id prefix from a tagged bot message, or None."""
    match = _SESSION_TAG_RE.search(str(text or ""))
    return match.group("short").lower() if match else None


def _jsonld_chat_id(jsonld: dict[str, Any]) -> int | None:
    raw = jsonld.get("telegram_chat_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _remember_operator(operator: Entity) -> None:
    chat_id = _jsonld_chat_id(operator.jsonld or {})
    if chat_id is None:
        return
    _operator_chats[chat_id] = operator.id
    for sid in router.prompted_ids(operator):
        _watched_chats[sid] = chat_id


def _forget_watch(session_id: UUID) -> None:
    _watched_chats.pop(session_id, None)


async def notify_telegram_chat(chat_id: int, text: str, *, force_reply: bool = False) -> None:
    token = settings.telegram_bot_token
    if not token or not text:
        return
    import httpx

    body: dict[str, Any] = {"chat_id": chat_id, "text": text[:3500]}
    if force_reply:
        # The client opens a reply to this message, so the answer carries the session tag.
        body["reply_markup"] = {"force_reply": True}
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=body,
        )


APPROVAL_CALLBACK_PREFIX = "apr"
_APPROVAL_BUTTONS: tuple[tuple[str, str, str], ...] = (
    ("Allow", "allow", "once"),
    ("Allow for session", "allow", "session"),
    ("Deny", "deny", "once"),
)


def approval_request_text(payload: dict, *, tag: str = "") -> str:
    name = str(payload.get("name") or "tool")
    summary = str(payload.get("summary") or "")
    reason = str(payload.get("reason") or "")
    lines = []
    if tag:
        lines.append(tag)
    lines.append(f"Approval needed: {name}")
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


def approval_result_label(decision: str, scope: str) -> str:
    return {
        ("allow", "once"): "Allowed",
        ("allow", "session"): "Allowed for this session",
        ("deny", "once"): "Denied",
    }.get((decision, scope), decision)


async def close_approval_message(query: Any, label: str) -> None:
    """Retire an answered approval: append the outcome and take the buttons away.

    ``edit_message_text`` without ``reply_markup`` drops the inline keyboard, so
    the Allow / Deny buttons cannot be pressed a second time. If the edit fails
    (message too old, identical text), still remove the keyboard on its own.
    """
    try:
        original = str(getattr(query.message, "text", None) or "")
        await query.edit_message_text(f"{original}\n\n→ {label}"[:4000])
        return
    except Exception:
        log.debug("telegram approval message edit failed", exc_info=True)
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        log.debug("telegram approval keyboard removal failed", exc_info=True)


async def notify_telegram_approval(chat_id: int, payload: dict, *, tag: str = "") -> None:
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
                "text": approval_request_text(payload, tag=tag),
                "reply_markup": approval_keyboard(tool_use_id),
            },
        )


# ------------------------------------------------------------------ global sink


def _spawn(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _tasks.add(task)

    def _done(t: asyncio.Task[Any]) -> None:
        _tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("telegram task %s failed", name, exc_info=t.exception())

    task.add_done_callback(_done)
    return task


def _turn_status_text(status: str) -> str:
    return {
        "stopped": "Turn stopped.",
        "discarded": "Turn discarded.",
        "error": "Turn failed.",
        "waiting_ask": "Waiting for your answer.",
        "aborted": "Turn aborted.",
    }.get(status, "Turn finished.")


def _home_channel(sess: Entity) -> str:
    from orbweaver.agent import normalize_channel

    return normalize_channel(str((sess.jsonld or {}).get("channel") or "")) or "web"


async def _telegram_operators(store: Store) -> list[tuple[Entity, int]]:
    out: list[tuple[Entity, int]] = []
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") is None or is_deleted_session(ent):
            continue
        chat_id = _jsonld_chat_id(ent.jsonld or {})
        if chat_id is None:
            continue
        out.append((ent, chat_id))
    return out


async def _tagged_title(store: Store, session_id: UUID, sess: Entity | None) -> str:
    events = await store.list_events(session_id)
    title = router.display_title(sess.jsonld or {}, events) if sess else str(session_id)[:8]
    return session_tag(title, session_id)


async def deliver_turn_result(chat_id: int, session_id: UUID, done: dict[str, Any]) -> None:
    """A turn on another session finished: report it with a reply-to tag."""
    store = get_store()
    events = await store.list_events(session_id)
    try:
        user_seq = int(done.get("user_seq") or 0)
    except (TypeError, ValueError):
        user_seq = 0
    tail = [e for e in events if e.seq > user_seq]
    text = texts_for_reply(tail) or _turn_status_text(str(done.get("status") or ""))
    sess = await store.get_entity(session_id)
    tag = await _tagged_title(store, session_id, sess)
    await notify_telegram_chat(chat_id, f"{tag}\n{text}")


def _should_report_completion(
    status: str, *, watched: bool, aborted: bool
) -> tuple[bool, bool]:
    """(report, unwatch). waiting_ask is skipped; errors/aborts always report."""
    if status == "waiting_ask":
        return False, False
    always = status in {"error", "aborted"} or aborted
    if watched or always:
        terminal = status in {"ok", "stopped", "discarded", "error", "aborted"} or aborted
        return True, watched and terminal
    return False, False


async def _handle_global_event(session_id: UUID, msg: dict[str, Any]) -> None:
    store = get_store()
    sess = await store.get_entity(session_id)
    if sess is None or router.is_operator_session(sess) or router.is_subagent_session(sess):
        return
    operators = await _telegram_operators(store)
    if not operators:
        return
    kind = msg.get("kind")
    tag = await _tagged_title(store, session_id, sess)
    if kind == "permission_request":
        payload = dict(msg.get("payload") or {})
        for _op, chat_id in operators:
            await notify_telegram_approval(chat_id, payload, tag=tag)
        return
    if kind == "ask_user":
        question = str((msg.get("payload") or {}).get("question") or "").strip()
        body = f"{tag}\nWaiting for your answer:\n{question}" if question else f"{tag}\nWaiting for your answer."
        for op, chat_id in operators:
            await remember_open_ask(store, op, session_id)
            await notify_telegram_chat(chat_id, body, force_reply=True)
        return
    if kind != "turn_done":
        return
    status = str(msg.get("status") or "")
    events = await store.list_events(session_id)
    try:
        user_seq = int(msg.get("user_seq") or 0)
    except (TypeError, ValueError):
        user_seq = 0
    tail = [e for e in events if e.seq > user_seq]
    aborted = any(e.kind == "turn_aborted" for e in tail)
    for op, chat_id in operators:
        watched = session_id in router.prompted_ids(op)
        report, unwatch = _should_report_completion(status, watched=watched, aborted=aborted)
        if report:
            await deliver_turn_result(chat_id, session_id, dict(msg))
        if unwatch:
            await router.unwatch_prompted(store, op, session_id)
            _forget_watch(session_id)


def _telegram_global_sink(session_id: UUID, msg: dict[str, Any]) -> None:
    kind = msg.get("kind")
    if kind not in {"permission_request", "ask_user", "turn_done"}:
        return
    _spawn(_handle_global_event(session_id, dict(msg)), f"tg-sink-{kind}")


def install_telegram_sink() -> None:
    global _sink_installed
    if _sink_installed:
        return
    router.add_global_sink(_telegram_global_sink)
    _sink_installed = True


def install_router_hooks() -> None:
    """Back-compat alias used by older tests; installs the global dispatcher sink."""
    install_telegram_sink()


async def rehydrate_chat_sinks(store: Store) -> int:
    """After a restart, restore operator chat ids and the prompted watch set."""
    install_telegram_sink()
    count = 0
    for ent, _chat_id in await _telegram_operators(store):
        _remember_operator(ent)
        count += 1
    return count


def reset_for_tests() -> None:
    global _sink_installed
    _operator_chats.clear()
    _watched_chats.clear()
    _tasks.clear()
    router.remove_global_sink(_telegram_global_sink)
    _sink_installed = False


# -------------------------------------------------------------- photos / images


def _telegram_chat_id(sess: Entity | None) -> int | None:
    """Chat bound to ``sess``: its own Telegram binding, else a watching operator."""
    if not sess:
        return None
    own = _jsonld_chat_id(sess.jsonld or {})
    if own is not None:
        return own
    return _watched_chats.get(sess.id)


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


# ---------------------------------------------------------- inbound documents


def document_too_large_text(name: str, size: int) -> str:
    """Telegram refuses to serve bot downloads above ~20 MB; say so plainly."""
    return (
        f"{name} is {human_size(size)}, over Telegram's {human_size(TELEGRAM_MAX_BYTES)} "
        "bot download limit. Put the file in the workspace another way "
        "(scp, git, or the web UI) and tell me the path."
    )


async def receive_document(workspace: Any, data: bytes, filename: str) -> StoredUpload:
    """Store a Telegram document off the event loop (it writes to disk)."""
    return await asyncio.to_thread(store_upload, workspace, data, filename)


async def _reply_target_workspace(update, context, ref: str) -> tuple[Any, None] | tuple[None, str]:
    """Bind the reply-target session workspace, or return an error for the user."""
    store, operator, _chat_id = await _operator_session(update, context)
    try:
        target = await router.find_session(store, ref, exclude={operator.id})
    except router.AmbiguousSessionRef as e:
        return None, str(e)
    if target is None:
        return None, f"No session matches '{ref}'."
    ws, _kind, changed = bind_workspace(
        target.jsonld, settings.workspace_root, session_key=str(target.id)
    )
    if changed:
        await store.put_entity(target)
    return ws, None


async def handle_photo(update, context) -> None:
    """A compressed photo: Telegram re-encoded it, so only the vision path applies.

    Module level rather than nested in ``start_telegram`` so tests can drive it
    with a fake update. A reply to a tagged report stores the image on the
    target session and prompts it; otherwise the operator takes the turn.
    """
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
    caption = (update.message.caption or "").strip() or "[Photo]"
    stem = f"photo_{update.message.message_id}"
    ref = await inbound_target_ref(update, context)
    if ref:
        ws, err = await _reply_target_workspace(update, context, ref)
        if err:
            await update.message.reply_text(err)
            return
        try:
            img = save_inbound_image(ws, data, stem)
        except Exception as e:
            await update.message.reply_text(f"photo processing failed: {e}")
            return
        start_prompted(update, context, ref, caption, images=[img])
        return
    bound = await _session_workspace(update, context)
    try:
        img = save_inbound_image(bound.ws, data, stem)
    except Exception as e:
        await update.message.reply_text(f"photo processing failed: {e}")
        return
    start_turn(update, context, caption, images=[img])


async def handle_document(update, context) -> None:
    """Any uncompressed attachment.

    Images keep the vision path they always had; everything else (PDF, text,
    code, docx) is stored under ``attachments/`` and its extracted text goes
    into the turn message alongside the workspace path. A reply to a tagged
    report stores the file on the target session and prompts it.
    """
    if not update.effective_user or not update.message or not update.message.document:
        return
    if not user_allowed(update.effective_user.id):
        return
    doc = update.message.document
    name = doc.file_name or f"document_{update.message.message_id}"
    size = int(doc.file_size or 0)
    # Check the declared size before get_file(): Telegram answers that call with
    # a bare "file is too big" error, which is a worse thing to show the user.
    if size > TELEGRAM_MAX_BYTES:
        await update.message.reply_text(document_too_large_text(name, size))
        return
    try:
        file = await doc.get_file()
        data = bytes(await file.download_as_bytearray())
    except Exception as e:
        await update.message.reply_text(f"download failed: {e}")
        return
    if len(data) > TELEGRAM_MAX_BYTES:
        await update.message.reply_text(document_too_large_text(name, len(data)))
        return
    ref = await inbound_target_ref(update, context)
    if ref:
        ws, err = await _reply_target_workspace(update, context, ref)
        if err:
            await update.message.reply_text(err)
            return
    else:
        bound = await _session_workspace(update, context)
        ws = bound.ws
    try:
        stored = await receive_document(ws, data, name)
    except UploadError as e:
        await update.message.reply_text(f"cannot accept {name}: {e}")
        return
    except Exception as e:
        log.exception("telegram document handling failed")
        await update.message.reply_text(f"could not store {name}: {e}")
        return
    caption = (update.message.caption or "").strip()
    note = describe_for_turn(stored)
    text = f"{caption}\n\n{note}" if caption else note
    images = [stored.image] if stored.image is not None else None
    if ref:
        start_prompted(update, context, ref, text, images=images)
        return
    start_turn(update, context, text, images=images)


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
    if router.clear_legacy_attach(jsonld):
        changed = True
    return changed


async def session_for_telegram_user(
    store: Store, user_id: int, chat_id: int | None = None
) -> Entity:
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") == user_id:
            # A deleted operator session must not come back: fall through and
            # create a fresh one rather than reusing the hidden row.
            if is_deleted_session(ent):
                continue
            if _normalize_operator(ent.jsonld, chat_id):
                await store.put_entity(ent)
            _remember_operator(ent)
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
    _remember_operator(ent)
    return ent


@dataclass
class Bound:
    """The operator session a Telegram message runs on."""

    store: Store
    operator: Entity
    ws: Any
    kind: str
    chat_id: int | None
    target: Entity | None = None

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = self.operator

    @property
    def session_id(self) -> UUID:
        return self.operator.id


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
    install_telegram_sink()
    _remember_operator(ent)
    return store, ent, chat_id


async def _session_workspace(update, context, *, operator_only: bool = False) -> Bound:
    del operator_only
    store, operator, chat_id = await _operator_session(update, context)
    ws, kind, kind_changed = bind_workspace(
        operator.jsonld, settings.workspace_root, session_key=str(operator.id)
    )
    if kind_changed:
        await store.put_entity(operator)
    return Bound(store=store, operator=operator, target=operator, ws=ws, kind=kind, chat_id=chat_id)


def _queued_reply(state: RunningTurn | None) -> str:
    via = f" ({state.channel})" if state and state.channel else ""
    return (
        f"A turn is already running on this session{via}; your message is queued "
        "and will start when it finishes. Send /stop to cancel the current turn."
    )


def _title_of(ent: Entity, events) -> str:
    return router.display_title(ent.jsonld or {}, events)


def _ack_line(title: str, session_id: UUID) -> str:
    return f"→ {session_tag(title, session_id)}"


# ------------------------------------------------------------------- prompt / turns


async def _mark_watched(store: Store, operator: Entity, target: Entity) -> None:
    await router.mark_prompted(store, operator, target.id)
    chat_id = _jsonld_chat_id(operator.jsonld or {})
    if chat_id is not None:
        _watched_chats[target.id] = chat_id


async def _run_prompted_turn(
    store: Store,
    target: Entity,
    text: str,
    images: list[dict[str, str]] | None = None,
) -> None:
    """Start ``agent_turn`` on ``target`` as that session's home channel, via Telegram."""
    session_id = target.id
    channel = _home_channel(target)
    state: RunningTurn | None = None
    status = "error"
    try:
        try:
            state = acquire_turn(session_id, channel=channel)
        except GatewayDraining:
            return
        if state is None:
            from orbweaver.app import inject_into_turn

            running = get_running_turn(session_id)
            if running is not None:
                await inject_into_turn(store, session_id, running, text, images, via=CHANNEL)
            return
        ws, kind, kind_changed = bind_workspace(
            target.jsonld, settings.workspace_root, session_key=str(session_id)
        )
        if kind_changed:
            await store.put_entity(target)
        await agent_turn(
            store,
            session_id,
            text,
            ws,
            workspace_kind=kind,
            emit=router.emitter(session_id),
            headless=True,
            interactive=True,
            images=images,
            channel=channel,
            cancel=state.cancel,
            turn_state=state,
            via=CHANNEL,
        )
        status = "waiting_ask" if pending_ask_user(await store.list_events(session_id)) else "ok"
    except TurnCancelled as e:
        status = "stopped"
        marker = await store.append_event(
            session_id,
            "turn_interrupted",
            {"reason": (state.interrupt_reason if state else "") or "stop"},
        )
        router.emit(
            session_id,
            {"kind": marker.kind, "payload": marker.payload, "id": str(marker.id), "seq": marker.seq},
        )
        del e
    except asyncio.CancelledError:
        status = "stopped"
        try:
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
            log.exception("telegram: could not record cancelled prompted turn")
        raise
    except Exception:
        log.exception("telegram prompted turn failed for %s", session_id)
        status = "error"
    finally:
        if state is not None:
            release_turn(session_id, state)
            router.emit(
                session_id,
                {
                    "kind": "turn_done",
                    "status": status,
                    "user_seq": state.user_seq,
                    "channel": channel,
                },
            )


async def prompt_session(
    store: Store,
    operator: Entity,
    ref: str,
    text: str,
    images: list[dict[str, str]] | None = None,
    *,
    ack_chat_id: int | None = None,
) -> str:
    """Resolve ``ref``, watch it, inject or start a turn. JSON for PromptSession."""
    ref = str(ref or "").strip()
    text = str(text or "").strip()
    if not ref:
        return "error: session is required (id, id prefix, or title fragment)"
    if not text and not images:
        return "error: text is required"
    if not text:
        text = "[Photo]" if images else ""
    try:
        target = await router.find_session(store, ref, exclude={operator.id})
    except router.AmbiguousSessionRef as e:
        return json.dumps(
            {
                "error": str(e),
                "matches": [
                    {"id": str(m.id), "title": str((m.jsonld or {}).get("title") or "")}
                    for m in e.matches
                ],
            }
        )
    if target is None:
        return f"error: no session matches '{ref}'"
    events = await store.list_events(target.id)
    title = _title_of(target, events)
    await _mark_watched(store, operator, target)
    running = get_running_turn(target.id)
    if running is not None:
        from orbweaver.app import inject_into_turn

        await inject_into_turn(store, target.id, running, text, images, via=CHANNEL)
        action = "injected"
    else:
        _spawn(_run_prompted_turn(store, target, text, images), f"tg-prompt-{target.id}")
        action = "started"
    if ack_chat_id is not None:
        await notify_telegram_chat(ack_chat_id, _ack_line(title, target.id))
    return json.dumps(
        {
            "action": action,
            "session": str(target.id),
            "title": title,
            "tag": session_tag(title, target.id),
            "note": "The target keeps its own history; this chat stays the operator.",
        }
    )


async def prompt_session_from_operator(store: Store, operator: Entity, inp: dict[str, Any]) -> str:
    fresh = await store.get_entity(operator.id) or operator
    return await prompt_session(
        store,
        fresh,
        str(inp.get("session") or ""),
        str(inp.get("text") or ""),
        ack_chat_id=_jsonld_chat_id(fresh.jsonld or {}),
    )


async def stop_session_from_operator(store: Store, operator: Entity, inp: dict[str, Any]) -> str:
    ref = str(inp.get("session") or "").strip()
    if not ref:
        return "error: session is required (id, id prefix, or title fragment)"
    try:
        target = await router.find_session(store, ref, exclude={operator.id})
    except router.AmbiguousSessionRef as e:
        return json.dumps(
            {
                "error": str(e),
                "matches": [
                    {"id": str(m.id), "title": str((m.jsonld or {}).get("title") or "")}
                    for m in e.matches
                ],
            }
        )
    if target is None:
        return f"error: no session matches '{ref}'"
    state = get_running_turn(target.id)
    if state is None:
        return json.dumps({"stopped": False, "session": str(target.id), "reason": "nothing running"})
    state.cancel.set()
    return json.dumps({"stopped": True, "session": str(target.id)})


async def remember_open_ask(store: Store, operator: Entity, session_id: UUID) -> None:
    """The next plain Telegram message answers ``session_id``'s AskUser."""
    fresh = await store.get_entity(operator.id) or operator
    if str((fresh.jsonld or {}).get(OPEN_ASK_KEY) or "") == str(session_id):
        return
    fresh.jsonld[OPEN_ASK_KEY] = str(session_id)
    await store.put_entity(fresh)


async def open_ask_target(store: Store, operator: Entity) -> Entity | None:
    """Session a forwarded AskUser is still waiting on, or None.

    A stale pointer (answered on the web, deleted session) is cleared.
    """
    raw = str((operator.jsonld or {}).get(OPEN_ASK_KEY) or "").strip()
    if not raw:
        return None
    try:
        sid = UUID(raw)
    except ValueError:
        sid = None
    target = await store.get_entity(sid) if sid is not None else None
    pending = None
    if target is not None:
        pending = pending_ask_user(await store.list_events(target.id))
    if target is None or pending is None:
        operator.jsonld.pop(OPEN_ASK_KEY, None)
        await store.put_entity(operator)
        return None
    return target


async def inbound_target_ref(update, context) -> str | None:
    """Session prefix this message should run on, or None for the operator.

    A reply to a tagged report wins. Otherwise a plain message answers the
    AskUser Telegram just forwarded, unless the operator itself is mid-turn.
    """
    ref = _reply_session_ref(update)
    if ref:
        return ref
    store, operator, _chat_id = await _operator_session(update, context)
    if get_running_turn(operator.id) is not None:
        return None
    target = await open_ask_target(store, operator)
    if target is None:
        return None
    log.info("telegram: plain message answers open ask %s", target.id)
    return str(target.id)[:8]


async def dispatch_inbound_text(
    update,
    context,
    text: str,
    images: list[dict[str, str]] | None = None,
) -> None:
    """Route text to a tagged reply, an open AskUser, or the operator."""
    ref = await inbound_target_ref(update, context)
    if ref:
        start_prompted(update, context, ref, text, images)
        return
    start_turn(update, context, text, images)


def _reply_session_ref(update) -> str | None:
    """Short id from a reply to a tagged bot message, or None."""
    message = getattr(update, "message", None)
    if message is None:
        return None
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return None
    from_user = getattr(reply, "from_user", None)
    if from_user is not None and getattr(from_user, "is_bot", None) is False:
        return None
    text = getattr(reply, "text", None) or getattr(reply, "caption", None) or ""
    return parse_session_tag(str(text))


async def _run_turn(
    update,
    context,
    text: str,
    images: list[dict[str, str]] | None = None,
    *,
    operator_only: bool = False,
) -> None:
    del operator_only
    state: RunningTurn | None = None
    session_id: UUID | None = None
    status = "error"
    # Always set before the post-finally reply so CancelledError (BaseException)
    # during a gateway restart cannot leave ``reply`` unbound and swallow the turn.
    reply = "Turn interrupted."
    try:
        bound = await _session_workspace(update, context)
        store, session_id = bound.store, bound.session_id
        try:
            state = acquire_turn(session_id, channel=CHANNEL)
        except GatewayDraining:
            reply = "Orbweaver is updating; send that again in a minute."
            status = "stopped"
            return
        if state is None:
            running = get_running_turn(session_id)
            if running is None:
                reply = "A turn is already running on this session; send that again in a moment."
                log.info("telegram: session %s busy but holder vanished; asking user to retry", session_id)
                return
            if update.message:
                try:
                    await update.message.reply_text(_queued_reply(running))
                except Exception:
                    log.exception("telegram queue ack failed")
            log.info(
                "telegram: session %s busy via %s; queuing user message for the next turn",
                session_id,
                running.channel or "unknown",
            )
            try:
                state = await acquire_when_idle(session_id, CHANNEL)
            except GatewayDraining:
                reply = "Orbweaver is updating; send that again in a minute."
                status = "stopped"
                return
        if state is not None:
            extra = TELEGRAM_IMAGE_HINT + " " + OPERATOR_SYSTEM_EXTRA
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
            )
            reply = texts_for_reply(events) or "(no assistant text)"
            status = "waiting_ask" if pending_ask_user(await store.list_events(session_id)) else "ok"
    except TurnCancelled as e:
        status = "stopped"
        if session_id is not None:
            marker = await store.append_event(
                session_id,
                "turn_interrupted",
                {"reason": (state.interrupt_reason if state else "") or "stop"},
            )
            router.emit(
                session_id,
                {"kind": marker.kind, "payload": marker.payload, "id": str(marker.id), "seq": marker.seq},
            )
        reply = texts_for_reply(e.produced) or "Turn stopped."
    except asyncio.CancelledError:
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
    """Run the operator turn off the handler so /stop, approvals, and new messages keep flowing."""
    return _spawn(_run_turn(update, context, text, images, **kw), "tg-turn")


async def _prompt_from_reply(
    update,
    context,
    text: str,
    ref: str,
    images: list[dict[str, str]] | None = None,
) -> None:
    store, operator, _chat_id = await _operator_session(update, context)
    result = await prompt_session(store, operator, ref, text, images)
    ack = "Prompted."
    try:
        payload = json.loads(result)
        tag = payload.get("tag")
        if tag:
            ack = f"→ {tag}"
        elif result.startswith("error"):
            ack = result
    except (TypeError, ValueError, json.JSONDecodeError):
        if result.startswith("error"):
            ack = result
    if update.message:
        try:
            await update.message.reply_text(ack)
        except Exception:
            log.exception("telegram reply-to ack failed")


def start_prompted(
    update, context, ref: str, text: str, images: list[dict[str, str]] | None = None
) -> asyncio.Task[Any]:
    return _spawn(_prompt_from_reply(update, context, text, ref, images), "tg-prompt-reply")


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
    lines.append(f"operator session {str(bound.operator.id)[:8]} (workspace:default)")
    lines.append(_session_model_line(bound.operator))
    watching = router.prompted_ids(bound.operator)
    if watching:
        names: list[str] = []
        for sid in watching:
            ent = await store.get_entity(sid)
            if ent is None:
                continue
            events = await store.list_events(sid)
            names.append(f"{_title_of(ent, events)} ({str(sid)[:8]})")
            pending = pending_ask_user(events)
            if pending is not None:
                question = str(((pending.payload or {}).get("input") or {}).get("question") or "")
                if not question:
                    for ev in reversed(events):
                        if ev.kind == "ask_user":
                            question = str((ev.payload or {}).get("question") or "")
                            break
                if question:
                    lines.append(f"waiting in {_title_of(ent, events)}: {question}")
            state = get_running_turn(sid)
            if state is not None:
                lines.append(f"a turn is running on {_title_of(ent, events)} (via {state.channel or 'unknown'})")
        if names:
            lines.append("watching: " + ", ".join(names))
    else:
        lines.append("not watching any other session")
    state = get_running_turn(bound.session_id)
    if state is not None:
        lines.append(f"a turn is running on the operator (via {state.channel or 'unknown'})")
    events = await store.list_events(bound.operator.id)
    pending = pending_ask_user(events)
    if pending is not None:
        question = str(((pending.payload or {}).get("input") or {}).get("question") or "")
        lines.append(f"waiting for your answer: {question}" if question else "waiting for your answer")
    waiting = await open_ask_target(store, bound.operator)
    if waiting is not None:
        lines.append(f"your next message answers {_title_of(waiting, await store.list_events(waiting.id))}")
    lines.append("/sessions to list chats; reply to a tagged report to inject")
    return "\n".join(lines)


async def sessions_text(bound: Bound) -> str:
    rows = await router.list_sessions(bound.store, exclude={bound.operator.id})
    body = router.format_session_list(rows, router.prompted_ids(bound.operator))
    return body + "\n\nReply to a tagged report to inject; or ask the operator to PromptSession."


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

    sess = bound.operator
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


async def _stop_target(store: Store, operator: Entity, session_id: UUID) -> str:
    state = get_running_turn(session_id)
    if state is None:
        return "Nothing is running on that session."
    state.cancel.set()
    return "Stopping the running turn."


async def stop_text(bound: Bound, *, reply_ref: str | None) -> str:
    store = bound.store
    if reply_ref:
        try:
            target = await router.find_session(store, reply_ref, exclude={bound.operator.id})
        except router.AmbiguousSessionRef as e:
            return f"Several sessions match; pick one by id: {e}"
        if target is None:
            return f"No session matches '{reply_ref}'."
        return await _stop_target(store, bound.operator, target.id)
    state = get_running_turn(bound.session_id)
    if state is not None:
        state.cancel.set()
        return "Stopping the running turn."
    last = router.last_prompted_id(bound.operator)
    if last is not None:
        return await _stop_target(store, bound.operator, last)
    return "Nothing is running on this session."


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

    install_telegram_sink()
    try:
        bound_count = await rehydrate_chat_sinks(get_store())
        log.info("telegram: rebuilt dispatcher for %d operator session(s)", bound_count)
    except Exception:
        log.exception("telegram: rehydrating dispatcher failed")

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
        bound = await _session_workspace(update, context)
        await update.message.reply_text(await sessions_text(bound))

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
        await update.message.reply_text(await stop_text(bound, reply_ref=_reply_session_ref(update)))

    async def on_text(update: Update, context) -> None:
        if not update.effective_user or not update.message or not update.message.text:
            return
        if not user_allowed(update.effective_user.id):
            return
        await dispatch_inbound_text(update, context, update.message.text)

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
        await dispatch_inbound_text(update, context, text)

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
        await _operator_session(update, context)
        target_sid = find_pending_approval(tool_use_id)
        resolved = target_sid is not None and resolve_approval(target_sid, tool_use_id, decision, scope)
        if resolved:
            label = approval_result_label(decision, scope)
        else:
            label = "No longer pending (cancelled or already answered)"
        await query.answer(label)
        await close_approval_message(query, label)

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("version", on_version))
    app.add_handler(CommandHandler("status", on_status))
    app.add_handler(CommandHandler("sessions", on_sessions))
    app.add_handler(CommandHandler(["model", "models"], on_model))
    app.add_handler(CommandHandler("stop", on_stop))
    app.add_handler(
        CallbackQueryHandler(on_approval, pattern=rf"^{APPROVAL_CALLBACK_PREFIX}:")
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # Every document, not just images: PDFs, text, code. filters.PHOTO above
    # still claims compressed photos, so this cannot shadow them.
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
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
