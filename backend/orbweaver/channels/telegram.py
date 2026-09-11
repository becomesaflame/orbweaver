from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from orbweaver import __version__
from orbweaver.agent import agent_turn, resolve_approval
from orbweaver.auth import mint_token
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
from orbweaver.workspace import WORKSPACE_KIND_LOCAL, apply_local_workspace_kind, bind_workspace

log = logging.getLogger(__name__)

TELEGRAM_IMAGE_HINT = (
    "The user may send photos; those arrive as images you can see. "
    "To send a photo on Telegram, write an image file in the workspace and call SendPhoto. "
    "GenerateImage creates a file and, on Telegram sessions, sends it to the chat."
)


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


def _approval_emitter(chat_id: int | None):
    """agent_turn emit hook: push permission_request events to the chat as a keyboard."""
    if chat_id is None:
        return None

    def emit(msg: dict) -> None:
        if msg.get("kind") != "permission_request":
            return
        payload = dict(msg.get("payload") or {})

        async def _send() -> None:
            try:
                await notify_telegram_approval(chat_id, payload)
            except Exception:
                log.exception("telegram approval keyboard failed")

        asyncio.get_running_loop().create_task(_send())

    return emit


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


def _telegram_chat_id(sess: Entity | None) -> int | None:
    if not sess:
        return None
    raw = sess.jsonld.get("telegram_chat_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


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


async def session_for_telegram_user(
    store: Store, user_id: int, chat_id: int | None = None
) -> Entity:
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") == user_id:
            changed = apply_local_workspace_kind(ent.jsonld)
            if chat_id is not None and ent.jsonld.get("telegram_chat_id") != chat_id:
                ent.jsonld["telegram_chat_id"] = chat_id
                changed = True
            if ent.jsonld.get("channel") != "telegram":
                ent.jsonld["channel"] = "telegram"
                changed = True
            if changed:
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
            "channel": "telegram",
            "telegram_user_id": user_id,
            "telegram_chat_id": chat_id if chat_id is not None else user_id,
            "status": "active",
        },
    )
    await store.put_entity(ent)
    return ent


async def _session_workspace(update, context):
    store = get_store()
    sid = context.user_data.get("session_id")
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not sid:
        ent = await session_for_telegram_user(store, update.effective_user.id, chat_id)
        sid = str(ent.id)
        context.user_data["session_id"] = sid
    session_id = UUID(sid)
    sess = await store.get_entity(session_id)
    if sess:
        changed = False
        if chat_id is not None and sess.jsonld.get("telegram_chat_id") != chat_id:
            sess.jsonld["telegram_chat_id"] = chat_id
            changed = True
        if sess.jsonld.get("channel") != "telegram":
            sess.jsonld["channel"] = "telegram"
            changed = True
        ws, kind, kind_changed = bind_workspace(sess.jsonld, settings.workspace_root)
        if changed or kind_changed:
            await store.put_entity(sess)
    else:
        ws, kind, _ = bind_workspace({"workspace_uri": "workspace:default"}, settings.workspace_root)
    return store, session_id, ws, kind


async def _run_turn(update, context, text: str, images: list[dict[str, str]] | None = None) -> None:
    try:
        store, session_id, ws, kind = await _session_workspace(update, context)
        chat = getattr(update, "effective_chat", None)
        chat_id = chat.id if chat is not None else None
        events = await agent_turn(
            store,
            session_id,
            text,
            ws,
            workspace_kind=kind,
            emit=_approval_emitter(chat_id),
            headless=True,
            interactive=True,
            images=images,
            system_extra=TELEGRAM_IMAGE_HINT,
            channel="telegram",
        )
        reply = texts_for_reply(events) or "(no assistant text)"
    except Exception as e:
        log.exception("telegram turn failed")
        reply = f"Turn failed: {e}"[:3500]
    if update.message:
        await update.message.reply_text(reply)


async def start_telegram() -> None:
    from telegram import Update
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    app = Application.builder().token(settings.telegram_bot_token).build()

    async def on_start(update: Update, context) -> None:
        if not update.effective_user or not update.message:
            return
        if not user_allowed(update.effective_user.id):
            await update.message.reply_text("not allowlisted")
            return
        store = get_store()
        chat_id = update.effective_chat.id if update.effective_chat else None
        ent = await session_for_telegram_user(store, update.effective_user.id, chat_id)
        context.user_data["session_id"] = str(ent.id)
        mint_token(f"telegram:{update.effective_user.id}", extra={"channel": "telegram"})
        await update.message.reply_text(f"session {ent.id}\nOrbweaver {__version__}")
        log.info("telegram session %s jwt issued", ent.id)

    async def on_version(update: Update, context) -> None:
        del context
        if not update.effective_user or not update.message:
            return
        if not user_allowed(update.effective_user.id):
            await update.message.reply_text("not allowlisted")
            return
        await update.message.reply_text(f"Orbweaver {__version__}")

    async def on_text(update: Update, context) -> None:
        if not update.effective_user or not update.message or not update.message.text:
            return
        if not user_allowed(update.effective_user.id):
            return
        await _run_turn(update, context, update.message.text)

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
        _store, _session_id, ws, _kind = await _session_workspace(update, context)
        try:
            img = save_inbound_image(ws, data, f"photo_{update.message.message_id}")
        except Exception as e:
            await update.message.reply_text(f"photo processing failed: {e}")
            return
        caption = (update.message.caption or "").strip() or "[Photo]"
        await _run_turn(update, context, caption, images=[img])

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
        _store, _session_id, ws, _kind = await _session_workspace(update, context)
        stem = (doc.file_name or f"image_{update.message.message_id}").rsplit(".", 1)[0]
        try:
            img = save_inbound_image(ws, data, stem)
        except Exception as e:
            await update.message.reply_text(f"image processing failed: {e}")
            return
        caption = (update.message.caption or "").strip() or f"[Image: {img['path']}]"
        await _run_turn(update, context, caption, images=[img])

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
        await _run_turn(update, context, text)

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
        sid = context.user_data.get("session_id")
        if not sid:
            chat_id = update.effective_chat.id if update.effective_chat else None
            ent = await session_for_telegram_user(get_store(), update.effective_user.id, chat_id)
            sid = str(ent.id)
            context.user_data["session_id"] = sid
        resolved = resolve_approval(UUID(sid), tool_use_id, decision, scope)
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
