from __future__ import annotations

import logging
from uuid import UUID

from orbweaver import __version__
from orbweaver.agent import agent_turn
from orbweaver.auth import mint_token
from orbweaver.config import settings
from orbweaver.store import (
    SESSION_TYPE,
    Entity,
    Store,
    get_store,
    new_uuid,
    session_at_id,
)
from orbweaver.workspace import make_workspace

log = logging.getLogger(__name__)


def user_allowed(user_id: int) -> bool:
    ids = settings.telegram_user_ids
    return not ids or user_id in ids


def texts_for_reply(events) -> str:
    parts: list[str] = []
    for e in events:
        text = (e.payload or {}).get("text")
        if text and e.kind in {"assistant", "turn_aborted"}:
            parts.append(str(text))
    return "\n".join(parts)[:3500]


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


async def session_for_telegram_user(
    store: Store, user_id: int, chat_id: int | None = None
) -> Entity:
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") == user_id:
            if chat_id is not None and ent.jsonld.get("telegram_chat_id") != chat_id:
                ent.jsonld["telegram_chat_id"] = chat_id
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
            "workspace_kind": "docker",
            "title": f"telegram:{user_id}",
            "telegram_user_id": user_id,
            "telegram_chat_id": chat_id if chat_id is not None else user_id,
            "status": "active",
        },
    )
    await store.put_entity(ent)
    return ent


async def start_telegram() -> None:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

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
        store = get_store()
        sid = context.user_data.get("session_id")
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not sid:
            ent = await session_for_telegram_user(store, update.effective_user.id, chat_id)
            sid = str(ent.id)
            context.user_data["session_id"] = sid
        session_id = UUID(sid)
        sess = await store.get_entity(session_id)
        if sess and chat_id is not None:
            sess.jsonld["telegram_chat_id"] = chat_id
            await store.put_entity(sess)
        kind = str(sess.jsonld.get("workspace_kind") or "docker") if sess else "docker"
        uri = str(sess.jsonld.get("workspace_uri") or "workspace:default") if sess else "workspace:default"
        ws = make_workspace(kind, uri, settings.workspace_root)
        events = await agent_turn(
            store, session_id, update.message.text, ws, workspace_kind=kind, headless=True,
            system_extra=(
                "Format all responses as plain text. Do not use Markdown — Telegram does not "
                "render it as rich text. Use plain text with spacing, emoji, and simple "
                "punctuation to organize responses instead."
            ),
        )
        await update.message.reply_text(texts_for_reply(events) or "(no assistant text)")

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
        except Exception as e:  # noqa: BLE001
            await update.message.reply_text(f"STT failed: {e}")
            return
        update.message.text = text
        await on_text(update, context)

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("version", on_version))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    log.info("telegram polling started")
