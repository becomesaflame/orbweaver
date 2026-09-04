from __future__ import annotations

import logging
from uuid import UUID

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


async def session_for_telegram_user(store: Store, user_id: int) -> Entity:
    for ent in await store.list_entities(SESSION_TYPE):
        if ent.jsonld.get("telegram_user_id") == user_id:
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
        ent = await session_for_telegram_user(store, update.effective_user.id)
        context.user_data["session_id"] = str(ent.id)
        mint_token(f"telegram:{update.effective_user.id}", extra={"channel": "telegram"})
        await update.message.reply_text(f"session {ent.id}")
        log.info("telegram session %s jwt issued", ent.id)

    async def on_text(update: Update, context) -> None:
        if not update.effective_user or not update.message or not update.message.text:
            return
        if not user_allowed(update.effective_user.id):
            return
        store = get_store()
        sid = context.user_data.get("session_id")
        if not sid:
            ent = await session_for_telegram_user(store, update.effective_user.id)
            sid = str(ent.id)
            context.user_data["session_id"] = sid
        session_id = UUID(sid)
        sess = await store.get_entity(session_id)
        kind = str(sess.jsonld.get("workspace_kind") or "docker") if sess else "docker"
        uri = str(sess.jsonld.get("workspace_uri") or "workspace:default") if sess else "workspace:default"
        ws = make_workspace(kind, uri, settings.workspace_root)
        events = await agent_turn(store, session_id, update.message.text, ws, workspace_kind=kind)
        texts = [e.payload.get("text") for e in events if e.kind == "assistant" and e.payload.get("text")]
        await update.message.reply_text("\n".join(texts)[:3500] or "(no assistant text)")

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
        update.message.text = text
        await on_text(update, context)

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    log.info("telegram polling started")
