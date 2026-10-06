"""FastAPI app: WhatsApp + Telegram webhooks, health check, scheduler lifecycle."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request, Response

from app.briefing.ingest import sync_sources
from app.channels import get_channel
from app.channels.telegram import TelegramChannel, parse_update
from app.channels.whatsapp import parse_webhook, verify_signature
from app.config import get_settings
from app.db import init_db
from app.handlers import handle
from app.scheduler import build_scheduler

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await sync_sources()
    sched = build_scheduler()
    sched.start()
    poller = None
    if settings.telegram_bot_token:
        tg: TelegramChannel = get_channel("telegram")  # type: ignore[assignment]
        if settings.telegram_mode == "polling":
            poller = asyncio.create_task(tg.poll_forever(handle))
        elif settings.public_base_url:
            await tg.set_webhook(f"{settings.public_base_url.rstrip('/')}/webhooks/telegram",
                                 settings.telegram_webhook_secret)
    log.info("started; primary channel = %s", settings.primary_channel)
    yield
    sched.shutdown(wait=False)
    if poller:
        poller.cancel()


app = FastAPI(title="Personal briefing assistant", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "primary_channel": settings.primary_channel}


# ---------------------------------------------------------------- WhatsApp

@app.get("/webhooks/whatsapp")
async def whatsapp_verify(
    mode: str = Query("", alias="hub.mode"),
    token: str = Query("", alias="hub.verify_token"),
    challenge: str = Query("", alias="hub.challenge"),
):
    if mode == "subscribe" and settings.whatsapp_webhook_verify_token and token == settings.whatsapp_webhook_verify_token:
        return Response(content=challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification failed")


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
):
    body = await request.body()
    if not verify_signature(settings.whatsapp_app_secret, body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="bad signature")
    messages, statuses = parse_webhook(json.loads(body or b"{}"))
    for st in statuses:
        if st.get("status") == "failed":
            log.warning("WhatsApp delivery failed: %s", st.get("errors"))
    for m in messages:
        background.add_task(handle, m)  # ack fast; Meta retries slow webhooks
    return {"ok": True}


# ---------------------------------------------------------------- Telegram (webhook mode)

@app.post("/webhooks/telegram")
async def telegram_webhook(
    request: Request,
    background: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if not settings.telegram_webhook_secret or x_telegram_bot_api_secret_token != settings.telegram_webhook_secret:
        raise HTTPException(status_code=401, detail="bad secret")
    inc = parse_update(await request.json())
    if inc:
        background.add_task(handle, inc)
    return {"ok": True}
