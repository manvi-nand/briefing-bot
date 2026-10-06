from functools import lru_cache

from app.channels.base import Button, Channel, Incoming
from app.config import get_settings

__all__ = ["Button", "Channel", "Incoming", "get_channel", "primary_channel"]


@lru_cache
def get_channel(name: str) -> Channel:
    if name == "whatsapp":
        from app.channels.whatsapp import WhatsAppChannel

        return WhatsAppChannel()
    if name == "telegram":
        from app.channels.telegram import TelegramChannel

        return TelegramChannel()
    raise ValueError(f"unknown channel {name}")


def primary_channel() -> Channel:
    return get_channel(get_settings().primary_channel)
