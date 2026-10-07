"""All runtime configuration, loaded from environment variables / .env."""

from datetime import time
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Channels ----
    primary_channel: str = "telegram"  # telegram | whatsapp

    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_business_account_id: str = ""
    whatsapp_app_secret: str = ""
    whatsapp_webhook_verify_token: str = ""
    whatsapp_api_version: str = "v23.0"
    whatsapp_briefing_template: str = "daily_briefing_ready"
    whatsapp_alert_template: str = "assistant_alert"
    whatsapp_template_language: str = "en"
    whatsapp_audio_as_voice: bool = True
    my_whatsapp_number: str = ""  # E.164 without '+', e.g. 919876543210

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""  # your numeric chat id (allow-list)
    telegram_mode: str = "polling"  # polling | webhook
    telegram_webhook_secret: str = ""

    # ---- LLM (Gemini) ----
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    llm_thinking_chat: str = "low"  # minimal | low | medium | high
    llm_thinking_script: str = "medium"
    llm_thinking_rank: str = "low"
    web_search_max_uses: int = 5

    # ---- Speech (Sarvam) ----
    sarvam_api_key: str = ""
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_language: str = "en-IN"
    sarvam_tts_pace: float = 1.0
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_language: str = "unknown"

    # ---- Ranking ----
    ranker: str = "llm"  # llm | jev
    typesafe_api_key: str = ""
    jev_model: str = "jev-latest"
    jev_low_confidence: float = 0.5
    jev_concurrency: int = 8
    dedupe_threshold: float = 0.88
    max_candidates_per_section: int = 40

    # ---- Data ----
    database_url: str = "postgresql+asyncpg://helper:helper@localhost:5544/helper"
    data_dir: Path = Path("data")
    sources_file: Path = Path("config/sources.yaml")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ---- Ops ----
    timezone: str = "Asia/Kolkata"
    briefing_time: str = "07:30"
    briefing_build_lead_minutes: int = 30
    briefing_minutes: int = 8
    words_per_minute: int = 170  # measured Sarvam bulbul:v3 pace at pace=1.0 (~172 wpm)
    daily_budget_usd: float = 1.0
    user_interests: str = (
        "AI/ML engineer in Bangalore. Interested in LLMs, agents, RAG, evals, open-source "
        "models, Indian AI ecosystem (e.g. Sarvam), AI infra, and major world news."
    )
    public_base_url: str = ""  # e.g. https://mybot.up.railway.app (for Telegram webhook)
    log_level: str = "INFO"

    # ---- External uptime alerting (optional; app/heartbeat.py) ----
    heartbeat_url: str = ""           # e.g. a healthchecks.io ping URL; pinged every 5 min while alive
    briefing_heartbeat_url: str = ""  # pinged once after each successful briefing delivery

    @field_validator("*", mode="before")
    @classmethod
    def _drop_inline_comment(cls, v):
        # `KEY=   # comment` in .env parses as the comment itself; treat it as empty
        if isinstance(v, str) and v.lstrip().startswith("#"):
            return ""
        return v

    @field_validator("primary_channel")
    @classmethod
    def _channel(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"telegram", "whatsapp"}:
            raise ValueError("PRIMARY_CHANNEL must be telegram or whatsapp")
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def briefing_clock(self) -> time:
        hh, mm = self.briefing_time.split(":")
        return time(int(hh), int(mm))

    @property
    def audio_dir(self) -> Path:
        p = self.data_dir / "audio"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
