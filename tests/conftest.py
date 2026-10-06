"""Tests use their own database so they never touch real briefings, reminders or topic weights."""

import os

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://helper:helper@localhost:5544/helper_test"
)
os.environ["PRIMARY_CHANNEL"] = "telegram"
os.environ["TELEGRAM_CHAT_ID"] = "42"
os.environ["DAILY_BUDGET_USD"] = "0"  # no cap in tests
# Never call paid APIs from tests, whatever .env says: blank keys, plain ranker.
for key in ("GEMINI_API_KEY", "SARVAM_API_KEY", "TYPESAFE_API_KEY", "TELEGRAM_BOT_TOKEN", "WHATSAPP_ACCESS_TOKEN"):
    os.environ[key] = ""
os.environ["RANKER"] = "llm"
