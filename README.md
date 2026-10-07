# Personal AI Helper (Telegram now, WhatsApp later)

A single-user assistant that sends an audio briefing every morning (by default about 8 minutes and 10 stories:
AI and new models, India, world, business and markets, science and tech, and something light) and answers anything by text or voice note. It can also remember things, keep lists,
set reminders and watch topics. It's built from `PRD Personal AI Helper on WhatsApp (Daily Briefing + Assistant).pdf`.
Accounts and keys are listed in `SETUP_REQUIREMENTS.md`.

## How it works

```
 scheduled (from /settings)                                          webhook / polling (any time)
 ┌──────────────────────────────────────────────┐          ┌──────────────────────────────────────┐
 │ ingest RSS / HN / Wikipedia  → full text      │          │ WhatsApp or Telegram message          │
 │ embed (local bge-small) → near-dup filter     │          │  voice? → Sarvam STT                  │
 │ rank: LLM baseline  or  Jev (+LLM fallback)   │          │  button? play / 👍👎 / go deeper       │
 │ top N per section → Gemini writes script      │          │ Gemini agent: Google Search + tools   │
 │   (only from article text)                    │          │  story lookup, memory, lists,         │
 │ Sarvam TTS (chunked) → stitch → OGG/Opus      │          │  watches, reminders, preferences      │
 │ at your time: "briefing ready" + ▶️ Play       │          │ reply as text, or voice if you spoke  │
 └───────────────────────┬──────────────────────┘          └─────────────────┬────────────────────┘
                         └──────────── Postgres + pgvector ────────────────────┘
```

| Path | What |
|---|---|
| `app/briefing/` | ingest → dedupe → rank (`rank.py`, `jev.py`) → script → pipeline |
| `app/agent/` | Gemini function-calling loop (`agent.py`) and its tools (`tools.py`) |
| `app/channels/` | `telegram.py` (in use), `whatsapp.py` (ready for later), shared interface `base.py` |
| `app/speech/` | Sarvam TTS/STT, WAV stitching, OGG/Opus encoding |
| `app/trackers.py` | memory facts, lists, reminders, topic watches, feedback weights, weekly summary (now includes a 7-day cost total) |
| `app/prefs.py` | chat-editable briefing settings: weekday/weekend time, length, section mix, skipped topics, focus, pause |
| `app/scheduler.py` | 1-minute tick reads the settings (builds 30 min early, delivers on time, catches up within 4 h), reminders, watches 4× a day, weekly summary Sunday 19:00, heartbeat ping every 5 min |
| `app/heartbeat.py` | optional external uptime alerting — see "Uptime alerting" below |
| `app/costs.py` | usage log per stage and the daily budget cap |
| `app/llm.py` | Gemini wrapper: thinking level, JSON output, retries, cost logging, Google-Search-grounded search |
| `app/evals/` | ranking (Jev vs LLM), script faithfulness, agent question set |
| `config/sources.yaml` | news sources; edit freely |

## Run locally

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker, and ffmpeg (`brew install ffmpeg`).

```bash
cp .env.example .env              # fill in keys (see SETUP_REQUIREMENTS.md)
docker compose up -d db           # Postgres + pgvector on localhost:5544
uv sync                           # installs Python 3.12 deps
uv run pytest                     # offline tests
```

Try pieces one at a time:

```bash
uv run python -m app.cli ingest                 # no keys needed: fetch + dedupe today's news
uv run python -m app.cli say "Hello Manvitha"   # Sarvam TTS test  -> data/say.ogg
uv run python -m app.cli chat                   # talk to the assistant in the terminal (Gemini key)
uv run python -m app.cli build --no-audio       # rank + write today's script (Gemini key)
uv run python -m app.cli build                  # full episode with audio (Gemini + Sarvam)
uv run python -m app.cli build --ranker jev     # same, ranked by Jev
uv run python -m app.cli cost                   # spend per provider/stage
```

Start the bot:

```bash
uv run uvicorn app.main:app --port 8000
```

**Telegram** (`TELEGRAM_MODE=polling`) runs on your laptop with no public URL. Send your bot any message and
it replies with your chat id. Put that in `TELEGRAM_CHAT_ID`, restart, and send `/build` to get a briefing
right away.

**WhatsApp (later):** the adapter in `app/channels/whatsapp.py` is complete and tested offline. To switch, set
`PRIMARY_CHANNEL=whatsapp` and the WhatsApp env vars, expose a public HTTPS URL (for example `ngrok http 8000`),
and point Meta's webhook at `/webhooks/whatsapp`. The details are in `SETUP_REQUIREMENTS.md`.

Chat commands: `/settings`, `/play` (latest briefing), `/build` (build a fresh one now), `/clear`, `/cost`, `/help`.

Scheduled tasks and reminders (like Instinct's proactive messages): "every weekday at 6pm send me the Nifty and
Sensex close" re-runs that request at 18:00 Mon-Fri (with web search) and sends the fresh answer; "every Monday at
9am remind me to submit my timesheet" sends fixed text. Schedules: once, daily, weekdays, weekly on chosen days,
every N hours (min 1). "What tasks do I have?", "pause/resume/change/delete …". Limits: 15 open tasks, 20 task runs
a day, and runs count toward the daily budget. Runs missed while the bot was off by more than 3 h are skipped.

Change the briefing by chatting: "send it at 8:15 on weekdays and 9:30 on weekends", "make it 5 minutes",
"more India news", "skip sports", "pause the briefing", "what are my briefing settings?".

## Deploy

`docker compose --profile app up -d --build` runs the app and the database on any VPS. On Railway, Render or Fly,
deploy the `Dockerfile`, attach a Postgres that has pgvector, and set the env vars. Polling works on a server too.
Webhook mode is optional: set `TELEGRAM_MODE=webhook`, `PUBLIC_BASE_URL` and `TELEGRAM_WEBHOOK_SECRET`.

Run **one** instance only. The scheduler runs inside the app process, so two copies would send two briefings.

### Uptime alerting

Once this isn't running on your laptop, something has to notice if it goes down — and the bot can't tell you
itself if the whole process is dead. Create a free account at [healthchecks.io](https://healthchecks.io), add a
check expecting a ping roughly every 10 minutes, and put its ping URL in `HEARTBEAT_URL`; the scheduler pings it
every 5 minutes while the app is alive, and healthchecks.io emails/texts you (your choice) the moment it stops
hearing from you. Optionally add a second check expecting one ping a day around your briefing time, and put its
URL in `BRIEFING_HEARTBEAT_URL` — that one catches a quieter failure: the app staying up but the briefing itself
silently failing to build after retries. Leave either blank to skip it.

## Evals (PRD section "Evals and quality metrics")

```bash
uv run python -m app.cli label               # daily for ~2 weeks: y/n per candidate
uv run python -m app.cli eval-ranking        # precision@N, missed stories, latency, cost: LLM vs Jev
uv run python -m app.cli eval-faithfulness   # Gemini-judged claim check of recent scripts (target: 0 unsupported)
uv run python -m app.cli eval-agent          # 30 fixed questions (deep-dive / general / tracker)
```

## Notes and decisions

- **LLM is Gemini** (`google-genai` SDK, `generate_content`). The default is `gemini-3.8-flash`, which is
  $0.75 / $3.75 per 1M tokens until 2026-12-31 and doubles after that. Thinking is `low` for chat and ranking
  and `medium` for scripts. To cut cost, set `LLM_MODEL=gemini-3.5-flash-lite`.
- **Web search** is a `web_search` tool that runs a separate Gemini call grounded with Google Search (5,000 free
  searches a month, then $14 per 1,000). Keeping it separate from the other tools avoids depending on
  Gemini support for mixing built-in and custom tools. `read_url` fetches pages directly.
- **No LangGraph.** The agent is a direct function-calling loop of about 130 lines in `app/agent/agent.py`.
  Adding LangGraph later only touches that one file.
- **Budget cap.** Once `DAILY_BUDGET_USD` is reached, LLM calls stop for the rest of the day and the bot says so.
- **Security.** Webhook signatures are verified, only your own number or chat id is answered, duplicate
  deliveries are dropped, and article text is treated as data. Jev's `injection` question and the LLM ranker's
  `injection_suspected` flag drop suspicious items.
- **Regional language.** Set `SARVAM_TTS_LANGUAGE` (for example `te-IN`). The script is still written in
  English, so the regional track still needs a translation step (next milestone).
