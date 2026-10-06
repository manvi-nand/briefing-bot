# Personal AI Helper on WhatsApp — Feasibility & Setup Requirements

Companion to `PRD Personal AI Helper on WhatsApp (Daily Briefing + Assistant).pdf`.
Written 2026-09-30. Vendor facts below should be re-checked when you start building.

---

## 1. Verdict: Is this possible?

**Yes — every piece of v1 is buildable today with public APIs.** Nothing in the PRD needs
research-grade work; it's integration work. A realistic part-time build is 4–6 weeks to a
working v1 (briefing + Q&A + trackers), which matches the PRD's 4-phase plan.

| Capability | Feasible? | Notes |
|---|---|---|
| Daily audio briefing pushed to WhatsApp | ✅ | Needs an approved **template message** to open the 24h window each morning |
| Text Q&A with web search + citations | ✅ | Standard LLM + search tool pattern |
| Voice notes in → STT → answer → voice note out | ✅ | Sarvam STT/TTS + `ffmpeg` for format conversion |
| "Go deeper on story 2" | ✅ | Just a DB lookup on `episodes.story_order` |
| Memory, trackers, reminders | ✅ | Postgres tables + a scheduler |
| Jev ranking | ✅ | Public API (`api.typesafe.ai/v1/systemone`), also on OpenRouter |
| Regional-language audio | ✅ | Sarvam Bulbul v3 supports Indian languages |
| Live voice calls | ⚠️ out of scope (correctly) | |

### The real risks (not technical)

1. **WhatsApp policy — the biggest one.** Meta's updated WhatsApp Business Solution terms
   (new accounts from 15 Oct 2025, all accounts from 15 Jan 2026) bar using the Business API
   to distribute a **general-purpose AI assistant as the primary function**. It was aimed at
   ChatGPT/Perplexity-style public bots, and a single-user personal tool is low-profile, but
   it's still a grey area: your account could get flagged.
   **Mitigation (strongly recommended):** keep the agent channel-agnostic from day 1
   (a `Channel` interface with `send_text / send_audio / send_buttons`) and build a
   **Telegram adapter** too. Telegram has no 24h window, no templates, no per-message fees,
   no business verification, and supports voice notes natively. Honestly, Telegram is the
   easier dev channel. Get the pipeline working there first, then add WhatsApp.
2. **You need a separate phone number** for WhatsApp Cloud API. A number registered on
   the Cloud API can't be used in the normal WhatsApp app at the same time. Meta's
   **free test number** works for development and can message up to 5 verified recipients,
   so that covers your single-user case for a while.
3. **Audio format.** For audio to show up as a *voice note* (waveform, play inline),
   WhatsApp expects **OGG with the Opus codec**. MP3/WAV get delivered as an audio file
   attachment instead. Plan: Sarvam TTS → WAV chunks → stitch + convert with `ffmpeg`
   (`-c:a libopus`). Check early (the PRD already flags this).
4. **Voice-note length.** A 10-minute briefing is ~8–10 MB as Opus, well under WhatsApp's
   16 MB audio limit, but test it. If you need to, split into 2–3 voice notes (AI / World / Fun).
5. **Jev is new and vendor-benchmarked.** The PRD's plan (LLM ranker baseline in phase 1,
   Jev in phase 3, fallback on low confidence) is the right way to handle this.

---

## 2. Tech stack (confirmed from PRD, with gaps filled)

| Layer | Choice | Why / notes |
|---|---|---|
| Language | Python 3.12 | |
| Web server | FastAPI + Uvicorn | Webhook endpoint (`GET` verify + `POST` messages) |
| Agent | LangGraph | Tools: web_search, get_story, related_stories, trackers, reminders |
| LLM | Gemini 3.8 Flash (default) / 3.5 Flash-Lite (cheap) / 3.1 Pro (quality) | Swappable via `LLM_MODEL` |
| Ranking | Embeddings (dedupe) → Jev (score) | LLM ranker as baseline/fallback |
| Embeddings | OpenAI `text-embedding-3-small`, Voyage, or local `bge-small` | Local = free, fine for dedupe |
| STT / TTS | Sarvam (Saarika/Saaras STT, Bulbul v3 TTS) | TTS max 2,500 chars/request → chunk at sentences |
| Audio processing | `ffmpeg` (+ `pydub`) | Stitch chunks, add pauses, convert to OGG/Opus |
| Messaging | WhatsApp Cloud API (Meta direct) **+ Telegram Bot API fallback** | Meta direct is cheaper & simpler than Twilio for 1 user |
| Database | Postgres 16 + `pgvector` | Supabase / Neon free tier works |
| Object storage | S3 / Cloudflare R2 / Supabase Storage | Audio files + transcripts |
| Scheduler | APScheduler (in-process) or cron / GitHub Actions / cloud scheduler | Daily run + reminder checks every minute |
| Ingestion | `feedparser` (RSS), `trafilatura` (article text), HN Algolia API, arXiv API | All free, no keys |
| Web search tool | Tavily / Brave Search / Exa | Pick one |
| Tracing | Langfuse (optional) | Cloud free tier or self-host |
| Hosting | Railway / Render / Fly.io / a small VPS | Needs a **public HTTPS URL** for the webhook |
| Local dev tunnel | ngrok or Cloudflare Tunnel | So Meta can reach your laptop |

### Free / no-key data sources for the briefing
- **Hacker News**: `https://hn.algolia.com/api/v1/` (no key)
- **arXiv**: `http://export.arxiv.org/api/query` (no key)
- **RSS**: most AI blogs and news sites (Reuters, BBC, The Verge, etc.)
- **"On this day"**: Wikipedia `https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/...` (no key)
- **Weather** (for "commute weather"): Open-Meteo (no key)
- *Optional:* NewsAPI / GNews (key needed) if RSS coverage isn't enough

---

## 3. What YOU need to set up (accounts, keys, secrets)

### 3.1 Required

| # | Service | What to get | Where | Cost |
|---|---|---|---|---|
| 1 | **Meta for Developers** | Meta developer account → create an App (type *Business*) → add the **WhatsApp** product | developers.facebook.com | Free |
| 2 | **Meta Business Portfolio** | Business account (auto-created, or at business.facebook.com). Business verification is *not* needed for the test number | business.facebook.com | Free |
| 3 | **WhatsApp test number** | Comes with the WhatsApp product. Add *your* personal number as a verified recipient | App dashboard → WhatsApp → API Setup | Free |
| 4 | **Permanent access token** | Create a **System User** (Admin) in Business Settings → assign the app + WABA → generate a token with `whatsapp_business_messaging` + `whatsapp_business_management`. (The dashboard token expires after 24h, so use it only for a first test) | Business Settings → Users → System users | Free |
| 5 | **Message template** | Create a *Utility* template, e.g. `daily_briefing_ready`: "Your {{1}} briefing is ready 🎧" with a Quick Reply button **Play**. Wait for approval (usually minutes to hours) | WhatsApp Manager → Message templates | Per-message charge (small, India pricing) |
| 6 | **Sarvam AI** | API subscription key | dashboard.sarvam.ai | ₹1,000 free credits; TTS ≈ ₹30 / 10k chars; STT ≈ ₹30 / hr |
| 7 | **Google Gemini** | API key. It also covers web search (Google Search grounding: 5,000 free searches a month, then $14 per 1,000), so no Tavily or Brave key is needed | aistudio.google.com → Get API key | Free tier, then pay-per-use |
| 8 | ~~Web search API~~ | Not needed, see row 7 | | |
| 9 | **Postgres + pgvector** | Connection string | Supabase / Neon / local Docker | Free tier OK |
| 10 | **Hosting** | Account on Railway/Render/Fly, or a VPS | | ~$0–7/month |

### 3.2 Needed for Jev (phase 3)

| Service | What to get | Where |
|---|---|---|
| **TypeSafe (Jev)** | API key (Bearer token) for `https://api.typesafe.ai/v1/systemone` | docs.typesafe.ai (also on OpenRouter as `typesafe/jev-1.13`) |

Answer the PRD's open question first: read TypeSafe's data retention terms before sending personal messages through it. Article text only is low risk.

### 3.3 Optional / nice to have

| Service | Purpose |
|---|---|
| **Telegram BotFather** | Fallback channel: message `@BotFather`, `/newbot`, and you get a bot token. **Highly recommended.** |
| Embeddings API key (OpenAI / Voyage) | Only if you don't use a local embedding model |
| Langfuse | Tracing + cost per stage |
| ngrok | Free account + authtoken for a stable dev tunnel |
| Cloudflare R2 / S3 | Audio storage (or use Supabase Storage) |
| Twilio | Only if you choose Twilio over Meta direct (not recommended for 1 user) |

---

## 4. Exactly what to give me (current plan: Telegram + Gemini)

Copy `.env.example` to `.env` and fill in these values:

| Variable | Where it comes from | Needed for |
|---|---|---|
| `GEMINI_API_KEY` | aistudio.google.com → Get API key (turn on billing for higher limits) | ranking, scripts, chat, web search |
| `SARVAM_API_KEY` | dashboard.sarvam.ai → API keys | audio briefing, voice notes |
| `TELEGRAM_BOT_TOKEN` | Telegram → @BotFather → `/newbot` | the chat channel |
| `TELEGRAM_CHAT_ID` | leave empty, message your bot once, and it replies with the id | allow-list (only you get answers) |
| `TYPESAFE_API_KEY` | console.typesafe.ai/keys (optional, phase 3) | Jev ranking (`RANKER=jev`) |

**Local tools:** `brew install ffmpeg` (audio encoding) plus Docker and uv (already on your machine).

### Later: WhatsApp

When you switch to WhatsApp, you'll also need `WHATSAPP_ACCESS_TOKEN` (System User token), `WHATSAPP_PHONE_NUMBER_ID`,
`WHATSAPP_BUSINESS_ACCOUNT_ID`, `WHATSAPP_APP_SECRET`, `WHATSAPP_WEBHOOK_VERIFY_TOKEN` (you make it up),
`MY_WHATSAPP_NUMBER`, a public HTTPS URL, and two *Utility* templates:
`daily_briefing_ready` (body `Good morning! Your {{1}} briefing is ready 🎧`, quick-reply button `Play`) and
`assistant_alert` (body `Update from your assistant: {{1}}`). Section 3.1 covers where to get each one.

### Security must-haves
- **Verify the webhook signature** (`X-Hub-Signature-256` HMAC with `WHATSAPP_APP_SECRET`) on every POST.
- **Allow-list your own number.** Drop any message not from `MY_WHATSAPP_NUMBER`, so strangers can't run up your bill.
- Return `200` to Meta fast and process in the background, or Meta will retry and you'll get duplicates. Dedupe on the message `id`.

---

## 5. Rough monthly cost (single user)

| Item | Estimate |
|---|---|
| Sarvam TTS: 7-min briefing ≈ 1,000 words ≈ 6,000 chars/day → ~180k chars/mo | ≈ ₹540 |
| Sarvam STT: ~30 min of voice notes/mo | ≈ ₹15 |
| Gemini 3.8 Flash: scripts + ranking + ~20 Q&A/day | ≈ $2–5 (free tier may cover testing) |
| Telegram | free |
| Google Search grounding (~5/day) | free (within 5,000/month) |
| DB + hosting | $0–7 |
| **Total** | **≈ ₹800–1,000/month** (~$10–12), mostly Sarvam TTS |

---

## 6. Suggested build order (maps to the PRD's phases)

1. **Skeleton**: FastAPI app, Postgres schema, `Channel` interface, Telegram adapter. Echo bot works.
2. **WhatsApp adapter**: webhook verify, signature check, send text/audio/buttons, template send. Test voice-note format (OGG/Opus).
3. **Briefing v0**: ingest RSS/HN/arXiv → embed + dedupe → **LLM ranker** → script → Sarvam TTS chunks → ffmpeg stitch → deliver. Store the episode.
4. **Agent**: Gemini function-calling loop with Google-Search-grounded web search, story lookup ("story 2"), voice-in/voice-out.
5. **Memory + trackers + reminders**: tables, scheduler tick, confirmation echoes.
6. **Feedback loop**: 👍/👎 buttons, topic weights, weekly summary.
7. **Jev**: integrate behind the `RANKER` flag, run the 2-week labeled comparison against the LLM ranker.
8. **Regional language track.**

---

## 7. Decisions you need to make (from the PRD's open questions)

- [ ] Meta direct vs Twilio → **recommend Meta direct**
- [x] Channel → Telegram first, WhatsApp later
- [x] LLM provider/model → Gemini (`gemini-3.8-flash`)
- [x] Web search provider → Google Search grounding via Gemini
- [ ] Hosting platform
- [ ] Default briefing time and length
- [ ] Regional language (Telugu? Kannada? Hindi?)
- [ ] Single host vs two-host style (two-host = 2 Sarvam speakers, roughly 1.2× the TTS chars)
- [ ] Are TypeSafe's data terms OK?
