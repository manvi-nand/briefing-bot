"""Command line for local testing:  python -m app.cli <command>  (uv run python -m app.cli ...)"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config import get_settings


async def _main(args: argparse.Namespace) -> None:
    from app.db import init_db

    await init_db()
    cmd = args.cmd

    if cmd == "init-db":
        print("database ready")

    elif cmd == "sync-sources":
        from app.briefing.ingest import sync_sources

        print(f"{len(await sync_sources())} sources synced")

    elif cmd == "ingest":
        from app.briefing import dedupe, ingest

        ids = await ingest.ingest()
        dups = await dedupe.embed_and_dedupe(ids)
        print(f"{len(ids)} new items ({dups} near-duplicates)")

    elif cmd == "build":
        from app.briefing import pipeline

        ep = await pipeline.build_episode(ranker=args.ranker, with_audio=not args.no_audio)
        print(f"episode {ep.id} built")
        for n, item, source in await pipeline.episode_stories(await pipeline.latest_episode()):
            print(f"  {n}. [{source}] {item.title}")
        latest = await pipeline.latest_episode()
        print("\n" + (latest.script or ""))
        if latest.audio_path:
            print(f"\naudio: {latest.audio_path} ({latest.audio_seconds or 0:.0f}s)")
        if args.send:
            await pipeline.deliver_notice(latest)
            print("notice sent")

    elif cmd == "play":
        from app.briefing import pipeline

        await pipeline.play_episode()

    elif cmd == "chat":
        from app.agent.agent import respond
        from app.handlers import _save

        print("Local chat with the assistant (Ctrl-D to quit). Uses the real DB, tools and web search.")
        while True:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
            if not text:
                continue
            mid = await _save("cli", "in", "text", text)
            answer = str(await respond(text, channel="cli", message_id=mid))
            await _save("cli", "out", "text", answer)
            print(f"\nbot> {answer}")

    elif cmd == "say":
        from app.speech import audio, sarvam

        wav, secs = await sarvam.synthesize([args.text], stage="tts_test")
        out = Path(args.out)
        if audio.ffmpeg_available():
            await audio.to_ogg_opus(wav, out)
        else:
            out = out.with_suffix(".wav")
            out.write_bytes(wav)
        print(f"wrote {out} ({secs:.1f}s)")

    elif cmd == "send-test":
        from app.channels import primary_channel

        ch = primary_channel()
        print(await ch.send_text(ch.owner, args.text))

    elif cmd == "send-template":
        from app.channels import primary_channel
        from app.briefing.pipeline import latest_episode

        ch = primary_channel()
        ep = await latest_episode()
        print(await ch.send_briefing_notice(ch.owner, "test", ep.id if ep else 0))

    elif cmd == "cost":
        from app import costs

        rows = await costs.cost_report(args.days)
        for p, st, c in rows:
            print(f"{p:10} {st:18} ${c:.4f}")
        print(f"total ${sum(c for *_, c in rows):.4f}; today ${await costs.spent_today():.4f}")

    elif cmd == "label":
        from app.evals.ranking import label_today

        await label_today()

    elif cmd == "eval-ranking":
        from app.evals.ranking import compare

        await compare(args.days)

    elif cmd == "compare-rankers":
        from app.evals.ranking import side_by_side

        await side_by_side()

    elif cmd == "eval-faithfulness":
        from app.evals.faithfulness import audit

        await audit(args.episodes)

    elif cmd == "eval-agent":
        from app.evals.agent_eval import run

        await run()


def main() -> None:
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("sync-sources")
    sub.add_parser("ingest")
    b = sub.add_parser("build", help="build a briefing now")
    b.add_argument("--no-audio", action="store_true")
    b.add_argument("--ranker", choices=["llm", "jev"], default=None)
    b.add_argument("--send", action="store_true", help="send the 'briefing ready' notice")
    sub.add_parser("play", help="send the latest briefing audio + story cards")
    sub.add_parser("chat", help="talk to the assistant in the terminal")
    sy = sub.add_parser("say", help="TTS test")
    sy.add_argument("text")
    sy.add_argument("--out", default="data/say.ogg")
    st = sub.add_parser("send-test", help="send a text via the primary channel")
    st.add_argument("text")
    sub.add_parser("send-template", help="send the briefing-ready template/notice")
    c = sub.add_parser("cost")
    c.add_argument("--days", type=int, default=7)
    sub.add_parser("label", help="label today's candidates for the ranking eval")
    er = sub.add_parser("eval-ranking")
    er.add_argument("--days", type=int, default=14)
    sub.add_parser("compare-rankers", help="rank today's pool with LLM and Jev, show picks side by side")
    ef = sub.add_parser("eval-faithfulness")
    ef.add_argument("--episodes", type=int, default=3)
    sub.add_parser("eval-agent")
    args = p.parse_args()

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
