"""Script faithfulness audit (PRD eval 2): LLM judge checks every claim in each story against its source text."""

from sqlalchemy import select

from app import llm
from app.briefing.pipeline import episode_stories
from app.db import session_scope
from app.models import Episode

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
                    "evidence": {"type": "string", "description": "short quote from the source, or empty"},
                },
                "required": ["claim", "verdict", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

SYSTEM = """You audit a spoken news script for faithfulness. Split the SCRIPT into atomic factual claims
(ignore greetings, transitions and opinions clearly framed as such). For each claim decide whether the SOURCE text
supports it. "unsupported" = not stated in the source; "contradicted" = the source says otherwise.
Be strict: numbers, dates, names and attributions must match."""


async def audit(episodes: int = 3) -> None:
    async with session_scope() as s:
        eps = list((await s.scalars(
            select(Episode).where(Episode.status.in_(["ready", "sent"])).order_by(Episode.created_at.desc()).limit(episodes)
        )).all())
    total = bad = 0
    for ep in eps:
        stories = {n: (item, src) for n, item, src in await episode_stories(ep)}
        print(f"\n=== Episode {ep.date} ({ep.run_id}) ===")
        for seg in ep.segments or []:
            if seg.get("kind") != "story" or seg.get("story_number") not in stories:
                continue
            item, src = stories[seg["story_number"]]
            data = await llm.complete_json(
                SYSTEM,
                f"<source name=\"{src}\">\n{(item.full_text or item.summary)[:3500]}\n</source>\n\n"
                f"<script>\n{seg['text']}\n</script>",
                SCHEMA, stage="eval_faithfulness", thinking="medium",
            )
            claims = data["claims"]
            flagged = [c for c in claims if c["verdict"] != "supported"]
            total += len(claims)
            bad += len(flagged)
            status = "OK" if not flagged else f"{len(flagged)} FLAGGED"
            print(f"story {seg['story_number']}: {len(claims)} claims, {status} — {item.title[:70]}")
            for c in flagged:
                print(f"   ✗ [{c['verdict']}] {c['claim']}")
    print(f"\nTotal: {bad}/{total} claims not supported (target 0). Spot-check flagged ones by hand.")
