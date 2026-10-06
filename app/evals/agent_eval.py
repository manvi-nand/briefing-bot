"""Agent answer eval (PRD eval 3): fixed question set, LLM-judged, rerun after prompt/model changes.

Questions live in app/evals/agent_questions.yaml. Trackers/memory written during the run are deleted afterwards.
"""

from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import delete

from app import llm
from app.agent.agent import respond
from app.briefing.pipeline import episode_stories, latest_episode
from app.db import session_scope
from app.models import MemoryFact, Message, Tracker

QUESTIONS = Path(__file__).with_name("agent_questions.yaml")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

JUDGE = """You grade an assistant's chat answer. Given the question, grading notes and the answer, decide if the
answer is correct and does what the notes require. Be strict but fair; brevity is fine."""


async def run() -> None:
    qs = yaml.safe_load(QUESTIONS.read_text())["questions"]
    ep = await latest_episode()
    stories = {n: item.title for n, item, _ in await episode_stories(ep)} if ep else {}
    started = datetime.now(timezone.utc)
    results = []
    try:
        for q in qs:
            notes = q.get("notes", "")
            if "story" in q:
                if q["story"] not in stories:
                    continue
                notes += f" The answer must be about the story titled: {stories[q['story']]!r}."
            answer = str(await respond(q["q"], channel="eval"))
            cited = "http" in answer
            verdict = await llm.complete_json(
                JUDGE, f"Question: {q['q']}\nGrading notes: {notes}\n\nAnswer:\n{answer}",
                JUDGE_SCHEMA, stage="eval_agent", thinking="low",
            )
            ok = verdict["correct"] and (cited or not q.get("must_cite"))
            results.append((q["kind"], ok))
            print(f"{'PASS' if ok else 'FAIL'} [{q['kind']}] {q['q']}"
                  + ("" if ok else f"\n     -> {verdict['reason']}{'' if cited or not q.get('must_cite') else ' (no citation)'}"))
    finally:
        async with session_scope() as s:  # clean up side effects
            await s.execute(delete(Tracker).where(Tracker.created_at >= started))
            await s.execute(delete(MemoryFact).where(MemoryFact.created_at >= started))
            await s.execute(delete(Message).where(Message.channel == "eval"))
    for kind in sorted({k for k, _ in results}):
        rs = [ok for k, ok in results if k == kind]
        print(f"{kind:10} {sum(rs)}/{len(rs)}")
    print(f"overall    {sum(ok for _, ok in results)}/{len(results)}")
