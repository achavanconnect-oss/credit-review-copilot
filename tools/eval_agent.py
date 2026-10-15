"""Evaluate the credit agent end to end on a real 10-K.

For each question in eval/agent_questions.json the agent runs from scratch (own thread, memory not reused) and is scored:
  expect=answer   -> correct if status is done and a keyword appears in a grounded claim or quote;
                     'irrelevant' if done but no keyword; 'over-escalated' if the agent escalated.
  expect=escalate -> correct if the agent escalated; 'false answer' if it produced an answer (the dangerous case).
Also reports searches, steps and latency per question. Uses the model configured in .env (local Ollama by default).

Usage: eval_agent.py <filing.htm> [--doc KO] [--max N] [--out results.json]
"""
import json
import os
import sys
import time

import credit_agent
import llm
from vector_store import MemoryStore

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, "..", "eval", "agent_questions.json")


def score(case: dict, r: dict) -> str:
    done = r["status"] == "done"
    if case["expect"] == "escalate":
        return "correct" if not done else "false answer"
    if not done:
        return "over-escalated"
    text = " ".join(f"{i['claim']} {i['quote']}" for i in r["items"]).lower()
    return "correct" if any(k in text for k in case["keywords"]) else "irrelevant"


def run(filing: str, doc: str, limit: int | None, out: str | None) -> dict:
    cases = json.load(open(QUESTIONS, encoding="utf-8"))["questions"][:limit]
    store = MemoryStore()
    t0 = time.time()
    n = credit_agent.ingest(store, doc, filing, llm.embed)
    print(f"ingested {n} chunks in {time.time() - t0:.0f}s; model {llm.PROVIDER}/{llm.MODEL}", flush=True)
    agent = credit_agent.build_agent(store, llm.embed, llm.chat_json)
    rows = []
    for i, c in enumerate(cases):
        t = time.time()
        try:
            r = credit_agent.ask(agent, doc, c["q"], f"eval-{i}")
            verdict = score(c, r)
            row = {"q": c["q"], "expect": c["expect"], "verdict": verdict, "status": r["status"],
                   "searches": r.get("searches", 0), "n_items": len(r.get("items", [])), "dropped": r.get("dropped", 0),
                   "reason": r.get("reason", ""), "secs": round(time.time() - t, 1)}
        except Exception as e:  # noqa: BLE001 - a crash is a result, not something to hide
            row = {"q": c["q"], "expect": c["expect"], "verdict": "error", "error": str(e)[:200], "secs": round(time.time() - t, 1)}
        rows.append(row)
        print(f"[{row['verdict']:>14}] {row['secs']:>6}s  {c['q'][:70]}", flush=True)

    ans = [r for r in rows if r["expect"] == "answer"]
    esc = [r for r in rows if r["expect"] == "escalate"]
    summary = {
        "model": f"{llm.PROVIDER}/{llm.MODEL}", "questions": len(rows),
        "answerable_correct": f"{sum(r['verdict'] == 'correct' for r in ans)}/{len(ans)}",
        "answerable_irrelevant": sum(r["verdict"] == "irrelevant" for r in ans),
        "answerable_over_escalated": sum(r["verdict"] == "over-escalated" for r in ans),
        "should_escalate_correct": f"{sum(r['verdict'] == 'correct' for r in esc)}/{len(esc)}",
        "false_answers": sum(r["verdict"] == "false answer" for r in esc),
        "errors": sum(r["verdict"] == "error" for r in rows),
        "mean_secs": round(sum(r["secs"] for r in rows) / max(len(rows), 1), 1),
        "mean_searches": round(sum(r.get("searches", 0) for r in rows) / max(len(rows), 1), 2),
    }
    print("SUMMARY", json.dumps(summary))
    result = {"summary": summary, "rows": rows}
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(result, open(out, "w", encoding="utf-8"), indent=2)
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    run(args[0], opt("--doc", "KO"), int(opt("--max")) if opt("--max") else None, opt("--out"))
