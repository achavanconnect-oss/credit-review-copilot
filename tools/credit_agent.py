"""Credit-review ReAct agent on LangGraph.

Loop: recall -> plan (thought + action) -> search (observation) -> plan ... -> answer -> verify -> finalize.
  - plan:     the LLM chooses search / answer / escalate (ReAct: thought, action, observation).
  - verify:   deterministic. Every claim needs a verbatim quote found in the retrieved text (guardrails.py).
  - escalate: decision/rating requests, no evidence after several searches, or repeated ungrounded answers go to a human.
  - memory:   short-term via a LangGraph checkpointer (thread_id keeps the conversation history);
              long-term via the vector store: verified findings are stored and reused for near-identical questions.

The LLM and the embedder are injected, so the whole graph is testable without a model or a database.

Usage:
  credit_agent.py ask <doc> <filing.htm> "<question>"      (set DATABASE_URL to use pgvector, else in-memory)
"""
import operator
import os
import re
import sys
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

import guardrails

MAX_ATTEMPTS = 3          # ungrounded answers tolerated before escalating
MAX_SEARCHES = 4          # retrieval steps per question
MEMORY_MAX_DISTANCE = 0.10
# Remembered findings expire (default 7 days) so a stale answer is not reused after the filing set changes.
# Also use a new `doc` id whenever the underlying filing changes.
MEMORY_TTL_S = float(os.getenv("MEMORY_TTL_S", 7 * 24 * 3600))

# Requests for a decision or a rating are outside this tool's remit: always a human's call.
DECISION_REQUEST = re.compile(
    r"\b(should (?:we|i) (?:lend|approve|buy|sell|invest)|(?:approve|reject) (?:the|this) (?:loan|credit)"
    r"|(?:assign|give|set|recommend) (?:a |an |the )?(?:credit )?rating|rate (?:this|the) (?:company|borrower))\b", re.I)

SYSTEM = ("You are a credit analyst assistant. Use ONLY the provided excerpts. "
          "Every item must include a verbatim quote (copied exactly, 15+ characters) from the excerpts. "
          "Never assign or suggest a credit rating or lending decision.")

PLAN_SYSTEM = ("You plan retrieval over an SEC 10-K for a credit analyst. Reply with JSON. "
               "action is 'search' (give a short query and optionally a section such as 'Item 1A'), "
               "'answer' (the observations are enough), or 'escalate' (the question cannot be answered from the filing). "
               "Think briefly in 'thought'. Do not answer the question yet.")

PLAN_SCHEMA = {"type": "object", "required": ["thought", "action"], "properties": {
    "thought": {"type": "string"}, "action": {"type": "string", "enum": ["search", "answer", "escalate"]},
    "query": {"type": "string"}, "section": {"type": "string"}}}

ITEMS_SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["claim", "quote"],
    "properties": {"claim": {"type": "string"}, "quote": {"type": "string"}}}}}}


class State(TypedDict, total=False):
    question: str
    doc: str
    observations: list
    action: dict
    searches: int
    attempts: int
    feedback: str
    items: list
    dropped: int
    status: str                       # running | done | escalated
    reason: str
    source: str                       # live | memory
    steps: list                       # trace of this question: ["recall: miss", "plan: search ...", ...]
    history: Annotated[list, operator.add]   # grows across questions on the same thread (short-term memory)


def build_agent(store, embed: Callable, llm_json: Callable, checkpointer=None):
    """store: MemoryStore or PgVectorStore. embed(list[str]) -> list[vector]. llm_json(system, user, schema) -> dict."""

    def add(state, msg):
        return list(state.get("steps", [])) + [msg]

    def start(state):
        return {"observations": [], "searches": 0, "attempts": 0, "feedback": "", "items": [], "dropped": 0,
                "status": "running", "reason": "", "source": "live", "steps": [], "action": {}}

    def guard(state):
        if DECISION_REQUEST.search(state["question"]):
            return {"status": "escalated", "reason": "Rating or lending decisions are a human analyst's call.",
                    "steps": add(state, "guard: decision request -> escalate")}
        return {"steps": add(state, "guard: ok")}

    def recall(state):
        hit = store.recall(state["doc"], embed([state["question"]])[0], MEMORY_MAX_DISTANCE, MEMORY_TTL_S)
        if hit:
            return {"status": "done", "source": "memory", "items": hit["answer"]["items"],
                    "steps": add(state, f"recall: hit (distance {hit['distance']:.3f})")}
        return {"steps": add(state, "recall: miss")}

    def plan(state):
        obs = "\n".join(f"- [{o['section']}] {o['text'][:200].replace(chr(10), ' ')}" for o in state["observations"][:8])
        user = (f"QUESTION: {state['question']}\nSEARCHES DONE: {state['searches']}/{MAX_SEARCHES}\n"
                f"OBSERVATIONS SO FAR:\n{obs or '(none)'}\n"
                + (f"FEEDBACK FROM LAST ATTEMPT: {state['feedback']}\n" if state["feedback"] else ""))
        a = llm_json(PLAN_SYSTEM, user, PLAN_SCHEMA)
        action = a.get("action")
        if action not in ("search", "answer", "escalate"):
            action = "search"
        # deterministic constraints on the model's choice
        if action == "answer" and not state["observations"]:
            action = "search"
        if action == "search" and state["searches"] >= MAX_SEARCHES:
            action = "answer" if state["observations"] else "escalate"
        out = {"action": {"action": action, "query": a.get("query") or state["question"], "section": a.get("section") or ""},
               "steps": add(state, f"plan: {action} ({(a.get('thought') or '')[:80]})")}
        if action == "escalate":
            out.update(status="escalated", reason="No answer can be supported from the filing text.")
        return out

    def search(state):
        act = state["action"]
        qv = embed([act["query"]])[0]
        res = (store.search(state["doc"], qv, 6, act["section"] or None) if act["section"] else []) \
            or store.search(state["doc"], qv, 6)
        seen = {o["text"] for o in state["observations"]}
        new = [r for r in res if r["text"] not in seen]
        return {"observations": state["observations"] + new, "searches": state["searches"] + 1,
                "steps": add(state, f"search: '{act['query'][:50]}' -> {len(new)} new passages")}

    def answer(state):
        ctx = "\n\n".join(f"[{o['section']}]\n{o['text']}" for o in state["observations"][:10])
        raw = llm_json(SYSTEM, f"Answer the question with up to 5 items.\nQUESTION: {state['question']}\n\nEXCERPTS:\n{ctx}",
                       ITEMS_SCHEMA).get("items", [])
        return {"items": raw, "steps": add(state, f"answer: {len(raw)} candidate items")}

    def verify(state):
        kept = [it for it in state["items"]
                if any(guardrails.quote_is_grounded(it.get("quote", ""), o["text"]) for o in state["observations"])
                and not guardrails.banned_hits(it.get("claim", ""), guardrails.CREDIT_BANNED)]
        dropped = len(state["items"]) - len(kept)
        if kept:
            return {"items": kept, "dropped": dropped, "status": "done",
                    "steps": add(state, f"verify: {len(kept)} grounded, {dropped} dropped")}
        attempts = state["attempts"] + 1
        if attempts >= MAX_ATTEMPTS:
            return {"attempts": attempts, "status": "escalated", "items": [],
                    "reason": "Answers could not be grounded in verbatim quotes after repeated attempts.",
                    "steps": add(state, "verify: nothing grounded -> escalate")}
        return {"attempts": attempts, "items": [],
                "feedback": "None of your quotes appeared verbatim in the observations. Search for more specific passages.",
                "steps": add(state, f"verify: nothing grounded (attempt {attempts}) -> replan")}

    def finalize(state):
        if state["status"] == "done" and state.get("source") == "live" and state["items"]:
            store.add_memory(state["doc"], state["question"], {"items": state["items"]}, embed([state["question"]])[0])
        rec = {"question": state["question"], "status": state["status"], "source": state.get("source", "live"),
               "n_items": len(state.get("items", [])), "reason": state.get("reason", "")}
        return {"history": [rec], "steps": add(state, f"finalize: {state['status']}")}

    g = StateGraph(State)
    for name, fn in [("start", start), ("guard", guard), ("recall", recall), ("plan", plan), ("search", search),
                     ("answer", answer), ("verify", verify), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "start")
    g.add_edge("start", "guard")
    g.add_conditional_edges("guard", lambda s: "finalize" if s["status"] == "escalated" else "recall")
    g.add_conditional_edges("recall", lambda s: "finalize" if s["status"] == "done" else "plan")
    g.add_conditional_edges("plan", lambda s: {"search": "search", "answer": "answer"}.get(s["action"]["action"], "finalize"))
    g.add_edge("search", "plan")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", lambda s: "plan" if s["status"] == "running" else "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def ask(agent, doc: str, question: str, thread_id: str = "default") -> dict:
    return agent.invoke({"doc": doc, "question": question}, {"configurable": {"thread_id": thread_id}})


def ingest(store, doc: str, html_path: str, embed: Callable, batch: int = 64) -> int:
    import rag
    chunks = rag.chunk(rag.html_to_sections(html_path), doc)
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        store.add_chunks(doc, part, embed([c["text"] for c in part]))
    return len(chunks)


def format_result(r: dict) -> str:
    lines = [f"Status: {r['status']} (source: {r.get('source', 'live')})"]
    if r["status"] == "escalated":
        lines.append(f"ESCALATED TO HUMAN ANALYST: {r['reason']}")
    for it in r.get("items", []):
        lines.append(f"- {it['claim']}\n    \"{it['quote']}\"")
    lines.append("Trace: " + " | ".join(r.get("steps", [])))
    return "\n".join(lines)


def main(argv):
    if len(argv) != 5 or argv[1] != "ask":
        print(__doc__)
        return 1
    import llm
    _, _, doc, path, question = argv
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        from vector_store import PgVectorStore
        store = PgVectorStore(dsn, dim=int(os.getenv("EMBED_DIM", "768")))
    else:
        from vector_store import MemoryStore
        store = MemoryStore()
    if store.count(doc) == 0:
        print(f"Ingesting {path} ...", flush=True)
        print(ingest(store, doc, path, llm.embed), "chunks")
    agent = build_agent(store, llm.embed, llm.chat_json)
    print(format_result(ask(agent, doc, question)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
