import pytest

import credit_agent
from vector_store import MemoryStore

DOC = "acme"
GOOD_QUOTE = "Our substantial indebtedness could adversely affect our financial condition"


class ScriptedLLM:
    """Replays planner/answer replies in order and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user, schema):
        self.calls.append(user)
        return self.replies.pop(0)


def plan(action, query="", section="", thought="t"):
    return {"thought": thought, "action": action, "query": query, "section": section}


def answer(*quotes):
    return {"items": [{"claim": f"claim {i}", "quote": q} for i, q in enumerate(quotes)]}


@pytest.fixture
def loaded(embed, chunks):
    s = MemoryStore()
    s.add_chunks(DOC, chunks, embed([c["text"] for c in chunks]))
    return s


def run(store, embed, llm, q="What are the main debt risks?", thread="t1", agent=None):
    agent = agent or credit_agent.build_agent(store, embed, llm)
    return agent, credit_agent.ask(agent, DOC, q, thread)


def test_search_then_grounded_answer_is_done_and_remembered(loaded, embed):
    llm = ScriptedLLM(plan("search", "indebtedness risk", "Item 1A"), plan("answer"), answer(GOOD_QUOTE))
    _, r = run(loaded, embed, llm)
    assert r["status"] == "done" and r["source"] == "live" and len(r["items"]) == 1
    assert any(s.startswith("search:") for s in r["steps"])
    assert loaded.memory and loaded.memory[0]["question"] == "What are the main debt risks?"


def test_ungrounded_quote_is_dropped_and_grounded_one_kept(loaded, embed):
    llm = ScriptedLLM(plan("search", "debt"), plan("answer"), answer(GOOD_QUOTE, "this sentence was made up by the model"))
    _, r = run(loaded, embed, llm)
    assert r["status"] == "done" and len(r["items"]) == 1 and r["dropped"] == 1


def test_ungrounded_then_replans_and_succeeds(loaded, embed):
    llm = ScriptedLLM(plan("search", "debt"), plan("answer"), answer("invented quote that is not in the filing"),
                      plan("search", "indebtedness"), plan("answer"), answer(GOOD_QUOTE))
    _, r = run(loaded, embed, llm)
    assert r["status"] == "done" and r["attempts"] == 1
    assert "None of your quotes" in llm.calls[3]          # the failure feedback reached the planner


def test_repeated_ungrounded_answers_escalate(loaded, embed):
    bad = answer("nothing like this exists in the text anywhere")
    llm = ScriptedLLM(plan("search", "x"), plan("answer"), bad, plan("answer"), bad, plan("answer"), bad)
    _, r = run(loaded, embed, llm)
    assert r["status"] == "escalated" and "grounded" in r["reason"] and r["items"] == []
    assert not loaded.memory                               # nothing unverified is remembered


def test_decision_request_escalates_without_calling_the_llm(loaded, embed):
    llm = ScriptedLLM()
    _, r = run(loaded, embed, llm, "Should we lend to this borrower? Please assign a credit rating.")
    assert r["status"] == "escalated" and llm.calls == []


def test_planner_can_escalate(loaded, embed):
    _, r = run(loaded, embed, ScriptedLLM(plan("escalate")), "What is the favourite colour of the CEO?")
    assert r["status"] == "escalated"


def test_answer_before_any_search_is_forced_to_search(loaded, embed):
    llm = ScriptedLLM(plan("answer"), plan("answer"), answer(GOOD_QUOTE))
    _, r = run(loaded, embed, llm)
    assert r["status"] == "done" and any(s.startswith("search:") for s in r["steps"])


def test_search_budget_is_capped(loaded, embed):
    replies = [plan("search", f"q{i}") for i in range(credit_agent.MAX_SEARCHES)] + [plan("search", "again"), answer(GOOD_QUOTE)]
    _, r = run(loaded, embed, ScriptedLLM(*replies))
    assert r["searches"] == credit_agent.MAX_SEARCHES and r["status"] == "done"


def test_near_identical_question_is_served_from_memory_without_llm(loaded, embed):
    llm = ScriptedLLM(plan("search", "debt risks"), plan("answer"), answer(GOOD_QUOTE))
    agent, first = run(loaded, embed, llm)
    assert first["source"] == "live"
    calls_before = len(llm.calls)
    r2 = credit_agent.ask(agent, DOC, "What are the main debt risks?", "t2")
    assert r2["source"] == "memory" and r2["status"] == "done" and len(llm.calls) == calls_before


def test_thread_history_persists_across_questions(loaded, embed):
    llm = ScriptedLLM(plan("search", "debt"), plan("answer"), answer(GOOD_QUOTE), plan("escalate"))
    agent, _ = run(loaded, embed, llm, "What are the main debt risks?", thread="same")
    r = credit_agent.ask(agent, DOC, "Something unanswerable about the weather", "same")
    assert [h["status"] for h in r["history"]] == ["done", "escalated"]
    other = credit_agent.ask(agent, DOC, "Should we approve the loan?", "different")
    assert len(other["history"]) == 1                       # a different thread has its own memory


def test_cli_formatting_marks_escalation(loaded, embed):
    _, r = run(loaded, embed, ScriptedLLM(), "Should we lend?")
    out = credit_agent.format_result(r)
    assert "ESCALATED TO HUMAN ANALYST" in out and "Trace:" in out
