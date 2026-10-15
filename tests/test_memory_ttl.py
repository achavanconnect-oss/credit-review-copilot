import time

import credit_agent
from vector_store import MemoryStore, PgVectorStore


def age_memory(store, seconds):
    """Make every remembered finding look `seconds` old, in whichever store is under test."""
    if isinstance(store, MemoryStore):
        for m in store.memory:
            m["ts"] -= seconds
    elif isinstance(store, PgVectorStore):
        store.conn.execute(f"UPDATE {store.p}_memory SET created_at = created_at - make_interval(secs => %s)", (seconds,))


def test_expired_memory_is_not_recalled_but_fresh_memory_is(store, embed):
    q = embed(["what are the key risk factors"])[0]
    store.add_memory("acme", "what are the key risk factors", {"items": []}, q)
    assert store.recall("acme", q, 0.1, max_age_s=3600) is not None
    age_memory(store, 2 * 3600)
    assert store.recall("acme", q, 0.1, max_age_s=3600) is None          # older than the TTL
    assert store.recall("acme", q, 0.1) is not None                       # no TTL given: still there


def test_agent_does_not_reuse_a_finding_older_than_its_ttl(embed, chunks, monkeypatch):
    from tests.test_credit_agent import DOC, GOOD_QUOTE, ScriptedLLM, answer, plan
    store = MemoryStore()
    store.add_chunks(DOC, chunks, embed([c["text"] for c in chunks]))
    llm = ScriptedLLM(plan("search", "debt"), plan("answer"), answer(GOOD_QUOTE),
                      plan("search", "debt"), plan("answer"), answer(GOOD_QUOTE))
    agent = credit_agent.build_agent(store, embed, llm)
    first = credit_agent.ask(agent, DOC, "What are the main debt risks?", "t1")
    assert first["source"] == "live"
    again = credit_agent.ask(agent, DOC, "What are the main debt risks?", "t2")
    assert again["source"] == "memory"
    age_memory(store, credit_agent.MEMORY_TTL_S + 60)
    stale = credit_agent.ask(agent, DOC, "What are the main debt risks?", "t3")
    assert stale["source"] == "live"                                      # re-answered from the filing, not memory
    assert time.time() > 0
