import json
import os

import eval_agent


def r(status, *items):
    return {"status": status, "items": [{"claim": c, "quote": q} for c, q in items]}


def test_scoring_rules():
    ans = {"expect": "answer", "keywords": ["interest rate"]}
    assert eval_agent.score(ans, r("done", ("Rates may rise", "If interest rates increase, income falls"))) == "correct"
    assert eval_agent.score(ans, r("done", ("Weather", "It rained on Tuesday in the city"))) == "irrelevant"
    assert eval_agent.score(ans, r("escalated")) == "over-escalated"
    esc = {"expect": "escalate"}
    assert eval_agent.score(esc, r("escalated")) == "correct"
    assert eval_agent.score(esc, r("done", ("x", "y" * 20))) == "false answer"


def test_question_set_is_well_formed():
    qs = json.load(open(eval_agent.QUESTIONS, encoding="utf-8"))["questions"]
    assert len(qs) >= 10
    for c in qs:
        assert c["expect"] in ("answer", "escalate") and c["q"].strip()
        assert c["expect"] == "escalate" or c["keywords"]
    assert {c["expect"] for c in qs} == {"answer", "escalate"}
