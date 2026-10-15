# Credit Review Copilot

An analyst co-pilot that drafts a credit memo from two consecutive SEC 10-K filings, built around one idea:
**an LLM should never be trusted with numbers or unsourced claims.** It makes no rating or lending decision.

## How it works

| Step | What happens | LLM involved? |
|---|---|---|
| Ingest | 10-K HTML is split by `Item` section into chunks, embedded, stored as a NumPy index (`rag.py`) | embeddings only |
| Financial trends | Figures come from SEC XBRL `companyfacts`, never from the model (`financials.py`) | commentary only; every number it uses is checked against the table |
| Risk factors, covenants, liquidity | Retrieval over the relevant Items; model returns claim + **verbatim quote** | yes, then quote is verified |
| Grounding check | A quote is kept only if it appears verbatim in the retrieved source chunk (`guardrails.py`) | no |
| Auditor early-warning flags | Regex flags for going-concern doubt, restatement, material weakness, covenant breach (`auditor.py`) | model only names critical audit matters, quote must ground |
| Year-on-year risk-factor change | Passages flagged by embedding similarity or plain word diff; the LLM only labels what code flagged | labelling only |
| Audit trail | Append-only SQLite log of every call, with prompt hash and model (`audit.py`) | no |
| Output | Markdown memo marked DRAFT with an analyst sign-off checklist; banned phrases (ratings, decisions) trigger a warning | |

Provider-swappable: runs on a local Ollama model (free) or Anthropic via `.env`.

## Evaluation (the part that matters)

Extraction accuracy is scored against SEC XBRL ground truth (`eval_extraction.py`, `eval_tables.py`, `eval_yoy.py`).

- First run on Ford's filings, local 7B model: 8 of 34 figures right (23.5% in the logged run), 16 refusals,
  and the grounding check caught 5 wrong answers. Much of the early miss was the test harness, not the model.
- Table-aware retrieval + reading, 10 companies, 132 questions, Qwen2.5-7B: **70.5% accuracy**
  (87.1% retrieval recall; 80.9% when the figure was in the context).
- Tuning on misses lifted a held-out set to 83%. On 6 companies never touched, accuracy dropped to **61.5%**,
  and a plain rule-based script with no AI scored 68%. The 83% was overfitting.

**Conclusion:** numbers come from XBRL; the model only describes them, and every quote must exist in the filing.
A human signs off.

## Agent mode (LangGraph + pgvector)

`tools/credit_agent.py` turns the pipeline into a ReAct-style agent built on LangGraph:

```
start -> guard -> recall -> plan <-> search
                              \-> answer -> verify -> (plan | finalize)
```

- **Plan / act / observe:** the LLM chooses `search` (query + optional 10-K Item), `answer`, or `escalate`; each search result is an observation fed back into the next plan.
- **Deterministic limits on the model:** it cannot answer before searching, search budget is capped (4), and answers are only accepted if every kept claim has a verbatim quote found in the retrieved text.
- **Escalation to a human:** rating or lending-decision requests are refused before any LLM call; repeated ungrounded answers, or a planner that finds no evidence, end as `escalated` with a reason.
- **Memory:** short-term via a LangGraph checkpointer (`thread_id` keeps per-conversation history); long-term via the vector store, where verified findings are saved and near-identical questions are answered from memory with no LLM call.
- **Vector store:** `tools/vector_store.py` has `PgVectorStore` (PostgreSQL + pgvector, cosine distance, HNSW index, SQL section filter) and an in-memory `MemoryStore` with the same interface.

```bash
# in-memory (no database), local Ollama models
python tools/credit_agent.py ask KO .tmp/filings/KO.htm "What are the main debt and liquidity risks?"

# with pgvector
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
python tools/credit_agent.py ask KO .tmp/filings/KO.htm "What are the main debt and liquidity risks?"
```

Tests (`pytest`) cover the agent with a scripted LLM (grounding, replanning, escalation, budget caps, memory, thread isolation)
and run the same store tests against both stores. The pgvector tests are skipped without `DATABASE_URL`; GitHub Actions runs them against a
`pgvector/pgvector:pg16` service container.

## Limitations
Large US companies only, two fiscal years each, one local 7B model, public data. Not credit or investment advice.

## Usage
```bash
pip install -r requirements.txt
cp .env.example .env            # then edit
python tools/fetch_10k.py F     # downloads the latest 10-K to .tmp/filings/
python tools/credit_review.py Ford prior.htm current.htm out.md 0000037996
```
Requires [Ollama](https://ollama.com) with `qwen2.5:7b-instruct` and `nomic-embed-text`, or an Anthropic key.
