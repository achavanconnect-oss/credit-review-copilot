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
