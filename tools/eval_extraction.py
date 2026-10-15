"""Accuracy test: can the LLM read reported figures from a 10-K, scored against SEC XBRL ground truth?

For each filing and metric, the model is asked for the value for each fiscal year in the filing (RAG over the
filing's own chunks). Scored: correct (within 0.5%), wrong, refused/missing, and whether the cited quote is
grounded in the retrieved text. Provider/model come from .env, so the same test runs on Ollama or Claude.

Usage: eval_extraction.py <cik10> <doc_name:fiscal_year> [<doc_name:fiscal_year> ...] [--tag label]
e.g.   eval_extraction.py 0000037996 ford_prior:2024 ford_current:2025
"""
import sys, json, os, time
import llm, rag, guardrails, financials

NL = chr(10)
SCHEMA = {"type": "object", "required": ["found", "value_in_millions_usd", "quote"],
          "properties": {"found": {"type": "boolean"},
                         "value_in_millions_usd": {"type": "number"},
                         "quote": {"type": "string"}}}
# metric -> retrieval query, fiscal-year offsets to ask for (0 = filing year)
Q = {
    "Revenue": ("Total revenues consolidated income statement", [0, -1, -2]),
    "Net income": ("Net income (loss) attributable to Ford Motor Company consolidated income statement", [0, -1, -2]),
    "Operating cash flow": ("Net cash provided by (used in) operating activities consolidated statement of cash flows", [0, -1, -2]),
    "Total assets": ("Total assets consolidated balance sheet", [0, -1]),
    "Total liabilities": ("Total liabilities consolidated balance sheet", [0, -1]),
    "Equity": ("Total equity attributable to Ford Motor Company balance sheet", [0, -1]),
    "Cash & equivalents": ("Cash and cash equivalents balance sheet", [0, -1]),
}
SYSTEM = ("You read excerpts of a 10-K. Return the requested figure exactly as reported in the CONSOLIDATED statements, "
          "converted to millions of USD. If the excerpts do not contain it, set found=false. "
          "Quote the exact text (a table row or sentence) that contains the figure.")


def truth_for(table, metric, year):
    r = table.get(metric, {}).get(str(year))
    return r["val"] / 1e6 if r else None


def main(cik, specs, tag):
    name, table = financials.build(cik, years=6)
    rows = []
    for spec in specs:
        doc, fy = spec.split(":")
        fy = int(fy)
        index = rag.load_index(os.path.join(".tmp", "index", doc))
        chunks_all = index[1]
        for metric, (query, offsets) in Q.items():
            ctx = rag.search(index, query, k=5)
            text = NL.join(c["text"] for c in ctx)
            for off in offsets:
                year = fy + off
                tv = truth_for(table, metric, year)
                if tv is None:
                    continue
                ask = f"What is {metric} for fiscal year {year}? (consolidated, in millions of USD)" + NL + NL + "EXCERPTS:" + NL + text
                t0 = time.time()
                try:
                    r = llm.chat_json(SYSTEM, ask, SCHEMA)
                except Exception as e:
                    r = {"found": False, "value_in_millions_usd": 0, "quote": "", "error": str(e)[:80]}
                got = r["value_in_millions_usd"] if r["found"] else None
                ok = got is not None and abs(got - tv) <= 0.005 * abs(tv)
                # model sometimes returns billions or raw dollars: record the scale error separately
                scale = got is not None and not ok and any(abs(got * m - tv) <= 0.005 * abs(tv) for m in (1000, 1e6, 0.001))
                grounded = bool(r["quote"]) and any(guardrails.quote_is_grounded(r["quote"], c["text"]) for c in ctx)
                rows.append({"filing": doc, "metric": metric, "year": year, "truth": round(tv, 1), "got": got,
                             "correct": ok, "scale_error": scale, "refused": got is None, "quote_grounded": grounded,
                             "secs": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
    n = len(rows)
    c = sum(r["correct"] for r in rows)
    ref = sum(r["refused"] for r in rows)
    sc = sum(r["scale_error"] for r in rows)
    gr = sum(r["quote_grounded"] for r in rows)
    wrong_ungrounded = sum((not r["correct"]) and (not r["refused"]) and not r["quote_grounded"] for r in rows)
    wrong_total = n - c - ref
    summ = {"model": f"{llm.PROVIDER}/{llm.MODEL}", "questions": n, "accuracy": round(c / n, 3),
            "refused": ref, "wrong": wrong_total, "scale_errors": sc, "quotes_grounded": round(gr / n, 3),
            "wrong_answers_caught_by_grounding_check": wrong_ungrounded}
    print("SUMMARY", json.dumps(summ))
    os.makedirs(".tmp/eval", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(f".tmp/eval/extraction_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    tag = "local"
    if "--tag" in a:
        i = a.index("--tag")
        tag = a[i + 1]
        a = a[:i] + a[i + 2:]
    main(a[0], a[1:], tag)
