"""Multi-company accuracy test: table-aware retrieval + LLM reading, scored against SEC XBRL ground truth.

Per company: latest 10-K (fetch_10k.py) and, for each metric, the latest two fiscal years. Reports
 - retrieval recall (was the true figure visible in the context given to the model?)
 - accuracy overall and given retrieval success
 - failure types: refused, wrong year, scale error, other wrong
Provider/model from .env, so the same run works on Ollama or Claude.

Usage: eval_tables.py [--retrieval-only] [--tag label] TICKER [TICKER ...]
"""
import sys, os, json, time
import llm, financials, tables

NL = chr(10)
SCHEMA = {"type": "object", "required": ["found", "value", "unit", "quote"],
          "properties": {"found": {"type": "boolean"}, "value": {"type": "number"},
                         "unit": {"type": "string", "enum": ["units", "thousands", "millions", "billions"]},
                         "quote": {"type": "string"}}}
SYSTEM = ("You read table excerpts from a 10-K. Find the requested figure for the requested fiscal period using the "
          "column headers. Return the number exactly as printed in the table (do not rescale), with the table's unit "
          "(see the [Table units] line). If the figure is not in the excerpts, set found=false. "
          "Quote the exact table row. If no column header shows the period, the first number column is normally the most "
          "recent fiscal year, the next column the prior year.")
FACT = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
FLOW = {"Revenue", "Net income", "Operating cash flow"}


import re
ANS_NUM = re.compile(r"(\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?)")
UNIT_WORD = re.compile(r"\b(thousand|million|billion)s?\b|\b([kmb])\b|\b(bn|mm)\b", re.I)
PLAIN_SYSTEM = ("You read table excerpts from a 10-K. Find the requested figure for the requested fiscal period using the "
                "column headers. If no header shows the period, the first number column is normally the most recent year. "
                "Answer with the number and unit only, e.g. '49,953 million'. Only if the row is absent, answer 'NOT FOUND'.")


def ask_plain(ask, ctx):
    text = llm.chat(PLAIN_SYSTEM, ask).strip()
    if re.search(r"not found|cannot|unable|n/a", text, re.I) and not ANS_NUM.search(text):
        return {"found": False, "value": 0, "unit": "units", "quote": ""}
    m = ANS_NUM.search(text)
    if not m:
        return {"found": False, "value": 0, "unit": "units", "quote": ""}
    tok = m.group(1)
    neg = tok.startswith("(") or "-" in tok
    val = float(re.sub(r"[^\d.]", "", tok))
    um = UNIT_WORD.search(text[m.end():]) or UNIT_WORD.search(text)
    w = ((um.group(1) or um.group(2) or um.group(3) or "") if um else "").lower()
    unit = {"thousand": "thousands", "k": "thousands", "million": "millions", "m": "millions", "mm": "millions",
            "billion": "billions", "b": "billions", "bn": "billions"}.get(w)
    if unit is None:  # fall back to the table's own unit line
        mu = re.search(r"\[Table units: in (\w+)\]", ctx)
        unit = mu.group(1) if mu and mu.group(1) in FACT else "units"
    return {"found": True, "value": -val if neg else val, "unit": unit, "quote": text}


def truths(cik, latest_end, n=2):
    """Ground truth for the filing's latest two fiscal years only (stale XBRL tags can otherwise supply periods that
    are not in the filing at all)."""
    import datetime as dt
    name, table = financials.build(cik, years=6)
    out = {}
    for metric, d in table.items():
        for yr, row in d.items():
            age = (dt.date.fromisoformat(latest_end) - dt.date.fromisoformat(row["end"])).days / 365.25
            if -0.1 <= age <= 1.6:
                out.setdefault(metric, []).append((row["end"], row["val"]))
    return out


def main(tickers, retrieval_only, tag):
    rows = []
    for tk in tickers:
        meta = json.load(open(f".tmp/filings/{tk}.json"))
        try:
            fin = truths(meta["cik"], meta["report_date"])
            tabs = tables.parse(f".tmp/filings/{tk}.htm")
        except Exception as e:
            print(tk, "skipped:", e)
            continue
        print(f"{tk}: {len(tabs)} tables", flush=True)
        for metric in tables.METRIC_PATTERNS:
            for end, tv in sorted(fin.get(metric, []))[-2:]:
                year = end[:4]
                ctx, used = tables.find_context(tabs, metric, year, end_date=end, latest_end=meta['report_date'])
                present = tables.number_present(used, tv)
                rec = {"ticker": tk, "metric": metric, "end": end, "truth_usd": tv, "context_has_figure": present,
                       "context_found": bool(ctx)}
                cv, _note = tables.code_extract(tabs, metric, end, latest_end=meta['report_date'])
                rec.update(code_attempted=cv is not None, code_assumed=str(_note).startswith('assumed'),
                           code_correct=cv is not None and abs(cv - tv) <= 0.005 * abs(tv) and (cv < 0) == (tv < 0))
                if not retrieval_only and ctx:
                    ask = (f"Requested: {metric} for the fiscal period ending {end} (company: {meta['name']}). "
                           f"Pick the column whose header matches that period (a column headed {year} or the date); "
                           f"companies with January year-ends label columns by the calendar year of the period end. "
                           f"Set found=false only if the row is absent." + NL + NL + "EXCERPTS:" + NL + ctx[:7000])
                    try:
                        r = ask_plain(ask, ctx)
                    except Exception as e:
                        r = {"found": False, "value": 0, "unit": "units", "quote": "", "err": str(e)[:60]}
                    got = r["value"] * FACT[r["unit"]] if r["found"] else None
                    ok = got is not None and abs(abs(got) - abs(tv)) <= 0.005 * abs(tv) and (got < 0) == (tv < 0)
                    # wrong-year: model returned another column's value for the same row
                    rec.update(got_usd=got, correct=ok, refused=got is None)
                    if got is not None and not ok:
                        other = [abs(tables._num(v) or 0) * (t["factor"] or 1) for t in used for lab, vals in t["rows"] for v in vals]
                        rec["wrong_type"] = ("scale" if any(abs(abs(got) * m - abs(tv)) <= 0.005 * abs(tv) for m in (1e3, 1e-3, 1e6, 1e-6))
                                             else "other_column_or_year" if any(abs(abs(got) - o) <= 0.005 * abs(got) for o in other if o) else "not_in_context")
                rows.append(rec)
                print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rec.items() if k != "truth_usd"}, flush=True)
    n = len(rows)
    summ = {"model": f"{llm.PROVIDER}/{llm.MODEL}", "companies": len({r["ticker"] for r in rows}), "questions": n,
            "retrieval_recall": round(sum(r["context_has_figure"] for r in rows) / max(n, 1), 3)}
    if not retrieval_only:
        ok = [r for r in rows if r.get("correct")]
        rec_rows = [r for r in rows if r["context_has_figure"]]
        summ.update(accuracy=round(len(ok) / max(n, 1), 3),
                    accuracy_given_figure_in_context=round(sum(r.get("correct", False) for r in rec_rows) / max(len(rec_rows), 1), 3),
                    refused=sum(r.get("refused", False) for r in rows))
        wt = {}
        for r in rows:
            if r.get("wrong_type"):
                wt[r["wrong_type"]] = wt.get(r["wrong_type"], 0) + 1
        summ["wrong_types"] = wt
        # per-company accuracy and a simple spread, so a range can be reported
        per = {}
        for r in rows:
            per.setdefault(r["ticker"], []).append(r.get("correct", False))
        summ["per_company_accuracy"] = {k: round(sum(v) / len(v), 2) for k, v in per.items()}
    att = [r for r in rows if r["code_attempted"]]
    summ["code_only"] = {"accuracy": round(sum(r["code_correct"] for r in rows) / max(n, 1), 3),
                         "attempted": len(att),
                         "accuracy_when_attempted": round(sum(r["code_correct"] for r in att) / max(len(att), 1), 3)}
    print("SUMMARY", json.dumps(summ))
    os.makedirs(".tmp/eval", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(f".tmp/eval/tables_{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    a = sys.argv[1:]
    ro = "--retrieval-only" in a
    a = [x for x in a if x != "--retrieval-only"]
    tag = "local"
    if "--tag" in a:
        i = a.index("--tag"); tag = a[i + 1]; a = a[:i] + a[i + 2:]
    main(a, ro, tag)
