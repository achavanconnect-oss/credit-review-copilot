"""Case 1: LLM-driven credit memo draft from two 10-K filings (analyst co-pilot, no ratings).
Usage: credit_review.py <borrower> <prior.htm> <current.htm> [out.md]
"""
import sys, os, json, hashlib
import numpy as np
import llm, rag, guardrails, audit, financials, auditor, eval_yoy

SYSTEM = ("You are a credit analyst assistant. Use ONLY the provided excerpts. "
          "Every item must include a verbatim quote (copied exactly, 15+ characters) from the excerpts. "
          "Never assign or suggest a credit rating or lending decision.")
ITEMS = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["claim", "quote"],
    "properties": {"claim": {"type": "string"}, "quote": {"type": "string"}}}}}}
TASKS = {
    "Risk factors": ("key risk factors", "risk factors adverse material uncertainty", "Item 1A"),
    "Covenants & liquidity": ("covenant, debt, liquidity or credit-facility disclosures",
                              "covenants credit facilities liquidity debt maturities", "Item 7"),
}

def build(path: str, doc: str):
    idx_path = os.path.join(".tmp", "index", doc)
    if not os.path.exists(idx_path + ".npy"):
        chunks = rag.chunk(rag.html_to_sections(path), doc)
        print(f"{doc}: {len(chunks)} chunks, embedding...", flush=True)
        rag.build_index(chunks, idx_path)
    return rag.load_index(idx_path)

def extract(index, task: str, doc: str):
    what, query, sec = TASKS[task]
    ctx = rag.search(index, query, k=6, section_filter=sec) or rag.search(index, query, k=6)
    excerpts = "\n\n".join(f"[{c['section']}]\n{c['text']}" for c in ctx)
    raw = llm.chat_json(SYSTEM, f"Extract up to 5 {what}.\n\nEXCERPTS:\n{excerpts}", ITEMS)["items"]
    kept, dropped = [], 0
    for it in raw:
        src = next((c for c in ctx if guardrails.quote_is_grounded(it["quote"], c["text"])), None)
        if src:
            kept.append({**it, "section": src["section"], "doc": doc})
        else:
            dropped += 1
    audit.log("case1", "extract", {"doc": doc, "task": task, "kept": len(kept), "dropped": dropped,
                                   "model": llm.MODEL, "prompt_hash": hashlib.sha1(excerpts.encode()).hexdigest()})
    return kept, dropped


NL = chr(10)
CHUNK_SAME = 0.90  # cosine on source-text chunk embeddings (0.85 missed material changes vs the lexical baseline)
LEX_NEW = 0.30     # or: >=30% of the chunk's words sit in sentences absent from the other year (plain diff, no AI)

LABEL = {"type": "object", "required": ["topic", "quote"],
         "properties": {"topic": {"type": "string"}, "quote": {"type": "string"}}}

def compare_filings(prior_idx, current_idx, section_key="1A"):
    """Deterministic YoY on source text: a passage is flagged by embedding similarity or by plain word diff.
    The LLM only labels each flagged passage; its quote must be verbatim from that passage."""
    def sel(idx):
        v, c = idx
        ix = [i for i, x in enumerate(c) if section_key in x["section"]]
        w = v[ix] / np.linalg.norm(v[ix], axis=1, keepdims=True)
        return w, [c[i] for i in ix]
    pv, pc = sel(prior_idx)
    cv, cc = sel(current_idx)
    sim = cv @ pv.T
    pshing = set().union(*[eval_yoy.shingles(c["text"]) for c in pc])
    cshing = set().union(*[eval_yoy.shingles(c["text"]) for c in cc])
    cflag = [i for i in range(len(cc)) if sim[i].max() < CHUNK_SAME or eval_yoy.new_fraction(cc[i]["text"], pshing)[0] >= LEX_NEW]
    pflag = [j for j in range(len(pc)) if sim[:, j].max() < CHUNK_SAME or eval_yoy.new_fraction(pc[j]["text"], cshing)[0] >= LEX_NEW]
    flagged = {"NEW or CHANGED in current report": [(cc[i], float(sim[i].max())) for i in cflag],
               "No close match in current report (removed or reworded)": [(pc[j], float(sim[:, j].max())) for j in pflag]}
    matched = len(cc) - len(cflag)
    lines = [f"{matched} of {len(cc)} current-year Item {section_key} passages were unchanged versus the prior year "
             f"(flag rule: embedding similarity < {CHUNK_SAME} OR >= {LEX_NEW:.0%} new words). Flagged passages were selected by code, not by the LLM.", ""]
    for title, items in flagged.items():
        lines += [f"### {title} ({len(items)})"]
        for c, s in items:
            r = llm.chat_json(SYSTEM, "In one short phrase, name the topic of this passage, and copy one verbatim sentence from it." + NL + NL + c["text"], LABEL)
            ok = guardrails.quote_is_grounded(r["quote"], c["text"])
            lines.append(f"- **{r['topic']}** (best-match similarity {s:.2f}, {c['section'].strip()})")
            lines.append(f"  - " + (f"“{r['quote']}”" if ok else "(quote failed grounding check; passage start) “" + c["text"][:200].replace(NL, " ") + "”"))
        if not items:
            lines.append("- none")
    audit.log("case1", "compare_filings", {"matched": matched, "total": len(cc), "model": llm.MODEL})
    return NL.join(lines)

def main(borrower, prior, current, out, cik=None):
    cur_idx = build(current, f"{borrower}_current")
    prior_idx = build(prior, f"{borrower}_prior")
    results, stats = {}, {}
    for task in TASKS:  # current year only; the prior year is used for comparison, not summarised
        kept, dropped = extract(cur_idx, task, f"{borrower}_current")
        results[task] = kept
        stats[task] = (len(kept), dropped)
        print(f"{task}: kept {len(kept)}, dropped {dropped} ungrounded", flush=True)

    fin = financials.section(cik) if cik else "## Financial trends" + NL + "_No CIK given; skipped._"
    aud = auditor.findings(cur_idx[1])
    diff = compare_filings(prior_idx, cur_idx)
    hits = guardrails.banned_hits(NL.join([diff, aud, fin]), guardrails.CREDIT_BANNED)

    md = [f"# Credit memo draft: {borrower}",
          "> **DRAFT — AI-assisted. Analyst review and approval required. No rating or lending decision is made by this tool.**",
          f"> Model: {llm.PROVIDER}/{llm.MODEL}" + NL, fin, ""]
    for task in TASKS:
        md.append(f"## {task} (current year)")
        for x in results[task]:
            md.append(f"- {x['claim']}{NL}  - Source: *{x['section']}* — “{x['quote']}”")
        k, d = stats[task]
        md.append(f"{NL}_Grounding check: {k} kept, {d} dropped as not found in source._{NL}")
    md += [aud, "", "## Year-on-year change in risk-factor disclosure (Item 1A)" + NL + diff]
    if hits:
        md.append(f"{NL}**Guardrail warning: banned phrases in output: {hits}**")
    md.append(NL + "## Analyst sign-off" + NL + "- [ ] Reviewed against source filings  - [ ] Approved  - Name/date: ________")
    open(out, "w", encoding="utf-8").write(NL.join(md))
    audit.log("case1", "memo_written", {"out": out, "banned_hits": hits})
    print("wrote", out)

if __name__ == "__main__":
    b, p, c = sys.argv[1:4]
    o = sys.argv[4] if len(sys.argv) > 4 else f".tmp/{b}_credit_memo.md"
    main(b, p, c, o, sys.argv[5] if len(sys.argv) > 5 else None)
