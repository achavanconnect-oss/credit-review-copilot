"""Auditor-report early-warning checks on a 10-K. Deterministic keyword flags first; the LLM only
names critical audit matters, with a verbatim quote that must be grounded in the source chunk."""
import re
import llm, guardrails, audit

NL = chr(10)

# (flag, regex, why it matters) -- applied to the whole filing text, whitespace-normalised
FLAGS = [
    ("Going-concern doubt", r"substantial doubt (?:about|regarding|as to) (?:the |its |our )?(?:company'?s |entity'?s )?ability to continue as a going concern",
     "Auditor or management doubts survival; strongest early-warning signal"),
    ("Adverse or qualified opinion", r"(?:adverse|qualified) opinion", "Auditor did not give a clean opinion"),
    ("Material weakness (ICFR)", r"material weakness(?:es)? in (?:our )?internal control", "Control failures over financial reporting"),
    ("Restatement", r"(?:we (?:have )?restated|restatement of (?:our |the )?(?:previously issued |consolidated )?financial statements|as restated)",
     "Prior numbers were wrong (cover-page checkbox wording deliberately not matched)"),
    ("Covenant waiver/breach", r"(?:waiver of|not in compliance with|breach of|violation of) (?:a |the |certain )?(?:financial )?covenants?",
     "Possible covenant stress"),
]

CAM = {"type": "object", "required": ["matters"], "properties": {"matters": {"type": "array", "items": {
    "type": "object", "required": ["title", "quote"],
    "properties": {"title": {"type": "string"}, "quote": {"type": "string"}}}}}}


def _norm(t): return re.sub(r"\s+", " ", t)


def findings(chunks: list[dict]) -> str:
    full = _norm(" ".join(c["text"] for c in chunks))
    out = ["## Auditor report and early-warning flags", ""]
    for name, rx, why in FLAGS:
        m = re.search(rx, full, re.I)
        if m:
            s = max(0, m.start() - 160)
            out.append(f"- **{name}: mentioned.** {why}. Context: “…{full[s:m.end() + 160]}…” (analyst must read in full: a mention can be a negation or a risk-factor hypothetical)")
        else:
            out.append(f"- {name}: not found in text")
    # Critical audit matters: LLM names them, quote must be grounded
    cam_chunks = [c for c in chunks if re.search(r"critical audit matter", c["text"], re.I)]
    out += ["", "### Critical audit matters (CAMs)"]
    if not cam_chunks:
        out.append("- No 'critical audit matter' text found.")
    kept = dropped = 0
    seen = set()
    for c in cam_chunks:
        r = llm.chat_json("You read audit reports. Use only the excerpt.",
                          "List each critical audit matter described in this excerpt (title and one verbatim sentence). "
                          "If none are described, return an empty list." + NL + NL + c["text"], CAM)
        for m in r["matters"]:
            key = m["title"].strip().lower()
            if guardrails.quote_is_grounded(m["quote"], c["text"]) and key not in seen:
                seen.add(key)
                kept += 1
                out.append(f"- **{m['title']}** — “{m['quote']}” (*{c['section'].strip()}*)")
            else:
                dropped += 1
    audit.log("case1", "auditor_findings", {"cam_kept": kept, "cam_dropped": dropped, "model": llm.MODEL})
    out.append(f"{NL}_CAM grounding check: {kept} kept, {dropped} dropped. Earnings-call transcripts are not on EDGAR; add them from a licensed source._")
    return NL.join(out)
