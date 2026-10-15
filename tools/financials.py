"""Deterministic financial trends from SEC XBRL companyfacts. No LLM touches the numbers.
The LLM may only write commentary, and every number it uses is checked against the table.
Usage: financials.py <cik10> [out.md]
"""
import os, re, sys, json, datetime as dt, requests
import llm, guardrails, audit

UA = os.getenv("SEC_USER_AGENT", "CreditCopilotPilot research-pilot")
NL = chr(10)
# metric -> (kind, candidate us-gaap tags in priority order)
METRICS = {
    "Revenue": ("flow", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]),
    "Operating income": ("flow", ["OperatingIncomeLoss"]),
    "Net income": ("flow", ["NetIncomeLoss", "ProfitLoss"]),
    "Operating cash flow": ("flow", ["NetCashProvidedByUsedInOperatingActivities"]),
    "Total assets": ("stock", ["Assets"]),
    "Total liabilities": ("stock", ["Liabilities"]),
    "Equity": ("stock", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "Cash & equivalents": ("stock", ["CashAndCashEquivalentsAtCarryingValue"]),
}

def fetch_facts(cik: str) -> dict:
    path = os.path.join(".tmp", f"facts_{cik}.json")
    if not os.path.exists(path):
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        open(path, "w").write(r.text)
    return json.load(open(path))

def _days(a, b):
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days

def series(facts, kind, tags, years=3):
    """Per fiscal period end, take the latest-filed 10-K value; earlier tags win, later tags fill gaps."""
    best = {}
    for tag in tags:
        rows = facts["facts"]["us-gaap"].get(tag, {}).get("units", {}).get("USD", [])
        cur = {}
        for x in rows:
            if x.get("form") != "10-K":
                continue
            if kind == "flow" and not (350 <= _days(x["start"], x["end"]) <= 380):
                continue
            if x["end"] not in cur or x["filed"] > cur[x["end"]]["filed"]:
                cur[x["end"]] = {**x, "tag": tag}
        for e, v in cur.items():
            best.setdefault(e, v)
    return [best[e] for e in sorted(best)[-years:]]

def build(cik: str, years: int = 3):
    facts = fetch_facts(cik)
    table = {}
    for name, (kind, tags) in METRICS.items():
        table[name] = {x["end"][:4]: x for x in series(facts, kind, tags, years)}
    return facts["entityName"], table

def fmt(v): return f"-${abs(v)/1e9:,.1f}B" if v < 0 else f"${v/1e9:,.1f}B"

def ratios(table, y):
    g = lambda m: table.get(m, {}).get(y, {}).get("val")
    out = {}
    if g("Revenue") and g("Net income") is not None: out["Net margin"] = f"{g('Net income')/g('Revenue'):.1%}"
    if g("Revenue") and g("Operating income") is not None: out["Operating margin"] = f"{g('Operating income')/g('Revenue'):.1%}"
    if g("Equity") and g("Total liabilities"): out["Liabilities / equity"] = f"{g('Total liabilities')/g('Equity'):.1f}x"
    return out

def render(name, table):
    ys = sorted({y for m in table.values() for y in m})
    L = [f"## Financial trends: {name} (source: SEC XBRL, annual 10-K data)", "",
         "| Metric | " + " | ".join(ys) + " | Source tag |", "|---|" + "---|" * (len(ys) + 1)]
    for m, d in table.items():
        tag = ", ".join(sorted({v["tag"] for v in d.values()})) or "n/a"
        L.append(f"| {m} | " + " | ".join(fmt(d[y]["val"]) if y in d else "n/a" for y in ys) + f" | {tag} |")
    for r in ("Net margin", "Operating margin", "Liabilities / equity"):
        L.append(f"| {r} (computed) | " + " | ".join(ratios(table, y).get(r, "n/a") for y in ys) + " | derived |")
    L += ["", "_Total debt is not shown: filers use many different debt tags, so it is read from the filing text by the analyst._",
          "_Consolidated figures; for companies with finance arms (e.g. Ford Credit) these include the finance subsidiary._"]
    return NL.join(L)

NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

def commentary(name, table_md):
    """LLM summary of the table; rejected if it uses any number not present in the table."""
    text = llm.chat("You are a credit analyst assistant. Describe the multi-year trends in this table in 3-4 sentences. "
                    "Use only numbers exactly as written in the table. Do not give ratings or lending views.", table_md)
    allowed = set(NUM.findall(table_md))
    bad = [n for n in NUM.findall(text) if n not in allowed and not re.fullmatch(r"20\d\d", n)]
    audit.log("case1", "financial_commentary", {"entity": name, "bad_numbers": bad, "model": llm.MODEL})
    return (text, bad)

def section(cik: str) -> str:
    name, table = build(cik)
    md = render(name, table)
    text, bad = commentary(name, md)
    note = "" if not bad else f"{NL}> Commentary withheld: it used figures not in the table: {bad}"
    return md + NL + NL + ("**Commentary (AI-drafted):** " + text if not bad else note)

if __name__ == "__main__":
    out = section(sys.argv[1])
    print(out)
    if len(sys.argv) > 2: open(sys.argv[2], "w", encoding="utf-8").write(out)
