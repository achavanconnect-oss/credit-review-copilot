"""Table-aware parsing of 10-K HTML. Keeps each figure attached to its row label, column headers and units.
A table becomes a list of rows: label + cleaned numeric cells. Retrieval finds tables by ROW LABEL, not by embeddings.
"""
import re
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
NL = chr(10)
NUMCELL = re.compile(r"^\(?-?\$?\s*[\d,]+(?:\.\d+)?\)?%?$|^[—–-]$")
UNIT = re.compile(r"(?:\bin\s+|\()\s*(?:\$\s*)?(millions|thousands|billions)\b", re.I)
FACTOR = {"millions": 1e6, "thousands": 1e3, "billions": 1e9}


def _clean_cells(tr):
    raw = [re.sub(r"\s+", " ", c.get_text(" ").replace(chr(0x200b), "").replace(chr(0xa0), " ")).strip()
           for c in tr.find_all(["td", "th"])]
    out, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c in ("", "$", "%", "(", ")") :
            i += 1
            continue
        if c.startswith("(") and not c.endswith(")") and re.match(r"^\(\$?\s*[\d,\.]+$", c):
            c += ")"  # the closing parenthesis sits in the next cell
        out.append(c)
        i += 1
    return out


def _num(c):
    t = c.replace("$", "").replace(",", "").replace(" ", "")
    neg = t.startswith("(") or t.startswith("-")
    t = t.strip("()-%")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def parse(path):
    """Return list of tables: {rows:[(label,[cells])], header:[rows], unit:str, factor:float, idx:int}"""
    soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "lxml")
    tables = []
    for ti, tb in enumerate(soup.find_all("table")):
        rows = [_clean_cells(tr) for tr in tb.find_all("tr")]
        rows = [r for r in rows if r]
        if len(rows) < 3:
            continue
        # unit: look in the table itself, then in the text just before it
        head_txt = " ".join(" ".join(r) for r in rows[:4])
        m = UNIT.search(head_txt)
        if not m:
            prev = " ".join(s.strip() for s in list(tb.find_all_previous(string=True, limit=12)))
            ms = UNIT.findall(prev)
            m = UNIT.search(prev) if ms else None
        unit = m.group(1).lower() if m else "unknown"
        parsed = []
        for r in rows:
            vals = [c for c in r if NUMCELL.match(c)]
            label = r[0] if r and not NUMCELL.match(r[0]) else ""
            parsed.append((label, vals if label else r))
        tables.append({"idx": ti, "rows": parsed, "unit": unit, "factor": FACTOR.get(unit)})
    known = [t["unit"] for t in tables if t["unit"] != "unknown"]
    if known:  # a filing normally states one unit for its tables; fall back to the most common one (flagged)
        dom = max(set(known), key=known.count)
        for t in tables:
            if t["unit"] == "unknown":
                t["unit"], t["factor"], t["unit_inferred"] = dom, FACTOR[dom], True
    return tables


METRIC_PATTERNS = {
    "Revenue": [r"^total (net )?revenues?( and other income)?$", r"^(total )?net sales$", r"^net sales$", r"^total net revenue$",
                r"^revenues?$", r"^total revenues? and other income$", r"^sales and other operating revenue$", r"^revenues?, net$",
                r"^total operating revenue$", r"^net revenues?$", r"^net sales and other revenues?$",
                r"^total net operating revenues?$", r"^net operating revenues?$", r"^sales$", r"^total sales$",
                r"^total revenues?,? net of interest expense$"],
    "Net income": [r"^net (income|earnings)( \(loss\))?( attributable to [\w\.,&\s\-]+)?$", r"^net (loss|income \(loss\))$",
                   r"^net income \(loss\) attributable to [\w\.,&\s\-]+$", r"^consolidated net income( \(loss\))?$"],
    "Operating cash flow": [r"^(total )?(net )?cash( flows?)?[ ]*(provided|generated|from|\(used|used)[^|$]{0,50}operating activities[\w\s\(\)\-,/]*$",
                            r"^net cash (provided by|from) operations$", r"^cash flows? from operating activities.*$"],
    "Total assets": [r"^total assets$"],
    "Total liabilities": [r"^total liabilities$"],
    "Equity": [r"^total (stockholders|shareholders)['’]? ?(equity|investment)( \(deficit\))?$", r"^total equity$",
               r"^total (stockholders|shareholders)['’] equity \(deficit\)$", r"^total equity attributable to [\w\.,&\s\-]+$"],
    "Cash & equivalents": [r"^cash and cash equivalents$", r"^cash and cash equivalents, end of (year|period)$",
                           r"^cash, cash equivalents,? and restricted cash.*end of.*$"],
}


YEAR_RE = re.compile(r"(?:19|20)\d\d")


def header_years(t):
    """Find the header row holding the period labels. Returns (row_index, [year per number column]) or (None, [])."""
    best = (None, [])
    for i, (lab, vals) in enumerate(t["rows"][:10]):
        cells = list(vals) if lab == "" or not vals else [lab] + list(vals)
        ys = []
        for c in cells:
            m = YEAR_RE.findall(c)
            if m and (_num(c) is None or 1990 <= abs(_num(c)) <= 2100):
                ys.append(int(m[-1]))
        if len(ys) >= 2 and len(ys) > len(best[1]):
            best = (i, ys)
    return best


def target_col(years, end_date, latest_end=None):
    """Column index (0-based among the number columns) for the fiscal period ending end_date, or None.
    Filings list periods newest-first, so when the header years are strictly descending the column is the number of
    years between the filing's latest period end and end_date. This is robust to companies labelling fiscal years
    differently (by start year, end year, or 'fiscal 2024'). Falls back to matching the label."""
    import datetime as dt
    desc = all(years[i] > years[i + 1] for i in range(len(years) - 1))
    if desc and latest_end:
        d = (dt.date.fromisoformat(latest_end) - dt.date.fromisoformat(end_date)).days
        col = round(d / 365.25)
        if 0 <= col < len(years) and abs(d - col * 365.25) < 40:
            return col
        return None
    y = int(end_date[:4])
    return years.index(y) if y in years else None


def assumed_col(end_date, latest_end):
    """Newest-first convention for tables with no year header."""
    import datetime as dt
    d = (dt.date.fromisoformat(latest_end) - dt.date.fromisoformat(end_date)).days
    col = round(d / 365.25)
    return col if 0 <= col <= 4 and abs(d - col * 365.25) < 40 else None


def _candidates(tables, metric, year_hint):
    pats = [re.compile(p, re.I) for p in METRIC_PATTERNS[metric]]
    cands = []
    for t in tables:
        hits = [i for i, (lab, vals) in enumerate(t["rows"]) if lab and any(p.match(lab.strip(" :")) for p in pats)
                and sum(1 for v in vals if _num(v) is not None) >= 2]
        if hits:
            head = " ".join(" ".join(r[1]) + " " + r[0] for r in t["rows"][:6])
            cands.append((str(year_hint) in head, len(t["rows"]), t, hits))
    # prefer tables whose header shows the target year, then larger tables (full statements beat segment tables)
    cands.sort(key=lambda x: (-x[0], -x[1]))
    return [(c[2], c[3]) for c in cands]


def _annotated(lab, vals, years):
    """'Total assets | 2026: 49,953 | 2025: 52,616' when the number of values matches the header; otherwise plain."""
    if years and len(vals) >= len(years):
        cells = [f"{y}: {v}" for y, v in zip(years, vals)] + list(vals[len(years):])
        return f"{lab} | " + " | ".join(cells)
    return f"{lab} | " + " | ".join(vals)


def find_context(tables, metric, year_hint, max_tables=3, neighbours=1, end_date=None, latest_end=None):
    """Return (text, tables_used). Each value is labelled with its column year (mapped in code from the header row)."""
    parts, used = [], []
    for t, hits in _candidates(tables, metric, year_hint)[:max_tables]:
        used.append(t)
        lines = [f"[Table units: in {t['unit']}]"]
        hi, years = header_years(t)
        if hi is not None:
            lab, vals = t["rows"][hi]
            lines.append(((lab + " | ") if lab else "") + " | ".join(vals))
        else:
            lines.append("(no column header with years found; first number column is normally the most recent year)")
        if hi is not None and end_date:
            col = target_col(years, end_date, latest_end)
            if col is not None:
                lines.append(f"(The period ending {end_date} is the column labelled {years[col]}.)")
        elif hi is None and end_date and latest_end:
            ac = assumed_col(end_date, latest_end)
            if ac is not None:
                lines.append(f"(No year header. Columns are assumed newest-first: the period ending {end_date} is number {ac + 1} in each row.)")
        if t.get("unit_inferred"):
            lines[0] = lines[0].replace("]", ", assumed from the rest of the filing]")
        shown = {hi}
        for h in hits[:2]:
            for j in range(max(0, h - neighbours), min(len(t["rows"]), h + neighbours + 1)):
                if j in shown or not (t["rows"][j][0] or t["rows"][j][1]):
                    continue
                shown.add(j)
                lab, vals = t["rows"][j]
                lines.append(_annotated(lab, vals, years))
        parts.append(NL.join(lines))
    return (NL + NL).join(parts), used


def code_extract(tables, metric, end_date, year_hint=None, latest_end=None):
    """Deterministic answer, no model. Prefers tables with a parseable year header (column mapped by date offset);
    otherwise assumes newest-first columns. Returns (value_usd, note); note says 'header' or 'assumed'."""
    cands = _candidates(tables, metric, year_hint or end_date[:4])[:3]
    for allow_assumed in (False, True):
        for t, hits in cands:
            if not t["factor"]:
                continue
            hi, years = header_years(t)
            if hi is not None:
                col = target_col(years, end_date, latest_end)
                need = len(years)
                basis = "header"
            elif allow_assumed and latest_end:
                col = assumed_col(end_date, latest_end)
                need = (col or 0) + 1
                basis = "assumed"
            else:
                continue
            if col is None or (basis == "assumed") != allow_assumed:
                continue
            for h in hits:
                lab, vals = t["rows"][h]
                nums = [_num(v) for v in vals]
                if len(vals) >= need and col < len(nums) and nums[col] is not None:
                    return nums[col] * t["factor"], f"{basis}: table {t['idx']} col {col}"
    return None, "no table with mappable column"


def number_present(context_tables, truth_usd, tol=0.0005):
    """Is the ground-truth figure visible in the supplied tables (after unit scaling)? Separates 'model could not
    read it' from 'we never gave it the figure'."""
    for t in context_tables:
        f = t["factor"]
        if not f:
            continue
        want = abs(truth_usd) / f
        for lab, vals in t["rows"]:
            for v in vals:
                n = _num(v)
                if n is not None and abs(abs(n) - want) <= max(1.0, tol * want):
                    return True
    return False
