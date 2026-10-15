"""Download the latest 10-K (primary HTML) for a list of tickers from SEC EDGAR.
Usage: fetch_10k.py TICKER [TICKER ...]   -> .tmp/filings/<TICKER>.htm and .tmp/filings/<TICKER>.json (cik, fiscal info)
SEC requires an identifying User-Agent: set SEC_USER_AGENT in .env (name + contact email).
"""
import sys, os, json, time, requests
from dotenv import load_dotenv

load_dotenv()
UA = os.getenv("SEC_USER_AGENT", "CreditCopilotPilot research-pilot")
H = {"User-Agent": UA}


def get(url, **kw):
    time.sleep(0.4)  # stay well under SEC's 10 requests/second limit
    r = requests.get(url, headers=H, timeout=120, **kw)
    r.raise_for_status()
    return r


def main(tickers):
    os.makedirs(".tmp/filings", exist_ok=True)
    tick = get("https://www.sec.gov/files/company_tickers.json").json()
    by = {v["ticker"].upper(): v for v in tick.values()}
    for t in tickers:
        t = t.upper()
        if t not in by:
            print(t, "not found")
            continue
        cik = f"{by[t]['cik_str']:010d}"
        sub = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()["filings"]["recent"]
        for i, f in enumerate(sub["form"]):
            if f == "10-K":
                acc = sub["accessionNumber"][i].replace("-", "")
                doc = sub["primaryDocument"][i]
                html = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}").text
                open(f".tmp/filings/{t}.htm", "w", encoding="utf-8").write(html)
                json.dump({"ticker": t, "cik": cik, "name": by[t]["title"], "report_date": sub["reportDate"][i],
                           "filed": sub["filingDate"][i]}, open(f".tmp/filings/{t}.json", "w"))
                print(t, cik, sub["reportDate"][i], f"{len(html) / 1e6:.1f}MB")
                break


if __name__ == "__main__":
    main(sys.argv[1:])
