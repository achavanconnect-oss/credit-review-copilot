"""Ingest filings (HTML) into section-aware chunks and a numpy vector index."""
import re, json, os, numpy as np
from bs4 import BeautifulSoup
import llm

ITEM_RE = re.compile(r"^\s*item\s+(\d+[ab]?)\s*[\.\-–—:]?\s*(.{0,80})$", re.I)

def html_to_sections(path: str) -> list[dict]:
    soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    lines = [re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").split("\n")]
    lines = [l for l in lines if l]
    sections, cur = [], {"section": "Preamble", "text": []}
    for l in lines:
        m = ITEM_RE.match(l)
        if m and len(l) < 90:
            if cur["text"]:
                sections.append(cur)
            cur = {"section": f"Item {m.group(1).upper()} {m.group(2).strip()}", "text": []}
        else:
            cur["text"].append(l)
    sections.append(cur)
    return [{"section": s["section"], "text": "\n".join(s["text"])} for s in sections]

def chunk(sections: list[dict], doc: str, size: int = 1800, overlap: int = 200) -> list[dict]:
    out = []
    for s in sections:
        t = s["text"]
        for i in range(0, max(len(t), 1), size - overlap):
            piece = t[i:i + size]
            if len(piece) > 200:
                out.append({"doc": doc, "section": s["section"], "text": piece})
    return out

def build_index(chunks: list[dict], path: str):
    vecs = np.array(llm.embed([c["text"] for c in chunks]), dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path + ".npy", vecs)
    json.dump(chunks, open(path + ".json", "w"))

def load_index(path: str):
    return np.load(path + ".npy"), json.load(open(path + ".json"))

def search(index, query: str, k: int = 6, section_filter: str | None = None):
    vecs, chunks = index
    q = np.array(llm.embed([query])[0], dtype=np.float32)
    sims = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(-sims)
    res = []
    for i in order:
        if section_filter and section_filter.lower() not in chunks[i]["section"].lower():
            continue
        res.append(chunks[i])
        if len(res) == k:
            break
    return res
