"""Evaluate the year-on-year flagging against an independent lexical baseline, and emit an analyst labelling sheet.

Baseline (no AI, no embeddings): a sentence is NEW if <30% of its 6-word shingles occur anywhere in the other
year's Item 1A. A chunk is 'lexically changed' if >=20% of its words sit in NEW sentences.
This is NOT human ground truth. It is an independent cross-check; the CSV is for an analyst to label.

Usage: eval_yoy.py <prior_doc> <current_doc> [threshold]   (docs are index names under .tmp/index)
"""
import sys, re, csv, json, numpy as np

NL = chr(10)
W = re.compile(r"[a-z0-9']+")


def load(name, key="1A"):
    v = np.load(f".tmp/index/{name}.npy")
    c = json.load(open(f".tmp/index/{name}.json"))
    ix = [i for i, x in enumerate(c) if key in x["section"]]
    w = v[ix] / np.linalg.norm(v[ix], axis=1, keepdims=True)
    return w, [c[i] for i in ix]


def shingles(text, n=6):
    t = W.findall(text.lower())
    return {" ".join(t[i:i + n]) for i in range(max(len(t) - n + 1, 1))}


def new_fraction(chunk_text, other_shingles):
    sents = [s for s in re.split(r"(?<=[.;:])\s+", chunk_text) if len(W.findall(s)) >= 8]
    tot = new = 0
    ex = []
    for s in sents:
        sh = shingles(s)
        n = len(W.findall(s))
        tot += n
        if len(sh & other_shingles) / max(len(sh), 1) < 0.30:
            new += n
            ex.append(s)
    return (new / tot if tot else 0.0), ex


def main(prior, current, thr=0.85):
    pv, pc = load(prior)
    cv, cc = load(current)
    sim = cv @ pv.T
    all_p = set().union(*[shingles(c["text"]) for c in pc])
    all_c = set().union(*[shingles(c["text"]) for c in cc])

    rows, tp = [], 0
    for side, chunks, best, other in (("current", cc, sim.max(1), all_p), ("prior", pc, sim.max(0), all_c)):
        for i, c in enumerate(chunks):
            frac, ex = new_fraction(c["text"], other)
            rows.append({"side": side, "chunk": i, "embed_flag": best[i] < thr, "best_sim": round(float(best[i]), 3),
                         "lex_new_frac": round(frac, 2), "lex_changed": frac >= 0.20,
                         "example_new_sentence": (ex[0][:240] if ex else ""), "analyst_label_material(y/n)": ""})
    for side in ("current", "prior"):
        R = [r for r in rows if r["side"] == side]
        e = {r["chunk"] for r in R if r["embed_flag"]}
        l = {r["chunk"] for r in R if r["lex_changed"]}
        both = len(e & l)
        print(f"{side}: chunks={len(R)} embed_flagged={len(e)} lexical_changed={len(l)} agree={both} "
              f"precision_vs_lexical={both / max(len(e), 1):.2f} recall_vs_lexical={both / max(len(l), 1):.2f}")
    with open(".tmp/yoy_eval_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # threshold sweep
    print("threshold sweep (current side): thr, flagged, precision, recall vs lexical")
    R = [r for r in rows if r["side"] == "current"]
    l = {r["chunk"] for r in R if r["lex_changed"]}
    for t in (0.80, 0.83, 0.85, 0.87, 0.90, 0.92):
        e = {r["chunk"] for r in R if r["best_sim"] < t}
        b = len(e & l)
        print(f"  {t:.2f} {len(e):3d} {b / max(len(e), 1):.2f} {b / max(len(l), 1):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.85)
