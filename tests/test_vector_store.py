def load(store, embed, chunks, doc="acme"):
    store.add_chunks(doc, chunks, embed([c["text"] for c in chunks]))


def test_count_and_nearest_chunk(store, embed, chunks):
    load(store, embed, chunks)
    assert store.count("acme") == 4 and store.count("other") == 0
    r = store.search("acme", embed(["debt maturities in the next five years"])[0], k=2)
    assert "maturities" in r[0]["text"]
    assert r[0]["distance"] <= r[1]["distance"]


def test_section_filter(store, embed, chunks):
    load(store, embed, chunks)
    r = store.search("acme", embed(["debt"])[0], k=4, section="Item 1A")
    assert r and all("Item 1A" in x["section"] for x in r)


def test_docs_are_isolated(store, embed, chunks):
    load(store, embed, chunks, "acme")
    assert store.search("other", embed(["debt"])[0]) == []


def test_memory_recall_threshold(store, embed):
    q = embed(["what are the key risk factors"])[0]
    store.add_memory("acme", "what are the key risk factors", {"items": [{"claim": "c", "quote": "q" * 20}]}, q)
    hit = store.recall("acme", q, max_distance=0.1)
    assert hit and hit["answer"]["items"][0]["claim"] == "c" and hit["distance"] < 0.01
    far = embed(["completely different topic about dividends"])[0]
    assert store.recall("acme", far, max_distance=0.1) is None
    assert store.recall("other", q, max_distance=0.1) is None
