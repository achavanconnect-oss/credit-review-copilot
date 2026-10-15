import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
DIM = 64


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedding: similar wording gives a small cosine distance."""
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        out.append(v)
    return out


CHUNKS = [
    {"section": "Item 1A Risk Factors", "text": "Our substantial indebtedness could adversely affect our financial condition and our ability to fulfil obligations under our credit facilities."},
    {"section": "Item 1A Risk Factors", "text": "Supply chain disruptions and commodity price volatility may reduce our gross margins and operating results."},
    {"section": "Item 7 Liquidity", "text": "We were in compliance with all financial covenants under the revolving credit facility as of year end, with liquidity of $5 billion."},
    {"section": "Item 7 Liquidity", "text": "Debt maturities over the next five years total $12 billion, with the largest maturity in the third year."},
]


@pytest.fixture
def embed():
    return fake_embed


@pytest.fixture
def chunks():
    return CHUNKS


@pytest.fixture(params=["memory", "pgvector"])
def store(request):
    from vector_store import MemoryStore, PgVectorStore
    if request.param == "memory":
        yield MemoryStore()
        return
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL not set (pgvector tests run in CI against a pgvector Postgres service)")
    prefix = "t" + re.sub(r"\W", "", request.node.name.split("[")[0])[-10:].lower()
    PgVectorStore(dsn, dim=DIM, prefix=prefix).drop()
    s = PgVectorStore(dsn, dim=DIM, prefix=prefix)
    yield s
    s.drop()
