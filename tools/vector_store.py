"""Vector storage for filing chunks and verified-finding memory.

Two interchangeable stores with the same interface:
  - PgVectorStore: PostgreSQL + pgvector (cosine distance, HNSW index). Persistent, shared, filterable by SQL.
  - MemoryStore:   NumPy, in-process. For tests and for running without a database.

Search results are dicts: {"doc", "section", "text", "distance"} where distance is cosine distance (0 = identical).
"""
import json
import numpy as np


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return a / n if n else a


class MemoryStore:
    def __init__(self):
        self.chunks: list[dict] = []
        self.vecs: list[np.ndarray] = []
        self.memory: list[dict] = []

    def count(self, doc: str) -> int:
        return sum(1 for c in self.chunks if c["doc"] == doc)

    def add_chunks(self, doc: str, chunks: list[dict], vectors) -> None:
        for c, v in zip(chunks, vectors):
            self.chunks.append({"doc": doc, "section": c["section"], "text": c["text"]})
            self.vecs.append(_unit(v))

    def search(self, doc: str, qvec, k: int = 6, section: str | None = None) -> list[dict]:
        q = _unit(qvec)
        scored = []
        for c, v in zip(self.chunks, self.vecs):
            if c["doc"] != doc or (section and section.lower() not in c["section"].lower()):
                continue
            scored.append({**c, "distance": float(1.0 - v @ q)})
        return sorted(scored, key=lambda r: r["distance"])[:k]

    def add_memory(self, doc: str, question: str, answer: dict, qvec) -> None:
        self.memory.append({"doc": doc, "question": question, "answer": answer, "vec": _unit(qvec)})

    def recall(self, doc: str, qvec, max_distance: float = 0.1) -> dict | None:
        q = _unit(qvec)
        best = None
        for m in self.memory:
            if m["doc"] != doc:
                continue
            d = float(1.0 - m["vec"] @ q)
            if d <= max_distance and (best is None or d < best["distance"]):
                best = {"question": m["question"], "answer": m["answer"], "distance": d}
        return best


class PgVectorStore:
    """Needs PostgreSQL with the pgvector extension (e.g. the pgvector/pgvector Docker image)."""

    def __init__(self, dsn: str, dim: int = 768, prefix: str = "cra"):
        import psycopg
        from pgvector.psycopg import register_vector
        self.dim, self.p = dim, prefix
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(f"""CREATE TABLE IF NOT EXISTS {prefix}_chunks (
            id bigserial PRIMARY KEY, doc text NOT NULL, section text NOT NULL, text text NOT NULL,
            embedding vector({dim}) NOT NULL)""")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {prefix}_chunks_doc ON {prefix}_chunks (doc)")
        self.conn.execute(f"""CREATE INDEX IF NOT EXISTS {prefix}_chunks_hnsw
            ON {prefix}_chunks USING hnsw (embedding vector_cosine_ops)""")
        self.conn.execute(f"""CREATE TABLE IF NOT EXISTS {prefix}_memory (
            id bigserial PRIMARY KEY, doc text NOT NULL, question text NOT NULL, answer jsonb NOT NULL,
            embedding vector({dim}) NOT NULL, created_at timestamptz DEFAULT now())""")

    def drop(self) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {self.p}_chunks, {self.p}_memory")

    def count(self, doc: str) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.p}_chunks WHERE doc = %s", (doc,)).fetchone()[0]

    def add_chunks(self, doc: str, chunks: list[dict], vectors) -> None:
        rows = [(doc, c["section"], c["text"], np.asarray(v, dtype=np.float32)) for c, v in zip(chunks, vectors)]
        with self.conn.cursor() as cur:
            cur.executemany(f"INSERT INTO {self.p}_chunks (doc, section, text, embedding) VALUES (%s, %s, %s, %s)", rows)

    def search(self, doc: str, qvec, k: int = 6, section: str | None = None) -> list[dict]:
        q = np.asarray(qvec, dtype=np.float32)
        pat = f"%{section}%" if section else None
        rows = self.conn.execute(
            f"""SELECT section, text, embedding <=> %s AS distance FROM {self.p}_chunks
                WHERE doc = %s AND (%s::text IS NULL OR section ILIKE %s)
                ORDER BY embedding <=> %s LIMIT %s""", (q, doc, pat, pat, q, k)).fetchall()
        return [{"doc": doc, "section": s, "text": t, "distance": float(d)} for s, t, d in rows]

    def add_memory(self, doc: str, question: str, answer: dict, qvec) -> None:
        from psycopg.types.json import Jsonb
        self.conn.execute(f"INSERT INTO {self.p}_memory (doc, question, answer, embedding) VALUES (%s, %s, %s, %s)",
                          (doc, question, Jsonb(answer), np.asarray(qvec, dtype=np.float32)))

    def recall(self, doc: str, qvec, max_distance: float = 0.1) -> dict | None:
        q = np.asarray(qvec, dtype=np.float32)
        row = self.conn.execute(
            f"""SELECT question, answer, embedding <=> %s AS distance FROM {self.p}_memory
                WHERE doc = %s ORDER BY embedding <=> %s LIMIT 1""", (q, doc, q)).fetchone()
        if row and row[2] <= max_distance:
            ans = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            return {"question": row[0], "answer": ans, "distance": float(row[2])}
        return None
