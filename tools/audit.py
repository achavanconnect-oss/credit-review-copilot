"""Append-only SQLite audit log: every LLM call, output and human decision."""
import sqlite3, json, time, os

DB = os.path.join(os.path.dirname(__file__), "..", ".tmp", "audit.db")


def log(case: str, event: str, payload: dict):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute("CREATE TABLE IF NOT EXISTS audit(ts REAL, case_id TEXT, event TEXT, payload TEXT)")
        c.execute("INSERT INTO audit VALUES (?,?,?,?)", (time.time(), case, event, json.dumps(payload)))
