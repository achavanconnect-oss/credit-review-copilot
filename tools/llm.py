"""Provider-swappable LLM + embedding client. Configure via .env (LLM_PROVIDER, LLM_MODEL)."""
import os, json, requests
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")


def chat(system: str, user: str, schema: dict | None = None, temperature: float = 0.0) -> str:
    """Return model text. If schema (JSON Schema) is given, output is JSON conforming to it."""
    if PROVIDER == "ollama":
        body = {"model": MODEL, "stream": False, "options": {"temperature": temperature, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema:
            body["format"] = schema
        r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=900)
        r.raise_for_status()
        return r.json()["message"]["content"]
    if PROVIDER == "anthropic":
        import anthropic
        if schema:
            system += "\nRespond with ONLY valid JSON matching this schema:\n" + json.dumps(schema)
        msg = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=4096, temperature=temperature, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    raise ValueError(f"unknown LLM_PROVIDER {PROVIDER}")


def chat_json(system: str, user: str, schema: dict) -> dict:
    return json.loads(chat(system, user, schema))


def embed(texts: list[str], batch: int = 16) -> list[list[float]]:
    """Embeddings always run on Ollama (Anthropic has no embedding endpoint). Batched for speed."""
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts[i:i + batch]}, timeout=600)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return out
