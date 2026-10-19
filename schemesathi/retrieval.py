"""Retrieval (RAG): embeddings -> vector store -> cosine top-k -> keyword re-rank.

Embedding backends (the active one is exposed as MODE so the UI can label it, never hidden):
  * sentence-transformers  - real semantic embeddings, used only if SCHEMESATHI_EMBED=st and the package is installed
  * hashed bag-of-words    - MOCK embeddings: IDF-weighted tokens hashed into a fixed-size vector. Lexical, not semantic.
"""
import hashlib, json, math, os, re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
STOP = set("the a an of to in is are and or for be must by on at as it that this with from".split())
DIM = 512
_ST = None
if os.environ.get("SCHEMESATHI_EMBED") == "st":
    try:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _ST = None
MODE = "sentence-transformers embeddings" if _ST else "hashed-vector store (mock embeddings, lexical)"


def load_schemes():
    """Scheme records. Chunks produced by ingest.py (PyMuPDF) override the hand-written ones for that scheme."""
    schemes = json.loads((DATA / "schemes.json").read_text())
    f = DATA / "ingested_chunks.json"
    if f.exists():
        ing = json.loads(f.read_text())
        for s in schemes:
            if s["id"] in ing:
                s["chunks"] = {**s["chunks"], **ing[s["id"]]["chunks"]}
                s["chunk_source"] = "PDF (PyMuPDF): " + ing[s["id"]]["source"]
    return schemes


def tokens(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


def embed(text, idf=None):
    """Text -> L2-normalised vector (list of floats)."""
    if _ST:
        v = _ST.encode(text, normalize_embeddings=True)
        return [float(x) for x in v]
    vec = [0.0] * DIM
    for w, c in Counter(tokens(text)).items():
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % DIM] += (1 + math.log(c)) * (idf.get(w, 1.0) if idf else 1.0) * (1 if (h >> 64) & 1 else -1)
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """Minimal in-memory vector store: add(chunk) then cosine search with optional scheme filter."""
    def __init__(self, idf=None):
        self.items, self.idf = [], idf

    def add(self, chunk):
        self.items.append({**chunk, "vec": embed(chunk["text"], self.idf)})

    def search(self, query, top_k, scheme_id=None):
        q = embed(query, self.idf)
        scored = [(_cos(q, it["vec"]), it) for it in self.items if scheme_id in (None, it["scheme_id"])]
        scored.sort(key=lambda x: -x[0])
        return [({k: it[k] for k in ("id", "scheme_id", "kind", "text")}, s) for s, it in scored[:top_k]]


def build_index(schemes):
    chunks = [{"id": f"{s['id']}-{k}", "scheme_id": s["id"], "kind": k, "text": t}
              for s in schemes for k, t in s["chunks"].items()]
    df = Counter(w for c in chunks for w in set(tokens(c["text"])))
    n = len(chunks)
    store = VectorStore({w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()})
    for c in chunks:
        store.add(c)
    return store


def rerank(query, hits, keys=None):
    """Second-stage keyword re-ranker: vector score + query-term overlap + bonus when the chunk contains a scheme key phrase."""
    q = set(tokens(query)); keys = [str(v).lower() for v in (keys or {}).values()]
    out = []
    for c, s in hits:
        toks = set(tokens(c["text"]))
        overlap = len(q & toks) / (len(q) or 1)
        bonus = 0.15 * sum(k in c["text"].lower() for k in keys) / (len(keys) or 1)
        out.append((s + 0.5 * overlap + bonus, c, s))
    out.sort(key=lambda x: -x[0])
    return [dict(c, score=round(final, 3), vector_score=round(s, 3)) for final, c, s in out]


def search(store, query, scheme_id, top_k, keys=None):
    """Retrieve a wider candidate set from the vector store, re-rank it, keep top_k."""
    cands = store.search(query, top_k=max(top_k * 2, 4), scheme_id=scheme_id)
    return rerank(query, cands, keys)[:top_k]


def search_all(store, query, top_k):
    """Search every scheme's chunks (used by the chat assistant)."""
    hits = rerank(query, store.search(query, top_k * 2))
    return [h for h in hits if h["vector_score"] > 0][:top_k]
