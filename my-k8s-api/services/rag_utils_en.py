# services/rag_utils_en.py
"""
FAISS retrieval helper for the English corpus.
- uses sentence‑transformers for embeddings
- returns top‑k relevant docs for a query
"""

from __future__ import annotations
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer

from .rag_corpus import CORPUS_DOCS

# ── 1. build / load embedding model ───────────────────────────────────
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
_EMBS     = _EMBEDDER.encode(CORPUS_DOCS, normalize_embeddings=True)

_DIM   = _EMBS.shape[1]
_INDEX = faiss.IndexFlatIP(_DIM)     # cosine similarity (with normalized vecs = inner‑product)
_INDEX.add(np.asarray(_EMBS, dtype="float32"))

# ── 2. retrieval function ─────────────────────────────────────────────
def rag_retrieve(query: str, top_k: int = 3) -> str:
    """
    Return the most relevant corpus snippets joined by double newlines.
    """
    q_vec = _EMBEDDER.encode([query], normalize_embeddings=True)
    D, I = _INDEX.search(np.asarray(q_vec, dtype="float32"), top_k)
    hits = [CORPUS_DOCS[i] for i in I[0] if i != -1]
    return "\n\n".join(hits)

# ── 3. lightweight semantic guard / filter ────────────────────────────
_ALLOWED_RES   = r"\b(pod|namespace)\b"
_ALLOWED_METRIC= r"\b(cpu|memory|disk|network)\b"
_VALID_RANGE   = r"\b(\d+)(m|h|d|mo)\b"

def is_supported(query: str) -> bool:
    """
    Very simple rule‑based filter:
    – must mention pod or namespace
    – must mention one supported metric
    – must include a legal time token (30m / 1h / …) somewhere
    """
    if not re.search(_ALLOWED_RES,   query, re.I): return False
    if not re.search(_ALLOWED_METRIC,query, re.I): return False
    if not re.search(_VALID_RANGE,   query, re.I): return False
    return True
