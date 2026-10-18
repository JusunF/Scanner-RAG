"""
dense_retriever.py
--------------------
Semantic retrieval using real neural embeddings (sentence-transformers) +
cosine similarity.

Where BM25 matches words, dense retrieval matches *meaning*. Text is mapped
into a vector space such that semantically similar text ends up close
together, even with zero word overlap.

SentenceTransformerEmbedder uses a small pretrained neural model
(all-MiniLM-L6-v2) trained on hundreds of millions of sentence pairs — it
understands context, so "Python the language" and "python the snake" land
in genuinely different parts of the vector space, unlike plain TF-IDF.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes cosine similarity well-behaved
        # and avoids zero-vector edge cases we ran into with LSA
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class LocalLSAEmbedder:
    """Offline fallback (no internet/model download needed) — see earlier version."""

    def __init__(self, n_components: int = 50):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd = None

    def fit(self, texts: list[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        return self.svd.fit_transform(tfidf)

    def encode(self, texts: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf)


def get_default_embedder():
    """Try real embeddings first; fall back to LSA if sentence-transformers isn't available."""
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        print("sentence-transformers not installed — falling back to TF-IDF+SVD.")
        return LocalLSAEmbedder()


class DenseRetriever:
    def __init__(self, chunks: list, embedder=None, embeddings=None):
        """
        chunks: list of chunking.Chunk objects
        embedder: optional embedder instance; defaults to get_default_embedder()
        embeddings: optional precomputed embeddings matrix — if given, skips
                    re-encoding the whole corpus (used when loading a saved index)
        """
        self.chunks = chunks
        self.embedder = embedder or get_default_embedder()
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = self.embedder.fit([c.text for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[tuple]:
        query_vec = self.embedder.encode([query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        ranked = sorted(zip(self.chunks, sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

class PgDenseRetriever:
    def __init__(self, chunks_by_id, embedder):
        self.chunks_by_id = chunks_by_id
        self.embedder = embedder

    def search(self, query, top_k=5, doc_id=None):
        from sqlalchemy import select
        from app.database import SessionLocal
        from app.models import Chunk as ChunkRow

        qv = self.embedder.encode([query])[0].astype("float32")
        dist = ChunkRow.embedding.cosine_distance(qv)
        stmt = select(ChunkRow.chunk_id, dist).order_by(dist).limit(top_k)
        if doc_id is not None:
            stmt = stmt.where(ChunkRow.document_id == doc_id)
        with SessionLocal() as db:
            rows = db.execute(stmt).all()
        return [(self.chunks_by_id[cid], 1 - d) for cid, d in rows]


if __name__ == "__main__":
    import os
    from chunking import build_corpus

    chunks = build_corpus(os.path.join(os.path.dirname(__file__), "..", "documents"))
    retriever = DenseRetriever(chunks)

    for query in ["programming language for data science", "large snake that squeezes its prey"]:
        print(f"\nQuery: {query!r}")
        for chunk, score in retriever.search(query, top_k=3):
            print(f"  {score:.3f}  [{chunk.chunk_id}]  {chunk.text[:70]}...")