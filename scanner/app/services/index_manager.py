import numpy as np
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.models import Chunk as ChunkRow
from app.services.chunking import Chunk, chunk_text
from app.services.dense_retriever import DenseRetriever, SentenceTransformerEmbedder, PgDenseRetriever
from app.services.hybrid_retriever import HybridRetriever
from app.services.sparse_retriever import SparseRetriever

_embedder = None
_hybrid = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder

def rebuild(db: Session) -> None:
    """Rebuild the in-memory search index from the chunks table."""
    global _hybrid
    rows = db.execute(
        select(ChunkRow.chunk_id, ChunkRow.document_id, ChunkRow.text).order_by(ChunkRow.id)
    ).all()
    if not rows:
        _hybrid = None
        return
    chunks = [Chunk(r.chunk_id, str(r.document_id), r.text) for r in rows]
    sparse = SparseRetriever(chunks)
    dense = PgDenseRetriever({c.chunk_id: c for c in chunks}, get_embedder())
    _hybrid = HybridRetriever(sparse, dense)

def ingest(db: Session, doc) -> int:
    """Chunk + embed a document's text, save to DB, refresh the index."""
    chunks = chunk_text(str(doc.id), doc.extracted_text, chunk_size=300, overlap=50)
    if chunks:
        vectors = get_embedder().encode([c.text for c in chunks]).astype(np.float32)
        for c, v in zip(chunks, vectors):
            db.add(ChunkRow(document_id=doc.id, chunk_id=c.chunk_id,
                            text=c.text, embedding=v))
        db.commit()
    rebuild(db)
    return len(chunks)

def remove(db: Session, doc_id: int) -> None:
    db.execute(delete(ChunkRow).where(ChunkRow.document_id == doc_id))
    db.commit()

def search(query: str, top_k: int = 5, doc_id: int | None = None):
    if _hybrid is None:
        return []
    return _hybrid.search(query, top_k=top_k)