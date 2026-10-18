from rank_bm25 import BM25Okapi
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "at", "for", "with", "and",
    "or", "but", "what", "good", "this", "that", "it", "as", "by", "from",
}

def tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in text.split() if w not in STOPWORDS]

class SparseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5, min_score=0.01, doc_id=None):
        scores = self.bm25.get_scores(tokenize(query))
        pairs = list(zip(self.chunks, scores))
        if doc_id is not None:
            pairs = [(c, s) for c, s in pairs if c.doc_id == str(doc_id)]
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
        ranked = [(chunk, score) for chunk, score in ranked if score > min_score]
        return ranked[:top_k]