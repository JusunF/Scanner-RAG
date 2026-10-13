from collections import defaultdict

class HybridRetriever:
    def __init__(self, sparse_retriever, dense_retriever):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.chunks_by_id = {c.chunk_id: c for c in sparse_retriever.chunks}

    def search(self, query, top_k=5, candidate_k=20, rrf_k=60):
        sparse_results = self.sparse.search(query, top_k=candidate_k) # BM25 ranking
        dense_results = self.dense.search(query, top_k=candidate_k) # embedding ranking

        fused = defaultdict(float)
        for rank, (chunk, _) in enumerate(sparse_results):
            fused[chunk.chunk_id] += 1.0 / (rrf_k + rank + 1)
        for rank, (chunk, _) in enumerate(dense_results):
            fused[chunk.chunk_id] += 1.0 / (rrf_k + rank + 1)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks_by_id[cid], score) for cid, score in ranked]