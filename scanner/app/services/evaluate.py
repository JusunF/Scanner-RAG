import sys, os
from app.services.chunking import build_corpus

from chunking import build_corpus
from sparse_retriever import SparseRetriever
from dense_retriever import DenseRetriever
from hybrid_retriever import HybridRetriever

TEST_SET = [
    ("Python programming", "doc1::chunk0"),
    ("writing code and building apps", "doc4::chunk0"),
    ("reptiles that live outdoors", "doc6::chunk0"),
    ("baking bread at home", "doc5::chunk0"),
    ("banks and loans", "doc7::chunk0"),
]

def recall_at_k(retriever, test_set, k=3):
    hits=0
    for query, expected_chunk_id in test_set:
        results = retriever.search(query, top_k=k)
        retrieved_ids = [chunk.chunk_id for chunk, _score in results]
        if expected_chunk_id in retrieved_ids:
            hits += 1
    return hits / len(test_set)

chunks = build_corpus("documents", chunk_size=40, overlap=10)
sparse = SparseRetriever(chunks)
dense = DenseRetriever(chunks)
hybrid = HybridRetriever(sparse, dense)

print("Sparse recall@3:", recall_at_k(sparse, TEST_SET))
print("Dense recall@3:", recall_at_k(dense, TEST_SET))
print("Hybrid recall@3:", recall_at_k(hybrid, TEST_SET))

def recall_at_k_verbose(name, retriever, test_set, k=3):
    print(f"\n--- {name} (k={k}) ---")
    hits = 0
    for query, expected_chunk_id in test_set:
        results = retriever.search(query, top_k=k)
        retrieved_ids = [chunk.chunk_id for chunk, _score in results]
        hit = expected_chunk_id in retrieved_ids
        hits += hit
        print(f"  {'HIT ' if hit else 'MISS'}  {query!r:45s} expected={expected_chunk_id}  got={retrieved_ids}")
    print(f"  recall@{k} = {hits/len(test_set):.2f}")
    return hits / len(test_set)

recall_at_k_verbose("Sparse", sparse, TEST_SET, k=1)
recall_at_k_verbose("Dense", dense, TEST_SET, k=1)
recall_at_k_verbose("Hybrid", hybrid, TEST_SET, k=1)