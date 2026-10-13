from dataclasses import dataclass, field
import glob, os

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

def load_documents(folder):
    docs={}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs

def chunk_text(doc_id, text, chunk_size=60, overlap=15):
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks, start, idx = [],0 ,0
    while start < len(words):
        window = words[start:start + chunk_size]
        chunks.append(Chunk(f"{doc_id}::chunk{idx}", doc_id, " ".join(window)))
        idx += 1
        start += step
    return chunks

def build_corpus(folder, chunk_size=60, overlap=15):
    docs = load_documents(folder)
    all_chunks = []
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_text(doc_id, text, chunk_size, overlap))
    return all_chunks