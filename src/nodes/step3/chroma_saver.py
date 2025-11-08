from chromadb import PersistentClient
from pathlib import Path
import time

def save_to_chroma(chunks, collection_name="instruct2ds", persist_dir="db/chroma_instruct2ds"):
    """Lưu toàn bộ chunk + embedding vào ChromaDB."""
    client = PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(collection_name)

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for c in chunks:
        ids.append(f"{c.get('source','')}_{c['chunk_id']}")
        embeddings.append(c["embedding"])
        metadatas.append({"source": c.get("source", ""), "chunk_id": c["chunk_id"]})
        documents.append(c["text"])

    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Đã lưu {len(chunks)} chunks vào Chroma ({collection_name}).")
