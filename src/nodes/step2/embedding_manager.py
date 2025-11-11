import os
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DIR, EMBEDDING_MODEL

CHROMA_DIR = CHROMA_DIR
EMBED_MODEL_NAME = EMBEDDING_MODEL

# Cache embedding model (CPU-only)
_EMB = None
def get_embedder():
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMB

def upsert_chunks_into_chroma(chunks: List[Dict], batch_size: int = 16) -> int:
    """
    Ghi batch vào Chroma (tạo nếu chưa có). Trả về tổng số chunk đã ghi.
    """
    if not chunks:
        return 0

    embeddings = get_embedder()
    store = Chroma(
        collection_name="instruct2ds_all",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Upsert theo batch
    total = 0
    texts, metas, ids = [], [], []
    for i, ch in enumerate(tqdm(chunks, desc="Lưu vào Chroma", unit="chunk")):
        text = ch["chunk_text"]
        meta = ch["metadata"]
        # id ổn định: từ source_pdf + chunk_id
        cid = f'{meta.get("source_pdf","")}|{meta.get("chunk_id","0")}'
        texts.append(text)
        metas.append(meta)
        ids.append(cid)

        if len(texts) >= batch_size:
            store.add_texts(texts=texts, metadatas=metas, ids=ids)
            total += len(texts)
            texts, metas, ids = [], [], []

    if texts:
        store.add_texts(texts=texts, metadatas=metas, ids=ids)
        total += len(texts)

    store.persist()
    return total
