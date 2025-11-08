from sentence_transformers import SentenceTransformer
import numpy as np
import time

def get_model():
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", device="cpu")
    return model

def embed_chunks(model, chunks):
    """Sinh embedding cho danh sách các chunk."""
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    for i, vec in enumerate(embeddings):
        chunks[i]["embedding"] = vec.astype(np.float32).tolist()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Embedding {len(chunks)} chunks xong trong {time.time()-t0:.1f}s.")
    return chunks
