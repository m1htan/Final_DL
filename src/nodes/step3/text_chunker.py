from typing import List, Dict
import re

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[Dict]:
    """Chia text thành các đoạn có độ dài khoảng max_chars (có overlap nhẹ)."""
    clean_text = re.sub(r"\s+", " ", text.strip())
    chunks = []
    start = 0
    i = 0
    while start < len(clean_text):
        end = start + max_chars
        chunk = clean_text[start:end]
        chunks.append({
            "chunk_id": i,
            "text": chunk
        })
        start = end - overlap
        i += 1
    return chunks
