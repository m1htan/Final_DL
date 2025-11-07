from typing import List, Dict
from langchain_text_splitters import TokenTextSplitter

# Cấu hình token-aware: 1200 tokens + overlap 240 theo yêu cầu
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 240

def chunk_text_records(records: List[Dict],
                       chunk_size: int = DEFAULT_CHUNK_SIZE,
                       overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    """
    Input: list bản ghi có 'TEXT' + metadata
    Output: list chunk dict: { 'chunk_text', 'metadata' }
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks: List[Dict] = []
    for rec in records:
        raw = rec.get("TEXT", "")
        if not raw.strip():
            continue
        parts = splitter.split_text(raw)

        base_meta = {
            "title": rec.get("TITLE"),
            "authors": rec.get("AUTHORS"),
            "conference": rec.get("CONFERENCE"),
            "abbr": rec.get("ABBR"),
            "source_pdf": rec.get("PDF_PATH"),
            "paper_url": (rec.get("LINKS") or {}).get("PAPER"),
        }
        for i, part in enumerate(parts):
            meta = dict(base_meta)
            meta["chunk_id"] = i
            chunks.append({"chunk_text": part, "metadata": meta})
    return chunks
