from pathlib import Path
from typing import Dict, List
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    p = Path(pdf_path)
    if not p.exists():
        return ""
    try:
        reader = PdfReader(str(p))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        text = "\n".join(texts)
        # làm sạch nhẹ
        text = text.replace("\u00ad", "")  # soft hyphen
        return text
    except Exception as e:
        print(f"[WARN] Không trích xuất được {pdf_path}: {e}")
        return ""

def parse_many_pdf(records: List[Dict]) -> List[Dict]:
    """
    Nhận danh sách metadata (đã có 'PDF_PATH'), trả về kèm 'TEXT' (nếu có).
    """
    out = []
    for rec in records:
        pdf_path = rec.get("PDF_PATH")
        if not pdf_path:
            continue
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue
        out.append({**rec, "TEXT": text})
    return out
