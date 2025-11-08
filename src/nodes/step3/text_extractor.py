import fitz
from pathlib import Path
import time

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "papers_raw"
TEXT_DIR = DATA_DIR / "papers_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Trích toàn bộ text từ 1 file PDF bằng PyMuPDF."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text.strip()
    except Exception as e:
        print(f"[WARN] Không đọc được {pdf_path.name}: {e}")
        return ""

def extract_all_pdfs():
    """Trích text từ tất cả PDF trong data/papers_raw/."""
    t0 = time.time()
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu trích xuất {len(pdf_files)} PDF...")

    for pdf in pdf_files:
        txt_name = pdf.stem + ".txt"
        txt_path = TEXT_DIR / txt_name
        if txt_path.exists():
            continue  # bỏ qua file đã có
        text = extract_text_from_pdf(pdf)
        if text:
            txt_path.write_text(text, encoding="utf-8")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hoàn thành trích xuất. Thời gian: {time.time()-t0:.1f}s")
