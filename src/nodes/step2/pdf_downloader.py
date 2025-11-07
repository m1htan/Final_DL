import os
import json
import re
from pathlib import Path
from typing import List, Dict, Iterable
import urllib.request
from tqdm import tqdm
import ast
import requests
import time

DATA_DIR = Path("data")
RAW_PDF_DIR = DATA_DIR / "papers_raw"
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:180]

def iter_metadata_files() -> Iterable[Path]:
    # Duyệt tất cả file JSON metadata trong data/*/
    for p in (p for p in DATA_DIR.iterdir() if p.is_dir()):
        for f in sorted(p.glob("paper_metadata_*.json")):
            yield f

def _load_json_lines(path):
    """Đọc file JSON hoặc JSON-lines, tự động chuyển dict nhiều khóa thành list record."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        # Nếu là JSON hợp lệ dạng dict {key: {...}}, chuyển thành list
        data = json.loads(text)
        if isinstance(data, dict):
            # Chuyển mỗi value thành record, đồng thời lưu lại key nếu cần
            records = []
            for key, val in data.items():
                if isinstance(val, dict):
                    val["_id"] = key
                    records.append(val)
            return records

        # Nếu là list JSON
        if isinstance(data, list):
            return data

        # Nếu là JSON lines (mỗi dòng một record)
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [rec for rec in lines if isinstance(rec, dict)]

    except Exception as e:
        print(f"[WARN] Không đọc được {path.name}: {e}")
        return []

def build_pdf_filename(meta: Dict) -> str:
    title = meta.get("TITLE") or "untitled"
    abbr = meta.get("ABBR") or "GEN"
    # Cố gắng khui năm từ CONFERENCE hoặc từ tên file metadata (fallback)
    conf = (meta.get("CONFERENCE") or "").strip()
    year = re.search(r"(19|20)\d{2}", conf)
    year = year.group(0) if year else "xxxx"
    base = f"{abbr}_{year}_{_slugify(title)}.pdf"
    return base

def download_all_pdfs(skip_existing=True, limit=None):
    """
    Tải hoặc xác minh toàn bộ PDF từ tất cả metadata trong thư mục data/.
    Có log tổng quát domain, đếm metadata toàn bộ, và log số PDF hiện có / mới tải.
    """
    base_dir = Path("data")
    output_dir = Path("data/papers_raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ingest_progress.log"

    def log(msg):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} {msg}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} {msg}\n")

    # Đếm PDF hiện có
    existing_pdfs = list(output_dir.glob("*.pdf"))
    existing_count = len(existing_pdfs)
    log(f"Hiện có {existing_count} PDF trong thư mục papers_raw (trước khi ingest).")

    # Quét toàn bộ metadata file
    all_jsons = list(base_dir.rglob("paper_metadata_*.json"))
    total_files = len(all_jsons)
    log(f"Đã phát hiện {total_files} metadata files trên toàn hệ thống.")

    total_records = 0
    total_downloaded = 0
    failed = 0

    # Gom theo domain
    domains = {}
    for json_path in all_jsons:
        domain = json_path.parent.name
        domains.setdefault(domain, []).append(json_path)

    log(f"Có {len(domains)} domain: {', '.join(domains.keys())}")

    for d_idx, (domain, paths) in enumerate(domains.items(), 1):
        log(f"=== [{d_idx}/{len(domains)}] BẮT ĐẦU DOMAIN: {domain} ===")

        for f_idx, json_path in enumerate(sorted(paths), 1):
            log(f"({f_idx}/{len(paths)}) Xử lý {json_path.name} trong domain {domain}...")
            records = _load_json_lines(json_path)
            if not records:
                log(f"[WARN] File {json_path.name} rỗng hoặc lỗi.")
                continue

            for rec in tqdm(records, desc=f"Tải PDF từ {json_path.name}", leave=False):
                link_field = rec.get("LINKS") or rec.get("LINK") or rec.get("URL") or rec.get("pdf_url") or rec.get("url")
                paper_url = ""
                if isinstance(link_field, dict):
                    for k, v in link_field.items():
                        if isinstance(v, str) and any(x in k.lower() for x in ["pdf", "paper", "url"]):
                            paper_url = v.strip()
                            break
                elif isinstance(link_field, str):
                    paper_url = link_field.strip()

                if not paper_url or not paper_url.startswith("http"):
                    continue

                fname = build_pdf_filename(rec)
                save_path = output_dir / fname

                if skip_existing and save_path.exists():
                    continue

                try:
                    r = requests.get(paper_url, timeout=15)
                    if r.status_code == 200:
                        save_path.write_bytes(r.content)
                        total_downloaded += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

                total_records += 1
                if limit and total_records >= limit:
                    log("Đã đạt giới hạn ingest, dừng sớm.")
                    new_total = len(list(output_dir.glob('*.pdf')))
                    log(f"Tổng số PDF hiện có sau khi ingest: {new_total} (mới tải thêm {new_total - existing_count})")
                    return total_records

        log(f"=== KẾT THÚC DOMAIN: {domain} ({len(paths)} metadata) ===")

    # Tổng kết sau khi ingest xong toàn bộ
    new_total = len(list(output_dir.glob("*.pdf")))
    log(f"Tổng số record quét: {total_records}")
    log(f"Tổng số PDF tải mới: {total_downloaded}")
    log(f"Tổng số lỗi tải: {failed}")
    log(f"Tổng số PDF hiện có sau khi ingest: {new_total} (mới tải thêm {new_total - existing_count})")
    log("=== HOÀN THÀNH TOÀN BỘ INGESTION PDF ===")

    return total_records