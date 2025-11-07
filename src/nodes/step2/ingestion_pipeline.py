from typing import Dict
from src.state import OrchestratorState
from src.nodes.step2.pdf_downloader import download_all_pdfs
from src.nodes.step2.pdf_parser import parse_many_pdf
from src.nodes.step2.text_chunker import chunk_text_records
from src.nodes.step2.embedding_manager import upsert_chunks_into_chroma
from src.nodes.step2.ohcache import load_cache, save_cache
from src.utils.logger import log

def ingestion_pipeline_node(state: Dict) -> Dict:
    trace = list(state.get("trace", []))
    log("=== BẮT ĐẦU INGESTION PIPELINE ===")

    trace.append("[ingest] Bắt đầu ingestion pipeline (tải PDF → parse → chunk → embed → Chroma).")
    log("Bắt đầu tải PDF...")

    # 1) Tải PDF
    recs = download_all_pdfs(skip_existing=True, limit=None)
    trace.append(f"[ingest] Đã có/tải được {len(recs)} bản ghi có đường dẫn PDF.")
    log(f"Đã có/tải được {len(recs)} bản ghi có đường dẫn PDF.")

    # 2) Trích text
    log("Bắt đầu trích xuất văn bản từ PDF...")
    parsed = parse_many_pdf(recs)
    trace.append(f"[ingest] Trích văn bản thành công {len(parsed)}/ {len(recs)} PDF.")
    log(f"Trích văn bản thành công {len(parsed)}/{len(recs)} PDF.")

    if not parsed:
        resp = "Không trích xuất được PDF nào. Kiểm tra lại metadata hoặc mạng."
        log(resp)
        return {**state, "response": resp, "trace": trace}

    # 3) Chunk theo token
    log("Bắt đầu chia văn bản thành các đoạn...")
    chunks = chunk_text_records(parsed, chunk_size=1200, overlap=240)
    trace.append(f"[ingest] Sinh được {len(chunks)} chunks từ {len(parsed)} PDF.")
    log(f"Sinh được {len(chunks)} chunks từ {len(parsed)} PDF.")

    # 4) Embedding + Chroma
    log("Bắt đầu tính embedding và ghi vào Chroma...")
    inserted = upsert_chunks_into_chroma(chunks, batch_size=16)
    trace.append(f"[ingest] Đã ghi {inserted} chunks vào ChromaDB (db/chroma_instruct2ds).")
    log(f"Đã ghi {inserted} chunks vào ChromaDB (db/chroma_instruct2ds).")

    # 5) OHCache mock
    log("Cập nhật OHCache...")
    cache = load_cache()
    cache.setdefault("ingestion_runs", 0)
    cache["ingestion_runs"] += 1
    cache["last_stats"] = {
        "records_with_pdf": len(recs),
        "parsed_pdfs": len(parsed),
        "chunks_inserted": inserted
    }
    save_cache(cache)
    trace.append("[ingest] OHCache cập nhật (db/ohcache.json).")
    log("Cập nhật OHCache hoàn tất.")

    resp = (
        "Ingestion hoàn tất: "
        f"{len(parsed)} PDF → {inserted} chunks đã lưu vào ChromaDB. "
        "Sẵn sàng cho Bước 3 (Retriever → LLM → Post-processing)."
    )
    log("=== KẾT THÚC INGESTION PIPELINE ===")
    return {**state, "response": resp, "trace": trace}
