import time
from pathlib import Path
from typing import List
from src.utils.logger import log
from src.llm import make_llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langdetect import detect
from deep_translator import GoogleTranslator

from src.config import CHROMA_DIR, EMBEDDING_MODEL
from src.nodes.step4.users_query import users_query

DB_DIR = Path(CHROMA_DIR)
DB_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "instruct2ds"
QUERY_LOG = Path("logs/query.log")
QUERY_LOG.parent.mkdir(exist_ok=True)

def qlog(message: str, print_also: bool = False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if print_also:
        print(line)

def _format_sources(docs: List, include_chunks=True) -> str:
    lines = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown.pdf")
        # lấy chunk_id nếu có, nếu không fallback index (i+1)
        chunk_id = d.metadata.get("chunk_id", i + 1) if include_chunks else None
        if include_chunks:
            lines.append(f"- {src} [chunk {chunk_id}]")
        else:
            lines.append(f"- {src}")
    return "\n".join(lines)

def _maybe_filter_nlp(docs, query):
    q = query.lower()
    if any(k in q for k in ["nlp", "ngôn ngữ", "language", "text", "dịch máy", "parsing", "token", "transformer"]):
        prefer = ("ACL_", "EMNLP_", "NAACL_", "ICLR_", "NeurIPS_", "ICML_")
        filtered = [d for d in docs if str(d.metadata.get("source","")).startswith(prefer)]
        # nếu lọc ra rỗng, trả lại docs gốc
        return filtered if filtered else docs
    return docs

def _detect_and_translate_to_english(text: str) -> str:
    """Nếu câu hỏi là tiếng Việt → tự dịch sang tiếng Anh."""
    try:
        lang = detect(text)
        if lang == "vi":
            return GoogleTranslator(source="vi", target="en").translate(text)
        return text
    except:
        return text

def _translate_answer_to_vietnamese(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="vi").translate(text)
    except:
        return text

def query_pipeline_node(state: dict) -> dict:
    t0 = time.time()
    user_query = (state.get("user_input") or "").strip()
    if not user_query:
        msg = "Không có câu truy vấn nào được cung cấp."
        log(msg)     # dùng logger mặc định
        qlog(msg)    # và ghi thêm vào query.log
        state["response"] = msg
        return state

    log("=== BẮT ĐẦU BƯỚC 4: QUERY & RETRIEVAL ===")
    qlog("=== BẮT ĐẦU BƯỚC 4: QUERY & RETRIEVAL ===")
    log(f"Câu hỏi: {user_query}")
    qlog(f"Câu hỏi: {user_query}")

    query_en = _detect_and_translate_to_english(user_query)
    if query_en != user_query:
        log(f"→ Dịch sang tiếng Anh để truy vấn: {query_en}")
        qlog(f"→ Dịch sang tiếng Anh để truy vấn: {query_en}")
    else:
        query_en = user_query

    # Khởi tạo embedding model cho truy vấn
    log("Khởi tạo embedding: Alibaba-NLP/gte-Qwen2-1.5B-instruct.")
    qlog("Khởi tạo embedding: Alibaba-NLP/gte-Qwen2-1.5B-instruct.")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Kết nối Chroma
    if not any(DB_DIR.iterdir()):
        msg = (
            "ChromaDB chưa có dữ liệu. Hãy chạy pipeline ingest/embed trước khi truy vấn."
        )
        log(msg)
        qlog(msg)
        state["response"] = msg
        state.setdefault("trace", []).append("[query] Không tìm thấy dữ liệu trong Chroma.")
        return state

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embedder,
    )

    # Retrieval
    top_k = 10
    log(f"Thực hiện similarity_search(top_k={top_k})...")
    qlog(f"Thực hiện similarity_search(top_k={top_k})...")
    docs = vectordb.similarity_search(query_en, k=top_k)
    docs = _maybe_filter_nlp(docs, user_query)

    if not docs:
        msg = "Không tìm thấy context phù hợp trong cơ sở dữ liệu. Hãy thử từ khóa khác hoặc câu hỏi cụ thể hơn."

        if len(docs[0].page_content.strip()) < 50:
            msg = "Không đủ thông tin trong NGỮ CẢNH để trả lời câu hỏi này."
            log(msg)
            qlog(msg)
            state["response"] = msg
            state.setdefault("trace", []).append("[query] context quá yếu → từ chối trả lời.")
            return state

        log(msg)
        qlog(msg)
        state["response"] = msg
        state.setdefault("trace", []).append("[query] 0 context, không có kết quả.")
        return state

    log(f"Truy xuất được {len(docs)} đoạn context.")
    qlog(f"Truy xuất được {len(docs)} đoạn context.")

    # Chuẩn bị context rút gọn để prompt (tránh quá dài)
    # Lấy text + nguồn (giảm rủi ro prompt quá lớn)
    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown.pdf")
        chunk_id = d.metadata.get("chunk_id", i)  # fallback i nếu không có
        snippet = d.page_content.strip()
        # cắt ngắn mỗi đoạn để prompt gọn (tuỳ CPU/LLM hạn mức)
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        context_blocks.append(f"[{i}] {src} [chunk {chunk_id}]\n{snippet}")

    context_str = "\n\n".join(context_blocks)

    # Prompt LLM (Qwen2.5)
    prompt = (
        "Bạn là trợ lý học thuật. Trả lời CHỈ dựa trên NGỮ CẢNH cho trước.\n"
        "Nếu NGỮ CẢNH không chứa đủ bằng chứng, hãy nói rõ: 'Không đủ thông tin trong NGỮ CẢNH để kết luận.'\n\n"
        f"CÂU HỎI: {query_en}\n\n"
        f"NGỮ CẢNH:\n{context_str}\n\n"
        "YÊU CẦU:\n"
        "1) Trả lời ngắn gọn, chính xác, bằng tiếng Việt.\n"
        "2) Không suy diễn ngoài NGỮ CẢNH.\n"
        "3) Cuối câu trả lời, liệt kê Nguồn theo dạng <tên file> [chunk X].\n"
    )

    llm = make_llm()
    log("Gọi Qwen2.5 để sinh câu trả lời...")
    qlog("Gọi Qwen2.5 để sinh câu trả lời...")

    llm_raw = llm.invoke(prompt)
    llm_answer = getattr(llm_raw, "content", str(llm_raw)).strip()

    context_text_small = (context_str[:1000] or "").lower()
    answer_small = llm_answer.lower()

    if not any(tok in context_text_small for tok in answer_small.split()[:5]):
        llm_answer = "Không đủ thông tin trong NGỮ CẢNH để kết luận."

    llm_answer_vi = _translate_answer_to_vietnamese(llm_answer)

    elapsed = time.time() - t0
    log("=== HOÀN TẤT QUERY PIPELINE ===")
    qlog("=== HOÀN TẤT QUERY PIPELINE ===")
    log(f"Thời gian thực thi: {elapsed:.2f}s")
    qlog(f"Thời gian thực thi: {elapsed:.2f}s")

    # Ghép “nguồn” đúng yêu cầu: file + vị trí chunk
    sources_str = _format_sources(docs, include_chunks=True)

    full_response = (
        f"**Câu trả lời:**\n{llm_answer_vi}\n\n"
        f"**Nguồn tham chiếu:**\n{sources_str}"
    )

    # Cập nhật state
    state["llm_answer"] = llm_answer_vi
    state["response"] = users_query(user_query, full_response)
    state.setdefault("trace", []).append(
        f"[query] top_k={top_k}, results={len(docs)}, elapsed={elapsed:.2f}s"
    )

    # Ghi log chi tiết kết quả
    qlog("----- KẾT QUẢ LLM -----")
    qlog(llm_answer if len(llm_answer) <= 2000 else llm_answer[:2000] + "...[truncated]")
    qlog("----- NGUỒN -----")
    qlog(sources_str)

    return state