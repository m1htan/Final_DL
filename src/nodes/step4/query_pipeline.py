import time
from pathlib import Path
from typing import List
from src.utils.logger import log
from src.llm import make_llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langdetect import detect
from deep_translator import GoogleTranslator
import re

from src.config import CHROMA_DIR, EMBEDDING_MODEL
from src.nodes.step4.users_query import users_query

DB_DIR = Path(CHROMA_DIR)
DB_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "instruct2ds"
QUERY_LOG = Path("logs/query.log")
QUERY_LOG.parent.mkdir(exist_ok=True)

def _normalize(text: str) -> str:
    return (text or "").lower()

def _question_supported_by_context(question: str, context: str) -> bool:
    """
    Trả về True nếu trong NGỮ CẢNH có ít nhất một từ khóa 'dài' của câu hỏi.
    Nếu không, coi như không có ngữ cảnh phù hợp để trả lời.
    """
    q = _normalize(question)
    c = _normalize(context)

    # Lấy token chữ+ số, bỏ dấu câu
    tokens = re.findall(r"\w+", q)
    # Giữ lại token dài >=5 ký tự (lọc bớt từ chức năng: có, được, trong, ...)
    long_tokens = [t for t in tokens if len(t) >= 5]

    if not long_tokens:
        # Nếu câu hỏi toàn từ ngắn, chấp nhận là "được support"
        return True

    # Cần ít nhất 1 token dài xuất hiện trong context
    return any(t in c for t in long_tokens)


def _answer_consistent_with_context(answer: str, question: str, context: str) -> bool:
    """
    Hậu kiểm câu trả lời:
    - Phải có từ khóa từ câu hỏi
    - Phải dùng từ vựng xuất hiện trong context
    Nếu không, coi như không đủ thông tin.
    """
    a = _normalize(answer)
    q = _normalize(question)
    c = _normalize(context)

    # Nếu LLM đã trả câu "không đủ thông tin..." thì coi như ok
    if "không đủ thông tin trong ngữ cảnh" in a:
        return True

    # 1) Câu trả lời phải đụng đến một số từ trong câu hỏi
    q_tokens = [t for t in re.findall(r"\w+", q) if len(t) >= 4]
    if q_tokens and not any(t in a for t in q_tokens):
        return False

    # 2) Câu trả lời phải dùng ít nhất một số từ 'dài' đã có trong context
    a_tokens = [t for t in re.findall(r"\w+", a) if len(t) >= 5]
    if a_tokens and not any(t in c for t in a_tokens):
        return False

    return True

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
        log(msg)
        state["response"] = msg
        return state

    log("=== BẮT ĐẦU BƯỚC 4: QUERY & RETRIEVAL ===")
    log(f"Câu hỏi: {user_query}")

    query_en = _detect_and_translate_to_english(user_query)
    if query_en != user_query:
        log(f"→ Dịch sang tiếng Anh để truy vấn: {query_en}")
    else:
        query_en = user_query

    # Khởi tạo embedding model cho truy vấn
    log("Khởi tạo embedding: Alibaba-NLP/gte-Qwen2-1.5B-instruct.")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Kết nối Chroma
    if not any(DB_DIR.iterdir()):
        msg = (
            "ChromaDB chưa có dữ liệu. Hãy chạy pipeline ingest/embed trước khi truy vấn."
        )
        log(msg)
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
    docs = vectordb.similarity_search(query_en, k=top_k)
    docs = _maybe_filter_nlp(docs, user_query)

    if not docs or all(len(d.page_content.strip()) < 100 for d in docs):
        msg = "Không đủ thông tin trong NGỮ CẢNH để trả lời câu hỏi này."
        log(msg)
        state["response"] = msg
        return state

    log(f"Truy xuất được {len(docs)} đoạn context.")

    # Chuẩn bị context rút gọn để prompt (tránh quá dài)
    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown.pdf")
        chunk_id = d.metadata.get("chunk_id", i)  # fallback i nếu không có
        snippet = d.page_content.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        context_blocks.append(f"[{i}] {src} [chunk {chunk_id}]\n{snippet}")

    context_str = "\n\n".join(context_blocks)

    # ===== CHẶN TỪ GỐC: nếu context không support câu hỏi → KHÔNG GỌI LLM =====
    if not _question_supported_by_context(query_en, context_str):
        msg = "Không đủ thông tin trong NGỮ CẢNH để kết luận."
        log("Context không chứa từ khóa trọng tâm của câu hỏi → bỏ qua LLM.")
        elapsed = time.time() - t0
        log("=== HOÀN TẤT QUERY PIPELINE ===")
        log(f"Thời gian thực thi: {elapsed:.2f}s")

        sources_str = _format_sources(docs, include_chunks=True)
        full_response = (
            f"**Câu trả lời:**\n{msg}\n\n"
            f"**Nguồn tham chiếu:**\n{sources_str}"
        )
        state["llm_answer"] = msg
        state["response"] = users_query(user_query, full_response)
        state.setdefault("trace", []).append(
            f"[query] top_k={top_k}, results={len(docs)}, elapsed={elapsed:.2f}s, no_support=True"
        )
        return state

    # Prompt LLM (Qwen2.5)
    prompt = (
        "Bạn TUYỆT ĐỐI KHÔNG được sử dụng kiến thức ngoài NGỮ CẢNH. "
        "Nếu NGỮ CẢNH không chứa thông tin để trả lời CÂU HỎI, hãy trả lời đúng 1 câu:\n"
        "'Không đủ thông tin trong NGỮ CẢNH để kết luận.'\n\n"
        "Không được suy diễn, không được tưởng tượng, không được dựa trên hiểu biết bên ngoài.\n\n"
        f"CÂU HỎI: {query_en}\n\n"
        f"NGỮ CẢNH:\n{context_str}\n\n"
        "YÊU CẦU:\n"
        "- Trả lời bằng tiếng Việt, dựa 100% vào NGỮ CẢNH.\n"
        "- Nếu không chắc 100%, trả lời: 'Không đủ thông tin trong NGỮ CẢNH để kết luận.'\n"
        "- Cuối câu trả lời, liệt kê nguồn dạng <file> [chunk X].\n"
    )

    llm = make_llm()
    log("Gọi Qwen2.5 để sinh câu trả lời...")

    llm_raw = llm.invoke(prompt)
    llm_answer = getattr(llm_raw, "content", str(llm_raw)).strip()

    # ===== HẬU KIỂM: nếu answer không nhất quán với context & question → ép fallback =====
    if not _answer_consistent_with_context(llm_answer, user_query, context_str):
        log("Câu trả lời không nhất quán với NGỮ CẢNH hoặc CÂU HỎI → ép fallback.")
        llm_answer = "Không đủ thông tin trong NGỮ CẢNH để kết luận."

    llm_answer_vi = _translate_answer_to_vietnamese(llm_answer)

    elapsed = time.time() - t0
    log("=== HOÀN TẤT QUERY PIPELINE ===")
    log(f"Thời gian thực thi: {elapsed:.2f}s")

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
    return state
