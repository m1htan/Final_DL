import re
from langchain_ollama import ChatOllama
def a(rag_answer: str) -> bool:
    if not rag_answer:
        return True
    patterns = [
        r"không đủ thông tin",
        r"không có thông tin",
        r"không thể kết luận",
        r"ngữ cảnh.*không.*cung cấp",
    ]
    text = rag_answer.lower()
    return any(re.search(p, text) for p in patterns)
def b(question: str) -> str:
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.3,
        num_predict=512,
        num_ctx=2048
    )
    prompt = (
        "Bạn là trợ lý tri thức. Nếu có thể, trả lời chính xác câu hỏi dưới đây.\n"
        "Nếu câu hỏi quá khó, hãy trả lời ngắn gọn nhưng có ý nghĩa.\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời bằng tiếng Việt."
    )
    try:
        out = llm.invoke(prompt)
        return getattr(out, "content", str(out)).strip()
    except Exception as e:
        return f"Không thể được! {e}"
def users_query(question: str, rag_answer: str) -> str:
    if not a(rag_answer):
        return rag_answer
    fallback_resp = b(question)
    wrapped = (
        "**Câu trả lời:**\n"
        f"{fallback_resp}\n\n"
    )
    return wrapped