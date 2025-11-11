from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import make_llm

SYSTEM_PROMPT = """Bạn là Orchestrator cho hệ thống RAG xây dựng bằng LangGraph.
Chỉ trả về JSON với 2 trường: "next_action" và "plan".
next_action ∈ {"INGEST", "EMBED", "QUERY", "EVAL", "END"}.

Hướng dẫn:
- Nếu người dùng yêu cầu ingest, build dữ liệu, tải PDF → "INGEST".
- Nếu người dùng muốn nhúng lại dữ liệu, tái tạo vector → "EMBED".
- Nếu người dùng đặt câu hỏi, cần trả lời dựa trên tri thức → "QUERY".
- Nếu người dùng nói đánh giá, evaluate, kiểm thử chất lượng → "EVAL".
- Nếu không có hành động phù hợp → "END" với plan giải thích ngắn gọn.
"""

def planner_node(state: Dict) -> Dict:
    llm = make_llm()
    user = state.get("user_input","").strip()
    mode = state.get("mode","plan_only")

    # Nếu mode ép sẵn (cho CLI), ưu tiên mode:
    if mode == "ingest":
        next_action = "INGEST"
        plan = "Thực hiện pipeline ingestion."
    elif mode == "embed":
        next_action = "EMBED"
        plan = "Chạy lại pipeline embedding cho toàn bộ tài liệu."
    elif mode == "query":
        next_action = "QUERY"
        plan = "Thực hiện truy vấn RAG."
    elif mode == "evaluate":
        next_action = "EVAL"
        plan = "Chạy pipeline đánh giá chất lượng hệ thống."
    else:
        # Hỏi LLM để quyết định
        msgs = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User said: {user}")
        ]
        out = llm.invoke(msgs).content  # kỳ vọng JSON đơn giản
        # Robust parsing tối giản
        import json, re
        try:
            # Trích JSON đầu tiên
            jtxt = re.search(r"\{.*\}", out, re.S)
            data = json.loads(jtxt.group(0)) if jtxt else {}
            next_action = data.get("next_action","END")
            plan = data.get("plan","Không có kế hoạch chi tiết.")
            if next_action not in {"INGEST", "EMBED", "QUERY", "EVAL", "END"}:
                next_action = "END"
                plan = "LLM trả về hành động không hợp lệ. Kết thúc."
        except Exception:
            next_action, plan = "END", "Không parse được phản hồi từ Orchestrator."

    trace = list(state.get("trace", []))
    trace.append(f"[planner] mode={mode} -> next={next_action}; plan={plan}")

    return {**state, "next_action": next_action, "plan": plan, "trace": trace}

