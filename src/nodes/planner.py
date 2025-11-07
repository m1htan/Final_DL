from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import make_gemini

SYSTEM_PROMPT = """Bạn là Orchestrator cho hệ thống RAG.
Bạn CHỈ trả về JSON với field: next_action in ["INGEST","QUERY","END"] và plan (vắn tắt).
Quy tắc:
- Nếu người dùng nói 'ingest', 'build', 'load data' -> INGEST
- Nếu người dùng đặt câu hỏi nội dung -> QUERY
- Nếu yêu cầu chỉ kiểm tra hệ thống -> END
"""

def planner_node(state: Dict) -> Dict:
    llm = make_gemini()
    user = state.get("user_input","").strip()
    mode = state.get("mode","plan_only")

    # Nếu mode ép sẵn (cho CLI), ưu tiên mode:
    if mode == "ingest":
        next_action = "INGEST"
        plan = "Thực hiện pipeline ingestion."
    elif mode == "query":
        next_action = "QUERY"
        plan = "Thực hiện truy vấn RAG."
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
            plan = data.get("plan","")
        except Exception:
            next_action, plan = "END", "Không parse được, kết thúc tạm thời."

    trace = list(state.get("trace", []))
    trace.append(f"[planner] mode={mode} -> next={next_action}; plan={plan}")

    return {**state, "next_action": next_action, "plan": plan, "trace": trace}
