from typing import Dict
from src.nodes.step2.ingestion_pipeline import ingestion_pipeline_node

def ingest_stub(state: Dict) -> Dict:
    # Không dùng nữa, để ngừa ai đó import cũ
    return ingestion_pipeline_node(state)

def query_stub(state: Dict) -> Dict:
    trace = list(state.get("trace", []))
    trace.append("[query_stub] Chưa triển khai. Sẽ thực hiện ở Bước 3.")
    resp = "Query stub: hệ thống đã sẵn sàng cho bước 3 (retriever → Gemini → post-process)."
    return {**state, "response": resp, "trace": trace}
