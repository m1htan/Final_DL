from typing import Dict

def ingest_stub(state: Dict) -> Dict:
    trace = list(state.get("trace", []))
    trace.append("[ingest_stub] Chưa triển khai. Sẽ thực hiện ở Bước 2.")
    resp = "Ingestion stub: hệ thống đã sẵn sàng cho bước 2 (tải PDF, chunk, embed, Chroma)."
    return {**state, "response": resp, "trace": trace}

def query_stub(state: Dict) -> Dict:
    trace = list(state.get("trace", []))
    trace.append("[query_stub] Chưa triển khai. Sẽ thực hiện ở Bước 3.")
    resp = "Query stub: hệ thống đã sẵn sàng cho bước 3 (retriever → Gemini → post-process)."
    return {**state, "response": resp, "trace": trace}
