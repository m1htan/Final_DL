from typing import TypedDict, List, Literal, Optional, Any

class OrchestratorState(TypedDict, total=False):
    # Đầu vào/đầu ra tổng thể
    user_input: str
    response: str

    # Điều phối (bổ sung 'embed')
    mode: Literal["plan_only", "ingest", "embed", "query"]
    plan: str
    next_action: Literal["INGEST", "EMBED", "QUERY", "END"]

    # Log/trace
    trace: List[str]

    # Payload tương lai (bước 2–3 sẽ dùng)
    ingestion_args: Optional[dict]
    embedding_args: Optional[dict]
    query_args: Optional[dict]
    context_chunks: Optional[List[dict]]
    llm_answer: Optional[str]
    post_processed: Optional[str]
    extra: Optional[Any]
