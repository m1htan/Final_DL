from typing import TypedDict, List, Literal, Optional, Any

class OrchestratorState(TypedDict, total=False):
    # Đầu vào/đầu ra tổng thể
    user_input: str
    response: str

    # Điều phối
    mode: Literal["plan_only", "ingest", "embed", "query", "evaluate"]
    plan: str
    next_action: Literal["INGEST", "EMBED", "QUERY", "EVAL", "END"]

    # Log/trace
    trace: List[str]

    # Payload tương lai (bước 2–3 sẽ dùng)
    ingestion_args: Optional[dict]
    embedding_args: Optional[dict]
    evaluation_args: Optional[dict]
    query_args: Optional[dict]
    context_chunks: Optional[List[dict]]
    llm_answer: Optional[str]
    post_processed: Optional[str]
    extra: Optional[Any]

    # Tham số phục vụ UI/query
    top_k: Optional[int]
    max_context_chars: Optional[int]
    retrieved_docs: Optional[List[dict]]
    sources_markdown: Optional[str]
    answer_model: Optional[str]
    latency_seconds: Optional[float]
    context_prompt: Optional[str]
    ui_request: Optional[bool]
