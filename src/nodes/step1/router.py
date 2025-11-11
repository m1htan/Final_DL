from typing import Dict

from src.nodes.step2.ingestion_pipeline import ingestion_pipeline_node


def ingest_entry(state: Dict) -> Dict:
    """Entry wrapper để đảm bảo trace được cập nhật trước khi chạy ingestion."""
    trace = list(state.get("trace", []))
    trace.append("[router] chuyển sang pipeline ingestion")
    updated_state = {**state, "trace": trace}
    return ingestion_pipeline_node(updated_state)

