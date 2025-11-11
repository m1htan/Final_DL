import argparse
from typing import Dict

from langgraph.graph import END, StateGraph

from src.nodes.step1.planner import planner_node
from src.nodes.step1.router import ingest_entry
from src.nodes.step3.embedding_pipeline import embedding_pipeline_node
from src.nodes.step4.query_pipeline import query_pipeline_node
from src.nodes.step5.evaluation_pipeline import evaluation_pipeline_node
from src.state import OrchestratorState


def build_graph(force_mode=None):
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("ingest", ingest_entry)
    graph.add_node("embed", embedding_pipeline_node)
    graph.add_node("query", query_pipeline_node)
    graph.add_node("evaluate", evaluation_pipeline_node)

    # Entry
    if force_mode == "evaluate":
        graph.set_entry_point("evaluate")
    elif force_mode == "ingest":
        graph.set_entry_point("ingest")
    elif force_mode == "embed":
        graph.set_entry_point("embed")
    elif force_mode == "query":
        graph.set_entry_point("query")
    else:
        graph.set_entry_point("planner")

    # Planner transitions
    graph.add_conditional_edges(
        "planner",
        lambda s: s.get("next_action", "END"),
        {
            "INGEST": "ingest",
            "EMBED": "embed",
            "QUERY": "query",
            "EVAL": "evaluate",
            "END": END,
        },
    )

    graph.add_edge("ingest", END)
    graph.add_edge("embed", END)
    graph.add_edge("query", END)
    graph.add_edge("evaluate", END)

    return graph.compile()


def run_cli():
    parser = argparse.ArgumentParser(
        description="RAG-Orchestrator (LangGraph + Qwen2.5)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="plan_only",
        choices=["plan_only", "ingest", "embed", "query", "evaluate"],
        help="Chế độ ép nhánh.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Câu lệnh/câu hỏi từ người dùng.",
    )
    args = parser.parse_args()

    app = build_graph(force_mode=args.mode)
    init_state: Dict = {
        "user_input": args.input,
        "mode": args.mode,
        "trace": [],
    }

    final = app.invoke(init_state)
    print("=== RESPONSE ===")
    print(final.get("response", "<no response>"))
    print("\n=== TRACE ===")
    for t in final.get("trace", []):
        print(t)
    print("\n=== PLAN ===")
    print(final.get("plan", ""))


if __name__ == "__main__":
    run_cli()
