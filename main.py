import argparse
from typing import Dict
from langgraph.graph import StateGraph, END
from src.state import OrchestratorState
from src.nodes.step1.planner import planner_node
from src.nodes.step1.router import ingest_stub, query_stub
from src.nodes.step3.embedding_pipeline import embedding_pipeline_node

def build_graph(force_mode=None):
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("ingest", ingest_stub)
    graph.add_node("query", query_stub)
    graph.add_node("embed", embedding_pipeline_node)

    # Entry
    # Nếu ép mode thì vào thẳng node đó, bỏ qua planner
    if force_mode == "ingest":
        graph.set_entry_point("ingest")
    elif force_mode == "embed":
        graph.set_entry_point("embed")
    elif force_mode == "query":
        graph.set_entry_point("query")
    else:
        graph.set_entry_point("planner")

    # Edges cho các trường hợp bình thường
    graph.add_conditional_edges(
        "planner",
        lambda s: s.get("next_action", "END"),
        {
            "INGEST": "ingest",
            "EMBED": "embed",
            "QUERY": "query",
            "END": END
        },
    )

    graph.add_edge("ingest", END)
    graph.add_edge("embed", END)
    graph.add_edge("query", END)

    return graph.compile()

def run_cli():
    parser = argparse.ArgumentParser(description="RAG-Orchestrator (LangGraph + Gemini)")
    parser.add_argument("--mode", type=str, default="plan_only",
                        choices=["plan_only", "ingest", "embed", "query"],
                        help="Chế độ ép nhánh.")
    parser.add_argument("--input", type=str, default="",
                        help="Câu lệnh/câu hỏi từ người dùng.")
    args = parser.parse_args()

    app = build_graph(force_mode=args.mode)
    init_state: Dict = {
        "user_input": args.input,
        "mode": args.mode,
        "trace": []
    }

    final = app.invoke(init_state)
    print("=== RESPONSE ===")
    print(final.get("response","<no response>"))
    print("\n=== TRACE ===")
    for t in final.get("trace", []):
        print(t)
    print("\n=== PLAN ===")
    print(final.get("plan",""))

if __name__ == "__main__":
    run_cli()