import argparse
from typing import Dict
from langgraph.graph import StateGraph, END
from src.state import OrchestratorState
from src.nodes.planner import planner_node
from src.nodes.router import ingest_stub, query_stub

def build_graph():
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("ingest", ingest_stub)
    graph.add_node("query", query_stub)

    # Entry
    graph.set_entry_point("planner")

    # Edges theo quyết định của planner
    graph.add_conditional_edges(
        "planner",
        lambda s: s.get("next_action","END"),
        {
            "INGEST": "ingest",
            "QUERY": "query",
            "END": END
        },
    )

    # Sau khi chạy nhánh nào thì kết thúc (bước 1)
    graph.add_edge("ingest", END)
    graph.add_edge("query", END)

    return graph.compile()

def run_cli():
    parser = argparse.ArgumentParser(description="RAG-Orchestrator (LangGraph + Gemini)")
    parser.add_argument("--mode", type=str, default="plan_only",
                        choices=["plan_only","ingest","query"],
                        help="Chế độ ép nhánh.")
    parser.add_argument("--input", type=str, default="",
                        help="Câu lệnh/câu hỏi từ người dùng.")
    args = parser.parse_args()

    app = build_graph()
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