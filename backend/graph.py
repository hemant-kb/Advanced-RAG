"""
Master LangGraph — pure RAG chatbot.

Flow:
  entry → rag_subgraph → END

Every query goes straight into the RAG subgraph.
All queries go through retrieval — there is no direct-answer bypass.
"""
from __future__ import annotations

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph

from backend.rag.rag_graph import RAGState, build_rag_subgraph


def build_graph(checkpointer: AsyncSqliteSaver):
    """
    Compile the master graph.

    Because the only node is the RAG subgraph (compiled as a native node),
    LangSmith sees the full trace tree from one root run.
    """
    rag_subgraph = build_rag_subgraph().compile()

    graph = StateGraph(RAGState)
    graph.add_node("rag", rag_subgraph)
    graph.set_entry_point("rag")
    graph.add_edge("rag", END)

    return graph.compile(checkpointer=checkpointer)
