from langgraph.graph import END, StateGraph, START
from .function_node import Node
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    question_type: str
    query_trans: str
    documents: List[str]

class RAGFlow(Node):
    def __init__(
            self,
            llm,
            retrieve_tool,
            web_search_tool
    ):
        super().__init__(
            llm=llm,
            retrieve_tool=retrieve_tool,
            web_search_tool=web_search_tool
        )
        self.graph_state = GraphState

    def rag_flow(self):
        """
               Create and configure the RAG workflow state graph

               Returns:
                   Compiled state graph for RAG workflow
               """
        ragflow = StateGraph(self.graph_state)
        # Add nodes with methods from parent class
        ragflow.add_node("retrieve", self.retrieve)
        ragflow.add_node("generate", self.generate)
        # Configure graph workflow
        ragflow.add_edge(START, "retrieve")
        ragflow.add_edge("retrieve", "generate")
        ragflow.add_edge("generate", END)
        return ragflow.compile()
