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

#----
class CRAGFlow(Node):
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

    def crag_flow(self):
        """
               Create and configure the CRAG workflow state graph

               Returns:
                   Compiled state graph for CRAG workflow
               """


        cragflow = StateGraph(self.graph_state)
        # Add nodes with methods from parent class
        cragflow.add_node("retrieve", self.retrieve)
        cragflow.add_node("grade_documents", self.grade_documents)
        cragflow.add_node("generate", self.generate)
        cragflow.add_node("transform_query", self.transform_query)
        cragflow.add_node("web_search_node", self.web_search)
        # Configure graph workflow
        cragflow.add_edge(START, "retrieve")
        cragflow.add_edge("retrieve", "grade_documents")
        cragflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        cragflow.add_edge("transform_query", "web_search_node")
        cragflow.add_edge("web_search_node", "generate")
        cragflow.add_edge("generate", END)
        return cragflow.compile()