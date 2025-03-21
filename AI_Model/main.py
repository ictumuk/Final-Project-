import argparse
from tools import initialize_qdrant_hybrid, initialize_tavily_tool
from ragflow import RAGFlow
from langchain_groq import ChatGroq
from cragflow import CRAGFlow
from react import ReactAgent
from config import GROQ_API_KEY, LANGSMITH_KEY, GENERATIVE_MODEL_NAME
import time
import os
from langchain_core.tools import StructuredTool
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = LANGSMITH_KEY

qdrant_hybrid = initialize_qdrant_hybrid()
tavily_tool = initialize_tavily_tool()
# Function to get relevant documents based on a query
def get_relevant_document(query: str) -> str:
    total_content = ""
    # Use similarity search from Qdrant with top 3 results
    results = qdrant_hybrid.similarity_search(query=query, k=3)

    # Concatenate the content of the results
    for doc in results:
        total_content += doc.page_content + "\n"

    return total_content


# Create the structured tool for retrieving relevant documents
get_relevant_document_tool = StructuredTool.from_function(
    name="Get Relevant document",
    func=get_relevant_document,
    description="Useful for getting relevant documents from local Qdrant store"
)
def parse_args():
    """
    Parse command-line arguments for selecting the chat mode, query, and question type.
    """
    parser = argparse.ArgumentParser(description="Select chat mode and process a query.")

    # Add input arguments
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query/question to be processed."
    )
    parser.add_argument(
        "--chat_mode",
        type=str,
        choices=["CRAG", "RAG", "REACT"],
        required=True,
        help="Select the chat mode: 'CRAG' for CRAGFlow, 'RAG' for RAGFlow, or 'REACT'."
    )
    parser.add_argument(
        "--question_type",
        type=str,
        required=True,
        help="Type of the question (e.g., 'Đúng/Sai', 'Trắc nghiệm','Tự luận')."
    )

    return parser.parse_args()

def main():
    # Record start time
    start_time = time.time()

    # Parse arguments
    args = parse_args()

    query = args.query
    question_type = args.question_type

    #Initialize common tools using lazy initialization
    llm = ChatGroq(model=GENERATIVE_MODEL_NAME, api_key=GROQ_API_KEY)

    # Select workflow based on chat mode
    if args.chat_mode == "CRAG":
        workflow = CRAGFlow(llm, qdrant_hybrid, tavily_tool).crag_flow()
    elif args.chat_mode == "RAG":
        workflow = RAGFlow(llm, qdrant_hybrid, tavily_tool).rag_flow()
    elif args.chat_mode == "REACT":
        workflow = ReactAgent(llm, [get_relevant_document_tool,tavily_tool]).react_agent()
    # Retrieve response
    response = workflow.invoke({"question": query, "question_type": question_type})

    # Print response
    print(response)

    # Record end time
    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.4f} seconds")

if __name__ == "__main__":
    main()
