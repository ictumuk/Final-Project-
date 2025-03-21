from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from .config import QDRANT_KEY, QDRANT_URL,TAVILY_API_KEY, SPARSE_MODEL_NAME, DENSE_MODEL_NAME, COLLECTION_NAME

embeddings = None
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)


# Function to initialize qdrant hybrid store
def initialize_qdrant_hybrid():
    global embeddings, sparse_embeddings
    # Initialize embeddings if they are None
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL_NAME)

    if sparse_embeddings is None:
        sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)

    # Initialize QdrantVectorStore with hybrid retrieval mode
    qdrant_store = QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_KEY
    )
    return qdrant_store

def initialize_tavily_tool():
    """
    Initialize Tavily Search tool.
    """
    api_wrapper = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
    return TavilySearchResults(api_wrapper=api_wrapper)
