import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from .config import DENSE_MODEL_NAME, SPARSE_MODEL_NAME, QDRANT_KEY, QDRANT_URL

def load_documents(file_path: str) -> list[Document]:
    """
    Loads and transforms data from a JSON file into Document objects.

    Args:
    - file_path: Path to the JSON file containing the document data.

    Returns:
    - A list of Document objects.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    return [
        Document(
            page_content=entry["text"],
            metadata={
                "law_id": entry["law_id"],
                "article_id": entry["article_id"]
            }
        ) for entry in data
    ]


def create_text_splitter(chunk_size: int = 400) -> RecursiveCharacterTextSplitter:
    """
    Creates a text splitter instance with specified chunk size

    Args:
    - chunk_size: The size of each text chunk.

    Returns:
    - A RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
    )


def create_embeddings() -> tuple[HuggingFaceEmbeddings, FastEmbedSparse]:
    """
    Creates and returns dense and sparse embeddings.

    Returns:
    - A tuple containing dense and sparse embeddings.
    """
    dense_embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL_NAME)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)

    return dense_embeddings, sparse_embeddings


def create_qdrant_vector_store(docs: list[Document],
                               embeddings: HuggingFaceEmbeddings,
                               sparse_embeddings: FastEmbedSparse) -> QdrantVectorStore:
    """
    Creates a Qdrant vector store with given documents and embeddings.

    Args:
    - docs: The list of documents to store.
    - embeddings: Dense embeddings.
    - sparse_embeddings: Sparse embeddings.

    Returns:
    - A QdrantVectorStore instance.
    """
    return QdrantVectorStore.from_documents(
        docs,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name="legal_documents",
        retrieval_mode=RetrievalMode.HYBRID,
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_KEY,
        force_recreate=True
    )


def main():
    # Load documents
    documents = load_documents('transformed_law.json')

    # Split documents into smaller chunks
    text_splitter = create_text_splitter()
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings, sparse_embeddings = create_embeddings()

    # Create Qdrant vector store
    qdrant_hybrid = create_qdrant_vector_store(docs, embeddings, sparse_embeddings)


if __name__ == "__main__":
    main()