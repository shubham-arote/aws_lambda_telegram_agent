from functools import lru_cache

from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings  # noqa: F401 (kept intentionally)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from telegram_agent_aws.config import settings
from telegram_agent_aws.infrastructure.clients.qdrant import get_qdrant_client


@lru_cache(maxsize=1)
def get_retriever_tool():
    """Get the retriever tool as a singleton using LRU cache."""
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL, google_api_key=settings.GEMINI_API_KEY)

    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name="telegram_agent_aws_collection",
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="retrieve_karan_information_tool",
        description="Retrieve information about the Telegram Agent's background, academic journey, professional experience, major projects, philosophy, values, hobbies and personal interests",
    )

    return retriever_tool
