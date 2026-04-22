from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from telegram_agent_aws.config import settings
from telegram_agent_aws.infrastructure.clients.qdrant import get_qdrant_client


def generate_split_documents():
    loader = PyPDFLoader("./data/karan_full_biography.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)

    return all_splits


def index_documents():
    all_splits = generate_split_documents()
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL, google_api_key=settings.GEMINI_API_KEY)

    QdrantVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name="telegram_agent_aws_collection",
    )

    logger.info("Documents indexed successfully.")
