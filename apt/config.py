from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA = DATA_DIR / "raw"
    PROCESSED_DATA = DATA_DIR / "processed"
    CHROMA_DB = DATA_DIR / "chroma_db"
    REPORTS_DIR = DATA_DIR / "reports"

    COLLECTION_NAME = "apt_reports"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    PDF_LOADER = os.getenv("PDF_LOADER", "pymupdf4llm")

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 5

    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "apt-rag")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("access_token")
