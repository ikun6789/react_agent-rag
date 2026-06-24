import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（根据当前文件位置自动推断，本地和容器都适用）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    ARK_API_KEY = os.getenv("ARK_API_KEY")
    ARK_BASE_URL = os.getenv("ARK_BASE_URL")

    LLM_MODEL = os.getenv("LLM_MODEL", "doubao-seed-2-0-pro-260215")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

    EMBEDDING_MODEL_PATH = os.getenv(
        "EMBEDDING_MODEL_PATH",
        str(PROJECT_ROOT.parent.parent / "sentence-transformers/BAAI/bge-small-zh-v1.5")
    )

    VECTOR_DB_PATH = os.getenv(
        "VECTOR_DB_PATH",
        str(PROJECT_ROOT / "data/vector_db/mixed_docs_faiss")
    )

    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
    RERANK_MODEL_PATH = os.getenv(
        "RERANK_MODEL_PATH",
        str(PROJECT_ROOT.parent.parent / "sentence-transformers/BAAI/bge-reranker-v2-m3")
    )
    RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")


settings = Settings()
