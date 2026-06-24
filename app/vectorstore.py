from langchain_community.vectorstores import FAISS

from app.config import settings
from app.embeddings import get_embeddings


def get_vectorstore():
    embeddings = get_embeddings()

    db = FAISS.load_local(
        settings.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return db


def get_retriever(use_hybrid: bool = False):
    """
    获取检索器。
    
    Args:
        use_hybrid: 是否使用混合检索（向量+BM25），默认 False 保持向后兼容
    """
    if use_hybrid:
        from app.hybrid_retriever import get_hybrid_retriever
        return get_hybrid_retriever(k=settings.RETRIEVER_K)

    db = get_vectorstore()

    return db.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_K}
    )