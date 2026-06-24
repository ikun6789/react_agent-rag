"""
混合检索器：向量检索 + BM25 关键词检索

将两种检索结果加权融合，兼顾语义相似度和关键词匹配。
"""

import pickle
from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.vectorstore import get_vectorstore


# BM25 权重（可调，越大越偏向关键词匹配）
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


def _load_bm25_retriever(k: int = 30):
    """加载 BM25 检索器（从之前保存的 chunks.pkl）"""
    from app.config import settings
    chunks_path = Path(settings.VECTOR_DB_PATH) / "chunks.pkl"

    if not chunks_path.exists():
        print(f"[警告] BM25 文档块不存在: {chunks_path}，请重新运行 build_vector_db.py")
        return None

    with open(chunks_path, "rb") as f:
        docs = pickle.load(f)

    # 提取纯文本用于 BM25
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    retriever = BM25Retriever.from_texts(
        texts,
        metadatas=metadatas,
        k=k,
    )
    return retriever


def _reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 60,
) -> List[Document]:
    """
    使用 Reciprocal Rank Fusion (RRF) 融合两种检索结果。
    RRF 分数 = 1 / (k + rank)，对排序位置敏感，即使分数尺度不同也能很好地融合。
    """
    seen = {}
    
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        if content not in seen:
            seen[content] = {
                "doc": doc,
                "score": 0.0,
                "vector_rank": rank,
            }
        seen[content]["score"] += VECTOR_WEIGHT / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        content = doc.page_content
        if content not in seen:
            seen[content] = {
                "doc": doc,
                "score": 0.0,
                "bm25_rank": rank,
            }
        seen[content]["score"] += BM25_WEIGHT / (k + rank + 1)

    # 按融合分数排序
    sorted_docs = sorted(
        seen.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    result = []
    for item in sorted_docs:
        doc = item["doc"]
        doc.metadata["hybrid_score"] = round(item["score"], 4)
        result.append(doc)

    return result


def get_hybrid_retriever(k: int = 15):
    """
    获取混合检索器。

    用法:
        retriever = get_hybrid_retriever(k=15)
        docs = retriever.invoke("汽车网关的信息安全技术要求")
    """
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = _load_bm25_retriever(k=k)

    class HybridRetriever:
        def invoke(self, query: str) -> List[Document]:
            # 向量检索（使用 MMR 增加多样性，减少冗余结果）
            vector_docs = vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 2, lambda_mult=0.7
            )

            # BM25 检索
            if bm25_retriever:
                bm25_docs = bm25_retriever.invoke(query)
            else:
                bm25_docs = []

            # RRF 融合
            fused = _reciprocal_rank_fusion(vector_docs, bm25_docs)

            return fused[:k]

    return HybridRetriever()
