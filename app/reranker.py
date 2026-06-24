import json
import re
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.llm import get_llm


RERANK_TEMPLATE = """
你是一个严格的 RAG 检索结果重排序器。

用户问题：
{question}

候选检索片段：
{docs}

请根据候选片段对用户问题的回答价值进行重排序。

排序标准：
1. 优先选择能直接回答问题、包含关键事实、定义、要求、条款、指标、流程的片段。
2. 其次选择主题相关但信息不完整的片段。
3. 排除只有泛泛背景、重复内容、与问题无关的片段。
4. 不要因为片段更长就给更高分，只看是否能支撑回答。

请只输出最相关的前 {top_k} 个片段 ID。
请严格输出 JSON，不要输出 Markdown，不要输出额外解释。

输出格式：
{{
  "ranked_ids": [
    {{"id": 3, "score": 0.95, "reason": "直接包含问题所需要求"}},
    {{"id": 1, "score": 0.82, "reason": "包含相关定义"}}
  ]
}}
"""


def _extract_json(content: str) -> dict:
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError("未找到 JSON 内容")

    return json.loads(match.group(0))


def _format_docs_for_rerank(docs) -> str:
    result = []

    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "未知文件")
        file_type = doc.metadata.get("file_type", "未知类型")
        page = doc.metadata.get("page", None)
        page_info = f"第 {page + 1} 页" if page is not None else "无页码"

        result.append(
            f"【候选片段 {i}】\n"
            f"片段ID：{i}\n"
            f"文件名：{file_name}\n"
            f"文件类型：{file_type}\n"
            f"页码：{page_info}\n"
            f"内容：{doc.page_content}\n"
        )

    return "\n\n".join(result)


@lru_cache(maxsize=1)
def get_rerank_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        settings.RERANK_MODEL_PATH,
        device=settings.RERANK_DEVICE,
    )


def rerank_documents_by_model(question: str, docs, top_k: int):
    model = get_rerank_model()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored_docs = []

    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        doc.metadata["rerank_reason"] = "rerank 模型打分"
        scored_docs.append((doc, float(score)))

    scored_docs.sort(key=lambda item: item[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]


def rerank_documents_by_llm(question: str, docs, top_k: int):
    prompt = ChatPromptTemplate.from_template(RERANK_TEMPLATE)
    messages = prompt.format_messages(
        question=question,
        docs=_format_docs_for_rerank(docs),
        top_k=top_k,
    )

    response = get_llm().invoke(messages)
    result = _extract_json(response.content)

    ranked_items = result.get("ranked_ids", [])
    docs_by_id = {i: doc for i, doc in enumerate(docs, start=1)}

    reranked_docs = []
    used_ids = set()

    for item in ranked_items:
        doc_id = item.get("id")

        if doc_id not in docs_by_id or doc_id in used_ids:
            continue

        doc = docs_by_id[doc_id]
        doc.metadata["rerank_score"] = item.get("score")
        doc.metadata["rerank_reason"] = item.get("reason", "LLM 重排序")
        reranked_docs.append(doc)
        used_ids.add(doc_id)

        if len(reranked_docs) >= top_k:
            break

    if len(reranked_docs) < top_k:
        for i, doc in docs_by_id.items():
            if i in used_ids:
                continue
            reranked_docs.append(doc)
            if len(reranked_docs) >= top_k:
                break

    return reranked_docs


def rerank_documents(question: str, docs):
    """
    对向量检索召回的候选文档进行重排序。

    如果重排序失败，返回原始检索结果的前 RERANK_TOP_K 个。
    """
    if not docs:
        return []

    top_k = min(settings.RERANK_TOP_K, len(docs))

    try:
        return rerank_documents_by_model(question, docs, top_k)
    except Exception:
        pass

    try:
        return rerank_documents_by_llm(question, docs, top_k)
    except Exception:
        return docs[:top_k]
