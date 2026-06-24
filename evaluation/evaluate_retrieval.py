"""
检索和重排序质量评估

评估维度：
1. 向量检索质量：Recall@k, Precision@k, MRR@k
2. 重排序质量：Recall@k (rerank后), NDCG@k
3. 证据判断质量：Evidence Evaluator 的准确率

用法：
    python evaluation/evaluate_retrieval.py
"""

import json
import sys
from pathlib import Path
from typing import List, Set

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.vectorstore import get_vectorstore, get_retriever
from app.reranker import rerank_documents
from app.evidence_evaluator import evaluate_evidence
from app.config import settings
from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    average_precision,
    ndcg_at_k,
    mean_average_precision,
    classification_metrics,
    summarize_metrics,
)


TESTSET_PATH = Path(__file__).resolve().parent / "testset.json"


def load_testset() -> List[dict]:
    """加载测试集"""
    if not TESTSET_PATH.exists():
        print(f"[错误] 测试集文件不存在: {TESTSET_PATH}")
        print("请先创建测试集文件，参考 evaluation/testset_template.json")
        sys.exit(1)

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_relevant_doc_ids(
    retriever,
    relevant_sources: List[str],
    all_docs_with_idx
) -> Set[int]:
    """
    根据源文件名找出相关文档在检索结果中的索引。
    
    这是简化版：如果有带标注的 ground-truth doc IDs 就直接用，
    否则通过源文件名模糊匹配。
    """
    relevant_set = set()
    
    for idx, doc in all_docs_with_idx:
        source = doc.metadata.get("file_name", "") or doc.metadata.get("source", "")
        for rs in relevant_sources:
            if rs.lower() in source.lower():
                relevant_set.add(idx)
                break
    
    return relevant_set


def evaluate_single_query(
    question: str,
    relevant_sources: List[str],
    relevant_doc_ids: List[int] = None,
) -> dict:
    """评估单个查询的检索和重排序质量"""
    from app.embeddings import get_embeddings

    vectorstore = get_vectorstore()
    retriever = get_retriever(use_hybrid=True)

    # --- 1. 原始检索 ---
    raw_docs = retriever.invoke(question)
    
    # 获取所有文档及其索引（用于标注 ground truth）
    all_docs_with_idx = list(enumerate(raw_docs))
    
    # 确定相关文档集合
    if relevant_doc_ids is not None and len(relevant_doc_ids) > 0:
        relevant_set = set(relevant_doc_ids)
    elif relevant_sources:
        relevant_set = find_relevant_doc_ids(retriever, relevant_sources, all_docs_with_idx)
    else:
        relevant_set = set()

    raw_retrieved_ids = list(range(len(raw_docs)))

    # --- 2. 检索指标 ---
    K_VALUES = [1, 3, 5, 10, 20, 30]

    retrieval_metrics = {}
    for k in K_VALUES:
        if k <= len(raw_docs):
            retrieval_metrics[f"Recall@{k}"] = recall_at_k(raw_retrieved_ids, relevant_set, k)
            retrieval_metrics[f"Precision@{k}"] = precision_at_k(raw_retrieved_ids, relevant_set, k)

    retrieval_metrics["MRR"] = mean_reciprocal_rank(raw_retrieved_ids, relevant_set)
    retrieval_metrics["AP"] = average_precision(raw_retrieved_ids, relevant_set)

    # --- 3. 重排序指标 ---
    reranked_docs = rerank_documents(question, raw_docs)
    reranked_ids = []
    reranked_scores = []

    for rd in reranked_docs:
        # 找到该文档在原始列表中的索引
        for i, raw_doc in enumerate(raw_docs):
            if raw_doc.page_content == rd.page_content:
                reranked_ids.append(i)
                reranked_scores.append(rd.metadata.get("rerank_score", 0))
                break

    rerank_metrics = {}
    for k in K_VALUES:
        if k <= len(reranked_docs):
            rerank_metrics[f"Rerank_Recall@{k}"] = recall_at_k(reranked_ids, relevant_set, k)
            rerank_metrics[f"Rerank_Precision@{k}"] = precision_at_k(reranked_ids, relevant_set, k)

    # NDCG: 理想排序是相关文档在前，不相关在后
    ideal_scores = []
    for i in range(len(reranked_ids)):
        if reranked_ids[i] in relevant_set:
            ideal_scores.append(1.0)
        else:
            ideal_scores.append(0.0)
    ideal_scores.sort(reverse=True)

    predicted_relevance = [1.0 if rid in relevant_set else 0.0 for rid in reranked_ids]
    rerank_metrics["NDCG@10"] = ndcg_at_k(predicted_relevance, ideal_scores, k=10)
    rerank_metrics["NDCG@k"] = ndcg_at_k(predicted_relevance, ideal_scores, k=len(reranked_docs))

    # --- 4. 证据判断评估 ---
    evaluations = []
    try:
        eval_result = evaluate_evidence(question, reranked_docs)
        evaluations = eval_result.get("evaluations", [])
    except Exception as e:
        print(f"  ⚠️ 证据判断跳过（LLM不可用）: {e}")

    evidence_metrics = {}
    if evaluations:
        true_labels = []
        pred_labels = []

        for i, doc in enumerate(reranked_docs):
            # ground truth: 该文档是否相关
            original_idx = reranked_ids[i] if i < len(reranked_ids) else -1
            is_relevant = original_idx in relevant_set
            true_label = "有用" if is_relevant else "无用"

            # 实际判断
            eval_item = next(
                (e for e in evaluations if e.get("id") == i + 1),
                None
            )
            pred_label = eval_item.get("label", "未判断") if eval_item else "未判断"

            true_labels.append(true_label)
            pred_labels.append(pred_label)

        evidence_metrics = classification_metrics(true_labels, pred_labels, positive_label="有用")

    return {
        "question": question,
        "num_retrieved": len(raw_docs),
        "num_relevant": len(relevant_set),
        **retrieval_metrics,
        **rerank_metrics,
        **evidence_metrics,
    }


def main():
    print("=" * 60)
    print("  RAG 检索 & 重排序 & 证据判断 评估")
    print("=" * 60)

    testset = load_testset()
    print(f"\n加载测试集: {len(testset)} 条查询\n")

    all_results = []

    for i, item in enumerate(testset, 1):
        question = item["question"]
        relevant_sources = item.get("relevant_doc_sources", [])
        relevant_doc_ids = item.get("relevant_doc_ids", None)

        print(f"[{i}/{len(testset)}] 评估: {question[:40]}...")

        result = evaluate_single_query(
            question=question,
            relevant_sources=relevant_sources,
            relevant_doc_ids=relevant_doc_ids,
        )

        all_results.append(result)

    # --- 汇总结果 ---
    print("\n" + "=" * 60)
    print("  汇总结果")
    print("=" * 60)

    summary = summarize_metrics(all_results)

    for key, value in summary.items():
        print(f"  {key}: {value}")

    # --- 保存结果 ---
    output_path = Path(__file__).resolve().parent / "results" / "retrieval_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL_PATH,
            "vector_db": settings.VECTOR_DB_PATH,
            "retriever_k": settings.RETRIEVER_K,
            "rerank_model": settings.RERANK_MODEL_PATH,
            "rerank_top_k": settings.RERANK_TOP_K,
        },
        "summary": summary,
        "details": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
