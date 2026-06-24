"""
RAG 评估指标实现

包含指标：
- Retrieval: Recall@k, Precision@k, MRR@k
- Reranking: NDCG@k, MAP
- Classification: Accuracy, Precision, Recall, F1
"""

import math
from typing import List, Set


# ============================================================
# 检索指标
# ============================================================

def recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int = None
) -> float:
    """Recall@k: 前 k 个检索结果中相关文档的比例"""
    if k is None:
        k = len(retrieved_ids)

    top_k = set(retrieved_ids[:k])
    
    if len(relevant_ids) == 0:
        return 0.0
    
    hit = len(top_k & relevant_ids)
    return hit / len(relevant_ids)


def precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int = None
) -> float:
    """Precision@k: 前 k 个检索结果中相关文档的比例"""
    if k is None:
        k = len(retrieved_ids)

    top_k = set(retrieved_ids[:k])
    
    if k == 0:
        return 0.0
    
    hit = len(top_k & relevant_ids)
    return hit / k


def mean_reciprocal_rank(
    retrieved_ids: List[int],
    relevant_ids: Set[int]
) -> float:
    """MRR: 第一个相关文档的排名倒数"""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved_ids: List[int],
    relevant_ids: Set[int]
) -> float:
    """AP: 平均精确率"""
    if len(relevant_ids) == 0:
        return 0.0

    score = 0.0
    hit_count = 0

    for k, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hit_count += 1
            score += hit_count / k

    return score / len(relevant_ids)


# ============================================================
# 重排序指标
# ============================================================

def dcg_at_k(
    scores: List[float],
    k: int = None
) -> float:
    """DCG@k: 折损累计增益"""
    if k is None:
        k = len(scores)

    scores = scores[:k]
    
    if not scores:
        return 0.0
    
    dcg = scores[0]
    
    for i in range(1, len(scores)):
        dcg += scores[i] / math.log2(i + 1)
    
    return dcg


def ndcg_at_k(
    predicted_scores: List[float],
    ideal_scores: List[float],
    k: int = None
) -> float:
    """NDCG@k: 归一化折损累计增益"""
    if k is None:
        k = len(predicted_scores)

    dcg = dcg_at_k(predicted_scores, k)
    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_average_precision(
    all_retrieved: List[List[int]],
    all_relevant: List[Set[int]]
) -> float:
    """MAP: 平均精确率均值"""
    if not all_retrieved:
        return 0.0

    ap_sum = 0.0

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        ap_sum += average_precision(retrieved, relevant)

    return ap_sum / len(all_retrieved)


# ============================================================
# 分类指标（用于证据判断评估）
# ============================================================

def classification_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    positive_label: str = "有用"
) -> dict:
    """计算分类指标：Accuracy, Precision, Recall, F1"""
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels 和 pred_labels 长度不一致")

    total = len(true_labels)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)

    tp = sum(
        1 for t, p in zip(true_labels, pred_labels)
        if p == positive_label and t == positive_label
    )
    fp = sum(
        1 for t, p in zip(true_labels, pred_labels)
        if p == positive_label and t != positive_label
    )
    fn = sum(
        1 for t, p in zip(true_labels, pred_labels)
        if p != positive_label and t == positive_label
    )

    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total": total,
        "correct": correct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 汇总统计
# ============================================================

def summarize_metrics(results: List[dict]) -> dict:
    """汇总多个查询的评估结果"""
    if not results:
        return {}

    summary = {}

    for metric_name in results[0].keys():
        values = [r[metric_name] for r in results if isinstance(r.get(metric_name), (int, float))]
        
        if values:
            summary[f"{metric_name}_mean"] = round(sum(values) / len(values), 4)
            summary[f"{metric_name}_min"] = round(min(values), 4)
            summary[f"{metric_name}_max"] = round(max(values), 4)

    return summary
