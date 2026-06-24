"""
一键全流程 RAG 评估

依次运行：
1. 检索质量评估 (evaluate_retrieval.py)
2. 端到端答案质量评估 (evaluate_answer.py)
3. 生成综合评估报告

用法：
    python evaluation/evaluate_all.py
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


def run_retrieval_eval():
    """运行检索评估"""
    print("\n" + "=" * 60)
    print("  【阶段 1/2】检索 & 重排序 & 证据判断评估")
    print("=" * 60)

    from evaluation.evaluate_retrieval import evaluate_single_query, load_testset

    testset = load_testset()[:5]  # 先测前5题
    all_results = []

    for i, item in enumerate(testset, 1):
        question = item["question"]
        relevant_sources = item.get("relevant_doc_sources", [])
        relevant_doc_ids = item.get("relevant_doc_ids", None)

        print(f"\n[{i}/{len(testset)}] {question[:50]}")

        result = evaluate_single_query(
            question=question,
            relevant_sources=relevant_sources,
            relevant_doc_ids=relevant_doc_ids,
        )
        all_results.append(result)

        # 显示检索评估结果（原始检索 + 重排序后）
        recall_10 = result.get("Recall@10", "?")
        precision_5 = result.get("Precision@5", "?")
        mrr = result.get("MRR", "?")
        rerank_p5 = result.get("Rerank_Precision@5", "?")
        ndcg = result.get("NDCG@10", "?")
        print(f"  📊 Recall@10={recall_10}  Precision@5={precision_5}  MRR={mrr}")
        print(f"  🔄 重排序后: Precision@5={rerank_p5}  NDCG@10={ndcg}")

    return all_results


def run_answer_eval():
    """运行端到端答案质量评估"""
    print("\n" + "=" * 60)
    print("  【阶段 2/2】端到端答案质量评估（LLM-as-Judge）")
    print("=" * 60)

    from evaluation.evaluate_answer import (
        evaluate_single_answer,
        load_testset,
    )

    testset = load_testset()[:5]  # 先测前5题
    all_results = []

    for i, item in enumerate(testset, 1):
        question = item["question"]
        reference_answer = item.get("reference_answer", None)

        print(f"\n[{i}/{len(testset)}] {question[:50]}")

        result = evaluate_single_answer(
            question=question,
            reference_answer=reference_answer,
        )
        all_results.append(result)

        # 显示回答内容和评分
        print(f"  📝 回答: {result.get('answer', '')[:200]}...")
        print(f"  ⭐ Faithfulness: {result.get('faithfulness_score', '?')}/5")
        print(f"  ⭐ Answer Relevance: {result.get('answer_relevance_score', '?')}/5")
        print(f"  ⭐ Context Precision: {result.get('context_precision_score', '?')}/5")

    return all_results


def generate_report(retrieval_results, answer_results):
    """生成综合评估报告"""
    print("\n" + "=" * 60)
    print("  生成综合评估报告")
    print("=" * 60)

    # 计算检索指标均值
    retrieval_summary = {}
    if retrieval_results:
        metrics_of_interest = [
            "Recall@5", "Recall@10", "Recall@30",
            "Precision@5", "Precision@10",
            "Rerank_Recall@5", "Rerank_Precision@5",
            "MRR", "AP",
            "NDCG@10",
            "accuracy", "f1",
        ]

        for metric in metrics_of_interest:
            values = [
                r[metric] for r in retrieval_results
                if isinstance(r.get(metric), (int, float))
            ]
            if values:
                retrieval_summary[metric] = round(sum(values) / len(values), 4)

    # 计算答案指标均值
    answer_summary = {}
    if answer_results:
        for metric in ["faithfulness_score", "answer_relevance_score", "context_precision_score"]:
            values = [r[metric] for r in answer_results if isinstance(r.get(metric), (int, float))]
            if values:
                answer_summary[metric] = round(sum(values) / len(values), 4)

    # 生成报告文本
    report_lines = []
    report_lines.append("# RAG 系统综合评估报告")
    report_lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n## 系统配置")
    report_lines.append(f"\n| 配置项 | 值 |")
    report_lines.append(f"|--------|-----|")
    report_lines.append(f"| Embedding 模型 | `{settings.EMBEDDING_MODEL_PATH}` |")
    report_lines.append(f"| 向量库 | `{settings.VECTOR_DB_PATH}` |")
    report_lines.append(f"| 检索 Top-K | `{settings.RETRIEVER_K}` |")
    report_lines.append(f"| 重排序模型 | `{settings.RERANK_MODEL_PATH}` |")
    report_lines.append(f"| 重排序 Top-K | `{settings.RERANK_TOP_K}` |")
    report_lines.append(f"| LLM 模型 | `{settings.LLM_MODEL}` |")

    if retrieval_summary:
        report_lines.append(f"\n## 检索 & 重排序质量")
        report_lines.append(f"\n| 指标 | 值 |")
        report_lines.append(f"|------|-----|")
        for k, v in retrieval_summary.items():
            report_lines.append(f"| {k} | {v} |")

    if answer_summary:
        report_lines.append(f"\n## 端到端答案质量（LLM-as-Judge，满分5分）")
        report_lines.append(f"\n| 指标 | 均值 |")
        report_lines.append(f"|------|------|")
        name_map = {
            "faithfulness_score": "忠实性 (Faithfulness)",
            "answer_relevance_score": "答案相关性 (Answer Relevance)",
            "context_precision_score": "上下文精确性 (Context Precision)",
        }
        for k, v in answer_summary.items():
            display_name = name_map.get(k, k)
            report_lines.append(f"| {display_name} | {v}/5 |")

    # 改进建议
    report_lines.append(f"\n## 分析与建议")

    if retrieval_summary:
        recall_10 = retrieval_summary.get("Recall@10", 1)
        if recall_10 < 0.5:
            report_lines.append(f"\n⚠️ **召回率偏低 (Recall@10={recall_10})**：建议考虑：")
            report_lines.append(f"- 更换更强的 Embedding 模型")
            report_lines.append(f"- 调整 chunk_size 或 chunk_overlap")
            report_lines.append(f"- 增加检索 Top-K 值")

        precision_5 = retrieval_summary.get("Precision@5", 0)
        if precision_5 < 0.3:
            report_lines.append(f"\n⚠️ **精确率偏低 (Precision@5={precision_5})**：建议考虑：")
            report_lines.append(f"- 加强重排序效果")
            report_lines.append(f"- 优化文档切分策略")

    if answer_summary:
        faithfulness = answer_summary.get("faithfulness_score", 5)
        if faithfulness < 3:
            report_lines.append(f"\n⚠️ **忠实性偏低 (Faithfulness={faithfulness}/5)**：建议考虑：")
            report_lines.append(f"- 在 Prompt 中更严格约束 LLM 基于检索结果回答")
            report_lines.append(f"- 检查是否 LLM 忽略检索结果自行编造")

    # 保存报告
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n综合报告已保存: {report_path}")
    print("\n".join(report_lines))


def main():
    start_time = time.time()

    print("=" * 60)
    print("  RAG 系统全流程评估")
    print(f"  测试集: evaluation/testset.json")
    print("=" * 60)

    # 检查测试集
    testset_path = Path(__file__).resolve().parent / "testset.json"
    if not testset_path.exists():
        print("\n[错误] 测试集不存在！请先运行:")
        print("  python evaluation/prepare_testset.py")
        print("  然后手动编辑 evaluation/testset.json")
        sys.exit(1)

    # 阶段 1：检索评估
    retrieval_results = run_retrieval_eval()

    # 阶段 2：答案评估
    answer_results = run_answer_eval()

    # 生成报告
    generate_report(retrieval_results, answer_results)

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()
