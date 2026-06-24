"""
端到端答案质量评估（LLM-as-Judge）

使用 LLM 作为裁判，评估 RAG 系统生成答案的质量。

评估维度：
1. faithfulness (忠实性): 答案是否基于检索到的上下文，没有幻觉
2. answer_relevance (答案相关性): 答案是否回答了用户的问题
3. context_precision (上下文精确性): 检索到的上下文是否包含回答问题所需的信息

用法：
    python evaluation/evaluate_answer.py
"""

import json
import sys
from pathlib import Path
from typing import List

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import create_agent_executor
from app.config import settings
from app.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from evaluation.metrics import summarize_metrics


TESTSET_PATH = Path(__file__).resolve().parent / "testset.json"


# ============================================================
# LLM-as-Judge 评判 Prompt
# ============================================================

FAITHFULNESS_JUDGE_TEMPLATE = """
你是一个严格的 AI 评测员。请判断以下回答是否忠实于给定的上下文。

评分标准（1-5分）：
5分：回答完全基于上下文，没有任何编造或幻觉，所有信息都能在上下文中找到依据。
4分：回答基本基于上下文，但存在少量推断或轻微不准确。
3分：回答部分基于上下文，但有一些信息在上下文中找不到依据。
2分：回答大部分内容在上下文中找不到依据，有明显的幻觉。
1分：回答与上下文完全无关，或者完全凭空编造。

注意：如果上下文为空或相关，回答诚实说明"找不到信息"，且未编造，应给5分。

上下文：
{context}

回答：
{answer}

请只输出一个 JSON，格式如下：
{{
  "score": 5,
  "reason": "理由..."
}}
"""

ANSWER_RELEVANCE_JUDGE_TEMPLATE = """
你是一个严格的 AI 评测员。请判断以下回答是否与用户问题相关。

评分标准（1-5分）：
5分：回答直接、完整地回答了用户的问题，没有遗漏关键信息。
4分：回答与问题相关，给出了有用信息，但不够全面或完整。
3分：回答与问题部分相关，但存在不相关的内容或遗漏了重要方面。
2分：回答与问题关系不大，只有极少量相关内容。
1分：回答与问题完全无关。

用户问题：
{question}

回答：
{answer}

请只输出一个 JSON，格式如下：
{{
  "score": 5,
  "reason": "理由..."
}}
"""

CONTEXT_PRECISION_JUDGE_TEMPLATE = """
你是一个严格的 AI 评测员。请判断以下检索到的上下文是否包含回答用户问题所需的信息。

评分标准（1-5分）：
5分：上下文直接包含回答问题的所有必要信息。
4分：上下文包含大部分必要信息，但缺少少量细节。
3分：上下文包含部分相关信息，但不足以完整回答问题。
2分：上下文只有很少的相关信息，大部分信息与问题无关。
1分：上下文完全不包含任何与问题相关的信息。

用户问题：
{question}

上下文：
{context}

请只输出一个 JSON，格式如下：
{{
  "score": 5,
  "reason": "理由..."
}}
"""


def load_testset() -> List[dict]:
    if not TESTSET_PATH.exists():
        print(f"[错误] 测试集文件不存在: {TESTSET_PATH}")
        sys.exit(1)

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_context_from_docs(docs) -> str:
    """将检索到的文档拼接成上下文文本"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "未知")
        parts.append(f"[{i}] 来源: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def llm_judge(prompt_template: str, **kwargs) -> dict:
    """使用 LLM 进行评判"""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.format_messages(**kwargs)
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # 尝试解析 JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", content, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"score": 0, "reason": f"解析失败: {content[:200]}"}


def evaluate_single_answer(
    question: str,
    reference_answer: str = None,
) -> dict:
    """
    评估单个问题的端到端回答质量。
    """
    from app.vectorstore import get_retriever
    from app.reranker import rerank_documents

    # 初始化 Agent
    agent_executor = create_agent_executor(verbose=False)

    # ---- 获取检索上下文 ----
    retriever = get_retriever(use_hybrid=True)
    raw_docs = retriever.invoke(question)
    reranked_docs = rerank_documents(question, raw_docs)
    context = get_context_from_docs(reranked_docs)

    # ---- 生成回答 ----
    try:
        result = agent_executor.invoke({"input": question})
        answer = result.get("output", str(result))
    except Exception as e:
        answer = f"[Agent 执行错误] {e}"

    print(f"  📝 回答: {answer[:300]}")

    # ---- LLM-as-Judge 评估 ----
    print(f"  评估 faithfulness...")
    faithfulness = llm_judge(
        FAITHFULNESS_JUDGE_TEMPLATE,
        context=context,
        answer=answer,
    )

    print(f"  评估 answer_relevance...")
    answer_relevance = llm_judge(
        ANSWER_RELEVANCE_JUDGE_TEMPLATE,
        question=question,
        answer=answer,
    )

    print(f"  评估 context_precision...")
    context_precision = llm_judge(
        CONTEXT_PRECISION_JUDGE_TEMPLATE,
        question=question,
        context=context,
    )

    result = {
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "faithfulness_score": faithfulness.get("score", 0),
        "faithfulness_reason": faithfulness.get("reason", ""),
        "answer_relevance_score": answer_relevance.get("score", 0),
        "answer_relevance_reason": answer_relevance.get("reason", ""),
        "context_precision_score": context_precision.get("score", 0),
        "context_precision_reason": context_precision.get("reason", ""),
        "context_length": len(context),
        "answer_length": len(answer),
    }

    return result


def main():
    print("=" * 60)
    print("  端到端答案质量评估（LLM-as-Judge）")
    print("=" * 60)

    testset = load_testset()
    print(f"\n加载测试集: {len(testset)} 条查询")
    print(f"LLM 模型: {settings.LLM_MODEL}\n")

    # 过滤出有参考答案的或所有问题
    eval_items = [
        item for item in testset
        # 默认评估所有问题，也可以只评估有 reference_answer 的
    ]

    all_results = []

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        reference_answer = item.get("reference_answer", None)

        print(f"[{i}/{len(eval_items)}] {question[:50]}...")

        result = evaluate_single_answer(
            question=question,
            reference_answer=reference_answer,
        )

        all_results.append(result)

        # 打印当前结果
        print(f"  Faithfulness: {result['faithfulness_score']}/5")
        print(f"  Answer Relevance: {result['answer_relevance_score']}/5")
        print(f"  Context Precision: {result['context_precision_score']}/5")
        print()

    # ---- 汇总 ----
    print("=" * 60)
    print("  汇总结果")
    print("=" * 60)

    scores = {
        "faithfulness": [r["faithfulness_score"] for r in all_results],
        "answer_relevance": [r["answer_relevance_score"] for r in all_results],
        "context_precision": [r["context_precision_score"] for r in all_results],
    }

    for name, values in scores.items():
        avg = sum(values) / len(values) if values else 0
        print(f"  {name}: {avg:.2f}/5 (平均)")

    # ---- 保存 ----
    output_path = Path(__file__).resolve().parent / "results" / "answer_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL_PATH,
            "rerank_model": settings.RERANK_MODEL_PATH,
        },
        "summary": {
            "faithfulness_avg": round(sum(scores["faithfulness"]) / len(scores["faithfulness"]), 4),
            "answer_relevance_avg": round(sum(scores["answer_relevance"]) / len(scores["answer_relevance"]), 4),
            "context_precision_avg": round(sum(scores["context_precision"]) / len(scores["context_precision"]), 4),
        },
        "details": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
