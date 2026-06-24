"""
（旧版简单测试 - 已废弃）

请使用 evaluation/ 下的新评估框架：

  # 1. 生成测试集模板
  python evaluation/prepare_testset.py

  # 2. 手动编辑 evaluation/testset.json

  # 3. 运行全流程评估
  python evaluation/evaluate_all.py

新评估框架支持：
- Recall@k, Precision@k, MRR@k
- NDCG@k, MAP
- LLM-as-Judge 答案质量评估
- Faithfulness, Answer Relevance, Context Precision
- 综合评估报告
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  旧版测试已废弃")
    print("=" * 60)
    print()
    print("请使用 evaluation/ 下的新评估框架:")
    print()
    print("  # 1. 生成测试集模板")
    print("  python evaluation/prepare_testset.py")
    print()
    print("  # 2. 手动编辑 evaluation/testset.json")
    print()
    print("  # 3. 运行全流程评估")
    print("  python evaluation/evaluate_all.py")