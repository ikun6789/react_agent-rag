# RAG 评估框架

## 评估维度

| 维度 | 指标 | 说明 |
|------|------|------|
| **检索质量** | Recall@k, Precision@k, MRR@k | 评估向量检索的召回效果 |
| **重排序质量** | NDCG@k, MAP | 评估重排序器是否把有用的结果排到前面 |
| **证据判断质量** | Accuracy, Precision, Recall, F1 | 评估 LLM 判断片段是否有用的准确性 |
| **端到端答案质量** |  faithfulness, answer_relevance, context_precision | LLM-as-Judge 评估答案质量 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `testset.json` | 测试集（你手动标注的问题 + 期望答案 + 相关文档ID） |
| `evaluate_retrieval.py` | 检索和重排序质量评估 |
| `evaluate_answer.py` | 端到端答案质量评估（LLM-as-Judge） |
| `evaluate_all.py` | 一键全流程评估 |
| `metrics.py` | 评估指标实现 |

## 使用方法

### 1. 准备测试集

先准备测试集 `evaluation/testset.json`，格式如下：

```json
[
  {
    "id": 1,
    "question": "汽车网关的信息安全技术要求是什么？",
    "reference_answer": "网关应具备防火墙、入侵检测、安全日志等功能...",
    "relevant_doc_ids": [0, 3, 5],
    "relevant_doc_sources": ["GBT+40857-2021 汽车网关信息安全技术要求及试验方法.md"]
  }
]
```

- `relevant_doc_ids`: 在向量库中与该问题相关的文档块索引（需要先通过检索观察确定）
- `relevant_doc_sources`: 相关的源文件名

### 2. 运行评估

```bash
# 先准备测试集
python evaluation/prepare_testset.py

# 评估检索质量
python evaluation/evaluate_retrieval.py

# 评估端到端答案质量
python evaluation/evaluate_answer.py

# 一键全流程评估
python evaluation/evaluate_all.py
```
