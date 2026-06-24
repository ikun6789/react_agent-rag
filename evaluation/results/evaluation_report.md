# RAG 系统综合评估报告

生成时间: 2026-06-09 15:26:43

## 系统配置

| 配置项 | 值 |
|--------|-----|
| Embedding 模型 | `/home/lyj/sentence-transformers/BAAI/bge-small-zh-v1.5` |
| 向量库 | `/home/lyj/Agent/react_agent+rag/data/vector_db/mixed_docs_faiss` |
| 检索 Top-K | `10` |
| 重排序模型 | `/home/lyj/sentence-transformers/BAAI/bge-reranker-v2-m3` |
| 重排序 Top-K | `8` |
| LLM 模型 | `doubao-seed-1-8-251228` |

## 检索 & 重排序质量

| 指标 | 值 |
|------|-----|
| Recall@5 | 0.7 |
| Recall@10 | 1.0 |
| Precision@5 | 0.28 |
| Precision@10 | 0.22 |
| Rerank_Recall@5 | 0.9 |
| Rerank_Precision@5 | 0.4 |
| MRR | 0.6333 |
| AP | 0.5709 |
| NDCG@10 | 0.8865 |
| accuracy | 0.45 |
| f1 | 0.5667 |

## 端到端答案质量（LLM-as-Judge，满分5分）

| 指标 | 均值 |
|------|------|
| 忠实性 (Faithfulness) | 5.0/5 |
| 答案相关性 (Answer Relevance) | 5.0/5 |
| 上下文精确性 (Context Precision) | 5.0/5 |

## 分析与建议

⚠️ **精确率偏低 (Precision@5=0.28)**：建议考虑：
- 加强重排序效果
- 优化文档切分策略