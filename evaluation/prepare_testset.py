"""
测试集准备工具

这个脚本帮助你：
1. 列出向量库中的所有文档来源（文件名）
2. 对每个文档进行采样，展示内容片段
3. 生成 testset.json 模板

用法：
    python evaluation/prepare_testset.py          # 生成模板
    python evaluation/prepare_testset.py --sample # 采样查看文档内容
"""

import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.vectorstore import get_vectorstore


def list_document_sources():
    """列出向量库中所有文档的源文件"""
    vectorstore = get_vectorstore()

    # FAISS 没有直接的 get_all_docs 方法，需要通过检索来探索
    # 这里用一些宽泛的查询来获取文档样本
    source_files = set()
    sample_queries = ["安全", "要求", "技术", "数据", "网关", "通信", "网络", "隐私"]

    for query in sample_queries:
        docs = vectorstore.similarity_search(query, k=50)
        for doc in docs:
            file_name = doc.metadata.get("file_name", "未知")
            source_files.add(file_name)

    print(f"\n向量库中的文档源文件 ({len(source_files)} 个):")
    print("-" * 60)

    for f in sorted(source_files):
        print(f"  {f}")

    return sorted(source_files)


def sample_document_content(num_samples: int = 3):
    """采样查看文档内容片段"""
    vectorstore = get_vectorstore()

    sample_queries = ["安全", "要求", "技术", "数据", "网关", "通信"]

    seen_files = set()
    samples = []

    for query in sample_queries:
        if len(samples) >= num_samples:
            break

        docs = vectorstore.similarity_search(query, k=10)

        for doc in docs:
            file_name = doc.metadata.get("file_name", "未知")
            if file_name not in seen_files:
                seen_files.add(file_name)
                samples.append({
                    "file_name": file_name,
                    "file_type": doc.metadata.get("file_type", "未知"),
                    "content_preview": doc.page_content[:300],
                })

                if len(samples) >= num_samples:
                    break

    print(f"\n文档内容采样 ({len(samples)} 个):")
    print("-" * 60)

    for s in samples:
        print(f"\n文件名: {s['file_name']}")
        print(f"类型: {s['file_type']}")
        print(f"内容预览: {s['content_preview']}...")
        print()

    return samples


def generate_testset_template(source_files):
    """生成测试集 JSON 模板"""
    template = [
        {
            "id": 1,
            "question": "汽车网关的信息安全技术要求有哪些？",
            "reference_answer": "汽车网关应具备防火墙、入侵检测、安全日志、安全启动等功能...",
            "relevant_doc_sources": [
                "GBT+40857-2021 汽车网关信息安全技术要求及试验方法.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 2,
            "question": "ISO 21434 标准对道路车辆网络安全工程的要求是什么？",
            "reference_answer": "ISO 21434 要求建立网络安全管理系统...",
            "relevant_doc_sources": [
                "ISO21434-2021原版.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 3,
            "question": "R155 法规对车辆网络安全认证的要求是什么？",
            "reference_answer": "R155 要求车辆类型批准需要满足网络安全管理系统...",
            "relevant_doc_sources": [
                "R155.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 4,
            "question": "《中华人民共和国数据安全法》对数据分类分级保护的要求是什么？",
            "reference_answer": "数据安全法要求建立数据分类分级保护制度...",
            "relevant_doc_sources": [
                "【法规】中华人民共和国数据安全法.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 5,
            "question": "电动汽车充电系统的信息安全技术要求是什么？",
            "reference_answer": "充电系统应具备身份认证、加密通信、安全监测等能力...",
            "relevant_doc_sources": [
                "GBT+41578-2022 电动汽车充电系统信息安全技术要求及试验方法.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 6,
            "question": "车联网服务平台需要满足哪些网络安全技术要求？",
            "reference_answer": "车联网服务平台应具备安全认证、访问控制、日志审计...",
            "relevant_doc_sources": [
                "DB31T 1561-2025【上海】车联网服务平台网络安全技术要求.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 7,
            "question": "个人信息保护法对汽车数据处理有哪些要求？",
            "reference_answer": "个人信息保护法要求告知同意、最小必要、目的限制...",
            "relevant_doc_sources": [
                "【法规】中华人民共和国个人信息保护法.md",
                "GBT+41871-2022 信息安全技术 汽车数据处理安全要求.md"
            ],
            "relevant_doc_ids": []
        },
        {
            "id": 8,
            "question": "智能网联汽车整车信息安全技术要求有哪些？",
            "reference_answer": "整车信息安全要求包括安全启动、安全通信、安全升级...",
            "relevant_doc_sources": [
                "DB4403T355-2023【深圳】智能网联汽车整车信息安全技术要求.md",
                "GB+44495-2024 汽车整车信息安全技术要求（即将实施）.md"
            ],
            "relevant_doc_ids": []
        },
    ]

    return template


def main():
    parser = argparse.ArgumentParser(description="RAG 测试集准备工具")
    parser.add_argument("--sample", action="store_true", help="采样查看文档内容")
    args = parser.parse_args()

    print("=" * 60)
    print("  测试集准备工具")
    print("=" * 60)

    source_files = list_document_sources()

    if args.sample:
        sample_document_content()

    output_path = Path(__file__).resolve().parent / "testset.json"

    if output_path.exists():
        overwrite = input(f"\n{output_path.name} 已存在，是否覆盖？(y/n): ").strip().lower()
        if overwrite != "y":
            print("已取消")
            return

    template = generate_testset_template(source_files)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"\n测试集模板已生成: {output_path}")
    print(f"共 {len(template)} 条测试问题")
    print("\n⚠️  重要提示：")
    print("  1. 请手动检查并修改 testset.json 中的问题")
    print("  2. relevant_doc_sources 已根据文件名预设，但需要验证准确性")
    print("  3. 如果有精确的 ground truth doc IDs，请填入 relevant_doc_ids")
    print("  4. reference_answer 只是模板占位，请根据标准答案修改")


if __name__ == "__main__":
    main()
