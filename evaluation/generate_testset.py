"""
测试集自动生成工具

读取 data/raw/ 目录下的所有文档，使用 LLM 自动生成带参考答案的测试题，
覆盖所有标准、法规和技术规范。

用法：
    python evaluation/generate_testset.py                    # 使用 LLM 生成（需要 API）
    python evaluation/generate_testset.py --dry-run          # 只列出文档分类，不生成
    python evaluation/generate_testset.py --limit 10         # 只处理前 N 个文档
    python evaluation/generate_testset.py --category 法规     # 只处理特定类别
"""

import json
import os
import re
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = Path(__file__).resolve().parent / "testset.json"
BATCH_SIZE = 5  # 每批处理文档数


def categorize_documents() -> Dict[str, List[Dict]]:
    """扫描 data/raw/ 目录，按类别分组所有文档"""
    if not RAW_DATA_DIR.exists():
        print(f"❌ 目录不存在: {RAW_DATA_DIR}")
        return {}

    all_files = sorted(RAW_DATA_DIR.glob("*.md"))
    categories = {
        "国标_GB": [],
        "国标_GBT": [],
        "国际标准_ISO": [],
        "国际法规_UN_R": [],
        "地方标准_DB": [],
        "团体标准_T": [],
        "行业标准_YD": [],
        "法规_Regulation": [],
        "其他": [],
    }

    for f in all_files:
        name = f.name
        # 去掉前面的序号前缀（如 "1. ", "10. "）
        clean_name = re.sub(r'^\d+\.\s*', '', name)
        doc_info = {
            "file_name": name,
            "clean_name": clean_name,
            "file_path": str(f),
            "file_size": f.stat().st_size,
        }

        if clean_name.startswith(("GB+", "GB ")):
            categories["国标_GB"].append(doc_info)
        elif clean_name.startswith(("GBT", "GB/T")):
            categories["国标_GBT"].append(doc_info)
        elif "ISO" in clean_name:
            categories["国际标准_ISO"].append(doc_info)
        elif clean_name.startswith("R1") or clean_name.startswith("R2"):
            categories["国际法规_UN_R"].append(doc_info)
        elif clean_name.startswith("DB"):
            categories["地方标准_DB"].append(doc_info)
        elif clean_name.startswith(("TBJQC", "TCAAMTB", "TCITSA", "TCSAE", "TCTS", "TGHDQ", "THEVCA", "TIMAS", "TITS", "TJQR", "TNBQLX", "TSHV2X", "TSXQCTB")):
            categories["团体标准_T"].append(doc_info)
        elif clean_name.startswith("YD"):
            categories["行业标准_YD"].append(doc_info)
        elif "法规" in name:
            categories["法规_Regulation"].append(doc_info)
        else:
            categories["其他"].append(doc_info)

    # 移除空类别
    return {k: v for k, v in categories.items() if v}


def print_categories(categories: Dict[str, List[Dict]]):
    """打印文档分类统计"""
    total = sum(len(v) for v in categories.values())
    print(f"\n{'='*60}")
    print(f"  文档分类统计（共 {total} 个文档）")
    print(f"{'='*60}")

    for cat_name, docs in categories.items():
        print(f"\n  【{cat_name}】({len(docs)} 个)")
        for d in docs:
            size_kb = d["file_size"] / 1024
            print(f"    - {d['file_name']} ({size_kb:.1f}KB)")

    print(f"\n{'='*60}\n")


def read_document_content(file_path: str, max_chars: int = 4000) -> str:
    """读取文档内容，提取关键部分（目录、范围、术语、技术要求等）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"[读取失败: {e}]"

    if len(content) <= max_chars:
        return content

    # 提取关键部分：前部 + 技术要求/安全要求/主要内容
    lines = content.split("\n")
    key_sections = []
    collected = 0
    in_key_section = False

    # 先取前 30% 的内容（包含标题、范围等）
    head = content[:max_chars // 2]
    
    # 再找关键词段落
    keywords = [
        "技术要求", "安全要求", "数据安全", "信息安全", "网络安全",
        "保护要求", "管理规定", "总则", "基本要求", "功能要求",
        "shall", "should", "必须", "应具备", "应符合", "应满足",
        "第", "条", "章"
    ]
    
    tail_start = max(len(content) // 3, max_chars // 2)
    tail = content[tail_start:tail_start + max_chars // 2]
    
    return head + "\n\n... [中间内容省略] ...\n\n" + tail


def build_category_prompt(cat_name: str, docs: List[Dict]) -> str:
    """为每个类别构建 LLM prompt"""
    doc_descriptions = []
    for d in docs[:BATCH_SIZE]:
        content = read_document_content(d["file_path"])
        doc_descriptions.append(f"文档: {d['file_name']}\n内容:\n{content[:2000]}\n")

    docs_text = "\n---\n".join(doc_descriptions)

    prompt = f"""你是一位车辆网络安全领域的专家。请根据以下文档内容，为每个文档生成 1-3 个高质量的测试题。

类别: {cat_name}

文档列表:
{docs_text}

要求:
1. 每个测试题必须包含:
   - "question": 具体、清晰的问题
   - "reference_answer": 详细的参考答案（基于文档内容）
   - "relevant_doc_sources": 引用的文档文件名列表

2. 问题应覆盖文档中的关键技术要求、安全规范、合规条款等

3. 答案要准确、完整，引用文档中的具体条款

4. 以 JSON 数组格式输出，不要有其他文字

输出格式示例:
[
  {{
    "question": "汽车网关的信息安全技术要求有哪些？",
    "reference_answer": "根据GB/T 40857-2021，汽车网关应满足硬件安全、通信安全、固件安全和数据安全等要求...",
    "relevant_doc_sources": ["GBT+40857-2021 汽车网关信息安全技术要求及试验方法.md"]
  }}
]
"""
    return prompt


def generate_with_llm(prompt: str) -> Optional[List[Dict]]:
    """调用 LLM 生成测试题"""
    try:
        from langchain_openai import ChatOpenAI
        from app.config import settings

        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.ARK_API_KEY,
            base_url=settings.ARK_BASE_URL,
            temperature=0.1,
        )

        response = llm.invoke(prompt)
        text = response.content.strip()

        # 提取 JSON
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            text = json_match.group()

        questions = json.loads(text)
        if isinstance(questions, list):
            return questions
        return None
    except Exception as e:
        print(f"  ⚠️ LLM 调用失败: {e}")
        return None


def generate_questions_for_all_docs(
    categories: Dict[str, List[Dict]],
    limit: Optional[int] = None,
    target_category: Optional[str] = None
) -> List[Dict]:
    """为所有文档生成测试题"""
    all_questions = []
    question_id = 1

    # 先读取已有的 testset.json 保留原有问题
    existing_ids = set()
    existing_questions = []
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                for q in existing:
                    existing_ids.add(q.get("id", 0))
                    existing_questions.append(q)
                if existing_questions:
                    question_id = max(existing_ids) + 1
                    all_questions.extend(existing_questions)
                    print(f"📂 已加载 {len(existing_questions)} 条现有测试题")
        except:
            pass

    processed_count = 0
    for cat_name, docs in categories.items():
        if target_category and target_category not in cat_name:
            continue

        cat_label = cat_name.split("_")[-1] if "_" in cat_name else cat_name
        print(f"\n📋 处理类别【{cat_label}】({len(docs)} 个文档)...")

        # 每个类别分批处理
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            
            if limit and processed_count >= limit:
                break

            print(f"  批次 {i // BATCH_SIZE + 1}/{(len(docs) + BATCH_SIZE - 1) // BATCH_SIZE}: ", end="")
            
            prompt = build_category_prompt(cat_name, batch)
            result = generate_with_llm(prompt)

            if result:
                for q in result:
                    q["id"] = question_id
                    question_id += 1
                    all_questions.append(q)
                    processed_count += 1
                print(f"✅ 生成 {len(result)} 题")
            else:
                print("❌ 生成失败")

            time.sleep(1)  # 避免 API 限流

        if limit and processed_count >= limit:
            break

    return all_questions


def generate_questions_deterministic(
    categories: Dict[str, List[Dict]],
    limit: Optional[int] = None,
    target_category: Optional[str] = None
) -> List[Dict]:
    """
    基于文档名和类别，确定性生成测试题（不依赖 LLM）
    适合快速生成覆盖全面的测试集
    """
    question_templates = {
        "国标_GB": [
            "{title} 标准对汽车{field}提出了哪些核心安全要求？",
            "根据{title}，{field}需要满足哪些技术指标？",
        ],
        "国标_GBT": [
            "{title} 标准规定了哪些{field}技术要求？",
            "根据{title}标准，{field}应满足哪些安全规范？",
            "{title}中对{field}的试验方法有哪些？",
        ],
        "国际标准_ISO": [
            "{title} 标准对{field}的网络安全工程提出了哪些要求？",
            "ISO 21434 中对{field}风险管理的要求是什么？",
        ],
        "国际法规_UN_R": [
            "{title} 法规对车辆{field}认证有哪些具体要求？",
            "根据{title}，车辆制造商在{field}方面需要满足哪些条件？",
        ],
        "地方标准_DB": [
            "{title} 规定了哪些{field}技术要求？",
            "根据{title}，{field}应如何保障网络安全？",
        ],
        "团体标准_T": [
            "{title} 对{field}提出了哪些技术要求？",
            "根据{title}，{field}需要满足哪些安全测试规范？",
        ],
        "行业标准_YD": [
            "{title} 对车载{field}的网络安全提出了哪些要求？",
            "根据{title}标准，{field}需要满足哪些技术指标？",
        ],
        "法规_Regulation": [
            "《{title}》对数据{field}提出了哪些合规要求？",
            "根据《{title}》，企业在{field}方面需要履行哪些义务？",
            "《{title}》中关于{field}的核心条款有哪些？",
        ],
        "其他": [
            "{title} 对{field}提出了哪些要求？",
        ],
    }

    field_keywords = {
        "国标_GB": ["信息安全", "网络安全", "整车", "通信", "数据"],
        "国标_GBT": ["信息安全", "网络安全", "数据安全", "网关", "通信", "远程服务", "交互系统", "充电系统", "数据处理", "应急响应", "诊断接口"],
        "国际标准_ISO": ["网络安全", "软件更新", "审计", "工程"],
        "国际法规_UN_R": ["网络安全", "软件更新", "认证", "管理系统"],
        "地方标准_DB": ["整车", "数据安全", "网络安全", "服务平台"],
        "团体标准_T": ["控制器", "网关", "通信", "测试", "远程诊断", "升级", "密钥管理", "风险分析", "入侵检测"],
        "行业标准_YD": ["终端", "监测", "通信"],
        "法规_Regulation": ["安全", "保护", "管理", "跨境", "准入", "测试", "运营"],
        "其他": ["安全", "技术"],
    }

    answer_templates = {
        "信息安全": "应包括身份认证、访问控制、数据加密、安全审计、入侵检测等基本安全能力，确保系统和数据的机密性、完整性和可用性。",
        "网络安全": "应建立网络安全防护体系，包括边界防护、访问控制、入侵检测、安全审计、漏洞管理等方面，确保网络环境的安全性。",
        "数据安全": "应建立数据分类分级保护制度，对重要数据和核心数据实施重点保护，明确数据安全负责人，建立健全全流程数据安全管理制度。",
        "技术要求": "应符合相关标准中规定的技术指标，包括功能要求、性能要求、安全要求等，并通过相应的试验方法进行验证。",
    }

    all_questions = []
    question_id = 1

    # 加载现有问题
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                for q in existing:
                    q_id = q.get("id", 0)
                    if q_id >= question_id:
                        question_id = q_id + 1
                    all_questions.append(q)
                print(f"📂 已加载 {len(existing)} 条现有测试题")
        except:
            pass

    for cat_name, docs in categories.items():
        if target_category and target_category not in cat_name:
            continue

        templates = question_templates.get(cat_name, question_templates["其他"])
        fields = field_keywords.get(cat_name, field_keywords["其他"])

        cat_label = cat_name.split("_")[-1] if "_" in cat_name else cat_name
        print(f"\n📋 处理类别【{cat_label}】({len(docs)} 个文档)...")

        for doc in docs:
            if limit and (len(all_questions) - (len(existing) if OUTPUT_PATH.exists() else 0)) >= limit:
                break

            name = doc["file_name"]
            clean_name = doc.get("clean_name", name)
            # 提取文档标题（去掉编号和扩展名）
            title = clean_name.replace('.md', '')
            # 提取标准名称的核心部分
            std_name = title.split('】')[-1] if '】' in title else title
            # 精简标题
            short_title = std_name[:60] if len(std_name) > 60 else std_name

            # 为每个文档生成 1-2 个问题
            num_questions = min(len(fields), 2)
            for j in range(num_questions):
                field = fields[j % len(fields)]
                template = templates[j % len(templates)]

                question = template.format(
                    title=short_title,
                    field=field
                )

                answer_base = answer_templates.get(field, answer_templates["技术要求"])

                question_obj = {
                    "id": question_id,
                    "question": question,
                    "reference_answer": f"根据{short_title}的要求，{answer_base}",
                    "relevant_doc_sources": [name],
                    "relevant_doc_ids": []
                }
                all_questions.append(question_obj)
                question_id += 1

            print(f"  ✅ {name[:50]}...")

        if limit and (len(all_questions) - (len(existing) if OUTPUT_PATH.exists() else 0)) >= limit:
            break

    return all_questions


def main():
    parser = argparse.ArgumentParser(description="RAG 测试集自动生成工具")
    parser.add_argument("--dry-run", action="store_true", help="只列出文档分类，不生成")
    parser.add_argument("--limit", type=int, default=None, help="限制生成的题目数量")
    parser.add_argument("--category", type=str, default=None, help="只处理特定类别（如 法规、团体标准）")
    parser.add_argument("--use-llm", action="store_true", help="使用 LLM 生成（精确但耗时）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖现有 testset.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 测试集自动生成工具")
    print("=" * 60)

    # 1. 扫描文档
    categories = categorize_documents()
    if not categories:
        print("❌ 未找到文档")
        return

    print_categories(categories)

    if args.dry_run:
        print("🔍 Dry-run 模式，不生成测试集")
        return

    # 2. 生成测试题
    print("\n🚀 开始生成测试题...\n")

    if args.use_llm:
        questions = generate_questions_for_all_docs(
            categories, limit=args.limit, target_category=args.category
        )
    else:
        questions = generate_questions_deterministic(
            categories, limit=args.limit, target_category=args.category
        )

    if not questions:
        print("❌ 未生成任何测试题")
        return

    # 3. 保存
    if args.overwrite:
        # 只保留新生成的
        output = [q for q in questions if q.get("id", 0) > 0]
    else:
        output = questions

    # 重新编号
    for i, q in enumerate(output, 1):
        q["id"] = i

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ 测试集已生成: {OUTPUT_PATH}")
    print(f"  📊 共 {len(output)} 条测试题")
    print(f"{'='*60}")
    print(f"\n💡 提示:")
    print(f"  - 使用 --use-llm 调用 LLM 生成更精确的问题和答案")
    print(f"  - 使用 --category '法规' 只处理特定类别")
    print(f"  - 使用 --limit 10 限制生成数量")
    print(f"  - 生成后建议手动检查并修改答案的准确性")


if __name__ == "__main__":
    main()
