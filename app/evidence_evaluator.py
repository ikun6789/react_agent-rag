import json
from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm


EVIDENCE_EVALUATOR_TEMPLATE = """
你是一个严格的证据评估器。

你的任务是判断检索片段是否能够支持回答用户问题。

用户问题：
{question}

检索片段：
{docs}

请逐条判断每个片段的有用性。

标签只能是以下三种之一：
- 有用
- 部分有用
- 无用

判断标准：
1. 有用：片段可以直接回答问题，或包含回答所需的关键事实、数据、定义、结论。
2. 部分有用：片段与问题主题相关，但信息不完整，不能单独支撑完整回答。
3. 无用：片段与问题无关，或者不能支持回答。

请严格输出 JSON，不要输出 Markdown，不要输出额外解释。

输出格式如下：

{{
  "evaluations": [
    {{
      "id": 1,
      "label": "有用",
      "reason": "该片段直接说明了问题所需的信息"
    }},
    {{
      "id": 2,
      "label": "无用",
      "reason": "该片段与用户问题无关"
    }}
  ]
}}
"""


def format_docs_for_evaluation(docs):
    result = []

    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "未知文件")
        file_type = doc.metadata.get("file_type", "未知类型")
        page = doc.metadata.get("page", None)
        page_info = f"第 {page + 1} 页" if page is not None else "无页码"

        result.append(
            f"【检索片段 {i}】\n"
            f"片段ID：{i}\n"
            f"文件名：{file_name}\n"
            f"文件类型：{file_type}\n"
            f"页码：{page_info}\n"
            f"内容：{doc.page_content}\n"
        )

    return "\n\n".join(result)


def evaluate_evidence(question, docs):
    """
    判断每个检索片段是否有用。

    返回：
    {
        "evaluations": [
            {"id": 1, "label": "有用", "reason": "..."},
            {"id": 2, "label": "无用", "reason": "..."}
        ]
    }
    """
    llm = get_llm()

    docs_text = format_docs_for_evaluation(docs)

    prompt = ChatPromptTemplate.from_template(EVIDENCE_EVALUATOR_TEMPLATE)

    messages = prompt.format_messages(
        question=question,
        docs=docs_text
    )

    response = llm.invoke(messages)
    content = response.content

    try:
        return json.loads(content)
    except Exception:
        return {
            "evaluations": [],
            "parse_error": content
        }