from langchain.tools import tool
from app.vectorstore import get_retriever
from app.evidence_evaluator import evaluate_evidence
from app.reranker import rerank_documents

retriever = get_retriever(use_hybrid=True)


@tool
def search_local_kb(query: str) -> str:
    """
    查询本地知识库，对召回片段重排序，并判断检索片段是否有用。
    """
    docs = retriever.invoke(query)

    if not docs:
        return "知识库中没有找到相关信息。"

    docs = rerank_documents(query, docs)

    evaluation_result = evaluate_evidence(query, docs)

    evaluations = evaluation_result.get("evaluations", [])

    result_list = []
    useful_contents = []

    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "未知文件")
        file_type = doc.metadata.get("file_type", "未知类型")
        page = doc.metadata.get("page", None)
        page_info = f"第 {page + 1} 页" if page is not None else "无页码"
        rerank_score = doc.metadata.get("rerank_score", "无")
        rerank_reason = doc.metadata.get("rerank_reason", "无")

        eval_item = next(
            (item for item in evaluations if item.get("id") == i),
            None
        )

        label = eval_item.get("label", "未判断") if eval_item else "未判断"
        reason = eval_item.get("reason", "无") if eval_item else "无"

        result_list.append(
            f"【检索片段 {i}】\n"
            f"文件名：{file_name}\n"
            f"文件类型：{file_type}\n"
            f"页码：{page_info}\n"
            f"重排序分数：{rerank_score}\n"
            f"重排序原因：{rerank_reason}\n"
            f"有用性判断：{label}\n"
            f"判断原因：{reason}\n"
            f"内容：{doc.page_content}"
        )

        if label in ["有用", "部分有用"]:
            useful_contents.append(
                f"【可用证据 {i}】\n"
                f"来源：{file_name}，{page_info}\n"
                f"内容：{doc.page_content}"
            )

    if not useful_contents:#如果为空 没有任何片段被评估为有用/部分有用
        return (
            "【证据判断结果】\n"
            + "\n\n".join(result_list)
            + "\n\n【结论】\n所有检索片段均无法有效支撑回答，知识库中没有足够信息。"
        )

    return (
        "【证据判断结果】\n"
        + "\n\n".join(result_list)
        + "\n\n【可用于回答的证据】\n"
        + "\n\n".join(useful_contents)
    )

@tool
def calculator(expression: str) -> str:
    """
    用于执行数学表达式计算。

    当用户要求计算加减乘除、括号表达式、小数计算时，必须调用该工具。
    Action Input 必须完整保留用户提供的数学表达式，不要省略任何数字或运算符。

    """
    try:
        expression = expression.replace("（", "(").replace("）", ")")
        expression = expression.replace("×", "*").replace("÷", "/")

        allowed_chars = "0123456789+-*/(). "
        if not all(ch in allowed_chars for ch in expression):
            return f"表达式中包含不允许的字符：{expression}"

        return str(eval(expression))

    except Exception as e:
        return f"计算失败：{e}"


def get_tools():
    tools = [search_local_kb, calculator]

    # 加载社区 MCP Server（优先）
    try:
        from app.mcp_client import get_community_mcp_tools
        community_tools = get_community_mcp_tools()
        tools.extend(community_tools)
        if community_tools:
            print(f"  ✅ 社区 MCP 工具已加载: {[t.name for t in community_tools]}")
    except Exception as e:
        print(f"  ⚠️ 社区 MCP 加载失败，尝试自定义 MCP: {e}")
        # 回退：加载自定义 MCP 工具
        try:
            from app.mcp_tools import get_mcp_tools_for_agent
            mcp_tools = get_mcp_tools_for_agent()
            tools.extend(mcp_tools)
            print(f"  ✅ 自定义 MCP 工具已加载: {[t.name for t in mcp_tools]}")
        except ImportError:
            pass  # MCP 未安装，跳过

    return tools

    return tools
