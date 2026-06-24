"""
MCP Server — 将 RAG 知识库检索和计算器包装为 MCP 工具

可被任何 MCP 客户端调用（VS Code Copilot、Claude Desktop 等）。

用法：
    python app/mcp_rag_server.py          # 启动 MCP Server
"""

import json
import re
from pathlib import Path

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions
import mcp.server.stdio
import mcp.types as types

# ─── 导入项目已有的工具逻辑 ───
# 延迟导入，避免启动时加载所有模型
_search_kb_fn = None
_calc_fn = None


def get_search_kb():
    global _search_kb_fn
    if _search_kb_fn is None:
        from app.vectorstore import get_retriever
        from app.evidence_evaluator import evaluate_evidence
        from app.reranker import rerank_documents

        retriever = get_retriever(use_hybrid=True)

        def _search(query: str) -> str:
            docs = retriever.invoke(query)
            if not docs:
                return "知识库中没有找到相关信息。"
            docs = rerank_documents(query, docs)
            eval_result = evaluate_evidence(query, docs)
            evaluations = eval_result.get("evaluations", [])
            result_list = []
            useful_contents = []
            for i, doc in enumerate(docs, start=1):
                file_name = doc.metadata.get("file_name", "未知文件")
                score = doc.metadata.get("rerank_score", 0)
                label = "未判断"
                for e in evaluations:
                    if e.get("id") == i:
                        label = e.get("label", "未判断")
                        break
                snippet = doc.page_content[:300]
                result_list.append(
                    f"【片段 {i}】来源：{file_name} | 相关性：{score:.3f} | 判断：{label}\n{snippet}"
                )
                if label == "有用":
                    useful_contents.append(f"【证据 {len(useful_contents)+1}】\n{doc.page_content}")
            if not useful_contents:
                return "所有检索片段均无法有效支撑回答。\n\n" + "\n\n".join(result_list)
            return "【检索结果】\n" + "\n\n".join(result_list)

        _search_kb_fn = _search
    return _search_kb_fn


def get_calculator():
    global _calc_fn
    if _calc_fn is None:
        def _calc(expression: str) -> str:
            try:
                expr = expression.replace("（", "(").replace("）", ")").replace("×", "*").replace("÷", "/")
                allowed = "0123456789+-*/(). "
                if not all(ch in allowed for ch in expr):
                    return f"包含不允许的字符"
                return str(eval(expr))
            except Exception as e:
                return f"计算失败：{e}"
        _calc_fn = _calc
    return _calc_fn


# ─── MCP Server ───

app = Server("rag-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_local_kb",
            description="查询本地车联网信息安全知识库，检索相关标准文档内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，如'汽车网关信息安全技术要求'",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="calculator",
            description="执行数学表达式计算",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 3.14*5^2",
                    },
                },
                "required": ["expression"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_local_kb":
        query = arguments.get("query", "")
        result = get_search_kb()(query)
        return [types.TextContent(type="text", text=result)]

    elif name == "calculator":
        expression = arguments.get("expression", "")
        result = get_calculator()(expression)
        return [types.TextContent(type="text", text=result)]

    else:
        raise ValueError(f"未知工具: {name}")


async def run_server():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-server",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(
                        prompts_changed=False,
                        resources_changed=False,
                        tools_changed=False,
                    ),
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())
