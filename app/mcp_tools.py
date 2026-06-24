"""
MCP 工具集成

通过 MCP 协议标准化连接外部工具和数据源。
当前提供：文件搜索、网页抓取工具。

用法：
    python app/mcp_tools.py          # 独立运行 MCP Server
"""

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

import os
import json
from pathlib import Path


# ============================================================
# MCP Server 定义
# ============================================================

app = Server("rag-knowledge-tools")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """注册可用工具"""
    return [
        types.Tool(
            name="search_files",
            description="在本地文件系统中搜索文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "文件名或路径模式（glob格式，如 **/*.md）",
                    },
                    "root_dir": {
                        "type": "string",
                        "description": "搜索根目录，默认为项目 data/raw",
                    },
                },
                "required": ["pattern"],
            },
        ),
        types.Tool(
            name="fetch_webpage",
            description="获取网页内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "网页URL",
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """执行工具调用"""
    if name == "search_files":
        pattern = arguments.get("pattern", "")
        root_dir = arguments.get("root_dir", "")

        if not root_dir:
            root_dir = str(Path(__file__).resolve().parent.parent / "data" / "raw")

        search_path = Path(root_dir)
        if not search_path.exists():
            return [types.TextContent(type="text", text=f"目录不存在: {root_dir}")]

        results = list(search_path.rglob(pattern))
        if len(results) > 20:
            results = results[:20]
            summary = f"找到超过20个文件，显示前20个:\n"
        else:
            summary = f"找到 {len(results)} 个文件:\n"

        for r in results:
            size = r.stat().st_size
            summary += f"  - {r.name} ({size/1024:.1f}KB)\n"

        return [types.TextContent(type="text", text=summary)]

    elif name == "fetch_webpage":
        url = arguments.get("url", "")
        import urllib.request

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                # 简单提取文本（去掉HTML标签）
                import re
                text = re.sub(r"<[^>]+>", "", content)
                text = re.sub(r"\s+", " ", text).strip()[:3000]
                return [types.TextContent(type="text", text=text[:3000])]
        except Exception as e:
            return [types.TextContent(type="text", text=f"抓取失败: {e}")]

    else:
        return [types.TextContent(type="text", text=f"未知工具: {name}")]


# ============================================================
# 在 LangChain Agent 中使用 MCP 工具
# ============================================================

def get_mcp_tools_for_agent():
    """
    返回 MCP 工具列表，让 Agent 可以调用。
    用法：
        from app.mcp_tools import get_mcp_tools_for_agent
        from app.tools import get_tools

        all_tools = get_tools() + get_mcp_tools_for_agent()
    """
    from langchain.tools import tool

    @tool
    def mcp_search_files(pattern: str, root_dir: str = "") -> str:
        """搜索本地文件。输入 pattern 如 '**/*.md'，root_dir 可选"""
        import asyncio
        result = asyncio.run(call_tool("search_files", {
            "pattern": pattern,
            "root_dir": root_dir,
        }))
        return result[0].text if result else "未找到"

    @tool
    def mcp_fetch_webpage(url: str) -> str:
        """获取网页文本内容。输入 url"""
        import asyncio
        result = asyncio.run(call_tool("fetch_webpage", {"url": url}))
        return result[0].text if result else "抓取失败"

    return [mcp_search_files, mcp_fetch_webpage]


# ============================================================
# 独立运行 MCP Server
# ============================================================

async def run_server():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-knowledge-tools",
                server_version="0.1.0",
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())
