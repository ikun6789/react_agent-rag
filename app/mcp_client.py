"""
MCP Client — 连接社区 MCP Server 并包装为 LangChain Tool

用法：安装社区 MCP Server
    npm install -g @modelcontextprotocol/server-filesystem
    # 启动：
    npx @modelcontextprotocol/server-filesystem /home/lyj/Agent/react_agent+rag/data
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional

from langchain.tools import tool


def get_community_mcp_tools():
    """
    连接并获取社区 MCP Server 提供的工具。
    返回 LangChain Tool 列表，可直接加入 Agent。
    """
    tools = []

    # ─── 文件系统 MCP Server ───
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # 定义要启动的 MCP Server（开放整个 /home/lyj 目录）
        project_root = Path(__file__).resolve().parent.parent
        allowed_dirs = [
            str(project_root.parent),  # /home/lyj
        ]
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                *allowed_dirs,
            ],
        )

        # 这里仅展示定义，实际运行时 @tool 会启动连接
        @tool
        def mcp_filesystem_read(path: str) -> str:
            """读取文件内容。输入文件路径（相对 data/ 目录）"""
            import anyio
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            async def _read():
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "read", {"path": path}
                        )
                        return result.content[0].text if result.content else ""

            return anyio.run(_read)

        @tool
        def mcp_filesystem_list(directory: str = ".") -> str:
            """列出目录内容。输入目录路径（相对 data/ 目录）"""
            import anyio
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            async def _list():
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "list_directory", {"path": directory}
                        )
                        return result.content[0].text if result.content else ""

            return anyio.run(_list)

        tools.extend([mcp_filesystem_read, mcp_filesystem_list])
        print("  ✅ 社区 MCP Server (filesystem) 已加载")

    except Exception as e:
        print(f"  ⚠️ 社区 MCP Server 加载失败: {e}")

    return tools
