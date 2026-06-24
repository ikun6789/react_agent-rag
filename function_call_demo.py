"""
Function Calling 机制演示 — 调用真实 LLM（豆包大模型）

流程：
  1. 定义工具（函数 + schema）
  2. 把 schema 发给真实 LLM
  3. LLM 返回：直接回答 / 要调某个工具 + 参数
  4. 代码执行工具 → 结果再发给 LLM
  5. LLM 基于结果生成最终回答
"""

import json
import os
from typing import get_type_hints

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# 第一步：定义工具函数
# ============================================================

def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    database = {
        "北京": "25°C，晴",
        "上海": "28°C，多云",
        "深圳": "30°C，阵雨",
        "广州": "27°C，阴天",
    }
    return database.get(city, f"没有{city}的天气数据")


def calculate(expression: str) -> str:
    """执行数学计算"""
    allowed = "0123456789+-*/(). "
    if not all(c in allowed for c in expression):
        return "包含非法字符"
    return str(eval(expression))


# 工具注册表
TOOLS_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ============================================================
# 第二步：从函数自动生成 OpenAI 格式的 Tool Schema
# ============================================================

def generate_tool_schema(func) -> dict:
    """从函数签名生成 OpenAI 标准的 tool schema"""
    hints = get_type_hints(func)
    doc = func.__doc__ or ""

    properties = {}
    required = []

    for name, typ in hints.items():
        if name == "return":
            continue
        type_map = {str: "string", int: "number", float: "number"}
        properties[name] = {
            "type": type_map.get(typ, "string"),
            "description": f"参数 {name}",
        }
        required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


tools_schema = [generate_tool_schema(fn) for fn in TOOLS_MAP.values()]

print("=" * 70)
print("注册的 Tool Schemas：")
for s in tools_schema:
    print(json.dumps(s, ensure_ascii=False, indent=2))
print("=" * 70)


# ============================================================
# 第三步：Function Calling 主循环
# ============================================================

client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("ARK_BASE_URL"),
)


def function_calling_loop(question: str):
    """一次真实的 Function Calling 完整流程"""
    print(f"\n{'=' * 60}")
    print(f"用户提问：{question}")
    print(f"{'=' * 60}")

    messages = [{"role": "user", "content": question}]

    # ── 第 1 轮：LLM 决定要不要调工具 ──
    print("\n▶ 第1轮：向 LLM 发送问题 + Tools Schema...")

    response = client.chat.completions.create(
        model="doubao-seed-1-8-251228",
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
        temperature=0.1,
    )

    msg = response.choices[0].message

    # 情况 A：LLM 直接回答（没调工具）
    if msg.content and not msg.tool_calls:
        print(f"\n[LLM 直接回答] {msg.content}")
        return msg.content

    # 情况 B：LLM 要调工具
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            print(f"\n[LLM 决定调工具] {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

            # 执行工具
            fn = TOOLS_MAP.get(fn_name)
            if not fn:
                result = f"未知工具: {fn_name}"
            else:
                result = fn(**fn_args)

            print(f"[工具返回] {result}")

            # 把工具调用和结果加入对话
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # ── 第 2 轮：LLM 基于工具结果生成回答 ──
        print("\n▶ 第2轮：把工具结果送回 LLM，生成最终回答...")

        final = client.chat.completions.create(
            model="doubao-seed-1-8-251228",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.1,
        )

        final_answer = final.choices[0].message.content
        print(f"\n[最终回答] {final_answer}")
        return final_answer


# ============================================================
# 运行演示
# ============================================================

if __name__ == "__main__":
    function_calling_loop("北京今天天气怎么样？")
    function_calling_loop("计算 3.14 * 5 + 2 等于多少？")
    function_calling_loop("地球是圆的吗？")
