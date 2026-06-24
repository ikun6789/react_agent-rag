from datetime import date

from app.agent import create_agent_executor
from app.logger import logger


def get_default_session_id() -> str:
    """
    获取默认会话 ID。
    按天划分会话，同一天的对话属于同一个会话。
    """
    return f"session_{date.today().isoformat()}"


def run_cli(session_id: str = None):
    """
    启动 CLI。
    
    Args:
        session_id: 会话 ID。不传则按日期自动生成。
    """
    if session_id is None:
        session_id = get_default_session_id()

    agent_executor = create_agent_executor(verbose=True, session_id=session_id)

    print(f"🧠 ReAct + 本地知识库 + 长期记忆 Agent 已启动")
    print(f"📋 会话 ID: {session_id}")
    print(f"💡 输入 exit / quit / q / 退出 可以结束程序")
    print(f"💡 输入 /new 开启新会话")
    print(f"💡 输入 /session <名称> 切换到指定会话")
    print("=" * 60)

    while True:
        raw_input = input("请提问：").strip()

        # ── 命令处理 ──────────────────────────────────────
        if raw_input.lower() in ["exit", "quit", "q", "退出"]:
            print("👋 已退出")
            logger.info("用户退出程序")
            break

        if raw_input == "/new":
            session_id = get_default_session_id()
            agent_executor = create_agent_executor(verbose=True, session_id=session_id)
            print(f"📋 已切换到新会话: {session_id}")
            continue

        if raw_input.startswith("/session "):
            session_id = raw_input[len("/session "):].strip()
            agent_executor = create_agent_executor(verbose=True, session_id=session_id)
            print(f"📋 已切换到会话: {session_id}")
            continue

        # ── 普通对话 ──────────────────────────────────────
        if not raw_input:
            continue

        try:
            logger.info(f"用户输入：{raw_input}")

            result = agent_executor.invoke(
                {"input": raw_input}
            )

            output = result["output"]

            print("\n🤖 Agent 回答：")
            print(output)
            logger.info(f"Agent 输出：{output}")

            print("=" * 60)

        except Exception as e:
            print("\n❌ 程序运行出错：")
            print(e)
            logger.exception(f"程序运行出错，用户输入为：{raw_input}")
            print("=" * 60)