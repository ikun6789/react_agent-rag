from langchain.agents import AgentExecutor, create_react_agent

from app.llm import get_llm
from app.tools import get_tools
from app.memory import get_memory
from app.prompts import get_react_prompt


def create_agent_executor(
    verbose: bool = True,
    session_id: str = "default",
):
    """
    创建 Agent Executor。
    
    Args:
        verbose: 是否输出详细日志
        session_id: 会话 ID，用于区分不同会话的记忆
    """
    llm = get_llm()
    tools = get_tools()
    memory = get_memory(session_id=session_id)
    prompt = get_react_prompt()

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return agent_executor

#result=create_agent_executor(verbose=True).invoke({"input": "请介绍一下你自己？"})
#print(result)