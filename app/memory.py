from langchain.memory import ConversationBufferMemory
from langchain_core.memory import BaseMemory
from pydantic import PrivateAttr

from app.long_term_memory import LongTermMemory


class HybridMemory(BaseMemory):
    """
    混合记忆：短期（ConversationBufferMemory）+ 长期（SQLite 持久化）。

    工作流：
      1. 启动时从 SQLite 加载历史对话填充到 Buffer
      2. 每次交互后自动保存到 SQLite
      3. 当历史对话超过阈值时，自动摘要旧对话
    """

    session_id: str = "default"
    recent_turns: int = 20
    summarization_threshold: int = 50

    # Pydantic 私有字段（不参与序列化/验证）
    _ltm: LongTermMemory = PrivateAttr()
    _buffer: ConversationBufferMemory = PrivateAttr()

    def __init__(
        self,
        session_id: str = "default",
        recent_turns: int = 20,
        summarization_threshold: int = 50,
    ):
        super().__init__(
            session_id=session_id,
            recent_turns=recent_turns,
            summarization_threshold=summarization_threshold,
        )

        # 长期记忆（SQLite）
        self._ltm = LongTermMemory(
            session_id=session_id,
            recent_turns=recent_turns,
            summarization_threshold=summarization_threshold,
        )

        # 短期记忆（Buffer）
        self._buffer = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,
        )

        # 启动时加载历史
        self._load_history()

    @property
    def memory_variables(self) -> list[str]:
        """BaseMemory 必需属性：返回此记忆管理的变量名列表"""
        return self._buffer.memory_variables

    def _load_history(self):
        """从 SQLite 加载最近对话到 Buffer"""
        recent = self._ltm.load_recent_messages(self.recent_turns)
        if not recent:
            return

        self._buffer.chat_memory.messages = []
        for role, content in recent:
            if role == "user":
                self._buffer.chat_memory.add_user_message(content)
            else:
                self._buffer.chat_memory.add_ai_message(content)

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """
        保存一轮对话到 Buffer + SQLite。
        兼容 LangChain AgentExecutor 的调用方式：
            memory.save_context({"input": ...}, {"output": ...})
        """
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")

        # 保存到 Buffer
        self._buffer.save_context(inputs, outputs)
        # 持久化到 SQLite
        if input_str:
            self._ltm.save_message("user", input_str)
        if output_str:
            self._ltm.save_message("assistant", output_str)

    def load_memory_variables(self, inputs: dict) -> dict:
        """获取记忆变量（用于 AgentExecutor）"""
        variables = self._buffer.load_memory_variables(inputs)

        # 如果历史太长，注入摘要信息
        total_turns = self._ltm.get_total_turns()
        if total_turns > self._ltm.summarization_threshold:
            summary_text = self._ltm._summarize_old_conversations()
            if summary_text:
                original = variables.get("chat_history", "")
                variables["chat_history"] = (
                    f"[历史对话摘要]\n{summary_text}\n\n"
                    f"[最近 {self.recent_turns} 轮对话]\n{original}"
                )

        return variables

    def clear(self):
        """清空记忆"""
        self._buffer.clear()


def get_memory(session_id: str = "default"):
    """工厂函数，创建混合记忆实例"""
    return HybridMemory(session_id=session_id)