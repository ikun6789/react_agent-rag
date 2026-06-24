# app/stream_callback.py
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler


class QueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.buffer = ""
        self.streaming_final = False

        self.final_markers = [
            "Final Answer:",
            "Final Answer：",
            "最终答案:",
            "最终答案：",
        ]

    def on_llm_start(self, serialized, prompts, **kwargs):
        """
        每次 LLM 开始生成时重置状态。
        ReAct Agent 可能会多次调用 LLM。
        """
        self.buffer = ""
        self.streaming_final = False

    def on_llm_new_token(self, token: str, **kwargs):
        if not token:
            return

        # 如果已经进入最终答案阶段，直接输出 token
        if self.streaming_final:
            self.queue.put(token)
            return

        # 还没进入最终答案阶段，先缓存
        self.buffer += token

        # 检查是否出现 Final Answer 标记
        for marker in self.final_markers:
            if marker in self.buffer:
                self.streaming_final = True

                # 只输出 marker 后面的内容
                final_part = self.buffer.split(marker, 1)[1]

                if final_part:
                    self.queue.put(final_part)

                return

    def on_llm_error(self, error, **kwargs):
        self.queue.put(f"\n[LLM错误] {error}")