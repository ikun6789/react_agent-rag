"""
持久化长期记忆模块（SQLite 实现）

功能：
- 将每一轮对话保存到 SQLite 数据库，程序重启后不丢失
- 启动时自动加载最近 N 轮对话 + 历史会话摘要
- 当对话历史超过阈值时，自动摘要旧对话以节省 Token
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import HumanMessage, AIMessage, SystemMessage


class LongTermMemory:
    """SQLite 持久化长期记忆"""

    def __init__(
        self,
        db_dir: str = None,
        session_id: str = "default",
        recent_turns: int = 20,
        summarization_threshold: int = 50,
    ):
        if db_dir is None:
            db_dir = str(Path(__file__).resolve().parent.parent / "data" / "memory")

        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.db_dir / "long_term_memory.db")

        self.session_id = session_id
        self.recent_turns = recent_turns
        self.summarization_threshold = summarization_threshold

        self._init_db()

    # ── 数据库初始化 ──────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role      TEXT    NOT NULL,   -- "user" | "assistant"
                content   TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                summary    TEXT    NOT NULL,
                timestamp  TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversations (session_id, id)
        """)
        conn.commit()
        conn.close()

    # ── 写入 ──────────────────────────────────────────────────

    def save_message(self, role: str, content: str):
        """保存单条消息"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def save_summary(self, summary: str):
        """保存当前会话的摘要"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO session_summaries (session_id, summary, timestamp) VALUES (?, ?, ?)",
            (self.session_id, summary, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    # ── 读取 ──────────────────────────────────────────────────

    def load_recent_messages(self, n: int = None) -> List[Tuple[str, str]]:
        """加载最近 n 轮对话，返回 [(role, content), ...] 时间升序"""
        if n is None:
            n = self.recent_turns
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT role, content FROM conversations
             WHERE session_id = ?
             ORDER BY id DESC LIMIT ?
            """,
            (self.session_id, n * 2),  # *2 因为 user+assistant 算一轮
        )
        rows = cursor.fetchall()
        conn.close()
        rows.reverse()  # 转为时间升序
        return rows

    def load_all_session_summaries(self) -> List[str]:
        """加载所有历史会话的摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            SELECT summary FROM session_summaries
             WHERE session_id = ?
             ORDER BY id DESC LIMIT 10
            """,
            (self.session_id,),
        )
        rows = [row[0] for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_total_turns(self) -> int:
        """获取当前会话总对话轮数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) / 2 FROM conversations WHERE session_id = ?",
            (self.session_id,),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # ── 构建给 Agent 的历史消息 ──────────────────────────────

    def build_chat_history_for_agent(self):
        """
        构造传入 Agent 的 chat_history（HumanMessage / AIMessage 列表）。
        策略：
          1. 如果历史轮数 > summarization_threshold，先加载摘要作为 SystemMessage
          2. 再加载最近 recent_turns 轮的详细消息
        """
        total_turns = self.get_total_turns()
        messages = []

        # 1) 历史摘要（旧对话压缩）
        if total_turns > self.summarization_threshold:
            summaries = self.load_all_session_summaries()
            # 也尝试用 LLM 生成一个新摘要
            summary_text = self._summarize_old_conversations()
            if summary_text:
                messages.append(
                    SystemMessage(
                        content=f"【历史对话摘要】\n{summary_text}\n\n以下是最新的详细对话："
                    )
                )

        # 2) 最近对话详情
        recent = self.load_recent_messages(self.recent_turns)
        for role, content in recent:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    def build_formatted_history(self) -> str:
        """
        构建文本格式的历史记录（用于 Prompt Template 的 {chat_history}）
        策略同上：摘要 + 最近对话
        """
        total_turns = self.get_total_turns()
        parts = []

        if total_turns > self.summarization_threshold:
            summary_text = self._summarize_old_conversations()
            if summary_text:
                parts.append(f"[历史对话摘要]\n{summary_text}\n")

        recent = self.load_recent_messages(self.recent_turns)
        for role, content in recent:
            if role == "user":
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")

        return "\n".join(parts)

    # ── 摘要 ──────────────────────────────────────────────────

    def _summarize_old_conversations(self) -> Optional[str]:
        """
        获取所有历史摘要 + 当前会话中超出 recent_turns 的对话，
        拼成一段摘要文本。
        （实际项目中可调用 LLM 做智能摘要，此处先做简单拼接）
        """
        summaries = self.load_all_session_summaries()
        if summaries:
            return " | ".join(summaries)

        # 如果没有摘要，取超出 recent_turns 的旧消息做简单摘要
        conn = sqlite3.connect(self.db_path)
        all_rows = conn.execute(
            """
            SELECT role, content FROM conversations
             WHERE session_id = ?
             ORDER BY id ASC
            """,
            (self.session_id,),
        ).fetchall()
        conn.close()

        # 去掉最近 recent_turns 轮
        cutoff = self.recent_turns * 2
        old_rows = all_rows[:-cutoff] if len(all_rows) > cutoff else []

        if not old_rows:
            return None

        # 简单摘要：取前几轮的关键信息
        key_points = []
        for role, content in old_rows[:6]:  # 只取前 3 轮
            content_preview = content[:100] if len(content) > 100 else content
            key_points.append(f"[{role}] {content_preview}")

        return "早期对话要点：\n" + "\n".join(key_points)

    # ── 维护 ──────────────────────────────────────────────────

    def trim_old_history(self, keep_turns: int = 100):
        """
        清理过旧的对话记录，仅保留最近 keep_turns 轮。
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            DELETE FROM conversations
             WHERE session_id = ?
               AND id NOT IN (
                   SELECT id FROM conversations
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
               )
            """,
            (self.session_id, self.session_id, keep_turns * 2),
        )
        conn.commit()
        conn.close()
