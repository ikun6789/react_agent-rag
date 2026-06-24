# app/main_fastapi.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.agent import create_agent_executor
from app.logger import logger
from app.stream_callback import QueueCallbackHandler

from queue import Queue
from threading import Thread
import asyncio


# 创建 Agent Executor（使用固定 session_id，生产环境建议从请求中获取用户标识）
agent_executor = create_agent_executor(
    verbose=False,
    session_id="api_default",
)

app = FastAPI(title="智能问答 Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "")
        session_id = body.get("session_id", "api_default")

        if not question:
            return JSONResponse({"answer": "问题不能为空"})

        # 支持每用户独立记忆
        local_agent = create_agent_executor(
            verbose=False,
            session_id=session_id,
        )

        result = local_agent.invoke({"input": question})

        if isinstance(result, dict):
            answer = result.get("output", str(result))
        else:
            answer = str(result)

        return JSONResponse({"answer": answer})

    except Exception as e:
        logger.exception(e)
        return JSONResponse({"answer": f"后端错误: {e}"})


@app.post("/api/ask_stream")
async def ask_stream(request: Request):
    body = await request.json()
    question = body.get("question", "")
    session_id = body.get("session_id", "api_default")

    if not question:
        return StreamingResponse(
            iter(["问题不能为空"]),
            media_type="text/plain; charset=utf-8"
        )

    # 每个请求创建独立的 agent_executor（含独立记忆）
    local_agent = create_agent_executor(
        verbose=False,
        session_id=session_id,
    )

    token_queue = Queue()
    callback_handler = QueueCallbackHandler(token_queue)

    DONE = "[[STREAM_DONE]]"

    def run_agent():
        try:
            local_agent.invoke(
                {"input": question},
                config={
                    "callbacks": [callback_handler]
                }
            )
        except Exception as e:
            token_queue.put(f"\n[后端错误] {e}")
        finally:
            token_queue.put(DONE)

    Thread(target=run_agent, daemon=True).start()

    async def token_generator():
        while True:
            token = await asyncio.to_thread(token_queue.get)

            if token == DONE:
                break

            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8"
    )
#cd /home/lyj/Agent/react_agent+rag
#uvicorn main_fastapi:app --host 127.0.0.1 --port 8000 --reload