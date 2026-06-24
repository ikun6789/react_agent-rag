from langchain_openai import ChatOpenAI
from app.config import settings


def get_llm():
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.ARK_API_KEY,
        base_url=settings.ARK_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        streaming=True,#流式输出
    )