# streamlit_app.py
import streamlit as st
import requests
import html

API_URL = "http://127.0.0.1:8000/api/ask_stream"

st.set_page_config(
    page_title="智能问答 Agent",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f7f7f8;
}

.chat-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 8px;
}

.chat-subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

.user-message {
    background-color: #DCF8C6;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 10px 0;
    max-width: 75%;
    margin-left: auto;
    color: #111;
    line-height: 1.6;
    white-space: pre-wrap;
}

.agent-message {
    background-color: #FFFFFF;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 10px 0;
    max-width: 75%;
    margin-right: auto;
    color: #111;
    line-height: 1.6;
    border: 1px solid #e5e5e5;
    white-space: pre-wrap;
}

.message-label {
    font-size: 13px;
    color: #888;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="chat-title">🤖 智能问答 Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">完全流式输出版本</div>', unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("设置")

    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("流式接口：")
    st.code(API_URL)


# 显示历史消息
for message in st.session_state.messages:
    role = message["role"]
    content = html.escape(message["content"])

    if role == "user":
        st.markdown(
            f"""
            <div class="message-label" style="text-align:right;">你</div>
            <div class="user-message">{content}</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="message-label">Agent</div>
            <div class="agent-message">{content}</div>
            """,
            unsafe_allow_html=True
        )


user_input = st.chat_input("请输入你的问题...")


if user_input:
    question = user_input.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.markdown(
        f"""
        <div class="message-label" style="text-align:right;">你</div>
        <div class="user-message">{html.escape(question)}</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="message-label">Agent</div>', unsafe_allow_html=True)

    answer_placeholder = st.empty()
    full_answer = ""

    try:
        with requests.post(
            API_URL,
            json={"question": question},
            stream=True,
            timeout=300
        ) as response:

            response.raise_for_status()

            for chunk in response.iter_content(
                chunk_size=None,
                decode_unicode=True
            ):
                if chunk:
                    full_answer += chunk

                    safe_answer = html.escape(full_answer)

                    answer_placeholder.markdown(
                        f"""
                        <div class="agent-message">{safe_answer}▌</div>
                        """,
                        unsafe_allow_html=True
                    )

        safe_answer = html.escape(full_answer)

        answer_placeholder.markdown(
            f"""
            <div class="agent-message">{safe_answer}</div>
            """,
            unsafe_allow_html=True
        )

    except requests.exceptions.ConnectionError:
        full_answer = "请求错误：无法连接后端服务，请确认 FastAPI 已启动。"
        answer_placeholder.markdown(
            f'<div class="agent-message">{html.escape(full_answer)}</div>',
            unsafe_allow_html=True
        )

    except requests.exceptions.Timeout:
        full_answer = "请求超时：Agent 响应时间过长。"
        answer_placeholder.markdown(
            f'<div class="agent-message">{html.escape(full_answer)}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        full_answer = f"请求错误：{e}"
        answer_placeholder.markdown(
            f'<div class="agent-message">{html.escape(full_answer)}</div>',
            unsafe_allow_html=True
        )

    st.session_state.messages.append({
        "role": "agent",
        "content": full_answer
    })
#cd /home/lyj/Agent/react_agent+rag
#streamlit run streamlit_app.py