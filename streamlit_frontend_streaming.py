import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot


def extract_text(content) -> str:
    """Normalize an AIMessage.content into plain text.

    Newer langchain_google_genai versions return content as a list of
    content blocks (e.g. [{'type': 'text', 'text': '...'}]) instead of
    a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        [data-testid="stChatMessage"] { animation: fadein 0.25s ease-in; }
        @keyframes fadein {
            from { opacity: 0; transform: translateY(4px); }
            to   { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def new_thread_id():
    return str(uuid.uuid4())


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = new_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


def start_new_chat():
    st.session_state["thread_id"] = new_thread_id()
    st.session_state["message_history"] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## LangGraph Chatbot")
    st.caption("Powered by Gemini + LangGraph")

    if st.button("New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.caption(f"Thread ID\n\n`{st.session_state['thread_id'][:8]}…`")
    st.caption(f"Messages: {len(st.session_state['message_history'])}")

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Chat")

if not st.session_state["message_history"]:
    st.info("Ask me anything to get started.")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here…")

if user_input:
    # Show and store the user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the assistant's response, streamed into the UI token by token
    def stream_response():
        for message_chunk, _metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            text = extract_text(message_chunk.content)
            if text:
                yield text

    with st.chat_message("assistant"):
        ai_message = st.write_stream(stream_response)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
