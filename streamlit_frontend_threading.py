import re
import time
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_backend import chatbot

TYPING_DELAY = 0.02  # seconds between words
TITLE_LENGTH = 40
SUGGESTIONS = [
    "Explain LangGraph in simple terms",
    "Write a Python function to reverse a string",
    "Give me 3 tips for learning machine learning",
]

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())


def extract_text(content) -> str:
    # Gemini returns content as a list of blocks like [{'type': 'text', 'text': '...'}]
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


def make_title(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= TITLE_LENGTH else text[:TITLE_LENGTH].rstrip() + "…"


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    history = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, (HumanMessage, AIMessage)):
            text = extract_text(msg.content)
            if text:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                history.append({"role": role, "content": text})
    return history


def new_chat():
    # Don't pile up empty threads when "New chat" is clicked repeatedly
    if st.session_state["message_history"]:
        st.session_state["thread_id"] = generate_thread_id()
        st.session_state["message_history"] = []


def switch_thread(thread_id):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)


def queue_prompt(prompt):
    st.session_state["pending_prompt"] = prompt


# **************************************** Page setup *********************************

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
        /* Left-align conversation titles in the sidebar */
        [data-testid="stSidebar"] .stButton button { justify-content: flex-start; }
        [data-testid="stSidebar"] .stButton button p {
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# **************************************** Session Setup ******************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []  # oldest first

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# Read the input before drawing the sidebar so a new conversation shows up immediately
user_input = st.chat_input("Message the chatbot…") or st.session_state.pop("pending_prompt", None)

current_thread = st.session_state["thread_id"]
if user_input and current_thread not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].append(current_thread)
    st.session_state["thread_titles"][current_thread] = make_title(user_input)


# **************************************** Sidebar UI *********************************

with st.sidebar:
    st.markdown("## :material/smart_toy: LangGraph Chatbot")
    st.caption("Powered by Gemini + LangGraph")

    st.button(
        "New chat",
        icon=":material/add:",
        type="primary",
        width="stretch",
        on_click=new_chat,
    )

    st.divider()
    st.markdown("**Conversations**")

    if not st.session_state["chat_threads"]:
        st.caption("No conversations yet. Send a message to start one.")

    for thread_id in reversed(st.session_state["chat_threads"]):
        is_active = thread_id == current_thread
        st.button(
            st.session_state["thread_titles"].get(thread_id, "Untitled chat"),
            key=f"thread_{thread_id}",
            icon=":material/chat_bubble:" if is_active else ":material/chat_bubble_outline:",
            type="secondary" if is_active else "tertiary",
            width="stretch",
            on_click=switch_thread,
            args=(thread_id,),
        )


# **************************************** Main UI ************************************

title = st.session_state["thread_titles"].get(current_thread, "New chat")
st.title(title)

if not st.session_state["message_history"] and not user_input:
    st.markdown("#### How can I help you today?")
    st.caption("Pick a suggestion or type your own message below.")
    for i, suggestion in enumerate(SUGGESTIONS):
        st.button(
            suggestion,
            key=f"suggestion_{i}",
            icon=":material/lightbulb:",
            on_click=queue_prompt,
            args=(suggestion,),
        )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": current_thread}}

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("_Thinking…_")

        # Gemini sends large chunks, so re-split them into words for a typing effect
        def ai_only_stream():
            for message_chunk, _metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if not isinstance(message_chunk, AIMessage):
                    continue
                text = extract_text(message_chunk.content)
                if not text:
                    continue
                thinking.empty()
                for word in re.findall(r"\S+\s*|\s+", text):
                    yield word
                    time.sleep(TYPING_DELAY)

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as e:
            thinking.empty()
            st.error(f"Couldn't get a response from the model: {e}", icon=":material/error:")
        else:
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
