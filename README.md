# LangGraph Chatbot

A conversational chatbot agent built with [LangGraph](https://www.langchain.com/langgraph), powered by Google Gemini, with a Streamlit chat interface.

## Architecture

The agent is a single-node LangGraph `StateGraph`:

```
START → chat_node → END
```

- **State** — a `ChatState` holding one field, `messages`: a list of LangChain messages, accumulated via the `add_messages` reducer so each turn appends to history rather than overwriting it.
- **chat_node** — takes the current message list, sends it to the Gemini model, and returns the model's reply as a new message to append to state.
- **Model** — `ChatGoogleGenerativeAI` running `gemini-3.8-flash`.
- **Memory** — an `InMemorySaver` checkpointer, so the graph remembers prior turns within a conversation `thread_id`. Memory is process-local (in-memory), not persisted to disk.

This lives in [langgraph_backend.py](langgraph_backend.py).

## Frontend

[streamlit_frontend.py](streamlit_frontend.py) is a chat UI built on Streamlit's `st.chat_message` / `st.chat_input`:

- Each browser session is assigned its own `thread_id` (a UUID), so the LangGraph checkpointer keeps separate histories per session.
- Message history is mirrored in `st.session_state` so the UI can re-render past turns after every rerun.
- A "New chat" button in the sidebar clears history and starts a fresh `thread_id`.
- Gemini's response content can come back as a list of content blocks rather than a plain string; a small `extract_text` helper normalizes it to plain text before display.
- Dark theme is configured via [.streamlit/config.toml](.streamlit/config.toml).

## Current capabilities

- Single-turn Q&A with conversational memory within a session.
- No tools, retrieval, or multi-agent routing yet — it's a plain chat node.
- No persistent storage — history resets on process restart.

## Project files

| File | Purpose |
|---|---|
| [langgraph_backend.py](langgraph_backend.py) | Graph definition, state, model, checkpointer |
| [streamlit_frontend.py](streamlit_frontend.py) | Chat UI |
| [.env](.env) | `GEMINI_API_KEY` (git-ignored) |
| [.streamlit/config.toml](.streamlit/config.toml) | Dark theme settings |
