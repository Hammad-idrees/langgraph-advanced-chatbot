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
- **Memory** — `langgraph_backend.py` uses an `InMemorySaver`, while `langgraph_database_backend.py` uses an SQLite `SqliteSaver` to persist conversation checkpoints in `chatbot.db`.

Both backends use the same Gemini chat model and graph structure. The database-backed version is used by [streamlit_frontend_database.py](streamlit_frontend_database.py).

## LangSmith tracing

LangChain and LangGraph send traces automatically when tracing is enabled. The backends load settings from `.env`; configure `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, and `LANGCHAIN_ENDPOINT` there. The key must be authorized for the configured LangSmith workspace. Restart Streamlit after changing environment settings.

## Frontend

[The Streamlit frontends](streamlit_frontend.py) provide chat interfaces built on Streamlit's `st.chat_message` / `st.chat_input`. `streamlit_frontend_database.py` uses SQLite-backed conversations; the other frontends use the in-memory backend.

- Each browser session is assigned its own `thread_id` (a UUID), so the LangGraph checkpointer keeps separate histories per session.
- Message history is mirrored in `st.session_state` so the UI can re-render past turns after every rerun.
- A "New chat" button in the sidebar clears history and starts a fresh `thread_id`.
- Gemini's response content can come back as a list of content blocks rather than a plain string; a small `extract_text` helper normalizes it to plain text before display.
- Dark theme is configured via [.streamlit/config.toml](.streamlit/config.toml).

## Current capabilities

- Single-turn Q&A with conversational memory within a session.
- No tools, retrieval, or multi-agent routing yet — it's a plain chat node.
- In-memory conversations reset on process restart; the database-backed frontend persists checkpoints in `chatbot.db`.

## Project files

| File                                                             | Purpose                                                   |
| ---------------------------------------------------------------- | --------------------------------------------------------- |
| [langgraph_backend.py](langgraph_backend.py)                     | Graph definition, state, model, checkpointer              |
| [langgraph_database_backend.py](langgraph_database_backend.py)   | Graph definition with SQLite checkpoint persistence       |
| [streamlit_frontend.py](streamlit_frontend.py)                   | Chat UI                                                   |
| [streamlit_frontend_database.py](streamlit_frontend_database.py) | Chat UI with persistent conversations                     |
| [.env](.env)                                                     | Gemini and optional LangSmith configuration (git-ignored) |
| [.streamlit/config.toml](.streamlit/config.toml)                 | Dark theme settings                                       |
