import uuid

_DEFAULT = {
    "system_prompt": "You are a helpful assistant with access to various tools via MCP servers. Use the available tools when needed to answer user questions accurately.",
    "provider": None,
    "api_key": None,
    "model": None,
    "mcp_servers": []
}

_sessions: dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = dict(_DEFAULT)
    return _sessions[session_id]

def save_session(session_id: str, data: dict):
    _sessions[session_id] = data

def reset_session(session_id: str):
    _sessions.pop(session_id, None)

def new_session_id() -> str:
    return str(uuid.uuid4())

_chat_histories: dict[str, list] = {}

def get_chat_history(session_id: str) -> list:
    return _chat_histories.setdefault(session_id, [])

def append_chat(session_id: str, message: dict):
    get_chat_history(session_id).append(message)

def clear_chat(session_id: str):
    _chat_histories.pop(session_id, None)
