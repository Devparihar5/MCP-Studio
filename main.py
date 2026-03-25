import json
from fastapi import FastAPI, HTTPException, Cookie, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from config import get_session, save_session, reset_session, new_session_id, get_chat_history, append_chat, clear_chat
from llm import fetch_models
from mcp_manager import get_server_tools
from chat import chat_anthropic, chat_gemini

app = FastAPI()

SESSION_COOKIE = "mcp_session"

def _get_or_create_session(response: Response, session_id: Optional[str]) -> tuple[str, dict]:
    if not session_id:
        session_id = new_session_id()
        response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return session_id, get_session(session_id)


# ── Settings ──────────────────────────────────────────────────────────────────

class ProviderSetup(BaseModel):
    provider: str
    api_key: str

class ModelSelect(BaseModel):
    model: str

class PromptUpdate(BaseModel):
    system_prompt: str


@app.get("/api/settings")
def get_settings(response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, cfg = _get_or_create_session(response, mcp_session)
    return {
        "provider": cfg.get("provider"),
        "model": cfg.get("model"),
        "system_prompt": cfg.get("system_prompt"),
        "api_key_masked": ("*" * 8 + cfg["api_key"][-4:]) if cfg.get("api_key") else None,
        "api_key": cfg.get("api_key"),
    }


@app.post("/api/settings/reset")
def reset_settings(response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    if mcp_session:
        reset_session(mcp_session)
        clear_chat(mcp_session)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.post("/api/settings/provider")
async def set_provider(body: ProviderSetup, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    if body.provider not in ("anthropic", "gemini"):
        raise HTTPException(400, "Provider must be 'anthropic' or 'gemini'")
    try:
        models = await fetch_models(body.provider, body.api_key)
    except Exception as e:
        raise HTTPException(400, f"Invalid API key or provider error: {e}")

    sid, cfg = _get_or_create_session(response, mcp_session)
    cfg["provider"] = body.provider
    cfg["api_key"] = body.api_key
    cfg["model"] = None
    save_session(sid, cfg)
    return {"models": models}


@app.post("/api/settings/model")
def set_model(body: ModelSelect, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, cfg = _get_or_create_session(response, mcp_session)
    cfg["model"] = body.model
    save_session(sid, cfg)
    return {"ok": True}


@app.post("/api/settings/prompt")
def update_prompt(body: PromptUpdate, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, cfg = _get_or_create_session(response, mcp_session)
    cfg["system_prompt"] = body.system_prompt
    save_session(sid, cfg)
    return {"ok": True}


# ── MCP Servers ───────────────────────────────────────────────────────────────

class MCPServer(BaseModel):
    name: str
    type: str
    command: str = ""
    args: list[str] = []
    env: dict = {}
    url: str = ""


@app.get("/api/mcp/servers")
def list_servers(response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    _, cfg = _get_or_create_session(response, mcp_session)
    return cfg.get("mcp_servers", [])


@app.post("/api/mcp/servers")
def add_server(server: MCPServer, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, cfg = _get_or_create_session(response, mcp_session)
    servers = [s for s in cfg.get("mcp_servers", []) if s["name"] != server.name]
    servers.append(server.model_dump())
    cfg["mcp_servers"] = servers
    save_session(sid, cfg)
    return {"ok": True}


@app.delete("/api/mcp/servers/{name}")
def delete_server(name: str, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, cfg = _get_or_create_session(response, mcp_session)
    cfg["mcp_servers"] = [s for s in cfg.get("mcp_servers", []) if s["name"] != name]
    save_session(sid, cfg)
    return {"ok": True}


@app.post("/api/mcp/test/{name}")
async def test_server(name: str, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    _, cfg = _get_or_create_session(response, mcp_session)
    server = next((s for s in cfg.get("mcp_servers", []) if s["name"] == name), None)
    if not server:
        raise HTTPException(404, "Server not found")
    try:
        tools = await get_server_tools(server)
        return {"ok": True, "tools": tools}
    except Exception as e:
        raise HTTPException(400, f"Connection failed: {e}")


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.get("/api/chat/history")
def chat_history(response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, _ = _get_or_create_session(response, mcp_session)
    return get_chat_history(sid)


@app.delete("/api/chat/history")
def clear_chat_history(response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, _ = _get_or_create_session(response, mcp_session)
    clear_chat(sid)
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatRequest, response: Response, mcp_session: Optional[str] = Cookie(default=None)):
    sid, cfg = _get_or_create_session(response, mcp_session)
    if not cfg.get("api_key") or not cfg.get("model"):
        raise HTTPException(400, "Provider and model must be configured first")

    provider = cfg["provider"]
    system_prompt = cfg.get("system_prompt", "")
    servers = cfg.get("mcp_servers", [])

    append_chat(sid, {"role": "user", "content": body.message})
    messages = get_chat_history(sid)

    async def stream():
        assistant_text = ""
        gen = (
            chat_anthropic(messages, system_prompt, cfg["api_key"], cfg["model"], servers)
            if provider == "anthropic"
            else chat_gemini(messages, system_prompt, cfg["api_key"], cfg["model"], servers)
        )
        async for chunk in gen:
            if chunk["type"] == "text":
                assistant_text += chunk["content"]
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
        append_chat(sid, {"role": "assistant", "content": assistant_text})

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Static UI ─────────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")
