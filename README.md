# MCP Studio

A local web UI to connect LLMs (Anthropic / Gemini) with MCP servers and chat using their tools.

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000

## Features

- **Setup** — Choose Anthropic or Gemini, enter API key, fetch available models, select model. Current session details (provider, model, masked API key with eye toggle) shown once configured. Clear settings button to reset.
- **Prompt** — Edit the system prompt sent with every chat
- **MCP Servers** — Add stdio or HTTP MCP servers, test connections, inspect tools and input schemas
- **Chat** — Chat with the LLM; responses rendered as markdown. Tool calls run silently in the background.

## Multi-user

Each browser gets its own isolated session via a cookie. Users on different browsers or machines have completely separate configs. All state is in-memory — cleared when the server restarts.

## Adding MCP Servers

**stdio**
```
Name:    filesystem
Command: npx
Args:    -y @modelcontextprotocol/server-filesystem /tmp
```

**HTTP**
```
Name: my-server
URL:  http://localhost:8080/mcp
```

## Project Structure

```
main.py          # FastAPI routes + per-user session cookie handling
chat.py          # LLM chat loop with tool calling (Anthropic + Gemini)
mcp_manager.py   # MCP server connections (stdio + http)
llm.py           # Fetch available models from provider
config.py        # In-memory per-user session storage
static/
└── index.html   # Single-file UI
```
