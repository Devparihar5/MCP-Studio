import json
import asyncio
import anthropic
from google import genai
from google.genai import types as genai_types
from mcp_manager import get_server_tools, call_tool


def _mcp_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t["inputSchema"],
        }
        for t in tools
    ]


def _mcp_tools_to_gemini(tools: list[dict]) -> list:
    declarations = []
    for t in tools:
        schema = t.get("inputSchema", {})
        declarations.append(
            genai_types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=schema if schema.get("properties") else None,
            )
        )
    return [genai_types.Tool(function_declarations=declarations)]


async def _gather_all_tools(servers: list[dict]) -> tuple[list[dict], dict]:
    """Returns (all_tools, tool_to_server_map)"""
    all_tools = []
    tool_server_map = {}
    for server in servers:
        try:
            tools = await get_server_tools(server)
            for t in tools:
                tool_server_map[t["name"]] = server
            all_tools.extend(tools)
        except Exception as e:
            print(f"Failed to load tools from {server.get('name')}: {e}")
    return all_tools, tool_server_map


async def chat_anthropic(messages: list[dict], system_prompt: str, api_key: str, model: str, servers: list[dict]):
    all_tools, tool_server_map = await _gather_all_tools(servers)
    client = anthropic.Anthropic(api_key=api_key)
    anthropic_tools = _mcp_tools_to_anthropic(all_tools)

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools if anthropic_tools else [],
        )

        # Yield text chunks
        for block in response.content:
            if hasattr(block, "text"):
                yield {"type": "text", "content": block.text}

        if response.stop_reason != "tool_use":
            break

        # Handle tool calls
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            yield {"type": "tool_call", "name": tool_use.name, "input": tool_use.input}
            server = tool_server_map.get(tool_use.name)
            if server:
                result = await call_tool(server, tool_use.name, tool_use.input)
            else:
                result = "Tool not found"
            yield {"type": "tool_result", "name": tool_use.name, "result": result}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


async def chat_gemini(messages: list[dict], system_prompt: str, api_key: str, model: str, servers: list[dict]):
    all_tools, tool_server_map = await _gather_all_tools(servers)
    client = genai.Client(api_key=api_key)
    gemini_tools = _mcp_tools_to_gemini(all_tools) if all_tools else []

    # Convert messages to Gemini format
    history = []
    for m in messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append(genai_types.Content(role=role, parts=[genai_types.Part(text=m["content"])]))

    last_message = messages[-1]["content"]

    chat = client.chats.create(
        model=model,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools if gemini_tools else None,
        ),
        history=history,
    )

    while True:
        response = chat.send_message(last_message)
        text_parts = []

        for part in response.candidates[0].content.parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                fc = part.function_call
                yield {"type": "tool_call", "name": fc.name, "input": dict(fc.args)}
                server = tool_server_map.get(fc.name)
                result = await call_tool(server, fc.name, dict(fc.args)) if server else "Tool not found"
                yield {"type": "tool_result", "name": fc.name, "result": result}
                # Send tool result back
                last_message = genai_types.Part(
                    function_response=genai_types.FunctionResponse(name=fc.name, response={"result": result})
                )
                break
        else:
            # No function calls, yield text and stop
            yield {"type": "text", "content": "".join(text_parts)}
            break
