import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

async def get_server_tools(server: dict) -> list[dict]:
    """Connect to an MCP server and return its tools with input schemas."""
    tools = []

    if server["type"] == "stdio":
        params = StdioServerParameters(
            command=server["command"],
            args=server.get("args", []),
            env=server.get("env", None),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema,
                    }
                    for t in result.tools
                ]

    elif server["type"] == "http":
        async with streamablehttp_client(server["url"]) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema,
                    }
                    for t in result.tools
                ]

    return tools


async def call_tool(server: dict, tool_name: str, tool_input: dict) -> str:
    if server["type"] == "stdio":
        params = StdioServerParameters(
            command=server["command"],
            args=server.get("args", []),
            env=server.get("env", None),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_input)
                return str(result.content)

    elif server["type"] == "http":
        async with streamablehttp_client(server["url"]) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_input)
                return str(result.content)
