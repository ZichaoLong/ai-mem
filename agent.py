from __future__ import annotations
import os
import asyncio
from pydantic import BaseModel
from typing import ClassVar
try:
    from mem0 import AsyncMemoryClient
except ImportError:
    raise ImportError("mem0 is not installed. Please install it using 'pip install mem0ai'.")
from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
)
from agents.models import _openai_shared
import httpx
from agents.run import RunConfig

# Set environment variables at module level
os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
os.environ["OPENAI_API_KEY"] = "sk-jfG6565cacad97e9033779cb36ad9b7a917ba3782d9oThMu"
os.environ["MEM0_API_KEY"] = "m0-36uzjrFD0xdyJokomHMTesdbE2KrwOXfh9OhEzgM"

# Disable tracing to prevent timeout issues
os.environ["AGENTS_DISABLE_TRACING"] = "true"
os.environ["AGENTS_DISABLE_TELEMETRY"] = "true"
os.environ["OPENAI_DISABLE_TRACING"] = "true"

class Mem0Context(BaseModel):
    user_id: str | None = None
    client: ClassVar = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))

_openai_shared.set_use_responses_by_default(False)

@function_tool
async def add_to_memory(
    context: RunContextWrapper[Mem0Context],
    content: str,
) -> str:
    """
    Add a message to Mem0
    Args:
        content: The content to store in memory.
    """
    try:
        messages = [{"role": "user", "content": content}]
        user_id = context.context.user_id or "default_user"
        await context.context.client.add(messages, user_id=user_id)
        return f"Stored message: {content}"
    except Exception as e:
        return f"Error storing message: {str(e)}"

@function_tool
async def search_memory(
    context: RunContextWrapper[Mem0Context],
    query: str,
) -> str:
    """
    Search for memories in Mem0
    Args:
        query: The search query.
    """
    try:
        user_id = context.context.user_id or "default_user"
        memories = await context.context.client.search(query, user_id=user_id, output_format="v1.1")
        results = '\n'.join([result["memory"] for result in memories["results"]])
        return str(results) if results else "No memories found for this query."
    except Exception as e:
        return f"Error searching memory: {str(e)}"

@function_tool
async def get_all_memory(
    context: RunContextWrapper[Mem0Context],
) -> str:
    """Retrieve all memories from Mem0"""
    try:
        user_id = context.context.user_id or "default_user"
        memories = await context.context.client.get_all(user_id=user_id, output_format="v1.1")
        results = '\n'.join([result["memory"] for result in memories["results"]])
        return str(results) if results else "No memories stored yet."
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"

memory_agent = Agent[Mem0Context](
    name="Memory Assistant",
    instructions="""You are a helpful assistant with memory capabilities. You can:
    1. Store new information using add_to_memory
    2. Search existing information using search_memory
    3. Retrieve all stored information using get_all_memory
    When users ask questions:w
    - If they want to store information, use add_to_memory
    - If they're searching for specific information, use search_memory
    - If they want to see everything stored, use get_all_memory""",
    tools=[add_to_memory, search_memory, get_all_memory],
    model="gpt-3.5-turbo",
)


async def main():
    current_agent: Agent[Mem0Context] = memory_agent
    input_items: list[TResponseInputItem] = []
    context = Mem0Context()
    while True:
        user_input = input("Enter your message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        input_items.append({"content": user_input, "role": "user"})
        try:
            result = await Runner.run(
                current_agent,
                input_items,
                context=context,
                run_config=RunConfig(tracing_disabled=True)
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
        for new_item in result.new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
            elif isinstance(new_item, ToolCallItem):
                print(f"{agent_name}: Calling a tool")
            elif isinstance(new_item, ToolCallOutputItem):
                print(f"{agent_name}: Tool call output: {new_item.output}")
            else:
                print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
        input_items = result.to_input_list()

if __name__ == "__main__":
    asyncio.run(main())

