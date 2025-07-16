#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%

import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
from agents import set_tracing_disabled
from agents.models import _openai_shared
_openai_shared.set_use_responses_by_default(False)
from agents.run import RunConfig

os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
os.environ["OPENAI_API_KEY"] = "sk-jfG6565cacad97e9033779cb36ad9b7a917ba3782d9oThMu"
os.environ["OPENAI_DISABLE_TRACING"] = "true"

"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
"""

english_agent = Agent(
    name="english_agent",
    instructions="You translate the user's message to English",
    handoff_description="An chinese to english translator",
    model="gpt-3.5-turbo",
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An chinese to french translator",
    model="gpt-3.5-turbo",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    model="gpt-3.5-turbo",
    tools=[
        english_agent.as_tool(
            tool_name="translate_to_english",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
    model="gpt-3.5-turbo",
)


async def main():
    # msg = input("Hi! What would you like translated, and to which languages? ")
    msg = "我是一名工程师。翻译为英文。"

    # Run the entire orchestration in a single trace
    with trace("Orchestrator evaluator"):
        set_tracing_disabled(True)
        orchestrator_result = await Runner.run(
                orchestrator_agent, msg,
                run_config=RunConfig(tracing_disabled=True)
                )

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation step: {text}")

        synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list(),
            run_config=RunConfig(tracing_disabled=True)
        )

    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
#%%


