#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%

import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
from agents import set_tracing_disabled
from agents.models import _openai_shared
_openai_shared.set_use_responses_by_default(False)
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent

model_name = 'gpt-4o'

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
    instructions="你把用户的文本翻译为英文",
    handoff_description="中文到英文翻译器",
    model=model_name,
)

french_agent = Agent(
    name="french_agent",
    instructions="你把用户的文本翻译为法语",
    handoff_description="中文到法语翻译器",
    model=model_name,
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "你是一个优秀的助手，且由于掌握了我给你的翻译工具，你尤为擅长翻译。"
        "你可以根据用户的输入和相关指令，把翻译工作交给我提供给你的工具，而其他任务则由你自己完成。"
        "注意，不要自己作翻译工作，但其他的用户指令你可以自己完成。"
    ),
    model=model_name,
    tools=[
        english_agent.as_tool(
            tool_name="translate_to_english",
            tool_description="把用户的文本翻译成英文",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="把用户的文本翻译成法语",
        ),
    ],
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions=(
        "你负责检查翻译结果。"
        "如果需要的话还应该把翻译结果进行修正，并且提供一个最终的合并版回答。"),
    model=model_name,
)

#%%
# msg = input("Hi! What would you like translated, and to which languages? ")
msg = "请你首先写一篇约500字的中文作文并展现给我，然后把这篇作文交给工具翻译为英文。"
# msg = "请你生成一段500字的中文，不用翻译。"
set_tracing_disabled(True)
if True:
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
else:
    orchestrator_result = Runner.run_streamed(
            orchestrator_agent, msg,
            run_config=RunConfig(tracing_disabled=True)
            )
    async for event in orchestrator_result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    synthesizer_result = Runner.run_streamed(
        synthesizer_agent, orchestrator_result.to_input_list(),
        run_config=RunConfig(tracing_disabled=True)
    )
    async for event in synthesizer_result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


#%%


