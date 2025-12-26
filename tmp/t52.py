import asyncio
from logging import disable
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, FunctionTool, ModelSettings, StopAtTools, set_tracing_disabled
from openai import AsyncOpenAI
import os

set_tracing_disabled(disabled=True)
provider = AsyncOpenAI()
model = OpenAIChatCompletionsModel(model="qwen3-max", openai_client=provider)

async def get_weather(ctx, args: str) -> str:
    import json
    city = json.loads(args)["city"]
    return f"The weather in {city} is sunny."

params_json_schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"]
}

tool = FunctionTool(
    name="get_weather",
    description="Get the weather in a city.",
    params_json_schema=params_json_schema,
    on_invoke_tool=get_weather,
)

agent = Agent(
    name="weather agent",
    instructions="You are a helpful agent.",
    tools=[tool],
    model=model,
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_weather"]),
    model_settings=ModelSettings(tool_choice="required")
)

async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())