# 常见智能体模式

此文件夹包含了不同常见的智能体（Agent）模式示例。



## 确定性流程（Deterministic flows）

一种常见策略是将一个任务拆解为一系列更小的步骤。每一个小任务都可以由一个智能体来完成，并且前一个智能体的输出会作为下一个智能体的输入。例如，如果你的任务是生成一个故事，你可以把它拆分成如下步骤：

1. 生成提纲
2. 生成正文
3. 生成结尾

每一步都可以由一个智能体来执行。每个智能体的输出会作为下一个智能体的输入。

具体示例请见 [`deterministic.py`](./deterministic.py) 文件。



这个例子演示了一个确定性流程，每一步都由一个智能体完成：

1. 第一个智能体生成故事提纲
2. 我们把提纲传递给第二个智能体
3. 第二个智能体检查提纲的质量是否合格，以及它是否属于科幻故事
4. 如果提纲质量不合格，或者不是科幻故事，流程在这里终止
5. 如果提纲质量合格且为科幻故事，我们再把提纲传递给第三个智能体
6. 第三个智能体根据提纲创作完整的故事



```python
import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os

set_tracing_disabled(disabled=True)

"""
This example demonstrates a deterministic flow, where each step is performed by an agent.
1. The first agent generates a story outline
2. We feed the outline into the second agent
3. The second agent checks if the outline is good quality and if it is a scifi story
4. If the outline is not good quality or not a scifi story, we stop here
5. If the outline is good quality and a scifi story, we feed the outline into the third agent
6. The third agent writes the story
"""

provider = AsyncOpenAI()
model = OpenAIChatCompletionsModel(model="qwen3-max", openai_client=provider)

story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
    model=model
)

class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool

outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions="Read the given story outline, and judge the quality. Also, determine if it is a scifi story.",
    output_type=OutlineCheckerOutput,
    model=model
)

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    output_type=str,
    model=model
)

async def main():
    input_prompt = input("What kind of story do you want? ")

    # Ensure the entire workflow is a single trace
    with trace("Deterministic story flow"):
        # 1. Generate an outline
        outline_result = await Runner.run(
            story_outline_agent,
            input_prompt,
        )
        print("Outline generated")

        # 2. Check the outline
        outline_checker_result = await Runner.run(
            outline_checker_agent,
            outline_result.final_output,
        )

        # 3. Add a gate to stop if the outline is not good quality or not a scifi story
        assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
        if not outline_checker_result.final_output.good_quality:
            print("Outline is not good quality, so we stop here.")
            exit(0)

        if not outline_checker_result.final_output.is_scifi:
            print("Outline is not a scifi story, so we stop here.")
            exit(0)

        print("Outline is good quality and a scifi story, so we continue to write the story.")

        # 4. Write the story
        story_result = await Runner.run(
            story_agent,
            outline_result.final_output,
        )
        print(f"Story: {story_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
```



## 交接与路由（Handoffs and routing）

在很多情况下，你会有一些专门处理特定任务的子智能体（sub-agent）。你可以通过任务交接（handoff）机制将任务路由到合适的智能体处理。

例如，你可以有一个前台智能体（frontline agent）接收所有请求，然后根据请求的语言，将其交接给对应的专门智能体进行处理。

具体示例请参见 [`routing.py`](./routing.py) 文件。



## 智能体作为工具（Agents as tools）

关于任务交接的一种思维模型是：新的智能体“接管”任务。它可以看到之前的对话历史，并从那一刻起主导对话。不过，这并不是使用智能体的唯一方式。你也可以把智能体当作工具来使用——工具型智能体会独立运行，然后把结果返回给原始的智能体。

例如，上面提到的翻译任务，你可以用“工具调用”的方式来建模：不是将任务交接给特定语言的智能体，而是把该智能体当作工具来调用，然后将结果用于下一步。这种方式支持比如同时翻译多种语言等功能。

具体示例请参见 [`agents_as_tools.py`](./agents_as_tools.py) 文件。



## LLM作为评审（LLM-as-a-judge）

LLM在获得反馈的情况下，通常能提升其输出质量。一种常见模式是，先用一个模型生成结果，然后用第二个模型对结果进行反馈。为了优化成本，你甚至可以用小模型做初步生成，用大模型做反馈评估。

例如，你可以用一个 LLM 生成故事提纲，然后用另一个 LLM 对提纲进行评价和反馈。接下来可以根据反馈完善提纲，并重复这个过程，直到 LLM 对提纲满意为止。

具体示例请参见 [`llm_as_a_judge.py`](./llm_as_a_judge.py) 文件。



## 并行化（Parallelization）

让多个智能体并行运行是一种常见模式。这可以用于降低延迟（例如，如果你有多个步骤彼此独立可以同时进行），也可以用于其他用途，比如生成多个不同的结果，再选择其中最优的那个。

具体示例请参见 [`parallelization.py`](./parallelization.py) 文件。示例中会并行多次运行翻译智能体，然后选择最好的翻译结果。



## 防护措施（Guardrails）

和并行化相关，你经常会希望在智能体处理前运行输入防护措施（guardrails），以确保交给智能体的输入是有效的。例如，如果你有一个客服智能体，你可能想要确保用户不是在请求数学帮助这样的不相关内容。

你完全可以不用特殊的Agents SDK特性，只用并行化来实现这一点。但我们还支持一种特殊的防护措施原语（guardrail primitive）。防护措施可以设置一个“绊线（tripwire）”——一旦被触发，智能体的执行会立刻中止，并抛出一个 `GuardrailTripwireTriggered` 异常。

这种机制对于降低延迟非常有用：比如，你可以用一个非常快的模型来做防护检查，用一个较慢的模型来运行实际的智能体。如果输入无效，你无需等待慢模型跑完，防护措施就能快速拒绝无效输入。

具体示例请参见 [`input_guardrails.py`](./input_guardrails.py) 和 [`output_guardrails.py`](./output_guardrails.py) 文件。