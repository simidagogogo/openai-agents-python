# 常见Agent模式

此文件夹包含了不同常见的Agent模式示例。



## 确定性流程（Deterministic flows）

一种常见策略是将一个任务拆解为一系列更小的步骤。每一个小任务都可以由一个Agent来完成，并且前一个Agent的输出会作为下一个Agent的输入。例如，如果你的任务是生成一个故事，你可以把它拆分成如下步骤：

1. 生成提纲
2. 生成正文
3. 生成结尾

每一步都可以由一个Agent来执行。每个Agent的输出会作为下一个Agent的输入。



**具体示例**

详见 [`deterministic.py`](./deterministic.py) 。这个例子演示了一个确定性流程，每一步都由一个Agent完成：

1. 第一个Agent生成故事提纲，然后把提纲传递给第二个Agent
2. 第二个Agent检查提纲的质量是否合格，以及它是否属于科幻故事
   1. 如果提纲质量不合格或者不是科幻故事，终止流程
   2. 如果提纲质量合格且为科幻故事，再把提纲传递给第三个Agent

3. 第三个Agent根据提纲创作完整的故事

```python
import asyncio
import os

from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled, trace

set_tracing_disabled(disabled=True)

provider = AsyncOpenAI()
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=provider)

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

在很多情况下，会有一些专门处理特定任务的子智能体（sub-agent）。可以通过任务交接（handoff）机制将任务路由到合适的agent处理。

例如，可以有一个前台agent接收所有请求，然后根据请求的语言，将其交接给对应的专门agent进行处理。



**具体示例**

参见 [`routing.py`](./routing.py) 文件。这个例子展示了“转接/路由”模式。分诊agent会接收用户的首条消息，然后根据请求的语言将对话转接给相应的agent。用户会实时收到响应内容。

```python
import asyncio
import uuid

from openai import AsyncOpenAI
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RawResponsesStreamEvent,
    Runner,
    TResponseInputItem,
    trace,
)

model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
    model=model
)

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You only speak Spanish",
    model=model
)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
    model=model
)

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, spanish_agent, english_agent],
    model=model
)

async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    msg = input("Hi! We speak French, Spanish and English. How can I help? ")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # Each conversation turn is a single trace. Normally, each input from the user would be an
        # API request to your app, and you can wrap the request in a trace()
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent

if __name__ == "__main__":
    asyncio.run(main())
```



## 智能体作为工具（Agents as tools）

关于任务交接的一种思维模型是：新的Agents“接管”任务。它可以看到之前的对话历史，并从那一刻起主导对话。不过，这并不是使用Agents的唯一方式。也可以把Agents当作工具来使用——工具型Agents会独立运行，然后把结果返回给原始的Agents。

例如，上面提到的翻译任务，你可以用“工具调用”的方式来建模：不是将任务交接给特定语言的智能体，而是把该Agents当作工具来调用，然后将结果用于下一步。这种方式支持比如同时翻译多种语言等功能。

将Agent转换为一个工具，可以被其他智能体调用。

agents-as-tools vs handoffs

- agents-as-tools（工具）：
  - 主控智能体只把当前生成的input发给子工具Agent，即新Agent只会收到生成的输入内容。
  - 新的Agent作为一个工具被调用，对话权仍在主控agent手里。翻译工具输出后，对话继续由主控agent管理和继续进行
- handoffs（转接）：
  - 切换到新Agent后，新Agent会接管整个对话，原 agent不再参与对话



**具体示例**

请参见 [`agents_as_tools.py`](./agents_as_tools.py) 文件。这段代码演示了自动分配翻译任务、工具式调用翻译子智能体并汇总结果。

- 有一个主控Agent，它负责根据用户的需求，调用一组翻译工具Agent
- 每一个目标语言都定义了一个“子智能体”，它们只做各自语言的翻译
- 子Agent通过 `as_tool(...)` 被注册为工具，这样主控Agent可以像调用函数一样使用它们，而不是交接整个会话

```python
import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and then picks which agents to call, as tools. In this case, it picks from a set of translation agents.
"""

set_tracing_disabled(disabled=True)
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An english to spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An english to french translator",
    model=model
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An english to italian translator",
    model=model
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools in order."
        "You never translate on your own, you always use the provided tools."
    ),
    model=model,
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
    model=model
)

async def main():
    msg = input("Hi! What would you like translated, and to which languages? ")

    # Run the entire orchestration in a single trace
    with trace("Orchestrator evaluator"):
        orchestrator_result = await Runner.run(orchestrator_agent, msg)

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation step: {text}")

        synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list()
        )

    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
```





## LLM作为评审（LLM-as-a-judge）

LLM在获得反馈的情况下，通常能提升其输出质量。一种常见模式是，先用一个模型生成结果，然后用第二个模型对结果进行反馈。为了优化成本，你甚至可以用小模型做初步生成，用大模型做反馈评估。



具体示例

请参见 [`llm_as_a_judge.py`](./llm_as_a_judge.py) 文件。用一个LLM生成故事提纲，然后用另一个LLM对提纲进行评价和反馈。接下来可以根据反馈完善提纲，并重复这个过程，直到 LLM 对提纲满意为止。

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI

from agents import (
    Agent,
    ItemHelpers,
    OpenAIChatCompletionsModel,
    Runner,
    TResponseInputItem,
    set_tracing_disabled,
    trace,
)
from agents.tracing import provider

"""
This example shows the LLM as a judge pattern. The first agent generates an outline for a story.
The second agent judges the outline and provides feedback. We loop until the judge is satisfied with the outline.
"""

set_tracing_disabled(disabled=True)
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

story_outline_generator = Agent(
    name="story_outline_generator",
    instructions=(
        "You generate a very short story outline based on the user's input. "
        "If there is any feedback provided, use it to improve the outline."
    ),
    model=model,
)


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


evaluator = Agent[None](
    name="evaluator",
    instructions=(
        "You evaluate a story outline and decide if it's good enough. "
        "If it's not good enough, you provide feedback on what needs to be improved. "
        "Never give it a pass on the first try. After 5 attempts, you can give it a pass if the story outline is good enough - do not go for perfection"
    ),
    output_type=EvaluationFeedback,
    model=model,
)


async def main() -> None:
    msg = input("What kind of story would you like to hear? ")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    latest_outline: str | None = None

    # We'll run the entire workflow in a single trace
    with trace("LLM as a judge"):
        while True:
            story_outline_result = await Runner.run(
                story_outline_generator,
                input_items,
            )

            input_items = story_outline_result.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(story_outline_result.new_items)
            print("Story outline generated")

            evaluator_result = await Runner.run(evaluator, input_items)
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")

            if result.score == "pass":
                print("Story outline is good enough, exiting.")
                break

            print("Re-running with feedback")

            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    print(f"Final story outline: {latest_outline}")


if __name__ == "__main__":
    asyncio.run(main())
```



## 并行化（Parallelization）

让多个agent并行运行是一种常见模式。这可以用于降低延迟（例如，如果你有多个步骤彼此独立可以同时进行），也可以用于其他用途，比如生成多个不同的结果，再选择其中最优的那个。

**具体示例**

请参见 [`parallelization.py`](./parallelization.py) 文件。示例中会并行多次运行翻译agent，然后选择最好的翻译结果。

```python
import asyncio

from openai import AsyncOpenAI

from agents import (
    Agent,
    ItemHelpers,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
    trace,
)

"""
This example shows the parallelization pattern. We run the agent three times in parallel, and pick the best result.
"""
set_tracing_disabled(disabled=True)
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    model=model,
)

translation_picker = Agent(
    name="translation_picker",
    instructions="You pick the best Spanish translation from the given options.",
    model=model,
)


async def main():
    msg = input("Hi! Enter a message, and we'll translate it to Spanish.\n\n")

    # Ensure the entire workflow is a single trace
    with trace("Parallel translation"):
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(
                spanish_agent,
                msg,
            ),
            Runner.run(
                spanish_agent,
                msg,
            ),
            Runner.run(
                spanish_agent,
                msg,
            ),
        )

        outputs = [
            ItemHelpers.text_message_outputs(res_1.new_items),
            ItemHelpers.text_message_outputs(res_2.new_items),
            ItemHelpers.text_message_outputs(res_3.new_items),
        ]

        translations = "\n\n".join(outputs)
        print(f"\n\nTranslations:\n\n{translations}")

        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n{translations}",
        )

    print("\n\n-----")

    print(f"Best translation: {best_translation.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
```





## 防护措施（Guardrails）

和并行化相关，你经常会希望在Agent处理前运行输入防护措施，以确保交给Agent的输入是有效的。例如，如果有一个客服Agent，你可能想要确保用户不是在请求数学帮助这样的不相关内容。

你完全可以不用特殊的Agents SDK特性，只用并行化来实现这一点。但我们还支持一种特殊的防护措施原语（guardrail primitive）。防护措施可以设置一个“绊线（tripwire）”——一旦被触发，Agent的执行会立刻中止，并抛出一个 `GuardrailTripwireTriggered` 异常。

这种机制对于降低延迟非常有用：比如，你可以用一个非常快的模型来做防护检查，用一个较慢的模型来运行实际的Agent。如果输入无效，无需等待慢模型跑完，防护措施就能快速拒绝无效输入。



**具体示例1**

请参见 [`input_guardrails.py`](./input_guardrails.py) 

这个示例展示了如何使用 guardrails（护栏机制）。

Guardrails 是在智能体执行过程中并行运行的检查程序。 它们可以用于以下场景：

- 检查输入信息是否跑题
- 检查输入信息是否违反相关政策
- 如果检测到意外输入，可以接管智能体的执行流程

在这个示例中，我们设置了一个 input guardrail，用来检测用户是否在请求做数学作业。 如果 guardrail 被触发，我们将返回一个拒绝响应。

```python
from __future__ import annotations

import asyncio
from openai import AsyncOpenAI

from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    set_tracing_disabled,
    OpenAIChatCompletionsModel
)

"""
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that input messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll setup an input guardrail that trips if the user is asking to do math homework.
If the guardrail trips, we'll respond with a refusal message.
"""

set_tracing_disabled(disabled=True)
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

### 1. An agent-based guardrail that is triggered if the user is asking to do math homework
class MathHomeworkOutput(BaseModel):
    reasoning: str
    is_math_homework: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
    model=model,
)

@input_guardrail
async def math_guardrail(context: RunContextWrapper[None], 
                        agent: Agent, 
                        input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    """
    This is an input guardrail function, which happens to call an agent to check if the input
    is a math homework question.
    """
    result = await Runner.run(guardrail_agent, input, context=context.context)
    final_output = result.final_output_as(MathHomeworkOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_math_homework,
    )

### 2. The run loop
async def main():
    agent = Agent(
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
        input_guardrails=[math_guardrail],
        model=model,
    )

    input_data: list[TResponseInputItem] = []
    while True:
        user_input = input("Enter a message: ")
        input_data.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = await Runner.run(agent, input_data)
            print(result.final_output)
            # If the guardrail didn't trigger, we use the result as the input for the next run
            input_data = result.to_input_list()
        except InputGuardrailTripwireTriggered:
            # If the guardrail triggered, we instead add a refusal message to the input
            message = "Sorry, I can't help you with your math homework."
            print(message)
            input_data.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )

    # Sample run:
    # Enter a message: What's the capital of California?
    # The capital of California is Sacramento.
    # Enter a message: Can you help me solve for x: 2x + 5 = 11
    # Sorry, I can't help you with your math homework.

if __name__ == "__main__":
    asyncio.run(main())
```



**具体示例2**

请参见 [`output_guardrails.py`](./output_guardrails.py) 文件。这个示例展示了如何使用输出 guardrails（护栏机制）。

输出 guardrails 是在agent生成最终输出后进行的检查。 它们可以用于以下场景：

- 检查输出内容是否包含敏感数据
- 检查输出是否是对用户消息的有效回复

在本例中将用一个（人为设计的）例子——检查agent的回复中是否包含电话号码650。

```python
from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
    set_tracing_disabled,
)

"""
This example shows how to use output guardrails.

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

In this example, we'll use a (contrived) example where we check if the agent's response contains
a phone number.
"""

set_tracing_disabled(disabled=True)
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

# The agent's output type
class MessageOutput(BaseModel):
    reasoning: str = Field(description="Thoughts on how to respond to the user's message")
    response: str = Field(description="The response to the user's message")
    user_name: str | None = Field(description="The name of the user who sent the message, if known")

@output_guardrail
async def sensitive_data_check(context: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    phone_number_in_response = "650" in output.response
    phone_number_in_reasoning = "650" in output.reasoning

    return GuardrailFunctionOutput(
        output_info={
            "phone_number_in_response": phone_number_in_response,
            "phone_number_in_reasoning": phone_number_in_reasoning,
        },
        tripwire_triggered=phone_number_in_response or phone_number_in_reasoning,
    )

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_type=MessageOutput,
    output_guardrails=[sensitive_data_check],
    model=model,
)

async def main():
    # This should be ok
    await Runner.run(agent, "What's the capital of California?")
    print("First message passed")

    # This should trip the guardrail
    try:
        result = await Runner.run(agent, "My phone number is 650-123-4567. Where do you think I live?")
        print(f"Guardrail didn't trip - this is unexpected. Output: {json.dumps(result.final_output.model_dump(), indent=2)}")
    except OutputGuardrailTripwireTriggered as e:
        print(f"Guardrail tripped. Info: {e.guardrail_result.output.output_info}")

if __name__ == "__main__":
    asyncio.run(main())
```



**具体示例3**

请参见 [`output_guardrails.py`](./output_guardrails.py)文件。这个示例展示了如何在模型流式输出（streaming）时使用 guardrails（护栏机制）。

通常情况下，输出 guardrails 会在agent生成最终输出后再运行；而本例中，guardrails会每生成N个token就运行一次，这样在检测到不良输出时可以提前终止生成。

预期的输出效果是：你会看到一串token被陆续流式输出，然后在guardrail被触发时，流式输出会被立刻中断。

```python
from __future__ import annotations

import asyncio

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

"""
This example shows how to use guardrails as the model is streaming. Output guardrails run after the
final output has been generated; this example runs guardails every N tokens, allowing for early
termination if bad output is detected.

The expected output is that you'll see a bunch of tokens stream in, then the guardrail will trigger
and stop the streaming.
"""

set_tracing_disabled(disabled=True)
model = OpenAIChatCompletionsModel(model="gpt-4.1", openai_client=AsyncOpenAI())

agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpful assistant. You ALWAYS write long responses, making sure to be verbose "
        "and detailed."
    ),
    model=model,
)

class GuardrailOutput(BaseModel):
    reasoning: str = Field(description="Reasoning about whether the response could be understood by a ten year old.")
    is_readable_by_ten_year_old: bool = Field(description="Whether the response is understandable by a ten year old.")

guardrail_agent = Agent(
    name="Checker",
    instructions=(
        "You will be given a question and a response. Your goal is to judge whether the response "
        "is simple enough to be understood by a ten year old."
    ),
    model=model,
    output_type=GuardrailOutput,
)

async def check_guardrail(text: str) -> GuardrailOutput:
    result = await Runner.run(guardrail_agent, text)
    return result.final_output_as(GuardrailOutput)

async def main():
    question = "What is a black hole, and how does it behave? 使用中文回答"
    result = Runner.run_streamed(agent, question)
    current_text = ""

    # We will check the guardrail every N characters
    next_guardrail_check_len = 300
    guardrail_task = None

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
            current_text += event.data.delta

            # Check if it's time to run the guardrail check
            # Note that we don't run the guardrail check if there's already a task running. An
            # alternate implementation is to have N guardrails running, or cancel the previous
            # one.
            if len(current_text) >= next_guardrail_check_len and not guardrail_task:
                print("Running guardrail check")
                guardrail_task = asyncio.create_task(check_guardrail(current_text))
                next_guardrail_check_len += 300

        # Every iteration of the loop, check if the guardrail has been triggered
        if guardrail_task and guardrail_task.done():
            guardrail_result = guardrail_task.result()
            if not guardrail_result.is_readable_by_ten_year_old:
                print("\n\n================\n\n")
                print(f"Guardrail triggered. Reasoning:\n{guardrail_result.reasoning}")
                break

    # Do one final check on the final output
    guardrail_result = await check_guardrail(current_text)
    if not guardrail_result.is_readable_by_ten_year_old:
        print("\n\n================\n\n")
        print(f"Guardrail triggered. Reasoning:\n{guardrail_result.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())

```

输出

```bash
黑洞是一种由极度致密的物质组成的天体，其引力极强，以至于连光线都无法逃脱其引力范围。具体来说，黑洞是当一颗大质量恒星在耗尽燃料后，其核心在引力作用下坍缩到一个体积极小、密度极大的状态时形成的。它的边界被称为“视界”或“事件视界”（Event Horizon），在此范围内，一旦物质或能量进入，就无法再逃脱出来。

**黑洞的主要行为和特性有：**

1. **引力极强**：黑洞的引力如此之大，以致于任何靠近它的物体都会被它吸引，最终落入黑洞内部，不能逃逸。
2. **不发出光**：由于连光线都无法逃脱，黑洞本身不会发出可见光，这也是为何“黑洞”得名。我们通常通过观测黑洞周围的天体或气体的异常运动Running guardrail check
，或辐射来间接探测黑洞的存在。
3. **吞噬物质**：如果有天体或气体靠近黑洞，会被黑洞的引力拉伸、撕碎，然后逐渐吞噬。这一过程可能会在黑洞外围形成亮度极高的吸积盘。
4. **霍金辐射**：根据理论物理学家史蒂芬·霍金的预测，黑洞并非完全“黑暗”，它会以极缓慢的速度向外辐射“霍金辐射”，并最终蒸发、消失。这一现象主要在极小质量的黑洞中才有意义。
5. **时空弯曲**：根据爱因斯坦的广义相对论，黑洞周围的时空被极度弯曲，使得时间和空间在其附近发生巨大变化。例如，靠近黑洞的时间会比

================
Guardrail triggered. Reasoning:
这段内容使用了许多科学术语，比如“致密的物质”、“天体”、“视界”、“事件视界”、“核心坍缩”等，也提到了恒星耗尽燃料、密度极大的状态等复杂物理过程。这些概念对于十岁的小孩来说，理解起来有一定难度。虽然通过举例和简单描述可以帮助他们理解黑洞的强引力和“连光都逃不出去”，但整体文本还是比较偏向成人或具有初步科学知识的青少年。需要更简单直接的语言以及生活化的比喻，才能让十岁的小孩易于理解。
================
Guardrail triggered. Reasoning:
该回答用了一些复杂的科学术语，比如“物质致密”、“恒星坍缩”、“事件视界”、“引力极强”、“霍金辐射”、“广义相对论”等，还涉及了黑洞的理论行为和时空弯曲，对十岁孩子来说不容易全部理解。虽然部分内容（比如黑洞很有引力、会吞噬物体、不发光）较为直观，但总体结构和用词偏难，尤其是论述黑洞形成机制、时空弯曲和霍金辐射部分，超出了十岁孩子的认知范畴。
```



## 强制Agent使用工具（forceing tool use）

**具体示例3**

请参见 [`forcing_tool_use.py`](./forcing_tool_use.py)文件。本示例展示了如何强制智能体使用工具。它通过设置 `ModelSettings(tool_choice="required")`，让智能体必须调用某个工具。



```py
```



你可以用三种方式运行它：

1. `default`：默认行为，即将工具的输出传递给LLM。在这种情况下 `tool_choice`设置为`None`，否则会无限循环——LLM 会一直调用工具，工具运行后把结果给 LLM，然后 LLM 再次被强制调用工具，如此反复。
2. `first_tool_result`：把第一个工具的返回结果作为最终输出。
3. `custom`：使用自定义的工具调用行为函数。这个自定义函数会接收所有工具的返回结果，并选择第一个工具结果作为最终输出。

用法示例：

```bash
python examples/agent_patterns/forcing_tool_use.py -t default
python examples/agent_patterns/forcing_tool_use.py -t first_tool
python examples/agent_patterns/forcing_tool_use.py -t custom
```

输出

```bash
python forcing_tool_use.py -t default
[debug] get_weather called
The weather in Tokyo is sunny with windy conditions. Temperatures range from 14°C to 20°C.

python forcing_tool_use.py -t first_tool       
[debug] get_weather called
city='Tokyo' temperature_range='14-20C' conditions='Sunny with wind'

python forcing_tool_use.py -t custom          
[debug] get_weather called
Tokyo is Sunny with wind.
```


