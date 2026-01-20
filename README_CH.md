# OpenAI Agents SDK



## 简介

**openai/openai-agents-python**是OpenAI 官方推出的 **OpenAI Agents SDK**的Python版本。

这是一个轻量级但功能强大的框架，旨在帮助开发者构建和编排复杂的**多智能体（Multi-Agent）工作流**。它不局限于特定的提供商，但主要围绕OpenAI的API（如OpenAI Responses 和 Chat Completions API）构建，以及 100 多种其他大型语言模型 (LLM)。



### 核心目标

该项目旨在解决单一LLM难以处理复杂任务的问题。通过将任务分解，利用多个配置了不同指令和工具的“智能体”协作，来构建更可靠、更强大的 AI 应用。

<img src="https://cdn.openai.com/API/docs/images/orchestration.png" alt="Image of the Agents Tracing UI" style="max-height: 803px;">

> [!NOTE]
>
> 多智能体（Multi-agent）工作流 Trace仪表板
>
> 这是一个用于**调试和监控 AI 智能体工作流**的开发者工具界面（AI 编排平台的后台监控界面）。它帮助开发者理解多个 AI 智能体是如何协作完成一个复杂任务（如处理保险索赔）的，包括每个步骤的耗时、调用的工具以及具体的输入输出
>
> 1. 执行流程可视化（左侧/中间）：
>    - 显示一个任务是如何在不同的 **智能体（Agent）** 之间流转
>    - 三个主要的智能体步骤：
>      - **Triage Agent（分类/分诊智能体）：** 耗时 1,233 ms。
>      - **Approval Agent（审批智能体）：** 耗时 4,320 ms。这是最复杂的步骤，包含多次 API 调用 (`POST /v1/responses`) 和工具调用（`fetch_data`, `check_eligibility`, `send_email`）。
>      - **Summarizer Agent（总结智能体）：** 耗时 2,151 ms。
>    - 还显示了智能体之间的 **Handoff（交接）** 过程。
> 2. 详细请求信息（右侧）：
>    - 右侧面板显示了选中的某个操作（这里是一个 `POST /v1/responses` 请求）的详细信息。
>    - **Properties（属性）：** 包含创建时间、ID、使用的模型（`gpt-4o-2024-08-06`）、消耗的 Token 数（499 total）以及可用的函数/工具列表。
>    - **Instructions（指令）：** 显示了给 AI 的系统提示词（System Instructions）。这段指令描述了该智能体的具体任务：获取索赔详情、判断是否符合审批条件、起草并发送邮件、最后进行总结。
>



### 核心概念

1. **智能体 (Agents)：** 框架的基本单元。每个Agent都是一个配置了特定指令（System Prompt）、工具（Tools）、护栏（Guardrails）和交接机制的LLM。
2. **交接 (Handoffs)：** Agents SDK使用的一种**专用工具调用**，用于**在智能体之间转移控制权**。这是框架的一大特色。它允许一个智能体在执行过程中，将当前的对话控制权完全“移交”给另一个更专业的智能体（例如，一个负责“接待”的智能体将用户转接给负责“写代码”的智能体）。
3. **护栏 (Guardrails)：** 用于输入和输出验证的可配置**安全检查**。确保智能体的行为符合预期，防止恶意输入或错误输出。
4. **会话 (Sessions)：** 跨智能体运行的自动**对话历史管理**。能够自动维护跨多次运行的对话历史记录（Memory），支持内存、SQLite 或 Redis 等多种存储方式。
5. **追踪 (Tracing)：** 内置的智能体**运行跟踪**功能，方便开发者查看、调试和优化智能体之间的交互流程。

请探索 examples 目录以了解 SDK 的实际应用，并阅读我们的文档以获取更多详情。



### 支持的模式

该框架支持多种常见的 Agent 设计模式，包括：

- **确定性流程 (Deterministic flows)**：按预定步骤**顺序执行任务**。
- **路由与分发 (Routing)**：根据用户意图将任务分发给不同的专家智能体。
- **智能体作为工具 (Agents as tools)**：一个智能体可以将另一个智能体像**调用函数**一样调用，获取结果后再继续。
- **并行处理 (Parallelization)**：同时运行多个智能体以提高效率。
- **LLM 评审 (LLM-as-a-judge)**：使用一个模型来评估和改进另一个模型的输出。



### 适用人群

如果你正在使用Python开发AI应用，并且发现**单一的Prompt或简单的API调用**无法满足复杂的业务逻辑（需要状态管理、多步骤推理、角色切换等），那么这个框架非常适合你。



## Get started

To get started, set up your Python environment (Python 3.9 or newer required), and then install OpenAI Agents SDK package.

### venv

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install openai-agents
```

For voice support, install with the optional `voice` group: `pip install 'openai-agents[voice]'`.

### uv

If you're familiar with [uv](https://docs.astral.sh/uv/), using the tool would be even similar:

```bash
uv init
uv add openai-agents
```

For voice support, install with the optional `voice` group: `uv add 'openai-agents[voice]'`.



## Hello world example

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(_If running this, ensure you set the `OPENAI_API_KEY` environment variable_)

(_For Jupyter notebook users, see [hello_world_jupyter.ipynb](examples/basic/hello_world_jupyter.ipynb)_)



## Handoffs example

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)

async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?

if __name__ == "__main__":
    asyncio.run(main())
```



## Functions example

```python
import asyncio

from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)

async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.

if __name__ == "__main__":
    asyncio.run(main())
```



## Agent 循环 (The agent loop)

当你调用 `Runner.run()` 时，我们会运行一个循环，直到获得最终输出为止。

1. 调用LLM，会使用 agent上的模型、设置和消息历史作为输入。
2. LLM返回一个响应，这个响应可能包含工具调用。
3. 如果响应中含有最终输出（final output，详见下文），我们就返回它并结束循环。
4. 如果响应中包含分派（handoff），我们会把 agent 切换到新的 agent，然后回到第1步。
5. 我们处理所有工具调用（如有），并把工具调用结果信息加入消息队列。然后回到第1步。

另外，还有一个 `max_turns` 参数，你可以用它来限制循环执行的最大轮数。

> [!NOTE]
>
> 这段话描述了**agent**在调用 `Runner.run()` 时的完整决策与交互流程：
>
> - 每一轮，LLM 都会产生可能涉及工具调用的回答；
>- 如果给出的就是最终答案，循环立即结束；
> - 如果需要分派到另一个 agent，就切换后重头再来一轮；
> - 如果涉及工具，则实际调用工具，拿到结果后继续反馈给 LLM，循环继续下去；
> - 你可以通过 `max_turns` 限制最多循环多少步（避免死循环）。
> 
> **本质：这是一个自动决策+工具/分派增强的“对话状态机”，直到产生最终结果或达到最大步数为止。**



### 最终输出

最终输出是agent在循环中产生的最后结果。

1. 如果你在 agent 上设置了 `output_type`，那么最终输出就是当 LLM 返回该类型的数据时。对于这种情况，我们使用[结构化输出](https://platform.openai.com/docs/guides/structured-outputs)。
2. 如果没有设置 `output_type`（比如只返回纯文本），那么**第一个没有任何工具调用或分派（handoff）的 LLM 响应**就会被视作最终输出。

因此，agent循环的思维模型是：

1. 如果当前 agent 有 `output_type`，循环会持续，直到agent生成一个匹配该类型的结构化输出。
2. 如果当前 agent 没有 `output_type`，循环会持续，直到当前 agent 生成一个没有任何工具调用或handoffs的消息。

这就是 agent 流程的“输出终止判据”，“什么样的响应算整个自动链路结束、可以把结果返回给用户”。



## 常见Agentic模式

Agents SDK 设计得非常灵活，允许你构建各种 LLM 工作流，包括**确定性流程**、**迭代循环**等。可以参考 [`examples/agent_patterns`](examples/agent_patterns) 目录中的示例。这里有大篇幅的介绍和代码。



## 追踪（Tracing）

Agents SDK 会自动追踪你的agent运行，使你能够轻松追踪和调试agent的行为。追踪功能本身是可扩展的，支持自定义追踪片段（span）以及多种外部追踪目的地，包括 [Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents)、[AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk)、[Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk)、[Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration) 和 [Keywords AI](https://docs.keywordsai.co/integration/development-frameworks/openai-agent)。关于如何自定义或关闭追踪功能的更多细节，请参见[Tracing](http://openai.github.io/openai-agents-python/tracing)，那里还包含更多 [外部追踪集成工具的列表](http://openai.github.io/openai-agents-python/tracing/#external-tracing-processors-list)。



## 长时间运行的 Agent 与human-in-the-loop

你可以使用 Agents SDK 的 [Temporal](https://temporal.io/) 集成来运行持久的、长时间运行的工作流，包括human-in-the-loop（人类参与流程）任务。你可以[在这个视频](https://www.youtube.com/watch?v=fFBZqzT4DD8)中观看 Temporal与Agents SDK 协作完成长时任务的演示，并且可以[在这里查看文档](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents)。



## 会话（Sessions）

Agents SDK 内置了会话记忆功能，能够在多次agent 运行中**自动维护对话历史**，无需你在每轮交互之间手动处理 `.to_input_list()`。

### Quick start

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a session instance
session = SQLiteSession("conversation_123")

# First turn
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# Second turn - agent automatically remembers previous context
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"

# Also works with synchronous runner
result = Runner.run_sync(
    agent,
    "What's the population?",
    session=session
)
print(result.final_output)  # "Approximately 39 million"
#San Francisco.
#California.
#As of 2023, San Francisco's population is approximately 808,000.
```



### Session options会话选项

- **无记忆**（默认）：如果未传递 session 参数，则对话无会话记忆
- **`session: Session = DatabaseSession(...)`**：使用 Session 实例来管理对话历史

```python
from agents import Agent, Runner, SQLiteSession

# Custom SQLite database file
session = SQLiteSession("user_123", "conversations.db")
agent = Agent(name="Assistant")

# Different session IDs maintain separate conversation histories
result1 = await Runner.run(
    agent,
    "Hello",
    session=session
)
result2 = await Runner.run(
    agent,
    "Hello",
    session=SQLiteSession("user_456", "conversations.db")
)
```



### 自定义Session实现

可以通过创建一个遵循 `Session` 协议的类，来实现自定义的会话记忆。

```python
from agents.memory import Session
from typing import List

class MyCustomSession:
    """Custom session implementation following the Session protocol."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Your initialization here

    async def get_items(self, limit: int | None = None) -> List[dict]:
        # Retrieve conversation history for the session
        pass

    async def add_items(self, items: List[dict]) -> None:
        # Store new items for the session
        pass

    async def pop_item(self) -> dict | None:
        # Remove and return the most recent item from the session
        pass

    async def clear_session(self) -> None:
        # Clear all items for the session
        pass

# Use your custom session
agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    session=MyCustomSession("my_session")
)
```



## 开发环境搭建（仅当需要编辑SDK或示例代码时）

0. Ensure you have [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv --version
```

1. 安装依赖项

```bash
make sync
```

2. （进行了代码修改后）运行代码检查和测试

```bash
make check # 运行测试、代码风格检查器和类型检查器
```

Or to run them individually:

```bash
make tests        # 运行测试
make mypy         # 运行类型检查器
make lint         # 运行代码风格检查
make format-check # 运行代码格式检查
```



## 致谢

我们要感谢开源社区的杰出工作，特别是：

- [Pydantic](https://docs.pydantic.dev/latest/)（数据校验）和 [PydanticAI](https://ai.pydantic.dev/)（高级智能体框架）
- [LiteLLM](https://github.com/BerriAI/litellm)（支持100+大模型的统一接口）
- [MkDocs](https://github.com/squidfunk/mkdocs-material)
- [Griffe](https://github.com/mkdocstrings/griffe)
- [uv](https://github.com/astral-sh/uv) 和 [ruff](https://github.com/astral-sh/ruff)（py代码风格检查工具）

我们承诺将持续以开源框架的方式构建 Agents SDK，方便社区成员能够基于我们的思路进行拓展和创新。



## 其他

```bash
# 只检查import问题(比如排序,分组等), 并且自动修复
ruff check --select I --fix .
```

import 顺序会由 Ruff 自动调整如下格式：

1. 标准库
2. 第三方包
3. 本项目模块
4. 各分组间有空行
5. 每组内字典序排序



参考文档：https://openai.github.io/openai-agents-python/
