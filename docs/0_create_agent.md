# 0_create_agent 笔记

这是整个课程的**第零课（入门课）**，目标是介绍 LangChain 预构建的 **ReAct Agent** 抽象，为后续构建"深度 Agent"打基础。内容分为四大块：

## 1. 什么是 ReAct Agent

ReAct = **Re**asoning + **Act**ing，源自论文 *ReAct: Synergizing Reasoning and Acting in Language Models*。

一个 ReAct Agent 由三个部分组成：
- **LLM**（大语言模型）
- **Tools**（工具集）
- **Prompt**（系统提示词）

运行机制是一个**循环**：LLM 查看上下文 -> 决定是否调用工具 -> 调用工具 -> 接收工具返回的结果（observation）-> 继续推理或结束。循环直到 LLM 认为不需要再调用工具为止。

## 2. 用工具构建 Agent（基础版）

Notebook 用一个 `calculator` 工具做演示：

```python
@tool
def calculator(operation, a, b):
    # 支持 add/subtract/multiply/divide
```

然后用 `create_agent()` 创建 agent：

```python
agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
```

底层会编译成一个 LangGraph 的 `CompiledStateGraph`，包含两个节点：**LLM 节点** 和 **ToolNode**（工具执行节点）。

调用示例：用户问 "What is 3.1 * 4.2?"，LLM 生成一个带 `tool_calls` 的 `AIMessage`，ToolNode 执行计算返回 `ToolMessage`，LLM 再组织最终回答。

> 注意：`create_react_agent` 在 LangChain 1.0 中已被移到 `langchain` 包并重命名为 `create_agent`。

## 3. 图、状态与消息（Graph, State, Messages）

默认使用的状态是 `AgentState`，核心就是一个 `messages` 列表：

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
```

- `add_messages` 是 reducer，负责把新消息追加到列表末尾
- `remaining_steps` 跟踪图的递归步数

## 4. 在工具中访问和修改状态（重点）

这是本节最核心的进阶内容，分三步讲解：

### a) 自定义 State

扩展 `AgentState`，增加一个 `ops` 字段记录所有运算历史：

```python
class CalcState(AgentState):
    ops: Annotated[List[str], reduce_list]
```

### b) 注入状态（InjectedState）

工具需要访问 state，但 LLM 不知道 state 的存在。解决方案是用 `InjectedState` 注解，让 ToolNode 在执行时自动注入：

```python
def calculator_wstate(
    operation, a, b,
    state: Annotated[CalcState, InjectedState],        # LLM 看不到这个参数
    tool_call_id: Annotated[str, InjectedToolCallId],  # LLM 也看不到
):
```

### c) 用 Command 更新状态

工具不再直接返回值，而是返回 `Command` 对象，同时更新 `ops` 和 `messages`：

```python
return Command(update={
    "ops": [f"({operation}, {a}, {b}),"],
    "messages": [ToolMessage(f"{result}", tool_call_id=tool_call_id)]
})
```

最后还演示了**并行工具调用**的场景：计算 `3.1 * 4.2 + 5.5 * 6.5` 时，LLM 在一轮中同时发出两个 multiply 调用，ToolNode 并行执行，然后 LLM 再发一个 add 调用汇总结果。

## 总结

这节课从零开始，通过一个计算器示例，教会了如何用 `create_agent` 构建 ReAct Agent、理解消息流转机制、自定义状态、以及通过 `InjectedState` + `Command` 在工具中读写图状态。这些都是后续课程（TODO 工具、虚拟文件系统、子 Agent）的基础。
