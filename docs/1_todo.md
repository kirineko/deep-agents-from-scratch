# 1_todo 笔记

这是课程的**第一课**，核心主题是：**用 TODO 列表工具让 Agent 在长任务中保持专注**。

## 背景问题：上下文腐化（Context Rot）

Notebook 开篇引用了两个实际案例：
- **Claude Code** 使用 `TodoWrite` 工具在执行前创建结构化任务列表
- **Manus**（AI Agent 产品）的平均任务使用约 50 次工具调用

当上下文窗口不断增长时，Agent 容易出现"上下文腐化"——遗忘早期目标、偏离主题。解决方案是**不断重写和更新 TODO 列表**，让目标始终出现在上下文末尾，帮助 Agent 保持聚焦。

## 1. 状态定义（DeepAgentState）

在 `state.py` 中定义了扩展状态，在 `AgentState` 基础上新增两个字段：

```python
class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]                         # TODO 列表
    files: Annotated[NotRequired[dict[str, str]], file_reducer]  # 虚拟文件系统（下节课用）
```

其中 `Todo` 的结构为：

```python
class Todo(TypedDict):
    content: str                                        # 任务描述
    status: Literal["pending", "in_progress", "completed"]  # 状态
```

关键设计：`todos` 没有自定义 reducer，每次更新都是**全量覆盖**，这样 LLM 可以在推进过程中重新调整整个任务计划。

## 2. TODO 工具描述（WRITE_TODOS_DESCRIPTION）

提供给 LLM 的工具说明，核心规则：
- **何时使用**：多步骤或复杂任务；用户提供多个任务时
- **结构**：每个 todo 包含 content、status、id
- **最佳实践**：同一时间只有一个 `in_progress` 任务；完成后立即标记；每次发送完整列表
- **进度更新**：实时反映进度，不要批量标完；遇到阻塞时新增描述阻塞原因的任务

## 3. 两个工具实现（todo_tools.py）

### write_todos

将 LLM 生成的 todo 列表写入 state：

```python
@tool
def write_todos(todos: list[Todo], tool_call_id):
    return Command(update={
        "todos": todos,
        "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)]
    })
```

- 参数 `todos` 由 LLM 生成
- 用 `Command` 同时更新 state 中的 `todos` 和 `messages`
- 写入动作本身也会作为 ToolMessage 出现在对话历史中，强化 LLM 的记忆

### read_todos

从 state 中读取当前 todo 列表：

```python
@tool
def read_todos(state: Annotated[DeepAgentState, InjectedState], tool_call_id):
    # 格式化输出，带状态 emoji：⏳ pending / 🔄 in_progress / ✅ completed
```

- 使用 `InjectedState` 注入 state（LLM 看不到这个参数）
- 返回格式化的 todo 列表字符串

## 4. Agent 构建与系统提示词

系统提示词 `TODO_USAGE_INSTRUCTIONS` 规定了 Agent 的工作流程：

1. 收到用户请求后，先用 `write_todos` 创建 TODO 计划
2. 完成一个 TODO 后，用 `read_todos` 重新读取列表（**提醒自己当前进度**）
3. 反思已完成的工作和剩余 TODO
4. 标记任务为 completed，继续下一个
5. 重复直到全部完成

还有两个重要指令：
- **任何请求都要先创建 TODO 计划**
- **尽量将研究任务合并为单个 TODO**，减少跟踪负担

## 5. 演示运行

Notebook 使用一个 **mock 搜索工具**（返回固定的 MCP 协议介绍文本）来演示完整流程。用户问 "Give me a short summary of the Model Context Protocol (MCP)"，Agent 的执行流程大致为：

1. LLM 调用 `write_todos` 创建计划（如：搜索 MCP -> 总结结果）
2. LLM 调用 `web_search` 执行搜索
3. LLM 调用 `read_todos` 回顾进度
4. LLM 调用 `write_todos` 更新状态（标记完成）
5. LLM 输出最终总结

## 核心要点

这节课的关键思想是**通过反复读写 TODO 列表来对抗上下文腐化**：
- TODO 列表作为 Agent 的"工作记忆"存在于 state 中
- 每次读写都会在 messages 中产生新的记录，确保目标始终在上下文窗口的"末尾"附近
- 全量覆盖写入允许 Agent 动态调整计划
- 这是从 Claude Code 和 Manus 等实际产品中提炼出的上下文工程（Context Engineering）技术
