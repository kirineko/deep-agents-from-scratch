# 2_files: 上下文卸载 — 虚拟文件系统

## 核心思想

随着 Agent 执行复杂任务，上下文窗口会迅速膨胀（Manus 的平均任务约使用 50 次工具调用）。**上下文卸载（Context Offloading）** 是一种关键的上下文工程技术：与其将所有工具调用结果和中间数据直接保留在上下文窗口中，不如让 Agent 将信息写入文件，在需要时再按需读取。

这种模式在 Manus、Hugging Face Open Deep Research 以及 Anthropic 的多 Agent 研究系统中都得到了成功应用。它的核心优势在于：

- **防止信息退化**：避免信息在多个 Agent 之间传递时产生的"传话游戏"效应
- **保持上下文清洁**：新生成的子 Agent 可以从干净的上下文出发，按需获取已存储的信息
- **适用于长期运行的任务**：中间结果被持久化，无需持续占用上下文注意力

## 架构设计

### 虚拟文件系统

本节采用了一个**虚拟文件系统**方案——使用一个 Python 字典模拟传统文件系统，存储在 LangGraph 的 Agent State 中：

- **键（key）**：文件路径/文件名
- **值（value）**：文件内容（纯文本）

这种方式提供线程级的短期持久化，适合在单次 Agent 对话内维持上下文，但不适用于跨对话线程的信息持久化。

### State 定义

在 `state.py` 中，`DeepAgentState` 继承了 LangGraph 的 `AgentState`，并扩展了两个字段：

```python
class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
```

其中 `files` 字段使用了自定义的 `file_reducer` 作为 reducer 函数：

```python
def file_reducer(left, right):
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}
```

这个 reducer 的作用是在状态更新时合并文件字典：`{**left, **right}` 先展开已有文件，再展开新文件，重复的键会被新值覆盖。这使得 `write_file` 工具可以通过 LangGraph 的 `Command` 类型增量更新虚拟文件系统。

## 三个文件工具

### 1. `ls()` — 列出文件

列出虚拟文件系统中所有已存在的文件。无需参数，返回文件路径列表。

```python
@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    return list(state.get("files", {}).keys())
```

用途：在开始文件操作前定位方向，了解当前有哪些文件可用。

### 2. `read_file()` — 读取文件

从虚拟文件系统中读取指定文件的内容，支持分页读取。

```python
@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(file_path: str, state, offset: int = 0, limit: int = 2000) -> str:
```

参数：
- `file_path`（必填）：要读取的文件路径
- `offset`（可选，默认 0）：起始行号
- `limit`（可选，默认 2000）：最多读取行数

返回内容带有行号（类似 `cat -n`），长行会被截断到 2000 字符。分页能力避免了将大文件一次性全部加载到上下文中。

### 3. `write_file()` — 写入文件

创建新文件或完全覆盖已有文件。

```python
@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(file_path: str, content: str, state, tool_call_id) -> Command:
```

参数：
- `file_path`（必填）：文件路径
- `content`（必填）：写入的完整内容

该工具返回一个 `Command` 对象，用于更新 Agent State 中的 `files` 字典，并返回一条 `ToolMessage` 确认操作。

## 设计要点

### 工具描述与 Prompt 分离

工具使用 `@tool(description=PROMPT)` 装饰器将描述传递给 LLM（此时 docstring 被抑制）。将较长的描述放在单独的 `prompts.py` 文件中，有更大的空间来解释工具的操作方式和使用场景。

### 面向 LLM 的错误信息

错误消息的设计对象是 LLM 而非人类用户。在 Agentic 系统中，LLM 可以利用错误信息中的细节来重试操作，例如：

- `"Error: File '{file_path}' not found"` — LLM 可以据此检查文件名或先调用 `ls()`
- `"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"` — LLM 可以调整 offset

### InjectedState 和 InjectedToolCallId

`state` 和 `tool_call_id` 参数使用 `Annotated` 类型标注（`InjectedState`、`InjectedToolCallId`），它们不会暴露给 LLM，而是由 LangGraph 的 Tool Node 在运行时自动注入。

## 实际演示

Notebook 中构建了一个简单的研究 Agent 来演示文件系统的使用。Agent 的工作流程为：

1. **定位**：调用 `ls()` 查看现有文件
2. **保存**：调用 `write_file()` 将用户请求存储到文件中
3. **研究**：调用 `web_search` 工具搜索信息
4. **读取**：调用 `read_file()` 读回保存的文件，确保准确回答用户问题

在示例中，用户请求"Give me an overview of Model Context Protocol (MCP)"后，Agent 先将请求保存到 `user_request.txt`，再搜索并组织答案。运行结束后，`result["files"]` 中可以看到：

```python
{'user_request.txt': 'User asked: Give me an overview of Model Context Protocol (MCP).'}
```

虽然在这个简单示例中信息完全可以保留在上下文窗口内，但对于**长期运行的 Agent 任务**，上下文内容可能被压缩或丢弃。在压缩前将信息存入文件，并在需要时按需检索，正是智能上下文工程的核心实践。

## 与教程整体的关系

本节是教程系列的第二部分：

| 序号 | 主题 | 核心能力 |
|------|------|----------|
| 0 | create_agent | 基础 Agent 创建 |
| 1 | TODO 工具 | 任务规划与进度追踪 |
| **2** | **文件系统** | **上下文卸载与信息持久化** |
| 3 | 子 Agent | 任务委托与上下文隔离 |
| 4 | 完整 Agent | 组合所有能力 |

文件系统工具为后续的子 Agent（Notebook 3）和完整 Agent（Notebook 4）提供了基础：子 Agent 可以将研究结果写入文件，协调 Agent 通过读取文件获取结果，而不必通过上下文传递大量信息。
