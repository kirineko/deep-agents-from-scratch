# 4_full_agent 完整深度智能体详解

本文档详细解释 `notebooks/4_full_agent.ipynb` notebook 中的完整深度智能体(Deep Agent)实现。

## 概述

本节将前面学习的所有组件整合在一起，构建一个完整的深度研究智能体：

- **TODO 工具**：跟踪任务进度
- **文件系统**：存储原始工具调用结果
- **子智能体委托**：实现上下文隔离的研究任务

```
┌─────────────────────────────────────────────────────────────────┐
│                     Deep Agent 完整架构                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  TODO 工具   │  │  文件系统   │  │    子智能体委托          │ │
│  │ write_todos │  │    ls       │  │                         │ │
│  │ read_todos  │  │ read_file   │  │  task(description,      │ │
│  └─────────────┘  │ write_file  │  │        subagent_type)   │ │
│                   └─────────────┘  └─────────────────────────┘ │
│                                                                 │
│                          ▼                                      │
│              ┌─────────────────────┐                           │
│              │   research-agent    │                           │
│              │  (子智能体)          │                           │
│              │  - tavily_search    │                           │
│              │  - think_tool       │                           │
│              └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件：搜索工具

搜索工具实现了**上下文卸载(Context Offloading)**模式，将原始内容保存到文件，只返回摘要给智能体。这是长时间运行的智能体轨迹中的常见模式。

### 1. 搜索执行 (`run_tavily_search`)

使用 Tavily API 执行实际的网络搜索：

```python
def run_tavily_search(
    search_query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """使用 Tavily API 执行搜索"""
    result = tavily_client.search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )
    return result
```

### 2. 内容摘要 (`summarize_webpage_content`)

使用轻量级模型生成网页内容的结构化摘要：

```python
class Summary(BaseModel):
    """网页内容摘要的模式"""
    filename: str = Field(description="存储文件的名称")
    summary: str = Field(description="网页的关键学习内容")

def summarize_webpage_content(webpage_content: str) -> Summary:
    """使用配置的摘要模型总结网页内容"""
    structured_model = summarization_model.with_structured_output(Summary)
    summary_and_filename = structured_model.invoke([
        HumanMessage(content=SUMMARIZE_WEB_SEARCH.format(
            webpage_content=webpage_content,
            date=get_today_str()
        ))
    ])
    return summary_and_filename
```

### 3. 结果处理 (`process_search_results`)

获取完整网页内容，将 HTML 转换为 markdown，并为每个结果生成摘要：

```python
def process_search_results(results: dict) -> list[dict]:
    """处理搜索结果，在可用时总结内容"""
    processed_results = []
    HTTPX_CLIENT = httpx.Client(timeout=30.0)

    for result in results.get('results', []):
        url = result['url']
        response = HTTPX_CLIENT.get(url)

        if response.status_code == 200:
            # 将 HTML 转换为 markdown
            raw_content = markdownify(response.text)
            summary_obj = summarize_webpage_content(raw_content)
        # ... 错误处理

        processed_results.append({
            'url': result['url'],
            'title': result['title'],
            'summary': summary_obj.summary,
            'filename': summary_obj.filename,
            'raw_content': raw_content,
        })

    return processed_results
```

### 4. 上下文卸载 (`tavily_search` 工具)

这是搜索工具的核心，实现上下文卸载模式：

```python
@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """搜索网络并将详细结果保存到文件，同时返回最小上下文"""

    # 1. 执行搜索
    search_results = run_tavily_search(query, max_results, topic)

    # 2. 处理和总结结果
    processed_results = process_search_results(search_results)

    # 3. 将每个结果保存到文件（上下文卸载）
    files = state.get("files", {})
    for result in processed_results:
        files[result['filename']] = f"""# Search Result: {result['title']}
**URL:** {result['url']}
## Summary
{result['summary']}
## Raw Content
{result['raw_content']}
"""

    # 4. 返回最小摘要给智能体（防止上下文膨胀）
    summary_text = f"🔍 Found {len(processed_results)} result(s)..."

    return Command(
        update={
            "files": files,  # 完整内容保存到文件
            "messages": [
                ToolMessage(summary_text, tool_call_id=tool_call_id)  # 只返回摘要
            ],
        }
    )
```

**关键点**：
- 完整的搜索结果保存到虚拟文件系统
- 只有摘要返回给智能体的上下文
- 这解决了 token 效率问题，保持智能体的工作上下文最小化和聚焦

### 5. 战略思考 (`think_tool`)

提供结构化的反思机制：

```python
@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """用于研究进度和决策的战略反思工具

    使用时机：
    - 收到搜索结果后：我找到了什么关键信息？
    - 决定下一步之前：我有足够的信息来全面回答吗？
    - 评估研究差距时：我还缺少什么具体信息？
    - 结束研究前：我现在能提供完整答案吗？

    反思应该包括：
    1. 当前发现分析 - 我收集了什么具体信息？
    2. 差距评估 - 还缺少什么关键信息？
    3. 质量评估 - 我有足够的证据/示例吗？
    4. 战略决策 - 应该继续搜索还是提供答案？
    """
    return f"Reflection recorded: {reflection}"
```

## 完整智能体构建

### 工具配置

```python
# 子智能体工具 - 用于研究
sub_agent_tools = [tavily_search, think_tool]

# 主智能体内置工具
built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]

# 研究子智能体配置
research_sub_agent = {
    "name": "research-agent",
    "description": "委托研究给子智能体研究员。每次只给这个研究员一个主题。",
    "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": ["tavily_search", "think_tool"],
}

# 创建任务委托工具
task_tool = _create_task_tool(
    sub_agent_tools, [research_sub_agent], model, DeepAgentState
)

# 所有工具
all_tools = built_in_tools + [task_tool]
```

### 研究员指令

研究子智能体有专门的指令来指导其行为：

```xml
<Task>
你的工作是使用工具收集关于用户输入主题的信息。
你可以串行或并行调用这些工具，你的研究在工具调用循环中进行。
</Task>

<Instructions>
像一个时间有限的人类研究员一样思考：
1. 仔细阅读问题 - 用户需要什么具体信息？
2. 从更广泛的搜索开始 - 首先使用广泛、全面的查询
3. 每次搜索后，暂停评估 - 我有足够的信息来回答吗？还缺少什么？
4. 随着信息收集执行更窄的搜索 - 填补空白
5. 当你能自信地回答时停止 - 不要为了完美而继续搜索
</Instructions>

<Hard Limits>
工具调用预算（防止过度搜索）：
- 简单查询：最多使用 1-2 次搜索工具调用
- 普通查询：最多使用 2-3 次搜索工具调用
- 非常复杂的查询：最多使用 5 次搜索工具调用
- 始终在 5 次搜索工具调用后停止

立即停止当：
- 你可以全面回答用户的问题
- 你有 3+ 个相关示例/来源
- 最后 2 次搜索返回了类似的信息
</Hard Limits>
```

### 主智能体指令

主智能体的完整指令整合了所有组件：

```
# TODO 管理
[TODO_USAGE_INSTRUCTIONS]

================================================================================

# 文件系统使用
[FILE_USAGE_INSTRUCTIONS]

================================================================================

# 子智能体委托
[SUBAGENT_USAGE_INSTRUCTIONS]
```

## 数据流

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      主智能体                                │
│                                                             │
│  1. 使用 write_todos 创建任务计划                            │
│  2. 使用 task 工具委托研究任务给 research-agent              │
│                                                             │
│     ┌─────────────────────────────────────────────────┐    │
│     │              research-agent                      │    │
│     │                                                  │    │
│     │  1. 执行 tavily_search                           │    │
│     │  2. 搜索结果 → 文件系统（上下文卸载）              │    │
│     │  3. 摘要 → 返回给智能体                           │    │
│     │  4. 使用 think_tool 反思                          │    │
│     │  5. 重复直到有足够信息                            │    │
│     │  6. 返回研究结果                                  │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
│  3. 接收子智能体结果                                         │
│  4. 使用 read_file 访问详细内容（如需要）                    │
│  5. 使用 write_file 存储最终报告                            │
│  6. 更新 todos 状态                                         │
│  7. 返回最终答案给用户                                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
最终答案 + 文件系统中的研究资料
```

## 使用 deepagents 包

项目提供了 `deepagents` 包作为简单的抽象层：

```python
from deepagents import create_deep_agent

# 创建研究子智能体配置
research_sub_agent = {
    "name": "research-agent",
    "description": "委托研究给子智能体研究员",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    "tools": [tavily_search, think_tool],
}

# 创建深度智能体
agent = create_deep_agent(
    tools=sub_agent_tools,
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    model=model,
)

# 调用智能体
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "给我一个关于 Model Context Protocol (MCP) 的简要概述",
    }],
})
```

`deepagents` 包已经包含：
- 文件系统工具
- TODO 工具
- Task 工具

你只需要提供子智能体配置和子智能体使用的工具。

## 关键设计模式总结

| 模式 | 说明 | 解决的问题 |
|------|------|-----------|
| **上下文卸载** | 将详细内容保存到文件，只返回摘要 | Token 效率、上下文膨胀 |
| **TODO 追踪** | 使用结构化任务列表管理进度 | 复杂任务的组织和追踪 |
| **子智能体委托** | 将任务委托给隔离的专业智能体 | 上下文冲突、专业化需求 |
| **战略反思** | 使用 think_tool 进行结构化思考 | 决策质量、研究效率 |
| **虚拟文件系统** | 在状态中维护文件系统 | 持久化、回溯支持 |

## 最佳实践

1. **搜索预算管理**：为不同复杂度的查询设置合理的搜索次数限制
2. **反思驱动的决策**：每次搜索后使用 think_tool 评估进度
3. **上下文最小化**：只将必要的摘要保留在工作上下文中
4. **任务分解**：使用 TODO 工具将复杂任务分解为可管理的步骤
5. **并行研究**：对独立的研究方向使用多个子智能体并行处理

## 与前面章节的关系

| 章节 | 引入的概念 | 在本章的应用 |
|------|-----------|-------------|
| 1_todo | TODO 工具 | 任务规划和进度追踪 |
| 2_files | 虚拟文件系统 | 搜索结果的上下文卸载 |
| 3_subagents | 子智能体委托 | research-agent 的隔离执行 |
| **4_full_agent** | **完整整合** | **所有组件协同工作** |

这个完整的深度智能体展示了如何将所有上下文工程技术组合在一起，构建一个强大、高效的研究智能体系统。
