# 3_subagents 子智能体详解

本文档详细解释 `notebooks/3_subagents.ipynb` notebook 中的子智能体(Sub-agents)概念和实现。

## 核心问题：上下文膨胀

随着对话的进行，Agent 的上下文会快速增长，这会导致几个严重问题：

- **上下文冲突(Context Clash)**：同一上下文窗口中混合多个目标，导致性能下降
- **上下文混淆(Context Confusion)**：Agent 难以区分不同任务的相关信息
- **上下文污染(Context Poisoning)**：无关信息干扰当前任务的执行
- **上下文稀释(Context Dilution)**：重要信息被大量无关内容稀释

## 解决方案：上下文隔离

**上下文隔离(Context Isolation)** 通过将任务委托给专门的子智能体来解决这些问题。每个子智能体在自己独立的上下文窗口中运行，实现：

- 防止上下文冲突和混淆
- 支持专注的、专业化的任务执行
- 保持父智能体上下文的整洁

## 架构设计

```
┌─────────────────────────────────────────────────────┐
│                   主智能体(Supervisor)               │
│                                                     │
│  task(description, subagent_type)                   │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────────────────────────────────┐   │
│  │           子智能体注册表(Registry)            │   │
│  │                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐        │   │
│  │  │research-agent│  │ other-agent  │  ...   │   │
│  │  └──────────────┘  └──────────────┘        │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  子智能体结果 → ToolMessage → 返回主智能体          │
└─────────────────────────────────────────────────────┘
```

## 核心概念

### 1. SubAgent 配置类型

```python
class SubAgent(TypedDict):
    """专门子智能体的配置"""
    name: str           # 子智能体名称，用于调用时指定
    description: str    # 描述，告诉主智能体何时使用此子智能体
    prompt: str         # 系统提示词，指导子智能体如何执行任务
    tools: NotRequired[list[str]]  # 可选的工具列表
```

子智能体具有**双重角色**：
- **作为工具**：为主智能体提供关于其能力的信息，以及如何调用它
- **作为智能体**：需要提示词来指导任务执行，以及完成任务所需的工具集

### 2. task 工具

`task` 工具是实现上下文隔离的核心：

```python
def task(
    description: str,     # 子智能体需要执行的任务描述
    subagent_type: str,   # 使用哪个子智能体
    state: DeepAgentState,
    tool_call_id: str,
):
    # 1. 验证子智能体类型是否存在
    if subagent_type not in agents:
        return f"Error: ..."

    # 2. 获取对应的子智能体
    sub_agent = agents[subagent_type]

    # 3. 创建隔离的上下文 - 只包含任务描述
    # 这是上下文隔离的关键：不包含父智能体的历史
    state["messages"] = [{"role": "user", "content": description}]

    # 4. 在隔离环境中执行子智能体
    result = sub_agent.invoke(state)

    # 5. 通过 Command 返回结果给父智能体
    return Command(
        update={
            "files": result.get("files", {}),  # 合并文件更改
            "messages": [
                ToolMessage(
                    result["messages"][-1].content,
                    tool_call_id=tool_call_id
                )
            ],
        }
    )
```

### 3. 上下文隔离的关键

```python
# 这一行是上下文隔离的核心
state["messages"] = [{"role": "user", "content": description}]
```

子智能体接收的是**全新的消息列表**，只包含任务描述，完全不包含父智能体的对话历史。这确保了：
- 子智能体专注于单一任务
- 不受父智能体上下文的干扰
- 上下文窗口保持清洁和高效

## 实际示例

### 创建研究子智能体

```python
# 研究子智能体配置
research_sub_agent = {
    "name": "research-agent",
    "description": "委托研究任务给子智能体研究员。每次只给这个研究员一个主题。",
    "prompt": SIMPLE_RESEARCH_INSTRUCTIONS,
    "tools": ["web_search"],  # 只提供搜索工具
}
```

### 主智能体的使用指南

主智能体的提示词需要包含如何使用子智能体的说明：

```xml
<Available Tools>
1. **task(description, subagent_type)**: 将研究任务委托给专门的子智能体
   - description: 清晰、具体的研究问题或任务
   - subagent_type: 要使用的智能体类型（如 "research-agent"）
2. **think_tool(reflection)**: 反思每个委托任务的结果并规划下一步
</Available Tools>
```

## 缩放规则

### 简单任务
单个子智能体即可：
- 示例："列出旧金山排名前10的咖啡店" → 使用 1 个子智能体

### 比较任务
为每个比较元素使用一个子智能体：
- 示例："比较 OpenAI vs Anthropic vs DeepMind 的 AI 安全方法" → 使用 3 个子智能体
- 将发现存储在单独的文件中

### 多方面研究
为不同方面使用并行智能体：
- 示例："研究可再生能源：成本、环境影响和采用率" → 使用 3 个子智能体
- 按方面组织发现

## 重要提醒

1. **每个 task 调用创建一个独立的研究智能体**，具有隔离的上下文
2. **子智能体之间无法看到彼此的工作**，需要提供完整的独立指令
3. **使用清晰、具体的语言**，避免在任务描述中使用缩写
4. **适时停止**：当有足够信息时不要过度研究
5. **限制迭代**：设置最大委托次数以防止无限循环

## 与其他组件的集成

子智能体系统与之前的组件无缝集成：
- **TODO 工具**：主智能体可以使用 TODO 来规划需要委托的任务
- **文件系统**：子智能体的结果可以写入虚拟文件系统，供后续使用
- **状态管理**：通过 `Command` 类型，子智能体的文件更改会合并回主智能体状态

## 总结

子智能体模式是构建复杂 Agent 系统的关键技术：

| 特性 | 说明 |
|------|------|
| 上下文隔离 | 每个子智能体有独立的上下文窗口 |
| 专业化 | 子智能体可以有不同的工具集和提示词 |
| 并行执行 | 可以同时运行多个独立的子智能体 |
| 状态同步 | 子智能体的结果自动合并回主智能体 |
| 可扩展性 | 通过注册表轻松添加新的子智能体类型 |

这种模式特别适合：
- 多步骤研究任务
- 需要不同专业能力的复杂任务
- 需要并行处理的场景
- 上下文管理要求严格的应用
