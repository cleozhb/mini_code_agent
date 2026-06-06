# LLM 协议转换优化计划

> 日期：2026-06-05
>
> 目标：让 OpenAI 和 Anthropic 都成为本项目的一等 LLM provider，可随时切换；Agent 核心循环继续只依赖统一抽象，不直接感知底层 API 协议差异。

## 1. 背景知识

本项目的核心不是简单调用聊天接口，而是自己实现 Coding Agent Runtime：维护对话、暴露工具、执行工具、把工具结果回传模型、继续多轮推理。这个架构要求 LLM 层做一件关键事情：

> 把项目内部统一的 `Message`、`ToolParam`、`ToolCall`、`ToolResult`、`LLMResponse` 转换成各家模型 API 的原生协议，再把原生响应转回统一结构。

如果没有这层转换，`core/agent.py` 会被 OpenAI 或 Anthropic 的细节污染，后续切换模型、做 eval、接入新 API 都会变成大面积改造。

### 1.1 OpenAI Chat Completions

当前项目的 OpenAI 主路径使用 Chat Completions。

它的核心形态是：

- 请求是 `messages` 数组，角色包括 `system`、`user`、`assistant`、`tool`。
- 工具定义通过 `tools=[{"type": "function", "function": ...}]` 传入。
- 模型要调用工具时，assistant message 会带 `tool_calls`。
- Agent 执行工具后，用 `role="tool"` 且带 `tool_call_id` 的消息把结果回传。
- 结构化输出通过 `response_format={"type": "json_schema", ...}` 表达。

Chat Completions 的优点是协议直接，适合学习和实现自己的 ReAct loop；也方便兼容 DeepSeek 等 OpenAI-compatible 服务。缺点是它不是 OpenAI 现在最现代的 agentic API。

参考：[OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)、[OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat)。

### 1.2 OpenAI Responses API

Responses API 是 OpenAI 新一代统一接口。它支持文本、图像、结构化输出、function calling、conversation state，以及 web/file/computer 等内置工具。

它和 Chat Completions 的关键区别是：

- 请求使用 `input` items，而不是纯 Chat messages。
- 工具调用在输出里表现为 `function_call` item。
- 工具执行结果回传为 `function_call_output` item。
- 结构化输出放在 `text.format` 里，`{"type": "json_schema"}` 会启用 Structured Outputs。
- Responses 可以用 `previous_response_id` 管理状态，但本项目不使用这点，因为项目自己的 Agent loop 需要保留对状态和工具执行的控制。

本项目新增 Responses backend 时，只实现自己的 Agent loop 需要的最小子集：

- message
- function_call
- function_call_output
- structured output
- usage

暂不接 OpenAI 内置 web/file/computer tools，避免把学习项目的工具系统替换掉。

参考：[OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses/create?api-mode=responses)、[OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)。

### 1.3 Anthropic Claude Messages API

Claude Messages API 的工具协议和 OpenAI 不同。

它不使用 OpenAI 那种独立的 `role="tool"` 消息。Anthropic 把工具协议集成到 user/assistant 消息的 content block 里：

- system prompt 是顶层 `system` 参数，不是 messages 里的 `system` role。
- assistant 要调用工具时，返回 `tool_use` content block。
- Agent 执行工具后，下一条 user message 里放 `tool_result` content block。
- `tool_result` 必须紧跟对应的 `tool_use`，中间不能插入无关消息。
- `tool_result` 在 user content 数组中要排在普通 text block 前面。

Claude structured outputs 不是 prompt fallback。Anthropic API 现在原生支持：

- JSON outputs：通过 `output_config={"format": {"type": "json_schema", "schema": ...}}` 约束最终回答格式。
- Strict tool use：通过工具定义里的 `strict: true` 约束工具名和工具参数。

两者解决的问题不同：

- `output_config.format` 约束模型最终“说什么”。
- `strict: true` 约束模型“如何调用工具”。

参考：[Claude Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[Claude Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)。

### 1.4 Streaming 差异

两家 streaming 的增量事件形态也不同。

OpenAI Chat Completions streaming 中，tool call arguments 会通过 `delta.tool_calls[index].function.arguments` 分片返回。Agent 需要按 index 或 id 累积参数字符串，等 finish 后再解析 JSON。

OpenAI Responses streaming 中，function call arguments 也会以专门的 response event 分片返回。新增 Responses backend 需要把这些事件统一转换成项目已有的 `StreamDeltaType.TOOL_CALL_START/DELTA/END`。

Claude streaming 中，工具调用以 content block 形式出现：

- `content_block_start` 表示一个 `tool_use` block 开始。
- `content_block_delta` 中的 `input_json_delta` 分片返回工具参数 JSON。
- `content_block_stop` 表示 block 结束。

本项目的统一 streaming 抽象应该继续只暴露 `TEXT`、`TOOL_CALL_START`、`TOOL_CALL_DELTA`、`TOOL_CALL_END`、`FINISH`，把这些 provider event 都藏在 client adapter 内部。

参考：[Claude Streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)。

### 1.5 上下文缓存、reasoning tokens 和成本

Coding Agent 请求通常包含稳定的大前缀：system prompt、工具 schema、项目说明、Repo Map。这些内容非常适合缓存。

OpenAI prompt caching 自动工作，但需要把稳定内容放在 prompt 前缀，并可通过 usage 里的 `cached_tokens`、`reasoning_tokens` 等字段观测缓存和推理 token。

Anthropic prompt caching 需要通过 `cache_control` 标记可缓存块，usage 里会返回：

- `cache_creation_input_tokens`
- `cache_read_input_tokens`
- `input_tokens`
- `output_tokens`

本项目目前只统计 input/output token，价格表也硬编码在 CLI。优化后应把 usage 和 pricing 独立出来，否则无法准确估算缓存、reasoning、cache read/write 的成本。

参考：[OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)、[Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)。

## 2. 当前实现缺口

当前代码已经有协议转换雏形：

- `llm/base.py` 定义统一 `Message`、`ToolCall`、`ToolParam`、`LLMResponse`、`StreamDelta`。
- `llm/openai_client.py` 已把统一结构转换为 OpenAI Chat Completions。
- `llm/claude_client.py` 已把统一结构转换为 Claude Messages API 的基础 tool_use/tool_result block。

但还有这些问题：

- Claude client 的 `chat()` / `chat_stream()` 签名没有接 `response_format`，导致 Planner/GraphPlanner 无法随时切换到 Anthropic。
- Claude structured outputs 未映射到原生 `output_config.format`。
- Claude strict tool use 未映射到工具定义里的 `strict: true`。
- `.env.example` 当前默认 `ANTHROPIC_MODEL=claude-sonnet-4-20250514`，需要迁移到支持 structured outputs 的 Claude 4.5/4.6+ 模型。
- OpenAI tool arguments JSON 解析失败时直接变 `{}`，错误被吞掉。
- usage 只有 `input_tokens` / `output_tokens`，没有 cached/reasoning/cache read/write。
- `_PRICE_TABLE` 和 `_estimate_cost()` 硬编码在 `src/mini_code_agent/cli/repl.py`，CLI 层承担了 billing 逻辑。
- 缺少不依赖真实 API 的协议转换单元测试。
- 文档里仍有“Anthropic 预留”的表述，和“随时切换 provider”的目标不一致。

## 3. 目标架构

目标是三层结构：

```text
core/agent.py
  只认识统一抽象：Message / ToolParam / ToolCall / ToolResult / LLMResponse

llm/base.py
  定义统一协议、能力声明、usage、成本输入结构

llm/openai_chat_client.py
llm/openai_responses_client.py
llm/claude_client.py
  各自负责 provider 原生协议转换
```

Agent 核心循环不应该知道：

- OpenAI Chat 的 `role="tool"`。
- OpenAI Responses 的 `function_call_output`。
- Claude 的 `tool_use` / `tool_result` content block。
- Claude 的 system prompt 是顶层字段。
- Structured output 在 OpenAI Chat 叫 `response_format`，在 Responses 叫 `text.format`，在 Claude 叫 `output_config.format`。

这些差异都应被 LLM client 层吸收。

## 4. 实施计划

### 4.1 SDK 与依赖

- 使用 `uv add -U openai anthropic` 升级 SDK，并更新 `uv.lock`。
- `pyproject.toml` 保留合理下限；下限必须覆盖 OpenAI Responses API 和 Claude `output_config.format`。
- `.env.example` 的 Anthropic 默认模型迁移为 `claude-sonnet-4-6`，确保默认配置满足 Claude structured outputs 要求。
- 升级后先用签名检查确认：
  - `openai.AsyncOpenAI().responses.create` 可用。
  - `anthropic.AsyncAnthropic().messages.create` 支持 `output_config`。
  - `anthropic.AsyncAnthropic().messages.stream` 支持 structured output 相关参数。

### 4.2 统一抽象

扩展 `llm/base.py`：

- `TokenUsage` 增加：
  - `cached_input_tokens: int = 0`
  - `reasoning_tokens: int = 0`
  - `cache_creation_input_tokens: int = 0`
  - `cache_read_input_tokens: int = 0`
- `ToolCall` 增加：
  - `raw_arguments: str = ""`
  - `parse_error: str | None = None`
- `ToolParam` 增加：
  - `strict: bool = False`
- 新增 `LLMCapabilities`：
  - `structured_outputs: bool`
  - `strict_tools: bool`
  - `streaming_tools: bool`
  - `prompt_cache: bool`
  - `reasoning_usage: bool`

注意：`strict` 不能默认全开。OpenAI 和 Claude strict mode 都只支持 JSON Schema 子集；必须先规范化工具 schema，确认兼容后再启用。

### 4.3 OpenAI Chat backend

- 保留当前 `openai` 默认路径，继续支持 OpenAI-compatible 服务。
- 补齐 usage 细字段：
  - `prompt_tokens_details.cached_tokens`
  - `completion_tokens_details.reasoning_tokens`
- tool arguments JSON 解析失败时：
  - 保留 `raw_arguments`
  - 写入 `parse_error`
  - 不静默伪装成空参数
- `response_format` 继续原样映射到 Chat Completions。
- tools 转换时显式设置 strict：
  - `strict=True` 时传 strict。
  - `strict=False` 时保持非 strict，避免破坏 DeepSeek 等兼容服务。

### 4.4 OpenAI Responses backend

- 新增 `OpenAIResponsesClient`，provider 名为 `openai-responses`。
- `.env.example` 增加：
  - `OPENAI_API_STYLE=chat`
  - 可选值：`chat | responses`
- `create_client("openai")` 默认仍走 Chat；当 `OPENAI_API_STYLE=responses` 时走 Responses。
- `create_client("openai-responses")` 直接创建 Responses backend。
- Responses backend 只实现本项目需要的最小子集：
  - system/user/assistant message
  - function_call
  - function_call_output
  - structured output
  - usage
  - streaming function arguments
- 不使用 `previous_response_id`。
- 不接内置 web/file/computer tools。
- 统一 `response_format` 映射到 Responses `text.format`。

### 4.5 Anthropic Claude backend

- Claude 是正式一等 provider，不再作为预留实现。
- `ClaudeClient.chat()` 和 `ClaudeClient.chat_stream()` 必须接受 `response_format`。
- 统一 `response_format` 映射规则：
  - 输入仍使用项目当前 OpenAI 风格：`{"type": "json_schema", "json_schema": {"name": ..., "strict": ..., "schema": ...}}`
  - Claude 只接收 schema 本体：`output_config={"format": {"type": "json_schema", "schema": schema}}`
  - OpenAI 包装里的 `name` 和 `strict` 不传给 Claude `output_config`
- Claude tool use 映射规则：
  - assistant message + `ToolCall` -> `tool_use` content block
  - `Message.tool(ToolResult)` -> 下一条 user message 的 `tool_result` content block
  - 保证 `tool_result` 紧跟对应 `tool_use`
  - 如果一个 user message 同时包含 tool_result 和 text，tool_result 必须排在 text 前面
- strict tool use：
  - `ToolParam.strict=True` -> Claude tool 定义 `strict: true`
  - `ToolParam.strict=False` -> 不传 strict 或传 false
- structured outputs 失败处理：
  - `stop_reason="refusal"`：转为可读 `LLMError`，说明安全拒答可能不满足 schema
  - `stop_reason="max_tokens"`：提示提高 `max_tokens` 或让 Planner/GraphPlanner 重试
  - schema 复杂度 400：报告 schema 不兼容 strict/structured outputs
  - streaming JSON 不完整：保留 raw 内容并报解析错误
- prompt caching：
  - `.env.example` 增加 `ANTHROPIC_CACHE_TTL=off`
  - 可选值：`off | 5m | 1h`
  - 给稳定 tools/system/project context 添加 `cache_control`
  - 读取 `cache_creation_input_tokens`、`cache_read_input_tokens`

### 4.6 成本估算

- 把 `src/mini_code_agent/cli/repl.py` 中的 `_PRICE_TABLE` 和 `_estimate_cost()` 抽到独立模块，例如 `src/mini_code_agent/llm/pricing.py`。
- 新模块提供：
  - `CostEstimate`
  - `estimate_cost(model: str, usage: TokenUsage) -> CostEstimate | None`
- CLI 只负责展示，不维护价格表和计费逻辑。
- 成本估算支持：
  - input tokens
  - output tokens
  - cached input tokens
  - reasoning tokens
  - Anthropic cache creation tokens
  - Anthropic cache read tokens
- 未知模型或未知价格项返回部分估算，并标明缺失项。
- 本计划不承诺实时同步官方最新价格；价格表只是结构化迁移。

### 4.7 文档更新

- 更新 `docs/agent-interview-guide.md`：
  - 移除“Anthropic 预留”的表述。
  - 明确 OpenAI Chat、OpenAI Responses、Anthropic Messages 都是正式 backend。
  - 说明 Claude structured outputs 是原生支持，不是 prompt fallback。
  - 说明 Responses 是可选 backend，不替代项目自己的 ReAct loop。
- 保留本文档作为协议优化专项计划。

## 5. 测试计划

新增 `tests/test_llm_protocol.py`，不依赖真实 API：

- OpenAI Chat：
  - message 转换
  - tools 转换
  - tool result 转换
  - response_format 透传
  - tool arguments JSON 解析失败
  - streaming parallel tool calls
- OpenAI Responses：
  - message -> input item
  - assistant tool call -> function_call item
  - tool result -> function_call_output item
  - response_format -> text.format
  - usage 细字段映射
  - streaming function arguments
- Anthropic：
  - system prompt 顶层分离
  - message content block 转换
  - output_config.format 映射
  - strict tool use 映射
  - tool_use/tool_result 多轮转换
  - streaming input_json_delta
  - cache usage 映射

新增 `tests/test_pricing.py`：

- 基础 input/output 估算
- OpenAI cached input token
- OpenAI reasoning token
- Anthropic cache creation/read
- 未知模型
- 部分价格缺失

回归现有测试：

```bash
uv run pytest tests/test_llm.py tests/test_planner.py tests/test_task_graph.py tests/test_agent.py tests/test_llm_protocol.py tests/test_pricing.py -xvs
```

真实 API 冒烟测试继续默认 skip，通过 API key 显式启用。

## 6. 验收标准

- `create_client("openai")` 默认仍可用，并继续走 Chat Completions。
- `create_client("openai-responses")` 可用。
- `OPENAI_API_STYLE=responses` 时，`create_client("openai")` 走 Responses backend。
- `create_client("anthropic")` 可用于普通 Agent loop。
- Planner/GraphPlanner 可在 OpenAI Chat、OpenAI Responses、Anthropic Claude 三个 backend 上使用统一 `response_format`。
- 默认 `ANTHROPIC_MODEL` 指向支持 structured outputs 的模型；如果用户显式配置了不支持的 Claude 模型，首次结构化调用必须报清晰错误。
- Agent 核心代码不出现 provider 特定字段名，例如 `tool_use`、`function_call_output`、`output_config`。
- 协议转换核心路径有不依赖真实 API 的单元测试。
- CLI 成本展示不再直接维护价格表。

## 7. 明确不做

- 不用 OpenAI Responses 的 `previous_response_id` 替代项目自己的 conversation 管理。
- 不接 OpenAI 内置 web/file/computer tools。
- 不把 Claude structured outputs 降级成 prompt fallback。
- 不默认对所有工具开启 strict。
- 不承诺价格表实时同步官方最新价格。
