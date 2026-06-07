# Mini Code Agent

从零构建的编程 Agent，完整实现了从 ReAct 循环到 DAG 任务编排、长程任务管理、增量验证、安全控制等 Coding Agent 核心能力。

## 核心特性

- **多模型支持** — 统一抽象层对接兼容 Anthropic Claude / OpenAI 协议的模型，支持 Prompt Cache、Streaming、Structured Outputs
- **Tool 体系** — 工具定义与执行（文件/Shell/搜索/Git/LSP/记忆）
- **上下文工程** — Token 预算管理、KV Cache 友好排序、超长对话自动摘要压缩
- **三种执行模式** — 直接对话（ReAct Loop）/ 计划模式（Plan-then-Execute）/ 图模式（DAG 任务编排）
- **DAG 任务图** — LLM 生成有向无环图，拓扑排序执行，支持依赖阻塞传播、关键路径分析、Mermaid 导出
- **长程任务管理** — Task Ledger 外部记忆 + Checkpoint 断点续跑 + 崩溃恢复，支持跨会话的复杂任务
- **增量验证** — Level 1 编辑后即时检查（AST/Import/LSP）+ Level 2 子任务完成后单元测试，失败自动重试
- **四层安全体系** — 命令过滤 / 文件守卫 / Git Checkpoint / 循环防护
- **全链路 Trace** — JSONL 格式记录完整 LLM 调用和工具执行过程，用于调试和分析
- **Eval 基准测试** — 内置评测框架，支持任务定义、快照对比、验证脚本、结果分析

## 架构总览
<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/f7c50a60-fc99-4c62-b819-d1bded65f6ea" />


## 项目结构

```
src/mini_code_agent/
├── core/           Agent 核心循环、Planner、DAG 图执行器、验证器、重试控制
├── llm/            多模型客户端抽象（Anthropic/OpenAI/Responses）、Token 计费
├── tools/          工具定义与执行（文件/Shell/搜索/Git/LSP/记忆），Pydantic 校验
├── context/        上下文工程：Token 预算、项目分析、Repo Map、对话压缩
├── memory/         对话管理（摘要压缩）+ 项目记忆（持久化 JSON）
├── safety/         命令过滤、文件守卫、Git Checkpoint、循环防护
├── verify/         增量验证：L1 快速检查 + L2 单元测试
├── longrun/        长程任务：Ledger / Checkpoint / Resume
├── artifacts/      子任务产物协议：Patch / Verification / Scope / Decision
├── trace/          全链路 JSONL 日志记录
├── eval/           评测框架：Runner / Tracker / Snapshot / Analyze
└── cli/            REPL 交互、确认 UI、图可视化、计划展示
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.12+|
| 依赖管理 | uv |
| LLM SDK | anthropic, openai |
| 终端 UI | Rich (渲染) + prompt_toolkit (输入) |
| 数据建模 | dataclass + Pydantic (Schema 生成 & 结构化输出) |
| 测试 | pytest + pytest-asyncio |
| 配置 | python-dotenv (.env 文件) |

## 快速开始

```bash
# 克隆项目
git clone https://github.com/cleozhb/mini_code_agent.git
cd mini_code_agent

# 安装依赖（需要先安装 uv）
uv sync

# 配置 LLM（复制 .env.example 并填入 API Key）
cp .env.example .env

# 启动 Agent
uv run python main.py

# 指定模型/模式
uv run python main.py --project-dir ~/.mini_code_agent  # 指定工作目录
uv run python main.py --plan                            # 计划模式
uv run python main.py --graph                           # DAG 图模式
uv run python main.py --long-run "重构认证模块"           # 长程任务模式
uv run python main.py --resume <task_id>                # 断点续跑
```

## 设计亮点

### ReAct + Plan + Graph 三级执行策略

简单任务直接 ReAct 循环完成；中等复杂度任务先规划后执行，失败可重规划；复杂任务分解为 DAG，拓扑序并行执行，子任务独立上下文互不污染。

### 上下文工程

不是简单地把所有信息塞进 Prompt，而是按 Token 预算分配、KV Cache 命中率优先排序、超长对话自动 LLM 摘要压缩（保留最近 10 轮完整对话）、Repo Map 分级降级（完整签名 → 路径列表 → 截断）。

### 四层安全防护

1. **命令过滤** — 三级分类（安全/需确认/禁止），黑名单模式匹配 `rm -rf /`、`fork bomb` 等
2. **文件守卫** — 工作目录隔离、敏感文件拦截、写前自动备份、一键回滚
3. **Git Checkpoint** — 任务前后自动存档，只提交 Agent 产生的变更，支持回退
4. **循环防护** — 最大轮次 + 重复检测 + Token 预算，防止 Agent 陷入死循环

### 长程任务可靠性

Task Ledger 作为 Agent 的"外部记忆"，持久化目标、里程碑、已完成步骤、决策记录、失败尝试。Checkpoint Manager 带 SHA256 完整性校验，基于 Token 消耗和任务完成率自动存档。崩溃后通过 Resume Manager 从最近检查点恢复。

### 增量验证而非事后补救

每次文件编辑后完成 Level 1 检查（语法/导入/LSP 诊断），子任务完成后运行相关单元测试（Level 2），整体任务结束后全量验证。验证失败自动重试最多 3 次，携带错误上下文。

## REPL 命令

| 命令 | 功能 |
|------|------|
| `/plan` | 进入计划模式 |
| `/graph` | 进入 DAG 图模式 |
| `/graph-export` | 导出 Mermaid 图 |
| `/undo` | 回滚上一次文件修改 |
| `/checkpoints` | 查看 Git Checkpoint 列表 |
| `/diff` | 查看当前变更 |
| `/cost` | 查看 Token 用量和费用 |
| `/model` | 切换模型 |
| `/memory` | 查看项目记忆 |
| `/ledger` | 查看长程任务 Ledger |
| `/status` | 查看当前任务状态 |
| `/save` | 保存对话 |
| `/clear` | 清空上下文 |


## 模块详解

### Core — Agent 核心循环 [`src/mini_code_agent/core/`](src/mini_code_agent/core/)

ReAct 循环（最大 25 轮）+ Plan-then-Execute + DAG 图执行三种模式。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`agent.py`](src/mini_code_agent/core/agent.py) | Agent 主类，ReAct 循环：LLM → 工具执行 → 安全检查 → 验证 → 重试 |
| [`planner.py`](src/mini_code_agent/core/planner.py) | LLM 生成结构化 Plan，Pydantic + `response_format` 约束输出 |
| [`task_graph.py`](src/mini_code_agent/core/task_graph.py) | DAG 实现：环检测、拓扑排序、依赖阻塞传播、关键路径、Mermaid 导出 |
| [`graph_planner.py`](src/mini_code_agent/core/graph_planner.py) | LLM 从自然语言生成 TaskGraph |
| [`graph_executor.py`](src/mini_code_agent/core/graph_executor.py) | 拓扑序执行 DAG，子任务独立上下文，失败重试/阻塞回调 |
| [`subtask_runner.py`](src/mini_code_agent/core/subtask_runner.py) | 桥接 Executor↔Agent，产出 SubtaskArtifact |
| [`verifier.py`](src/mini_code_agent/core/verifier.py) | 任务后验证：语法/Lint/相关测试 |
| [`retry.py`](src/mini_code_agent/core/retry.py) | 验证失败时携带错误上下文自动重试（最多 3 次） |
| [`system_prompt.py`](src/mini_code_agent/core/system_prompt.py) | 系统 Prompt 构建，注入项目上下文 |

</details>

### LLM — 多模型抽象层 [`src/mini_code_agent/llm/`](src/mini_code_agent/llm/)

统一接口对接 Claude / OpenAI 协议，支持 Prompt Cache、Streaming、Structured Outputs。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`base.py`](src/mini_code_agent/llm/base.py) | 抽象基类，统一 Message/ToolCall/TokenUsage 类型，声明 LLMCapabilities |
| [`claude_client.py`](src/mini_code_agent/llm/claude_client.py) | Anthropic Claude 后端，Prompt Cache（5min/1h TTL） |
| [`openai_client.py`](src/mini_code_agent/llm/openai_client.py) | OpenAI Chat Completions（兼容 DeepSeek 等） |
| [`openai_responses_client.py`](src/mini_code_agent/llm/openai_responses_client.py) | OpenAI Responses API 后端 |
| [`factory.py`](src/mini_code_agent/llm/factory.py) | 工厂函数，从 `.env` 读取配置创建客户端 |
| [`pricing.py`](src/mini_code_agent/llm/pricing.py) | Token 费用计算，区分 cache read/write |

</details>

### Tools — 工具系统 [`src/mini_code_agent/tools/`](src/mini_code_agent/tools/)

Pydantic Schema 自动生成 + 三级权限（AUTO/CONFIRM/DENY），16 个内置工具。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`base.py`](src/mini_code_agent/tools/base.py) | Tool 抽象基类 + ToolRegistry，Pydantic InputModel 自动生成 JSON Schema |
| [`file_tools.py`](src/mini_code_agent/tools/file_tools.py) | ReadFile / WriteFile / EditFile（模糊匹配 + 回滚） |
| [`shell_tools.py`](src/mini_code_agent/tools/shell_tools.py) | Bash 执行，30s 超时，输出截断，经 CommandFilter 过滤 |
| [`search_tools.py`](src/mini_code_agent/tools/search_tools.py) | Grep（递归正则）/ ListDir（树形列表） |
| [`git_tools.py`](src/mini_code_agent/tools/git_tools.py) | GitStatus / GitDiff / GitCommit / GitLog |
| [`lsp.py`](src/mini_code_agent/tools/lsp.py) | LSP 集成：Python/TS/Go/Rust 的定义跳转、引用、类型、诊断 |
| [`memory_tools.py`](src/mini_code_agent/tools/memory_tools.py) | 项目记忆的添加与关键词检索 |

</details>

### Context — 上下文工程 [`src/mini_code_agent/context/`](src/mini_code_agent/context/)

Token 预算分配（对话 60% / 上下文 40%）、KV Cache 友好排序、Repo Map 分级降级。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`context_builder.py`](src/mini_code_agent/context/context_builder.py) | 预算管理 + Cache 友好排序 + 降级策略 |
| [`project_analyzer.py`](src/mini_code_agent/context/project_analyzer.py) | 自动检测项目语言/框架/包管理器 |
| [`repo_map.py`](src/mini_code_agent/context/repo_map.py) | 仓库地图：文件路径 + 函数/类签名，分级降级 |

</details>

### Memory — 记忆系统 [`src/mini_code_agent/memory/`](src/mini_code_agent/memory/)

对话摘要压缩（70% 预算触发）+ 项目记忆持久化（规范/决策/已知问题）。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`conversation.py`](src/mini_code_agent/memory/conversation.py) | 对话管理：超预算时 LLM 摘要压缩，保留最近 10 轮 |
| [`project_memory.py`](src/mini_code_agent/memory/project_memory.py) | 项目记忆持久化（`.agent/memory.json`），支持关键词检索 |

</details>

### Safety — 安全控制 [`src/mini_code_agent/safety/`](src/mini_code_agent/safety/)

四层防护：命令黑名单 → 文件隔离/备份 → Git 自动存档 → 循环检测。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`command_filter.py`](src/mini_code_agent/safety/command_filter.py) | 三级分类 + 黑名单模式匹配 + 敏感路径检测 |
| [`file_guard.py`](src/mini_code_agent/safety/file_guard.py) | 工作目录隔离、敏感文件拦截、写前备份、回滚 |
| [`git_checkpoint.py`](src/mini_code_agent/safety/git_checkpoint.py) | 任务前后自动 checkpoint，差异化提交，支持回退 |
| [`loop_guard.py`](src/mini_code_agent/safety/loop_guard.py) | 最大轮次 + 重复检测 + Token 预算，注入警告 |

</details>

### Verify — 增量验证 [`src/mini_code_agent/verify/`](src/mini_code_agent/verify/)

L1 每次编辑后 5s 快速检查，L2 子任务完成后运行相关测试。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`verifier.py`](src/mini_code_agent/verify/verifier.py) | 编排 L1/L2 触发时机 |
| [`level1.py`](src/mini_code_agent/verify/level1.py) | 快速验证：AST 语法 + import + LSP 诊断 |
| [`level2.py`](src/mini_code_agent/verify/level2.py) | 运行与修改文件相关的单元测试 |

</details>

### Longrun — 长程任务 [`src/mini_code_agent/longrun/`](src/mini_code_agent/longrun/)

Task Ledger 外部记忆 + Checkpoint 断点续跑 + 崩溃恢复，支持跨会话复杂任务。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`task_ledger.py`](src/mini_code_agent/longrun/task_ledger.py) | 外部记忆：目标/里程碑/决策/失败尝试/资源用量 |
| [`ledger_manager.py`](src/mini_code_agent/longrun/ledger_manager.py) | 持久化 + append-only 历史 + 上下文摘要注入 |
| [`checkpoint_manager.py`](src/mini_code_agent/longrun/checkpoint_manager.py) | SHA256 校验 + 自动存档策略 + 原子写入 |
| [`resume_manager.py`](src/mini_code_agent/longrun/resume_manager.py) | 从最近检查点恢复中断任务 |
| [`session_state.py`](src/mini_code_agent/longrun/session_state.py) | 会话快照：Ledger + Graph + Messages + Git Hash |

</details>

### Artifacts — 子任务产物协议 [`src/mini_code_agent/artifacts/`](src/mini_code_agent/artifacts/)

标准化子任务输出（Patch/验证/作用域/决策/置信度），支持编排器审批流。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`artifact.py`](src/mini_code_agent/artifacts/artifact.py) | SubtaskArtifact：置信度 + 状态生命周期（DRAFT→APPLIED） |
| [`builder.py`](src/mini_code_agent/artifacts/builder.py) | 链式构建器 |
| [`patch.py`](src/mini_code_agent/artifacts/patch.py) | FileEdit（create/modify/delete/rename）+ unified diff |
| [`verification.py`](src/mini_code_agent/artifacts/verification.py) | 自验证：syntax/lint/type/test/import |
| [`scope.py`](src/mini_code_agent/artifacts/scope.py) | 确保 Worker 只修改允许路径 |
| [`storage.py`](src/mini_code_agent/artifacts/storage.py) | Artifact 持久化 |

</details>

### Trace — 全链路日志 [`src/mini_code_agent/trace/`](src/mini_code_agent/trace/)

Append-only JSONL，记录完整 LLM 调用和工具执行过程。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`recorder.py`](src/mini_code_agent/trace/recorder.py) | JSONL Trace 记录器，按 session 分目录存储 |
| [`serializer.py`](src/mini_code_agent/trace/serializer.py) | 事件序列化 |

</details>

### Eval — 评测框架 [`src/mini_code_agent/eval/`](src/mini_code_agent/eval/)

内置 Benchmark 运行器，支持任务定义、快照 diff、验证脚本、结果分析。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`runner.py`](src/mini_code_agent/eval/runner.py) | 临时工作区执行 + diff + 验证脚本 |
| [`benchmark.py`](src/mini_code_agent/eval/benchmark.py) | BenchmarkSuite / BenchmarkTask 定义 |
| [`tracker.py`](src/mini_code_agent/eval/tracker.py) | 指标追踪 |
| [`snapshot.py`](src/mini_code_agent/eval/snapshot.py) | 工作区前后快照 |
| [`analyze.py`](src/mini_code_agent/eval/analyze.py) | 结果分析 |

</details>

### CLI — 命令行界面 [`src/mini_code_agent/cli/`](src/mini_code_agent/cli/)

Rich 流式渲染 + prompt_toolkit 交互，14 个 Slash 命令。

<details>
<summary>展开文件列表</summary>

| 文件 | 职责 |
|------|------|
| [`repl.py`](src/mini_code_agent/cli/repl.py) | 交互式 REPL，历史/补全/流式输出 |
| [`confirm.py`](src/mini_code_agent/cli/confirm.py) | 工具调用确认 UI |
| [`graph_display.py`](src/mini_code_agent/cli/graph_display.py) | DAG 终端可视化 |
| [`plan_display.py`](src/mini_code_agent/cli/plan_display.py) | 计划展示 |

</details>
