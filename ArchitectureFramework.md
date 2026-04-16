# Code Agent 架构框架

> 从零构建一个编程 Agent 的完整知识地图。
> 覆盖核心模块划分、实现优先级、关键设计决策与常见陷阱。

---

## 一、系统架构总览

一个 Code Agent 由 **5 层 + 1 个全局关注点** 构成，信息从上到下流转：

```
┌─────────────────────────────────────────────────┐
│             用户交互层 (Interface)                │
│   CLI/REPL · Streaming 渲染 · 确认/审批机制       │
└──────────────┬────────────────────▲──────────────┘
               │ 用户输入            │ Agent 回复
               ▼                    │
┌─────────────────────────────────────────────────┐
│             大脑层 (Brain)                        │
│                                                  │
│  System Prompt ──→ Agent Loop ←── 推理策略        │
│  构建器             核心循环       (ReAct/Plan)    │
│                       │                          │
│  Completion Judge     │      Loop Guardian        │
│  (任务完成判定)        │      (循环控制)            │
└───────────────────────┼──────────────────────────┘
                        │ tool_use / tool_result
                        ▼
┌─────────────────────────────────────────────────┐
│             能力层 (Capabilities)                 │
│                                                  │
│  文件操作        Shell 执行       代码搜索         │
│  read/write     bash cmd        grep/glob        │
│  edit(patch)    timeout 控制     符号查找          │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Tool Registry: 注册 / 描述 / 路由 / 权限  │    │
│  └─────────────────────────────────────────┘    │
└──────────────┬────────────────────▲──────────────┘
               │                    │
               ▼                    │
┌─────────────────────────────────────────────────┐
│           环境感知层 (Perception)                  │
│                                                  │
│  项目结构分析       运行时错误捕获     版本控制      │
│  文件树/依赖关系    lint/测试输出      git 状态     │
│  语言/框架检测      编译错误           diff/blame  │
└──────────────┬────────────────────▲──────────────┘
               │                    │
               ▼                    │
┌─────────────────────────────────────────────────┐
│             记忆层 (Memory)                       │
│                                                  │
│  短期记忆           长期记忆          Context      │
│  当前会话            项目规则          Window       │
│  对话历史            偏好/约定         管理器       │
│  (消息列表)          (AGENT.md)       (截断/压缩)  │
└─────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════
  全局关注点: Token Budget (成本控制，贯穿所有层)
  ═══════════════════════════════════════════════
```

**核心信息流：**

```
用户输入
  → 大脑层组装 context（从记忆层拉历史，从环境感知层拉项目状态）
  → 调 LLM
  → LLM 返回 tool_use → 能力层执行 → 结果返回大脑层 → 继续调 LLM
  → LLM 返回纯文本 → 回到用户交互层
```

---

## 二、六大核心模块

### 模块 1：Agent 基础抽象

Agent 的推理与执行骨架。

| 子模块 | 说明 |
|--------|------|
| ReAct 循环 | Reasoning + Acting 交替进行，LLM 思考后调用工具，观察结果后再思考 |
| Tool Use 协议 | 遵循 LLM 原生的 function calling / tool_use 协议 |
| Planning | 可选的规划阶段：执行前先生成计划，用户确认后再执行 |
| Execution Loop | `while (LLM 返回 tool_use) { 执行工具 → 结果喂回 LLM }` |
| Observation/Action | 每一轮的输入（observation）和输出（action）的标准化格式 |
| Completion Judge | 判断任务是否真正完成：LLM 自判 + 客观验证（测试通过、编译成功） |
| Loop Guardian | 防止无限循环：硬性轮数上限 + 重复检测 + 进展检测 |
| 自我纠错信号 | Agent 能识别"我在做无用功"并主动调整策略或请求帮助 |

**核心数据结构：**

```typescript
// Agent Loop 的核心状态
interface AgentState {
  messages: Message[];          // 完整对话历史
  currentGoal: string;          // 当前任务目标
  iterationCount: number;       // 已执行轮数
  tokenUsage: TokenUsage;       // 累计 token 消耗
  status: 'running' | 'done' | 'failed' | 'awaiting_user';
}

// 单轮 Agent 循环
async function agentStep(state: AgentState): Promise<AgentState> {
  const response = await llm.call(state.messages);

  if (response.type === 'text') {
    // LLM 认为任务完成，进入验证
    return { ...state, status: 'done' };
  }

  if (response.type === 'tool_use') {
    // 安全检查 → 执行工具 → 把结果追加到 messages
    const result = await executeTool(response.toolCall);
    state.messages.push(toolResultMessage(result));
    state.iterationCount++;
    return state;
  }
}
```

---

### 模块 2：上下文工程

Coding Agent 最核心的能力，直接决定 Agent 的能力上限。

**核心问题：在有限的 context window 里塞进最相关的代码信息。**

| 子模块 | 说明 |
|--------|------|
| 项目概览构建 | 会话开始时加载：项目类型、目录树（2-3 层）、关键配置摘要 |
| 按需文件加载 | Agent 要改某文件时，同时拉入相关文件（import 分析、符号引用） |
| Repo Map | 项目的"地图"——文件列表 + 每个文件的简要描述（类/函数签名） |
| Context Window 管理 | 监控 token 用量，接近上限时触发压缩策略 |
| 对话摘要压缩 | 早期对话压缩成摘要，保留关键决策和结果 |
| 项目级指令文件 | 类似 CLAUDE.md / AGENT.md，用户提供的项目约定和偏好 |

**分层上下文策略：**

```
层级 1 — 静态全局（session 开始加载一次）
  ├─ 项目类型（package.json / pyproject.toml 推断）
  ├─ 目录树（top 2-3 levels, 忽略 node_modules）
  ├─ AGENT.md 项目指令
  └─ 关键配置摘要

层级 2 — 动态按需（Agent 决定读什么）
  ├─ 当前要修改的文件的完整内容
  ├─ import/require 依赖文件
  ├─ grep/ripgrep 搜索结果
  └─ 相关测试文件

层级 3 — 运行时反馈
  ├─ 命令执行的 stdout/stderr
  ├─ 测试结果
  ├─ lint/编译错误
  └─ git diff
```

**关键陷阱：** 不要一开始就搞 embedding + 向量检索。文件树 + import 分析 + grep 已经能覆盖 90% 场景，而且可解释、可调试。

---

### 模块 3：Tooling 设计

Agent 通过工具与外部世界交互。

| 子模块 | 说明 |
|--------|------|
| Tool Schema | 每个工具的名称、描述、参数定义、返回值格式 |
| Tool Registry | 集中管理所有工具的注册、查找、路由 |
| 返回值结构化 | 统一返回 `{result, error, metadata}` 格式 |
| 错误表示 | 区分"工具执行失败"和"工具成功但结果不符合预期" |
| 幂等性 | read_file 天然幂等；write_file 需要考虑重复执行的影响 |
| 权限控制 | 工具分级：自动执行 / 需要确认 / 禁止执行 |

**最小工具集（3 个就够启动）：**

```typescript
// 1. 读文件
{
  name: "read_file",
  description: "Read the contents of a file",
  parameters: { path: "string" },
  permission: "auto"   // 读操作无风险，自动执行
}

// 2. 写文件
{
  name: "write_file",
  description: "Write content to a file (creates or overwrites)",
  parameters: { path: "string", content: "string" },
  permission: "confirm" // 需要用户确认
}

// 3. 执行命令
{
  name: "bash",
  description: "Execute a shell command",
  parameters: { command: "string" },
  permission: "confirm" // 需要用户确认
}
```

**进阶工具（按需添加）：**

```
edit_file     — search-and-replace 局部编辑（减少 token 消耗）
list_dir      — 列出目录结构
search_code   — ripgrep 搜索代码
```

**工具输出格式设计原则：** 输出格式直接影响 LLM 推理质量。返回 `{stdout, stderr, exit_code}` 比纯文本好得多，LLM 可以直接看 exit_code 判断成功失败。

---

### 模块 4：Memory 设计

**核心哲学：记住该记的，忘掉该忘的。**

| 维度 | 说明 |
|------|------|
| 写什么 | 关键决策、用户偏好、项目约定、失败原因 |
| 不写什么 | 中间过程、重复信息、可从文件系统恢复的内容 |
| 什么时候更新 | 任务完成时、发现新约定时、用户纠正时 |
| 如何检索 | 短期靠最近 N 条消息；长期靠关键词匹配或 embedding |
| 如何遗忘 | 对话截断 + 摘要压缩；长期记忆加过期时间或覆盖更新 |

**三层记忆架构：**

```
短期记忆 — 当前会话的消息列表
  ├─ 策略：保留最近 N 条 + system prompt
  ├─ 超出时：摘要压缩早期对话
  └─ 实现：内存中的数组

长期记忆 — 跨会话的项目知识
  ├─ 存储：项目根目录下的 AGENT.md 或 .agent/memory.json
  ├─ 内容：项目约定、用户偏好、常见错误的解法
  └─ 更新：任务结束时 Agent 提议写入，用户确认

语义记忆 — 基于内容的检索（进阶）
  ├─ 存储：本地向量数据库（如 sqlite-vss）
  ├─ 内容：代码片段、文档、历史解决方案
  └─ 检索：embedding 相似度匹配
```

**关键难题 — Memory 一致性：**
Agent 在第 5 步记住了"项目用 Jest 测试"，但第 20 步迁移到了 Vitest。旧 memory 变成错误信息。解决方案：memory 条目带时间戳 + 来源，冲突时以文件系统实际状态为准。

---

### 模块 5：Evaluation

**没有 eval 就不知道 Agent 在变好还是变差。**

| 指标 | 说明 |
|------|------|
| Task Success Rate | 任务完成率，最核心的指标 |
| Test Pass Rate | 生成的代码能通过测试的比例 |
| Edit Accuracy | 修改的代码中不需要用户手动修正的比例 |
| Step Count | 完成任务所需的 tool calling 轮数（越少越好） |
| Token Cost | 单个任务消耗的 token 数 |
| Recovery Rate | 遇到错误后自动恢复的成功率 |

**最小 Benchmark 设计：**

准备 5-10 个固定任务，涵盖不同难度：

```
Level 1 — 单文件修改
  "在 utils.ts 里加一个 formatDate 函数"
  "修复 parser.ts 第 42 行的 off-by-one error"

Level 2 — 多文件协作
  "给 User model 加一个 email 验证字段，包括迁移和测试"
  "把所有 console.log 替换成 logger 调用"

Level 3 — 探索 + 修改
  "这个项目的测试为什么会失败？找到原因并修复"
  "重构 auth 模块，把 session 逻辑抽出来"
```

每次改完 Agent 代码后跑一遍，记录每个任务的各项指标。

---

### 模块 6：安全与约束

Coding Agent 的危险操作清单及防护策略。

| 风险 | 防护方式 |
|------|---------|
| Shell 注入 | 命令白名单（ls, cat, grep 自动放行）+ 危险命令需确认 |
| 删除文件 | rm 命令默认禁止自动执行；所有删除操作需二次确认 |
| 覆盖用户改动 | 写操作前先备份原文件；任务开始时 git checkpoint |
| 无限循环执行 | Loop Guardian：最大轮数 + 重复检测 + 进展检测 |
| 修改错误目录 | 限制 Agent 的工作目录；路径参数校验 |
| 读取敏感文件 | .env / 密钥文件 / ~/.ssh 等加入黑名单 |
| 超时命令 | bash 执行加 timeout（默认 30s），防止 while true 卡死 |

**权限分级模型：**

```
🟢 自动执行   — read_file, list_dir, search_code
🟡 需要确认   — write_file, edit_file, bash(非白名单命令)
🔴 默认禁止   — rm -rf, 修改 .env, 操作 ~/.ssh
```

**最可靠的安全网：Git。**
任务开始 → 自动创建 git checkpoint。所有修改可一键回滚。

---

## 三、实现优先级

从上到下，每一步都能跑、能演示。

```
Phase 1 — 最小可运行 Agent（第 1-2 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ① Agent Loop        核心循环 + LLM 调用 + streaming
 ② Tool System       read_file + write_file + bash（3 个工具）
 ③ Repo Context      项目目录树 + 关键文件自动加载
 ④ Shell Safety      基本的命令确认机制 + 输出截断 + timeout
 ⬇ 里程碑：能对话、能读写文件、能跑命令

Phase 2 — 可用于真实项目（第 3-5 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ⑤ Context Builder   System Prompt 工程 + 上下文组装策略
 ⑥ Edit/Patch        search-and-replace 局部编辑
 ⑦ Safety Layer      权限分级 + git checkpoint + 文件备份
 ⑧ Logging           工具调用日志 + token 用量追踪
 ⬇ 里程碑：能在真实 repo 中稳定工作

Phase 3 — 智能化（第 6-8 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ⑨ Memory            对话压缩 + 项目级长期记忆
 ⑩ Plan Mode         执行前规划 + 用户确认 + 按计划执行
 ⑪ Retry/Replan      失败检测 + 自动重试 + 方案切换
 ⑫ Evaluation        最小 benchmark + 指标追踪
 ⬇ 里程碑：能处理复杂任务，有数据证明效果

Phase 4 — 高级能力（第 9-12 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ⑬ Task Graph        DAG 任务拆分 + 动态调整
 ⑭ Skill System      可插拔的工具/能力模块
 ⑮ Sub-Agent         Orchestrator + Worker 多 Agent 协作
 ⬇ 里程碑：能拆分复杂任务，多 Agent 协作完成
```

---

## 四、关键设计决策速查

| 决策点 | 建议 | 理由 |
|--------|------|------|
| 语言选择 | TypeScript | LLM SDK 支持最好；类型系统管理复杂状态更安全；CLI 生态成熟 |
| 文件编辑方案 | Search-and-Replace 为主，Whole-file 兜底 | Token 效率高，容易验证，Aider/Claude Code 验证过的方案 |
| 安全网 | 依赖 Git | 不要自己造版本控制，git 是最可靠的 checkpoint 系统 |
| 上下文检索 | 文件树 + import 分析 + grep | 不要过早引入 embedding/向量检索，简单方案覆盖 90% 场景 |
| 对话历史管理 | 保留 N 条 + 摘要压缩 | 简单有效，实现复杂度低 |
| 长输出处理 | 保留头 + 尾，截断中间 | 错误信息通常在末尾，保留尾部比头部更重要 |
| 任务完成判定 | LLM 自判 + 客观验证 | 单靠 LLM 判断不可靠，需要 test/lint/build 验证 |
| 循环控制 | 硬性上限 + 重复检测 | 三道防线：最大轮数 / 重复操作检测 / 进展停滞检测 |

---

## 五、推荐学习资源

**按实现阶段分批参考，不要一次性全看：**

| 阶段 | 参考项目/论文 | 看什么 |
|------|-------------|--------|
| Phase 1 | Aider 源码 | edit format 设计、LLM 交互模式 |
| Phase 2 | SWE-agent | 任务循环设计、工具集选择 |
| Phase 3 | Claude Code 设计文档 | context 管理策略、memory 机制 |
| Phase 4 | MetaGPT / AutoGen | 多角色通信协议、conversation pattern |

---

## 六、面试展示建议

每个阶段写一篇文档，重点讲：

1. **遇到了什么问题** — "context window 满了之后简单截断丢失了关键信息"
2. **尝试了什么方案** — "我试了三种截断策略：固定窗口、滑动窗口、摘要压缩"
3. **做了什么 tradeoff** — "摘要压缩效果最好但额外消耗 token，我选择在超过 70% 容量时触发"
4. **用数据说话** — "在我的 benchmark 上 task success rate 从 40% 提升到 65%"

> "我在 context 管理上尝试了三种策略，跑了 benchmark"
> 远比 "我实现了 memory 机制" 有说服力。
