# Evaluation 系统设计文档

本文档是 mini_code_agent Evaluation 系统的完整设计说明与施工蓝图。每次新会话接手这个工作时，先读本文档 + 根目录 `CLAUDE.md` 即可接着干。

## 目的

在后续优化 Agent 时，能**量化**每次改动的影响。改完跑 eval 要能看到每个指标的具体变化，而不是凭感觉判断。

---

## 状态

| 模块 | 状态 |
|---|---|
| Agent 侧前置改动（BashTool cwd、stop_reason、max_wall_time、tool/verifier 统计） | ✅ 已完成并通过所有测试 |
| PR1: `benchmark.py` + 1 个最小任务 + `snapshot.py` | ✅ 已 merge (PR #8) |
| PR2: `runner.py` + `ModelPricing` + 端到端跑通 | ✅ 已 merge (PR #9) |
| PR3: `tracker.py` + compare + trend | ✅ 已 merge |
| PR4a: CLI 子命令 + 第 2 个任务（L1-add-classmethod） | ✅ 已 merge (PR #11) |
| PR4b: 补齐剩余 3 个 L1 任务（fix-failing-test / rename-var / add-docstring） | ✅ 已 merge (PR #12) |
| PR4c-fix: 给 eval 模式注入 workspace 上下文到 system prompt | ✅ 已 merge (PR #13) |
| PR4c-fix2: runner 在 agent.run() 外 chdir 到 workspace（修 FileGuard 相对路径误拦） | ✅ 已 merge (PR #14) |
| PR4d-fix: L1-add-classmethod 规约精确化 + snapshot 过滤 `.agent-backups`/`.pytest_cache` | ✅ 已 merge (PR #15) |
| PR4d-polish: `expected_files` 支持 `a|b` OR 组 + `KNOWN_MODELS` 加 deepseek-v3.2 定价 | ✅ 已 merge (PR #17) |
| PR4e: L2-split-large-function + L2-print-to-logger（2 个 L2 任务） | ✅ 已 merge (PR #18) |
| PR4e-fix: L2 任务 max_tokens 80k→150k、system prompt 加自测引导、Runner 加 trace 落盘 | ✅ 已 merge (PR #19) |
| PR4f: L2-add-model-field + L2-cache-decorator（补齐 L2×4） | ✅ 已 merge (PR #20) |
| PR4g+: 剩余 3 个 L3 任务（find-and-fix-bugs / refactor-module / api-integration，可分批） | ⬜ 未开始 |

---

## 0. 已完成：Agent 侧前置改动

这三件事不属于 eval 模块，但 eval 强依赖它们，已经实现并测过：

### 0.1 BashTool 加 cwd 字段
`src/mini_code_agent/tools/shell.py` — `BashTool` 多一个 `cwd: str | None = None`，透传给 `create_subprocess_shell`。默认 `None` 保持原行为；eval runner 构造时设为任务 workspace 目录，把 Agent 的 Bash 操作锁在隔离区内。

### 0.2 Agent 墙钟超时与硬中断
`src/mini_code_agent/core/agent.py`：

- `Agent.__init__` 加 `max_wall_time_seconds: float | None = None`
- `run()` 拆成外层 `asyncio.wait_for` + 内部 `_run_impl()`
- `LoopGuard.max_tokens` 超限时 `_run_once` 真的中断（之前只 log 不 break — 修复了）
- `AgentResult` 新增字段 `stop_reason: Literal["ok","max_rounds","max_tokens","timeout","error"]`
- `_force_final_response` 返回 `stop_reason="max_rounds"`
- 超时返 `stop_reason="timeout"`

### 0.3 AgentResult 暴露 eval 所需的统计量
`AgentResult` 同时新增这 4 个字段（默认值不破坏现有调用点）：

```python
tool_calls_errors: int = 0                    # is_error=True 的 tool_call 数
verifier_attempts: int = 0                    # 跑了几次 verifier
verifier_first_passed: bool | None = None     # 首次 verifier 是否 pass（None=没触发）
verifier_final_passed: bool | None = None     # 末次 verifier 是否 pass
```

`_run_once` 累计 `tool_calls_errors`；`_run_impl` 在 verifier 重试循环里记录 first/final。

### 0.4 测试
`tests/test_agent.py` 新增覆盖：
- `test_agent_max_rounds_protection` 断言 `stop_reason == "max_rounds"`
- `test_agent_stop_reason_ok_on_normal_finish`
- `test_agent_stop_reason_max_tokens`
- `test_agent_stop_reason_timeout`
- `test_agent_tool_calls_errors_counted`
- `test_agent_verifier_first_pass_no_retry`
- `test_agent_verifier_recovery_tracked`
- `test_agent_verifier_not_triggered`

`tests/test_tools.py` 新增：
- `test_cwd_scopes_subprocess`
- `test_cwd_isolates_relative_paths`

---

## 1. 目录结构

```
src/mini_code_agent/eval/
  __init__.py
  DESIGN.md             ← 本文档
  benchmark.py          # BenchmarkTask / BenchmarkSuite / hash 计算
  runner.py             # EvalRunner / TaskResult / SuiteResult / EvalSummary
  tracker.py            # EvalTracker / ComparisonReport / TrendReport
  snapshot.py           # workspace 文件快照 diff（edit_accuracy 用）
  METRICS.md            # 指标定义文档（判定规则写死，避免漂移）

eval/                   # 仓库根下，不在 src 下
  tasks/                # benchmark 任务定义
    L1-add-function/
      task.yaml
      workspace/        # Agent 的初始工作区
      validate.py       # 验证脚本，最后一行输出 JSON
    ...（共 12 个任务，见第 3 节）
  results/              # 每次 eval 结果，{timestamp}_{commit}_{seq}.json
    .gitignore          # 忽略本地结果

tests/
  test_eval_benchmark.py
  test_eval_runner.py
  test_eval_tracker.py
```

---

## 2. 核心数据结构

### BenchmarkTask

```python
@dataclass(frozen=True)
class BenchmarkTask:
    id: str                      # "L1-add-function"
    level: int                   # 1 / 2 / 3
    description: str             # 给 Agent 的任务描述（脱敏，像真实用户输入）
    workspace_dir: Path          # task 目录下的 workspace/
    validate_script: Path        # task 目录下的 validate.py
    expected_files: list[str]    # 预期改动的相对路径，支持 "a|b" 表示 a/b 任一命中（OR 组）
    max_steps: int               # 该任务的 LoopGuard.max_rounds
    max_tokens: int              # 该任务的 LoopGuard.max_tokens
    max_wall_time_seconds: int   # 该任务的 Agent.max_wall_time_seconds
    tags: list[str]              # ["single-file", "python", "bug-fix"]
    task_hash: str               # 对 task 目录内容算 sha256（不含 __pycache__/.pyc/.DS_Store）
```

YAML 示例（`eval/tasks/L1-add-function/task.yaml`）：

```yaml
id: L1-add-function
level: 1
description: |
  在 utils.py 中添加一个函数，把 Unix 时间戳转换成 YYYY-MM-DD 字符串，
  并补上测试。
expected_files:
  - src/utils.py
  - tests/test_utils.py
max_steps: 20
max_tokens: 50000
max_wall_time_seconds: 300
tags: [single-file, python]
```

### BenchmarkSuite

```python
class BenchmarkSuite:
    tasks: list[BenchmarkTask]
    suite_hash: str  # sha256 of sorted([task.id + ":" + task.task_hash for task in tasks])

    @classmethod
    def load_from_dir(cls, tasks_dir: Path) -> BenchmarkSuite: ...
    def filter_by_level(self, level: int) -> BenchmarkSuite: ...
    def filter_by_tag(self, tag: str) -> BenchmarkSuite: ...
    def get(self, task_id: str) -> BenchmarkTask | None: ...
```

**Hash 计算规则**：
- `task_hash`：遍历 task 目录下所有文件（按相对路径排序），逐个文件 `sha256(relpath || b"\0" || content)`，串起来再 sha256。忽略 `__pycache__/`、`*.pyc`、`.DS_Store`。
- `suite_hash`：`sha256(sorted([f"{t.id}:{t.task_hash}" for t in tasks]).join("\n"))`

### TaskResult / SuiteResult / EvalSummary

```python
@dataclass
class TaskResult:
    task_id: str
    task_hash: str
    run_index: int                     # 第几次运行（n>=3 时区分）
    passed: bool                       # validate.py 最终判定
    stop_reason: str                   # 来自 AgentResult.stop_reason
    step_count: int                    # tool_calls_count
    tool_error_count: int              # tool_calls_errors
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float                    # tokens × 模型单价
    wall_time_seconds: float
    verifier_first_passed: bool | None
    verifier_final_passed: bool | None
    files_changed_actual: list[str]    # 由 workspace snapshot diff 得出
    edit_precision: float              # |actual ∩ expected| / |actual|
    edit_recall: float                 # |actual ∩ expected| / |expected|
    failure_category: str | None       # 见下
    validation_details: str            # validate.py 的完整输出


@dataclass
class SuiteResult:
    timestamp: str                     # ISO8601
    git_commit: str | None
    suite_hash: str
    model_name: str
    results: list[TaskResult]          # 每个 (task_id, run_index) 一条
    summary: EvalSummary


@dataclass
class EvalSummary:
    # 任务先在 n 次 run 里聚合成"任务通过率"，再对所有任务平均
    task_success_rate: float
    by_level: dict[int, float]         # {1: 0.8, 2: 0.5, 3: 0.3}
    by_failure_category: dict[str, int]

    avg_step_count: float
    tool_error_rate: float             # sum(tool_errors) / sum(tool_calls)
    verifier_first_pass_rate: float
    verifier_recovery_rate: float      # 从失败中挽回的比例（见下）

    avg_prompt_tokens: float
    avg_completion_tokens: float
    total_cost_usd: float
    avg_wall_time_seconds: float
    avg_edit_precision: float
    avg_edit_recall: float
```

### failure_category（固定枚举）

| 值 | 判定 |
|---|---|
| `"timeout"` | `stop_reason == "timeout"` |
| `"max_tokens"` | `stop_reason == "max_tokens"` |
| `"max_rounds"` | `stop_reason == "max_rounds"` |
| `"validation_fail"` | Agent 正常结束但 validate.py 挂了 |
| `"agent_error"` | run() 抛异常 |
| `None` | 通过 |

### recovery_rate 定义

- `verifier_first_pass_rate` = 一把过的任务比例（`verifier_first_passed == True`）
- `verifier_recovery_rate` = 本来首次失败但最终救回来的比例
  - 分子：`verifier_first_passed == False && verifier_final_passed == True` 的任务数
  - 分母：`verifier_first_passed == False` 的任务数（即首次失败过的任务总数）

这是第三轮讨论时敲定的，替换了原计划模糊的 `error_recovery_count`。

---

## 3. Benchmark 任务清单（12 个）

任务描述必须**脱敏**（像真实用户输入，不泄漏修复思路或答案），否则测不出真实能力。

> **当前进度（2026-04-20）**：L1×5 + L2×4 全部落地，剩 L3×3 未开工。每项后面的 ✅/⬜ 代表 `eval/tasks/<id>/` 是否已存在。

### Level 1（5 个，单文件修改）

- ✅ `L1-add-function`：加一个时间戳格式化函数 + 测试
- ✅ `L1-fix-failing-test`：跑 `pytest tests/test_parser.py` 有测试失败，修好（**不**说是 off-by-one）
- ✅ `L1-add-classmethod`：给 Config 类加 `from_env()` 类方法
- ✅ `L1-rename-var`：把某模块的 `user_id` 统一重命名为 `uid`，保持测试通过
- ✅ `L1-add-docstring`：给某个无文档的模块批量加 docstring（验证：ast 解析出所有函数都有 docstring）

### Level 2（4 个，多文件协作）

- ✅ `L2-add-model-field`：给 User model 加 email 字段（model + schema + endpoint + test）
- ✅ `L2-print-to-logger`：把 `print()` 替换成 logger 调用，涉及创建 logger 模块
- ✅ `L2-cache-decorator`：给 3 个请求方法加缓存装饰器 + 失效测试
- ✅ `L2-split-large-function`：把 100+ 行的函数拆成小函数（验证：单函数行数上限 + 测试通过）

### Level 3（3 个，探索 + 修改）

- ⬜ `L3-find-and-fix-bugs`：测试 suite 里有 2 个 failing test，找出根因并修
- ⬜ `L3-refactor-module`：把 `auth.py` 里的 session 逻辑抽出到 `session.py`
- ⬜ `L3-api-integration`：基于现有 HTTP client 给未实现的 endpoint 补完（含错误处理 + 重试 + 测试）

**强制开发顺序**：先只写 `L1-add-function` 一个任务，把 runner / validate / snapshot / tracker 全流程打通再扩量。fixture 和 validate 的坑一次性暴露比较好。

> 已验证的 validate.py 工程经验（PR4a 踩过）：如果 workspace 里的模块用了 `@dataclass`，validate.py 通过 `importlib.util` 加载它之前必须把模块写进 `sys.modules`，否则 dataclass 内部查 `sys.modules[cls.__module__]` 会拿到 None，抛 "NoneType has no attribute __dict__"。后续任务写 validate 时先抄 `L1-add-classmethod/validate.py` 的 `_load_*` 函数。

### validate.py 协议

每个任务的 `validate.py` 以子进程方式运行，cwd 是该 run 的临时 workspace。

**约定**：脚本的**最后一行 stdout** 必须是一行合法 JSON：

```json
{"passed": true, "details": "all 3 tests passed", "tests_run": 3, "tests_passed": 3}
```

runner 只解析最后一行，前面的输出全部进 `TaskResult.validation_details`。JSON 解析失败或 `passed` 缺失 → `failure_category="validation_fail"`。

---

## 4. EvalRunner

```python
class EvalRunner:
    def __init__(
        self,
        *,
        agent_factory: Callable[[Path], Agent],  # 传 workspace，返回配好的 Agent
        runs_per_task: int = 3,                  # 默认 n=3（信噪比）
        parallel_tasks: int = 1,                 # 任务间并行度（默认串行）
        model_name: str,
        pricing: ModelPricing,                   # {input_per_1k, output_per_1k}
    ): ...

    async def run_task(self, task: BenchmarkTask) -> list[TaskResult]: ...
    async def run_suite(self, suite: BenchmarkSuite) -> SuiteResult: ...
```

### 单次 run_task 流程

1. 创建临时工作区：`shutil.copytree(task.workspace_dir, f"/tmp/eval-{uuid}/")`
2. `snapshot.capture(workspace)` — 记录初始文件 `{relpath: (size, mtime, sha256)}`
3. `agent = agent_factory(workspace_path)`：
   - `BashTool(cwd=workspace_path)`
   - `FileGuard(work_dir=workspace_path)`
   - `LoopGuard(max_rounds=task.max_steps, max_tokens=task.max_tokens)`
   - `Agent(..., max_wall_time_seconds=task.max_wall_time_seconds)`
4. `result: AgentResult = await agent.run(task.description)`
5. `files_changed_actual = snapshot.diff(workspace, initial_snapshot)` — 包括 Bash 间接改的
6. 跑 `validate.py`：`python {validate_script}`，cwd=workspace，超时 60s，解析最后一行 JSON
7. 计算 edit_precision/recall、failure_category、cost_usd → 组装 `TaskResult`

### 关键细节

- 工作区默认**不删除**（失败后能进去调试）。超过 7 天或超过 100 个由 tracker 在 trend/list 时清理老的。
- `cost_usd = (prompt * input_per_1k + completion * output_per_1k) / 1000`
- Agent 异常（不是 Agent 自己处理的那种）：`TaskResult.passed=False, failure_category="agent_error"`，不中断整个 suite
- `parallel_tasks > 1` 时用 `asyncio.Semaphore` 控制并发；任务间互不依赖（临时目录隔离）

### ModelPricing 内置表

```python
KNOWN_MODELS = {
    "deepseek-chat":        ModelPricing(0.00014, 0.00028),
    "gpt-4o":               ModelPricing(0.0025, 0.010),
    "gpt-4o-mini":          ModelPricing(0.00015, 0.00060),
    "claude-sonnet-4-5":    ModelPricing(0.003, 0.015),
    # 必要时扩充
}
```

用户可传 `pricing=ModelPricing(...)` 覆盖。

---

## 5. EvalTracker

```python
class EvalTracker:
    def __init__(self, results_dir: Path): ...

    def save(self, result: SuiteResult) -> Path:
        # 文件名 {timestamp}_{commit}_{seq}.json
        # 同 commit 多次跑，seq 递增，不覆盖

    def list_runs(self, last_n: int | None = None) -> list[SuiteResult]: ...

    def compare(self, run_a: SuiteResult, run_b: SuiteResult) -> ComparisonReport:
        # suite_hash 不同 → 只对 task_hash 相同的任务做 diff（交集对比）
        # 三部分：
        #   - 共有任务的指标 diff（含箭头 + 颜色）
        #   - 仅 A 有的任务（被删 / 定义变了）
        #   - 仅 B 有的任务（新增 / 定义变了）

    def trend(self, last_n: int = 10) -> TrendReport:
        # 用 rich 画 5 个核心指标 sparkline：
        #   task_success_rate / tool_error_rate / avg_cost / avg_wall_time / verifier_recovery_rate
```

---

## 6. CLI（扩 main.py）

```
main.py eval                              # 跑全部任务，n=3 次/任务
main.py eval --level 1                    # 只跑 L1
main.py eval --task L1-add-function       # 跑单任务
main.py eval --runs 1                     # 覆盖默认 n
main.py eval --parallel 4                 # 任务间并行度
main.py eval --no-save                    # 不落盘
main.py eval --compare                    # 对比最近两次
main.py eval --compare RUN_A RUN_B        # 对比指定两次
main.py eval --trend                      # 最近 10 次趋势
main.py eval --trend 20                   # 最近 N 次
```

---

## 7. 测试

### `test_eval_benchmark.py`
- YAML 加载、`filter_by_level` / `filter_by_tag`
- `task_hash` 稳定性：同内容同 hash，改一字节变 hash
- `suite_hash` 一致性

### `test_eval_runner.py`
- MockLLMClient + 临时最小任务，端到端跑通 `run_task`
- 超时任务被正确标 `failure_category="timeout"`
- `edit_precision/recall` 对照预期
- `validate.py` JSON 解析失败 → `failure_category="validation_fail"`

### `test_eval_tracker.py`
- 存读往返
- `compare` 在 `suite_hash` 不同时只对比交集
- 同 commit 多次保存，文件名 `seq` 递增不覆盖

---

## 8. 明确排除（避免 scope creep）

- ❌ **Plan mode 相关指标**（`plan_steps_failed` 等）— 等真跑 L3 有数据再说
- ❌ **tool_error 良性失败过滤**（`test -f` 这种）— 接受噪音，相对变化仍可比
- ❌ **JS/TS 任务** — 全 Python fixture，避免 node 依赖
- ❌ **多模型对比自动化** — 一次一个模型，跨 run 比靠 tracker
- ❌ **Web UI / notebook 分析工具** — JSON 落盘后用户自己分析

---

## 9. 关键决策备忘（为什么这么设计）

### 9.1 Benchmark 版本化（每任务 hash + 交集对比）

Benchmark 本身会演进（改描述、改 workspace、改 validate），跨时间对比会"苹果 vs 橘子"。

- 每任务单独算 `task_hash`，而不只是整体 `suite_hash` — 这样"你改了 1 个任务"不会让另外 11 个任务的对比作废
- `tracker.compare` 在 suite_hash 不同时只对比 task_hash 相同的任务子集，新增/删除/变更的任务单独列出
- 结果 JSON 同时记录 `suite_hash` 和每个 result 的 `task_hash`，这样事后分析也能判断可比性

### 9.2 错误恢复指标（三指标替代 error_recovery_count）

原计划 `error_recovery_count: int` 模糊、无法自动判定、一个数丢信息。换成：

- `tool_error_rate` — 犯错频率（客观）
- `verifier_first_pass_rate` — 一把过率
- `verifier_recovery_rate` — 挽回率（从失败中救回来的比例）

三个分开看：优化方向不同（prompt 改进 vs 自我修复能力），需要分别观察。

### 9.3 n=3 默认运行次数

8 个任务 pass/fail 最小变化粒度 = 12.5%，LLM 本身的随机性就能造成那么大的波动。n=3 牺牲成本换信噪比。

### 9.4 edit_accuracy 用 workspace snapshot diff 而不是 Agent 自报

Agent 的 `_files_changed` 只追踪 WriteFile/EditFile，Bash 用 `sed -i` / `echo >` 改的文件追不到。eval runner 自己做前后快照 diff 是可靠信号。

### 9.5 validate.py 用子进程 + stdout 末行 JSON

- 不 import 到 runner 进程里：隔离副作用
- JSON 格式规整、易解析；允许脚本前面打印任意调试信息都不影响解析

### 9.6 eval 模式的 system prompt 必须注入 workspace 上下文

**根因**：PR4a 刚落地的时候，eval 用的是一句话的常量 system prompt（"你是一个编程 Agent，使用工具完成任务"）。在 DeepSeek 上跑 L1-add-classmethod 观察到 Agent 连续 11 轮没调用过一次 WriteFile、prompt token 滚到 50k 被 LoopGuard 砍掉、`files_changed_actual=[]`、`tool_error_rate=0.273`。Claude 这样的强模型有先验能猜对 cwd，但弱模型不行。

**修法**：`src/mini_code_agent/cli/eval_cmd.py:_build_eval_system_prompt(workspace)` 按每个 workspace 动态生成 prompt，里面塞进：
- 当前 cwd 绝对路径
- 工作区初始文件相对路径列表（过滤 `__pycache__` / `.pytest_cache`，超过 30 个截断）
- WriteFile/EditFile path 用相对路径的明确约束
- Bash cwd 已锁住、不要 cd 出去的提示

REPL 模式下这层由 `core.build_system_prompt_with_context` 提供，eval 之前绕过了。

**但这只是故事的一半** —— 见 §9.7 的路径解析 bug，两个 bug 叠在一起才完整解释当时的 `tool_error_rate=0.27`。

### 9.7 eval 模式的 workspace 相对路径解析（chdir 临时修复 + 真正 fix 的 follow-up）

**追加观察**：PR4c-fix 上完后在 DeepSeek 重跑 L1-add-classmethod，`tool_error_rate` 从 0.273 **升到了 0.538**，`files_changed_actual` 仍然空。看起来 prompt 修好了反而更糟 —— 实际是 prompt 让模型严格按相对路径调 WriteFile，而**基础设施本身处理不了相对路径**：

- `FileGuard.is_path_allowed(path)` 里 `Path(path).resolve()` 用**进程 cwd**，不是 `self.work_dir`
- `WriteFileTool.execute` 同样 `Path(path_str).expanduser()`，直接依赖进程 cwd

跑 eval 时进程 cwd = 你启动 `main.py` 的目录（通常是仓库根），而 `FileGuard.work_dir = /var/folders/.../eval-xxx`。于是 Agent 发 `WriteFile(path="config.py")` 时：
1. FileGuard 把它 resolve 成 `<仓库根>/config.py`
2. 判定"路径不在工作目录内"
3. 返回 `blocked`，Agent 收到 `[安全拦截]` 当成 tool_error
4. Agent 试别的路径再被拦，死循环直到 token 爆

**PR4c-fix2 的临时修法**：`EvalRunner._run_task_once` 在 `await agent.run(...)` 外面用 `_chdir(workspace)` 包一层，离开时恢复。这样 `Path.resolve()` 的相对路径基准对齐到 workspace，FileGuard + WriteFileTool + EditFileTool + ReadFileTool + Grep + ListDir 都自动正确。

**代价 / 已知局限**：`os.chdir` 是进程全局状态 —— 当前实现因此不支持 `parallel_tasks > 1`（多个任务会抢 cwd）。`EvalRunner.__init__` 里在 `parallel_tasks > 1` 时 log 警告。真要跑并发必须走方案 B。

**方案 B（根治，未来做）**：把所有文件工具族（`WriteFileTool` / `EditFileTool` / `ReadFileTool` / `GrepTool` / `ListDirTool`）加 `work_dir: Path | None = None` 参数，效仿 `BashTool.cwd` 的做法。相对路径在工具内部解析到 `work_dir / path`；`FileGuard.is_path_allowed` 同步按 `self.work_dir` 解析而不是进程 cwd。改完之后 eval 不再需要 chdir，天然支持并行。改动面：6 个工具 + guard + 所有现有调用点 + 一批测试 —— 显然不能塞在这个 PR 里。

**教训**：PR4c-fix 以为根因是 prompt，实际是两层（prompt + 路径解析）叠加，误诊的代价是多跑一次真 API、多发一个 PR。诊断"为什么 Agent 没落地文件"时应同时查 prompt 层（告诉没告诉）和基础设施层（就算告诉了能不能执行），而不是看到 prompt 明显缺信息就下结论。这条记到 memory 里。

### 9.8 benchmark 规约里要明示文件布局 + snapshot 过滤要扩到工具副产物

**观察**：PR4c-fix2 合入后本机重跑 L1-add-classmethod，`tool_error_rate` 从 0.54 → 0.15，`files_changed_actual` 含 `config.py` + `tests/test_config.py`，validate 明确说"from_env 行为对" —— 基础设施和 Agent 核心产出都已经对。但任务仍 `passed=false`，原因是：

1. **task 规约模糊**：`description` 只说"补上相应测试"，没说测试文件放哪里。validate.py 只在 `workspace/test_config.py` 找，Agent 按 Python 惯例写到了 `workspace/tests/test_config.py`，于是"测试没写"被判失败。这不是 Agent 能力问题，是 benchmark 没说清楚。
2. **snapshot 没忽略工具副产物**：`files_changed_actual` 里混进了 `.agent-backups/config.py.xxx.bak`（WriteFileTool 写文件前自动备份的目录）和 `tests/.pytest_cache/*`（Agent 在 workspace 跑 pytest 的缓存）。这让 `edit_precision` 被拉到 0.14 —— 看起来 Agent 写了一堆 expected 外的文件，其实都是工具自己生成的。

**修法（PR4d-fix）**：
- `L1-add-classmethod/task.yaml` description 末尾显式写"测试文件命名为 test_config.py，放在根目录或 tests/ 下，两种都接受"；`expected_files` 只列核心产出 `config.py`，测试位置不锁，避免把 edit metric 绑死到单一路径
- `L1-add-classmethod/validate.py` 同时在 `workspace/test_config.py` 和 `workspace/tests/test_config.py` 找；子进程跑 pytest 时显式 `PYTHONPATH=workspace` 让 `tests/` 下的 `from config import Config` 能 import
- `snapshot._IGNORE_NAMES` 加入 `.agent-backups` 和 `.pytest_cache`，`test_eval_snapshot.py` 新增对应 case

**教训**：benchmark 任务的 description 必须精确到让"Agent 按合理做法来就能 pass"，凡是会被 validate 严格匹配的位置/命名（路径、文件名、函数签名）都要在 description 里明示。Agent 做出合理选择却因规约模糊被判失败，说明的是规约缺陷不是能力缺陷 —— 后续加 L2/L3 任务时要把这一条当检查项。

另外 `files_changed_actual` / `edit_precision` 作为观测指标，必须把工具副产物（备份目录、缓存目录）和 Agent 真实写入区分开，否则噪声会掩盖信号。

### 9.9 `expected_files` 的 OR 组语义（`a|b` 表示任一命中）

**背景**：PR4d-fix 之后 L1-add-classmethod 跑通了，`passed=true` 但 `edit_precision=0.5` —— 因为我们把 `expected_files` 精简到只剩 `config.py`，Agent 合规地写了 `config.py` + `test_config.py`，于是 precision = 1/2。指标看起来 Agent 做了多余的事，实际上它做的正是任务要求。根因是**"位置允许二选一"和"文件必须在 expected 集合里"这两个诉求打架**。

**解法**：`expected_files` 每项支持用 `|` 分隔的 alternative 组：
```yaml
expected_files:
  - config.py                              # 单文件，与以前一致
  - test_config.py|tests/test_config.py    # 任一命中即整组算命中
```

`BenchmarkTask.load` 把每项按 `|` split 成 `expected_file_groups: tuple[tuple[str, ...], ...]`；`compute_edit_metrics` 改成接 groups。`expected_files` 保留 YAML 原始字符串形式（hash 稳定 + human-readable）。

评分语义：
- recall = 命中的 expected 组数 / |groups|
- precision = |actual 中出现在任何 group 里的文件| / |actual|
- 一组里 Agent 两个都写了也不惩罚（都计入 precision 分子） —— 兜底双写没错

这样 L1-add-classmethod Agent 写 `tests/test_config.py` 的情况下，precision 和 recall 都到 1.0，指标真正反映能力。

**什么时候用 OR 组**：当**位置选择**对任务结果无关（放 tests/ 还是根目录都能跑通）时。不要用来掩盖规约模糊 —— 如果 Agent 可能写两种**不等价**的方案（例如写到 `config.py` 或写到 `settings.py`，语义不一样），仍应在 description 锁死。OR 组是"规约允许的多条合理路径"的显式表达，不是"反正都算对"。

### 9.10 LoopGuard 的 token 记账语义 + L2 任务 budget + trace 落盘

**背景**：PR4e 合并后，本地用 deepseek-v3.2 跑 `L2-split-large-function`，`passed=true` 但 `stop_reason=max_tokens`、`verifier_first_passed=null`、9 步 `prompt_tokens=80860`。同步任务 L1-add-classmethod 9 步只耗 29k prompt。看起来 Agent "侥幸过线"，实际分析下来发现三件事需要修。

**LoopGuard 的记账不是"prompt 上限"，是"API usage 累计"**：`loop_guard.add_tokens(response.usage.input_tokens + response.usage.output_tokens)` 每轮累加。对于 Chat Completions 这种多轮对话，每轮 input 都完整 replay 历史 → `sum_over_rounds(input_tokens)` 增长很快。80860 = Σ9 轮的 input_tokens，不是单轮 prompt 大小，也不是 conversation 实际 token 数。

**prompt cache 帮不上忙**：DeepSeek/OpenAI/Anthropic 都有 context cache，但 `usage.prompt_tokens` 返回的是**完整原始 prompt 的 token 数**，cache 命中只在辅助字段（如 `prompt_cache_hit_tokens`）体现。LoopGuard 读的是完整数，所以 cache 影响 $$ 和延迟，不影响预算判定。

**改动**：

1. **短期 — 放宽 L2 budget**：`L2-split-large-function` 和 `L2-print-to-logger` 的 `task.yaml` 把 `max_tokens: 80000` 提到 `150000`。L2 平均每轮 ~9k prompt（上下文 ×9 轮累加 ~81k），留出自测 + 1-2 步缓冲到 150k 更合理。L1 保持 50000 不变（实测 29k 够用，~40% 余量）。

2. **长期 — system prompt 加自测引导**：`_build_eval_system_prompt` 加一段"提交前必须用 Bash 跑一次 `pytest` 确认无回归"。目的是把 `verifier_first_pass_rate` 从恒 0 拉起来（现在这个指标没信号）。同时 reword 收尾语使汇报里带上测试结果（便于人看日志诊断）。这不等同于 `Verifier` 子系统，但先让 Agent 养成跑测的习惯。

3. **中期 — Runner 加 trace 落盘**：`EvalRunner.__init__` 接 `trace_dir: Path | None`（CLI 默认 `<results_dir>/traces/`，`--no-trace` 关，`--no-save` 隐含 `--no-trace`）。每个 run 落一份 `<ws_name>.trace.json`，含：
   - `task_id`/`run_index`/`timestamp`/`stop_reason`/`passed`/`usage`/`tool_calls_count`/`tool_calls_errors`/`validation_details`
   - `messages`: 完整对话历史（system + user + assistant + tool），每条带 `role`/`content`/`tool_calls`/`tool_result`

   `TaskResult` 加 `trace_path: str | None` 字段反查回 trace 文件。以后再问"为什么这次跑 80k token / 2 个 tool error 从哪来"，直接读 trace 不再靠猜。

**什么时候再动 LoopGuard 记账逻辑**：如果 L2/L3 任务跨多 run 普遍在 100-150k budget 里挣扎，说明"累计 usage" 语义对多轮任务天然不友好。届时两条备选：
- **A.** 按"当前 conversation token 数"而不是累计判 —— 语义更准但改 stop_reason 的意义，慎重。
- **B.** 把 cache hit 字段考虑进去按"折后 billable token" 累加 —— 贴近成本含义，但偏离"流量保护"。

眼下不碰，先用 trace 攒几次数据再说。

---

## 10. 实施顺序

| PR | 内容 | 验收 |
|---|---|---|
| 1 | `benchmark.py` + `snapshot.py` + 1 个最小任务（`L1-add-function`） | YAML 能 load、task_hash 稳定、snapshot diff 准确 |
| 2 | `runner.py` + `ModelPricing` + 端到端跑通最小任务 | MockLLM 跑 `L1-add-function` 产出完整 TaskResult |
| 3 | `tracker.py` + compare + trend | 存读往返、交集对比、sparkline 渲染 |
| 4 | CLI 子命令 + 剩余 11 个任务 | `main.py eval` 产出完整 summary |

**每个 PR 都是独立可 merge 的**，不要攒大包。

---

## 11. 还可能被调整的非关键默认值

这些之前列为"可以改"，默认是这样，执行时如果用户想调再说：

1. `runs_per_task=3`（成本敏感时可降到 1）
2. `parallel_tasks=1`（日志清晰，L1 理论上可并行）
3. `ModelPricing` 内置表 — 用户可传自定义
4. 工作区保留 7 天 / 100 个自动清理

---

## 参考：相关代码位置

- Agent 核心：`src/mini_code_agent/core/agent.py`
- LoopGuard：`src/mini_code_agent/safety/loop_guard.py`
- FileGuard：`src/mini_code_agent/safety/file_guard.py`
- BashTool：`src/mini_code_agent/tools/shell.py`
- Verifier：`src/mini_code_agent/core/verifier.py`
- RetryController：`src/mini_code_agent/core/retry.py`
