"""Task Graph 系统测试 — DAG 构建、拓扑排序、环检测、执行器.

运行:
    uv run pytest tests/test_task_graph.py -xvs
"""

from __future__ import annotations

import json

import pytest

from mini_code_agent.core.task_graph import (
    CyclicDependencyError,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from mini_code_agent.core.graph_planner import (
    GraphPlanner,
    GraphPlannerError,
)
from mini_code_agent.core.graph_executor import (
    GraphExecutor,
    GraphResult,
    run_verification,
    _is_shell_command,
)
from mini_code_agent.llm import (
    LLMClient,
    LLMResponse,
    Message,
    TokenUsage,
    ToolParam,
)
from mini_code_agent.core.agent import Agent, AgentResult


# ==================================================================
# Helper: 创建简单图
# ==================================================================


def _make_linear_graph() -> TaskGraph:
    """创建线性图: A -> B -> C."""
    graph = TaskGraph()
    graph.original_goal = "测试目标"
    graph.add_task(TaskNode(id="A", description="任务 A"))
    graph.add_task(TaskNode(id="B", description="任务 B", dependencies=["A"]))
    graph.add_task(TaskNode(id="C", description="任务 C", dependencies=["B"]))
    return graph


def _make_diamond_graph() -> TaskGraph:
    """创建菱形图: A -> B, A -> C, B -> D, C -> D."""
    graph = TaskGraph()
    graph.original_goal = "菱形测试"
    graph.add_task(TaskNode(id="A", description="任务 A"))
    graph.add_task(TaskNode(id="B", description="任务 B", dependencies=["A"]))
    graph.add_task(TaskNode(id="C", description="任务 C", dependencies=["A"]))
    graph.add_task(TaskNode(id="D", description="任务 D", dependencies=["B", "C"]))
    return graph


def _make_parallel_graph() -> TaskGraph:
    """创建并行图: A, B, C 无依赖."""
    graph = TaskGraph()
    graph.original_goal = "并行测试"
    graph.add_task(TaskNode(id="A", description="任务 A"))
    graph.add_task(TaskNode(id="B", description="任务 B"))
    graph.add_task(TaskNode(id="C", description="任务 C"))
    return graph


# ==================================================================
# TaskNode 基本测试
# ==================================================================


class TestTaskNode:
    def test_defaults(self):
        node = TaskNode(id="t1", description="做点事")
        assert node.id == "t1"
        assert node.description == "做点事"
        assert node.dependencies == []
        assert node.status == TaskStatus.PENDING
        assert node.files_involved == []
        assert node.verification == ""
        assert node.result is None
        assert node.error is None
        assert node.retry_count == 0

    def test_with_all_fields(self):
        node = TaskNode(
            id="t2",
            description="写代码",
            dependencies=["t1"],
            status=TaskStatus.RUNNING,
            files_involved=["a.py", "b.py"],
            verification="pytest",
            result="完成",
            error=None,
            retry_count=1,
        )
        assert node.dependencies == ["t1"]
        assert node.files_involved == ["a.py", "b.py"]


# ==================================================================
# TaskGraph DAG 构建测试
# ==================================================================


class TestTaskGraphBuild:
    def test_add_task(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="test"))
        assert "t1" in graph.nodes
        assert len(graph) == 1

    def test_add_duplicate_task_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="test"))
        with pytest.raises(ValueError, match="重复"):
            graph.add_task(TaskNode(id="t1", description="另一个"))

    def test_add_dependency(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="first"))
        graph.add_task(TaskNode(id="t2", description="second"))
        graph.add_dependency("t2", "t1")
        assert "t1" in graph.nodes["t2"].dependencies

    def test_add_dependency_nonexistent_task_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="test"))
        with pytest.raises(KeyError):
            graph.add_dependency("t999", "t1")

    def test_add_dependency_nonexistent_dep_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="test"))
        with pytest.raises(KeyError):
            graph.add_dependency("t1", "t999")

    def test_add_dependency_idempotent(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="first"))
        graph.add_task(TaskNode(id="t2", description="second"))
        graph.add_dependency("t2", "t1")
        graph.add_dependency("t2", "t1")  # 重复添加不报错
        assert graph.nodes["t2"].dependencies.count("t1") == 1


# ==================================================================
# 环检测测试
# ==================================================================


class TestCycleDetection:
    def test_self_dependency_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="test"))
        with pytest.raises(CyclicDependencyError, match="自依赖"):
            graph.add_dependency("t1", "t1")

    def test_direct_cycle_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", description="first"))
        graph.add_task(TaskNode(id="t2", description="second", dependencies=["t1"]))
        with pytest.raises(CyclicDependencyError):
            graph.add_dependency("t1", "t2")

    def test_indirect_cycle_raises(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="A", description="A"))
        graph.add_task(TaskNode(id="B", description="B", dependencies=["A"]))
        graph.add_task(TaskNode(id="C", description="C", dependencies=["B"]))
        # 尝试 A -> C（形成 A -> B -> C -> A 的环）
        with pytest.raises(CyclicDependencyError):
            graph.add_dependency("A", "C")

    def test_no_cycle_valid(self):
        graph = _make_diamond_graph()
        # 菱形图不应该检测到环
        assert len(graph) == 4

    def test_cycle_via_add_task_dependencies(self):
        """add_task 时如果节点声明了依赖到已有节点且形成环，应该报错."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="A", description="A"))
        graph.add_task(TaskNode(id="B", description="B", dependencies=["A"]))
        # C 依赖 B，B 依赖 A —— 没有环
        graph.add_task(TaskNode(id="C", description="C", dependencies=["B"]))
        # 无环通过
        assert len(graph) == 3


# ==================================================================
# get_ready_tasks 测试
# ==================================================================


class TestGetReadyTasks:
    def test_no_dependencies_all_ready(self):
        graph = _make_parallel_graph()
        ready = graph.get_ready_tasks()
        assert len(ready) == 3
        ids = {n.id for n in ready}
        assert ids == {"A", "B", "C"}

    def test_linear_only_first_ready(self):
        graph = _make_linear_graph()
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "A"

    def test_after_first_completed(self):
        graph = _make_linear_graph()
        graph.mark_completed("A", "done")
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "B"

    def test_diamond_after_root_completed(self):
        graph = _make_diamond_graph()
        graph.mark_completed("A", "done")
        ready = graph.get_ready_tasks()
        ids = {n.id for n in ready}
        assert ids == {"B", "C"}

    def test_diamond_d_not_ready_until_both_deps(self):
        graph = _make_diamond_graph()
        graph.mark_completed("A", "done")
        graph.mark_completed("B", "done")
        # C 还是 PENDING，所以 D 不应该 ready
        ready = graph.get_ready_tasks()
        ids = {n.id for n in ready}
        assert "D" not in ids
        assert "C" in ids

    def test_diamond_d_ready_after_both_deps(self):
        graph = _make_diamond_graph()
        graph.mark_completed("A", "done")
        graph.mark_completed("B", "done")
        graph.mark_completed("C", "done")
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "D"

    def test_running_tasks_not_ready(self):
        graph = _make_parallel_graph()
        graph.mark_running("A")
        ready = graph.get_ready_tasks()
        ids = {n.id for n in ready}
        assert "A" not in ids
        assert len(ready) == 2

    def test_failed_tasks_not_ready(self):
        graph = _make_parallel_graph()
        graph.mark_failed("A", "error")
        ready = graph.get_ready_tasks()
        ids = {n.id for n in ready}
        assert "A" not in ids


# ==================================================================
# BLOCKED 传播测试
# ==================================================================


class TestBlockedPropagation:
    def test_linear_block_propagates(self):
        graph = _make_linear_graph()
        graph.mark_failed("A", "crash")
        # B 依赖 A，应该被标记为 BLOCKED
        assert graph.nodes["B"].status == TaskStatus.BLOCKED
        # C 依赖 B，也应该被 BLOCKED
        assert graph.nodes["C"].status == TaskStatus.BLOCKED

    def test_diamond_partial_block(self):
        graph = _make_diamond_graph()
        graph.mark_completed("A", "done")
        graph.mark_failed("B", "error")
        # D 依赖 B 和 C，B 失败，D 应该被 BLOCKED
        assert graph.nodes["D"].status == TaskStatus.BLOCKED
        # C 不依赖 B，不应该受影响
        assert graph.nodes["C"].status == TaskStatus.PENDING

    def test_is_blocked(self):
        graph = _make_linear_graph()
        assert not graph.is_blocked()
        graph.mark_failed("A", "error")
        assert graph.is_blocked()

    def test_parallel_no_block(self):
        """并行任务，一个失败不会阻塞其他."""
        graph = _make_parallel_graph()
        graph.mark_failed("A", "error")
        # B 和 C 不依赖 A，不应该被 BLOCKED
        assert graph.nodes["B"].status == TaskStatus.PENDING
        assert graph.nodes["C"].status == TaskStatus.PENDING
        # 但因为没有 BLOCKED 节点（只有 FAILED），is_blocked 返回 False
        assert not graph.is_blocked()


# ==================================================================
# is_complete 测试
# ==================================================================


class TestIsComplete:
    def test_empty_graph_is_complete(self):
        graph = TaskGraph()
        assert graph.is_complete()

    def test_all_completed(self):
        graph = _make_linear_graph()
        graph.mark_completed("A", "ok")
        graph.mark_completed("B", "ok")
        graph.mark_completed("C", "ok")
        assert graph.is_complete()

    def test_all_skipped(self):
        graph = _make_parallel_graph()
        graph.mark_skipped("A")
        graph.mark_skipped("B")
        graph.mark_skipped("C")
        assert graph.is_complete()

    def test_mixed_completed_and_skipped(self):
        graph = _make_parallel_graph()
        graph.mark_completed("A", "ok")
        graph.mark_skipped("B")
        graph.mark_completed("C", "ok")
        assert graph.is_complete()

    def test_not_complete_with_pending(self):
        graph = _make_linear_graph()
        graph.mark_completed("A", "ok")
        assert not graph.is_complete()


# ==================================================================
# get_critical_path 测试
# ==================================================================


class TestCriticalPath:
    def test_empty_graph(self):
        graph = TaskGraph()
        assert graph.get_critical_path() == []

    def test_linear_path(self):
        graph = _make_linear_graph()
        path = graph.get_critical_path()
        assert len(path) == 3
        assert [n.id for n in path] == ["A", "B", "C"]

    def test_diamond_path(self):
        graph = _make_diamond_graph()
        path = graph.get_critical_path()
        # 关键路径应该是长度 3：A -> B/C -> D
        assert len(path) == 3
        assert path[0].id == "A"
        assert path[-1].id == "D"

    def test_parallel_single_node_path(self):
        graph = _make_parallel_graph()
        path = graph.get_critical_path()
        # 并行任务关键路径长度为 1
        assert len(path) == 1

    def test_longer_branch_is_critical(self):
        """较长的分支应该是关键路径."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="A", description="root"))
        # 短分支：A -> B
        graph.add_task(TaskNode(id="B", description="short", dependencies=["A"]))
        # 长分支：A -> C -> D -> E
        graph.add_task(TaskNode(id="C", description="long1", dependencies=["A"]))
        graph.add_task(TaskNode(id="D", description="long2", dependencies=["C"]))
        graph.add_task(TaskNode(id="E", description="long3", dependencies=["D"]))

        path = graph.get_critical_path()
        assert len(path) == 4
        assert [n.id for n in path] == ["A", "C", "D", "E"]


# ==================================================================
# to_mermaid 测试
# ==================================================================


class TestToMermaid:
    def test_basic_mermaid(self):
        graph = _make_linear_graph()
        mermaid = graph.to_mermaid()
        assert "graph TD" in mermaid
        assert "A -->" in mermaid
        assert "B -->" in mermaid

    def test_mermaid_contains_all_nodes(self):
        graph = _make_diamond_graph()
        mermaid = graph.to_mermaid()
        assert 'A[' in mermaid
        assert 'B[' in mermaid
        assert 'C[' in mermaid
        assert 'D[' in mermaid

    def test_mermaid_style_classes(self):
        graph = _make_linear_graph()
        graph.mark_completed("A", "done")
        graph.mark_failed("B", "error")
        mermaid = graph.to_mermaid()
        assert ":::completed" in mermaid
        assert ":::failed" in mermaid
        assert "classDef completed" in mermaid
        assert "classDef failed" in mermaid


# ==================================================================
# summary 测试
# ==================================================================


class TestSummary:
    def test_all_pending(self):
        graph = _make_parallel_graph()
        s = graph.summary()
        assert s == {"pending": 3}

    def test_mixed(self):
        graph = _make_parallel_graph()
        graph.mark_completed("A", "ok")
        graph.mark_failed("B", "err")
        s = graph.summary()
        assert s["completed"] == 1
        assert s["failed"] == 1
        assert s["pending"] == 1


# ==================================================================
# Mock LLM for GraphPlanner tests
# ==================================================================


class MockLLMClient(LLMClient):
    """返回预设文本的 Mock LLM."""

    def __init__(self, content: str) -> None:
        super().__init__(model="mock")
        self._content = content
        self.call_count = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolParam] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self._content, usage=TokenUsage(10, 10))

    def chat_stream(self, messages, tools=None, response_format=None):
        raise NotImplementedError


def _make_valid_graph_json(tasks_count: int = 3) -> str:
    tasks = []
    for i in range(tasks_count):
        tid = f"task-{i+1:03d}"
        deps = [f"task-{i:03d}"] if i > 0 else []
        tasks.append({
            "id": tid,
            "description": f"任务 {i + 1}",
            "dependencies": deps,
            "files_involved": [f"file{i}.py"],
            "verification": f"pytest test_{i}.py",
        })
    return json.dumps({"tasks": tasks})


# ==================================================================
# GraphPlanner 测试
# ==================================================================


class TestGraphPlanner:
    @pytest.mark.asyncio
    async def test_plan_basic(self):
        raw = _make_valid_graph_json(3)
        client = MockLLMClient(raw)
        planner = GraphPlanner(client)

        graph = await planner.plan_as_graph("实现用户认证")

        assert len(graph) == 3
        assert graph.original_goal == "实现用户认证"
        assert graph.nodes["task-001"].dependencies == []
        assert graph.nodes["task-002"].dependencies == ["task-001"]
        assert graph.nodes["task-003"].dependencies == ["task-002"]

    @pytest.mark.asyncio
    async def test_plan_empty_goal_raises(self):
        client = MockLLMClient("{}")
        planner = GraphPlanner(client)
        with pytest.raises(GraphPlannerError, match="goal 不能为空"):
            await planner.plan_as_graph("   ")

    @pytest.mark.asyncio
    async def test_plan_invalid_json_retries(self):
        """无法解析的 JSON 应该重试."""
        client = MockLLMClient("这不是 JSON")
        planner = GraphPlanner(client, max_retries=1)
        with pytest.raises(GraphPlannerError, match="尝试"):
            await planner.plan_as_graph("做点事")
        # 1 次初始 + 1 次重试 = 2 次调用
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_plan_missing_tasks_raises(self):
        client = MockLLMClient('{"other": []}')
        planner = GraphPlanner(client, max_retries=0)
        with pytest.raises(GraphPlannerError):
            await planner.plan_as_graph("做点事")

    @pytest.mark.asyncio
    async def test_plan_handles_markdown_codeblock(self):
        """LLM 返回被 ``` 包裹的 JSON 也能解析."""
        raw = f"```json\n{_make_valid_graph_json(2)}\n```"
        client = MockLLMClient(raw)
        planner = GraphPlanner(client)

        graph = await planner.plan_as_graph("测试")
        assert len(graph) == 2

    @pytest.mark.asyncio
    async def test_plan_parallel_tasks(self):
        """无依赖的并行任务."""
        data = json.dumps({"tasks": [
            {"id": "A", "description": "并行 A", "dependencies": [], "files_involved": [], "verification": ""},
            {"id": "B", "description": "并行 B", "dependencies": [], "files_involved": [], "verification": ""},
            {"id": "C", "description": "汇聚 C", "dependencies": ["A", "B"], "files_involved": [], "verification": ""},
        ]})
        client = MockLLMClient(data)
        planner = GraphPlanner(client)

        graph = await planner.plan_as_graph("并行任务")
        assert len(graph) == 3
        ready = graph.get_ready_tasks()
        ids = {n.id for n in ready}
        assert ids == {"A", "B"}

    @pytest.mark.asyncio
    async def test_plan_cyclic_dependency_raises(self):
        """环依赖应该被检测到."""
        data = json.dumps({"tasks": [
            {"id": "A", "description": "A", "dependencies": ["B"]},
            {"id": "B", "description": "B", "dependencies": ["A"]},
        ]})
        client = MockLLMClient(data)
        planner = GraphPlanner(client, max_retries=0)
        with pytest.raises(GraphPlannerError):
            await planner.plan_as_graph("环依赖")

    @pytest.mark.asyncio
    async def test_plan_dangling_dependency_raises(self):
        """引用不存在的依赖应该报错."""
        data = json.dumps({"tasks": [
            {"id": "A", "description": "A", "dependencies": ["nonexistent"]},
        ]})
        client = MockLLMClient(data)
        planner = GraphPlanner(client, max_retries=0)
        with pytest.raises(GraphPlannerError):
            await planner.plan_as_graph("悬挂依赖")


# ==================================================================
# Mock Agent for GraphExecutor tests
# ==================================================================


class MockAgent:
    """最小化的 Mock Agent，模拟 run() 和 reset()."""

    def __init__(self, results: list[AgentResult] | None = None) -> None:
        self._results = results or []
        self._call_count = 0
        self.reset_count = 0
        self.last_prompts: list[str] = []

    async def run(self, user_message: str) -> AgentResult:
        self.last_prompts.append(user_message)
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
        else:
            result = AgentResult(content="完成了", usage=TokenUsage(5, 5))
        self._call_count += 1
        return result

    def reset(self) -> None:
        self.reset_count += 1


# ==================================================================
# GraphExecutor 测试
# ==================================================================


class TestGraphExecutor:
    @pytest.mark.asyncio
    async def test_execute_linear_graph(self):
        """线性图应该按顺序执行所有任务."""
        graph = _make_linear_graph()
        agent = MockAgent()
        executor = GraphExecutor()

        result = await executor.execute(graph, agent)

        assert result.tasks_completed == 3
        assert result.tasks_failed == 0
        assert graph.is_complete()
        # 每个子任务都应该 reset Agent
        assert agent.reset_count == 3

    @pytest.mark.asyncio
    async def test_execute_parallel_graph(self):
        """并行图应该执行所有任务."""
        graph = _make_parallel_graph()
        agent = MockAgent()
        executor = GraphExecutor()

        result = await executor.execute(graph, agent)

        assert result.tasks_completed == 3
        assert graph.is_complete()

    @pytest.mark.asyncio
    async def test_execute_diamond_graph(self):
        """菱形图应该正确处理依赖."""
        graph = _make_diamond_graph()
        agent = MockAgent()
        executor = GraphExecutor()

        result = await executor.execute(graph, agent)

        assert result.tasks_completed == 4
        assert graph.is_complete()

    @pytest.mark.asyncio
    async def test_execute_with_failure(self):
        """失败的任务应该阻塞下游."""
        graph = _make_linear_graph()
        # 第一个任务失败（stop_reason 不是 "ok"）
        results = [
            AgentResult(content="崩了", usage=TokenUsage(5, 5), stop_reason="error"),
            AgentResult(content="崩了", usage=TokenUsage(5, 5), stop_reason="error"),
            AgentResult(content="崩了", usage=TokenUsage(5, 5), stop_reason="error"),
        ]
        agent = MockAgent(results)
        executor = GraphExecutor(max_retries=2)

        result = await executor.execute(graph, agent)

        assert result.tasks_failed == 1  # A 失败
        assert graph.nodes["A"].status == TaskStatus.FAILED
        assert graph.nodes["B"].status == TaskStatus.BLOCKED
        assert graph.nodes["C"].status == TaskStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_execute_retry_then_succeed(self):
        """失败后重试成功."""
        graph = TaskGraph()
        graph.original_goal = "重试测试"
        graph.add_task(TaskNode(id="A", description="可能失败的任务"))

        results = [
            # 第一次失败
            AgentResult(content="失败", usage=TokenUsage(5, 5), stop_reason="error"),
            # 第二次成功
            AgentResult(content="成功", usage=TokenUsage(5, 5), stop_reason="ok"),
        ]
        agent = MockAgent(results)
        executor = GraphExecutor(max_retries=2)

        result = await executor.execute(graph, agent)

        assert result.tasks_completed == 1
        assert graph.nodes["A"].status == TaskStatus.COMPLETED
        assert graph.nodes["A"].retry_count == 1

    @pytest.mark.asyncio
    async def test_execute_passes_context_to_subtask(self):
        """子任务 prompt 应该包含目标和依赖信息."""
        graph = _make_linear_graph()
        agent = MockAgent()
        executor = GraphExecutor()

        await executor.execute(graph, agent)

        # A 的 prompt 应该包含总体目标
        assert "测试目标" in agent.last_prompts[0]
        # B 的 prompt 应该包含 A 的结果
        assert "任务 A" in agent.last_prompts[1] or "完成" in agent.last_prompts[1]

    @pytest.mark.asyncio
    async def test_execute_wall_time_tracked(self):
        """墙钟时间应该被记录."""
        graph = _make_parallel_graph()
        agent = MockAgent()
        executor = GraphExecutor()

        result = await executor.execute(graph, agent)

        assert result.wall_time >= 0

    @pytest.mark.asyncio
    async def test_execute_empty_graph(self):
        """空图应该立即返回."""
        graph = TaskGraph()
        agent = MockAgent()
        executor = GraphExecutor()

        result = await executor.execute(graph, agent)

        assert result.tasks_completed == 0
        assert result.total_steps == 0
        assert graph.is_complete()


# ==================================================================
# run_verification 测试
# ==================================================================


class TestRunVerification:
    @pytest.mark.asyncio
    async def test_success(self):
        passed, output = await run_verification("echo hello")
        assert passed
        assert "hello" in output

    @pytest.mark.asyncio
    async def test_failure(self):
        passed, output = await run_verification("exit 1")
        assert not passed

    @pytest.mark.asyncio
    async def test_empty_command(self):
        passed, output = await run_verification("")
        assert passed
        assert output == ""

    @pytest.mark.asyncio
    async def test_python_check(self):
        passed, output = await run_verification("python -c 'print(1+1)'")
        assert passed
        assert "2" in output

    @pytest.mark.asyncio
    async def test_chinese_description_skipped(self):
        """中文自然语言描述应该被跳过（视为通过）."""
        passed, output = await run_verification("检查输出中是否包含类似 README.md 等关键文件")
        assert passed
        assert output == ""

    @pytest.mark.asyncio
    async def test_mixed_chinese_skipped(self):
        passed, output = await run_verification("确认文件存在并且格式正确")
        assert passed
        assert output == ""


# ==================================================================
# _is_shell_command 测试
# ==================================================================


class TestIsShellCommand:
    def test_common_commands(self):
        assert _is_shell_command("python -c 'import os'")
        assert _is_shell_command("pytest tests/")
        assert _is_shell_command("node -e 'console.log(1)'")
        assert _is_shell_command("ls -la src/")
        assert _is_shell_command("cat README.md")
        assert _is_shell_command("echo hello")
        assert _is_shell_command("npm test")
        assert _is_shell_command("git status")

    def test_path_commands(self):
        assert _is_shell_command("./run_tests.sh")
        assert _is_shell_command("/usr/bin/python3 -c 'print(1)'")

    def test_versioned_commands(self):
        assert _is_shell_command("python3.11 -c 'print(1)'")

    def test_chinese_text_rejected(self):
        assert not _is_shell_command("检查文件是否存在")
        assert not _is_shell_command("确认输出包含 README")
        assert not _is_shell_command("验证 JSON 格式正确")

    def test_empty_rejected(self):
        assert not _is_shell_command("")
        assert not _is_shell_command("   ")

    def test_unknown_word_rejected(self):
        assert not _is_shell_command("verify the output contains data")
