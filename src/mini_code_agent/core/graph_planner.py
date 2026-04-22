"""图规划器 — 调用 LLM 生成 Task Graph（DAG 式任务依赖图）."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, Message
from .task_graph import CyclicDependencyError, TaskGraph, TaskNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class GraphPlannerError(Exception):
    """图规划器相关错误."""


# ---------------------------------------------------------------------------
# Pydantic Schema — 限制 LLM 输出格式
# ---------------------------------------------------------------------------


class TaskItem(BaseModel):
    """单个任务."""

    id: str = Field(description="唯一 ID，如 task-001")
    description: str = Field(description="任务描述")
    dependencies: list[str] = Field(
        default_factory=list,
        description="依赖的 task id 列表",
    )
    files_involved: list[str] = Field(
        default_factory=list,
        description="涉及的文件列表",
    )
    verification: str = Field(
        default="",
        description=(
            "验证方式：必须是可在终端执行的 shell 命令，"
            "如 'pytest tests/' 或 'python -c \"import config\"'。"
            "禁止写自然语言。无法验证时写空字符串。"
        ),
    )


class TaskGraphSchema(BaseModel):
    """LLM 返回的任务图结构."""

    tasks: list[TaskItem] = Field(
        min_length=1,
        description="任务列表，至少包含一个任务",
    )


# ---------------------------------------------------------------------------
# 规划用 System Prompt
# ---------------------------------------------------------------------------


GRAPH_PLANNER_SYSTEM_PROMPT = """\
你是一个任务规划专家。分析用户的需求，输出一个任务执行图。

规则：
1. 每个子任务应该是 Agent 单次能完成的粒度
   （修改 1-3 个文件，或执行一个明确的操作）
2. 明确标注任务之间的依赖关系
   （B 依赖 A = A 完成后才能开始 B）
3. 没有依赖关系的任务可以并行
4. 每个任务包含验证方式

关于 verification 字段的重要要求：
- verification 必须是一条**可以在终端直接执行的 shell 命令**
- 例如: "python -c 'import config'", "pytest tests/", "ls src/config.py"
- **禁止**写自然语言描述，比如"检查文件是否存在"
- 如果某个任务确实难以用命令验证（如"阅读理解代码"），写空字符串 ""
"""


# ---------------------------------------------------------------------------
# GraphPlanner
# ---------------------------------------------------------------------------


class GraphPlanner:
    """调用 LLM 生成 Task Graph."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt or GRAPH_PLANNER_SYSTEM_PROMPT
        self.max_retries = max_retries

    async def plan_as_graph(
        self,
        goal: str,
        project_context: str = "",
    ) -> TaskGraph:
        """根据目标生成 Task Graph.

        Args:
            goal: 用户的任务目标 / 需求原文
            project_context: 可选的项目上下文

        Returns:
            构建好的 TaskGraph

        Raises:
            GraphPlannerError: 多次重试后仍无法解析
        """
        if not goal.strip():
            raise GraphPlannerError("goal 不能为空")

        user_lines = [f"用户需求：{goal.strip()}"]
        if project_context.strip():
            user_lines.append("")
            user_lines.append("项目上下文：")
            user_lines.append(project_context.strip())
        user_message = "\n".join(user_lines)

        messages = [
            Message.system(self.system_prompt),
            Message.user(user_message),
        ]

        # 构建 response_format：使用 OpenAI 的 json_schema 格式
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_graph",
                "strict": True,
                "schema": TaskGraphSchema.model_json_schema(),
            },
        }

        last_error: str = ""
        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                # 把错误信息反馈给 LLM 修正
                messages.append(
                    Message.user(
                        f"你上一次的输出无法解析，错误信息：{last_error}\n"
                        f"请重新输出正确的 JSON 格式。"
                    )
                )

            response = await self.llm_client.chat(
                messages=messages,
                tools=None,
                response_format=response_format,
            )
            logger.debug("GraphPlanner 原始输出 (attempt %d): %s", attempt, response.content)

            try:
                graph = self._parse_response(response.content, goal)
                return graph
            except (ValueError, CyclicDependencyError) as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "GraphPlanner 解析失败 (attempt %d/%d): %s",
                    attempt + 1,
                    1 + self.max_retries,
                    last_error,
                )
                # 把 LLM 的回复加入对话历史，以便下次修正
                messages.append(Message.assistant(response.content))

        raise GraphPlannerError(
            f"经过 {1 + self.max_retries} 次尝试仍无法解析 Task Graph: {last_error}"
        )

    def _parse_response(self, content: str, goal: str) -> TaskGraph:
        """用 Pydantic 解析 LLM 返回的 JSON，构建 TaskGraph.

        Raises:
            ValueError: Pydantic 校验失败、依赖不存在等
            CyclicDependencyError: 任务之间有循环依赖
        """
        # 兜底：某些模型不支持 structured output，可能返回 markdown codeblock
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start:end])

        try:
            parsed = TaskGraphSchema.model_validate_json(text)
        except Exception as e:
            raise ValueError(f"JSON 解析/校验失败: {e}") from e

        graph = TaskGraph()
        graph.original_goal = goal

        # 创建所有节点
        for task_data in parsed.tasks:
            node = TaskNode(
                id=task_data.id,
                description=task_data.description,
                dependencies=task_data.dependencies,
                files_involved=task_data.files_involved,
                verification=task_data.verification,
            )
            graph.add_task(node)

        # 校验所有依赖引用都合法
        for node in graph.nodes.values():
            for dep_id in node.dependencies:
                if dep_id not in graph.nodes:
                    raise ValueError(
                        f"任务 {node.id} 依赖了不存在的任务 {dep_id}"
                    )

        # 全部节点加完后，做完整的环检测
        self._validate_no_cycles(graph)

        return graph

    def _validate_no_cycles(self, graph: TaskGraph) -> None:
        """对整个图做完整的环检测（Kahn 算法）."""
        in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}
        for node in graph.nodes.values():
            in_degree[node.id] = len(
                [d for d in node.dependencies if d in graph.nodes]
            )

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            current = queue.pop(0)
            visited_count += 1
            # 找 current 的下游（所有依赖了 current 的节点）
            for node in graph.nodes.values():
                if current in node.dependencies:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        queue.append(node.id)

        if visited_count != len(graph.nodes):
            raise CyclicDependencyError(
                f"Task Graph 中存在环依赖（{len(graph.nodes)} 个节点中"
                f"只有 {visited_count} 个可拓扑排序）"
            )
