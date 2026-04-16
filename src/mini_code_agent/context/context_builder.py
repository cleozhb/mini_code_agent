"""上下文组装器 — Token 预算管理 + 上下文拼接.

核心设计原则：
- Token 是稀缺资源，每一块上下文都要"值得"占据它的位置
- 拼接顺序影响 KV cache 命中率，跨轮不变的内容放前面
- 设定 token 预算，硬性保证不超限，优雅降级而不是报错
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .project_analyzer import ProjectInfo, detect_project_type, get_directory_tree, get_key_files
from .repo_map import build_repo_map, build_repo_map_paths_only


# ---------------------------------------------------------------------------
# Token 预算管理
# ---------------------------------------------------------------------------

# 中日韩字符 Unicode 范围
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff"
    r"\uac00-\ud7af\u3400-\u4dbf]"
)


def estimate_tokens(text: str) -> int:
    """简易 token 估算函数.

    - 中日韩字符：约 1.5 字符/token → len / 1.5
    - 其他字符：约 4 字符/token → len / 4
    - 混合文本取加权平均

    不需要 100% 精确，误差 10% 以内即可。
    """
    if not text:
        return 0

    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars

    cjk_tokens = cjk_chars / 1.5
    other_tokens = other_chars / 4

    return int(cjk_tokens + other_tokens)


@dataclass
class ContextBudget:
    """Token 预算管理."""

    model_context_limit: int = 200_000  # 模型上下文窗口大小
    reserved_for_output: int = 8192  # 预留给模型输出的 token
    reserved_for_conversation_ratio: float = 0.6  # 对话历史占比

    @property
    def reserved_for_conversation(self) -> int:
        """预留给对话历史的 token."""
        usable = self.model_context_limit - self.reserved_for_output
        return int(usable * self.reserved_for_conversation_ratio)

    @property
    def available_for_context(self) -> int:
        """实际可用于上下文的 token.

        available = model_limit - reserved_output - reserved_conversation
        """
        return (
            self.model_context_limit
            - self.reserved_for_output
            - self.reserved_for_conversation
        )

    @property
    def initial_context_budget(self) -> int:
        """给 initial context（system prompt）的份额：available 的 40%."""
        return int(self.available_for_context * 0.4)

    @property
    def task_context_budget(self) -> int:
        """给 task context 的份额：available 的 60%."""
        return int(self.available_for_context * 0.6)


@dataclass
class ContextStats:
    """上下文使用统计."""

    total_budget: int = 0
    initial_context_tokens: int = 0
    task_context_tokens: int = 0
    conversation_tokens: int = 0
    remaining_tokens: int = 0
    cache_friendly_prefix_tokens: int = 0  # 可被 KV cache 复用的前缀长度


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------

@dataclass
class ContextBuilder:
    """上下文组装器 — 负责拼接 system prompt 和 task context."""

    project_path: Path = field(default_factory=lambda: Path.cwd())
    budget: ContextBudget = field(default_factory=ContextBudget)

    # 缓存
    _project_info: ProjectInfo | None = field(default=None, repr=False)
    _initial_context: str | None = field(default=None, repr=False)
    _initial_context_tokens: int = field(default=0, repr=False)
    _cache_prefix_tokens: int = field(default=0, repr=False)

    def analyze_project(self) -> ProjectInfo:
        """分析项目并缓存结果."""
        self._project_info = detect_project_type(self.project_path)
        return self._project_info

    @property
    def project_info(self) -> ProjectInfo:
        if self._project_info is None:
            self.analyze_project()
        return self._project_info  # type: ignore[return-value]

    def build_initial_context(self, base_instructions: str) -> str:
        """组装会话开始时注入 system prompt 的项目上下文.

        拼接顺序（按稳定性从高到低排列）：
          1. 基础指令（角色定义 + 工具使用规范）     — 永远不变
          2. AGENT.md / CLAUDE.md 项目指令文件       — 几乎不变
          3. 项目元信息（类型、语言、框架、包管理器）  — 几乎不变
          4. 目录树（top 2-3 levels）                — 偶尔变化
          5. Repo Map（文件列表 + 签名摘要）          — 偶尔变化

        为什么这个顺序重要？
        LLM API 对 system prompt 的前缀部分有 KV cache 优化：
        如果两次请求的 system prompt 前缀相同，第二次不需要重新计算
        attention，直接复用缓存。所以把跨轮永远不变的内容放在最前面。

        Args:
            base_instructions: 基础指令字符串（角色定义、工具使用规范）

        Returns:
            拼装好的 system prompt 字符串
        """
        budget = self.budget.initial_context_budget
        used_tokens = 0
        sections: list[str] = []

        # --- Section 1: 基础指令（必须包含，不可省略）---
        sections.append(base_instructions)
        used_tokens += estimate_tokens(base_instructions)
        # 基础指令是最稳定的部分，记为 cache-friendly 前缀
        cache_prefix_end = used_tokens

        # --- Section 2: 项目指令文件 AGENT.md / CLAUDE.md（必须包含）---
        project_instructions = self._read_project_instructions()
        if project_instructions:
            section = f"\n<project-instructions>\n{project_instructions}\n</project-instructions>"
            section_tokens = estimate_tokens(section)
            sections.append(section)
            used_tokens += section_tokens
            # 指令文件几乎不变，也是 cache-friendly 的
            cache_prefix_end = used_tokens

        # --- Section 3: 项目元信息（必须包含，很短 <200 token）---
        info = self.project_info
        meta_section = self._format_project_meta(info)
        if meta_section:
            section = f"\n<project-meta>\n{meta_section}\n</project-meta>"
            section_tokens = estimate_tokens(section)
            sections.append(section)
            used_tokens += section_tokens

        # --- Section 4: 目录树（如果超预算，减少 max_depth）---
        remaining = budget - used_tokens
        if remaining > 100:
            tree_section = self._build_tree_section(remaining)
            if tree_section:
                section = f"\n<directory-tree>\n{tree_section}\n</directory-tree>"
                section_tokens = estimate_tokens(section)
                if used_tokens + section_tokens <= budget:
                    sections.append(section)
                    used_tokens += section_tokens

        # --- Section 5: Repo Map（如果超预算，按策略降级）---
        remaining = budget - used_tokens
        if remaining > 200:
            map_section = self._build_repo_map_section(remaining)
            if map_section:
                section = f"\n<repo-map>\n{map_section}\n</repo-map>"
                section_tokens = estimate_tokens(section)
                if used_tokens + section_tokens <= budget:
                    sections.append(section)
                    used_tokens += section_tokens
                else:
                    # 尝试截断到预算以内
                    truncated = self._truncate_to_budget(
                        map_section, remaining - 50  # 留一点给标签
                    )
                    if truncated:
                        section = (
                            f"\n<repo-map>\n{truncated}\n"
                            "（Repo Map 已截断，Agent 可使用 ListDir / Grep 探索更多文件）\n"
                            "</repo-map>"
                        )
                        sections.append(section)
                        used_tokens += estimate_tokens(section)

        result = "\n".join(sections)
        self._initial_context = result
        self._initial_context_tokens = used_tokens
        self._cache_prefix_tokens = cache_prefix_end
        return result

    def build_task_context(
        self,
        relevant_files: list[str],
        *,
        budget: ContextBudget | None = None,
    ) -> str:
        """为特定任务动态组装上下文.

        把 relevant_files 的内容读出来，拼成一段上下文，
        每个文件标注路径和行号，方便 Agent 引用。

        Args:
            relevant_files: 与当前任务相关的文件路径列表
                （按优先级排序，越靠前越重要）
            budget: 可选的预算覆盖

        Returns:
            拼装好的 task context 字符串
        """
        effective_budget = budget or self.budget
        task_budget = effective_budget.task_context_budget
        single_file_max = int(task_budget * 0.5)  # 单文件最大占 50%

        used_tokens = 0
        sections: list[str] = []
        omitted_files: list[str] = []

        for file_path_str in relevant_files:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                file_path = self.project_path / file_path

            if not file_path.exists() or not file_path.is_file():
                continue

            remaining = task_budget - used_tokens
            if remaining < 100:
                omitted_files.append(file_path_str)
                continue

            content = self._read_file_for_context(
                file_path, min(remaining, single_file_max)
            )
            if not content:
                continue

            lines = content.splitlines()
            total_lines = self._count_file_lines(file_path)
            shown_lines = len(lines)

            if shown_lines == total_lines:
                header = f"=== {file_path_str} (1-{total_lines}, 共 {total_lines} 行) ==="
            else:
                header = (
                    f"=== {file_path_str} (已截断, 显示 {shown_lines}/{total_lines} 行) ==="
                )

            # 添加行号
            width = len(str(total_lines))
            numbered_lines = []
            for i, line in enumerate(lines, start=1):
                numbered_lines.append(f"{i:>{width}}  {line}")

            section = f"{header}\n" + "\n".join(numbered_lines)
            section_tokens = estimate_tokens(section)

            if used_tokens + section_tokens > task_budget:
                omitted_files.append(file_path_str)
                continue

            sections.append(section)
            used_tokens += section_tokens

        if omitted_files:
            sections.append(
                f"\n（以下文件已省略: {', '.join(omitted_files)}）\n"
                "（可使用 ReadFile 按需读取）"
            )

        return "\n\n".join(sections)

    def get_context_stats(self, conversation_tokens: int = 0) -> ContextStats:
        """返回当前上下文的使用情况."""
        initial_tokens = self._initial_context_tokens
        total = self.budget.available_for_context

        return ContextStats(
            total_budget=total,
            initial_context_tokens=initial_tokens,
            task_context_tokens=0,  # 调用者可以自行更新
            conversation_tokens=conversation_tokens,
            remaining_tokens=total - initial_tokens - conversation_tokens,
            cache_friendly_prefix_tokens=self._cache_prefix_tokens,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _read_project_instructions(self) -> str:
        """读取项目指令文件 CLAUDE.md / AGENT.md."""
        candidates = ["CLAUDE.md", "AGENT.md"]
        for name in candidates:
            p = self.project_path / name
            if p.exists() and p.is_file():
                try:
                    return p.read_text(encoding="utf-8").strip()
                except (UnicodeDecodeError, PermissionError):
                    continue
        return ""

    def _format_project_meta(self, info: ProjectInfo) -> str:
        """格式化项目元信息."""
        lines: list[str] = []
        lines.append(f"- 工作目录: {self.project_path}")

        if info.name:
            lines.append(f"- 项目名称: {info.name}")
        if info.language != "Unknown":
            lines.append(f"- 语言: {info.language}")
        if info.framework:
            lines.append(f"- 框架: {info.framework}")
        if info.package_manager:
            lines.append(f"- 包管理器: {info.package_manager}")
        if info.version:
            lines.append(f"- 版本: {info.version}")
        if info.entry_points:
            lines.append(f"- 入口文件: {', '.join(info.entry_points)}")

        return "\n".join(lines) if lines else ""

    def _build_tree_section(self, budget_tokens: int) -> str:
        """生成目录树，超预算时减少深度."""
        for depth in (3, 2, 1):
            tree = get_directory_tree(self.project_path, max_depth=depth)
            tokens = estimate_tokens(tree)
            if tokens <= budget_tokens:
                return tree
        return ""

    def _build_repo_map_section(self, budget_tokens: int) -> str:
        """生成 Repo Map，超预算时降级.

        降级策略：
          a. 先尝试完整 repo map（路径 + 签名摘要）
          b. 超预算 → 只保留文件路径列表（去掉签名摘要）
          c. 还超 → 截断到预算以内
        """
        # 尝试完整 repo map
        full_map = build_repo_map(self.project_path)
        if estimate_tokens(full_map) <= budget_tokens:
            return full_map

        # 降级：只列路径
        paths_only = build_repo_map_paths_only(self.project_path)
        if estimate_tokens(paths_only) <= budget_tokens:
            return paths_only

        # 再降级：截断
        return self._truncate_to_budget(paths_only, budget_tokens)

    def _truncate_to_budget(self, text: str, budget_tokens: int) -> str:
        """按行截断文本，确保不超过 token 预算."""
        lines = text.splitlines()
        result_lines: list[str] = []
        used = 0

        for line in lines:
            line_tokens = estimate_tokens(line)
            if used + line_tokens > budget_tokens:
                break
            result_lines.append(line)
            used += line_tokens

        if result_lines and len(result_lines) < len(lines):
            result_lines.append(
                f"... （共 {len(lines)} 个文件，已显示 {len(result_lines)} 个）"
            )

        return "\n".join(result_lines)

    def _read_file_for_context(self, file_path: Path, max_tokens: int) -> str:
        """读取文件内容，超长则截断（首尾保留）."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            return ""

        tokens = estimate_tokens(content)
        if tokens <= max_tokens:
            return content

        # 超长文件：保留首尾
        lines = content.splitlines()
        total = len(lines)

        # 头部占 70%，尾部占 30%
        head_budget = int(max_tokens * 0.7)
        tail_budget = max_tokens - head_budget

        head_lines: list[str] = []
        head_used = 0
        for line in lines:
            lt = estimate_tokens(line)
            if head_used + lt > head_budget:
                break
            head_lines.append(line)
            head_used += lt

        tail_lines: list[str] = []
        tail_used = 0
        for line in reversed(lines):
            lt = estimate_tokens(line)
            if tail_used + lt > tail_budget:
                break
            tail_lines.insert(0, line)
            tail_used += lt

        head_end = len(head_lines)
        tail_start = total - len(tail_lines) + 1

        truncation_msg = (
            f"\n... [此处省略第 {head_end + 1}-{tail_start - 1} 行，"
            f"共 {tail_start - head_end - 1} 行被截断]\n"
            f"（可使用 ReadFile 读取完整内容）"
        )

        return (
            "\n".join(head_lines)
            + truncation_msg
            + "\n".join(tail_lines)
        )

    def _count_file_lines(self, file_path: Path) -> int:
        """统计文件行数."""
        try:
            return len(file_path.read_text(encoding="utf-8").splitlines())
        except (UnicodeDecodeError, PermissionError, OSError):
            return 0
