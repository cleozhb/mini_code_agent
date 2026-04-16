"""System Prompt 构建 — 为 Agent 注入身份和行为指令."""

from __future__ import annotations

from pathlib import Path

from ..context.context_builder import ContextBudget, ContextBuilder, estimate_tokens

DEFAULT_SYSTEM_PROMPT = """\
你是一个专业的编程助手。你可以通过工具来读写文件、搜索代码和执行命令，帮助用户完成编程任务。

## 工作原则

1. **先思考再行动**：在使用工具之前，先分析用户的需求，制定计划。
2. **先读后改**：修改文件之前，先用 read_file 读取文件内容确认当前状态。
3. **谨慎执行**：执行命令前评估风险，避免破坏性操作。
4. **主动沟通**：遇到不确定的事情，主动询问用户，不要擅自做决定。

## 工具使用指南

- `read_file`：读取文件内容，支持指定行号范围
- `edit_file`：局部编辑文件，通过查找替换修改指定片段。修改已有文件时优先使用此工具
  - old_content 要包含足够的上下文行，确保在文件中唯一匹配
  - new_content 为空字符串表示删除对应片段
- `write_file`：写入文件，会覆盖原有内容。适合创建新文件或重写小文件（<50 行）
- `bash`：执行 shell 命令，有 30 秒超时限制
- `grep`：搜索代码，支持正则表达式
- `list_dir`：列出目录结构

## 回答要求

- 简洁清晰，直接解决问题
- 修改已有文件时使用 edit_file 进行局部编辑，避免用 write_file 覆盖整个文件
- 遇到错误时分析原因，提出修复方案
"""


def build_system_prompt(
    project_info: dict | None = None,
    *,
    project_path: str | Path | None = None,
    budget: ContextBudget | None = None,
) -> str:
    """构建完整的 system prompt，可注入项目上下文.

    如果提供了 project_path，将使用 ContextBuilder 自动分析项目
    并按 KV cache 友好的顺序拼接上下文。

    Args:
        project_info: 兼容旧接口的项目信息字典（可选）
        project_path: 项目根目录路径，传入后启用自动分析
        budget: Token 预算配置

    Returns:
        拼装好的 system prompt 字符串。
    """
    # 新路径：使用 ContextBuilder 自动分析
    if project_path is not None:
        effective_budget = budget or ContextBudget()
        builder = ContextBuilder(
            project_path=Path(project_path).resolve(),
            budget=effective_budget,
        )
        return builder.build_initial_context(DEFAULT_SYSTEM_PROMPT)

    # 旧路径：手动拼接简单信息（向后兼容）
    parts = [DEFAULT_SYSTEM_PROMPT]

    if project_info:
        context_lines = ["\n## 当前项目信息\n"]

        if cwd := project_info.get("cwd"):
            context_lines.append(f"- 工作目录：{cwd}")

        if name := project_info.get("project_name"):
            context_lines.append(f"- 项目名称：{name}")

        if stack := project_info.get("tech_stack"):
            context_lines.append(f"- 技术栈：{stack}")

        if conventions := project_info.get("conventions"):
            context_lines.append(f"\n### 编码规范\n{conventions}")

        if extra := project_info.get("extra"):
            context_lines.append(f"\n### 补充信息\n{extra}")

        parts.append("\n".join(context_lines))

    return "\n".join(parts)


def build_system_prompt_with_context(
    project_path: str | Path,
    budget: ContextBudget | None = None,
) -> tuple[str, ContextBuilder]:
    """构建 system prompt 并返回 ContextBuilder 供后续使用.

    Returns:
        (system_prompt, context_builder) 元组
    """
    effective_budget = budget or ContextBudget()
    builder = ContextBuilder(
        project_path=Path(project_path).resolve(),
        budget=effective_budget,
    )
    prompt = builder.build_initial_context(DEFAULT_SYSTEM_PROMPT)
    return prompt, builder
