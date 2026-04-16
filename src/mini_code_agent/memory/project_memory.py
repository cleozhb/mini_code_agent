"""项目级长期记忆 — 持久化到 .agent/memory.json."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_DIR = ".agent"
MEMORY_FILE = "memory.json"


@dataclass
class Decision:
    """一条决策记录."""

    date: str
    decision: str
    reason: str


@dataclass
class KnownIssue:
    """一条已知问题及解法."""

    issue: str
    solution: str


@dataclass
class ProjectMemoryData:
    """项目记忆的数据结构."""

    conventions: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    known_issues: list[KnownIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conventions": self.conventions,
            "decisions": [
                {"date": d.date, "decision": d.decision, "reason": d.reason}
                for d in self.decisions
            ],
            "known_issues": [
                {"issue": ki.issue, "solution": ki.solution}
                for ki in self.known_issues
            ],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ProjectMemoryData:
        return ProjectMemoryData(
            conventions=data.get("conventions", []),
            decisions=[
                Decision(
                    date=d.get("date", ""),
                    decision=d.get("decision", ""),
                    reason=d.get("reason", ""),
                )
                for d in data.get("decisions", [])
            ],
            known_issues=[
                KnownIssue(
                    issue=ki.get("issue", ""),
                    solution=ki.get("solution", ""),
                )
                for ki in data.get("known_issues", [])
            ],
        )


class ProjectMemory:
    """项目级长期记忆管理器.

    读写项目目录下的 .agent/memory.json，存储：
    - conventions: 项目约定
    - decisions: 技术决策及原因
    - known_issues: 已知问题及解法
    """

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir)
        self._memory_dir = self.project_dir / MEMORY_DIR
        self._memory_file = self._memory_dir / MEMORY_FILE
        self._data: ProjectMemoryData | None = None

    @property
    def data(self) -> ProjectMemoryData:
        """懒加载记忆数据."""
        if self._data is None:
            self._data = self._load()
        return self._data

    def _load(self) -> ProjectMemoryData:
        """从磁盘加载记忆."""
        if not self._memory_file.exists():
            return ProjectMemoryData()
        try:
            raw = json.loads(self._memory_file.read_text(encoding="utf-8"))
            return ProjectMemoryData.from_dict(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("加载项目记忆失败: %s，使用空记忆", e)
            return ProjectMemoryData()

    def save(self) -> None:
        """持久化到磁盘."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file.write_text(
            json.dumps(self.data.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("项目记忆已保存到 %s", self._memory_file)

    # ------------------------------------------------------------------
    # 增删查接口
    # ------------------------------------------------------------------

    def add_convention(self, convention: str) -> None:
        """添加一条项目约定."""
        if convention not in self.data.conventions:
            self.data.conventions.append(convention)
            self.save()

    def add_decision(self, decision: str, reason: str) -> None:
        """添加一条技术决策."""
        self.data.decisions.append(
            Decision(
                date=datetime.now().strftime("%Y-%m-%d"),
                decision=decision,
                reason=reason,
            )
        )
        self.save()

    def add_known_issue(self, issue: str, solution: str) -> None:
        """添加一条已知问题及解法."""
        self.data.known_issues.append(KnownIssue(issue=issue, solution=solution))
        self.save()

    def recall(self, keyword: str) -> list[str]:
        """搜索包含关键词的记忆条目.

        Returns:
            匹配到的记忆条目列表（字符串形式）。
        """
        keyword_lower = keyword.lower()
        results: list[str] = []

        for conv in self.data.conventions:
            if keyword_lower in conv.lower():
                results.append(f"[约定] {conv}")

        for dec in self.data.decisions:
            if keyword_lower in dec.decision.lower() or keyword_lower in dec.reason.lower():
                results.append(
                    f"[决策 {dec.date}] {dec.decision} — 原因: {dec.reason}"
                )

        for ki in self.data.known_issues:
            if keyword_lower in ki.issue.lower() or keyword_lower in ki.solution.lower():
                results.append(f"[已知问题] {ki.issue} — 解法: {ki.solution}")

        return results

    def format_for_prompt(self) -> str:
        """将记忆格式化为可注入 system prompt 的文本."""
        data = self.data
        if not data.conventions and not data.decisions and not data.known_issues:
            return ""

        sections: list[str] = []

        if data.conventions:
            items = "\n".join(f"  - {c}" for c in data.conventions)
            sections.append(f"项目约定:\n{items}")

        if data.decisions:
            items = "\n".join(
                f"  - [{d.date}] {d.decision}（原因: {d.reason}）"
                for d in data.decisions[-5:]  # 只保留最近 5 条
            )
            sections.append(f"技术决策:\n{items}")

        if data.known_issues:
            items = "\n".join(
                f"  - {ki.issue} → {ki.solution}"
                for ki in data.known_issues[-5:]  # 只保留最近 5 条
            )
            sections.append(f"已知问题:\n{items}")

        return "\n\n".join(sections)
