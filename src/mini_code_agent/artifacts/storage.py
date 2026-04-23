"""ArtifactStore — Artifact 持久化存储."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .artifact import SubtaskArtifact

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMeta:
    """Artifact 的元信息（用于列表展示，不含完整内容）."""

    artifact_id: str
    task_id: str
    status: str
    confidence: str
    created_at: str
    producer: str
    files_changed: int
    path: str  # 磁盘路径


class ArtifactStore:
    """Artifact 持久化存储.

    文件布局：
      {storage_dir}/
        {task_id}/
          {artifact_id}.json        # Artifact 本体
          {artifact_id}.patch       # 原始 patch，独立存储便于 diff 查看
    """

    def __init__(self, storage_dir: str = ".agent/artifacts/") -> None:
        self.storage_dir = storage_dir

    def save(self, artifact: SubtaskArtifact) -> str:
        """保存 Artifact 到磁盘，返回存储路径."""
        base = Path(self.storage_dir) / artifact.task_id
        base.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        json_path = base / f"{artifact.artifact_id}.json"
        artifact.save(str(json_path))

        # 保存 patch 文件
        patch_path = base / f"{artifact.artifact_id}.patch"
        diff_content = artifact.patch.to_unified_diff()
        patch_path.write_text(diff_content, encoding="utf-8")

        logger.info("Artifact 已保存: %s", json_path)
        return str(json_path)

    def load(self, artifact_id: str) -> SubtaskArtifact:
        """按 artifact_id 加载 Artifact（需要搜索所有 task 目录）."""
        root = Path(self.storage_dir)
        if not root.exists():
            raise FileNotFoundError(f"存储目录不存在: {self.storage_dir}")

        for task_dir in root.iterdir():
            if not task_dir.is_dir():
                continue
            json_path = task_dir / f"{artifact_id}.json"
            if json_path.exists():
                return SubtaskArtifact.load(str(json_path))

        raise FileNotFoundError(f"Artifact 不存在: {artifact_id}")

    def list_for_task(self, task_id: str) -> list[ArtifactMeta]:
        """列出某个 task 的所有 artifact（含被 reject 的历史版本）."""
        task_dir = Path(self.storage_dir) / task_id
        if not task_dir.exists():
            return []

        metas: list[ArtifactMeta] = []
        for json_file in sorted(task_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                metas.append(
                    ArtifactMeta(
                        artifact_id=data["artifact_id"],
                        task_id=data["task_id"],
                        status=data["status"],
                        confidence=data["confidence"],
                        created_at=data["created_at"],
                        producer=data["producer"],
                        files_changed=data["patch"]["total_files_changed"],
                        path=str(json_file),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("读取 artifact 元信息失败: %s: %s", json_file, e)
                continue

        return metas

    def get_latest_for_task(self, task_id: str) -> SubtaskArtifact | None:
        """获取某个 task 最新的 artifact."""
        metas = self.list_for_task(task_id)
        if not metas:
            return None

        # 按 created_at 排序取最新
        metas.sort(key=lambda m: m.created_at, reverse=True)
        return SubtaskArtifact.load(metas[0].path)
