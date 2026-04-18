"""Workspace snapshot & diff 测试."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mini_code_agent.eval.snapshot import (
    FileFingerprint,
    SnapshotDiff,
    capture,
    diff,
)


def _write(p: Path, content: str = "hello\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


class TestCapture:
    def test_captures_all_files(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.txt", "A")
        _write(tmp_path / "sub" / "b.py", "B")
        snap = capture(tmp_path)
        assert set(snap.keys()) == {"a.txt", "sub/b.py"}
        assert snap["a.txt"].size == 1
        assert snap["a.txt"].sha256.startswith(
            "559aead08264d5795d3909718cdd05abd49572e84fe55590eef31a88a08fdffd"
        )  # sha256("A")

    def test_ignores_pycache_and_pyc_and_ds_store(self, tmp_path: Path) -> None:
        _write(tmp_path / "keep.py", "x")
        _write(tmp_path / "__pycache__" / "mod.cpython-312.pyc", "garbage")
        _write(tmp_path / "sub" / "__pycache__" / "x.pyc", "garbage")
        _write(tmp_path / "scratch.pyc", "garbage")
        _write(tmp_path / ".DS_Store", "junk")
        snap = capture(tmp_path)
        assert set(snap.keys()) == {"keep.py"}

    def test_ignores_dot_git(self, tmp_path: Path) -> None:
        _write(tmp_path / "keep.py", "x")
        _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main")
        _write(tmp_path / ".git" / "objects" / "ab" / "cd", "blob")
        snap = capture(tmp_path)
        assert set(snap.keys()) == {"keep.py"}

    def test_ignores_agent_backups_and_pytest_cache(self, tmp_path: Path) -> None:
        """WriteFileTool 自动备份目录和 Agent 跑 pytest 的副产物都不算"真改动".

        出发点见 DESIGN §9.8：这些文件会把 edit_precision 拉低到离谱的数（0.14 实测），
        让 Agent 看起来写了一堆没用的文件，掩盖它真正的产出对不对。
        """
        _write(tmp_path / "keep.py", "x")
        _write(tmp_path / ".agent-backups" / "keep.py.1700000.bak", "old content")
        _write(tmp_path / ".agent-backups" / "nested" / "deep.bak", "x")
        _write(tmp_path / "tests" / ".pytest_cache" / "v" / "cache" / "nodeids", "{}")
        _write(tmp_path / "tests" / ".pytest_cache" / "CACHEDIR.TAG", "Signature")
        _write(tmp_path / ".pytest_cache" / "README.md", "# pytest cache")
        snap = capture(tmp_path)
        assert set(snap.keys()) == {"keep.py"}

    def test_missing_workspace_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            capture(tmp_path / "does-not-exist")


class TestDiff:
    def test_unchanged_is_empty(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.txt", "A")
        before = capture(tmp_path)
        d = diff(tmp_path, before)
        assert d.added == []
        assert d.modified == []
        assert d.removed == []
        assert d.is_empty()
        assert d.changed == []

    def test_added(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.txt", "A")
        before = capture(tmp_path)
        _write(tmp_path / "b.txt", "B")
        _write(tmp_path / "sub" / "c.py", "C")
        d = diff(tmp_path, before)
        assert d.added == ["b.txt", "sub/c.py"]
        assert d.modified == []
        assert d.removed == []

    def test_modified(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.txt", "A")
        before = capture(tmp_path)
        _write(tmp_path / "a.txt", "A-changed")
        d = diff(tmp_path, before)
        assert d.modified == ["a.txt"]
        assert d.added == []
        assert d.removed == []

    def test_removed(self, tmp_path: Path) -> None:
        _write(tmp_path / "a.txt", "A")
        _write(tmp_path / "b.txt", "B")
        before = capture(tmp_path)
        (tmp_path / "a.txt").unlink()
        d = diff(tmp_path, before)
        assert d.removed == ["a.txt"]
        assert d.added == []
        assert d.modified == []

    def test_mtime_only_change_not_modified(self, tmp_path: Path) -> None:
        """只改 mtime、内容不变 → 不算 modified（按 sha256 判断）."""
        p = tmp_path / "a.txt"
        _write(p, "same content")
        before = capture(tmp_path)
        # 把 mtime 往后推 5 秒
        st = p.stat()
        future = st.st_mtime + 5
        os.utime(p, (future, future))
        d = diff(tmp_path, before)
        assert d.modified == []
        assert d.is_empty()

    def test_changed_property_merges_all(self, tmp_path: Path) -> None:
        _write(tmp_path / "keep.txt", "k")
        _write(tmp_path / "del.txt", "d")
        _write(tmp_path / "mod.txt", "m")
        before = capture(tmp_path)
        _write(tmp_path / "new.txt", "n")
        (tmp_path / "del.txt").unlink()
        _write(tmp_path / "mod.txt", "m2")
        d = diff(tmp_path, before)
        assert d.added == ["new.txt"]
        assert d.removed == ["del.txt"]
        assert d.modified == ["mod.txt"]
        assert d.changed == ["del.txt", "mod.txt", "new.txt"]
        assert not d.is_empty()
