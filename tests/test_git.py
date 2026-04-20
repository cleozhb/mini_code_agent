"""Git 工具 + Checkpoint 测试.

在临时 git repo 中测试所有 Git 功能。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from mini_code_agent.tools.git import (
    GitCommitTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    _run_git,
)
from mini_code_agent.tools.base import PermissionLevel
from mini_code_agent.safety.git_checkpoint import GitCheckpoint, CHECKPOINT_PREFIX


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
async def git_repo(tmp_path: Path) -> Path:
    """创建一个临时 git repo，带一个初始 commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    await _run_git("init", cwd=str(repo))
    await _run_git("config", "user.email", "test@test.com", cwd=str(repo))
    await _run_git("config", "user.name", "Test", cwd=str(repo))

    # 创建初始文件并提交
    (repo / "README.md").write_text("# Test Project\n")
    await _run_git("add", "-A", cwd=str(repo))
    await _run_git("commit", "-m", "initial commit", cwd=str(repo))
    return repo


@pytest.fixture
def non_git_dir(tmp_path: Path) -> Path:
    """一个非 git 目录."""
    d = tmp_path / "not_git"
    d.mkdir()
    return d


# ===========================================================================
# GitStatusTool 测试
# ===========================================================================


class TestGitStatusTool:
    @pytest.fixture
    def tool(self) -> GitStatusTool:
        return GitStatusTool()

    async def test_permission_level(self, tool: GitStatusTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    async def test_clean_repo(self, tool: GitStatusTool, git_repo: Path) -> None:
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "分支:" in result.output
        assert "工作区干净" in result.output

    async def test_modified_file(self, tool: GitStatusTool, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Modified\n")
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "Unstaged" in result.output
        assert "README.md" in result.output

    async def test_untracked_file(self, tool: GitStatusTool, git_repo: Path) -> None:
        (git_repo / "new_file.txt").write_text("hello\n")
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "Untracked" in result.output
        assert "new_file.txt" in result.output

    async def test_staged_file(self, tool: GitStatusTool, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Staged\n")
        await _run_git("add", "README.md", cwd=str(git_repo))
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "Staged" in result.output

    async def test_non_git_dir(self, tool: GitStatusTool, non_git_dir: Path) -> None:
        result = await tool.execute(path=str(non_git_dir))
        assert result.is_error
        assert "不是 git 仓库" in result.error


# ===========================================================================
# GitDiffTool 测试
# ===========================================================================


class TestGitDiffTool:
    @pytest.fixture
    def tool(self) -> GitDiffTool:
        return GitDiffTool()

    async def test_permission_level(self, tool: GitDiffTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    async def test_no_changes(self, tool: GitDiffTool, git_repo: Path) -> None:
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "无 unstaged 改动" in result.output

    async def test_unstaged_diff(self, tool: GitDiffTool, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Changed\n")
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "Changed" in result.output
        assert "diff" in result.output.lower()

    async def test_staged_diff(self, tool: GitDiffTool, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Staged Change\n")
        await _run_git("add", "README.md", cwd=str(git_repo))
        result = await tool.execute(path=str(git_repo), staged=True)
        assert not result.is_error
        assert "Staged Change" in result.output

    async def test_truncation(self, git_repo: Path) -> None:
        tool = GitDiffTool(max_lines=5)
        # 创建大量改动
        (git_repo / "README.md").write_text("\n".join(f"line {i}" for i in range(500)))
        result = await tool.execute(path=str(git_repo))
        assert not result.is_error
        assert "截断" in result.output


# ===========================================================================
# GitCommitTool 测试
# ===========================================================================


class TestGitCommitTool:
    @pytest.fixture
    def tool(self) -> GitCommitTool:
        return GitCommitTool()

    async def test_permission_level(self, tool: GitCommitTool) -> None:
        assert tool.permission_level == PermissionLevel.CONFIRM

    async def test_commit_changes(self, tool: GitCommitTool, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("# Updated\n")
        # 需要切到 repo 目录（GitCommitTool 不接受 path 参数，它 stage -u 在 cwd）
        old_cwd = os.getcwd()
        try:
            os.chdir(git_repo)
            result = await tool.execute(message="test commit")
        finally:
            os.chdir(old_cwd)
        assert not result.is_error
        assert "提交成功" in result.output

        # 验证 commit 确实存在
        code, log = await _run_git("log", "--oneline", "-1", cwd=str(git_repo))
        assert "test commit" in log

    async def test_no_changes_to_commit(self, tool: GitCommitTool, git_repo: Path) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(git_repo)
            result = await tool.execute(message="empty")
        finally:
            os.chdir(old_cwd)
        assert result.is_error
        assert "没有已暂存的改动" in result.error


# ===========================================================================
# GitLogTool 测试
# ===========================================================================


class TestGitLogTool:
    @pytest.fixture
    def tool(self) -> GitLogTool:
        return GitLogTool()

    async def test_permission_level(self, tool: GitLogTool) -> None:
        assert tool.permission_level == PermissionLevel.AUTO

    async def test_basic_log(self, tool: GitLogTool, git_repo: Path) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(git_repo)
            result = await tool.execute()
        finally:
            os.chdir(old_cwd)
        assert not result.is_error
        assert "initial commit" in result.output

    async def test_log_count(self, tool: GitLogTool, git_repo: Path) -> None:
        # 创建几个额外的 commit
        for i in range(3):
            (git_repo / f"file{i}.txt").write_text(f"content {i}")
            await _run_git("add", "-A", cwd=str(git_repo))
            await _run_git("commit", "-m", f"commit {i}", cwd=str(git_repo))

        old_cwd = os.getcwd()
        try:
            os.chdir(git_repo)
            result = await tool.execute(count=2)
        finally:
            os.chdir(old_cwd)
        assert not result.is_error
        lines = result.output.strip().splitlines()
        assert len(lines) == 2


# ===========================================================================
# GitCheckpoint 测试
# ===========================================================================


class TestGitCheckpoint:
    @pytest.fixture
    def checkpoint(self, git_repo: Path) -> GitCheckpoint:
        return GitCheckpoint(cwd=str(git_repo))

    async def test_is_git_repo(self, checkpoint: GitCheckpoint) -> None:
        assert await checkpoint.is_git_repo() is True

    async def test_not_git_repo(self, non_git_dir: Path) -> None:
        cp = GitCheckpoint(cwd=str(non_git_dir))
        assert await cp.is_git_repo() is False

    async def test_save_head(self, checkpoint: GitCheckpoint, git_repo: Path) -> None:
        """save_head 只记录 HEAD hash，不创建 commit."""
        code, expected_head = await _run_git("rev-parse", "HEAD", cwd=str(git_repo))
        expected_head = expected_head.strip()

        result = await checkpoint.save_head()
        assert result == expected_head
        assert checkpoint._before_head == expected_head

        # 不应该有新的 commit
        code, current_head = await _run_git("rev-parse", "HEAD", cwd=str(git_repo))
        assert current_head.strip() == expected_head

    async def test_save_head_non_git(self, non_git_dir: Path) -> None:
        cp = GitCheckpoint(cwd=str(non_git_dir))
        result = await cp.save_head()
        assert result is None

    async def test_create_checkpoint(
        self, checkpoint: GitCheckpoint, git_repo: Path
    ) -> None:
        # 创建改动
        (git_repo / "new_file.py").write_text("print('hello')\n")
        commit_hash = await checkpoint.create_checkpoint("test checkpoint")
        assert commit_hash is not None
        assert len(commit_hash) == 40  # full SHA

        # 验证 commit message
        code, log = await _run_git("log", "--oneline", "-1", cwd=str(git_repo))
        assert CHECKPOINT_PREFIX in log
        assert "test checkpoint" in log

    async def test_create_checkpoint_no_changes(
        self, checkpoint: GitCheckpoint
    ) -> None:
        """没有改动时跳过 checkpoint."""
        result = await checkpoint.create_checkpoint("no changes")
        assert result is None

    async def test_create_checkpoint_non_git(self, non_git_dir: Path) -> None:
        """非 git 目录降级处理."""
        cp = GitCheckpoint(cwd=str(non_git_dir))
        result = await cp.create_checkpoint("should skip")
        assert result is None

    async def test_list_checkpoints(
        self, checkpoint: GitCheckpoint, git_repo: Path
    ) -> None:
        # 创建两个 checkpoint
        (git_repo / "a.txt").write_text("aaa")
        await checkpoint.create_checkpoint("first")
        (git_repo / "b.txt").write_text("bbb")
        await checkpoint.create_checkpoint("second")

        cps = await checkpoint.list_checkpoints()
        assert len(cps) == 2
        # 最近的在前
        assert "second" in cps[0].message
        assert "first" in cps[1].message

    async def test_rollback_to(
        self, checkpoint: GitCheckpoint, git_repo: Path
    ) -> None:
        original_content = (git_repo / "README.md").read_text()

        # 修改文件并创建 checkpoint
        (git_repo / "README.md").write_text("MODIFIED\n")
        commit_hash = await checkpoint.create_checkpoint("modify readme")
        assert commit_hash is not None

        # 回滚到 checkpoint 之前
        success = await checkpoint.rollback_to(f"{commit_hash}~1")
        assert success is True

        # 验证文件恢复
        assert (git_repo / "README.md").read_text() == original_content

    async def test_rollback_last(
        self, checkpoint: GitCheckpoint, git_repo: Path
    ) -> None:
        original_content = (git_repo / "README.md").read_text()

        # 修改文件并创建 checkpoint
        (git_repo / "README.md").write_text("CHANGED\n")
        await checkpoint.create_checkpoint("change readme")

        # 用 rollback_last 回滚
        success = await checkpoint.rollback_last()
        assert success is True

        # 验证文件恢复
        assert (git_repo / "README.md").read_text() == original_content

    async def test_rollback_last_no_checkpoints(
        self, checkpoint: GitCheckpoint
    ) -> None:
        success = await checkpoint.rollback_last()
        assert success is False

    async def test_rollback_restores_file_content(
        self, checkpoint: GitCheckpoint, git_repo: Path
    ) -> None:
        """确认回滚后文件内容完全恢复."""
        # 写入多个文件
        (git_repo / "a.py").write_text("def a(): pass\n")
        (git_repo / "b.py").write_text("def b(): pass\n")
        await checkpoint.create_checkpoint("add files")

        # 修改 + 删除
        (git_repo / "a.py").write_text("def a(): return 1\n")
        (git_repo / "b.py").unlink()
        (git_repo / "c.py").write_text("def c(): pass\n")
        await checkpoint.create_checkpoint("modify files")

        # 回滚到上一个 checkpoint
        success = await checkpoint.rollback_last()
        assert success is True

        # 验证状态恢复
        assert (git_repo / "a.py").read_text() == "def a(): pass\n"
        assert (git_repo / "b.py").read_text() == "def b(): pass\n"
        assert not (git_repo / "c.py").exists()

    async def test_cleanup_checkpoints(
        self, checkpoint: GitCheckpoint, git_repo: Path
    ) -> None:
        # 创建 7 个 checkpoint
        for i in range(7):
            (git_repo / f"file{i}.txt").write_text(f"content {i}")
            await checkpoint.create_checkpoint(f"checkpoint {i}")

        cps_before = await checkpoint.list_checkpoints()
        assert len(cps_before) == 7

        # 清理，保留最近 5 个
        cleaned = await checkpoint.cleanup_checkpoints(keep_last_n=5)
        assert cleaned == 2
