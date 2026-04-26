"""Incremental Verifier 测试 — 层级 1（QuickVerifier）+ 层级 2（UnitTestVerifier）.

运行:
    uv run pytest tests/test_incremental_verifier.py -xvs
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mini_code_agent.verify import (
    IncrementalVerifier,
    QuickVerifier,
    UnitTestVerifier,
)


# ---------------------------------------------------------------------------
# 辅助：建临时项目骨架
# ---------------------------------------------------------------------------


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip("\n"))


def _make_project(tmp_path: Path) -> Path:
    _write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "demo"
        version = "0.0.1"
        dependencies = ["pytest>=7.0"]
        """,
    )
    return tmp_path


# ===========================================================================
# 层级 1：QuickVerifier
# ===========================================================================


@pytest.mark.asyncio
async def test_quick_syntax_error_detected(tmp_path: Path):
    proj = _make_project(tmp_path)
    bad = proj / "src" / "demo" / "bad.py"
    _write(bad, "def foo(:\n    pass\n")

    verifier = QuickVerifier()
    result = await verifier.verify(["src/demo/bad.py"], str(proj))

    assert not result.overall_passed
    syntax = result.get_check("syntax")
    assert syntax is not None
    assert not syntax.passed
    assert syntax.items_failed == 1
    assert "SyntaxError" in syntax.details


@pytest.mark.asyncio
async def test_quick_unresolved_import_detected(tmp_path: Path):
    proj = _make_project(tmp_path)
    f = proj / "src" / "demo" / "ok.py"
    _write(f, "import this_does_not_exist_xyz\n")

    verifier = QuickVerifier()
    result = await verifier.verify(["src/demo/ok.py"], str(proj))

    imp = result.get_check("import")
    assert imp is not None
    assert not imp.passed
    assert "this_does_not_exist_xyz" in imp.details


@pytest.mark.asyncio
async def test_quick_clean_file_passes(tmp_path: Path):
    proj = _make_project(tmp_path)
    f = proj / "src" / "demo" / "ok.py"
    _write(f, "import os\nimport sys\n\ndef hello() -> str:\n    return 'hi'\n")

    verifier = QuickVerifier()
    result = await verifier.verify(["src/demo/ok.py"], str(proj))

    assert result.overall_passed
    syntax = result.get_check("syntax")
    assert syntax.passed
    assert syntax.items_checked == 1


@pytest.mark.asyncio
async def test_quick_mixed_files(tmp_path: Path):
    proj = _make_project(tmp_path)
    good = proj / "src" / "good.py"
    bad = proj / "src" / "bad.py"
    _write(good, "x = 1\n")
    _write(bad, "def(:\n")

    verifier = QuickVerifier()
    result = await verifier.verify(
        ["src/good.py", "src/bad.py"], str(proj)
    )

    assert not result.overall_passed
    syntax = result.get_check("syntax")
    assert syntax.items_checked == 2
    assert syntax.items_failed == 1


@pytest.mark.asyncio
async def test_quick_overall_timeout(tmp_path: Path):
    """整体超时 → 返回 partial result，不抛异常."""
    proj = _make_project(tmp_path)
    f = proj / "src" / "demo" / "ok.py"
    _write(f, "x = 1\n")

    verifier = QuickVerifier(overall_timeout=0.001)

    # 通过 monkeypatch 让 _run_all_checks 卡住
    async def hang(*a, **kw):
        import asyncio
        await asyncio.sleep(10)

    with patch.object(QuickVerifier, "_run_all_checks", hang):
        result = await verifier.verify(["src/demo/ok.py"], str(proj))

    assert not result.overall_passed
    assert "timeout" in result.checks[0].details.lower()


@pytest.mark.asyncio
async def test_quick_skips_non_python(tmp_path: Path):
    proj = _make_project(tmp_path)
    f = proj / "README.md"
    _write(f, "# hello\n")

    verifier = QuickVerifier()
    result = await verifier.verify(["README.md"], str(proj))

    syntax = result.get_check("syntax")
    assert syntax.skipped
    imp = result.get_check("import")
    assert imp.skipped


# ===========================================================================
# 层级 2：UnitTestVerifier — 通过 mock subprocess 验证逻辑
# ===========================================================================


def _make_proj_with_tests(tmp_path: Path) -> Path:
    proj = _make_project(tmp_path)
    _write(
        proj / "src" / "foo.py",
        "def add(a, b):\n    return a + b\n",
    )
    _write(
        proj / "tests" / "test_foo.py",
        """
        from src.foo import add

        def test_add():
            assert add(1, 2) == 3
        """,
    )
    return proj


def _mock_subprocess(stdout: bytes, stderr: bytes = b"", returncode: int = 0):
    """构造一个 fake create_subprocess_exec."""
    class FakeProc:
        def __init__(self):
            self.returncode = returncode
        async def communicate(self):
            return stdout, stderr
        async def wait(self):
            return None
        def kill(self):
            pass

    async def fake_exec(*args, **kwargs):
        return FakeProc()

    return fake_exec


@pytest.mark.asyncio
async def test_l2_finds_corresponding_test_and_passes(tmp_path: Path):
    proj = _make_proj_with_tests(tmp_path)

    verifier = UnitTestVerifier()
    fake = _mock_subprocess(b"===== 1 passed in 0.05s =====\n", returncode=0)
    with patch("asyncio.create_subprocess_exec", fake):
        result = await verifier.verify(["src/foo.py"], str(proj))

    assert result.overall_passed
    check = result.get_check("unit_test")
    assert check.passed
    assert check.items_checked >= 1


@pytest.mark.asyncio
async def test_l2_no_related_tests_skipped(tmp_path: Path):
    proj = _make_project(tmp_path)
    _write(proj / "src" / "lonely.py", "x = 1\n")

    verifier = UnitTestVerifier()
    result = await verifier.verify(["src/lonely.py"], str(proj))

    check = result.get_check("unit_test")
    assert check.skipped
    assert result.overall_passed  # skipped 不算失败


@pytest.mark.asyncio
async def test_l2_shared_module_runs_full_tests_with_long_timeout(tmp_path: Path):
    proj = _make_project(tmp_path)
    _write(proj / "src" / "utils.py", "x = 1\n")
    _write(proj / "tests" / "test_a.py", "def test_a(): pass\n")

    verifier = UnitTestVerifier(
        default_timeout=30, shared_module_timeout=60,
    )

    captured_timeout: dict[str, float | None] = {"t": None}

    fake_exec = _mock_subprocess(b"===== 1 passed in 0.05s =====\n")

    async def proxy_wait_for(coro, timeout):
        captured_timeout["t"] = timeout
        return await coro

    with patch("asyncio.create_subprocess_exec", fake_exec):
        with patch("asyncio.wait_for", proxy_wait_for):
            result = await verifier.verify(["src/utils.py"], str(proj))

    assert result.overall_passed
    # shared module 应使用 60s
    assert captured_timeout["t"] == 60


@pytest.mark.asyncio
async def test_l2_test_failure_reported(tmp_path: Path):
    proj = _make_proj_with_tests(tmp_path)

    out = (
        b"FAILED tests/test_foo.py::test_add - AssertionError: 1 != 2\n"
        b"===== 1 failed in 0.05s =====\n"
    )
    verifier = UnitTestVerifier()
    fake = _mock_subprocess(out, returncode=1)
    with patch("asyncio.create_subprocess_exec", fake):
        result = await verifier.verify(["src/foo.py"], str(proj))

    check = result.get_check("unit_test")
    assert not check.passed
    assert check.items_failed == 1
    assert "FAILED" in check.details


@pytest.mark.asyncio
async def test_l2_test_timeout(tmp_path: Path):
    proj = _make_proj_with_tests(tmp_path)
    verifier = UnitTestVerifier(default_timeout=0)

    class HangProc:
        returncode = None
        async def communicate(self):
            import asyncio
            await asyncio.sleep(100)
            return b"", b""
        async def wait(self):
            return None
        def kill(self):
            pass

    async def fake_exec(*a, **kw):
        return HangProc()

    with patch("asyncio.create_subprocess_exec", fake_exec):
        result = await verifier.verify(["src/foo.py"], str(proj))

    check = result.get_check("unit_test")
    assert not check.passed
    assert "timed out" in check.details.lower()


@pytest.mark.asyncio
async def test_l2_collection_error(tmp_path: Path):
    proj = _make_proj_with_tests(tmp_path)
    out = (
        b"ERROR tests/test_foo.py - ImportError: cannot import name 'add'\n"
    )
    verifier = UnitTestVerifier()
    fake = _mock_subprocess(out, returncode=2)
    with patch("asyncio.create_subprocess_exec", fake):
        result = await verifier.verify(["src/foo.py"], str(proj))

    check = result.get_check("unit_test")
    assert not check.passed


# ===========================================================================
# 整合：IncrementalVerifier
# ===========================================================================


@pytest.mark.asyncio
async def test_verify_after_subtask_l1_fail_skips_l2(tmp_path: Path):
    proj = _make_project(tmp_path)
    bad = proj / "src" / "bad.py"
    _write(bad, "def(:\n")

    l2_called = {"v": False}

    class StubL2(UnitTestVerifier):
        async def verify(self, *a, **kw):
            l2_called["v"] = True
            return await super().verify(*a, **kw)

    verifier = IncrementalVerifier(level2=StubL2())

    # 构造一个迷你 artifact stub
    class _Edit:
        def __init__(self, p): self.path = p
    class _Patch:
        def __init__(self, edits): self.edits = edits
    class _Art:
        task_id = "t1"
        patch = _Patch([_Edit("src/bad.py")])

    result = await verifier.verify_after_subtask(_Art(), str(proj))
    assert not result.overall_passed
    assert l2_called["v"] is False  # L1 失败应跳过 L2
    assert result.level == 1


@pytest.mark.asyncio
async def test_verify_after_edit_returns_l1(tmp_path: Path):
    proj = _make_project(tmp_path)
    f = proj / "src" / "ok.py"
    _write(f, "import os\nx = 1\n")

    verifier = IncrementalVerifier()
    result = await verifier.verify_after_edit(["src/ok.py"], str(proj))

    assert result.level == 1
    assert result.overall_passed


@pytest.mark.asyncio
async def test_verify_after_edit_empty_files(tmp_path: Path):
    verifier = IncrementalVerifier()
    result = await verifier.verify_after_edit([], str(tmp_path))
    assert result.overall_passed
    assert result.checks == []


# ===========================================================================
# Agent 集成：层级 1 警告注入到 tool_result
# ===========================================================================


@pytest.mark.asyncio
async def test_agent_quick_verify_injects_warning(tmp_path: Path):
    """write_file 之后，QuickVerifier 检测到错误，warning 应附加到 tool 输出."""
    from mini_code_agent.core.agent import Agent
    from mini_code_agent.llm import LLMClient, LLMResponse, TokenUsage, ToolParam
    from mini_code_agent.tools import ToolRegistry, WriteFileTool

    proj = _make_project(tmp_path)

    # 极简 LLM stub
    class StubLLM(LLMClient):
        def __init__(self):
            super().__init__(model="stub")
        async def chat(self, messages, tools=None, **_):
            return LLMResponse(content="ok", usage=TokenUsage(), tool_calls=[])
        def chat_stream(self, messages, tools=None):
            raise NotImplementedError

    registry = ToolRegistry()
    registry.register(WriteFileTool())

    agent = Agent(
        llm_client=StubLLM(),
        tool_registry=registry,
        system_prompt="you are a test agent",
        project_path=str(proj),
        incremental_verifier=IncrementalVerifier(),
    )

    from mini_code_agent.llm.base import ToolCall
    bad_path = str(proj / "src" / "bad.py")
    tc = ToolCall(
        id="c1",
        name="WriteFile",
        arguments={"path": bad_path, "content": "def(:\n"},
    )

    msg = await agent._execute_tool_call(tc)
    assert msg.tool_result is not None
    assert "VERIFICATION WARNING" in (msg.tool_result.content or "")
