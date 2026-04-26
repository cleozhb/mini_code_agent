"""层级 2：相关单元测试 —— 找到改动文件对应的测试，跑它们."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path

from .types import IncrementalVerificationResult, VerificationCheck

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
SHARED_MODULE_TIMEOUT = 60


# 公共模块判断：src/<pkg>/utils.py / common.py / config.py / __init__.py 等
_SHARED_MODULE_NAMES = {"utils", "common", "config", "constants", "__init__", "base"}


class UnitTestVerifier:
    """层级 2：跑相关单元测试."""

    def __init__(
        self,
        default_timeout: int = DEFAULT_TIMEOUT,
        shared_module_timeout: int = SHARED_MODULE_TIMEOUT,
    ) -> None:
        self._default_timeout = default_timeout
        self._shared_module_timeout = shared_module_timeout

    async def verify(
        self,
        files_changed: list[str],
        project_path: str,
        task_id: str = "",
    ) -> IncrementalVerificationResult:
        start = time.monotonic()

        related, hit_shared = self._find_related_tests(
            files_changed, project_path
        )

        if not related:
            check = VerificationCheck(
                check_name="unit_test",
                passed=True,
                skipped=True,
                skip_reason="no related tests found",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=0,
                items_failed=0,
            )
            return IncrementalVerificationResult(
                task_id=task_id,
                level=2,
                checks=[check],
                overall_passed=True,
                files_verified=list(files_changed),
                total_duration_seconds=time.monotonic() - start,
            )

        framework = self._detect_framework(project_path)
        if framework is None:
            check = VerificationCheck(
                check_name="unit_test",
                passed=True,
                skipped=True,
                skip_reason="no test framework detected",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=0,
                items_failed=0,
            )
            return IncrementalVerificationResult(
                task_id=task_id,
                level=2,
                checks=[check],
                overall_passed=True,
                files_verified=list(files_changed),
                total_duration_seconds=time.monotonic() - start,
            )

        timeout = (
            self._shared_module_timeout if hit_shared else self._default_timeout
        )

        check = await self._run_tests(related, project_path, framework, timeout)

        duration = time.monotonic() - start
        overall = check.passed or check.skipped

        return IncrementalVerificationResult(
            task_id=task_id,
            level=2,
            checks=[check],
            overall_passed=overall,
            files_verified=list(files_changed),
            total_duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # 测试定位
    # ------------------------------------------------------------------

    def _find_related_tests(
        self, files_changed: list[str], project_path: str
    ) -> tuple[list[str], bool]:
        """
        Returns (test_paths, hit_shared_module).

        规则按优先级:
          (a) 修改的本身就是 tests/ 文件 → 直接跑
          (b) src/foo/bar.py → tests/test_bar.py / tests/foo/test_bar.py /
              src/foo/test_bar.py / tests/test_foo.py
          (c) 公共模块 → 跑 tests/ 整个目录
        """
        proj = Path(project_path).resolve()
        results: list[str] = []
        seen: set[str] = set()
        hit_shared = False

        def _add(p: Path) -> None:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                results.append(sp)

        for f in files_changed:
            fp = Path(f)
            if not fp.is_absolute():
                fp = proj / fp
            try:
                rel = fp.relative_to(proj)
            except ValueError:
                rel = fp

            parts = rel.parts
            stem = fp.stem

            # (a) 测试文件本身
            if "tests" in parts or stem.startswith("test_") or stem.endswith("_test"):
                if fp.exists():
                    _add(fp)
                continue

            if fp.suffix.lower() not in (".py",):
                continue

            # (c) 公共模块判定 → 跑 tests/ 整个目录
            if stem in _SHARED_MODULE_NAMES:
                hit_shared = True
                tests_dir = proj / "tests"
                if tests_dir.exists() and tests_dir.is_dir():
                    _add(tests_dir)
                    continue
                # 没有 tests/，也跳过

            # (b) 候选位置 — 第一个存在的就用
            candidates = [
                proj / "tests" / f"test_{stem}.py",
                fp.parent / f"test_{stem}.py",
            ]
            # tests/<sub>/test_<stem>.py — 沿父目录拼
            # 找出 src/ 之后的 sub-path
            try:
                if "src" in parts:
                    idx = parts.index("src")
                    sub = parts[idx + 1 : -1]  # 去掉 src 和文件名
                else:
                    sub = parts[:-1]
            except ValueError:
                sub = ()

            if sub:
                candidates.append(proj / "tests" / Path(*sub) / f"test_{stem}.py")
                # 模块级测试
                top = sub[0]
                candidates.append(proj / "tests" / f"test_{top}.py")

            for c in candidates:
                if c.exists():
                    _add(c)
                    break

        return results, hit_shared

    # ------------------------------------------------------------------
    # 框架检测
    # ------------------------------------------------------------------

    def _detect_framework(self, project_path: str) -> str | None:
        """返回 'pytest' / 'jest' / 'npm' / None."""
        proj = Path(project_path)
        if (proj / "pyproject.toml").exists() or (proj / "pytest.ini").exists():
            return "pytest"
        # 退一步：有 tests/ 目录 + .py 项目
        if (proj / "tests").exists() and any(proj.glob("*.py")):
            return "pytest"
        if (proj / "package.json").exists():
            try:
                txt = (proj / "package.json").read_text(errors="replace")
                if '"jest"' in txt:
                    return "jest"
                return "npm"
            except OSError:
                return "npm"
        return None

    # ------------------------------------------------------------------
    # 跑测试
    # ------------------------------------------------------------------

    async def _run_tests(
        self,
        test_paths: list[str],
        project_path: str,
        framework: str,
        timeout: int,
    ) -> VerificationCheck:
        start = time.monotonic()

        if framework == "pytest":
            cmd = [
                "uv", "run", "pytest", *test_paths,
                "-x", "--tb=short", "--no-header", "-q",
            ]
        elif framework == "jest":
            cmd = ["npx", "jest", *test_paths]
        elif framework == "npm":
            cmd = ["npm", "test", "--", *test_paths]
        else:
            return VerificationCheck(
                check_name="unit_test",
                passed=True,
                skipped=True,
                skip_reason=f"unsupported framework: {framework}",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=0,
                items_failed=0,
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return VerificationCheck(
                check_name="unit_test",
                passed=True,
                skipped=True,
                skip_reason=f"runner not found: {e}",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=0,
                items_failed=0,
            )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return VerificationCheck(
                check_name="unit_test",
                passed=False,
                skipped=False,
                skip_reason=None,
                duration_seconds=time.monotonic() - start,
                details=f"tests timed out after {timeout} seconds",
                items_checked=0,
                items_failed=0,
            )

        stdout = (stdout_b or b"").decode(errors="replace")
        stderr = (stderr_b or b"").decode(errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        rc = proc.returncode or 0

        passed_n, failed_n, total_n, failed_summary = self._parse_output(
            combined, framework
        )

        if rc == 0 and failed_n == 0:
            return VerificationCheck(
                check_name="unit_test",
                passed=True,
                skipped=False,
                skip_reason=None,
                duration_seconds=time.monotonic() - start,
                details=f"{passed_n} passed",
                items_checked=total_n or passed_n,
                items_failed=0,
            )

        # collection error: rc != 0 但解析不到 failed
        details = (
            f"{failed_n} tests failed:\n{failed_summary}"
            if failed_n > 0
            else self._truncate_output(combined)
        )
        return VerificationCheck(
            check_name="unit_test",
            passed=False,
            skipped=False,
            skip_reason=None,
            duration_seconds=time.monotonic() - start,
            details=details,
            items_checked=total_n or (passed_n + failed_n),
            items_failed=failed_n if failed_n > 0 else 1,
        )

    @staticmethod
    def _parse_output(
        text: str, framework: str
    ) -> tuple[int, int, int, str]:
        """提取 passed/failed/total + failed-test 摘要."""
        passed_n = 0
        failed_n = 0
        total_n = 0
        summary_lines: list[str] = []

        if framework == "pytest":
            # ===== 5 passed, 2 failed in 1.23s =====
            tail_match = re.search(
                r"(?:=+\s*)?(?:(\d+)\s+failed)?[,\s]*(?:(\d+)\s+passed)?"
                r"[,\s]*(?:(\d+)\s+error)?",
                text,
            )
            # 简单逐项搜
            m_passed = re.search(r"(\d+)\s+passed", text)
            m_failed = re.search(r"(\d+)\s+failed", text)
            m_error = re.search(r"(\d+)\s+error", text)
            if m_passed:
                passed_n = int(m_passed.group(1))
            if m_failed:
                failed_n = int(m_failed.group(1))
            if m_error:
                failed_n += int(m_error.group(1))
            total_n = passed_n + failed_n

            # FAILED tests/test_x.py::test_foo - AssertionError: ...
            for line in text.splitlines():
                if line.startswith("FAILED ") or line.startswith("ERROR "):
                    summary_lines.append(line.strip()[:200])

        elif framework in ("jest", "npm"):
            m = re.search(
                r"Tests:\s*(?:(\d+)\s+failed,\s*)?(?:(\d+)\s+passed,\s*)?"
                r"(\d+)\s+total",
                text,
            )
            if m:
                failed_n = int(m.group(1) or 0)
                passed_n = int(m.group(2) or 0)
                total_n = int(m.group(3))
            for line in text.splitlines():
                if line.strip().startswith("✕") or "FAIL " in line:
                    summary_lines.append(line.strip()[:200])

        return passed_n, failed_n, total_n, "\n".join(summary_lines[:5])

    @staticmethod
    def _truncate_output(text: str, max_lines: int = 20) -> str:
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
