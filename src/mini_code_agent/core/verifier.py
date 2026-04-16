"""任务验证器 — 对 Agent 修改过的代码自动跑语法/lint/测试检查."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """验证结果."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def format_for_llm(self) -> str:
        """格式化成适合回传给 LLM 的文本."""
        if self.passed:
            return "验证通过：代码改动未发现问题。"
        lines = ["验证失败，发现以下问题："]
        for i, err in enumerate(self.errors, 1):
            lines.append(f"{i}. {err}")
        if self.suggestions:
            lines.append("\n建议：")
            for s in self.suggestions:
                lines.append(f"- {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


# 命令执行超时（秒）
DEFAULT_CMD_TIMEOUT = 60


class Verifier:
    """对修改过的代码文件运行多层验证.

    - 语法检查：python 用 py_compile，js/ts 用 node --check
    - Lint：如果检测到项目配置则执行
    - 测试：尝试定位并执行与改动文件相关的测试
    """

    PY_SUFFIXES = {".py"}
    JS_SUFFIXES = {".js", ".mjs", ".cjs"}
    TS_SUFFIXES = {".ts", ".tsx"}

    def __init__(self, cmd_timeout: int = DEFAULT_CMD_TIMEOUT) -> None:
        self.cmd_timeout = cmd_timeout

    # -------------------------- 公共 API --------------------------

    async def verify_code_change(
        self,
        files_changed: list[str] | list[Path],
        project_path: str | Path,
    ) -> VerificationResult:
        """对改动过的文件运行语法 / lint / 测试验证."""
        project_root = Path(project_path).resolve()
        paths = [self._resolve_path(f, project_root) for f in files_changed]
        # 只保留存在的文件
        paths = [p for p in paths if p.exists() and p.is_file()]

        errors: list[str] = []
        suggestions: list[str] = []

        if not paths:
            return VerificationResult(passed=True, errors=[], suggestions=[])

        # 1. 语法检查
        syntax_errors = await self._check_syntax(paths)
        errors.extend(syntax_errors)

        # 语法错误直接返回，不再跑 lint/test
        if syntax_errors:
            suggestions.append("先修复语法错误再继续。")
            return VerificationResult(
                passed=False, errors=errors, suggestions=suggestions
            )

        # 2. Lint
        lint_errors = await self._run_lint(paths, project_root)
        errors.extend(lint_errors)

        # 3. 相关测试
        test_errors = await self._run_related_tests(paths, project_root)
        errors.extend(test_errors)

        if errors:
            if lint_errors:
                suggestions.append("阅读 lint 输出，修正代码风格或潜在问题。")
            if test_errors:
                suggestions.append(
                    "阅读测试失败信息，定位到断言或异常所在的逻辑进行修复。"
                )

        return VerificationResult(
            passed=not errors, errors=errors, suggestions=suggestions
        )

    # -------------------------- 语法检查 --------------------------

    async def _check_syntax(self, files: list[Path]) -> list[str]:
        errors: list[str] = []
        for f in files:
            suffix = f.suffix.lower()
            if suffix in self.PY_SUFFIXES:
                msg = await self._py_compile(f)
                if msg:
                    errors.append(msg)
            elif suffix in self.JS_SUFFIXES or suffix in self.TS_SUFFIXES:
                msg = await self._node_check(f, is_ts=suffix in self.TS_SUFFIXES)
                if msg:
                    errors.append(msg)
            # 其他文件类型不做语法检查
        return errors

    async def _py_compile(self, file: Path) -> str | None:
        """用 py_compile 检查 Python 语法."""
        code = (
            "import py_compile, sys\n"
            "try:\n"
            f"    py_compile.compile({str(file)!r}, doraise=True)\n"
            "except py_compile.PyCompileError as e:\n"
            "    sys.stderr.write(str(e))\n"
            "    sys.exit(1)\n"
        )
        rc, _out, err = await self._run_cmd(["python", "-c", code])
        if rc != 0:
            return f"[语法错误] {file.name}: {err.strip() or '未知语法错误'}"
        return None

    async def _node_check(self, file: Path, is_ts: bool) -> str | None:
        """用 node --check 检查 JS 语法；TS 需要 tsc 才能精准检查，这里做尽力检查."""
        if is_ts and shutil.which("tsc"):
            rc, out, err = await self._run_cmd(
                ["tsc", "--noEmit", "--allowJs", str(file)]
            )
            if rc != 0:
                msg = (err or out).strip() or "TypeScript 类型/语法错误"
                return f"[TS 检查失败] {file.name}: {msg}"
            return None

        if not shutil.which("node"):
            return None  # 没有 node 环境，跳过

        rc, _out, err = await self._run_cmd(["node", "--check", str(file)])
        if rc != 0:
            return f"[语法错误] {file.name}: {err.strip() or '未知语法错误'}"
        return None

    # -------------------------- Lint --------------------------

    async def _run_lint(
        self, files: list[Path], project_root: Path
    ) -> list[str]:
        """如果项目配置了 lint，则对改动文件执行 lint."""
        errors: list[str] = []
        py_files = [f for f in files if f.suffix.lower() in self.PY_SUFFIXES]
        js_ts_files = [
            f for f in files
            if f.suffix.lower() in (self.JS_SUFFIXES | self.TS_SUFFIXES)
        ]

        # Python：优先 ruff，其次 flake8
        if py_files:
            if self._has_ruff_config(project_root) and shutil.which("ruff"):
                rc, out, err = await self._run_cmd(
                    ["ruff", "check", *[str(f) for f in py_files]],
                    cwd=project_root,
                )
                if rc != 0:
                    msg = (out or err).strip()
                    if msg:
                        errors.append(f"[ruff] {msg}")
            elif self._has_flake8_config(project_root) and shutil.which("flake8"):
                rc, out, err = await self._run_cmd(
                    ["flake8", *[str(f) for f in py_files]],
                    cwd=project_root,
                )
                if rc != 0:
                    msg = (out or err).strip()
                    if msg:
                        errors.append(f"[flake8] {msg}")

        # JS/TS：如果有 eslint 配置且装了 eslint，就跑
        if js_ts_files and self._has_eslint_config(project_root):
            eslint_bin = self._find_local_bin(project_root, "eslint") or shutil.which(
                "eslint"
            )
            if eslint_bin:
                rc, out, err = await self._run_cmd(
                    [eslint_bin, *[str(f) for f in js_ts_files]],
                    cwd=project_root,
                )
                if rc != 0:
                    msg = (out or err).strip()
                    if msg:
                        errors.append(f"[eslint] {msg}")

        return errors

    def _has_ruff_config(self, root: Path) -> bool:
        if (root / "ruff.toml").exists() or (root / ".ruff.toml").exists():
            return True
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                text = pyproject.read_text(encoding="utf-8", errors="ignore")
                if "[tool.ruff" in text:
                    return True
            except OSError:
                pass
        return False

    def _has_flake8_config(self, root: Path) -> bool:
        for name in (".flake8", "setup.cfg", "tox.ini"):
            if (root / name).exists():
                return True
        return False

    def _has_eslint_config(self, root: Path) -> bool:
        names = (
            ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
            ".eslintrc.yaml", ".eslintrc.yml", "eslint.config.js",
            "eslint.config.mjs", "eslint.config.cjs",
        )
        return any((root / n).exists() for n in names)

    def _find_local_bin(self, root: Path, name: str) -> str | None:
        candidate = root / "node_modules" / ".bin" / name
        if candidate.exists():
            return str(candidate)
        return None

    # -------------------------- 测试 --------------------------

    async def _run_related_tests(
        self, files: list[Path], project_root: Path
    ) -> list[str]:
        """定位并运行与改动文件相关的测试."""
        py_files = [f for f in files if f.suffix.lower() in self.PY_SUFFIXES]
        if not py_files:
            return []

        # 如果用户改的就是测试文件本身，直接跑这些文件
        direct_tests = [f for f in py_files if self._looks_like_test(f)]

        # 对非测试文件，尝试找对应的 test_*.py
        related: list[Path] = list(direct_tests)
        for f in py_files:
            if self._looks_like_test(f):
                continue
            related.extend(self._find_tests_for(f, project_root))

        # 去重
        seen: set[Path] = set()
        unique: list[Path] = []
        for p in related:
            rp = p.resolve()
            if rp not in seen and rp.exists():
                seen.add(rp)
                unique.append(rp)

        if not unique:
            return []

        # 检测 pytest 是否可用；优先 `python -m pytest`
        pytest_cmd = await self._resolve_pytest_cmd(project_root)
        if pytest_cmd is None:
            logger.info("未发现 pytest，跳过测试运行")
            return []

        rc, out, err = await self._run_cmd(
            [*pytest_cmd, "-x", "--tb=short", *[str(p) for p in unique]],
            cwd=project_root,
        )
        if rc != 0:
            msg = (out or err).strip()
            # 截断避免太长
            if len(msg) > 4000:
                msg = msg[:4000] + "\n...(输出已截断)"
            return [f"[pytest] 测试失败:\n{msg}"]
        return []

    def _looks_like_test(self, f: Path) -> bool:
        name = f.name
        return name.startswith("test_") or name.endswith("_test.py")

    def _find_tests_for(self, src: Path, project_root: Path) -> list[Path]:
        """按常见约定查找与 src 对应的测试文件."""
        candidates: list[Path] = []
        stem = src.stem  # e.g. 'verifier'
        target_names = {f"test_{stem}.py", f"{stem}_test.py"}

        # 优先 project_root/tests/
        tests_dir = project_root / "tests"
        if tests_dir.exists():
            for p in tests_dir.rglob("*.py"):
                if p.name in target_names:
                    candidates.append(p)

        # 源文件同级目录
        for p in src.parent.glob("*.py"):
            if p.name in target_names:
                candidates.append(p)

        return candidates

    async def _resolve_pytest_cmd(self, project_root: Path) -> list[str] | None:
        """决定如何调用 pytest."""
        # 优先 python -m pytest（兼容 venv 内的 pytest）
        rc, _out, _err = await self._run_cmd(
            ["python", "-c", "import pytest"], cwd=project_root
        )
        if rc == 0:
            return ["python", "-m", "pytest"]

        if shutil.which("pytest"):
            return ["pytest"]

        return None

    # -------------------------- 命令执行 --------------------------

    async def _run_cmd(
        self,
        cmd: list[str],
        cwd: Path | None = None,
    ) -> tuple[int, str, str]:
        """执行外部命令，返回 (returncode, stdout, stderr)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
            )
        except FileNotFoundError as e:
            return 127, "", f"命令未找到: {e}"
        except OSError as e:
            return 1, "", f"执行命令失败: {e}"

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self.cmd_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"命令超时（>{self.cmd_timeout}s）: {' '.join(cmd)}"

        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        return proc.returncode or 0, stdout, stderr

    # -------------------------- 工具 --------------------------

    @staticmethod
    def _resolve_path(f: str | Path, project_root: Path) -> Path:
        p = Path(f)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        return p
