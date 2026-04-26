"""层级 1：快速检查 —— 只对修改文件做语法 + import + LSP 诊断."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from .types import IncrementalVerificationResult, VerificationCheck

logger = logging.getLogger(__name__)


_PY_EXTS = {".py", ".pyi"}
_NODE_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS = {".ts", ".tsx"}

_SUBPROCESS_TIMEOUT = 2.0  # 单次子进程超时
_OVERALL_TIMEOUT = 5.0     # 整体 timeout
_LSP_TIMEOUT = 3.0


class QuickVerifier:
    """层级 1：5 秒内完成的快速检查."""

    def __init__(
        self,
        lsp_manager: Any | None = None,
        check_node: bool = False,
        overall_timeout: float = _OVERALL_TIMEOUT,
    ) -> None:
        self._lsp_manager = lsp_manager
        self._check_node = check_node
        self._overall_timeout = overall_timeout

    async def verify(
        self,
        files_changed: list[str],
        project_path: str,
        task_id: str = "",
    ) -> IncrementalVerificationResult:
        """对 files_changed 跑快速检查；总超时 self._overall_timeout."""
        start = time.monotonic()

        try:
            checks = await asyncio.wait_for(
                self._run_all_checks(files_changed, project_path),
                timeout=self._overall_timeout,
            )
        except asyncio.TimeoutError:
            checks = [
                VerificationCheck(
                    check_name="syntax",
                    passed=False,
                    skipped=False,
                    skip_reason=None,
                    duration_seconds=self._overall_timeout,
                    details=(
                        f"Level1 timeout after {self._overall_timeout:.1f}s "
                        f"(partial result)"
                    ),
                    items_checked=len(files_changed),
                    items_failed=0,
                )
            ]

        duration = time.monotonic() - start
        overall_passed = all(c.passed or c.skipped for c in checks) if checks else True

        return IncrementalVerificationResult(
            task_id=task_id,
            level=1,
            checks=checks,
            overall_passed=overall_passed,
            files_verified=list(files_changed),
            total_duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # 内部检查
    # ------------------------------------------------------------------

    async def _run_all_checks(
        self, files_changed: list[str], project_path: str
    ) -> list[VerificationCheck]:
        """三类检查并发跑."""
        py_files = [
            f for f in files_changed
            if Path(f).suffix.lower() in _PY_EXTS
        ]
        node_files = [
            f for f in files_changed
            if Path(f).suffix.lower() in (_NODE_EXTS | _TS_EXTS)
        ]

        tasks: list[asyncio.Task[VerificationCheck]] = []
        tasks.append(asyncio.create_task(
            self._syntax_check(py_files, node_files, project_path)
        ))
        tasks.append(asyncio.create_task(
            self._import_check(py_files, project_path)
        ))
        tasks.append(asyncio.create_task(
            self._lsp_check(files_changed, project_path)
        ))

        return list(await asyncio.gather(*tasks))

    # ---- syntax check -------------------------------------------------

    async def _syntax_check(
        self,
        py_files: list[str],
        node_files: list[str],
        project_path: str,
    ) -> VerificationCheck:
        """Python 用 ast.parse；JS/TS 可选用 node --check."""
        start = time.monotonic()
        items_checked = 0
        items_failed = 0
        errors: list[str] = []

        for f in py_files:
            items_checked += 1
            ok, err = self._check_python_syntax(f, project_path)
            if not ok:
                items_failed += 1
                errors.append(f"{f}: {err}")

        if self._check_node:
            for f in node_files:
                items_checked += 1
                ok, err = await self._check_node_syntax(f, project_path)
                if not ok:
                    items_failed += 1
                    errors.append(f"{f}: {err}")

        passed = items_failed == 0
        details = "ok" if passed else "\n".join(errors[:5])
        return VerificationCheck(
            check_name="syntax",
            passed=passed,
            skipped=items_checked == 0,
            skip_reason="no syntax-checkable files" if items_checked == 0 else None,
            duration_seconds=time.monotonic() - start,
            details=details,
            items_checked=items_checked,
            items_failed=items_failed,
        )

    @staticmethod
    def _check_python_syntax(
        path: str, project_path: str
    ) -> tuple[bool, str | None]:
        """用 ast.parse 解析 — 比 subprocess 快得多."""
        full = path if os.path.isabs(path) else os.path.join(project_path, path)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except FileNotFoundError:
            return False, "file not found"
        except OSError as e:
            return False, f"read error: {e}"
        try:
            ast.parse(source, filename=full)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError L{e.lineno}: {e.msg}"

    async def _check_node_syntax(
        self, path: str, project_path: str
    ) -> tuple[bool, str | None]:
        """用 node --check 检查 JS 语法（TS 跳过 — 太慢）."""
        ext = Path(path).suffix.lower()
        if ext in _TS_EXTS:
            # TS 太慢；不在 Level 1 范围
            return True, None

        full = path if os.path.isabs(path) else os.path.join(project_path, path)
        try:
            proc = await asyncio.create_subprocess_exec(
                "node", "--check", full,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=_SUBPROCESS_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"node --check timeout"
            if proc.returncode == 0:
                return True, None
            err = (stderr or b"").decode(errors="replace").strip().splitlines()
            return False, err[0] if err else "syntax error"
        except FileNotFoundError:
            return True, None  # 没装 node，跳过

    # ---- import check -------------------------------------------------

    async def _import_check(
        self,
        py_files: list[str],
        project_path: str,
    ) -> VerificationCheck:
        """检查 Python import 能不能 resolve（不实际 import）."""
        start = time.monotonic()
        items_checked = 0
        items_failed = 0
        errors: list[str] = []

        declared = self._read_declared_packages(project_path)

        for f in py_files:
            items_checked += 1
            full = f if os.path.isabs(f) else os.path.join(project_path, f)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fp:
                    source = fp.read()
            except OSError:
                items_failed += 1
                errors.append(f"{f}: cannot read")
                continue

            try:
                tree = ast.parse(source)
            except SyntaxError:
                # 语法错误归 syntax 检查；这里跳过
                continue

            bad = self._collect_unresolved_imports(
                tree, full, project_path, declared
            )
            if bad:
                items_failed += 1
                errors.append(f"{f}: " + "; ".join(bad[:3]))

        passed = items_failed == 0
        details = "ok" if passed else "\n".join(errors[:5])
        return VerificationCheck(
            check_name="import",
            passed=passed,
            skipped=items_checked == 0,
            skip_reason="no python files" if items_checked == 0 else None,
            duration_seconds=time.monotonic() - start,
            details=details,
            items_checked=items_checked,
            items_failed=items_failed,
        )

    @staticmethod
    def _read_declared_packages(project_path: str) -> set[str]:
        """从 pyproject.toml / requirements.txt 读出依赖包名（顶层名）."""
        declared: set[str] = set()
        # pyproject.toml — 简单字符串扫描，避免引入 toml 依赖
        py = Path(project_path) / "pyproject.toml"
        if py.exists():
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
                # 搜 dependencies = [ "foo>=...", ... ] / [project] dependencies
                import re
                for m in re.finditer(
                    r'"([a-zA-Z0-9_\-\.\[\]]+)(?:\s*[<>=!~]|\s*$)',
                    text,
                ):
                    name = m.group(1).split("[")[0].strip().lower()
                    if name:
                        declared.add(name.replace("-", "_"))
            except OSError:
                pass
        # requirements.txt
        req = Path(project_path) / "requirements.txt"
        if req.exists():
            try:
                for line in req.read_text(errors="replace").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    pkg = line.split("==")[0].split(">=")[0].split("<=")[0]
                    pkg = pkg.split("[")[0].strip().lower()
                    if pkg:
                        declared.add(pkg.replace("-", "_"))
            except OSError:
                pass
        return declared

    @staticmethod
    def _collect_unresolved_imports(
        tree: ast.AST,
        file_full_path: str,
        project_path: str,
        declared: set[str],
    ) -> list[str]:
        """返回每个解析不到的 import 描述."""
        import sys

        unresolved: list[str] = []
        proj_path = Path(project_path).resolve()
        file_dir = Path(file_full_path).resolve().parent

        # 收集项目内可能的顶层包目录
        local_top: set[str] = set()
        for candidate in proj_path.iterdir() if proj_path.exists() else []:
            if candidate.is_dir() and (candidate / "__init__.py").exists():
                local_top.add(candidate.name)
            if candidate.is_file() and candidate.suffix == ".py":
                local_top.add(candidate.stem)
        # src/ layout
        src_dir = proj_path / "src"
        if src_dir.exists():
            for candidate in src_dir.iterdir():
                if candidate.is_dir() and (candidate / "__init__.py").exists():
                    local_top.add(candidate.name)

        stdlib_names = _stdlib_module_names()

        def _resolves(top: str) -> bool:
            top_l = top.lower().replace("-", "_")
            if top in stdlib_names:
                return True
            if top in local_top:
                return True
            if top_l in declared:
                return True
            # 退一步：能在 sys.path 里找到？（避免误报）
            try:
                import importlib.util
                spec = importlib.util.find_spec(top)
                if spec is not None:
                    return True
            except (ImportError, ValueError, ModuleNotFoundError):
                pass
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if not _resolves(top):
                        unresolved.append(f"unresolved import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # 相对 import — 走目录解析
                    target_dir = file_dir
                    for _ in range(node.level - 1):
                        target_dir = target_dir.parent
                    mod = node.module or ""
                    if mod:
                        candidate1 = target_dir / mod.replace(".", "/") / "__init__.py"
                        candidate2 = target_dir / (mod.replace(".", "/") + ".py")
                        if not candidate1.exists() and not candidate2.exists():
                            # 也可能是从包里导入符号
                            candidate3 = target_dir / "__init__.py"
                            if not candidate3.exists():
                                unresolved.append(
                                    f"relative import '{'.' * node.level}{mod}' "
                                    f"unresolved"
                                )
                    # else: from . import X — 通常解析为同目录 __init__；放过
                else:
                    if node.module:
                        top = node.module.split(".")[0]
                        if not _resolves(top):
                            unresolved.append(
                                f"unresolved import 'from {node.module}'"
                            )
        return unresolved

    # ---- LSP check ----------------------------------------------------

    async def _lsp_check(
        self, files_changed: list[str], project_path: str
    ) -> VerificationCheck:
        """LSP 错误诊断；超时 _LSP_TIMEOUT 秒，超时即跳过."""
        start = time.monotonic()
        if self._lsp_manager is None:
            return VerificationCheck(
                check_name="lsp",
                passed=True,
                skipped=True,
                skip_reason="lsp manager not configured",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=0,
                items_failed=0,
            )

        try:
            errors = await asyncio.wait_for(
                self._collect_lsp_errors(files_changed, project_path),
                timeout=_LSP_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return VerificationCheck(
                check_name="lsp",
                passed=True,
                skipped=True,
                skip_reason=f"lsp timeout {_LSP_TIMEOUT}s",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=len(files_changed),
                items_failed=0,
            )
        except Exception as e:  # noqa: BLE001 — LSP 任何异常都不能挂掉验证
            logger.debug("LSP check error: %s", e)
            return VerificationCheck(
                check_name="lsp",
                passed=True,
                skipped=True,
                skip_reason=f"lsp error: {type(e).__name__}",
                duration_seconds=time.monotonic() - start,
                details="",
                items_checked=0,
                items_failed=0,
            )

        passed = len(errors) == 0
        return VerificationCheck(
            check_name="lsp",
            passed=passed,
            skipped=False,
            skip_reason=None,
            duration_seconds=time.monotonic() - start,
            details="ok" if passed else "\n".join(errors[:5]),
            items_checked=len(files_changed),
            items_failed=len(errors),
        )

    async def _collect_lsp_errors(
        self, files_changed: list[str], project_path: str
    ) -> list[str]:
        """从 LSP manager 拉错误级诊断."""
        manager = self._lsp_manager
        out: list[str] = []
        for f in files_changed:
            full = f if os.path.isabs(f) else os.path.join(project_path, f)
            try:
                await manager.ensure_ready(full)
                if not manager.is_ready():
                    continue
                uri = await manager.open_document(full)
                await asyncio.sleep(0.3)
                diags = manager._diagnostics.get(uri, []) if hasattr(
                    manager, "_diagnostics"
                ) else []
                for d in diags:
                    if d.get("severity", 1) == 1:  # error only
                        line = d.get("range", {}).get("start", {}).get("line", 0) + 1
                        msg = d.get("message", "")
                        out.append(f"{f}:L{line}: {msg}")
            except FileNotFoundError:
                continue
        return out


# ---------------------------------------------------------------------------
# stdlib 名字缓存
# ---------------------------------------------------------------------------

_STDLIB_CACHE: set[str] | None = None


def _stdlib_module_names() -> set[str]:
    global _STDLIB_CACHE
    if _STDLIB_CACHE is not None:
        return _STDLIB_CACHE
    import sys
    names: set[str] = set()
    if hasattr(sys, "stdlib_module_names"):
        names = set(sys.stdlib_module_names)
    # 加常见的不一定在 stdlib_module_names 里的
    names.update({"__future__", "typing_extensions"})
    _STDLIB_CACHE = names
    return names
