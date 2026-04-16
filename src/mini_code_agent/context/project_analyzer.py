"""项目分析器 — 检测项目类型、生成目录树、识别关键文件、提取文件摘要."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ProjectInfo:
    """项目元信息."""

    project_type: str  # python / node / rust / go / java / unknown
    language: str  # Python / JavaScript / TypeScript / Rust / Go / Java / Unknown
    framework: str = ""  # flask / django / react / vue 等
    package_manager: str = ""  # uv / pip / npm / yarn / cargo 等
    name: str = ""
    version: str = ""
    description: str = ""
    entry_points: list[str] = field(default_factory=list)  # 入口文件


# ---------------------------------------------------------------------------
# 忽略规则
# ---------------------------------------------------------------------------

_IGNORE_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".eggs", ".next", ".nuxt",
    "target",  # Rust / Java
    "vendor",  # Go
    ".idea", ".vscode",
}

_IGNORE_SUFFIXES: set[str] = {".egg-info", ".pyc", ".pyo"}


def _should_ignore(entry: Path) -> bool:
    """判断是否应该忽略该目录/文件."""
    name = entry.name
    if name in _IGNORE_DIRS:
        return True
    for suffix in _IGNORE_SUFFIXES:
        if name.endswith(suffix):
            return True
    if entry.is_dir() and name.startswith("."):
        return True
    return False


# ---------------------------------------------------------------------------
# 项目类型检测
# ---------------------------------------------------------------------------

def detect_project_type(path: str | Path) -> ProjectInfo:
    """从项目根目录的配置文件推断项目类型.

    检测顺序：pyproject.toml → package.json → Cargo.toml → go.mod
    """
    root = Path(path).resolve()
    info = ProjectInfo(project_type="unknown", language="Unknown")

    # Python: pyproject.toml / setup.py / requirements.txt
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        info = _parse_pyproject(pyproject, info)
    elif (root / "setup.py").exists() or (root / "requirements.txt").exists():
        info.project_type = "python"
        info.language = "Python"
        info.package_manager = "pip"

    # Node: package.json
    elif (root / "package.json").exists():
        info = _parse_package_json(root / "package.json", info)

    # Rust: Cargo.toml
    elif (root / "Cargo.toml").exists():
        info = _parse_cargo_toml(root / "Cargo.toml", info)

    # Go: go.mod
    elif (root / "go.mod").exists():
        info = _parse_go_mod(root / "go.mod", info)

    # Java: pom.xml / build.gradle
    elif (root / "pom.xml").exists() or (root / "build.gradle").exists():
        info.project_type = "java"
        info.language = "Java"
        info.package_manager = "maven" if (root / "pom.xml").exists() else "gradle"

    # 检测入口文件
    info.entry_points = _detect_entry_points(root, info.project_type)

    return info


def _parse_pyproject(path: Path, info: ProjectInfo) -> ProjectInfo:
    """解析 pyproject.toml 获取项目信息（简单正则，不依赖 toml 库）."""
    info.project_type = "python"
    info.language = "Python"

    text = path.read_text(encoding="utf-8")

    # 检测包管理器
    if "hatchling" in text or "[tool.hatch" in text:
        info.package_manager = "hatch"
    elif "[tool.poetry" in text:
        info.package_manager = "poetry"
    else:
        info.package_manager = "pip"

    # 如果有 uv.lock，说明用 uv
    if (path.parent / "uv.lock").exists():
        info.package_manager = "uv"

    # 项目名称
    m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        info.name = m.group(1)

    # 版本号
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        info.version = m.group(1)

    # 描述
    m = re.search(r'^description\s*=\s*"([^"]*)"', text, re.MULTILINE)
    if m:
        info.description = m.group(1)

    # 框架检测（从依赖列表中推断）
    deps_lower = text.lower()
    if "django" in deps_lower:
        info.framework = "django"
    elif "flask" in deps_lower:
        info.framework = "flask"
    elif "fastapi" in deps_lower:
        info.framework = "fastapi"

    return info


def _parse_package_json(path: Path, info: ProjectInfo) -> ProjectInfo:
    """解析 package.json 获取项目信息."""
    import json

    info.project_type = "node"
    info.language = "JavaScript"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return info

    info.name = data.get("name", "")
    info.version = data.get("version", "")
    info.description = data.get("description", "")

    # 检测 TypeScript
    if (path.parent / "tsconfig.json").exists():
        info.language = "TypeScript"

    # 包管理器
    if (path.parent / "yarn.lock").exists():
        info.package_manager = "yarn"
    elif (path.parent / "pnpm-lock.yaml").exists():
        info.package_manager = "pnpm"
    elif (path.parent / "bun.lockb").exists() or (path.parent / "bun.lock").exists():
        info.package_manager = "bun"
    else:
        info.package_manager = "npm"

    # 框架检测
    all_deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
    if "react" in all_deps:
        info.framework = "react"
    elif "vue" in all_deps:
        info.framework = "vue"
    elif "svelte" in all_deps:
        info.framework = "svelte"
    elif "next" in all_deps:
        info.framework = "next"

    return info


def _parse_cargo_toml(path: Path, info: ProjectInfo) -> ProjectInfo:
    """解析 Cargo.toml."""
    info.project_type = "rust"
    info.language = "Rust"
    info.package_manager = "cargo"

    text = path.read_text(encoding="utf-8")

    m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        info.name = m.group(1)

    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        info.version = m.group(1)

    return info


def _parse_go_mod(path: Path, info: ProjectInfo) -> ProjectInfo:
    """解析 go.mod."""
    info.project_type = "go"
    info.language = "Go"
    info.package_manager = "go modules"

    text = path.read_text(encoding="utf-8")

    m = re.search(r'^module\s+(\S+)', text, re.MULTILINE)
    if m:
        info.name = m.group(1)

    return info


def _detect_entry_points(root: Path, project_type: str) -> list[str]:
    """检测项目入口文件."""
    entries: list[str] = []

    candidates = {
        "python": ["main.py", "app.py", "manage.py", "cli.py", "__main__.py"],
        "node": ["index.js", "index.ts", "src/index.js", "src/index.ts",
                 "app.js", "app.ts", "server.js", "server.ts"],
        "rust": ["src/main.rs", "src/lib.rs"],
        "go": ["main.go", "cmd/main.go"],
        "java": ["src/main/java"],
    }

    for candidate in candidates.get(project_type, []):
        p = root / candidate
        if p.exists():
            entries.append(candidate)

    return entries


# ---------------------------------------------------------------------------
# 目录树生成
# ---------------------------------------------------------------------------

def get_directory_tree(path: str | Path, max_depth: int = 3) -> str:
    """生成目录树字符串，智能忽略 .git/node_modules/__pycache__ 等.

    Args:
        path: 项目根目录
        max_depth: 最大递归深度

    Returns:
        树形结构字符串
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return f"[不是目录: {root}]"

    lines: list[str] = [f"{root.name}/"]
    _walk_tree(root, lines, prefix="", depth=0, max_depth=max_depth)
    return "\n".join(lines)


def _walk_tree(
    directory: Path,
    lines: list[str],
    prefix: str,
    depth: int,
    max_depth: int,
) -> None:
    """递归生成目录树."""
    if depth >= max_depth:
        return

    try:
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        lines.append(f"{prefix}[权限不足]")
        return

    entries = [e for e in entries if not _should_ignore(e)]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            _walk_tree(entry, lines, child_prefix, depth + 1, max_depth)
        else:
            lines.append(f"{prefix}{connector}{entry.name}")


# ---------------------------------------------------------------------------
# 关键文件识别
# ---------------------------------------------------------------------------

def get_key_files(path: str | Path) -> list[str]:
    """返回应该优先读取的关键文件列表.

    包括：README, 配置文件, 入口文件, 项目指令文件等。
    """
    root = Path(path).resolve()
    key_files: list[str] = []

    # 优先级从高到低
    candidates = [
        # 项目指令文件
        "CLAUDE.md", "AGENT.md", ".cursorrules",
        # 文档
        "README.md", "README.rst", "README.txt", "README",
        # Python 配置
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
        # Node 配置
        "package.json", "tsconfig.json",
        # Rust 配置
        "Cargo.toml",
        # Go 配置
        "go.mod",
        # Java 配置
        "pom.xml", "build.gradle", "build.gradle.kts",
        # 通用配置
        "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".env.example",
        # 入口文件
        "main.py", "app.py", "index.js", "index.ts",
        "src/main.rs", "src/lib.rs", "main.go",
    ]

    for candidate in candidates:
        p = root / candidate
        if p.exists() and p.is_file():
            # 返回相对路径
            key_files.append(candidate)

    return key_files


# ---------------------------------------------------------------------------
# 文件摘要
# ---------------------------------------------------------------------------

# 匹配常见语言的顶级定义
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ".py": [
        re.compile(r"^class\s+(\w+)(?:\([^)]*\))?:", re.MULTILINE),
        re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE),
        re.compile(r"^async\s+def\s+(\w+)\s*\(", re.MULTILINE),
    ],
    ".js": [
        re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+(?:default\s+)?function\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+const\s+(\w+)", re.MULTILINE),
        re.compile(r"^class\s+(\w+)", re.MULTILINE),
        re.compile(r"^function\s+(\w+)", re.MULTILINE),
    ],
    ".ts": [
        re.compile(r"^export\s+(?:default\s+)?class\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+(?:default\s+)?function\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+(?:default\s+)?(?:const|let|var)\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+(?:default\s+)?interface\s+(\w+)", re.MULTILINE),
        re.compile(r"^export\s+(?:default\s+)?type\s+(\w+)", re.MULTILINE),
        re.compile(r"^class\s+(\w+)", re.MULTILINE),
        re.compile(r"^function\s+(\w+)", re.MULTILINE),
        re.compile(r"^interface\s+(\w+)", re.MULTILINE),
    ],
    ".rs": [
        re.compile(r"^pub\s+(?:async\s+)?fn\s+(\w+)", re.MULTILINE),
        re.compile(r"^pub\s+struct\s+(\w+)", re.MULTILINE),
        re.compile(r"^pub\s+enum\s+(\w+)", re.MULTILINE),
        re.compile(r"^pub\s+trait\s+(\w+)", re.MULTILINE),
    ],
    ".go": [
        re.compile(r"^func\s+(\w+)\s*\(", re.MULTILINE),
        re.compile(r"^func\s+\([^)]+\)\s+(\w+)\s*\(", re.MULTILINE),
        re.compile(r"^type\s+(\w+)\s+struct", re.MULTILINE),
        re.compile(r"^type\s+(\w+)\s+interface", re.MULTILINE),
    ],
    ".java": [
        re.compile(r"^\s*public\s+(?:static\s+)?class\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*public\s+(?:static\s+)?interface\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*public\s+(?:static\s+)?(?:\w+\s+)*(\w+)\s*\(", re.MULTILINE),
    ],
}

# .jsx/.tsx 复用 .js/.ts 的模式
_PATTERNS[".jsx"] = _PATTERNS[".js"]
_PATTERNS[".tsx"] = _PATTERNS[".ts"]


def summarize_file(path: str | Path) -> str:
    """对代码文件生成单行摘要：文件路径 + 导出的类/函数签名.

    不需要 LLM，用简单的正则匹配 class/def/function/export。

    Returns:
        格式如 "src/core/agent.py — Agent class: run(), reset()"
    """
    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        return str(file_path)

    suffix = file_path.suffix
    patterns = _PATTERNS.get(suffix)

    if patterns is None:
        # 非代码文件，只返回路径
        return str(file_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return str(file_path)

    # 收集所有匹配的名称，按类别分组
    classes: list[str] = []
    functions: list[str] = []

    seen: set[str] = set()

    for pattern in patterns:
        for m in pattern.finditer(content):
            name = m.group(1)
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)

            match_str = m.group(0)
            if any(kw in match_str for kw in ("class ", "struct ", "trait ", "interface ", "enum ", "type ")):
                classes.append(name)
            else:
                functions.append(name)

    # 构建摘要
    parts: list[str] = []
    if classes:
        for cls in classes:
            # 查找该类下的方法（Python 特定）
            if suffix == ".py":
                methods = _find_class_methods(content, cls)
                if methods:
                    parts.append(f"{cls}: {', '.join(methods)}")
                else:
                    parts.append(cls)
            else:
                parts.append(cls)

    if functions:
        parts.append(", ".join(functions))

    summary = "; ".join(parts) if parts else "（空文件或无公共定义）"
    return f"{file_path} — {summary}"


def _find_class_methods(content: str, class_name: str) -> list[str]:
    """查找 Python 类中的公开方法名（不以 _ 开头的 def）."""
    # 找到 class 定义的位置
    class_pattern = re.compile(
        rf"^class\s+{re.escape(class_name)}\b.*?:\s*$", re.MULTILINE
    )
    m = class_pattern.search(content)
    if not m:
        return []

    # 从 class 定义后开始，找到下一个顶级定义为止
    start = m.end()
    next_top = re.search(r"^\S", content[start:], re.MULTILINE)
    class_body = content[start:start + next_top.start()] if next_top else content[start:]

    # 匹配方法定义
    method_pattern = re.compile(r"^\s+(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
    methods = []
    for mm in method_pattern.finditer(class_body):
        name = mm.group(1)
        if not name.startswith("_"):
            methods.append(f"{name}()")
    return methods
