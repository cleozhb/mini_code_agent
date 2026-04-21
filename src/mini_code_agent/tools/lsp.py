"""LSP 工具集：通过 Language Server Protocol 提供精确代码分析能力."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


# ===========================================================================
# 常量与映射
# ===========================================================================

# 语言 → (server binary, 启动参数)
_SERVER_MAP: dict[str, tuple[str, list[str]]] = {
    "python": ("pyright-langserver", ["--stdio"]),
    "typescript": ("typescript-language-server", ["--stdio"]),
    "javascript": ("typescript-language-server", ["--stdio"]),
    "go": ("gopls", ["serve"]),
    "rust": ("rust-analyzer", []),
}

# 备选 server（主 server 未安装时尝试）
_SERVER_FALLBACKS: dict[str, tuple[str, list[str]]] = {
    "python": ("pylsp", []),
}

# 安装建议
_INSTALL_HINTS: dict[str, str] = {
    "python": "pip install pyright  # 或 pip install python-lsp-server",
    "typescript": "npm install -g typescript-language-server typescript",
    "javascript": "npm install -g typescript-language-server typescript",
    "go": "go install golang.org/x/tools/gopls@latest",
    "rust": "rustup component add rust-analyzer",
}

# 文件扩展名 → 语言
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
}

# 项目配置文件 → 语言（用于探测项目根目录）
_CONFIG_TO_LANGUAGE: dict[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "requirements.txt": "python",
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "go.mod": "go",
    "Cargo.toml": "rust",
}

# LSP languageId
_LANGUAGE_IDS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".go": "go",
    ".rs": "rust",
}


# ===========================================================================
# JSON-RPC 传输层
# ===========================================================================


class JsonRpcError(Exception):
    """JSON-RPC 协议错误."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(f"JSON-RPC error {code}: {message}")


async def _write_message(
    proc: asyncio.subprocess.Process, payload: dict
) -> None:
    """写入一条 JSON-RPC 消息（Content-Length 帧）."""
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    assert proc.stdin is not None
    proc.stdin.write(header + body)
    await proc.stdin.drain()


async def _read_message(
    proc: asyncio.subprocess.Process, timeout: float = 5.0
) -> dict:
    """从 proc.stdout 读取一条 JSON-RPC 消息，带超时."""
    assert proc.stdout is not None

    async def _do_read() -> dict:
        # 读取 headers
        content_length = -1
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise ConnectionError("Language server 连接断开")
            line_str = line.decode("ascii").strip()
            if line_str == "":
                break  # headers 结束
            if line_str.lower().startswith("content-length:"):
                content_length = int(line_str.split(":")[1].strip())

        if content_length < 0:
            raise ConnectionError("未收到 Content-Length header")

        # 读取 body
        body = await proc.stdout.readexactly(content_length)
        return json.loads(body.decode("utf-8"))

    return await asyncio.wait_for(_do_read(), timeout=timeout)


# ===========================================================================
# 辅助函数
# ===========================================================================


def _path_to_uri(path: str) -> str:
    """文件路径 → file:// URI."""
    abs_path = os.path.abspath(path)
    return f"file://{abs_path}"


def _uri_to_path(uri: str) -> str:
    """file:// URI → 文件路径."""
    if uri.startswith("file://"):
        return uri[7:]
    return uri


def _detect_language_from_ext(file_path: str) -> str | None:
    """从文件扩展名推断语言."""
    ext = Path(file_path).suffix.lower()
    return _EXT_TO_LANGUAGE.get(ext)


def _detect_project_root(file_path: str) -> str:
    """向上查找项目根目录（包含项目配置文件的最近祖先目录）."""
    current = Path(file_path).resolve().parent
    config_files = set(_CONFIG_TO_LANGUAGE.keys())
    while current != current.parent:
        if any((current / f).exists() for f in config_files):
            return str(current)
        current = current.parent
    return os.getcwd()


def _find_symbol_column(file_path: str, line: int, symbol_name: str) -> int | None:
    """在指定行中查找符号名的列号（1-based）。

    返回 None 表示该行中找不到该符号。
    优先匹配完整单词（避免 "foo" 匹配到 "foobar"）。
    """
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if line < 1 or line > len(lines):
        return None

    line_text = lines[line - 1]

    # 优先做完整单词匹配
    import re
    pattern = re.compile(r"(?<![a-zA-Z0-9_])" + re.escape(symbol_name) + r"(?![a-zA-Z0-9_])")
    match = pattern.search(line_text)
    if match:
        return match.start() + 1  # 转为 1-based

    # 回退到子串匹配
    idx = line_text.find(symbol_name)
    if idx >= 0:
        return idx + 1

    return None


def _read_surrounding_lines(file_path: str, line: int, context: int = 5) -> str:
    """读取目标行周围的代码（带行号），用于上下文展示."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"[文件不存在: {file_path}]"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line - 1 - context)
        end = min(len(lines), line - 1 + context + 1)
        result = []
        for i in range(start, end):
            marker = " → " if i == line - 1 else "   "
            result.append(f"{marker}{i + 1:4d} │ {lines[i]}")
        return "\n".join(result)
    except (OSError, UnicodeDecodeError):
        return f"[无法读取文件: {file_path}]"


def _language_id(file_path: str) -> str:
    """文件扩展名 → LSP languageId."""
    ext = Path(file_path).suffix.lower()
    return _LANGUAGE_IDS.get(ext, "plaintext")


# ===========================================================================
# LSPManager
# ===========================================================================


class LSPManager:
    """管理 Language Server 的生命周期和通信."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._language: str | None = None
        self._project_path: str | None = None
        self._request_id: int = 0
        self._initialized: bool = False
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._open_docs: dict[str, int] = {}  # uri → version
        self._diagnostics: dict[str, list[dict]] = {}  # uri → diagnostics

    # --- Public API ---

    async def start_server(self, language: str, project_path: str) -> None:
        """启动指定语言的 language server.

        Raises:
            FileNotFoundError: 找不到 server binary 时抛出，附带安装建议。
        """
        # 查找 binary
        binary, args = _SERVER_MAP.get(language, (None, []))
        resolved = shutil.which(binary) if binary else None

        if not resolved and language in _SERVER_FALLBACKS:
            binary, args = _SERVER_FALLBACKS[language]
            resolved = shutil.which(binary)

        if not resolved:
            hint = _INSTALL_HINTS.get(language, "请安装对应语言的 language server")
            raise FileNotFoundError(
                f"{language} 的 language server 未安装。\n"
                f"安装方法: {hint}"
            )

        # 启动进程
        self._proc = await asyncio.create_subprocess_exec(
            resolved,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 启动后台读取循环
        self._reader_task = asyncio.create_task(self._reader_loop())

        # LSP initialize 握手
        await self._initialize(project_path)

        self._language = language
        self._project_path = project_path

    async def stop_server(self) -> None:
        """优雅关闭 language server."""
        if self._proc is None or self._proc.returncode is not None:
            self._reset_state()
            return

        try:
            # shutdown 请求
            await self.request("shutdown", {})
        except (asyncio.TimeoutError, ConnectionError, JsonRpcError):
            pass

        try:
            # exit 通知
            await self.notify("exit", {})
        except (ConnectionError, OSError):
            pass

        # 等待进程退出
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()

        self._reset_state()

    def is_ready(self) -> bool:
        """检查 server 是否已初始化且进程存活."""
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._initialized
        )

    async def ensure_ready(self, file_path: str) -> None:
        """懒启动：从文件扩展名推断语言并启动 server（如尚未启动）.

        Raises:
            FileNotFoundError: server binary 不存在时抛出。
        """
        language = _detect_language_from_ext(file_path)
        if not language:
            raise FileNotFoundError(
                f"无法识别文件 '{file_path}' 的语言，LSP 不支持该文件类型"
            )

        # 已就绪且语言相同：直接返回
        if self.is_ready() and self._language == language:
            return

        # 语言不同或未启动：先停再启动
        if self.is_ready() and self._language != language:
            await self.stop_server()

        project_path = _detect_project_root(file_path)
        await self.start_server(language, project_path)

    async def request(self, method: str, params: dict) -> Any:
        """发送 JSON-RPC request 并等待响应（5 秒超时）."""
        if self._proc is None or self._proc.returncode is not None:
            raise ConnectionError("Language server 未运行")

        req_id = self._next_id()
        future: asyncio.Future[dict] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        try:
            await _write_message(self._proc, payload)
            response = await asyncio.wait_for(future, timeout=5.0)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise
        finally:
            self._pending.pop(req_id, None)

        if "error" in response:
            err = response["error"]
            raise JsonRpcError(err.get("code", -1), err.get("message", "未知错误"))

        return response.get("result")

    async def notify(self, method: str, params: dict) -> None:
        """发送 JSON-RPC notification（不期望响应）."""
        if self._proc is None or self._proc.returncode is not None:
            raise ConnectionError("Language server 未运行")

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await _write_message(self._proc, payload)

    async def open_document(self, file_path: str) -> str:
        """发送 textDocument/didOpen（如尚未打开）。返回文档 URI.

        如果文件已打开过，会重新发送 didClose + didOpen 以刷新内容。
        """
        uri = _path_to_uri(file_path)

        # 如果已打开过：关闭后重新打开（刷新编辑后的内容）
        if uri in self._open_docs:
            await self.notify("textDocument/didClose", {
                "textDocument": {"uri": uri},
            })

        # 读取文件内容
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""

        version = self._open_docs.get(uri, 0) + 1
        self._open_docs[uri] = version

        await self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": _language_id(file_path),
                "version": version,
                "text": content,
            },
        })

        return uri

    # --- Internal ---

    async def _reader_loop(self) -> None:
        """后台任务：持续从 server stdout 读取消息并分发."""
        try:
            while self._proc and self._proc.returncode is None:
                try:
                    msg = await _read_message(self._proc, timeout=30.0)
                except asyncio.TimeoutError:
                    continue  # 长时间无消息是正常的
                except (ConnectionError, OSError):
                    break  # server 断开

                # 有 id 的是 response → 分发给对应的 future
                if "id" in msg and msg["id"] in self._pending:
                    future = self._pending[msg["id"]]
                    if not future.done():
                        future.set_result(msg)
                # publishDiagnostics 通知 → 缓存
                elif msg.get("method") == "textDocument/publishDiagnostics":
                    params = msg.get("params", {})
                    uri = params.get("uri", "")
                    diagnostics = params.get("diagnostics", [])
                    self._diagnostics[uri] = diagnostics
                # 其他通知/请求：忽略
        except asyncio.CancelledError:
            pass
        finally:
            self._initialized = False

    async def _initialize(self, project_path: str) -> None:
        """执行 LSP initialize 握手."""
        result = await self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": _path_to_uri(project_path),
            "rootPath": project_path,
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "hover": {
                        "dynamicRegistration": False,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True,
                    },
                },
            },
        })

        # 发送 initialized 通知
        await self.notify("initialized", {})
        self._initialized = True
        return result

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _reset_state(self) -> None:
        """重置所有状态."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        self._proc = None
        self._language = None
        self._project_path = None
        self._initialized = False
        self._pending.clear()
        self._open_docs.clear()
        self._diagnostics.clear()
        self._reader_task = None


# ===========================================================================
# GotoDefinitionTool
# ===========================================================================


class GotoDefinitionInput(BaseModel):
    file_path: str = Field(description="文件路径")
    line: int = Field(description="行号（从 1 开始）")
    symbol_name: str = Field(description="要跳转到定义的符号名（函数名、类名、变量名等）")


@dataclass
class GotoDefinitionTool(Tool):
    """跳转到符号定义位置."""

    InputModel: ClassVar[type[BaseModel]] = GotoDefinitionInput

    name: str = "GotoDefinition"
    description: str = (
        "跳转到指定位置的符号定义。返回定义所在的文件路径、行号，"
        "以及定义处周围的代码片段。需要提供源文件路径、行号和符号名。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    _lsp_manager: Any = field(default=None, repr=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._lsp_manager is None:
            return ToolResult(output="", error="LSP 管理器未初始化")

        file_path: str = kwargs["file_path"]
        line: int = kwargs["line"]
        symbol_name: str = kwargs["symbol_name"]
        manager: LSPManager = self._lsp_manager

        # 自动定位列号
        column = _find_symbol_column(file_path, line, symbol_name)
        if column is None:
            return ToolResult(
                output="",
                error=f"在 {file_path} 第 {line} 行未找到符号 '{symbol_name}'",
            )

        # 懒启动 server
        try:
            await manager.ensure_ready(file_path)
        except FileNotFoundError as e:
            return ToolResult(output="", error=str(e))

        if not manager.is_ready():
            return ToolResult(output="", error="语言服务器未就绪")

        # 打开文档
        uri = await manager.open_document(file_path)

        # 请求跳转定义
        try:
            result = await manager.request("textDocument/definition", {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": column - 1},
            })
        except asyncio.TimeoutError:
            return ToolResult(output="", error="LSP 请求超时（5s）")
        except (JsonRpcError, ConnectionError) as e:
            return ToolResult(output="", error=f"LSP 请求失败: {e}")

        # 解析响应（Location | Location[] | LocationLink[]）
        if result is None or (isinstance(result, list) and len(result) == 0):
            return ToolResult(output="未找到定义")

        locations = result if isinstance(result, list) else [result]
        loc = locations[0]

        # 提取目标位置
        target_uri = loc.get("uri") or loc.get("targetUri", "")
        target_range = loc.get("range") or loc.get("targetRange", {})
        target_path = _uri_to_path(target_uri)
        target_line = target_range.get("start", {}).get("line", 0) + 1

        # 读取周围代码
        code_context = _read_surrounding_lines(target_path, target_line)

        return ToolResult(
            output=f"定义位置: {target_path}:{target_line}\n\n{code_context}"
        )


# ===========================================================================
# FindReferencesTool
# ===========================================================================


class FindReferencesInput(BaseModel):
    file_path: str = Field(description="文件路径")
    line: int = Field(description="行号（从 1 开始）")
    symbol_name: str = Field(description="要查找引用的符号名（函数名、类名、变量名等）")


@dataclass
class FindReferencesTool(Tool):
    """查找符号的所有引用位置."""

    InputModel: ClassVar[type[BaseModel]] = FindReferencesInput

    name: str = "FindReferences"
    description: str = (
        "查找指定位置符号的所有引用。返回所有引用该符号的文件路径和行号列表。"
        "可用于了解某个函数、类或变量在项目中的使用情况。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    _lsp_manager: Any = field(default=None, repr=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._lsp_manager is None:
            return ToolResult(output="", error="LSP 管理器未初始化")

        file_path: str = kwargs["file_path"]
        line: int = kwargs["line"]
        symbol_name: str = kwargs["symbol_name"]
        manager: LSPManager = self._lsp_manager

        # 自动定位列号
        column = _find_symbol_column(file_path, line, symbol_name)
        if column is None:
            return ToolResult(
                output="",
                error=f"在 {file_path} 第 {line} 行未找到符号 '{symbol_name}'",
            )

        try:
            await manager.ensure_ready(file_path)
        except FileNotFoundError as e:
            return ToolResult(output="", error=str(e))

        if not manager.is_ready():
            return ToolResult(output="", error="语言服务器未就绪")

        uri = await manager.open_document(file_path)

        try:
            result = await manager.request("textDocument/references", {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": column - 1},
                "context": {"includeDeclaration": True},
            })
        except asyncio.TimeoutError:
            return ToolResult(output="", error="LSP 请求超时（5s）")
        except (JsonRpcError, ConnectionError) as e:
            return ToolResult(output="", error=f"LSP 请求失败: {e}")

        if not result:
            return ToolResult(output="未找到任何引用")

        # 格式化引用列表（最多 50 条）
        locations = result[:50]
        lines_output = []
        for i, loc in enumerate(locations, 1):
            loc_uri = loc.get("uri", "")
            loc_path = _uri_to_path(loc_uri)
            loc_line = loc.get("range", {}).get("start", {}).get("line", 0) + 1
            lines_output.append(f"  {i}. {loc_path}:{loc_line}")

        total = len(result)
        header = f"找到 {total} 处引用"
        if total > 50:
            header += f"（显示前 50 条）"
        header += ":\n"

        return ToolResult(output=header + "\n".join(lines_output))


# ===========================================================================
# GetHoverInfoTool
# ===========================================================================


class GetHoverInfoInput(BaseModel):
    file_path: str = Field(description="文件路径")
    line: int = Field(description="行号（从 1 开始）")
    symbol_name: str = Field(description="要查看信息的符号名（函数名、类名、变量名等）")


@dataclass
class GetHoverInfoTool(Tool):
    """获取符号的类型信息和文档."""

    InputModel: ClassVar[type[BaseModel]] = GetHoverInfoInput

    name: str = "GetHoverInfo"
    description: str = (
        "获取指定位置符号的类型签名和文档字符串。"
        "可用于了解函数参数类型、返回值类型、类的接口说明等。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    _lsp_manager: Any = field(default=None, repr=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._lsp_manager is None:
            return ToolResult(output="", error="LSP 管理器未初始化")

        file_path: str = kwargs["file_path"]
        line: int = kwargs["line"]
        symbol_name: str = kwargs["symbol_name"]
        manager: LSPManager = self._lsp_manager

        # 自动定位列号
        column = _find_symbol_column(file_path, line, symbol_name)
        if column is None:
            return ToolResult(
                output="",
                error=f"在 {file_path} 第 {line} 行未找到符号 '{symbol_name}'",
            )

        try:
            await manager.ensure_ready(file_path)
        except FileNotFoundError as e:
            return ToolResult(output="", error=str(e))

        if not manager.is_ready():
            return ToolResult(output="", error="语言服务器未就绪")

        uri = await manager.open_document(file_path)

        try:
            result = await manager.request("textDocument/hover", {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": column - 1},
            })
        except asyncio.TimeoutError:
            return ToolResult(output="", error="LSP 请求超时（5s）")
        except (JsonRpcError, ConnectionError) as e:
            return ToolResult(output="", error=f"LSP 请求失败: {e}")

        if result is None:
            return ToolResult(output="该位置没有可用的悬停信息")

        # 解析 contents（MarkupContent | MarkedString | MarkedString[]）
        contents = result.get("contents", "")
        output = self._parse_hover_contents(contents)

        if not output.strip():
            return ToolResult(output="该位置没有可用的悬停信息")

        return ToolResult(output=output)

    @staticmethod
    def _parse_hover_contents(contents: Any) -> str:
        """解析 LSP Hover 的 contents 字段."""
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            # MarkedString: {"language": "python", "value": "..."} — 优先检查
            if "language" in contents and "value" in contents:
                return f"```{contents['language']}\n{contents['value']}\n```"
            # MarkupContent: {"kind": "markdown"|"plaintext", "value": "..."}
            if "value" in contents:
                return contents["value"]
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "language" in item and "value" in item:
                        parts.append(
                            f"```{item['language']}\n{item['value']}\n```"
                        )
                    elif "value" in item:
                        parts.append(item["value"])
            return "\n\n".join(parts)
        return str(contents)


# ===========================================================================
# GetDiagnosticsTool
# ===========================================================================

_SEVERITY_MAP: dict[int, str] = {
    1: "error",
    2: "warning",
    3: "info",
    4: "hint",
}


class GetDiagnosticsInput(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="文件路径（不传则返回所有已知诊断信息）",
    )


@dataclass
class GetDiagnosticsTool(Tool):
    """获取代码诊断信息（错误、警告）."""

    InputModel: ClassVar[type[BaseModel]] = GetDiagnosticsInput

    name: str = "GetDiagnostics"
    description: str = (
        "获取代码诊断信息，包括语法错误、类型错误、警告等。"
        "可指定文件路径获取单文件诊断，不传则返回所有已知诊断。"
        "修改代码后调用可立即检查是否引入了新错误。"
    )
    permission_level: PermissionLevel = PermissionLevel.AUTO

    _lsp_manager: Any = field(default=None, repr=False)

    async def execute(self, **kwargs: Any) -> ToolResult:
        if self._lsp_manager is None:
            return ToolResult(output="", error="LSP 管理器未初始化")

        file_path: str | None = kwargs.get("file_path")
        manager: LSPManager = self._lsp_manager

        # 如果指定了文件，确保 server 就绪并打开文档
        if file_path:
            try:
                await manager.ensure_ready(file_path)
            except FileNotFoundError as e:
                return ToolResult(output="", error=str(e))

            if not manager.is_ready():
                return ToolResult(output="", error="语言服务器未就绪")

            uri = await manager.open_document(file_path)

            # 等待 server 推送诊断（server 分析需要时间）
            await asyncio.sleep(0.5)

            diagnostics = {uri: manager._diagnostics.get(uri, [])}
        else:
            if not manager.is_ready():
                return ToolResult(output="语言服务器未运行，请先对某个文件使用 LSP 工具")
            diagnostics = dict(manager._diagnostics)

        # 格式化输出
        output_parts = []
        total_count = 0

        for uri, diag_list in diagnostics.items():
            if not diag_list:
                continue
            path = _uri_to_path(uri)
            file_lines = [f"{path}:"]
            for diag in diag_list:
                line_num = diag.get("range", {}).get("start", {}).get("line", 0) + 1
                severity = _SEVERITY_MAP.get(diag.get("severity", 1), "error")
                message = diag.get("message", "")
                file_lines.append(f"  L{line_num}: [{severity}] {message}")
                total_count += 1
            output_parts.append("\n".join(file_lines))

        if not output_parts:
            return ToolResult(output="没有发现诊断问题 ✓")

        header = f"发现 {total_count} 个诊断问题:\n\n"
        return ToolResult(output=header + "\n\n".join(output_parts))
