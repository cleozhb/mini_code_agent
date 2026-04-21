"""对比 pylsp vs pyright 的诊断能力.

用法：
    uv run python scripts/compare_lsp_servers.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

TEST_CODE = '''\
def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    return "hello " + name


# 类型错误 1: int 赋给 str
result: str = add(1, 2)

# 类型错误 2: str 传给 int 参数
add("hello", "world")

# 类型错误 3: 返回值类型不匹配
def get_count() -> int:
    return "not a number"

# 类型错误 4: 属性不存在
x: int = 42
x.append(1)

# 正常代码（不应报错）
y = add(3, 4)
msg = greet("alice")
'''


async def run_diagnostics(
    server_name: str, binary: str, args: list[str], file_path: str
) -> list[str]:
    """启动 LSP server，打开文件，收集诊断."""
    resolved = shutil.which(binary)
    if not resolved:
        return [f"  ⚠ {binary} 未安装，跳过"]

    proc = await asyncio.create_subprocess_exec(
        resolved, *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def write_msg(payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        proc.stdin.write(header + body)
        await proc.stdin.drain()

    async def read_msg() -> dict:
        content_length = -1
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise ConnectionError("Server disconnected")
            line_str = line.decode("ascii").strip()
            if line_str == "":
                break
            if line_str.lower().startswith("content-length:"):
                content_length = int(line_str.split(":")[1].strip())
        body = await proc.stdout.readexactly(content_length)
        return json.loads(body.decode("utf-8"))

    # initialize
    await write_msg({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "processId": None,
            "rootUri": f"file://{Path(file_path).parent}",
            "capabilities": {
                "textDocument": {"publishDiagnostics": {"relatedInformation": True}}
            },
        },
    })

    while True:
        msg = await asyncio.wait_for(read_msg(), timeout=30.0)
        if msg.get("id") == 1:
            break

    await write_msg({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    content = Path(file_path).read_text()
    uri = f"file://{file_path}"
    await write_msg({
        "jsonrpc": "2.0", "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": uri, "languageId": "python", "version": 1, "text": content,
            },
        },
    })

    severity_map = {1: "error", 2: "warning", 3: "info", 4: "hint"}
    diagnostics = []

    try:
        deadline = asyncio.get_event_loop().time() + 15.0
        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            msg = await asyncio.wait_for(read_msg(), timeout=remaining)
            if msg.get("method") == "textDocument/publishDiagnostics":
                for diag in msg["params"].get("diagnostics", []):
                    line = diag["range"]["start"]["line"] + 1
                    sev = severity_map.get(diag.get("severity", 1), "?")
                    message = diag.get("message", "")
                    diagnostics.append(f"  L{line:2d} [{sev:7s}] {message}")
                try:
                    extra_msg = await asyncio.wait_for(read_msg(), timeout=2.0)
                    if extra_msg.get("method") == "textDocument/publishDiagnostics":
                        for diag in extra_msg["params"].get("diagnostics", []):
                            line = diag["range"]["start"]["line"] + 1
                            sev = severity_map.get(diag.get("severity", 1), "?")
                            message = diag.get("message", "")
                            diagnostics.append(f"  L{line:2d} [{sev:7s}] {message}")
                except asyncio.TimeoutError:
                    pass
                break
    except asyncio.TimeoutError:
        if not diagnostics:
            diagnostics.append("  (等待超时，未收到诊断)")

    await write_msg({"jsonrpc": "2.0", "id": 99, "method": "shutdown", "params": None})
    await write_msg({"jsonrpc": "2.0", "method": "exit", "params": None})

    try:
        await asyncio.wait_for(proc.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        proc.kill()

    return diagnostics if diagnostics else ["  (无诊断输出)"]


async def main() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(TEST_CODE)
        test_file = f.name

    print("=" * 60)
    print("LSP Server 诊断能力对比")
    print("=" * 60)
    print(f"\n测试文件包含 4 处类型错误:\n")
    print("  L10: int 赋给 str 变量")
    print("  L13: str 传给 int 参数")
    print("  L17: 返回 str 但声明返回 int")
    print("  L21: int 没有 append 属性")
    print()

    print("-" * 60)
    print("▶ pylsp (python-lsp-server)")
    print("-" * 60)
    results = await run_diagnostics("pylsp", "pylsp", [], test_file)
    for line in results:
        print(line)
    print(f"\n  共发现: {len([r for r in results if '[error' in r or '[warning' in r])} 个问题")

    print()

    print("-" * 60)
    print("▶ pyright (pyright-langserver)")
    print("-" * 60)
    results = await run_diagnostics("pyright", "pyright-langserver", ["--stdio"], test_file)
    for line in results:
        print(line)
    print(f"\n  共发现: {len([r for r in results if '[error' in r or '[warning' in r])} 个问题")

    print()
    print("=" * 60)
    print("结论: pyright 在类型检查方面通常更严格、更全面")
    print("=" * 60)

    Path(test_file).unlink()


if __name__ == "__main__":
    asyncio.run(main())
