#!/usr/bin/env python3
"""文件格式转换器 — 一个单文件脚本，支持 CSV/JSON/YAML 之间互转.

这是一个需要被重构为完整项目结构的原始脚本。
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

# ===== 配置 =====

DEFAULT_CONFIG = {
    "input_dir": "input",
    "output_dir": "output",
    "format": "json",
    "verbose": False,
    "encoding": "utf-8",
}

# ===== 日志 =====

logger = logging.getLogger("converter")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(level)


# ===== 工具函数 =====


def detect_format(filepath: str) -> str:
    """根据文件扩展名检测格式."""
    ext = Path(filepath).suffix.lower()
    format_map = {
        ".csv": "csv",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return format_map.get(ext, "unknown")


def ensure_dir(path: str) -> None:
    """确保目录存在."""
    os.makedirs(path, exist_ok=True)


def read_file(filepath: str, encoding: str = "utf-8") -> str:
    """读取文件内容."""
    with open(filepath, "r", encoding=encoding) as f:
        return f.read()


def write_file(filepath: str, content: str, encoding: str = "utf-8") -> None:
    """写入文件内容."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding=encoding) as f:
        f.write(content)


# ===== 核心转换逻辑 =====


def parse_csv(content: str) -> list[dict]:
    """解析 CSV 内容为字典列表."""
    reader = csv.DictReader(content.splitlines())
    return list(reader)


def parse_json(content: str) -> list[dict]:
    """解析 JSON 内容."""
    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"不支持的 JSON 结构: {type(data)}")


def parse_yaml(content: str) -> list[dict]:
    """解析 YAML 内容."""
    try:
        import yaml
    except ImportError:
        raise ImportError("需要安装 pyyaml: pip install pyyaml")
    data = yaml.safe_load(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"不支持的 YAML 结构: {type(data)}")


def to_csv(records: list[dict]) -> str:
    """将记录列表转为 CSV 字符串."""
    if not records:
        return ""
    from io import StringIO
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def to_json(records: list[dict], indent: int = 2) -> str:
    """将记录列表转为 JSON 字符串."""
    return json.dumps(records, indent=indent, ensure_ascii=False)


def to_yaml(records: list[dict]) -> str:
    """将记录列表转为 YAML 字符串."""
    try:
        import yaml
    except ImportError:
        raise ImportError("需要安装 pyyaml: pip install pyyaml")
    return yaml.dump(records, allow_unicode=True, default_flow_style=False)


PARSERS = {
    "csv": parse_csv,
    "json": parse_json,
    "yaml": parse_yaml,
}

SERIALIZERS = {
    "csv": to_csv,
    "json": to_json,
    "yaml": to_yaml,
}


def convert(content: str, from_format: str, to_format: str) -> str:
    """核心转换函数：解析 → 转换 → 序列化."""
    if from_format not in PARSERS:
        raise ValueError(f"不支持的输入格式: {from_format}")
    if to_format not in SERIALIZERS:
        raise ValueError(f"不支持的输出格式: {to_format}")

    logger.debug("解析 %s 格式...", from_format)
    records = PARSERS[from_format](content)
    logger.info("解析到 %d 条记录", len(records))

    logger.debug("序列化为 %s 格式...", to_format)
    result = SERIALIZERS[to_format](records)
    return result


def convert_file(input_path: str, output_path: str, to_format: str) -> None:
    """转换单个文件."""
    from_format = detect_format(input_path)
    if from_format == "unknown":
        logger.warning("无法检测文件格式: %s，跳过", input_path)
        return

    logger.info("转换: %s (%s) -> %s (%s)", input_path, from_format, output_path, to_format)
    content = read_file(input_path)
    result = convert(content, from_format, to_format)
    write_file(output_path, result)
    logger.info("完成: %s", output_path)


# ===== CLI =====


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文件格式转换器")
    parser.add_argument("--input", "-i", help="输入文件或目录")
    parser.add_argument("--output", "-o", help="输出文件或目录")
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json", "yaml"],
        default="json",
        help="目标格式 (默认: json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.input:
        logger.error("请指定 --input 参数")
        sys.exit(1)

    input_path = Path(args.input)
    to_format = args.format

    if input_path.is_file():
        # 单文件转换
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_suffix(f".{to_format}"))
        convert_file(str(input_path), output_path, to_format)
    elif input_path.is_dir():
        # 目录批量转换
        output_dir = args.output or "output"
        ensure_dir(output_dir)
        for f in input_path.iterdir():
            if f.is_file() and detect_format(str(f)) != "unknown":
                out_name = f.stem + f".{to_format}"
                out_path = os.path.join(output_dir, out_name)
                convert_file(str(f), out_path, to_format)
    else:
        logger.error("输入路径不存在: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
