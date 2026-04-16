"""工厂函数：按 provider 名称创建对应的 LLM 客户端."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values

from .base import LLMClient, LLMError

# 从 .env 文件直接读取配置（不污染 os.environ）
_env_path = Path(__file__).resolve().parents[3] / ".env"
_config = dotenv_values(_env_path)


def create_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """创建 LLM 客户端.

    优先级: 显式参数 > .env 文件配置 > 硬编码默认值

    Args:
        provider: "anthropic" | "openai"
        model: 模型名称，为 None 时从 .env 读取
        api_key: API Key，为 None 时从 .env 读取
        base_url: API Base URL，为 None 时从 .env 读取

    Returns:
        对应的 LLMClient 实例

    Raises:
        LLMError: 不支持的 provider
    """
    provider = provider.lower()

    if provider == "anthropic":
        from .claude_client import ClaudeClient

        return ClaudeClient(
            model=model or _config.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            api_key=api_key or _config.get("ANTHROPIC_API_KEY"),
            base_url=base_url or _config.get("ANTHROPIC_BASE_URL"),
        )

    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(
            model=model or _config.get("OPENAI_MODEL", "gpt-4o"),
            api_key=api_key or _config.get("OPENAI_API_KEY"),
            base_url=base_url or _config.get("OPENAI_BASE_URL"),
        )

    raise LLMError(f"不支持的 provider: {provider!r}，可选: anthropic, openai")
