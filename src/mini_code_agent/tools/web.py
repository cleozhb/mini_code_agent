"""Web 工具：搜索和获取网页内容."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .base import PermissionLevel, Tool, ToolResult


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=5, description="返回结果数量上限")


class WebFetchInput(BaseModel):
    url: str = Field(description="要获取内容的 URL")
    max_length: int = Field(default=5000, description="返回内容最大字符数")


@dataclass
class WebSearchTool(Tool):
    """搜索网络内容，返回相关结果摘要."""

    InputModel: ClassVar[type[BaseModel]] = WebSearchInput

    name: str = "WebSearch"
    description: str = "搜索网络获取相关信息。返回搜索结果标题、URL 和摘要。"
    permission_level: PermissionLevel = PermissionLevel.CONFIRM

    async def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        max_results: int = kwargs.get("max_results", 5)

        try:
            import httpx
        except ImportError:
            return ToolResult(
                output="",
                error="httpx 未安装，无法执行搜索。请 `uv add httpx`。",
            )

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return ToolResult(output="", error=f"搜索请求失败: {e}")

        results: list[str] = []
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if "Text" in topic:
                url = topic.get("FirstURL", "")
                results.append(f"- {topic['Text']}\n  {url}")

        if not results:
            abstract = data.get("AbstractText", "")
            if abstract:
                results.append(abstract)

        output = "\n\n".join(results) if results else "未找到相关结果。"
        return ToolResult(output=output)


@dataclass
class WebFetchTool(Tool):
    """获取指定 URL 的网页内容，转为纯文本返回."""

    InputModel: ClassVar[type[BaseModel]] = WebFetchInput

    name: str = "WebFetch"
    description: str = "获取指定 URL 的网页内容。将 HTML 转为纯文本或 Markdown 返回。"
    permission_level: PermissionLevel = PermissionLevel.CONFIRM

    async def execute(self, **kwargs: Any) -> ToolResult:
        url: str = kwargs["url"]
        max_length: int = kwargs.get("max_length", 5000)

        try:
            import httpx
        except ImportError:
            return ToolResult(output="", error="httpx 未安装。请 `uv add httpx`。")

        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                raw = resp.text
        except Exception as e:
            return ToolResult(output="", error=f"获取 URL 失败: {e}")

        if "html" in content_type:
            text = self._html_to_text(raw)
        else:
            text = raw

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[...截断，总长 {len(raw)} 字符]"

        return ToolResult(output=text)

    @staticmethod
    def _html_to_text(html: str) -> str:
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
