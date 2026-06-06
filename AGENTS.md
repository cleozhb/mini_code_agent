# mini_code_agent

一个用 Python 从零构建的编程 Agent，学习项目。

## 技术栈
- Python 3.12+
- uv 管理依赖和虚拟环境
- anthropic SDK + openai SDK（双模型支持）
- Rich（终端 UI）
- prompt_toolkit（输入处理）
- pytest + pytest-asyncio（测试）

## 项目结构
src/mini_code_agent/
  core/       — Agent 核心循环
  llm/        — LLM 客户端抽象
  tools/      — 工具定义和执行
  context/    — 上下文工程
  memory/     — 记忆系统
  safety/     — 安全控制
  cli/        — 命令行界面

## 开发命令
- uv run python main.py                  — 启动 Agent
- uv run pytest tests/ -xvs      — 跑测试
- uv add <package>                — 添加依赖
- uv add --dev <package>          — 添加开发依赖

## 编码规范
- 使用 type hints
- 用 dataclass 或 Pydantic 定义数据结构
- 每个模块有 __init__.py 导出公共接口
- 异步优先（async/await）
- 错误处理用自定义异常，不要裸 except
- 需要 LLM 返回结构化数据时，用 Pydantic model 定义 schema + `response_format` 约束输出，不要手动解析 JSON

## LLM 配置
- 所有配置从 .env 文件读取（不读系统环境变量），参考 .env.example
- `create_client("openai")` 默认走 OpenAI Chat Completions，兼容 DeepSeek 等 OpenAI 协议的服务
- `OPENAI_API_STYLE=responses` 或 `create_client("openai-responses")` 可切换到 OpenAI Responses backend
- `create_client("anthropic")` 走 Anthropic Messages backend，支持原生 structured outputs
