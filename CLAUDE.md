# CLAUDE.md

本项目的完整工作契约在 `@AGENTS.md`，Claude Code 启动时自动加载本文件并展开该导入。

快速命令：

- 同步环境：`uv sync`（需要 LLM 栈时 `uv sync --extra llm`）
- 运行：`uv run python <script>`
- 测试：`uv run pytest`
- 新依赖：`uv add <pkg>`

记住：data / models / outputs / .venv 永不提交；日志用 SwanLab；GPU 只有 6 GB。
