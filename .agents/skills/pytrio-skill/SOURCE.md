# 项目级 PyTRIO skill

- 上游：https://github.com/SwanHubX/pytrio-skill
- Skill 版本：0.2.2
- 固定 commit：`cea3edbe1d6a8fbdb61c2e2ab3975b7806e1fa23`
- 原始路径：`skills/pytrio-skill/`
- 安装位置：本仓库 `.agents/skills/pytrio-skill/`，不涉及全局安装。
- `SKILL.md`、`references/`、`examples/` 保持上游原样；本文件为项目补充。

项目适用规则：

1. 一切 Python 安装与执行都使用 `uv`；上游示例里的 `pip` 命令在此项目改用 `uv add` / `uv sync`。
2. 默认本地编排、PyTRIO 云端模型计算。SwanLab 保持 offline，上传由用户显式开启。
3. 不执行 skill 示例的顶层训练代码来验证安装；示例会发起真实的云端任务。
4. 使用项目锁定的 SDK，并读取当前任务对应的 reference；不将示例模型名当作服务支持承诺。
5. 更新时仍安装到项目目录并固定 commit，保留上游许可证；不要使用 `-g`。

安装由 Codex skill-installer 的 `--dest <repo>/.agents/skills` 完成。
其他支持 Agent Skills 的工具可在仓库根目录运行：

```powershell
npx skills add SwanHubX/pytrio-skill -y
```

`-g` 表示用户级安装，因此不适用于本项目的要求。
