"""备用 RL 自己的配置路径；不读取 SFT 参数或启动运行时。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "rl"


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
