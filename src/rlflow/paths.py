"""项目路径的唯一真相来源。

所有代码都要通过这里拿路径，不要写死绝对路径、也不要用相对 CWD 的路径，
否则在 notebooks/ 和 scripts/ 下运行会得到不同的结果。

可用环境变量覆盖（.env 或 shell 中设置）：
    RLFLOW_DATA_DIR     默认 <repo>/data
    RLFLOW_MODELS_DIR   默认 <repo>/models
    RLFLOW_OUTPUTS_DIR  默认 <repo>/outputs
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# .env 是可选的本地覆盖；已存在的 shell 环境变量优先。
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _dir(env_var: str, default: str) -> Path:
    return Path(os.environ.get(env_var, PROJECT_ROOT / default)).expanduser().resolve()


DATA_DIR = _dir("RLFLOW_DATA_DIR", "data")
MODELS_DIR = _dir("RLFLOW_MODELS_DIR", "models")
OUTPUTS_DIR = _dir("RLFLOW_OUTPUTS_DIR", "outputs")
CONFIGS_DIR = PROJECT_ROOT / "configs"

RAW_DIR = DATA_DIR / "raw"        # 原始下载，只读
PROCESSED_DIR = DATA_DIR / "processed"  # 可重建的中间产物
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


def ensure_dirs() -> None:
    """创建运行时目录。空目录不进 git，所以需要用时才建。"""
    for d in (RAW_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
