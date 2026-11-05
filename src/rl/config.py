"""RL 独立备用配置；不复用 SFT 参数或加载器。"""

from pathlib import Path

import yaml

from rl.paths import project_path


def load_config(path: str | Path) -> dict:
    path = project_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["stage"] != "rl":
        raise ValueError("RL 入口不能加载 SFT 配置")
    cfg["_config_path"] = str(path)
    return cfg
