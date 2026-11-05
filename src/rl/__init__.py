"""备用 RL 包；当前训练入口只执行 SFT。"""


def load_method(name):
    if name == "ppo":
        from rl.ppo import UNAVAILABLE_REASON
    elif name == "grpo":
        from rl.grpo import UNAVAILABLE_REASON
    else:
        raise ValueError(f"未知 RL 算法: {name}")
    raise NotImplementedError(UNAVAILABLE_REASON)
