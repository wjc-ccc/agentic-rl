"""LLM RL 数据 Interface：prompt 与参考信息，不把 assistant 答案放进采样 prompt。

data/processed/<数据集>/<版本>/train.jsonl 一条示例：
{"messages": [{"role": "user", "content": "1+1?"}], "answer": "2"}
answer 可省略（由任务的 reward 决定）；这里只加载数据，尚无 PPO/GRPO 训练实现。
经典环境 PPO 使用 env.reset/step 的 observation/action/reward，不使用此 JSONL 格式。
"""

import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Prompt(TypedDict):
    messages: list[Message]
    answer: NotRequired[str]


class Rollout(TypedDict):
    """未来算法的输入：同一 prompt 的候选分组；old_logprobs 必须来自采样时的策略。

    例：prompt_tokens=[11,12], completion_tokens=[21,99], old_logprobs=[-0.2,-0.1],
    text="2", reward=1.0；completion tokens 与 old_logprobs 等长。
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    old_logprobs: list[float]
    text: str
    reward: float


def load_prompts(path: Path) -> list[Prompt]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("RL prompt 数据不能为空")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("RL JSONL 每行必须为对象")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("RL messages 必须为非空列表")
        for msg in messages:
            if (
                not isinstance(msg, dict)
                or msg.get("role") not in {"system", "user", "assistant"}
                or not isinstance(msg.get("content"), str)
                or not msg["content"].strip()
            ):
                raise ValueError("RL messages 必须包含有效 role 和非空 content")
        if messages[-1]["role"] != "user":
            raise ValueError("RL prompt 必须以 user 结束，不包含待生成的 assistant 答案")
        if "answer" in row and not isinstance(row["answer"], str):
            raise ValueError("RL answer 必须为文本")
    return rows
