"""固定 SFT case 的模型 Interface 调用与本地保存；不会把参考答案发送进 prompt。

输入与 sft.data 相同；输出写 outputs/<run_id>/cases.jsonl，一条示例：
{"case_id": 0, "messages": [{"role": "user", "content": "1+1?"}],
"reference": "2", "response": "2", "tokens": [21, 99], "model_path": "<sampler URI>"}。
model_path=None 表示基模；不同采样的指标与数据 hash 由 scripts/sample.py 单独登记。
"""

from pathlib import Path

from sft.experiment import append_record
from sft.files import Message
from sft.model_client import PyTrioClient


def sample_cases(
    cfg: dict,
    rows: list[list[Message]],
    run_dir: Path,
    client: PyTrioClient,
    model_path: str | None = None,
    step: int | None = None,
) -> None:
    for index, messages in enumerate(rows):
        prompt = client.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **cfg["model"]["chat_template_kwargs"],
        )
        sequence = client.sample(prompt, model_path=model_path)[0]
        append_record(
            run_dir / "cases.jsonl",
            {
                "step": step,
                "case_id": index,
                "base_model": cfg["model"]["base_model"],
                "model_path": model_path,
                "messages": messages[:-1],
                "reference": messages[-1]["content"],
                "response": sequence.text,
                "tokens": sequence.tokens,
            },
        )
        print(f"case={index}: {sequence.text}", flush=True)
