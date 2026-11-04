"""云端生成、本地保存逐条 case；推理不加载本地模型。"""

from __future__ import annotations

from rlflow.experiment import append_record
from rlflow.remote import check_model


def sample_cases(cfg: dict, rows: list[list[dict]], run_dir, service, model_path=None) -> None:
    import pytrio as trio

    model = cfg["model"]["base_model"]
    check_model(service, model)
    sampler = service.create_sampling_client(base_model=model, model_path=model_path)
    tokenizer = sampler.get_tokenizer()
    params = trio.SamplingParams(seed=cfg["seed"], **cfg["sampling"])
    for index, messages in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        response = sampler.sample(
            prompt=trio.ModelInput.from_ints(prompt),
            sampling_params=params,
            num_samples=1,
        ).result()
        sequence = response.sequences[0]
        record = {
            "case_id": index,
            "base_model": model,
            "model_path": model_path,
            "messages": messages[:-1],
            "reference": messages[-1]["content"],
            "response": sequence.text,
            "tokens": list(sequence.tokens),
        }
        append_record(run_dir / "cases.jsonl", record)
        print(f"case={index}: {sequence.text}", flush=True)
