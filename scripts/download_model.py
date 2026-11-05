"""按配置下载模型；cloud 只用 --tokenizer-only，HF_ENDPOINT 可选择 HF 镜像。"""

import argparse

from sft.config import load_config
from sft.paths import project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["execution"]["backend"] != "local" and not args.tokenizer_only:
        raise ValueError("cloud 不需要下载基模，请用 --tokenizer-only")
    key = "tokenizer_path" if args.tokenizer_only else "local_path"
    target = str(project_path(cfg["model"][key]))
    patterns = (
        [
            "tokenizer*",
            "vocab.json",
            "merges.txt",
            "chat_template*",
            "special_tokens_map.json",
            "config.json",
            "added_tokens.json",
        ]
        if args.tokenizer_only
        else ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
    )
    if cfg["model"]["download_source"] == "modelscope":
        from modelscope import snapshot_download

        print(
            snapshot_download(
                cfg["model"]["base_model"],
                local_dir=target,
                allow_patterns=patterns,
            )
        )
    else:
        from huggingface_hub import snapshot_download

        print(
            snapshot_download(
                cfg["model"]["base_model"],
                local_dir=target,
                allow_patterns=patterns,
            )
        )
