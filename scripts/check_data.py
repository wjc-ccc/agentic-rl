"""只读核验当前 SFT DATA_DIR 下 raw/processed 与 manifest，RL 数据暂不接入。"""

from sft.registry import check_manifest

if __name__ == "__main__":
    errors = check_manifest()
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("数据登记核验通过：raw/processed 无未登记、缺失或校验不一致的文件。")
