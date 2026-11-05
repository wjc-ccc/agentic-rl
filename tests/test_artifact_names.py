"""名称使用覆盖后的真实参数，记录目录不会覆盖之前的运行。"""

from sft.config import apply_overrides
from sft.experiment import Experiment, parameter_name


def test_names_follow_effective_parameters(cfg):
    before = parameter_name(cfg)
    assert before.startswith("sft-lora-cloud-qwen3.5-4b-gsm1k-n24-lr1e-4-ep2-r8-bs1-ga1")
    apply_overrides(
        cfg,
        [
            "train.learning_rate=1.25e-4",
            "train.per_device_train_batch_size=2",
            "train.num_train_epochs=3",
        ],
    )
    changed = parameter_name(cfg)
    assert "lr1.25e-4-ep3-r8-bs2-ga1-len512-s42-cfg" in changed
    cfg["train"]["adam_beta1"] = 0.8
    assert parameter_name(cfg) != changed


def test_names_ignore_labels_and_machine_paths_but_include_data_hash(cfg):
    cfg["data"]["train_manifest"] = {"path": "machine-a", "sha256": "a" * 64}
    before = parameter_name(cfg)
    cfg["run_name"] = "stale-label"
    cfg["_config_path"] = "other-machine"
    cfg["model"]["cache_dir"] = "C:/different/cache"
    cfg["model"]["local_path"] = "C:/different/model"
    cfg["logging"]["mode"] = "offline"
    cfg["data"]["train_manifest"]["path"] = "machine-b"
    assert parameter_name(cfg) == before
    cfg["data"]["train_manifest"]["sha256"] = "b" * 64
    assert parameter_name(cfg) != before


def test_repeated_runs_and_backend_switch_do_not_share_artifacts(cfg):
    first, second = Experiment(cfg), Experiment(cfg)
    assert first.path != second.path and first.checkpoint_root != second.checkpoint_root
    cfg["execution"]["backend"] = "local"
    assert "-local-" in parameter_name(cfg)
