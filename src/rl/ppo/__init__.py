"""PPO Interface 与状态：未实现，不提供可执行 train。

经典环境路径：cfg.env -> reset/step -> observation/action/reward/terminated/truncated；
之后需要 policy/value/old_logprob/advantage，当前仅保留 RL 自己的备用 CartPole YAML。
LLM PPO 未来还需要 reward/value/reference/KL；不能直接把 GRPO 的 group advantage 当 PPO。
实现后应遵循 train(cfg, data, experiment, client) Interface，按真实 optimizer step 保存产物。
"""

UNAVAILABLE_REASON = "PPO 训练尚未实现；RL 仅作为备用，不提供执行入口"
