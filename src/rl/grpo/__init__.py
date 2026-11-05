"""GRPO Interface 与状态：未实现，不提供可执行 train。

数据：rl.data.Prompt（messages 以 user 结束，answer 可选）。未来需要 RL 自己的模型 client，
不得调用 SFT client。对同 prompt 生成多个候选，任务 reward(prompt, generation) -> float。
同组 reward 减去均值得到 advantage；全相同时跳过。未来 RL 自己的训练输入应满足：
input=prompt_tokens+completion_tokens[:-1]；target=[0]*(prompt_len-1)+completion_tokens；
logprobs/advantages 的 prompt 区间为 0，completion 区间用采样时 old_logprobs 和 advantage。
使用 importance_sampling，不能套用 SFT 的 cross_entropy；当前未提供对应 client 或训练循环。
RL 自己的参数、日志和模型接口待实际进入该阶段再实现；备用产物位置由 rl.paths 独立定义。
"""

UNAVAILABLE_REASON = "GRPO 训练尚未实现；需要具体 reward、同 prompt 多候选 rollout 与分组更新"
