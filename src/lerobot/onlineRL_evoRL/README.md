# onlineRL_evoRL 使用说明

`onlineRL_evoRL` 是当前 EvoRL 在线采集/训练逻辑的工作目录。当前实现支持两种 actor 行为：

- 默认 SAC actor：沿用在线 RL 的 SAC action 生成和 learner 参数同步。
- VLA actor：沿用 `scripts/RL_data.sh --policy.path` 的 VLA checkpoint 推理方式生成动作。

同时支持两种启动模式：

- online：actor 连接 learner，episode 结束后发送 transitions。
- actor-only：actor 不连接 learner，episode 结束后按 episode 保存本地数据并生成可视化页面。

## 运行环境

建议从项目根目录启动：

```bash
cd /home/yz/projects/Evo-RL-loop-0810
export PYTHONPATH=/home/yz/projects/Evo-RL-loop-0810/src:$PYTHONPATH
```

推荐使用当前项目环境：

```bash
/home/yz/projects/env/package_sorting_env/bin/python
```

## 配置文件

当前参考配置：

```bash
src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl.json
```

关键硬件配置：

- follower：`piper_follower`，`can1`
- leader：`piper_leader`，`can0`
- wrist/top OpenCV 相机：默认 `10` 和 `4`
- task：`Flip the package if the barcode is not facing up. Advantage: positive`
- episode 控制时长：`120s`

使用前按现场修改：

- `env.robot.cameras.wrist.index_or_path`
- `env.robot.cameras.top.index_or_path`
- `dataset.root`
- `output_dir`
- online 模式下的 `policy.actor_learner_config.learner_host`

## VLA Actor 配置

`actor_vla_policy.enabled=false` 时，actor 使用原 SAC policy 生成动作。

开启 VLA 推理：

```json
"actor_vla_policy": {
  "enabled": true,
  "policy_path": "/home/yz/projects/outputs/pi05_base_smovla_v3_0720/train/checkpoints/050000/pi05_base_smovla_v3_0720_50k",
  "policy_poll_s": 5.0,
  "reload_on_episode_boundary": true
}
```

VLA 加载逻辑与 `RL_data.sh --policy.path` 对齐：

```text
PreTrainedConfig.from_pretrained(policy_path)
make_policy(policy_cfg, ds_meta=LeRobotDatasetMetadata(...))
make_pre_post_processors(policy_cfg, pretrained_path=policy_path, dataset_stats=...)
predict_action(raw_robot_observation, task=env.task)
```

VLA checkpoint 目录需要包含类似文件：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

`reload_on_episode_boundary=true` 时，actor 会在 episode 边界按 `policy_poll_s` 检查 checkpoint 文件变化；发生变化后，下一个 episode 前重新加载 VLA。

## Online 模式

先启动 learner：

```bash
/home/yz/projects/env/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.learner \
  --config_path src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl.json
```

再启动 actor。默认 SAC actor：

```bash
/home/yz/projects/env/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.actor \
  --config_path src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl.json
```

Online + VLA actor：

```bash
/home/yz/projects/env/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.actor \
  --config_path src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl.json \
  --actor_vla_policy.enabled true \
  --actor_vla_policy.policy_path /home/yz/projects/outputs/pi05_base_smovla_v3_0720/train/checkpoints/050000/pi05_base_smovla_v3_0720_50k
```

Online + VLA actor 时，actor 仍连接 learner、仍发送 transitions；但 action selection 来自 VLA，不使用 learner 推送的 SAC actor 参数。

## Actor-only 模式

Actor-only 不连接 learner，不检查 learner 是否存在。配置：

```json
"actor_only": {
  "enabled": true,
  "episode_output_dir": "/home/yz/projects/outputs/piper_package_sorting_online_rl/actor_episodes",
  "save_episode_images": true,
  "save_episode_viewer": true
}
```

Actor-only + VLA：

```bash
/home/yz/projects/env/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.actor \
  --config_path src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl_1F.json \
  --actor_only.enabled true \
  --actor_vla_policy.enabled true \
  --actor_vla_policy.policy_path /home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_RLT_30K \
  --actor_only.episode_output_dir /home/hpc/yuzhang/outputs/online_rl_outbox/pi05_base_smovla_v3_0720_RLT_30K/actor_episodes
```

保存结构：

```text
actor_episodes/
  episode_000000/
    metadata.json
    transitions.pt
    frames.json
    images/
      observation.images.top/
      observation.images.wrist/
    viewer.html
```

打开某个 episode 的 `viewer.html` 可查看：

- top/wrist 两路相机
- task 指令
- reward、done、truncated
- action 和 state 摘要
- success/failure
- intervention 状态
- VLA policy path 和 checkpoint fingerprint

## 热键和 Episode 规则

当前 actor 内置 HIL 热键，不依赖 `event_config.json` 的质量事件配置。

- `I`：切换人工接管。默认 policy/VLA 控制；按一次进入人工接管，再按一次释放回 policy/VLA。
- `S`：标记当前 episode 成功并结束。最后一条 transition 写入 `reward=1.0, done=true`。
- `F`：标记当前 episode 失败并结束。最后一条 transition 写入 `reward=0.0, done=true`。
- 左箭头：放弃当前 episode 并重录，不发送、不保存为有效 episode，双臂保持当前位置。
- `R`：放弃当前 episode，双臂回默认初始位后重录，不发送、不保存为有效 episode。
- `Esc`：停止 actor。

`S/F/左箭头` 结束后保持双臂当前位置；`R` 结束后双臂回初始位。

## Reward 设计

当前 reward 是稀疏终止奖励：

- 普通 step：使用环境/processor 当前 reward，通常为 `0.0`。
- `S`：覆盖最后一步为 `reward=1.0, done=true`。
- `F`：覆盖最后一步为 `reward=0.0, done=true`。
- timeout：`truncated=true`，reward 保持当前值。
- 左箭头 / `R`：当前 episode 丢弃，不进入 learner 或 actor-only 有效保存。

每条 transition 的 `complementary_info` 包含：

```text
discrete_penalty
is_intervention
intervention_state
success
failure
actor_policy_is_vla
```

当前配置 `policy.num_discrete_actions=null` 时，`discrete_penalty` 不参与 SAC 主损失。

## 数据流

Online SAC actor：

```text
processed observation -> SACPolicy.select_action -> action processor -> env.step -> transition -> learner
```

Online VLA actor：

```text
raw robot observation + env.task -> VLA predict_action -> action processor -> env.step -> transition -> learner
```

Actor-only VLA：

```text
raw robot observation + env.task -> VLA predict_action -> action processor -> env.step -> transition -> local episode files
```

## 验证命令

语法检查：

```bash
/home/yz/projects/env/package_sorting_env/bin/python -m compileall -q \
  /home/yz/projects/Evo-RL-loop-0810/src/lerobot/onlineRL_evoRL \
  /home/yz/projects/Evo-RL-loop-0810/src/lerobot/configs/train.py
```

配置解析：

```bash
/home/yz/projects/env/package_sorting_env/bin/python - <<'PY'
from lerobot.onlineRL_evoRL import actor  # noqa: F401
from lerobot.configs.train import TrainRLServerPipelineConfig
cfg = TrainRLServerPipelineConfig.from_pretrained(
    '/home/yz/projects/Evo-RL-loop-0810/src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl.json'
)
cfg.validate()
print(cfg.actor_vla_policy)
print(cfg.actor_only)
PY
```

## 日志

日志写入：

```text
${output_dir}/logs/
```

常见文件：

- `learner_${job_name}.log`
- `actor_${job_name}.log`
- 多进程模式下的 `actor_policy_*.log`、`actor_transitions_*.log`、`actor_interactions_*.log`

## 注意事项

- `actor_vla_policy.enabled=true` 只替换 actor action 生成，不训练 VLA 权重。
- online + VLA actor 下，learner 仍训练 SAC；actor 采样动作来自 VLA。
- actor-only 模式不会向 learner 发送数据。
- VLA 推理依赖 `dataset.root` 里的 metadata/stats，确保该路径可读取并包含两路相机和 action/state 统计。
