# 0911 在线 RL 配置

这里保存从 `Evo-RL-loop-0901` 迁移并按当前 LeRobot 配置结构重写的在线 RL 配置。
在线硬件端只使用 `lerobot.onlineRL_evoRL.actor_new`，learner 只从
`lerobot.onlineRL_evoRL.learner` 启动；两者复用当前 gRPC 字节传输协议。

## 已迁移配置

- `actor/Actor_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json`
- `learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json`
- `actor/Actor_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json`
- `learner/Leanrer_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json`
- `actor/Actor_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json`
- `learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json`
- `hotkeys/piper_cup_catch_task_hotkeys.json`

0901 的三个 `actor_mode=vla_only` / Actor-only 配置没有复制为可执行配置。0911 已停用
该入口；纯 VLA 数据采集由 `scripts/RL_data.sh` 和 `scripts/RL_data_bimanual.sh` 负责。

## 关键字段迁移

| 0901 字段 | 0911 字段或处理 |
| --- | --- |
| `policy.type=pi05_online_rl` | `policy.type=pi05_rlt` + `algorithm.type=rlt_chunk` |
| policy 内 Actor/Critic 超参数 | 移到 `algorithm` |
| `policy.tokenizer_name` | `policy.text_tokenizer_name` |
| `policy.storage_device` | `algorithm.storage_device` |
| `policy.actor_learner_config` | `algorithm.actor_learner_config` |
| `policy.concurrency` | `algorithm.concurrency` |
| `online_updates_per_episode` | actor/learner 各自 JSON 顶层 `gpu_handoff.updates_per_episode` |
| `online_only_after_initialization` | `offline_pretraining` 只在启动时运行，完成后在线 replay 仍只含在线数据 |
| learner 的离线 `dataset` | 始终为 `null`；原始 LeRobotDataset 只交给独立提取脚本，learner 读取 `compact_dataset_path` |
| `manual_exposure_us/manual_gain/white_balance_kelvin` | `exposure/gain/white_balance` |

旧 checkpoint 的配置仍标记为 `pi05`，但权重中含有 `model.rlt_module.*`。迁移 JSON 是
权威的新配置 schema；`actor_new` 按 `pi05_rlt` 从旧目录读取 `model.safetensors` 与
processor 文件。Learner 永远只构造 RLT heads，不加载 5B；独立的
`extract_offline_features` 程序负责在本地或云端加载 5B 并导出 compact episode。


## Actor 双格式本地保存

在线 Actor 总是先在 `online_transition.episode_output_dir` 保存 compact transition，保存成功后才把同一个 payload 发给 Learner；`save_local_copy` 因此必须为 `true`。

标准 LeRobotDataset 是可选的第二份副本：

- `save_lerobot_copy=false`：只保存并发送 compact transition；
- `save_lerobot_copy=true`：另外把同一 accepted episode 写到 `lerobot_output_dir`；
- 两个输出目录必须不同；LeRobotDataset 目录为防覆盖要求首次运行时不存在。

当前 0915 双臂配置已将 `save_lerobot_copy` 设为 `true`，所以两种格式都会保存在本地。

## 启动

默认使用 0915 双臂 Piper 的 PI05-RLT 配对配置：

```bash
source /home/lenovo/code/envs/evo_0911/bin/activate
cd /home/lenovo/code/Evo-RL-loop-0911
bash scripts/RL_online.sh learner
bash scripts/RL_online.sh actor
```

同机启动：

```bash
bash scripts/RL_online.sh both \
  --can0.control=true \
  --rtc.enabled=true \
  --rtc.mode=guided
```

切换到 35k 配置：

```bash
export PIPER_ONLINE_RL_ACTOR_CONFIG=src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json
export PIPER_ONLINE_RL_LEARNER_CONFIG=src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json
bash scripts/RL_online.sh both
```

`both` 后面的覆盖参数只传给 actor；单独运行 `learner` 时，其后参数会传给 learner。
`PIPER_ONLINE_RL_CONFIG` 仍可作为让两端使用同一文件的旧兼容变量，但 `rlt_chunk` 推荐使用
这里拆分后的 actor/learner 配置。

## 使用前检查

- actor 与 learner 的 `policy.pretrained_path`、RLT 维度、chunk size 必须一致。
- actor 与 learner 的 `algorithm.actor_learner_config` host/port 必须一致。
- `gpu_handoff` 只写在各自的 actor/learner JSON 顶层；两端 `enabled`、`update_quota_threshold` 和 `updates_per_episode` 必须相同，threshold 可以不是 200。
- learner 的 `dataset` 必须始终为 `null`；离线阶段从 `offline_pretraining.compact_dataset_path` 读取 Actor-format 文件。
- 当前 0915 learner 离线训练 20000 step；特征提取进度由独立脚本显示，learner 显示 compact 加载和 Actor/Critic 更新进度。
- actor 的 compact 本地副本必开；启用 LeRobot 副本时，两个输出目录必须不同。
- 首次运行应使用不存在的新 `output_dir`；已有 checkpoint 时按当前 resume 流程启动。
- 两套旧迁移配置是单臂 7 维 cup-catch checkpoint，不能用于当前双臂任务。
- 当前 0915 配置要求 14 维状态/动作、三路相机，以及包含 RLT 权重的 `pi05_rlt` checkpoint。
