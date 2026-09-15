# 0911 在线 RL 配置

这里保存从 `Evo-RL-loop-0901` 迁移并按当前 LeRobot 配置结构重写的在线 RL 配置。
在线硬件端只使用 `lerobot.onlineRL_evoRL.actor_new`，learner 使用当前
`lerobot.rl.learner`，二者继续使用当前 gRPC 字节传输协议。

## 已迁移配置

- `actor/Actor_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json`
- `learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json`
- `actor/Actor_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json`
- `learner/Leanrer_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json`
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
| `online_updates_per_episode` | 当前 learner 连续更新，无同名字段 |
| `online_only_after_initialization` | 当前 `rlt_chunk` 固定使用 compact online replay |
| learner 的离线 `dataset` | 设为 `null`；离线 RLT 训练仍使用 `RL_train.sh` |
| `manual_exposure_us/manual_gain/white_balance_kelvin` | `exposure/gain/white_balance` |

旧 checkpoint 的配置仍标记为 `pi05`，但权重中含有 `model.rlt_module.*`。迁移 JSON 是
权威的新配置 schema；`actor_new` 和 learner 按 `pi05_rlt` 构造模型，再从旧目录读取
`model.safetensors` 与 processor 文件，因此不修改原 checkpoint。

## 启动

默认使用 0901-40k 的迁移配置：

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
- actor 的 `dataset.root` 用于 task metadata 校验；learner 为 online-only，保持 `dataset=null`。
- 首次运行应使用不存在的新 `output_dir`；已有 checkpoint 时按当前 resume 流程启动。
- 当前两套迁移配置都是单臂 7 维 cup-catch checkpoint。双臂运行需要 14 维、三相机且已经
  包含 RLT 权重的 `pi05_rlt` checkpoint，不能直接把普通双臂 PI05 checkpoint 当成在线 RLT 模型。
