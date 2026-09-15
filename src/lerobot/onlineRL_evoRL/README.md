# onlineRL_evoRL 使用说明

当前 0911 融合版只使用 `actor_new.py` 作为在线 Actor 入口；`scripts/RL_online.sh actor/both`
也只会启动 `lerobot.onlineRL_evoRL.actor_new`。`actor.py` 与 `actor_old.py` 仅为历史兼容文件，
不承载新的控制功能。learner 和 gRPC 字节分块协议继续使用当前 LeRobot 0.6.1 实现。

当前有效模式固定为：

- `actor_mode=online_actor`
- `save_format=transition`
- `algorithm.type=rlt_chunk`
- `actor_vla_policy.policy_path` 指向 PI05-RLT 模型；learner 只更新独立 Actor head。

纯 VLA 或 VLA+人工录制继续使用 `RL_data.sh` / `RL_data_bimanual.sh`，不再由
`actor_new` 的 `vla_only` 分支执行。

在线 Actor 支持单臂/双臂、可选主臂跟随与人工接管、guided RTC、Rerun、当前 compact
transition 保存以及 episode 结束后的保持/回初始位逻辑。

## 运行环境

建议从项目根目录启动：

```bash
cd /home/lenovo/code/Evo-RL-loop-0911
export PYTHONPATH=/home/lenovo/code/Evo-RL-loop-0911/src:$PYTHONPATH
```

推荐使用当前项目环境：

```bash
source /home/lenovo/code/envs/evo_0911/bin/activate
```

## 配置文件

当前默认配置分为 actor 和 learner 两份：

```bash
src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json
src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json
```

完整迁移映射见 `src/lerobot/onlineRL_evoRL/configs/README.md`。

关键硬件配置：

- follower：`piper_follower`，`can1`
- leader：`piper_leader`，`can0`
- wrist/top RealSense：默认 `260422275792` 和 `261822303677`
- task：`Grab the left cup`，可用 `1/2/3` 切换
- episode 控制时长：`120s`

使用前按现场修改：

- `env.robot.cameras.*.serial_number_or_name`
- `dataset.root`
- `output_dir`
- online 模式下的 `algorithm.actor_learner_config.learner_host/learner_port`

## Learner：PI05 Online RL

PI05 learner 冻结 PI0.5/RLT，只训练 action-chunk Actor 和 twin-Q Critic。推荐配置：

`src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json`

启动命令必须使用带等号的配置参数：

```bash
source /home/lenovo/code/envs/evo_0911/bin/activate
cd /home/lenovo/code/Evo-RL-loop-0911
python -m lerobot.rl.learner \
  --config_path=src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json

source /home/lenovo/code/envs/evo_0911/bin/activate
cd /home/lenovo/code/Evo-RL-loop-0911
mkdir /home/lenovo/datasets/online_rl_outbox/logs/pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k
nohup env PYTHONUNBUFFERED=1 python -m lerobot.rl.learner \
    --config_path=src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json \
    > /home/lenovo/datasets/online_rl_outbox/logs/pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k/Leanrer_onlineRL_transition.log 2>&1 < /dev/null &
```

断点重启
```bash
source /home/lenovo/code/envs/evo_0911/bin/activate
cd /home/lenovo/code/Evo-RL-loop-0911
python -m lerobot.rl.learner \
  --config_path=/home/lenovo/datasets/online_rl_outbox/pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k_0911/learner/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  >> /home/lenovo/datasets/online_rl_outbox/logs/pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k/Leanrer_onlineRL_transition.log 2>&1 < /dev/null &

# rlt_chunk learner 只消费 actor 发送的 compact online replay。
# 离线 RLT 训练继续使用 scripts/RL_train.sh。
```


配置文件是基准值；需要临时实验时，可在命令行用同名参数覆盖。参数统一使用 `--参数=值`，布尔值使用小写 `true/false`：

```bash
python -m lerobot.rl.learner \
  --config_path=src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json \
  --steps=1000 \
  --algorithm.actor_update_interval=4 \
  --algorithm.online_step_before_learning=100 \
  --algorithm.actor_learner_config.learner_port=50051 \
  --batch_size=16 \
  --num_workers=0
```

命令行覆盖只适合一次性实验；稳定参数应写回 learner-only JSON，便于 checkpoint 保存完整配置和复现实验。

### 0901 历史 learner 参数说明（不可直接用于 0911）

> 本节保留用于理解旧实验记录，其中 `policy.online_updates_per_episode`、
> `policy.online_only_after_initialization` 和顶层 `gradient_accumulation_steps` 已不属于当前
> `rlt_chunk` 配置。当前可执行字段以 `configs/README.md` 和迁移后的 learner JSON 为准。

| 配置 | 要求 |
| --- | --- |
| `policy.pretrained_path` | 本地 PI0.5+RLT checkpoint；权重必须包含 RLT。 |
| `policy.tokenizer_name` | 本地 PaliGemma tokenizer 目录，离线运行时不要填写远程模型名。 |
| `dataset.root` | LeRobotDataset 根目录，必须有 data、videos、`meta/tasks.parquet`、`meta/episodes` 和 stats。 |
| `output_dir` | learner checkpoint 和日志目录。 |
| `policy.dtype` | 当前只支持 `float32` 或 `bfloat16`。 |

PI05 learner 必须有离线 dataset。learner-only 配置不需要 `env`、机器人、teleop、相机端口、`actor_vla_policy` 或 `actor_only`。

### task 来源

learner 不接受外部统一 task，也不读取 `env.task`。

`LeRobotDataset.__getitem__` 根据每一帧的 `task_index` 查询 `meta/tasks.parquet`，生成对应的 `batch["task"]`。因此三任务数据可以混合训练，同一个 batch 中的每个样本使用自己的 task 文本。

在线 actor 已有的 episode metadata 会携带 task；learner 将它与同一 episode 的 transitions 配对，并且只接受出现在离线 dataset `meta/tasks.parquet` 中的 task，未知 task 会直接报错。

### reward 与人工介入

离线 parquet 不需要预先存在 `next.reward`。learner 从 episode metadata 动态构造稀疏 reward：

- 非末帧：`reward=0`
- `episode_success=success` 的末帧：`reward=1, done=true`
- `episode_success=failure` 的末帧：`reward=0, done=true`
- 缺失或非法 `episode_success`：启动失败

人工介入严格按逐帧来源判断：

```text
complementary_info.collector_policy_id == "human"
```

为 true 时，Actor BC target 使用数据集实际 action；否则使用冻结 VLA 的 reference action。不再使用 `complementary_info.is_intervention`，也没有可覆盖该语义的 `offline_intervention_field` 参数。

### 数据加载与内存

离线训练沿用标准 `lerobot-train` 的 DataLoader 模式：

1. DataLoader 按 `batch_size` 惰性解码视频，不再把全部帧转换成 Python transition list。
2. action 查询 `[0, chunk_size)`，observation 查询当前时刻和 `+chunk_size` 两个时刻。
3. 根据 episode 边界生成 reward、done、intervention、valid mask 和 bootstrap mask。
4. 当前/下一 observation 批量提取冻结 PI0.5/RLT 特征。
5. 离线初始化阶段使用全离线 batch；收到在线 episode 后，在线阶段按配置使用纯在线或 online/offline 混合 batch。

内存上限主要由 `batch_size × num_workers × prefetch_factor`、模型和 online replay 决定，不再随离线数据集总图像数线性增长。离线 replay 不再创建，也不会写入 checkpoint。

### 两阶段训练流程

1. learner 启动后先执行 `steps` 次离线初始化更新，同时可接收并缓存在线 episode。
2. 离线初始化完成后不自动退出，也不继续空转训练；没有在线 episode 时只等待。
3. 每接收并校验一个完整 compact episode（旧 raw 格式仍与 metadata 配对），就向累计额度增加 `policy.online_updates_per_episode` 次更新。多个 episode 会按 FIFO 配对，更新额度可以累加。
4. 在线 replay 累积保留所有已接收的在线 transition。`online_only_after_initialization=false` 时每个在线 batch 各取一半 online 和 offline；设为 `true` 时完全不再采样原离线 dataset。
5. `Ctrl+C` 或终止信号会设置 shutdown event，停止等待并关闭 gRPC 与队列。

### 阶段与采样参数

| 参数 | 当前示例 | 用法与精确含义 |
| --- | ---: | --- |
| `steps` | `200` | **仅表示离线初始化更新次数，不是总训练步数，也不是退出条件。** 达到该值后 learner 保持运行，等待在线 episode。 |
| `policy.online_updates_per_episode` | `4` | 每接收一个完整在线 episode 增加 4 次有效 optimizer update；当前每 2 个 episode 同步一次，因此一批执行 8 次 Critic 更新。 |
| `policy.online_only_after_initialization` | `false` | `false`：在线阶段每个 micro batch 混合 online/offline；`true`：离线初始化结束后只采样 online replay。该参数不会跳过最初的 `steps` 次初始化。 |
| `batch_size` | `16` | 单次前后向的 micro batch。混合模式每个 micro batch 为 8 个 online 和 8 个 offline 样本。 |
| `gradient_accumulation_steps` | `16` | 累计 16 个 micro batch 后执行一次 optimizer step；单卡 effective batch 为 `16 × 16 = 256`。 |
| `num_workers` | `4` | 离线 DataLoader worker 数。显存或主存紧张时先减 `batch_size`，并相应增加累计次数以维持 effective batch。 |
| `policy.online_buffer_capacity` | `100000` | online compact feature replay 最多保存的 sliding-window transition 数，不是 episode 数。增大它主要增加 CPU 内存和 checkpoint 体积。 |
| `policy.actor_update_interval` | `4` | Critic 每次有效 step 都更新，Actor 每 4 次 Critic step 更新一次。 |
| `log_freq` / `save_freq` | `5 / 100` | 每多少个累计 `learner_step` 记录日志/保存 checkpoint；`save_freq` 仅在 `save_checkpoint=true` 时生效。 |

如果目标是对齐 RLinf 的每轮 8 次 Critic 更新，且保持当前每 2 个 episode 同步一次，设置：

```json
{
  "steps": 1000,
  "batch_size": 16,
  "gradient_accumulation_steps": 16,
  "policy": {
    "online_updates_per_episode": 4,
    "actor_update_interval": 4,
    "online_only_after_initialization": false
  }
}
```

只想让在线阶段使用新数据时，仅把 `online_only_after_initialization` 改为 `true`；离线 dataset 仍会用于启动时的 meta、task、stats 和前 `steps` 次初始化。

### 模型与优化参数

| 参数 | 当前示例 | 用法与精确含义 |
| --- | ---: | --- |
| `policy.pretrained_path` | 本地 checkpoint | 实际加载的 PI0.5+RLT 权重目录。该 checkpoint 必须与数据集的 observation/action features 兼容。 |
| `policy.use_rlt` | `true` | 启用并读取 RLT 特征；本 learner 依赖 `z_rl`，应保持为 `true`。 |
| `policy.base_policy_path` | 与 `pretrained_path` 相同 | 当前 PI05 Online RL learner 不单独读取此字段；保留是为了配置兼容。需要更换基座时必须修改 `pretrained_path`。 |
| `policy.chunk_size` / `policy.n_action_steps` | `50 / 50` | Actor 输出的完整 action chunk 长度和训练 horizon；当前实现要求二者严格相等。 |
| `policy.z_dim` | `2048` | 冻结 RLT 输出特征维度，必须与所加载 checkpoint 一致。 |
| `policy.proprio_dim` | `7` | 本体状态维度，必须与数据集和 checkpoint 一致。 |
| `policy.actor_hidden_dims` / `policy.critic_hidden_dims` | `[256,256,256]` | Actor 使用 Tanh 隐藏层；Critic 使用 Linear→LayerNorm→Tanh，均与 RLinf Stage 2 对齐。 |
| `policy.num_critics` | `2` | Q 网络数量；当前实现至少需要 2 个。 |
| `policy.fixed_std` | `0.002` | 固定高斯噪声加在 raw mean 上，再经过 Tanh；Actor loss、TD target 和非确定性在线 rollout 都实际使用。 |
| `policy.actor_rollout_deterministic` | `false` | `false` 用于在线训练采集并加入固定噪声；纯评估时设为 `true`。两种模式都不使用 reference dropout。 |
| `policy.reference_dropout_prob` | `0.5` | Actor 训练时丢弃冻结 VLA reference action 条件的概率，用于避免 Actor 只复制 reference；范围 `[0,1]`。 |
| `policy.q_weight` / `policy.bc_weight` | `0.1 / 5.0` | Actor loss 为 `-q_weight × Q + bc_weight × masked_BC`。增大前者更偏向奖励，增大后者更贴近人工动作/VLA reference。 |
| `policy.discount` | `0.96` | chunk 内 reward 折扣以及完整窗口 bootstrap 折扣，范围 `(0,1]`。 |
| `policy.critic_target_update_weight` | `0.005` | target Critic 的 Polyak 软更新系数；越小更新越平滑，范围 `(0,1]`。 |
| `policy.actor_lr` / `policy.critic_lr` | `3e-4 / 3e-4` | learner 实际使用的 Actor/Critic Adam 学习率。 |
| `policy.grad_clip_norm` | `10.0` | Actor 与 Critic 各自的梯度范数裁剪上限。 |

PI0.5/RLT 骨干在本 learner 中被冻结；日志里的 `bc_loss` 下降表示新 Actor 更接近 BC target，不表示 VLA 骨干正在更新。通用 `optimizer` 和 `scheduler` 仍不控制这里的两个 Adam optimizer；`gradient_accumulation_steps` 现已由 PI05 learner 使用。学习率和裁剪继续使用 `policy.actor_lr`、`policy.critic_lr` 与 `policy.grad_clip_norm`。

### 资源、服务与输出参数

| 参数 | 建议/当前示例 | 用法与精确含义 |
| --- | ---: | --- |
| `policy.device` | `cuda` | 冻结骨干、Actor/Critic 和计算 batch 所在设备。 |
| `policy.storage_device` | `cpu` | online replay 特征的存储设备。保持 `cpu` 可避免 replay 长期占用显存。 |
| `policy.offload_to_cpu_while_waiting` | `true`（默认 `false`） | 为 `true` 时，没有待执行的在线更新会将模型和 Adam 动量迁到 CPU，并在收到新数据后恢复到 `policy.device`。这会增加 CPU 内存占用和恢复训练延迟；CUDA context 仍可能保留少量显存。 |
| `policy.dtype` | `bfloat16` | 骨干计算精度；当前只接受 `bfloat16` 或 `float32`，不能写 `float16`。 |
| `dataset.streaming` | `false` | 当前 PI05 learner 强制要求 `false`。 |
| `policy.tokenizer_name` | 本地目录 | PaliGemma tokenizer 资产路径；离线机器必须提前准备本地文件。 |
| `policy.actor_learner_config.learner_port` | `50051` | learner gRPC 监听端口；必须与发送数据的一侧一致且未被占用。 |
| `policy.actor_learner_config.learner_host` | `127.0.0.1` | learner gRPC 的绑定地址。同机使用 `127.0.0.1`；跨机器可绑定实际网卡 IP 或 `0.0.0.0`，发送端需连接 learner 的可达 IP。 |
| `policy.actor_learner_config.queue_get_timeout` | `2.0` | 内部队列等待超时秒数；影响无数据等待和退出响应速度，不是网络训练超时。 |
| `output_dir` | 新目录 | 日志和 checkpoint 根目录。`resume=false` 时目录必须不存在。 |
| `resume` | `false` | 恢复时设为 `true`，并让配置指向已有 run/checkpoint；恢复会加载累计 step、optimizer 和已保存的 online replay。 |

`policy.actor_learner_config.policy_parameters_push_frequency` 当前不被 PI05 learner 使用，因为该路径尚未向 actor 部署新 Actor 参数。`policy.online_steps` 仅控制 actor 的环境交互上限，不控制 learner，因此 learner-only 配置有意省略它。

PI05 learner 同样不使用 `utd_ratio`、`offline_buffer_capacity`、`feature_extract_batch_size`、`offline_intervention_field` 或 `async_prefetch`。不要添加这些字段来调节当前 learner。日志中的阶段与计数器含义固定：

- `training_phase`：`offline_initialization` 或 `online`
- `learner_step`：离线和在线阶段累计的模型更新次数
- `interaction_step`：actor 环境交互
- `online_episode_count`：本次运行已接收的在线 episode 数
- `pending_online_update_steps`：尚未执行的在线更新额度

### Checkpoint

PI05 learner 只保存可训练的 Actor、Critic 和 target Critic，不重复保存冻结的 VLA/RLT 权重：

```text
checkpoints/<learner_step>/
  actor_critic.pt
  manifest.json
  pretrained_model/
    train_config.json
    config.json
  training_state/
  replay_online/       # 只有收到在线数据后才存在
checkpoints/last
```

`manifest.json` 同时记录基座路径、特征维度、RLinf-compatible Actor/噪声架构、effective batch 和 Critic:Actor 更新比。恢复及在线 Actor 加载会校验架构版本；旧 ReLU/post-tanh-noise Stage 2 checkpoint 会被拒绝。恢复时先从 `policy.pretrained_path` 加载冻结 VLA/RLT，再加载 `actor_critic.pt`。optimizer、`learner_step`、`interaction_step`、更新额度、episode 计数和 online replay 都会恢复。

离线 dataset 始终从 `dataset.root` 读取，不复制进 checkpoint。checkpoint 名称中的 step 是离线和在线阶段累计的 `learner_step`。

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

迁移配置固定 `reload_on_episode_boundary=false`。VLA/RLT 基座在一次在线运行中保持不变，
learner 只通过当前 gRPC 通道更新独立 Actor head。

## 统一 Actor 模式

启动入口固定为：

```bash
python -m lerobot.onlineRL_evoRL.actor_new \
  --config_path=src/lerobot/onlineRL_evoRL/configs/actor/<actor-config>.json
```

配置矩阵：

| `actor_mode` | `save_format` | learner 连接 | 动作来源 |
| --- | --- | --- | --- |
| `online_actor` | `transition` | 是 | PI05-RLT Online Actor；`V` 可切换 VLA/Actor |

`vla_only` 和 `online_actor + lerobot` 都会在 `actor_new` 启动前报错。

在线模式参考配置：

```bash
source /home/lenovo/code/envs/evo_0911/bin/activate
cd /home/lenovo/code/Evo-RL-loop-0911
python -m lerobot.onlineRL_evoRL.actor_new \
  --config_path src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json

source /home/lenovo/code/envs/evo_0911/bin/activate
cd /home/lenovo/code/Evo-RL-loop-0911
python -m lerobot.onlineRL_evoRL.actor_new \
  --config_path src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_base_cup_catch_v2_0819_35k.json
```

它还需要：`policy.type=pi05_rlt`、`algorithm.type=rlt_chunk`、与 learner 相同的模型/数据配置，以及一致的 learner host/port。`actor_checkpoint_path` 可选；缺失时先用 VLA，收到 learner 权重后自动切换 Online Actor。

按 task 键会丢弃当前 episode、切换 `env.task` 并复位。`V` 在已加载 Actor head 时切换 Online Actor/VLA，并清空 RTC/动作缓存；`B` 专用于成功标记。

### 可选 RTC 动作执行

RTC 默认关闭；不配置时保留原有同步 VLA 路径和 Online Actor 的 `_actor_actions` chunk 缓存。
开启后，VLA 和 Online Actor 的完整动作块都交给同一个 RTC 执行队列，逐拍取出的动作再经过
平滑、夹爪处理和 robot processor 后发送给机械臂。Online Actor 模式下不会再使用
`_actor_actions`，避免双重动作队列。

配置文件中可加入：

```json
"rtc": {
  "enabled": true,
  "mode": "guided",
  "execution_horizon": 25,
  "prefix_attention_schedule": "LINEAR",
  "max_guidance_weight": 10.0
},
"rtc_action_queue_threshold": 32
```

也可以仅在启动时覆盖：

```bash
python -m lerobot.onlineRL_evoRL.actor_new \
  --config_path=src/lerobot/onlineRL_evoRL/configs/actor/<actor-config>.json \
  --rtc.enabled=true \
  --rtc.mode=guided \
  --rtc.execution_horizon=25 \
  --rtc_action_queue_threshold=32
```

`rtc.enabled=false` 时顶层开关会明确禁用 checkpoint 中可能保存的 RTC 配置。开始人工接管时
会立即废弃 RTC 队列；任务切换、VLA/Actor 切换、episode 边界和关闭进程时，还会等待旧的
后台推理安全退出后再重置或换权重。

## 已停用的 Actor-only 模式（历史说明）

> 0911 当前版本已停用 `actor_new` 的 Actor-only/VLA-only 分支；本节命令不可执行，仅用于
> 辨认旧输出。纯 VLA 采集请使用 `RL_data.sh` 或 `RL_data_bimanual.sh`。

`actor_mode=vla_only` 不连接 learner，不检查 learner 是否存在。顶层 `save_format` 选择两种保存方式，默认是 `lerobot`；`actor_only` 仅保留输出目录、图片和 viewer 细节：

- `transition`：保存与在线发送完全相同的 compact episode，同时可生成 JSON、JPEG 和 HTML viewer；要求启用 PI05+RLT VLA。
- `lerobot`：沿用原有 LeRobotDataset 保存逻辑，保存 Parquet、视频和 metadata。

### 方式一：保存 transition 包

配置文件中设置：

```json
"actor_mode": "vla_only",
"save_format": "transition",
"actor_only": {
  "episode_output_dir": "/home/hpc/yuzhang/outputs/online_rl_outbox/pi05_base_smovla_v3_0720_RLT_30K/actor_transition_episodes",
  "save_episode_images": true,
  "save_episode_viewer": true
}
```

启动命令：

```bash
cd /home/lenovo/code/Evo-RL-loop-0911
/home/lenovo/code/envs/evo_0911/bin/python -m lerobot.onlineRL_evoRL.actor_new \
  --config_path src/lerobot/onlineRL_evoRL/configs/actor/piper_cup_catch_pi05_Actor_actorOnly_transition.json
```

输出目录可以已经存在；重新启动时会从现有最大 episode 编号继续写入。保存结构：

```text
actor_transition_episodes/
  episode_000000/
    metadata.json
    compact_episode.pt
    frames.json
    images/
      observation.images.top/
      observation.images.wrist/
    viewer.html
```

打开某个 episode 的 `viewer.html` 即可查看 top/wrist 图像、task、reward、
done/truncated、action、state、介入状态和 VLA checkpoint 信息。

### 方式二：保存标准 LeRobotDataset

配置文件中设置：

```json
"actor_mode": "vla_only",
"save_format": "lerobot",
"actor_only": {
  "episode_output_dir": "/home/hpc/yuzhang/outputs/online_rl_outbox/pi05_base_smovla_v3_0720_RLT_30K/actor_lerobot_dataset",
  "save_episode_images": true,
  "save_episode_viewer": true
}
```

启动命令：

```bash
cd /home/hpc/yuzhang/Evo-RL-loop-0817
/home/hpc/yuzhang/envs/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.actor_new \
  --config_path src/lerobot/onlineRL_evoRL/configs/actor/piper_cup_catch_pi05_Actor_actorOnly_lerobot.json
```

`lerobot` 方式要求 `episode_output_dir` 在启动时不存在，因此每次新建数据集应使用新目录；
不要与 `transition` 方式共用同一目录。`save_episode_images` 和 `save_episode_viewer` 仅对
`transition` 方式生效，`lerobot` 方式固定由 LeRobotDataset 写入视频。

保存结构：

```text
actor_lerobot_dataset/
  data/
    chunk-000/
      file-000.parquet
  meta/
    episodes/
    info.json
    stats.json
    tasks.parquet
  videos/
    observation.images.top/
    observation.images.wrist/
```

数据集使用 `dataset.repo_id` 作为 repo id，使用 `env.fps` 作为帧率。可以用 Python 读取：

```bash
PYTHONPATH=/home/hpc/yuzhang/Evo-RL-loop-0817/src \
/home/hpc/yuzhang/envs/package_sorting_env/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local_data",
    root="/home/hpc/yuzhang/outputs/online_rl_outbox/pi05_base_smovla_v3_0720_RLT_30K/actor_lerobot_dataset",
)
print(dataset)
print(dataset[0]["action"])
PY
```

标准数据集中保存的主要字段与 `RL_data.sh` 一致：

- `observation.state`
- `observation.images.top` / `observation.images.wrist`
- `action`：实际传入 CAN1 从臂的动作
- `complementary_info.policy_action`
- `complementary_info.is_intervention`
- `complementary_info.state`
- `complementary_info.collector_policy_id`
- task 和 episode success/failure metadata

这些业务字段的名称与 dtype 对齐 0901 数据，磁盘结构仍使用当前 LeRobot v3 的
Parquet、MP4、`meta/episodes` 与流式编码；因此统一字段不等于退回旧存储格式。

## Actor 实时数据显示

配置文件中设置：

```json
"processor": {
  "observation": {
    "display_cameras": true
  }
}
```

启用后，Actor 使用与 `scripts/RL_data.sh --display_data=true` 相同的 Rerun
显示逻辑，实时显示处理后的摄像头图像、机械臂观测状态，以及策略或人工干预后
选中的动作。Actor 不再为该配置调用 `cv2.imshow()`。

设置为 `false` 时不启动 Rerun：

```json
"display_cameras": false
```

## 热键和 Episode 规则

当前 actor 内置 HIL 热键，不依赖 `event_config.json` 的质量事件配置。

- `C`：人工接管/释放。接管时立即暂停 RTC，废弃排队及在途旧动作，并在当前周期读取主臂。
- `B`：标记当前 episode 成功并结束。最后一条 transition 写入 `reward=1.0, done=true`。
- `F`：标记当前 episode 失败并结束。最后一条 transition 写入 `reward=0.0, done=true`。
- `A`：放弃当前 episode 并重录，不发送、不保存为有效 episode，双臂保持当前位置。
- `R`：放弃当前 episode，双臂回默认初始位后重录，不发送、不保存为有效 episode。
- `Esc`：停止 actor。

`actor_new` 还支持：

- `V`：在纯 VLA 和 VLA+Actor 之间切换；Actor 权重不存在时保持纯 VLA。
- task 配置中的按键（参考配置为 `1/2/3`）：切换 task，放弃当前 episode 并按 `R` 的流程归位。

`B/F/A` 结束后读取从臂最后位置并继续发送保持动作，不关闭使能；`R` 回到 actor 启动时捕获的初始位置。
单臂和双臂共用按键。`can0.control=false` 时不连接主臂且忽略 `C`；为 `true` 时 policy 阶段
主臂随从臂目标移动，按 `C` 后立即接管。

## Reward 设计

当前 reward 是稀疏终止奖励：

- 普通 step：使用环境/processor 当前 reward，通常为 `0.0`。
- `B`：覆盖最后一步为 `reward=1.0, done=true`。
- `F`：覆盖最后一步为 `reward=0.0, done=true`。
- timeout：`truncated=true`，reward 保持当前值。
- `A` / `R`：当前 episode 丢弃，不进入 learner 或本地有效保存。

每条 transition 的 `complementary_info` 包含：

```text
discrete_penalty
is_intervention
intervention_state
success
failure
actor_policy_is_vla
policy_action
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

Actor 2 的 VLA+Actor 模式：

```text
raw robot observation + selected task -> frozen VLA/RLT -> z_rl + proprio + ref_action
                                     -> trained chunk Actor -> postprocessor -> robot action
                                     -> transition -> learner
```

PI05 learner：

```text
offline DataLoader batch + meta task/reward/intervention -> frozen PI0.5/RLT features
                   -> optional online replay mix -> chunk Actor + twin-Q update -> checkpoint
```

Actor-only VLA：

```text
raw robot observation + env.task -> VLA predict_action -> action processor -> env.step -> transition -> local episode files
```

## 验证命令

语法检查：

```bash
/home/lenovo/code/envs/evo_0911/bin/python -m compileall -q \
  /home/lenovo/code/Evo-RL-loop-0911/src/lerobot/onlineRL_evoRL
```

配置解析：

```bash
/home/lenovo/code/envs/evo_0911/bin/python \
  -m lerobot.onlineRL_evoRL.actor_new --help
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

- `actor_vla_policy.enabled=true` 只为在线 Actor 提供 VLA/RLT 特征，不训练 VLA 权重。
- 当前有效组合是 `policy.type=pi05_rlt` 与 `algorithm.type=rlt_chunk`；learner 只更新独立 Actor/Critic。
- Actor-only/VLA-only 模式已经停用，纯 VLA 采集使用 `RL_data.sh` 或 `RL_data_bimanual.sh`。
- VLA 推理依赖 `dataset.root` 里的 metadata/stats，确保该路径可读取并包含两路相机和 action/state 统计。
