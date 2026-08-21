# onlineRL_evoRL 使用说明

`onlineRL_evoRL` 是当前 EvoRL 在线采集/训练逻辑的工作目录。当前保留原 Actor，并新增基于同一套采集/传输流程的 Actor 2，同时提供独立的 PI05 Online RL learner：

- 默认 SAC actor：沿用在线 RL 的 SAC action 生成和 learner 参数同步。
- VLA actor：沿用 `scripts/RL_data.sh --policy.path` 的 VLA checkpoint 推理方式生成动作。
- Actor 2：直接根据 `policy.type` 加载策略，可在纯 VLA 与 VLA+训练后 Actor 之间切换，并支持按键切换 task。
- PI05 Online RL learner：冻结 PI0.5/RLT，只训练完整 action chunk Actor 和 twin-Q Critic。

支持以下启动模式：

- online：actor 连接 learner，episode 结束后发送 transitions。
- actor-only：actor 不连接 learner，episode 结束后按 episode 保存本地数据并生成可视化页面。
- 离线初始化：learner 先从离线数据集训练 `steps` 次；完成后保持服务运行并等待在线 episode。
- online + offline：learner 加载离线数据，同时接收 actor 在 episode 结束后发送的在线 transitions。

## 运行环境

建议从项目根目录启动：

```bash
cd /home/yz/projects/Evo-RL-loop-0810
export PYTHONPATH=/home/yz/projects/Evo-RL-loop-0810/src:$PYTHONPATH
```

推荐使用当前项目环境：

```bash
source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
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

## Learner：PI05 Online RL

PI05 learner 冻结 PI0.5/RLT，只训练 action-chunk Actor 和 twin-Q Critic。推荐配置：

`src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_pi05_online_rl_learner_only.json`

启动命令必须使用带等号的配置参数：

```bash
source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
cd /home/hpc/yuzhang/Evo-RL-loop-0817
python -m lerobot.onlineRL_evoRL.learner \
  --config_path=src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_pi05_online_rl_learner_only.json
```

配置文件是基准值；需要临时实验时，可在命令行用同名参数覆盖。参数统一使用 `--参数=值`，布尔值使用小写 `true/false`：

```bash
python -m lerobot.onlineRL_evoRL.learner \
  --config_path=src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_pi05_online_rl_learner_only.json \
  --steps=1000 \
  --policy.online_updates_per_episode=100 \
  --policy.online_only_after_initialization=false \
  --batch_size=8 \
  --num_workers=2
```

命令行覆盖只适合一次性实验；稳定参数应写回 learner-only JSON，便于 checkpoint 保存完整配置和复现实验。

### 必须准备的输入

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
| `policy.online_updates_per_episode` | `100` | 每接收并成功转换一个完整在线 episode，增加多少次在线更新额度；多个 episode 的额度会累加。 |
| `policy.online_only_after_initialization` | `false` | `false`：在线阶段每个 batch 混合 online/offline；`true`：离线初始化结束后只采样 online replay。该参数不会跳过最初的 `steps` 次离线初始化。 |
| `batch_size` | `16` | 每次 learner update 的目标样本数。混合模式取 `max(1, batch_size // 2)` 个 online 样本，其余来自 offline；在线 replay 样本不足时实际 batch 可能更小。 |
| `num_workers` | `4` | 离线 DataLoader worker 数。显存或主存紧张时先减 `batch_size`，再减为 `2` 或 `0`；`0` 表示主进程加载。 |
| `policy.online_buffer_capacity` | `100000` | online compact feature replay 最多保存的 sliding-window transition 数，不是 episode 数。增大它主要增加 CPU 内存和 checkpoint 体积。 |
| `policy.actor_update_interval` | `1` | 每隔多少个累计 `learner_step` 更新一次 Actor；Critic 每一步都会更新。 |
| `log_freq` / `save_freq` | `5 / 100` | 每多少个累计 `learner_step` 记录日志/保存 checkpoint；`save_freq` 仅在 `save_checkpoint=true` 时生效。 |

如果目标是“先离线初始化 1000 步，之后每个新 episode 训练 100 步”，设置：

```json
{
  "steps": 1000,
  "policy": {
    "online_updates_per_episode": 100,
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
| `policy.actor_hidden_dims` / `policy.critic_hidden_dims` | `[256,256,256]` | 新训练的 Actor/Critic MLP 隐藏层宽度；增大后显存、计算量和 checkpoint 都会增加。 |
| `policy.num_critics` | `2` | Q 网络数量；当前实现至少需要 2 个。 |
| `policy.fixed_std` | `0.002` | Actor 非确定性调用时的固定高斯噪声标准差，必须大于 0；当前 learner 的 loss 路径使用 mean/deterministic action，因此该值暂不改变训练 loss。 |
| `policy.reference_dropout_prob` | `0.5` | Actor 训练时丢弃冻结 VLA reference action 条件的概率，用于避免 Actor 只复制 reference；范围 `[0,1]`。 |
| `policy.q_weight` / `policy.bc_weight` | `0.1 / 5.0` | Actor loss 为 `-q_weight × Q + bc_weight × masked_BC`。增大前者更偏向奖励，增大后者更贴近人工动作/VLA reference。 |
| `policy.discount` | `0.96` | chunk 内 reward 折扣以及完整窗口 bootstrap 折扣，范围 `(0,1]`。 |
| `policy.critic_target_update_weight` | `0.005` | target Critic 的 Polyak 软更新系数；越小更新越平滑，范围 `(0,1]`。 |
| `policy.actor_lr` / `policy.critic_lr` | `3e-4 / 3e-4` | learner 实际使用的 Actor/Critic Adam 学习率。 |
| `policy.grad_clip_norm` | `10.0` | Actor 与 Critic 各自的梯度范数裁剪上限。 |

PI0.5/RLT 骨干在本 learner 中被冻结；日志里的 `bc_loss` 下降表示新 Actor 更接近 BC target，不表示 VLA 骨干正在更新。通用训练配置中的 `optimizer`、`scheduler` 和 `gradient_accumulation_steps` 不控制这里的两个 Adam optimizer；应使用上表的 `policy.actor_lr`、`policy.critic_lr` 和 `policy.grad_clip_norm`。

### 资源、服务与输出参数

| 参数 | 建议/当前示例 | 用法与精确含义 |
| --- | ---: | --- |
| `policy.device` | `cuda` | 冻结骨干、Actor/Critic 和计算 batch 所在设备。 |
| `policy.storage_device` | `cpu` | online replay 特征的存储设备。保持 `cpu` 可避免 replay 长期占用显存。 |
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

`manifest.json` 记录基座 checkpoint 路径和特征维度；恢复时先从 `policy.pretrained_path` 加载冻结 VLA/RLT，再加载 `actor_critic.pt`。因此基座 checkpoint 必须继续可访问。optimizer、`learner_step`、`interaction_step`、`pending_online_update_steps`、`online_episode_count` 和 online replay 都会恢复。

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

`reload_on_episode_boundary=true` 时，actor 会在 episode 边界按 `policy_poll_s` 检查 checkpoint 文件变化；发生变化后，下一个 episode 前重新加载 VLA。

## Online 模式

与当前 cup-catch learner 匹配的 Actor 配置：

```bash
python -m lerobot.onlineRL_evoRL.actor \
  --config_path=src/lerobot/onlineRL_evoRL/configs/piper_cup_catch_pi05_online_transition_actor.json
```

该配置必须满足：`dataset.root` 与 learner 一致、`env.task` 精确存在于 `meta/tasks.parquet`、`actor_vla_policy.policy_path` 与 learner 的冻结基座一致、端口一致。`online_transition.enabled=true` 后，Actor 在 episode 结束时分批提取 `z_rl/proprio/ref_action`，发送一个原子 compact episode；learner 不再为该在线 episode 解码图像或重复运行 RLT。

```json
"online_transition": {
  "enabled": true,
  "save_local_copy": true,
  "episode_output_dir": ".../actor_transitions",
  "feature_batch_size": 8
}
```

`save_local_copy` 只控制在线 compact episode 的本地备份，不影响发送。Actor 权重回传尚未启用；Actor 继续使用冻结 VLA 和人工介入动作。旧 SAC raw-transition 在线路径仍兼容。

## Actor 2：VLA / VLA+Actor 与多任务切换

Actor 2 保留原 `actor.py` 的环境、人工介入、episode 保存、compact transition 和 gRPC 逻辑，仅替换模型加载与键盘控制。原 Actor 及其配置仍可继续使用。

参考配置：

```text
src/lerobot/onlineRL_evoRL/configs/piper_cup_catch_pi05_online_transition_actor_2.json
```

启动：

```bash
cd /home/hpc/yuzhang/Evo-RL-loop-0817
source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
python -m lerobot.onlineRL_evoRL.actor_2 \
  --config_path=src/lerobot/onlineRL_evoRL/configs/piper_cup_catch_pi05_online_transition_actor_2.json
```

### 模型与 Actor 权重

Actor 2 不再额外硬编码一个 SAC policy。主配置中的 `policy.type` 决定模型类型，当前参考配置使用：

```json
"policy": {
  "type": "pi05_online_rl",
  "pretrained_path": "/path/to/pi05_vla_rlt_base",
  "device": "cuda"
}
```

加载方式与 learner 对齐：使用 `dataset.root` 的 metadata/stats 调用 `make_policy(cfg.policy, ds_meta=...)`，因此 `dataset.root` 必须与 learner 使用的数据集兼容。`policy.pretrained_path` 提供冻结的 VLA/RLT 权重，`actor_checkpoint_path` 提供 learner 训练后的 Actor 权重：

```json
"actor_checkpoint_path": "/path/to/learner_output"
```

必须对齐的配置：

- `policy.type=pi05_online_rl`：启用当前 VLA/RLT+chunk Actor 接口。
- `policy.pretrained_path`：与 learner 的冻结基座相同。
- `actor_checkpoint_path`：learner 输出根目录、某个 checkpoint 目录或具体权重文件。
- `actor_vla_policy.enabled=true`：原 Actor 流程创建 VLA runtime 所必需；`actor_vla_policy.policy_path` 应与 `policy.pretrained_path` 保持一致。
- `dataset.root`：提供与 learner 一致的 features、normalization stats 和合法 task。
- `policy.actor_learner_config`：host/port 必须与 learner 一致。
- `task_hotkeys_path`：独立任务热键 JSON 的路径。

Actor 权重按以下顺序发现：

1. `<path>/actor_critic.pt`
2. `<path>/checkpoints/last/actor_critic.pt`
3. 兼容旧 checkpoint 的 `<path>/pretrained_model/model.safetensors`
4. 兼容旧 checkpoint 的 `<path>/checkpoints/last/pretrained_model/model.safetensors`

精简 checkpoint 只读取 `actor_critic.pt` 中的 `actor`，不会加载 Critic；旧 `model.safetensors` 也只读取 `actor.*` tensor。加载精简 checkpoint 时还会校验 `manifest.json` 中的基座路径、`chunk_size`、`z_dim` 和 `proprio_dim`，不匹配会拒绝启动 Actor 模式。

若启动时 Actor 权重尚不存在，程序正常使用纯 VLA 推理；此时按 `B` 只记录错误，不切换模式。设置 `actor_vla_policy.reload_on_episode_boundary=true` 后，每个 episode 边界都会按 `policy_poll_s` 重新查找 checkpoint，learner 后续生成 Actor 权重后即可被发现。

### `B`：切换动作生成模式

- 默认：纯 VLA 生成动作。
- 第一次按 `B`：切换为 VLA+Actor。VLA/RLT 生成 `z_rl/proprio/ref_action`，训练后的 Actor 输出 action chunk。
- 再次按 `B`：恢复纯 VLA。
- 每次成功切换都会清空 VLA action queue、Actor action queue 和动作平滑缓存，当前控制周期重新推理并发送新动作。
- Actor 权重不存在或当前 `policy.type` 不提供兼容的 RLT Actor 接口时，`B` 不生效，机械臂继续由纯 VLA 控制。

### 数字键切换 task

Actor 2 不再要求在主配置中固定单一 `env.task`，而是从独立配置读取按键和 task：

```text
src/lerobot/onlineRL_evoRL/configs/piper_cup_catch_task_hotkeys.json
```

```json
{
  "default_key": "3",
  "tasks": {
    "1": "Pick up the cup on the right",
    "2": "Take the middle cup away",
    "3": "Grab the left cup"
  }
}
```

主配置通过 `task_hotkeys_path` 指向该文件。启动时使用 `default_key` 对应的 task，并检查所有 task 必须精确存在于 `dataset.root/meta/tasks.parquet`。

运行中按 `1/2/3` 时：

1. 当前 episode 标记为重录并丢弃，不发送给 learner，也不保存为有效 episode。
2. 执行与 `R` 相同的双臂归位和 episode reset。
3. 下一个 episode 使用新 task；当前纯 VLA/VLA+Actor 模式保持不变。

任务热键必须是单个字符，且不能占用 `B/I/S/F/R`。

当前 Actor 2 未包含先前讨论的 Actor/Learner GPU 租约与空闲显存释放机制；同时启动两个进程时仍需按实际显存容量安排模型驻留。

## Actor-only 模式

Actor-only 不连接 learner，不检查 learner 是否存在。`actor_only.save_format` 用于选择两种保存方式，默认是 `lerobot`：

- `transition`：保存与在线发送完全相同的 compact episode，同时可生成 JSON、JPEG 和 HTML viewer；要求启用 PI05+RLT VLA。
- `lerobot`：沿用原有 LeRobotDataset 保存逻辑，保存 Parquet、视频和 metadata。

### 方式一：保存 transition 包

配置文件中设置：

```json
"actor_only": {
  "enabled": true,
  "episode_output_dir": "/home/hpc/yuzhang/outputs/online_rl_outbox/pi05_base_smovla_v3_0720_RLT_30K/actor_transition_episodes",
  "save_format": "transition",
  "save_episode_images": true,
  "save_episode_viewer": true
}
```

启动命令：

```bash
cd /home/hpc/yuzhang/Evo-RL-loop-0817
/home/hpc/yuzhang/envs/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.actor \
  --config_path src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl_actorOnly_transion.json
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
"actor_only": {
  "enabled": true,
  "episode_output_dir": "/home/hpc/yuzhang/outputs/online_rl_outbox/pi05_base_smovla_v3_0720_RLT_30K/actor_lerobot_dataset",
  "save_format": "lerobot",
  "save_episode_images": true,
  "save_episode_viewer": true
}
```

启动命令：

```bash
cd /home/hpc/yuzhang/Evo-RL-loop-0817
/home/hpc/yuzhang/envs/package_sorting_env/bin/python -m lerobot.onlineRL_evoRL.actor \
  --config_path src/lerobot/onlineRL_evoRL/configs/piper_package_sorting_online_rl_actorOnly_lerobot.json
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

### Actor 实时数据显示

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

- `I`：切换人工接管。默认 policy/VLA 控制；按一次进入人工接管，再按一次释放回 policy/VLA。
- `S`：标记当前 episode 成功并结束。最后一条 transition 写入 `reward=1.0, done=true`。
- `F`：标记当前 episode 失败并结束。最后一条 transition 写入 `reward=0.0, done=true`。
- 左箭头：放弃当前 episode 并重录，不发送、不保存为有效 episode，双臂保持当前位置。
- `R`：放弃当前 episode，双臂回默认初始位后重录，不发送、不保存为有效 episode。
- `Esc`：停止 actor。

仅 `actor_2.py` 额外支持：

- `B`：在纯 VLA 和 VLA+Actor 之间切换；Actor 权重不存在时保持纯 VLA。
- task 配置中的按键（参考配置为 `1/2/3`）：切换 task，放弃当前 episode 并按 `R` 的流程归位。

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
- 旧 `policy.type=sac` 配置仍训练 SAC；新 `policy.type=pi05_online_rl` 配置训练 chunk Actor/Critic。两者的在线 actor 都可由 VLA 采样。
- actor-only 模式不会向 learner 发送数据。
- VLA 推理依赖 `dataset.root` 里的 metadata/stats，确保该路径可读取并包含两路相机和 action/state 统计。
