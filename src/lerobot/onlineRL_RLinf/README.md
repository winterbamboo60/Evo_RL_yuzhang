# Piper RLT Online Actor-Critic

`onlineRL` 是单 Piper 从臂的在线 RLT actor-critic 运行模块。它复用项目已有的
`piper_follower` 和 `piper_leader` 控制接口：从臂负责实际任务执行，主臂在自动模式
接收同一动作作镜像同步，在人工接管模式中由操作者产生动作并同步给从臂。

当前默认运行方式是本机 actor + learner：actor 负责 Stage1/RLT actor 推理和机械臂控制，
learner 负责 episode replay、critic、target critic 和全部模型更新。两者之间通过独立
消息协议传递完整 episode 和更新后的 actor 权重，为将 learner 迁至另一台机器保留边界。

## 功能

- 保持 RLT Stage2 输入输出：`z_rl + proprio + ref_chunk -> action_chunk`。
- twin critic、target critic、TD update 与 BC/Q actor loss。
- 只记录 `follower.send_action()` 返回的实际动作；主臂动作不会作为训练 action 保存。
- 以**完整 episode**为 replay 缓存单位；默认最多保存 200 个 episode。
- 缓存至少积累 10 个完整 episode 后启动训练并允许 actor 更新；默认每个新 episode 带来 8 次更新预算，
  每 4 次 critic update 更新一次 actor。
- `S` 标记成功，`Q`/`F` 标记显式失败，超时也视为失败且禁止 TD bootstrap。
- `R` 放弃当前 episode、不进入训练，并将主从臂慢速回到启动时保存的初始关节姿态。
- `B` 切换使用 Stage1 VLA reference action 或 RLT actor action。
- `I` 持久切换人工接管：人工操作主臂，从臂执行同一动作；再次按 `I` 恢复自动模式。
- 交互式终端优先使用 TTY 热键监听；无 X11/`DISPLAY` 的 SSH 终端仍可使用 B/I/S/Q/F/R。
- 支持 fake Piper/fake Stage1 VLA 的 CPU dry-run，不会访问 CAN。
- 真机连接两路 OpenCV 相机；每次观测提供 `top_image` 和 `wrist_image`（RGB、uint8、HWC）给 Stage1 特征提取器。

## 目录

```text
onlineRL/
├── config.py              配置 dataclass 与 JSON 加载
├── run.py                 运行入口
├── actor_runtime.py       机械臂 rollout、episode 和 actor 权重应用
├── learner_runtime.py     episode replay 与 actor-critic 更新
├── rlt_model.py           actor、twin critic 和 target update
├── hardware.py            fake/real Piper 主从臂适配
├── vla.py                 Stage1 RLT feature 边界
├── stage1_rlt_adapter.py  本地 Stage1 Pi0.5+RLT 特征提取器
├── rlt_token_transformer.py  从 RLinf RLT 迁移来的 RL token transformer
├── keyboard.py            B/I/S/Q/R 键盘状态机
├── replay.py              episode 级 replay buffer
├── transport.py           本机通信与远程通信预留边界
├── configs/piper_rlt_ac.json
└── tests/
```

## 首次验证：不连接机械臂

在目标环境运行：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
source /home/yz/projects/env/package_sorting_env/bin/activate
python -m lerobot.onlineRL.tests.run_tests
```

仓库提供标准库测试入口，便于在不依赖 pytest 命令的情况下验证核心逻辑。应看到六个 `PASS`。

然后运行完整 dry-run：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
source /home/yz/projects/env/package_sorting_env/bin/activate
python -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting.json \
  --dry_run
```

dry-run 强制使用 CPU、fake Piper 和 fake Stage1 特征；不会打开 `can0/can1`，即使 JSON
中写了 CUDA device 也不会访问 GPU。

## 配置文件

复制或修改 `configs/piper_rlt_ac.json`。主要字段如下。

### `model`

| 字段 | 含义 | Piper Stage1 默认值 |
|---|---|---:|
| `z_dim` | Stage1 导出的 RLT token 特征维度 | `2048` |
| `proprio_dim` | 从臂本体状态维度 | `7` |
| `action_dim` | 单 Piper 动作维度 | `7` |
| `ref_num_action_chunks` | Stage1 reference action 长度 | `50` |
| `num_action_chunks` | RLT actor/VLA 单次连续执行长度 | `50` |
| `hidden_dim` | Stage2 MLP 隐层宽度 | `1024` |

必须满足：`ref_num_action_chunks >= num_action_chunks`，且真实 Stage1 输出的 shape 必须严格是：

```text
z_rl:     [1, z_dim]
proprio:  [1, proprio_dim]
ref_chunk:[1, ref_num_action_chunks, action_dim]
```

### `replay`

| 字段 | 含义 | 默认值 |
|---|---|---:|
| `max_cached_episodes` | learner 最多缓存的完整 episode 数 | `200` |
| `sample_window_episodes` | 从最近多少 episode 内采样 | `200` |
| `min_buffer_episodes` | 积累到该 episode 数后才允许更新 | `10` |
| `train_actor_episodes` | 积累到该 episode 数后才允许 actor 更新 | `10` |
| `batch_size` | 每个 learner update 的 transition batch | `32` |
| `update_epoch` | 每到一个 episode 所执行的更新预算 | `8` |
| `critic_actor_ratio` | critic 更新次数 : actor 更新次数 | `4` |
| `gamma` | TD 折扣 | `0.96` |
| `tau` | target 网络软更新系数 | `0.005` |
| `q_weight` | actor Q objective 权重 | `0.1` |
| `bc_weight` | actor BC objective 权重 | `5.0` |

缓存的粒度是 episode，不是 transition。learner 采样时先选 episode，再从其中选一个
action-chunk transition，因此长 episode 不会仅因 transition 更多而占据过高采样概率。

### `hardware`

| 字段 | 含义 | 默认值 |
|---|---|---|
| `follower_port` | 从臂 CAN 接口 | `can1` |
| `leader_port` | 主臂 CAN 接口 | `can0` |
| `follower_speed_ratio` | 从臂速度比例，范围 0--100 | `50` |
| `leader_speed_ratio` | 主臂 feedback 速度比例，范围 0--100 | `50` |
| `reset_duration_s` | `R` 回初始位姿的插值时长 | `3.0` |
| `control_hz` | 预留的控制循环频率配置 | `10.0` |
| `action_limit` | 发送前的安全动作幅值限制；Piper SDK 当前使用角度/夹爪单位，不是弧度 | `180.0` |
| `top_camera` | 顶部 OpenCV 相机：设备号/路径、宽、高、帧率 | 见示例配置 |
| `wrist_camera` | 手腕 OpenCV 相机：设备号/路径、宽、高、帧率 | 见示例配置 |

真机运行时两个相机都必须配置；执行器在每次读取从臂状态时读取两帧 RGB 图像，并以 `top_image`、`wrist_image` 字段交给 Stage1 特征适配器。图像格式为 RGB uint8 HWC；适配器负责沿用 Stage1 的图像顺序、预处理、任务文本和 norm_stats.json。

真实运行前，确认 CAN 接口、校准文件、Piper SDK、相机配置和机械臂初始姿态均与
`scripts/RL_data.sh` 的部署一致。

### `runtime`

| 字段 | 含义 |
|---|---|
| `mode` | 当前实现使用 `local` |
| `actor_device` | Stage1 VLA 与本地 actor 推理 GPU，例如 `cuda:0` |
| `learner_device` | learner 训练 GPU，例如 `cuda:1` |
| `dry_run` | 为 `true` 时永不连接 CAN，且强制 CPU |
| `max_episode_steps` | episode 最大原子动作步数；到达即 timeout failure |
| `outbox_dir` | 已封口、等待 learner 确认的 episode 临时目录 |
| `checkpoint_dir` | 后续检查点目录；当前示例也保留 Stage1 actor checkpoint 路径 |
| `feature_extractor_factory` | Stage1 特征适配器，格式 `python_module:function` |
| `feature_extractor_kwargs` | 传给 Stage1 特征适配器的模型路径、norm stats、任务文本和 RLT 参数 |

如果只有一张 GPU，可以将 `actor_device` 与 `learner_device` 都设置为同一张卡；但 Stage1
VLA 和 learner 会竞争显存，优先使用两张 GPU。

自动模式不使用后台 feature 缓存。运行时维护一个显式 action queue：队列动作数少于 20 时，用最新 observation 同步推理一次，得到完整 `z_rl/proprio/ref_chunk` 并把对应 action chunk 追加到队列；每轮控制最多从队列 `popleft()` 20 个动作下发。这样保留 LeRobot `select_action()` 的 action queue 语义，同时避免固定等到完整 50 步耗尽后才补队列。

## Stage1 Pi0.5 + RLT 适配

onlineRL 现在提供两条 Stage1 特征提取路径。对于 **RLinf 训练出来的 Stage1 VLA/RLT checkpoint**，默认应使用 RLinf-backed 路径；本地迁移路径只用于继续排查或做脱离 RLinf 的移植实验。

| 路径 | 配置 | 适用场景 | 说明 |
|---|---|---|---|
| `rlinf_stage1_adapter.py` | `configs/piper_rlt_ac_packageSorting_rlinf.json` | 推荐；运行 RLinf Stage1 checkpoint | 直接调用 RLinf `get_model(...).extract_rlt_obs()`，复用 Stage1 训练时的 OpenPI dataconfig、state/prompt/image transforms、RLT prefix 和动作采样逻辑 |
| `stage1_rlt_adapter.py` | `configs/piper_rlt_ac_packageSorting.json` | 本地迁移调试 | 使用 Evo 本地 `PI05Policy` 和迁移的 `RLTTokenTransformer`；当前与 RLinf OpenPI 推理图仍存在差异，不应作为真机主路径 |

推荐配置中的关键字段如下：

```json
{
  "runtime": {
    "feature_extractor_factory": "lerobot.onlineRL.rlinf_stage1_adapter:build_extractor",
    "feature_extractor_kwargs": {
      "rlinf_path": "/home/yz/projects/RLinf_yuzhang",
      "model_path": "/home/yz/projects/RLinf_yuzhang/logs/.../checkpoints/global_step_6000/actor",
      "norm_stats_path": "/home/yz/datasets/1F_0713/smovla_V3_0720_merged_RLinf/norm_stats.json",
      "repo_id": "realworld_package_sorting_rlt_stage1",
      "config_name": "pi05_piper_state",
      "num_images_in_input": 2,
      "action_chunk": 50,
      "num_steps": 10,
      "rlt_prefix_seq_len": 1024,
      "rlt_image_only": false,
      "rlt_use_mask": true
    }
  }
}
```

适配器输入来自 `RealPiperHardware.observation()`：

| onlineRL 字段 | RLinf Stage1 输入 |
|---|---|
| `state` | 原始 Piper 7D proprio，传入 RLinf `env_obs["states"]` |
| `top_image` | 顶部 RGB 图像，HWC，`[0,1]` float 或 `uint8`；传入 `main_images` |
| `wrist_image` | 腕部 RGB 图像，HWC，`[0,1]` float 或 `uint8`；传入 `wrist_images` |
| `prompt` / `task_description` | 任务文本，传入 `task_descriptions` |

输出必须满足并会被运行时校验：

```text
z_rl:     [1, 2048]
proprio:  [1, 7]
ref_chunk:[1, 50, 7]
```

当前排查结论：同一个 RLinf Stage1 checkpoint 在 RLinf `extract_rlt_obs()` 下能更贴近训练集未来动作；而本地 `stage1_rlt_adapter.py` 曾出现两类偏差：图像 resize padding 被二次归一化到 `-3`，以及本地 `PI05Policy` 的 action sampling 与 RLinf OpenPI 模型不完全一致。图像 padding 已修正，但为了保证真机动作行为和 RLinf Stage1 一致，主运行路径仍应使用 `rlinf_stage1_adapter.py`。

### 动作尺度排查

Stage1 的 `predict_action_chunk()` 输出位于训练时的归一化动作空间，本地适配器会用 `norm_stats["actions"]` 做一次 quantile inverse，得到 Piper SDK 当前使用的角度/夹爪单位。不要再把 `ref_chunk` 当弧度处理。

如果日志中 `command` 是几十到一百多的正常训练数据量级，但 `executed` 长期贴近 `+/-3.15`，说明发送前安全限幅仍在使用弧度尺度。Piper onlineRL 默认 `hardware.action_limit` 应为 `180.0`，用于保留 Stage1 训练统计范围内的关节角度动作。

自动模式下 action queue 少于 20 个动作时，Stage1 VLA/RLT 推理会产生一个动作 chunk，并缓存 `model.num_action_chunks` 个原子动作；默认每次追加 50 步 reference chunk。当前不使用后台 feature 缓存；补队列时用最新 observation 重新推理完整 `z_rl/proprio/ref_chunk`，每轮最多消费 20 个动作。不要改回每个原子动作都调用 Stage1 VLA，否则控制频率会被视觉模型推理降到约 1--2 Hz。

VLA 自动动作下发前沿用 `scripts/recording_loop.py` 的动作质量后处理：9 帧在线滑窗均值平滑所有动作维度，并对 `gripper.pos` 做同一套非线性衰减。日志里的 `command` 是后处理后的实际下发目标，`ref_chunk` 仍保留 Stage1 原始参考 chunk。

## 真实机械臂启动

如果运行 RLinf Stage1 checkpoint，先进入 RLinf Docker 并切换 OpenPI 环境；该环境包含 RLinf OpenPI 依赖和旧版 `lerobot.common`：

```bash
docker exec -it rlinf /bin/bash
source switch_env openpi
cd /home/yz/projects/Evo-RL-loop-0810/src
```

确认 `configs/piper_rlt_ac_packageSorting_rlinf.json` 中 Stage1 checkpoint、`norm_stats_path`、相机设备号和任务文本正确；启动时用 `--no-dry_run` 切到真机，并在交互式终端中运行以便读取 B/I/S/Q/F/R 热键。随后执行：

```bash
python -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --no-dry_run
```

如果只验证本地迁移版 adapter，可改用 `piper_rlt_ac_packageSorting.json`，但这条路径目前不作为 RLinf checkpoint 的推荐真机运行方式。

真机模式默认持续运行多个 episode，并在每个完整 episode 后把 episode 送入 learner。需要只跑固定轮数时加 `--episodes N`；dry-run 默认只跑 1 个 episode，避免验证命令无限运行。

首次连接时会读取从臂当前关节状态并将其作为 `R` 的初始回位姿态。请在启动前人工将主从臂放到安全初始位置。

对齐 `RL_data.sh` 的 episode 生命周期：真机启动后主从臂和相机连接会跨 episode 复用，`S/Q/F/R` 结束当前 episode 后不会主动断开再重开相机；下一 episode 开始前会在日志中进行 5 秒倒计时。`R` 会放弃当前 episode、回到启动时保存的初始姿态，然后重新进入倒计时。

### 单独启动 actor

actor-only 模式只运行 Stage1/RLT 推理、机械臂控制和 episode 采集，不创建 learner；完整 episode
保存在 `runtime.outbox_dir`，当前版本不会远程发送或训练：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
source switch_env openpi
python -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --mode actor \
  --no-dry_run
```

调试启动命令保留 `--display_data` 和 `--debug_every_steps 1`：前者把相机和状态写入 Rerun，后者每个原子动作都打印 state、image、VLA feature、command 和 executed 的范围，便于定位卡顿或动作尺度问题。输出会同步保存到 `onlineRL/logs/`。

```bash
mkdir -p lerobot/onlineRL/logs

cd /home/yz/projects/Evo-RL-loop-0810/src
source switch_env openpi
python -u -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --mode actor \
  --no-dry_run \
  --display_data \
  --debug_every_steps 1 \
  2>&1 | tee "lerobot/onlineRL/logs/onlineRL_$(date +%Y%m%d_%H%M%S).log"

cd /home/yz/projects/Evo-RL-loop-0810/src
source switch_env openpi
python -u -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --mode actor \
  --no-dry_run \
  --display_data \
  --debug_every_steps 100000 \
  2>&1 | tee "lerobot/onlineRL/logs/onlineRL_$(date +%Y%m%d_%H%M%S).log"在·
```

## 按键 

| 按键 | 行为 |
|---|---|
| `B` | 在 Stage1 VLA reference chunk 与 RLT actor chunk 间切换 |
| `I` | 进入/退出持续人工接管；人工操作主臂、从臂执行同一动作 |
| `S` | 成功：当前末步 reward=1、terminal，主从臂保持当前位置 |
| `Q`/`F` | 显式失败：当前末步 reward=0、terminal，主从臂保持当前位置 |
| `R` | 放弃当前 episode、不训练，主从臂慢速回初始位置 |
| 超时 | 失败：末步 reward=0、terminal，主从臂保持当前位置 |

`S`、`Q`/`F` 和 timeout 都不会自动回位，且会进入 hold 状态，不会擅自输出下一条动作。仅 `R`
调用回位逻辑。

## 当前远程部署边界

`protocol.py` 的完整 episode / actor weight 消息和 `transport.py` 已将 actor 与 learner
解耦。本版本已实现本机 `LocalTransport`；远程 gRPC transport 保留为独立实现边界，后续
增加远程 server/client 时不需要修改 Piper 控制、episode replay 或 RLT loss。

## 权重格式转换

`convert_stage1_to_lerobot_policy.py` 支持两个方向。

### RLT Stage1 转 LeRobot policy

把 RLinf/RLT Stage1 保存的 `model_state_dict/full_weights.pt + norm_stats.json` 转成
`RL_data.sh --policy.path` 可加载的 LeRobot PI05 policy 目录：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
source /home/yz/projects/env/package_sorting_env/bin/activate
python -m lerobot.onlineRL.convert_stage1_to_lerobot_policy \
  --stage1_dir /home/hpc/yuzhang/outputs/pi05_packageSorting_rlt_6k_0805 \
  --reference_policy_dir /home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_50k \
  --output_dir /home/hpc/yuzhang/outputs/pi05_packageSorting_rlt_6k_0805_evoRL \
  --image_size 224
```

转换只写标准 PI05/VLA policy 权重和 pre/postprocessor 统计；`rlt_module.*` 不会写入
`model.safetensors`，因为 `RL_data.sh` 的 `PI05Policy.from_pretrained()` 不加载 RLT 模块。
`norm_stats.json` 中 32 维 padded action 会截取前 7 维生成 LeRobot 的 `action` 统计。

### LeRobot policy 转 onlineRL Stage1/RLT

把标准 LeRobot PI05 policy 转成当前 onlineRL `stage1_rlt_adapter.py` 可加载的
`model_state_dict/full_weights.pt + norm_stats.json` 目录：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
source /home/yz/projects/env/package_sorting_env/bin/activate
python -m lerobot.onlineRL.convert_stage1_to_lerobot_policy \
  --lerobot_policy_dir /home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_50k \
  --output_dir /home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_50k_RLT \
  --template_stage1_dir /home/hpc/yuzhang/outputs/pi05_packageSorting_rlt_6k_0805 \
  --rlt_prefix_seq_len 1024
```

该方向保留 LeRobot 的 PI05/VLA `model.safetensors` 权重，并从 LeRobot processor state 提取
`state/actions` 的 quantile stats。`--template_stage1_dir` 只用于对齐参考 RLT checkpoint 的 key 布局、
RLT shape/dtype 和额外 `noise_head.*` key；不会拷贝参考模型的训练权重。源模型没有 RLT 模块时，
脚本会补齐全零初始化的 `rlt_module.*`，并以 bfloat16 保存以对齐参考 RLT checkpoint 的显存占用；
因此这个目录适合在 onlineRL 中用 `source=vla` 判断 base VLA 动作；不要把全零 RLT token 当成已训练 RLT actor 的有效特征。

转换后，将 onlineRL JSON 中的 `runtime.feature_extractor_kwargs.model_path` 和
`runtime.feature_extractor_kwargs.norm_stats_path` 指向新目录：

```json
"model_path": "/home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_50k_RLT",
"norm_stats_path": "/home/hpc/yuzhang/outputs/pi05_base_smovla_v3_0720_50k_RLT/norm_stats.json"
```

转换后的 LeRobot policy 目录也可继续用 0707 采集路径判断 Stage1 VLA 本身是否能完成任务：

```bash
bash /home/hpc/yuzhang/Evo-RL-loop-0707/scripts/RL_data.sh \
  --dataset.root /home/hpc/yuzhang/datasets/packageSorting_rlt_6k_0805_evoRL_test \
  --dataset.single_task "You are a parcel sorter. First, Grab the package and place it on the pallet. Then, Flip the package if the barcode is not facing up. Finally, Grab the scanned package and place it into the box." \
  --wrist_camera.index_or_path 4 \
  --top_camera.index_or_path 10 \
  --policy.path /home/hpc/yuzhang/outputs/pi05_packageSorting_rlt_6k_0805_evoRL \
  --event.config.path /home/hpc/yuzhang/Evo-RL-loop-0707/scripts/event_config.json
```

## 安全提示

- 先运行 dry-run 和测试，再接入 CAN。
- 模型 action 会做 NaN、Inf、维度和幅值检查；但软件限幅不能取代硬件急停。
- learner 断开时应保持最后一个完整 actor 权重；不要删除未确认 episode outbox。
- 真机运行中按 `Ctrl+C` 前，先确认从臂处于安全位置。

## 更新频率与一次训练的数据量

当前 JSON 的默认组合是：

```text
min_buffer_episodes = 10
train_actor_episodes = 10
batch_size = 32
update_epoch = 8
critic_actor_ratio = 4
```

它的准确含义如下：

1. 前 9 个完整 episode 到达 learner 后，只进入 episode replay；不会训练。
2. 第 10 个完整 episode 到达后，缓存达到 `min_buffer_episodes=10`，开始训练，并允许 actor 更新。
3. 此后每收到一个完整 episode，learner 增加 `update_epoch=8` 次 critic 更新预算。
4. 每一次 critic update 都构造一个 `batch_size=32` 的 batch：先从最近 `sample_window_episodes=200` 个完整 episode 中随机选 episode，再从各 episode 内随机选 action-chunk transition。
5. 每 4 次 critic update 更新一次 actor。因此，更新序号 1、5、9……会同时运行 critic 和 actor；其余更新序号只运行 critic。

所以，在默认的 8 次更新预算中，通常会发生：

```text
critic update：8 次，每次抽样 32 条 transition
actor update：2 次，每 4 次 critic update 一次
```

可以把 actor 的频率粗略看作每经历约 `4 x 32 = 128` 次 transition **抽样**更新一次，但这不表示“收集到 128 条新 transition 才更新 actor”：replay 是随机采样，同一条 transition 可以在不同 batch 中重复出现；actor update 本身也会使用一个 batch 计算 Q objective 与 BC objective。

这不是“每次训练所有缓存数据”的 epoch 式训练。每次仅训练固定大小的随机 batch，使每一步更新的计算量稳定；`max_cached_episodes=200` 只是可被采样的完整 episode 上限。

## 随机抽帧对比数据集动作、onlineRL 和 RLinf VLA 输出

`compare_stage1_parity.py` 用于从 Stage1 训练数据中抽帧，读取该帧之后的真实未来动作
chunk，并用同一帧的 `state + image + wrist_image + prompt` 做 Stage1 推理对照：

- `onlineRL` 本地迁移路径：`stage1_rlt_adapter.py` + 本地 `PI05Policy` + `RLTTokenTransformer`。
- RLinf 对照路径：`rlinf.models.embodiment.openpi.get_model()` + `extract_rlt_obs()`。
- 默认还会打印输入处理审计，逐步展示 dataset 原始字段、onlineRL batch/prompt/token/image mask、RLinf `obs_processor -> input_transform -> precision_processor` 后的字段。

当前典型现象是：RLinf 对照路径更贴近训练集未来动作；本地迁移路径即使修正图像 padding 后，仍可能因为 OpenPI 推理图和 state/action suffix 处理不一致而偏离。真机运行 RLinf checkpoint 时应使用 `piper_rlt_ac_packageSorting_rlinf.json`。

RLinf Stage1 默认训练方式如下。容器挂载 `/home/yz`，进入容器后切到 OpenPI 环境，再在
RLinf 仓库执行 SFT 脚本：

```bash
docker run -d --gpus all \
   --shm-size 120g \
   --network host \
   --name rlinf \
   -v /home/yz:/home/yz \
   docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
   tail -f /dev/null

docker exec -it rlinf /bin/bash
source switch_env openpi
cd /home/yz/projects/RLinf_yuzhang
bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05
```

脚本默认读取这些本机路径：

```text
onlineRL config: lerobot/onlineRL/configs/piper_rlt_ac_packageSorting.json
RLinf repo:      /home/yz/projects/RLinf_yuzhang
Stage1 config:   /home/yz/projects/RLinf_yuzhang/examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml
```

默认会从 RLinf Stage1 YAML 自动补齐 `data.train_data_paths`、`norm_stats_path`、`repo_id`、
`pi05_piper_state`、`num_action_chunks=50` 和 RLT 参数。Stage1 checkpoint 的选择优先级为：

```text
--stage1_checkpoint > onlineRL JSON runtime.feature_extractor_kwargs.model_path > RLinf logs 最新 global_step
```

只比较 dataset 与 onlineRL 本地推理：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
source /home/yz/projects/env/package_sorting_env/bin/activate
python -m lerobot.onlineRL.compare_stage1_parity \
  --random_sample \
  --seed 0
```

在 RLinf Docker 的 `openpi` 环境中，可以加 `--with_rlinf` 对同一帧同时运行 RLinf
`extract_rlt_obs()`，检查 onlineRL 迁移后的 VLA/RLT 推理是否和 RLinf Stage1 对齐：

```bash
cd /home/yz/projects/Evo-RL-loop-0810/src
python -m lerobot.onlineRL.compare_stage1_parity \
  --random_sample \
  --seed 0 \
  --with_rlinf
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--stage1_checkpoint` | 指定 Stage1 actor checkpoint 目录，例如 `.../checkpoints/global_step_6000/actor` |
| `--dataset_path` | 覆盖 Stage1 YAML 中的训练数据目录 |
| `--stage1_config` | 覆盖 RLinf Stage1 YAML 路径 |
| `--episode` / `--frames` | 固定 episode 和帧号，例如 `--episode 0 --frames 0,25,50` |
| `--num_samples` | 在指定 episode 内均匀抽取若干帧 |
| `--image_key` / `--wrist_image_key` | 覆盖 LeRobot 视频 key，默认 `image` 和 `wrist_image` |
| `--no_dataset_prompt` | 不使用数据集中的 prompt，改用 JSON 里的 `task_description` |
| `--no_inspect_inputs` | 关闭逐步输入处理审计，只打印动作对比 |
| `--print_stage1_launch` | 只打印 RLinf Stage1 Docker/OpenPI 训练启动命令 |

`--seed` 用于复现同一次随机抽样；不需要复现时可以去掉。脚本会避开 episode 尾部，保证抽到的帧后面有完整的未来动作窗口。输出中会打印第一步 action、chunk 最大绝对误差，以及前 5 步的 dataset / onlineRL / RLinf 动作。

解读审计输出时重点看三项：

1. 图像预处理后的范围应为 `[-1, 1]`，padding 不应出现 `-3`。
2. `lang_tokens mask_true`、prompt 文本和 state 归一化/padding 是否与 RLinf 路径一致。
3. 如果输入已对齐但 `onlineRL vs RLinf` 仍大，说明本地迁移模型图和 RLinf OpenPI action sampling 仍不一致，应切到 `rlinf_stage1_adapter.py`。

## Docker 镜像打包与跨服务器部署

当前推荐部署方式仍然同时使用两个项目：

```text
/home/yz/projects/Evo-RL-loop-0810/src      onlineRL actor / learner 代码
/home/yz/projects/RLinf_yuzhang             RLinf OpenPI Stage1 推理代码
```

Stage1 VLA/RLT 真机推理使用 `piper_rlt_ac_packageSorting_rlinf.json`，该配置会通过
`rlinf_stage1_adapter.py` 调用 RLinf 的 `extract_rlt_obs()`。因此迁移到其他服务器时，不仅要迁移
Docker 镜像，还要迁移两个代码目录、Stage1 checkpoint、norm stats 和本地模型资产。

### 1. 当前镜像信息

当前使用的镜像为：

```text
docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
```

本机查询到的镜像大小约为 `31.4GB`，精确 `Size` 为 `31,393,139,480` bytes。未压缩
`docker save` 文件通常会大于或接近该大小；目标磁盘建议至少预留 `50GB`，如果还要压缩或保留多份版本，建议预留 `100GB+`。

### 2. 在当前服务器打包镜像

先创建输出目录。这里沿用当前约定路径，注意目录名是 `dokcer_imgs`：

```bash
mkdir -p /home/yz/projects/outputs/dokcer_imgs
```

保存 Docker 镜像为 tar：

```bash
docker save \
  docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
  -o /home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar
```

检查文件大小：

```bash
ls -lh /home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar
```

可选：压缩 tar，节省传输空间但会增加压缩和解压时间：

```bash
gzip -1 /home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar
```

压缩后文件为：

```text
/home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar.gz
```

### 3. 需要一起迁移的文件和目录

最小迁移集合：

```text
/home/yz/projects/Evo-RL-loop-0810/src
/home/yz/projects/RLinf_yuzhang
/home/yz/projects/RLinf_yuzhang/logs/20260805-09:12:19-realworld_rlt_stage1_sft_openpi_pi05/realworld_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_6000/actor
/home/yz/datasets/1F_0713/smovla_V3_0720_merged_RLinf/norm_stats.json
/home/yz/modelZoo/pi05base
/home/yz/modelZoo/paligemma-3b-pt-224
/home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar 或 .tar.gz
```

如果目标服务器路径不是 `/home/yz/...`，迁移后需要修改：

```text
lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json
```

重点字段：

```json
"rlinf_path": ".../RLinf_yuzhang",
"model_path": ".../checkpoints/global_step_6000/actor",
"norm_stats_path": ".../norm_stats.json",
"tokenizer_name": ".../paligemma-3b-pt-224",
"outbox_dir": "...",
"checkpoint_dir": "..."
```

### 4. 将文件传到目标服务器

示例使用 `rsync`，按实际用户名、主机名和路径替换：

```bash
rsync -avh --progress /home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar target:/home/yz/projects/outputs/dokcer_imgs/
rsync -avh --progress /home/yz/projects/Evo-RL-loop-0810/src target:/home/yz/projects/Evo-RL-loop-0810/
rsync -avh --progress /home/yz/projects/RLinf_yuzhang target:/home/yz/projects/
rsync -avh --progress /home/yz/datasets/1F_0713/smovla_V3_0720_merged_RLinf/norm_stats.json target:/home/yz/datasets/1F_0713/smovla_V3_0720_merged_RLinf/
rsync -avh --progress /home/yz/modelZoo/pi05base target:/home/yz/modelZoo/
rsync -avh --progress /home/yz/modelZoo/paligemma-3b-pt-224 target:/home/yz/modelZoo/
```

如果只迁移 checkpoint 而不是整个 RLinf logs 目录，确保目标路径仍和 JSON 中的 `model_path` 一致。

### 5. 在目标服务器加载镜像

未压缩 tar：

```bash
docker load -i /home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar
```

如果是 gzip 压缩包：

```bash
gunzip -c /home/yz/projects/outputs/dokcer_imgs/rlinf_agentic-rlinf0.2-maniskill_libero.tar.gz | docker load
```

加载后确认镜像存在：

```bash
docker images docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero
```

### 6. 在目标服务器启动容器

推荐保持 `/home/yz:/home/yz` 挂载不变，这样 JSON 和脚本中的绝对路径无需修改：

```bash
docker run -d --gpus all \
   --shm-size 120g \
   --network host \
   --name rlinf \
   -v /home/yz:/home/yz \
   docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero \
   tail -f /dev/null
```

进入容器并切换 OpenPI 环境：

```bash
docker exec -it rlinf /bin/bash
source switch_env openpi
cd /home/yz/projects/Evo-RL-loop-0810/src
```

### 7. 部署后检查

先检查 Stage1 parity，确认 checkpoint、norm stats 和 RLinf-backed 推理路径可用：

```bash
python -m lerobot.onlineRL.compare_stage1_parity \
  --random_sample \
  --seed 0 \
  --with_rlinf \
  --no_inspect_inputs
```

再检查 onlineRL 配置是否能加载 RLinf-backed extractor。真机前可以先 dry-run：

```bash
python -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --dry_run
```

### 8. 在目标服务器运行 actor

连接真机时，确认 CAN、相机设备号、Piper 校准和权限已经在目标服务器上配置好。重点检查：

```text
hardware.follower_port
hardware.leader_port
hardware.top_camera.index_or_path
hardware.wrist_camera.index_or_path
```

actor-only 模式：

```bash
python -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --mode actor \
  --no-dry_run
```

本机 actor + learner 模式：

```bash
python -m lerobot.onlineRL.run \
  --config_path lerobot/onlineRL/configs/piper_rlt_ac_packageSorting_rlinf.json \
  --no-dry_run
```

### 9. 常见问题

- 如果 `docker run` 提示容器名已存在，先执行 `docker ps -a --filter name=rlinf`，确认后再 `docker stop rlinf && docker rm rlinf`。
- 如果 `source switch_env openpi` 不存在，说明目标镜像不是当前 RLinf 镜像，或环境未正确加载。
- 如果报路径不存在，优先检查 `piper_rlt_ac_packageSorting_rlinf.json` 中的 `rlinf_path`、`model_path`、`norm_stats_path` 和 `tokenizer_name`。
- 如果相机或 CAN 不可用，先在目标服务器重新确认 `/dev/video*` 编号、`can0/can1` 名称、用户权限和硬件连接。
- 如果 parity 中 `RLinf action` 能正常输出，但真机动作异常，优先检查 action 尺度、`hardware.action_limit`、动作平滑和实际下发日志中的 `command/executed`。

