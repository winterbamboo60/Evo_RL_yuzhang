# onlineRL_evoRL

本目录是 PI05-RLT 双臂 Piper 在线强化学习的完整运行时。Actor、独立 learner、在线 episode 组装、人工接管、RTC、GPU handoff、checkpoint、配置约束和启动前预检都由本目录维护；只复用 LeRobot 的 replay/trainer 等底层数据结构和 gRPC protobuf 字节传输。

当前可执行入口：

- Learner：`python -m lerobot.onlineRL_evoRL.learner`
- Actor：`python -m lerobot.onlineRL_evoRL.actor_new`
- 离线特征提取：`python -m lerobot.onlineRL_evoRL.extract_offline_features`
- 配置预检：`python -m lerobot.onlineRL_evoRL.preflight`
- 在线推荐启动脚本：`scripts/RL_online.sh`
- 离线提取推荐启动脚本：`scripts/RL_extract_offline_features.sh`

不要再使用 python -m lerobot.rl.learner 启动本项目。该入口不会经过本目录的 PI05-RLT 配置约束。

## 当前双臂任务

默认启动脚本指向以下配对配置：

- Learner：configs/learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json
- Actor：configs/actor/Actor_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json
- 数据集：/home/lenovo/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask
- 离线 compact 数据：/home/lenovo/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask_pi05_rlt_compact_stride2
- 已准备模型：/home/lenovo/outputs/pretrained_model
- 在线输出：/home/lenovo/datasets/online_rl_outbox/0915_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k

当前模型和数据集均为：

- policy.type = pi05_rlt
- algorithm.type = rlt_chunk
- observation.state = 14 维
- action = 14 维
- 三路图像：left_wrist、left_top、right_wrist
- chunk_size = 50
- 数据集与控制频率 = 30 FPS

双臂 CAN 分配：

- 左 follower：can1
- 右 follower：can3
- 左 leader：can0
- 右 leader：can2

四个端口必须唯一。启动前仍应确认 CAN 设备存在、机械臂完成标定且急停与工作空间安全。

## 与 0901 方案的区别

0901 目录中保存了一份完整 learner 分叉，但其 RL_online.sh 仍然启动 lerobot.rl.learner，导致目录内实现事实上没有执行。当前入口不再委托 `lerobot.rl.learner`，EvoRL 的训练生命周期在本目录内闭合。

当前方案采用明确边界：

1. scripts/RL_online.sh 从 lerobot.onlineRL_evoRL.learner 启动。
2. learner.py 独立管理服务、compact replay、更新额度、GPU handoff 和 checkpoint，不导入 `lerobot.rl.learner`。
3. actor_new.py 仍是唯一硬件 Actor。
4. preflight.py 在连接硬件前检查模型、数据集、14 维特征、三路相机、CAN 端口和配置配对。
5. Learner 只从 policy 配置读取 RLT 维度并构造小型 actor/critic，不实例化 5B PI0.5。

因此，在线项目入口、训练状态机和任务约束都在本目录；公共模块不再决定 EvoRL learner 的模型加载和生命周期。

## 在线数据流

1. Actor 从双臂 Piper 和三路相机取得 observation。
2. PI05-RLT VLA 产生参考 action chunk，并提取 z_rl、proprio 和 ref_action。
3. RTC 执行动作；按 C 可切换人工接管。
4. episode 结束后，Actor 构造 sliding-window compact transitions。
5. Actor 必须先把 compact episode 保存到 `episode_output_dir`。
6. 本地保存成功后，同一个 compact payload 经 gRPC 发送给 Learner。
7. `save_lerobot_copy=true` 时，同一 accepted episode 还写成与 RL_data 兼容的标准 LeRobotDataset。
8. Learner 将 compact transitions 写入 online replay。
9. 每个去重后的完整 episode 增加 `gpu_handoff.updates_per_episode` 额度；额度达到双方相同的 `update_quota_threshold` 后触发一轮训练。
10. 启用 handoff 时 Actor 先释放 PI0.5 CUDA 内存，Learner 执行恰好 threshold 次更新并释放 CUDA，再把在线 actor 权重和 handoff ID 发回 Actor。
11. Actor 重新加载 PI0.5 和最新在线 actor 权重，所有机械臂归位后恢复采集。

当前 0915 Learner 启用了可选的离线启动阶段，但不再读取原始 LeRobotDataset，也不会实例化 5B PI05。原始数据先由独立的 `extract_offline_features` 程序转换为与 Actor 本地保存完全相同的 `episode_xxxxxx/compact_episode.pt + metadata.json`；Learner 从 `compact_dataset_path` 加载这些文件，再训练 RLT Actor/Critic 20000 step。

提取程序显示每张 GPU 的帧级 `z_rl` 进度条；Learner 分别显示 `Offline compact loading` 和 `Offline Actor/Critic training`。所有离线 episode 都按成功处理（仅 terminal reward=1），所有有效 action 都按人工介入示教处理。离线训练完成后会保存并下发初始 Actor 权重，释放离线 replay，再启动 gRPC 服务进入原在线循环；离线 replay 不与在线 replay 混采。

## 离线 compact 数据准备与预训练

### 数据边界与输出格式

原始 LeRobotDataset 只由独立提取器读取。提取器在 `torch.inference_mode()` 下加载 PI0.5，批量
提取 `z_rl`、proprio 和 reference action，再按 sliding window 生成与在线 Actor 本地保存兼容的
compact transition。这个过程只做推理，不训练 Actor/Critic，也不会产生梯度。

Learner 顶层 `dataset` 必须为 `null`。它只读取提取后的 compact 数据、构造小型 RLT
Actor/Critic 并完成配置的离线训练，不再加载原始图像、视频或 5B PI0.5。每个离线 episode
统一按成功示教处理：仅 terminal reward 为 1，所有有效 action 的 intervention mask 为真。

输出目录结构如下：

```text
<output_dir>/
├── manifest.json
├── rank_000_manifest.json
├── rank_001_manifest.json
├── episode_000000/
│   ├── compact_episode.pt
│   └── metadata.json
└── episode_000001/
    ├── compact_episode.pt
    └── metadata.json
```

`manifest.json` 记录源数据、模型身份、期望/完成 episode 数、transition 数、源帧数、stride、
world size 和存储 dtype。只有所有请求的 episode 都成功写入，且输出中没有额外 episode 时，rank
0 才会原子写入全局 manifest。每个 episode 也通过临时目录原子落盘，避免中断后把半写入文件
误判为完成。

### 查看帮助

```bash
cd /home/lenovo/code/Evo-RL-loop-0911
bash scripts/RL_extract_offline_features.sh --help
```

脚本默认使用当前 0915 Learner JSON 作为特征维度、相机、chunk size 和模型身份契约。云端应有
同一版本的仓库、该配置文件、原始 LeRobotDataset 和 PI0.5 checkpoint。

### 单 GPU 提取

```bash
cd /home/lenovo/code/Evo-RL-loop-0911
EVORL_EXTRACT_DATASET_ROOT=/home/lenovo/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask \
EVORL_EXTRACT_OUTPUT_DIR=/home/lenovo/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask_pi05_rlt_compact_stride2 \
EVORL_EXTRACT_POLICY_PATH=/home/lenovo/outputs/pretrained_model \
EVORL_EXTRACT_BATCH_SIZE=8 \
EVORL_EXTRACT_WORKERS=4 \
bash scripts/RL_extract_offline_features.sh --episodes 0:1247
```

### 单机多 GPU 提取

推荐通过 wrapper 启动。`EVORL_EXTRACT_GPUS=N` 会调用 `torchrun --standalone
--nproc-per-node N`，每个进程固定使用一张 GPU，并以 `episodes[rank::world_size]` 的方式分配
完整 episode。进程间不做模型或梯度同步。

```bash
cd /cloud/Evo-RL-loop-0911
EVORL_ENV_ROOT=/cloud/envs/evo_0911 \
EVORL_EXTRACT_GPUS=4 \
EVORL_EXTRACT_BATCH_SIZE=32 \
EVORL_EXTRACT_WORKERS=8 \
EVORL_EXTRACT_STRIDE=2 \
EVORL_EXTRACT_STORAGE_DTYPE=float32 \
EVORL_EXTRACT_DATASET_ROOT=/cloud/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask \
EVORL_EXTRACT_OUTPUT_DIR=/cloud/outputs/20260915_bipiper_cube_catch_v2-1_compact_stride2 \
EVORL_EXTRACT_POLICY_PATH=/cloud/models/pretrained_model \
bash scripts/RL_extract_offline_features.sh --episodes 0:1247
```

每张 GPU 都有独立的帧级 `GPU <rank>/<world_size> z_rl` 进度条。上例中的 batch size 32 和
workers 8 都是**每个 GPU 进程**的值，四卡总推理 batch 上限为 128，总 DataLoader worker
数为 32。显存不足时先降低 `EVORL_EXTRACT_BATCH_SIZE`；CPU 内存、文件句柄或视频解码压力过高
时降低 `EVORL_EXTRACT_WORKERS`。

也可绕过 wrapper 直接启动：

```bash
cd /cloud/Evo-RL-loop-0911
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
/cloud/envs/evo_0911/bin/torchrun --standalone --nproc-per-node 4 \
  -m lerobot.onlineRL_evoRL.extract_offline_features \
  --config-path src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json \
  --dataset-root /cloud/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask \
  --dataset-repo-id local/20260915_bipiper_cube_catch_v2-1_merged_newTask \
  --output-dir /cloud/outputs/20260915_bipiper_cube_catch_v2-1_compact_stride2 \
  --policy-path /cloud/models/pretrained_model \
  --batch-size 32 \
  --num-workers 8 \
  --stride 2 \
  --episodes 0:1247
```

### 参数说明

Wrapper 通过以下环境变量配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EVORL_ENV_ROOT` | `/home/lenovo/code/envs/evo_0911` | Python/torchrun 环境根目录 |
| `EVORL_EXTRACT_CONFIG` | 当前 0915 Learner JSON | 提供 policy、env 和 feature contract |
| `EVORL_EXTRACT_DATASET_ROOT` | 无，必填 | 原始 LeRobotDataset 根目录 |
| `EVORL_EXTRACT_OUTPUT_DIR` | 无，必填 | compact 输出目录；不要混用不同 episode 集合或契约 |
| `EVORL_EXTRACT_DATASET_REPO_ID` | `local/offline_feature_extraction` | 写入 metadata 的源数据身份 |
| `EVORL_EXTRACT_GPUS` | `1` | 本机使用的 GPU/进程数量 |
| `EVORL_EXTRACT_BATCH_SIZE` | `8` | 每个 GPU 进程的推理 batch size |
| `EVORL_EXTRACT_WORKERS` | `4` | 每个 GPU 进程的 DataLoader worker 数 |
| `EVORL_EXTRACT_STRIDE` | `2` | sliding-window stride，当前须与 Learner 的 2 一致 |
| `EVORL_EXTRACT_STORAGE_DTYPE` | `float32` | compact 张量存储类型：`float32`、`float16` 或 `bfloat16` |
| `EVORL_EXTRACT_POLICY_PATH` | Learner JSON 的模型路径 | 当前机器实际加载的 checkpoint |
| `EVORL_EXTRACT_FEATURE_MODEL_PATH` | Learner JSON 的模型路径 | 写入 payload 的模型身份；仅修改契约时使用 |

Wrapper 后面可以继续传原生 CLI 参数：

- `--episodes 0:1247`：左闭右开范围，即 episode 0 到 1246。
- `--episodes 0:100,200:300:2`：支持逗号组合和步长。
- 省略 `--episodes`：处理数据集全部 episode。
- `--max-episodes N`：从解析后的 episode 列表截取前 N 个，适合小规模验证。
- `--resume` / `--no-resume`：默认开启断点续跑。
- `--device cuda`：单进程设备；多卡时各 rank 自动映射到对应 CUDA 设备。

`float16`/`bfloat16` 可减少产物体积和读取 I/O，但当前 Learner replay 最终使用 float32
storage，因此不会等比例降低训练期 replay 内存。第一次正式提取建议保留 `float32`；使用低精度前
先抽取少量 episode，比较离线训练稳定性。

### 云端模型路径与特征契约

云端 checkpoint 路径与机器人本机不同时，只设置 `EVORL_EXTRACT_POLICY_PATH`。例如云端从
`/cloud/models/pretrained_model` 实际加载模型，但 payload 仍默认写入 Learner JSON 中的
`/home/lenovo/outputs/pretrained_model` 作为模型身份，这样回传后可以直接通过本机契约检查。

不要仅因云端目录不同就设置 `EVORL_EXTRACT_FEATURE_MODEL_PATH`。该变量会改变产物中的模型
身份，只有同时有意修改 Learner 的 `policy.pretrained_path` 契约时才使用。

### 断点续跑

默认 `--resume`。模型身份和 stride 一致、且 `compact_episode.pt` 与 `metadata.json` 均有效的
episode 会被跳过；缺失或不兼容的已有 episode 目录会报
`Existing episode is incomplete or incompatible`，不会被静默覆盖。人工确认后将该 episode 目录
移走，再用完全相同的命令继续即可。

不要在同一个输出目录先后请求不同的 episode 集合，例如先提取 `0:100`，再直接提取
`0:1247`。全局 manifest 要求输出中的 episode 恰好等于本次请求集合。小规模验证应使用独立的
测试输出目录；`--no-resume` 也应只配合空目录或全新的输出目录使用。

### 完整性检查与回传

只有成功生成全局 `manifest.json` 的目录才能交给 Learner。云端提取完成后运行：

```bash
COMPACT_DIR=/cloud/outputs/20260915_bipiper_cube_catch_v2-1_compact_stride2
/cloud/envs/evo_0911/bin/python -c \
  'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); print(json.dumps(m, indent=2, ensure_ascii=False)); assert m["expected_episodes"] == m["completed_episodes"]; assert len(list(p.glob("episode_*/compact_episode.pt"))) == m["completed_episodes"]' \
  "$COMPACT_DIR"
```

然后将整个目录回传到 Learner JSON 的 `offline_pretraining.compact_dataset_path`。尾部 `/` 表示
复制目录内容，`manifest.json`、每个 episode 的 metadata 和 `.pt` 都必须保留：

```bash
rsync -a --info=progress2 \
  /cloud/outputs/20260915_bipiper_cube_catch_v2-1_compact_stride2/ \
  lenovo@<robot-ip>:/home/lenovo/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask_pi05_rlt_compact_stride2/
```

### 启动 Learner

当前 Learner 配置关键字段为：

```json
{
  "dataset": null,
  "offline_pretraining": {
    "enabled": true,
    "steps": 20000,
    "feature_batch_size": 16,
    "sliding_window_stride": 2,
    "compact_dataset_path": "/home/lenovo/datasets/20260915_bipiper_cube_catch_v2-1_merged_newTask_pi05_rlt_compact_stride2"
  }
}
```

回传完成后正常启动：

```bash
cd /home/lenovo/code/Evo-RL-loop-0911
bash scripts/RL_online.sh learner
```

预检先验证 manifest 的期望/完成 episode 数和实际文件数。Learner 随后显示
`Offline compact loading` 进度条，将 compact transition 装入临时离线 replay；再显示
`Offline Actor/Critic training` 进度条，训练配置的 20000 step。完成后保存并下发初始 Actor
权重，释放离线 replay，再启动 gRPC 服务进入原有在线循环。离线 replay 不与后续在线 replay
混采；在线旧数据仍由在线 replay/checkpoint 独立保留。

Actor 配置中的训练 `dataset.root` 仍用于 task metadata、特征契约和归一化处理，但不进入当前 learner replay 混采。

每个 episode 结束后，Actor 会像 RL_data 一样持续刷新 follower 的终止姿态，并同步任何可执行 leader，直到复位、下一 episode 或 handoff 完成，避免等待保存和训练时机械臂下坠。Actor 首次启动及每次重新获得 GPU 后都会把全部关节移动到 0，并把夹爪设置为 `max_gripper_pos`（当前为 100.0）。

## 启动前预检

在仓库根目录执行：

    cd /home/lenovo/code/Evo-RL-loop-0911
    /home/lenovo/code/envs/evo_0911/bin/python -m lerobot.onlineRL_evoRL.preflight \
      --learner-config src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json \
      --actor-config src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json \
      --mode both

预检不会连接机械臂，也不会加载完整模型到 GPU。它会检查：

- actor/learner 都是 pi05_rlt + rlt_chunk；
- 两端 policy 和 algorithm 关键字段一致；
- 模型目录包含 config、safetensors 和 pre/postprocessor；
- 模型的特征、chunk 和 RLT 参数与配置一致；
- 数据集 info.json 的状态、动作、相机和 action names 与模型一致；
- Learner 顶层 `dataset` 始终为 `null`；启用离线阶段时，compact 目录必须有完整
  `manifest.json`，期望/完成 episode 数必须相等，实际 episode 文件数也必须一致；
- 双端 GPU handoff 的 enabled、threshold 和 updates_per_episode 一致；threshold 不固定为 200；
- compact 本地副本必开，启用 LeRobot 副本时两个输出目录存在配置且互不相同；
- actor_checkpoint_path 指向 Learner output_dir；
- Piper robot/teleop 类型和四个 CAN 端口正确；
- RTC execution_horizon 不超过 chunk_size。

RL_online.sh 会自动执行对应模式的预检。预检失败时不会启动 Learner 或连接硬件。

## 启动方式

先看实际默认配置：

    bash scripts/RL_online.sh --help

推荐用两个终端，便于独立观察日志和停止进程。

终端 1：

    cd /home/lenovo/code/Evo-RL-loop-0911
    bash scripts/RL_online.sh learner

终端 2：

    cd /home/lenovo/code/Evo-RL-loop-0911
    bash scripts/RL_online.sh actor

也可以同时启动：

    cd /home/lenovo/code/Evo-RL-loop-0911
    bash scripts/RL_online.sh both

both 模式把日志写到：

- outputs/online_rl/logs/learner.log
- outputs/online_rl/logs/actor.log

需要临时替换配置时，使用：

    PIPER_ONLINE_RL_LEARNER_CONFIG=/abs/learner.json \
    PIPER_ONLINE_RL_ACTOR_CONFIG=/abs/actor.json \
    bash scripts/RL_online.sh both

both 模式后附的 Draccus 参数只传给 Actor；Learner 参数应写入其独立 JSON 或使用单独 learner 命令。

## Actor 热键

- C：切换人工接管
- B：结束当前 episode，标记成功，reward = 1
- F：结束当前 episode，标记失败，reward = 0
- A：放弃并重录当前 episode
- R：复位并重录当前 episode
- ESC：停止 Actor

Learner 权重可用后，V 可在 VLA 与在线 Actor 间切换。若配置 task_hotkeys_path，还可以用配置中的任务键切换 task；当前默认值为 null，因此任务提示由 env.task 固定提供。

热键优先使用 pynput 全局监听；不可用时回退到当前 TTY。若两者都不可用，日志会明确提示热键被禁用。

## 关键配置规则

Learner：

- `policy.type=pi05_rlt`、`algorithm.type=rlt_chunk`、`online_ratio=1.0` 是一组完整契约：PI05-RLT 产出 RLT 特征，rlt_chunk 消费 compact action-chunk transition，1.0 表示 mixer 只从该在线 replay 取样。
- Learner 的 `dataset` 必须始终为 `null`；启用离线阶段时必须提供完整的 `compact_dataset_path`。
- 当前 0915 配置的离线阶段为 20000 step、sliding-window stride 2；在线阶段仍使用独立的 online replay。
- 独立提取器使用 `torch.inference_mode()`；Actor 和 Critic 的梯度只存在于 learner 的小型 RLT heads。
- `gpu_handoff.update_quota_threshold` 是一次 handoff 训练额度，actor 和 learner 只要求取值相同；当前配置为 200。
- `gpu_handoff.updates_per_episode` 当前为 40，因此当前每 5 个 accepted episode 触发一次 handoff。
- `algorithm.online_step_before_learning` 仍只约束在线 replay warmup，与启动时的离线 Actor/Critic 预训练分开计数。
- algorithm.online_steps 是 Actor interaction loop 上限并参与 checkpoint 编号；Learner 本身在收到退出信号前持续运行。顶层 steps 不是当前在线循环的停止计数。
- save_freq 以 optimization step 为单位。
- resume=false 时，已有 checkpoints/last 会阻止覆盖。
- resume=true 时，从 output_dir/checkpoints/last 恢复 optimizer、algorithm 和 interaction step。

Actor：

- actor_mode 必须为 online_actor。
- save_format 必须为 transition。
- online_transition.enabled 和 actor_vla_policy.enabled 必须为 true。
- `online_transition.save_local_copy` 必须为 `true`；compact transition 本地副本不可关闭。
- `online_transition.episode_output_dir` 保存传输用的 `compact_episode.pt` 与 metadata。
- `online_transition.save_lerobot_copy` 控制是否额外保存标准 LeRobotDataset；当前双臂配置为 `true`。
- `online_transition.lerobot_output_dir` 是 LeRobotDataset 根目录，必须与 compact 目录不同且首次启动时不能已存在。
- actor_vla_policy.policy_path 应与 policy.pretrained_path 指向同一模型。
- actor_checkpoint_path 必须等于 Learner output_dir。
- RTC execution_horizon 当前为 25，不能大于 chunk_size 50。
- sliding_window_stride 当前为 2；增大可减少发送样本量，但会降低窗口密度。
- 每个 episode 结束立即进入持续保持姿态；首次启动和每次 GPU reacquire 后强制执行全零关节、最大夹爪归位。

## Checkpoint 与权重同步

Learner checkpoint 不复制或保存冻结的 5B PI0.5，包含：

- algorithm：在线 actor、critic ensemble 和 target critic；
- training_state：optimizer、离线/在线独立 step、离线完成标记、剩余额度、episode 去重集合和 handoff ID；
- compact_replay.pt：纯在线 compact replay 的持久化副本；
- train_config.json：PI0.5 特征契约和源 checkpoint 引用。

实时下发给 Actor 的是 rlt_chunk actor 权重，不是整套 PI05-RLT 大模型。Actor 在 episode 边界接收最新权重；Learner 尚未下发时，Actor 使用 VLA 参考动作继续运行。

## 常见故障

预检报告模型目录不存在：

确认两个 JSON 中 policy.pretrained_path 和 actor_vla_policy.policy_path。当前有效目录是 /home/lenovo/outputs/pretrained_model。

预检报告模型字段不一致：

不要只改在线 JSON 的维度或相机键。模型 config.json、数据集 meta/info.json、Actor policy、Learner policy 和 env.features/features_map 必须共同一致。

Learner 一启动就退出：

查看 Learner output_dir 是否已有 checkpoints/last。继续训练应设置 resume=true；新实验应换 output_dir。both 模式也会把启动阶段错误打印到终端并保留 learner.log。

预检报告 `compact manifest is missing`：

离线提取尚未完成，或云端产物没有被整体复制到
`offline_pretraining.compact_dataset_path`。不要创建空 manifest 绕过检查；确认全局
`manifest.json`、所有 `episode_*/compact_episode.pt` 和对应 `metadata.json` 均已回传。

预检报告 compact episode 数与 manifest 不一致：

回传中断、漏传了 episode，或把不同提取任务写进了同一目录。重新执行 rsync 补齐文件，并用
本节的完整性检查命令验证；不同 `--episodes`、stride 或模型契约必须使用不同输出目录。

提取器报告 `Existing episode is incomplete or incompatible`：

该目录可能是中断残留，或者由不同模型身份/stride 生成。先查看该 episode 的 `metadata.json`，
确认无须保留后将整个冲突 episode 目录移到输出目录之外，再用原命令断点续跑。提取器不会自动
覆盖它。

提取器 CUDA OOM：

`EVORL_EXTRACT_BATCH_SIZE` 是每卡值，先减小该值；增加 GPU 数只会把 episode 分片到更多独立
进程，不会分摊单个模型的显存。`EVORL_EXTRACT_STORAGE_DTYPE` 控制落盘张量类型，不代表
PI0.5 推理本身会以该精度加载，因此不能把它当作主要的显存开关。

云端提取成功，但本机报告 feature model 不匹配：

云端路径不同应只通过 `EVORL_EXTRACT_POLICY_PATH` 指定实际 checkpoint，不应覆盖
`EVORL_EXTRACT_FEATURE_MODEL_PATH`。若已经用错误身份完成提取，应使用正确契约重新导出，或在
明确确认模型相同后统一修改配置和产物契约；不要关闭校验。

Actor 无法连接 127.0.0.1:50051：

先确认 Learner 仍在运行，再确认两端 algorithm.actor_learner_config 的 host/port 完全一致。跨机器运行时，Learner host 不能继续使用仅本机可见的 127.0.0.1。

Actor 找不到初始在线权重：

启用离线阶段时，Learner 完成预训练后会在 output_dir/algorithm 保存 Actor/Critic，并在启动服务后发送初始 Actor 权重。Actor 只把权重标记为可用，仍保持 VLA 输出；必须主动按 V 才会切换到在线 Actor。

Actor 报告 LeRobotDataset 输出目录已存在：

标准 LeRobotDataset writer 为避免覆盖已有数据，要求 `lerobot_output_dir` 首次启动时不存在。请为新一次采集换新目录，或先人工确认并迁移旧目录；compact `episode_output_dir` 支持按 episode 编号续写。

显存或推理频率不足：

三路 480x640 图像最终按模型的 image_resolution 处理。优先检查 GPU 占用、RTC horizon、feature_batch_size 和相机 FPS，不要通过删除相机特征绕过，因为那会破坏模型与数据集契约。

## 无硬件验证

以下检查不会驱动机械臂：

    bash -n scripts/RL_extract_offline_features.sh
    bash scripts/RL_extract_offline_features.sh --help
    bash -n scripts/RL_online.sh
    bash scripts/RL_online.sh --help
    /home/lenovo/code/envs/evo_0911/bin/python -m lerobot.onlineRL_evoRL.preflight \
      --learner-config src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json \
      --actor-config src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_rlt_sft_20260915_bipiper_cube_catch_v21_merged_newTask_sft30k_rlt2k.json \
      --mode both

最后一条预检在启用离线阶段时要求 compact 数据已经放到配置路径；尚未回传时因 manifest 缺失
而失败是预期行为。
