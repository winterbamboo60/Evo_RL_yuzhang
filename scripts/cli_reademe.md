# Evo 0911 环境打包、上传与云端使用

`evo_0911` 是基于 Ubuntu 24.04、x86_64、Python 3.12 的 uv/venv 环境，不是 Conda
环境。它使用 `venv-pack` 生成可迁移归档，解压后不需要、也不能运行 `conda-unpack`。

归档内包含当前项目工作树构建出的非 editable LeRobot，因此只上传环境也能直接运行。
云端需要修改 LeRobot 源码时，再从云端项目目录执行 editable 重装；`--no-deps` 会保留
归档中的 PyTorch、CUDA、RealSense、Piper 等依赖版本。

## 1. 本地打包

```bash
cd /home/lenovo/code/Evo-RL-loop-0911
bash scripts/ci/env/pack_evo_0911.sh \
  --env-root /home/lenovo/code/envs/evo_0911 \
  --project-root /home/lenovo/code/Evo-RL-loop-0911 \
  --output-dir /home/lenovo/code/envs/artifacts/evo_0911
```

脚本不会修改源环境或项目工作树。输出目录包含：

- `evo_0911-*.tar.gz`：可以直接上传的环境归档；
- `evo_0911-*.tar.gz.sha256`：完整性校验文件；
- `evo_0911-*.manifest.txt`：构建主机、Python、Git 提交和 GPU 信息。

完整的 `pip freeze`、Git 工作树状态和 LeRobot wheel 校验值也保存在归档内的
`share/evo_0911/`。打包结束前会把归档解压到另一个临时路径，检查旧前缀、核心依赖、
LeRobot 安装位置和 `pip check`。

## 2. 上传云端

将项目目录以及 `.tar.gz`、`.sha256` 上传到云端。SSH 场景可以使用：

```bash
rsync -P \
  /home/lenovo/code/Evo-RL-loop-0911/artifacts/evo_0911/evo_0911-*.tar.gz \
  /home/lenovo/code/Evo-RL-loop-0911/artifacts/evo_0911/evo_0911-*.tar.gz.sha256 \
  USER@CLOUD_HOST:/cloud/uploads/
```

没有 SSH 的平台可直接在网页上传同样两个文件。不要只上传当前 Git 提交后重新构建环境：
当前工作树可能包含未提交的 LeRobot 功能，归档中的非 editable wheel 才是打包时的准确快照。

## 3. 云端解压和验证

目标目录必须尚不存在，安装脚本不会覆盖已有环境：

```bash
cd /cloud/code/Evo-RL-loop-0911
bash scripts/ci/env/install_evo_0911.sh \
  --archive /cloud/uploads/evo_0911-YYYYMMDD-HHMMSS-*.tar.gz \
  --dest /cloud/envs/evo_0911

export EVORL_ENV_ROOT=/cloud/envs/evo_0911
source "$EVORL_ENV_ROOT/bin/activate"

export EVORL_ENV_ROOT=/home/lenovo/code/envs/evo_0911
source "$EVORL_ENV_ROOT/bin/activate"
```

安装脚本会自动读取同目录的 `.sha256`、检查归档成员、防止覆盖目标目录，并在临时目录
验证通过后再原子移动到目标路径。需要强制验证 GPU 时追加 `--require-cuda`。

目标机要求：

- Ubuntu 24.04 x86_64，且 `/usr/bin/python3.12` 存在；
- NVIDIA 驱动能够运行归档中的 PyTorch CUDA 12.8；
- CAN、RealSense、udev 权限、`can-utils` 和 `ethtool` 由宿主机准备。

若系统或架构不同，不要强行使用二进制归档，应在目标机从 `uv.lock` 重建环境。

## 4. 可选：重新安装云端 LeRobot 源码

```bash
cd /cloud/code/Evo-RL-loop-0911
bash scripts/ci/env/reinstall_lerobot.sh \
  --env-root /cloud/envs/evo_0911 \
  --project-root /cloud/code/Evo-RL-loop-0911
```

也可以在解压时一次完成：

```bash
bash scripts/ci/env/install_evo_0911.sh \
  --archive /cloud/uploads/evo_0911-YYYYMMDD-HHMMSS-*.tar.gz \
  --dest /cloud/envs/evo_0911 \
  --project-root /cloud/code/Evo-RL-loop-0911 \
  --editable
```

editable 重装后，LeRobot 会从项目的 `src/` 目录加载。此后不要移动该环境目录；需要换路径时
重新从原始归档解压。重装不会访问网络，也不会安装或升级依赖。

## 5. 环境变量约定

本机默认路径如下；云端只需修改一次 `EVORL_ENV_ROOT`。`RL_online.sh` 会直接读取该变量，
数据采集和训练脚本则从已激活环境的 `PATH` 获取 Python。

```bash
export EVORL_ENV_ROOT="${EVORL_ENV_ROOT:-/home/lenovo/code/envs/evo_0911}"
source "$EVORL_ENV_ROOT/bin/activate"
```

# 找相机
mkdir -p /home/lenovo/code/Evo-RL-loop-0911/outputs/camera_check

source "$EVORL_ENV_ROOT/bin/activate"
source "$EVORL_ENV_ROOT/bin/activate"; cd /home/lenovo/code/Evo-RL-loop-0911; lerobot-find-cameras opencv
source "$EVORL_ENV_ROOT/bin/activate"; cd /home/lenovo/code/Evo-RL-loop-0911; lerobot-find-cameras realsense --output-dir /home/lenovo/code/Evo-RL-loop-0911/outputs/camera_check

调整相机参数


# 激活can口
Step3. 初始化CAN口
source "$EVORL_ENV_ROOT/bin/activate"
lerobot-setup-can --mode=setup --interfaces=can0,can1

默认 CAN 映射：
左主臂 can0 -> 左从臂 can1
右主臂 can2 -> 右从臂 can3
source "$EVORL_ENV_ROOT/bin/activate"
lerobot-setup-can --mode=setup --interfaces=can0,can1,can2,can3

Step4. CAN口模式测试，需要能看到持续输出的数据
lerobot-setup-can --mode=test --interfaces=can0
lerobot-setup-can --mode=test --interfaces=can1

# 北京5080数据采集
## 纯人工

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
  --dataset.root /home/lenovo/datasets/cube_catch_rollout_v3/0911_1 \
  --dataset.single_task "Grab the cube" \
  --wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6

## 策略模型 + 人工介入（显式启用 can0）
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
  --dataset.root /home/lenovo/datasets/0909_pi05_sft_cube_catch_belt_50k_test1 \
  --dataset.single_task "Grab the moving blocks on the conveyor belt" \
  --wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6 \
  --policy.path /home/lenovo/outputs/0909_pi05_sft_cube_catch_belt_50k \
  --can0.control true


## 合并数据集
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id /home/lenovo/datasets/v9_task123_0728_merged \
    --operation.type merge \
    --operation.repo_ids "['/home/lenovo/datasets/v9_task2_0728/v9_task2_0728_merged', '/home/lenovo/datasets/task0_grab_the_package_and_place_it_on_the_pal', '/home/lenovo/datasets/task2_grab_the_package_and_place_it_into_the_b']"

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id /home/lenovo/datasets/20260914_bipiper_cube_catch_v2_merged \
    --operation.type merge \
    --operation.source_dir /home/lenovo/datasets/20260914_bipiper_cube_catch_v2

source "$EVORL_ENV_ROOT/bin/activate"
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id /home/lenovo/datasets/cube_catch_rollout_v3_merge_test  \
    --operation.type merge \
    --operation.source_dir /home/lenovo/datasets/cube_catch_rollout_v3 \
    --operation.concatenate_videos false \
    --operation.concatenate_data false

## 查看并编辑数据集
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
python /home/lenovo/code/Evo-RL-loop-0911/scripts/ci/dataset_checker.py /home/lenovo/datasets/20260914_bipiper_cube_catch_v2/0915_1

## 修改数据集task
source "$EVORL_ENV_ROOT/bin/activate"
python /home/lenovo/code/Evo-RL-loop-0911/scripts/ci/replace_dataset_task.py /home/lenovo/datasets/20260914_bipiper_cube_catch_v2_merged --task "Sort the moving blocks on the conveyor belt: use the left arm to place only yellow blocks into the left basket, and use the right arm to place only red blocks into the right basket." --output-suffix "_newTask"

## 数据集转化

旧数据已经包含 complementary_info.policy_action、complementary_info.is_intervention、
complementary_info.state、complementary_info.collector_policy_id 时，可以由当前训练代码直接读取，
无需转换。缺少这些字段或使用临时 intervention bool 字段时，可非覆盖地规范化：

lerobot-migrate-evorl-dataset \
    --source-root SOURCE_DATASET \
    --destination-root DESTINATION_DATASET \
    --destination-repo-id local/canonical_dataset \
    --direction to-canonical \
    --collector-policy-id pi05-checkpoint

destination-root 是新副本的实际磁盘目录，destination-repo-id 是写入 meta/info.json
的数据集标识。转换不改源目录，输出仍使用当前 LeRobot v3 的 Parquet/MP4 存储；
只有 EvoRL 业务字段名和 dtype 对齐 0901。to-current 和 to-0901 仅保留作历史兼容方向。


# 模型训练
## pi05_base训练
需要根据机器修改环境ENV_ROOT的值

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
nohup bash scripts/RL_train.sh \
    --DATASET_ROOT /home/lenovo/datasets/20260914_bipiper_cube_catch_v1-1_merged \
    --ModelZoo /home/lenovo/modelZoo \
    --history_pretrained_path /home/lenovo/outputs/0914_pi05_sft_cube_catch_belt_dual_double_30000 \
    --OUTPUT_DIR /home/lenovo/outputs/0915_pi05_sft_cube_catch_belt_dual_v11_double_30000_needDelete \
    --policy_type pi05 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --steps 1000 \
    --save_freq 200 \
    --train_expert_only true \
    -- \
    --policy.compile_model=false \
    > /home/lenovo/outputs/logs/0915_pi05_sft_cube_catch_belt_dual_v11_double_30000_needDelete.log 2>&1 &

### 单机多卡 DDP 训练

当前 `lerobot_train.py` 使用 `torchrun`/PyTorch Distributed 作为多卡启动方式。`RL_train.sh`
增加 `--num_gpus N` 后会自动改用同一 Python 环境里的
`python -m torch.distributed.run --standalone`，并让当前 LeRobot 将默认并行拓扑解析为 DDP。
不要再套用 0901 的 `accelerate launch`：当前版本要求 Accelerate 参数统一来自
`TrainPipelineConfig`，外部 Accelerate 配置环境变量可能被拒绝。

两卡 PI0.5 示例（只需在单卡命令中选择显卡并增加 `--num_gpus 2`）：

```bash
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
CUDA_VISIBLE_DEVICES=0,1 bash scripts/RL_train.sh \
    --DATASET_ROOT /home/lenovo/datasets/20260914_bipiper_cube_catch_v1-1_merged \
    --history_pretrained_path /home/lenovo/outputs/0914_pi05_sft_cube_catch_belt_dual_double_30000 \
    --OUTPUT_DIR /home/lenovo/outputs/0915_pi05_sft_cube_catch_belt_dual_v11_ddp2 \
    --policy_type pi05 \
    --num_gpus 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --steps 1000 \
    --save_freq 200 \
    --train_expert_only true \
    --tensorboard \
    -- \
    --policy.compile_model=false
```

四卡时改为 `CUDA_VISIBLE_DEVICES=0,1,2,3` 和 `--num_gpus 4`。显卡编号的数量必须不小于
`--num_gpus`；多卡时 `--device` 必须保持为 `cuda`，不能写成 `cuda:0`。脚本仍只由主 rank
保存 checkpoint 和 TensorBoard 事件；所有进程的标准输出和报错会汇总到
`OUTPUT_DIR/RL_train.log`，训练器主日志同时位于 `OUTPUT_DIR/train/RL_train.log`，TensorBoard
目录仍为 `OUTPUT_DIR/tensoborad`。

全局有效 batch size 为：

```text
batch_size（每卡） × num_gpus × gradient_accumulation_steps
```

因此，从单卡 `batch_size=1、gradient_accumulation_steps=16` 切换到两卡且希望维持相同的
有效 batch size，应改成 `gradient_accumulation_steps=8`；如果仍设为 16，则有效 batch size
会由 16 增加到 32。`steps` 在当前训练器中是每个 data-parallel worker 消耗的 micro-step 数，
不会因为 GPU 数量增加而自动缩短。

可在不启动训练的情况下检查数据、checkpoint 并查看最终多卡命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/RL_train.sh \
    --DATASET_ROOT /path/to/dataset \
    --history_pretrained_path /path/to/checkpoint \
    --OUTPUT_DIR /path/to/output \
    --policy_type pi05 \
    --num_gpus 2 \
    --dry_run
```

当前 PI0.5、PI0.5-RLT 和 SmolVLA 未声明经过验证的 FSDP 包装单元，所以这里开放的是权重复制的
DDP：它能提高吞吐，但不会降低每张卡保存完整模型所需的显存。底层虽有 FSDP2 配置入口，暂不建议
对这三种策略直接启用；应先为具体策略确定并验证 `accelerator.fsdp.wrap_modules` 后再开放。

## pi05+rlt训练

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
nohup bash scripts/RL_train.sh \
    --DATASET_ROOT /home/lenovo/datasets/20260914_bipiper_cube_catch_v1-1_merged \
    --ModelZoo /home/lenovo/modelZoo \
    --history_pretrained_path /home/lenovo/outputs/0914_pi05_sft_cube_catch_belt_dual_double_30000 \
    --OUTPUT_DIR /home/lenovo/outputs/0914_pi05_rlt_sft_cube_catch_belt_dual_double_30000_needDelete \
    --policy_type pi05_rlt \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --steps 1000 \
    --save_freq 200 \
    --train_expert_only false \
    > /home/lenovo/outputs/logs/0914_pi05_rlt_sft_cube_catch_belt_dual_double_30000_needDelete.log 2>&1 &

# 模型推理
## 单臂 VLA 仅推理：
不连接 can0，C 键禁用；需要人工介入时追加 --can0.control true

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
  --dataset.root /home/lenovo/datasets/0910_pi05_sft_cube_catch_belt_50000_needDelete \
  --dataset.single_task "Grab the moving blocks on the conveyor belt" \
  --wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6 \
  --policy.path /home/lenovo/outputs/0910_pi05_sft_cube_catch_belt_50000

## 单臂 VLA + 人工介入
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
    --dataset.root /home/lenovo/datasets/0910_pi05_sft_cube_catch_belt_50000_needDelete \
    --dataset.single_task "Grab the moving blocks on the conveyor belt" \
    --wrist_camera.index_or_path 260422275792 \
    --top_camera.index_or_path 6 \
    --policy.path /home/lenovo/outputs/0910_pi05_sft_cube_catch_belt_50000 \
    --can0.control true

## 单臂 VLA + 人工介入 + RTC

RTC 使用 guided 模式异步生成动作块。按 C 后会立刻暂停 RTC、废弃队列中以及正在生成的旧
VLA 动作，并在当前控制周期交给 can0 主臂；再次按 C 后重置推理状态并恢复 VLA。

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data.sh \
    --dataset.root /home/lenovo/datasets/0910_pi05_sft_cube_catch_belt_50000_needDelete \
    --dataset.single_task "Grab the moving blocks on the conveyor belt" \
    --wrist_camera.index_or_path 260422275792 \
    --top_camera.index_or_path 6 \
    --policy.path /home/lenovo/outputs/0910_pi05_sft_cube_catch_belt_50000 \
    --can0.control true \
    --rtc.enabled true \
    --rtc.execution_horizon 25 \
    --rtc_action_queue_threshold 32

`--rtc.execution_horizon` 是每次 RTC 融合保留的动作窗口长度，必须为正整数；
`--rtc_action_queue_threshold` 是触发后台补充动作块的队列阈值，必须为非负整数。
不传 `--rtc.enabled true` 时仍使用默认同步推理。

## 双臂 VLA 仅推理

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data_bimanual.sh \
  --dataset.root /home/lenovo/datasets/0915_pi05_sft_cube_catch_belt_dual_v11_double_30000_needDelete_needDelete \
  --dataset.single_task "Sort the moving blocks on the conveyor belt: use the left arm to place only yellow blocks into the left basket, and use the right arm to place only red blocks into the right basket." \
  --left_wrist_camera.index_or_path 260422275773 \
  --right_wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6 \
  --policy.path /home/lenovo/outputs/0915_pi05_sft_cube_catch_belt_dual_v11_double_30000_needDelete/train/checkpoints/001000/pretrained_model

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data_bimanual.sh \
  --dataset.root /home/lenovo/datasets/0914_pi05_sft_cube_catch_belt_dual_double_30000_needDelete \
  --dataset.single_task "Sort the moving blocks on the conveyor belt: use the left arm to place only yellow blocks into the left basket, and use the right arm to place only red blocks into the right basket." \
  --left_wrist_camera.index_or_path 260422275773 \
  --right_wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6 \
  --policy.path /home/lenovo/outputs/0914_pi05_sft_cube_catch_belt_dual_double_30000 \
  --rtc.enabled true \
  --rtc.execution_horizon 25 \
  --rtc_action_queue_threshold 32

## 双臂 VLA + 人工介入
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data_bimanual.sh \
  --dataset.root /home/lenovo/datasets/0914_pi05_sft_cube_catch_belt_dual_double_30000_needDelete \
  --dataset.single_task "Sort the moving blocks on the conveyor belt: use the left arm to place only yellow blocks into the left basket, and use the right arm to place only red blocks into the right basket." \
  --left_wrist_camera.index_or_path 260422275773 \
  --right_wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6 \
  --policy.path /home/lenovo/outputs/0914_pi05_sft_cube_catch_belt_dual_double_30000 \
  --can0.control true

### 双臂 VLA + 人工介入 + RTC

`--can0.control true` 会同时启用 can0 左主臂和 can2 右主臂。VLA 阶段两只主臂随从臂
同步移动；按 C 后两只主臂立即接管，同时清除 RTC 中尚未执行及正在生成的旧动作。

source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
bash /home/lenovo/code/Evo-RL-loop-0911/scripts/RL_data_bimanual.sh \
  --dataset.root /home/lenovo/datasets/0914_pi05_sft_cube_catch_belt_dual_double_30000_needDelete \
  --dataset.single_task "Sort the moving blocks on the conveyor belt: use the left arm to place only yellow blocks into the left basket, and use the right arm to place only red blocks into the right basket." \
  --left_wrist_camera.index_or_path 260422275773 \
  --right_wrist_camera.index_or_path 260422275792 \
  --top_camera.index_or_path 6 \
  --policy.path /home/lenovo/outputs/0914_pi05_sft_cube_catch_belt_dual_double_30000 \
  --can0.control true \
  --rtc.enabled true \
  --rtc.execution_horizon 25 \
  --rtc_action_queue_threshold 32

## 在线 RL：actor_new + 人工介入 + RTC

在线 Actor 只从 `actor_new` 启动；learner 和传输协议仍使用当前 LeRobot 实现。单臂或双臂由
JSON 中的 `env.robot.type` / `env.teleop.type` 决定。双臂主臂配置应为 `bi_piper_leader`，
其中左、右端口分别配置 can0、can2。

```bash
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
PIPER_ONLINE_RL_ACTOR_CONFIG=src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json \
bash scripts/RL_online.sh actor \
  --can0.control=true \
  --rtc.enabled=true \
  --rtc.mode=guided \
  --rtc.execution_horizon=25 \
  --rtc_action_queue_threshold=32
```

同时启动 learner 和 actor：

```bash
source "$EVORL_ENV_ROOT/bin/activate"
cd /home/lenovo/code/Evo-RL-loop-0911
PIPER_ONLINE_RL_ACTOR_CONFIG=src/lerobot/onlineRL_evoRL/configs/actor/Actor_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json \
PIPER_ONLINE_RL_LEARNER_CONFIG=src/lerobot/onlineRL_evoRL/configs/learner/Leanrer_onlineRL_transition_pi05_base_rlt_sft_cup_catch_v4_merged_train0901_40k.json \
bash scripts/RL_online.sh both \
  --can0.control=true \
  --rtc.enabled=true \
  --rtc.mode=guided
```

`both` 模式后面的命令行覆盖只传给 `actor_new`，避免 `can0`、RTC 等硬件参数被 learner
误解析。learner 参数应修改 learner JSON，或者用 `RL_online.sh learner --参数=值` 单独启动。
不设置上述两个变量时，脚本默认使用这套 0901-40k 的 0911 适配配置。

统一按键：`C` 人工介入/释放、`B` 成功、`F` 失败、`A` 放弃并重录、`R` 回初始位并重录、
`Esc` 退出；`V` 切换 VLA/Online Actor。`can0.control=false` 时完全不连接主臂并禁用 `C`，
但 VLA/Actor + RTC 仍可近推理。policy 阶段若启用主臂，单臂 can0 或双臂 can0+can2 会随
从臂目标移动；按 `C` 后立即清空 RTC 队列并由主臂动作接管。

日志默认写入配置的 `output_dir/logs/actor_<job_name>.log`；若环境中的
`processor.observation.display_cameras=true`，会继续启动 Rerun。
