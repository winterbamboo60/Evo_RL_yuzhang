export EVORL_PROJECT_ROOT="${EVORL_PROJECT_ROOT:-/mnt/sda/yuzhang/projects/Evo_RL_yuzhang}"
export EVORL_ENV_ROOT="${EVORL_ENV_ROOT:-/mnt/sda/yuzhang/envs/evo_0911}"

source "$EVORL_ENV_ROOT/bin/activate"

cd "$EVORL_PROJECT_ROOT"

rm -r /mnt/sda/yuzhang/projects/Evo_RL_yuzhang/outputs/captured_images
lerobot-find-cameras opencv


lerobot-setup-can --mode=setup --interfaces=can0,can1,can2,can3


leader no data: 0 2
follow has data:1 3

lerobot-setup-can --mode=test --interfaces=can0
lerobot-setup-can --mode=test --interfaces=can1

bash scripts/RL_data.sh \
  --dataset.root /mnt/sda/datasets/1008_cube_catch/1008_cube_catch_12 \
  --dataset.single_task "Grab the block and put iBAt in the basket" \
  --wrist_camera.index_or_path 16 \
  --top_camera.index_or_path 24


dual piper data collect:
bash scripts/RL_data_bimanual.sh \
  --dataset.root /mnt/sda/datasets/1014_pourWater_general/1014_pourWater_general_1 \
  --dataset.single_task "catch the drink and pour water to cup" \
  --left_wrist_camera.index_or_path 22 \
  --right_wrist_camera.index_or_path 6 \
  --top_camera.index_or_path 14

# 数据集合并

python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id /mnt/sda/datasets/1012_pourWater_40ml_60ml_80ml_merged \
    --operation.type merge \
    --operation.source_dir /mnt/sda/datasets/1012_pourWater_40ml_60ml_80ml

# 查看数据集
cd "$EVORL_PROJECT_ROOT"
python scripts/ci/dataset_checker.py /mnt/sda/datasets/1014_pourWater_40ml/1014_pourWater_40ml_3

# 双臂 VLA 仅推理
bash scripts/RL_data_bimanual.sh \
  --dataset.root /mnt/sda/datasets/1012_pi05_rlt_sft_1012_pourWater_40ml_60ml_80ml_merged_8gpu_50k_testNeedDelete \
  --dataset.single_task "catch the drink and pour 40ml water to cup" \
  --left_wrist_camera.index_or_path 22 \
  --right_wrist_camera.index_or_path 6 \
  --top_camera.index_or_path 14 \
  --policy.path /mnt/sda/yuzhang/outputs/1012_pi05_rlt_sft_1012_pourWater_40ml_60ml_80ml_merged_8gpu_50k \
  --rename_map '{"observation.images.left_top":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}' \
  --rtc.enabled true \
  --rtc.execution_horizon 25 \
  --rtc_action_queue_threshold 32
