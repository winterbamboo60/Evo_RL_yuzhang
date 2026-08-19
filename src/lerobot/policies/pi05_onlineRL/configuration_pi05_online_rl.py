#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224

@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2.0


@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"

# 数据	        原始形状	                    预处理后	                      模型内部
# 单路图像      [B,3,H,W]                   [B,3,224,224]                    [B,256,2048]
# 多路图像      Ncam路                      每路独立处理                     [B,256Ncam,2048]
# 状态          [B,Ds]                     分位数归一化、离散化为整数文本     合并进语言 token
# 语言          每个样本一个任务字符串       [B,200] token IDs                 [B,200,2048]
# 训练动作      [B,50,Da]                   补齐为 [B,50,32]                  [B,50,1024]
# 动作输出	        —	                    模型输出 [B,50,32]	            截取为 [B,50,Da]
# 图像和语言前缀的总长度为：L_prefix​=256*N_cam​+200
# 后面还会连接长度为 50 的动作后缀。注意，前缀隐藏维度是 2048，动作后缀隐藏维度是 1024，因此并不是在最后一维上直接拼接，而是作为两条不同宽度的 Transformer 输入流进行联合注意力计算。


# 1、图像预处理
# 模型预期 LeRobot 输入图像的数值范围是：[0, 1]
# 原始图像
# [B, 3, H, W]
#         ↓
# 识别 channels-first / channels-last
#         ↓
# 保持长宽比缩放并填充
#         ↓
# [B, 3, 224, 224]
#         ↓
# x ← 2x - 1
#         ↓
# 数值范围变为 [-1, 1]
# 因此数据集中的图像应当保持在 [0,1]。假如上游已经处理成 [−1,1]，再次执行 2I−1 后范围会变成 [−3,1]，从而导致输入错误。

# 2、图像处理为视觉token
# PaliGemma 的视觉编码器采用：
#   图像分辨率：224×224
#   Patch 大小：14×14
#   视觉编码器内部宽度：1152
#   多模态投影维度：2048
#
# 变化过程可以理解为 patch embedding + transformer encoder + linear projection：[B,3,224,224]→[B,256,1152]→[B,256,2048]. （最后的 2048 与 gemma_2b 语言主干的隐藏维度一致）
# 多摄像头情况下，每幅图像分别经过视觉编码器，然后在序列维度上拼接：[B,256,2048]×N_cam​→[B,256*N_cam​,2048].
# 缺失的预期摄像头会被替换成数值全为 −1 的空图像，并使用 image mask 标记为无效，不参与有效注意力计算.

# 3、状态数据的维度和处理过程
# 配置中定义：max_state_dim: int = 32，允许机器人维度最多为32维
# 处理过程：分数归一化（将状态映射到[-1, 1]）+ 状态离散化（划分为0~255的整数）+ 转换为文本 token（每个整数对应一个 token ID）+ 补齐为长度为 200 的 token 序列
# 针对这部分数据，PI05不存在独立的 state_proj 层，而是直接作为显示数据加入任务提示词中：如 Task: Flip the package if the barcode is not facing up., State: 161 217 105 12 175 237 88 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128 128;

# 4、语言数据的维度和处理过程
# 4.1、tokenizer
# tokenizer_name = "google/paligemma-3b-pt-224"
# tokenizer_max_length = 200
# 使用 PaliGemma/Gemma 的 tokenizer，文本长度不足 200 时采用右侧 padding；超过 200 时会被截断，因为状态也被放入文本，所以 200 的长度需要同时容纳：
# Task 固定前缀 + 任务描述 + State 固定前缀 + 最多32维离散状态文本 + Action: + 特殊 token
#
# 4.2、语言 embedding
# 语言主干采用 gemma_2b
#   隐藏维度：2048
#   Transformer 层数：18
#   注意力头数：8
#   单头维度：256
#   MLP 中间维度：16384
# 语言 token 的形状变化为：[B,200]→[B,200,2048].
# 图像 token 和语言 token 随后在序列维度上拼接：[B,256*N_cam​,2048] + [B,200,2048] → [B,256*N_cam​+200,2048]

# 5、动作数据的维度和处理过程
# chunk_size=50：模型每次预测未来 50 步动作。
# n_action_steps=50：环境实际执行这 50 步动作。
# 当前设置相当于预测后完整执行整个动作块。如果控制频率为 20 Hz，那么 50 步对应：50/20=2.5 s.
# 动作维度补齐（以机械臂7维为例）：[B,50,7]→[B,50,32]
# 内部经过两层LN处理：
#   action_in_proj = Linear(32, 1024)
#   action_out_proj = Linear(1024, 32)
# 即：[B,50,32]→[B,50,1024]→[B,50,32].

# 6、Flow Matching 训练过程
# 设真实归一化动作块为：A∈R^B×50×32,
# 采样同形状的高斯噪声：ϵ∼N(0,I).
# 时间变量 t 根据 Beta 分布采样，构造带噪动作：X_t​=t*ϵ+(1−t)A
#   t→0 时，Xt 更接近真实动作；
#   t→1 时，Xt 更接近纯噪声。
# 训练过程中，动作的变化情况：
#   真实动作 A       [B, 50, 32]
#   高斯噪声 ε       [B, 50, 32]
#   带噪动作 Xt      [B, 50, 32]
#             ↓ Linear
#   动作 embedding   [B, 50, 1024]
#             ↓ Action Expert
#   速度预测         [B, 50, 1024]
#             ↓ Linear
#   预测速度         [B, 50, 32]
#
# 时间 t 会先编码为 1024 维正弦位置向量，再经过 MLP：# t→[B,1024]→[B,1024], 并通过条件归一化机制注入动作专家。


# 7、推理过程
# 推理时没有真实动作，首先随机采样：
# X1​∼N(0,I),X1​∈R^B×50×32.
# 随后，整体推理过程为：
#   图像 + 语言状态提示词
#          ↓
#   PaliGemma前缀编码
#          ↓
#   缓存 Prefix KV Cache
#          ↓
#   随机动作噪声 [B,50,32]
#          ↓
#   10次 Flow Matching / Euler 更新
#          ↓
#   归一化动作 [B,50,32]
#          ↓
#   截取真实动作维度
#          ↓
#   [B,50,Da]
#          ↓
#   分位数反归一化
#          ↓
#   机器人实际动作


# 8、完整数据流图
#                          ┌──────────────────────────────┐
# 图像1 [B,3,H,W] ────────→│ resize/pad 224×224           │
# 图像2 [B,3,H,W] ────────→│ [0,1] → [-1,1]              │
#                          └──────────────┬───────────────┘
#                                         ↓
#                          每路 [B,256,2048]
#                                         ↓
#                          多路拼接 [B,256×Nc,2048]
#                                         │
#                                         │
#                                         │
# 状态 [B,Ds]                             │
#    ↓ Quantile normalize                 │
# [-1,1]                                  │
#    ↓ 256-bin quantization               │
# 整数文本                                │
#    ↓                                    │
# Task + State prompt                     │
#    ↓ PaliGemma tokenizer                │
# Tokens [B,200]                          │
#    ↓ embedding                          │
# [B,200,2048]                            ↓
#    └──────────── concat ────────────────┘
#                   ↓
#               Prefix: [B,256×Nc+200,2048]───────────────┐
#                                                         │
#                                                         │
# 真实动作 [B,50,Da]                                       │
#    ↓ Quantile normalize                                 │
#    ↓ pad to 32                                          │
# [B,50,32]                                               │
#    ↓ 加噪 Xt                                            │
#    ↓ Linear(32→1024)                                    │
# Action suffix [B,50,1024]                               │
#    │                                                    │
#    └──────────── PaliGemma + Action Expert ─────────────┘
#                          联合注意力
#                               ↓
#                     Velocity [B,50,32]
#                               ↓
#                     Flow Matching积分
#                               ↓
#                     Action [B,50,Da]
#                               ↓
#                     Quantile反归一化
#                               ↓
#                       执行50步动作


@PreTrainedConfig.register_subclass("pi05_online_rl")
@dataclass
class PI05OnlineRLConfig(PreTrainedConfig):
    # A plain PI0.5+RLT checkpoint can initialize the frozen backbone.
    base_policy_path: str | None = None
    tokenizer_name: str = "google/paligemma-3b-pt-224"  # see openpi `__post_init__`
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    # 只使用当前时刻的一帧观测
    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    # empty_cameras=0 表示默认不主动增加空摄像头，如果设置为 2，validate_features() 会添加：observation.images.empty_camera_0；observation.images.empty_camera_1
    # 用于让不同数据集具有一致的摄像头数量，或者用于无图像数据。
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = True  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # RLT Stage1 settings. Enabled only when ``use_rlt`` is set on the base policy config.
    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_num_rl_tokens: int = 1
    rlt_prefix_seq_len: int = 1024
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_image_only: bool = False
    rlt_use_mask: bool = True

    # Chunk actor/critic.
    z_dim: int = 2048
    proprio_dim: int = 7
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    num_critics: int = 2
    fixed_std: float = 0.002
    reference_dropout_prob: float = 0.5
    q_weight: float = 0.1
    bc_weight: float = 5.0
    discount: float = 0.96
    critic_target_update_weight: float = 0.005
    use_backup_entropy: bool = False

    # Knobs consumed by the existing online learner runtime.
    storage_device: str = "cpu"
    shared_encoder: bool = False
    num_discrete_actions: int | None = None
    online_steps: int = 1_000_000
    online_buffer_capacity: int = 100_000
    offline_buffer_capacity: int = 100_000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    utd_ratio: int = 4
    grad_clip_norm: float = 10.0
    use_torch_compile: bool = False
    feature_extract_batch_size: int = 8
    offline_intervention_field: str | None = "complementary_info.is_intervention"
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.n_action_steps != self.chunk_size:
            raise ValueError("pi05_online_rl requires n_action_steps == chunk_size")
        if self.z_dim <= 0 or self.proprio_dim <= 0:
            raise ValueError("z_dim and proprio_dim must be positive")
        if self.num_critics < 2:
            raise ValueError("num_critics must be at least 2")
        if self.fixed_std <= 0:
            raise ValueError("fixed_std must be positive")
        if not 0 <= self.reference_dropout_prob <= 1:
            raise ValueError("reference_dropout_prob must be in [0, 1]")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if not 0 < self.critic_target_update_weight <= 1:
            raise ValueError("critic_target_update_weight must be in (0, 1]")
        if self.use_backup_entropy:
            raise ValueError("pi05_online_rl does not support entropy backup")

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
