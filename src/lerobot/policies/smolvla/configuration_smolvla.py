# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES

# 模型架构
#   图像：使用 SmolVLM2 的 SigLIP 视觉编码器，每张 512×512 图像最终得到 64 个视觉 token。
#   语言：最多 48 个 token，但默认按当前 batch 中的最长文本动态 padding。
#   状态：保留为连续向量，补齐到 32 维，再通过 Linear(32→960) 映射成一个独立状态 token。
#   动作：每次预测 50 步，每步补齐到 32 维，通过宽度为 720 的动作专家生成。
#   生成方式：采用 Flow Matching，推理时从高斯噪声开始执行 10 次 Euler 更新。

# 各模态数据处理过程
# 数据	        原始形状	                    预处理后	                      模型内部
# 单路图像      [B,3,H,W]                   [B,3,512,512]                    [B,64,960]
# 多路图像      Ncam路                      每路独立处理                     [B,64Ncam,960]
# 状态          [B,Ds]                     均值标准差归一化，补齐到 [B,32]     [B,1,960]
# 语言          每个样本一个任务字符串       [B,L_lang​]（按最长动态padding）    [B,L_lang​,960]
# 训练动作      [B,50,Da]                   归一化、补齐为 [B,50,32]           [B,50,720]
# 动作输出	        —	                    模型输出 [B,50,32]	            截取为 [B,50,Da]
# 图像和语言前缀的总长度为：L_prefix​=64*N_cam​+L_lang+1，最后的 1 是状态 token
# 动作专家的后缀长度固定为：L_suffix=50

# 整体模型结构
# vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
#    文本隐藏维度：960；
#    原始文本层数：32；
#    文本注意力头数：15；
#    KV 头数：5；
#    单头维度：64；
#    文本 MLP 中间维度：2560；
#    视觉隐藏维度：768；
#    图像尺寸：512；
#    Patch 大小：16；
#    视觉注意力头数：12；
#    Pixel Shuffle 比例：4
# 需要注意的是，模型并不使用 SmolVLM2 的全部 32 层，而是只保留前 16 层，动作专家默认也使用 16 层
# 
# 模型可概括为：
# 图像 ─→ SigLIP视觉编码器 ─┐
# 语言 ─→ SmolVLM词嵌入 ────┼─→ 960维VLM前缀 ─→ KV Cache
# 状态 ─→ Linear(32,960) ──┘                         │
#                                                    ↓
# 噪声动作 ─→ Linear(32,720) ─→ 16层动作专家 ─→ 预测速度


# 1、图像数据的维度和处理过程
# n_obs_steps = 1，表示每次只使用当前时刻的一组图像，不使用历史图像序列。
# 如果输入偶尔带有时间维度：[B,T,3,H,W],    代码只取最后一帧：img = batch[key][:, -1, :, :, :]
# 所以最终输入视觉编码器的仍是：所以最终输入视觉编码器的仍是：[B,3,H,W].
# 
# 图像 Resize 和 Padding
# 图像会保持长宽比缩放，然后补齐到：（512，512），且将 padding 加在左侧和上侧
# 
# 完整过程为：
# 原始图像
#   [B,3,H,W]
#       ↓ 保持比例缩放
#   [B,3,H',W']
#       ↓ 左上侧padding（填0）
#   [B,3,512,512]
#       ↓ x ← 2x - 1（归一化）
#   范围由 [0,1] 变为 [-1,1]


# 2、图像处理为视觉token
# 将经过：patch embedding + pixel shuffle + linear projection 得到图像token，得到 64 个视觉 token，每个 token 维度为 960。
# 即：[B,3,512,512] → [B,1024,768] → [B,64,12288] → [B,64,960]
# 对于N_cam路摄像头，每个摄像头独立产生[B,64,960]，随后序列维度拼接：[B,64,960]×N_cam → [B,64N_cam,960]


# 3、语言数据的维度和处理过程
# SmolVLA 的语言输入只包含自然语言任务描述，处理器会先确保任务文本末尾包含换行符：Pick up the red cube and place it in the box.\n
# 语言长度补齐到当前 batch 内最长文本的长度，同时限制最大长度为48：[B,L_lang​], L_lang​≤48


# 4、语言embedding
# 文本影藏维度为960，故：[B,Llang​]→[B,Llang​,960]
# 代码还会对 语言&图像 embedding 乘以：sqrt(960)


# 5、状态数据的维度和处理过程
# 处理过程：状态补齐到32维：[B,Ds]→[B,32]，然后通过Linear(32→960)映射成一个独立状态token：[B,1,960]：
#   机器人状态 [B,Ds]
#         ↓ Mean-Std归一化
#   归一化状态 [B,Ds]
#         ↓ 末尾补零
#   状态向量 [B,32]
#         ↓ Linear(32→960)
#   状态特征 [B,960]
#         ↓ 增加token维
#   状态token [B,1,960]


# 6. 前缀序列
# VLM前缀顺序为
#   [图像1 tokens]
#   [图像2 tokens]
#   ...
#   [语言 tokens]
#   [状态 token]
# 即：E_prefix​=Concat(E_image1​​,…,E_imageNcam​​​,E_language​,E_state​)
# 维度为：[B,64_Ncam​+L_lang​+1,960].


# 7、动作数据的维度和处理过程
# 原始动作 [B,50,Da​]
#       ↓ 动作补齐（训练时）
# 带噪动作 [B,50,32]
#       ↓ Linear(32→720)
# 动作特征 [B,50,720]
# 
# 时间 t [B]
#       ↓ Sin-Cos embedding
# 时间特征 [B,720]
#       ↓ 扩展
# [B,50,720]
# 
# 动作特征与时间特征拼接
#       ↓
# [B,50,1440]
#       ↓ Linear(1440→720)
#       ↓ SiLU
#       ↓ Linear(720→720)
# 动作专家输入 [B,50,720]


# 8、注意力机制：VLM 如何条件化动作专家
# 模型配置：
#   attention_mode = "cross_attn"
#   self_attn_every_n_layers = 2
#   num_vlm_layers = 16
# 表示动作专家主要通过 Cross-Attention 读取 VLM 前缀，同时每隔两层插入一层 Self-Attention，可以理解为：
#   第0层：动作Self-Attention
#   第1层：动作对VLM前缀Cross-Attention
#   第2层：动作Self-Attention
#   第3层：动作对VLM前缀Cross-Attention
#   ....
# Cross-Attention 时：
#   Query 来自动作 token，宽度 720；
#   Key/Value 来自图像、语言和状态前缀，原始宽度 960；
#   Key/Value 再投影到动作专家的注意力空间。


# 9、Flow Matching 训练过程
#   真实动作 A       [B,50,32]
#   高斯噪声 ε       [B,50,32]
#   时间 t           [B]
#           ↓
#   带噪动作 Xt      [B,50,32]
#           ↓
#   动作+时间嵌入     [B,50,720]
#           ↓
#   16层动作专家
#           ↓
#   动作隐藏特征      [B,50,720]
#           ↓ Linear(720→32)
#   预测速度 Vt       [B,50,32]
#           ↓
#   与 ε-A 计算MSE


# 10、推理过程
#   图像 + 语言 + 状态
#         ↓
#   构造VLM前缀
#         ↓
#   运行16层VLM
#         ↓
#   生成并缓存Prefix KV Cache
#         ↓
#   高斯动作噪声 [B,50,32]
#         ↓
#   10次动作专家去噪/Euler积分
#         ↓
#   归一化动作块 [B,50,32]
#         ↓
#   截取实际动作维度
#         ↓
#   [B,50,Da]
#         ↓
#   Mean-Std反归一化
#         ↓
#   实际机器人动作

# 11、完整数据流图
# 多路图像
# 每路 [B,3,H,W]
#         ↓ 保持比例resize + 左上padding
# 每路 [B,3,512,512]
#         ↓ [0,1] → [-1,1]
#         ↓ SigLIP，patch=16
# 每路 [B,1024,768]
#         ↓ Pixel Shuffle，factor=4
# 每路 [B,64,12288]
#         ↓ Linear(12288→960)
# 每路 [B,64,960]
#         ↓ 多路拼接
# 图像前缀 [B,64×Nc,960]       ┐
#                             │
#                             │
#                             │
# 语言任务字符串               │
#         ↓ 添加换行符         │
#         ↓ SmolVLM tokenizer │
# [B,Llang]，Llang≤48         │
#         ↓ Embedding        │
# [B,Llang,960]              ├─→ VLM前缀
#                            │   [B,64Nc+Llang+1,960]
#                            |
# 状态 [B,Ds]                 │
#         ↓ Mean-Std归一化    │
#         ↓ pad to 32        │
# [B,32]                     │
#         ↓ Linear(32→960)   │
# [B,1,960]                  ┘
#                                 │          
#                                 ↓
#                          16层VLM编码
#                                 ↓
#                           Prefix KV Cache
#                                 │
#                                 │ Cross Attention
#                                 ↓
# 真实动作 [B,50,Da]               │
#         ↓ Mean-Std归一化         │
#         ↓ pad to 32             │
# [B,50,32]                       │
#         ↓ 加入高斯噪声 Xt        │
#         ↓ Linear(32→720)        │
# 动作特征 [B,50,720]              │
#                                 │
# 时间 t [B]                      │
#         ↓ Sin-Cos               │
# [B,720]                         │
#         ↓ 扩展                  │
# [B,50,720]                      │
#         ↓ 与动作特征拼接         │
# [B,50,1440]                     │
#         ↓ MLP                   │
# 动作专家输入 [B,50,720] ─────────┘
#         ↓ 16层动作专家，CrossAttention
# [B,50,720]
#         ↓ Linear(720→32)
# 速度预测 [B,50,32]
#         ↓ 10次Euler积分
# 归一化动作 [B,50,32]
#         ↓ 截取真实维度
# [B,50,Da]
#         ↓ Mean-Std反归一化
# 机器人实际动作


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

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
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
