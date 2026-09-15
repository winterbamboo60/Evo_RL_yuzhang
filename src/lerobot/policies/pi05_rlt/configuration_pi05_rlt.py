from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_rlt")
@dataclass
class PI05RLTConfig(PI05Config):
    """Current PI0.5 policy with an independently trainable RL-token autoencoder."""

    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_num_rl_tokens: int = 1
    rlt_prefix_seq_len: int = 1024
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_dropout_rate: float = 0.0
    rlt_image_only: bool = False
    rlt_use_mask: bool = True

    # Advantage-conditioned prompting from the 0901 offline-RL training path.
    # These are policy parameters; the standard TrainPipelineConfig remains unchanged.
    acp_enabled: bool = False
    acp_indicator_field: str = "complementary_info.acp_indicator"
    acp_indicator_dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rlt_alpha < 0:
            raise ValueError("rlt_alpha must be non-negative")
        if min(self.rlt_input_dim, self.rlt_embed_dim, self.rlt_num_rl_tokens) <= 0:
            raise ValueError("RLT dimensions and token count must be positive")
        if self.rlt_prefix_seq_len <= 0:
            raise ValueError("rlt_prefix_seq_len must be positive")
        if self.rlt_num_layers <= 0:
            raise ValueError("rlt_num_layers must be positive")
        if self.rlt_embed_dim % self.rlt_num_heads != 0:
            raise ValueError("rlt_embed_dim must be divisible by rlt_num_heads")
        if self.rlt_mlp_ratio <= 0:
            raise ValueError("rlt_mlp_ratio must be positive")
        if not 0.0 <= self.rlt_dropout_rate < 1.0:
            raise ValueError("rlt_dropout_rate must be in [0, 1)")
        if not self.acp_indicator_field:
            raise ValueError("acp_indicator_field must not be empty")
        if not 0.0 <= self.acp_indicator_dropout_prob <= 1.0:
            raise ValueError("acp_indicator_dropout_prob must be in [0, 1]")
