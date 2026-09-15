import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies import PI05RLTConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.policies.pi05_rlt.processor_pi05_rlt import (
    PI05RLTACPTaskProcessorStep,
    reconcile_pi05_rlt_processors,
)
from lerobot.policies.pi05_rlt.rlt_token_transformer import RLTTokenTransformer


def test_pi05_rlt_is_independently_registered():
    assert PI05RLTConfig.get_choice_name(PI05RLTConfig) == "pi05_rlt"
    assert get_policy_class("pi05_rlt") is PI05RLTPolicy


def test_rlt_transformer_supports_dynamic_prefix_and_per_sample_loss():
    model = RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        num_rl_tokens=2,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
    )
    prefix = torch.randn(2, 7, 8)
    mask = torch.tensor([[True] * 7, [True, True, True, True, False, False, False]])

    loss, info = model(prefix, mask, reduction="none")

    assert loss.shape == (2,)
    assert info["z_rl"].shape == (2, 16)
    assert torch.isfinite(loss).all()


def test_pi05_rlt_acp_processor_preserves_0901_prompt_semantics():
    step = PI05RLTACPTaskProcessorStep(enabled=True)
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick", "place"],
            "complementary_info.acp_indicator": torch.tensor([1, 0], dtype=torch.int64),
        }
    }

    result = step(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == [
        "pick\nAdvantage: positive",
        "place\nAdvantage: negative",
    ]


def test_pi05_rlt_reconciles_plain_pi05_preprocessor():
    config = PI05RLTConfig(
        device="cpu",
        acp_enabled=True,
        acp_indicator_field="complementary_info.indicator_field",
        acp_indicator_dropout_prob=0.25,
    )
    tokenizer_step = Pi05PrepareStateTokenizerProcessorStep()
    preprocessor = type("Preprocessor", (), {"steps": [object(), tokenizer_step, object()]})()
    postprocessor = object()

    reconciled, returned_postprocessor = reconcile_pi05_rlt_processors(config, preprocessor, postprocessor)

    acp_steps = [step for step in reconciled.steps if isinstance(step, PI05RLTACPTaskProcessorStep)]
    assert len(acp_steps) == 1
    assert reconciled.steps.index(acp_steps[0]) < reconciled.steps.index(tokenizer_step)
    assert acp_steps[0].enabled is True
    assert acp_steps[0].indicator_field == "complementary_info.indicator_field"
    assert acp_steps[0].dropout_prob == 0.25
    assert returned_postprocessor is postprocessor
