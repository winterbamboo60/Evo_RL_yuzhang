import safetensors.torch
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies import PI05RLTConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
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


class _TinyStagedLoadPolicy(PI05Policy):
    init_devices: list[str] = []

    def __init__(self, config, **kwargs):
        del kwargs
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Linear(2, 2, bias=False, device=config.device, dtype=torch.bfloat16)
        self.init_devices.append(self.model.weight.device.type)


class _TinyStagedLoadPolicyWithFreshParameter(_TinyStagedLoadPolicy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.fresh = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16, device=config.device))


def _staged_load_config():
    return type(
        "StagedLoadConfig",
        (),
        {"device": "cpu", "use_proprioceptive_memory": False},
    )()


def test_pi05_staged_load_preserves_destination_dtype(tmp_path, monkeypatch):
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    safetensors.torch.save_file({"model.weight": expected}, tmp_path / "model.safetensors")
    loaded_devices = []
    real_load_file = safetensors.torch.load_file

    def recording_load_file(*args, **kwargs):
        loaded_devices.append(kwargs.get("device"))
        return real_load_file(*args, **kwargs)

    _TinyStagedLoadPolicy.init_devices = []
    monkeypatch.setattr(safetensors.torch, "load_file", recording_load_file)

    policy = _TinyStagedLoadPolicy.from_pretrained(tmp_path, config=_staged_load_config())

    assert _TinyStagedLoadPolicy.init_devices == ["cpu"]
    assert loaded_devices == [None]
    assert not any(tensor.is_meta for tensor in policy.state_dict().values())
    assert policy.model.weight.dtype == torch.bfloat16
    torch.testing.assert_close(policy.model.weight.float(), expected)


def test_pi05_staged_load_keeps_fresh_missing_parameters(tmp_path):
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    safetensors.torch.save_file({"model.weight": expected}, tmp_path / "model.safetensors")

    _TinyStagedLoadPolicyWithFreshParameter.init_devices = []

    policy = _TinyStagedLoadPolicyWithFreshParameter.from_pretrained(
        tmp_path,
        config=_staged_load_config(),
        strict=False,
    )

    assert _TinyStagedLoadPolicyWithFreshParameter.init_devices == ["cpu"]
    assert policy.fresh.device.type == "cpu"
    assert not policy.fresh.is_meta
    assert policy.fresh.dtype == torch.bfloat16
    assert policy.model.weight.dtype == torch.bfloat16
    torch.testing.assert_close(policy.model.weight.float(), expected)
