from types import SimpleNamespace

import torch

from lerobot.onlineRL_evoRL.buffer import ReplayBuffer
from lerobot.policies.pi05_onlineRL.chunk_transition import build_sliding_window_transitions
from lerobot.policies.pi05_onlineRL.modeling_pi05_online_rl import PI05OnlineRLPolicy
from lerobot.utils.constants import ACTION


def _episode(length: int):
    transitions = []
    for index in range(length):
        state = {
            "z_rl": torch.tensor([[float(index)]]),
            "proprio": torch.tensor([[float(index)]]),
            "ref_action": torch.full((1, 50, 1), float(index)),
        }
        next_state = {
            "z_rl": torch.tensor([[float(index + 1)]]),
            "proprio": torch.tensor([[float(index + 1)]]),
            "ref_action": torch.full((1, 50, 1), float(index + 1)),
        }
        transitions.append(
            {
                "state": state,
                ACTION: torch.tensor([[float(index)]]),
                "reward": float(index),
                "next_state": next_state,
                "done": index == length - 1,
                "truncated": False,
                "complementary_info": {"is_intervention": index % 2 == 0},
            }
        )
    return transitions


def test_image_preprocessing_matches_nchw_float_and_nhwc_uint8():
    key = "observation.images.top"
    image = torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8)
    policy = SimpleNamespace(
        config=SimpleNamespace(image_features={key: object()}, image_resolution=(4, 4)),
        parameters=lambda: iter((torch.empty(0),)),
    )

    nchw, _ = PI05OnlineRLPolicy._preprocess_images(policy, {key: image.float() / 255.0})
    nhwc, _ = PI05OnlineRLPolicy._preprocess_images(policy, {key: image.permute(0, 2, 3, 1)})

    assert nchw[0].shape == nhwc[0].shape == (2, 3, 4, 4)
    assert torch.allclose(nchw[0], nhwc[0])


def test_length_70_sliding_windows_and_bootstrap_mask():
    windows = build_sliding_window_transitions(_episode(70), horizon=50)

    assert len(windows) == 70
    assert windows[0]["valid_action_mask"].sum() == 50
    assert windows[0]["next_valid_action_mask"].sum() == 20
    assert windows[0]["next_state"]["z_rl"].item() == 50
    assert not windows[0]["done"]
    assert torch.equal(windows[0]["reward"], torch.arange(50, dtype=torch.float32))

    assert windows[20]["valid_action_mask"].sum() == 50
    assert windows[20]["done"]
    assert windows[50]["valid_action_mask"].sum() == 20
    assert windows[50]["next_valid_action_mask"].sum() == 0
    assert windows[50]["done"]
    assert windows[50]["target_action_chunk"][20:].eq(0).all()


def test_chunk_replay_keeps_explicit_next_state_and_masks():
    transition = build_sliding_window_transitions(_episode(70), horizon=50)[0]
    replay = ReplayBuffer(
        capacity=2,
        device="cpu",
        storage_device="cpu",
        state_keys=("z_rl", "proprio", "ref_action"),
        use_drq=False,
        optimize_memory=False,
    )
    replay.add(**transition)
    batch = replay.sample(1)

    assert batch["target_action_chunk"].shape == (1, 50, 1)
    assert batch["reward"].shape == (1, 50)
    assert batch["valid_action_mask"].sum() == 50
    assert batch["next_valid_action_mask"].sum() == 20
    assert batch["next_state"]["z_rl"].item() == 50


def test_actor_and_critic_shapes_and_padding_invariance():
    from lerobot.policies.pi05_onlineRL.modeling_pi05_online_rl import (
        RLTChunkActor,
        RLTChunkCriticEnsemble,
    )

    actor = RLTChunkActor(4, 2, 5, 3, (8,), 0.002, 0.5)
    critic = RLTChunkCriticEnsemble(4, 2, 5, 3, (8,), 2)
    z_rl, proprio = torch.randn(2, 4), torch.randn(2, 2)
    reference = torch.randn(2, 5, 3)
    action = actor(z_rl, proprio, reference, deterministic=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    changed_padding = action.clone()
    changed_padding[0, 3:] = 999

    assert action.shape == (2, 5, 3)
    assert critic(z_rl, proprio, action, mask).shape == (2, 2)
    assert torch.allclose(
        critic(z_rl, proprio, action, mask)[:, 0],
        critic(z_rl, proprio, changed_padding, mask)[:, 0],
    )


def test_td_target_bootstraps_only_full_nonterminal_chunk():
    from lerobot.policies.pi05_onlineRL.modeling_pi05_online_rl import compute_chunk_td_target

    discount = 0.96
    reward = torch.stack((torch.arange(50), torch.arange(50, 100))).float()
    mask = torch.ones(2, 50)
    mask[1, 20:] = 0
    target = compute_chunk_td_target(
        reward,
        mask,
        next_q=torch.tensor([3.0, 999.0]),
        done=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        discount=discount,
    )
    powers = discount ** torch.arange(50)
    assert torch.allclose(target[0], (reward[0] * powers).sum() + discount**50 * 3)
    assert torch.allclose(target[1], (reward[1, :20] * powers[:20]).sum())
