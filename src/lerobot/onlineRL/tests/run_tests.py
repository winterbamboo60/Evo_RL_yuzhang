"""Standard-library test runner for environments without pytest installed."""

from .test_online_rl import (
    test_actor_mode_persists_without_learner,
    test_actor_runtime_infers_once_per_action_chunk,
    test_actor_runtime_refills_vla_actions_below_low_watermark,
    test_actor_runtime_caches_generated_vla_actions,
    test_actor_runtime_smooths_automatic_actions,
    test_episode_replay_is_episode_counted,
    test_episode_countdown_sleeps_once_per_second,
    test_fake_hardware_only_returns_follower_action,
    test_keyboard_listener_is_optional_without_display,
    test_keyboard_listener_prefers_current_tty,
    test_keyboard_state_machine,
    test_random_parity_sample_keeps_full_future_action_chunk,
    test_learner_waits_for_two_complete_episodes,
    test_real_hardware_requires_existing_calibration_before_connect,
    test_real_hardware_reuses_sync_pool_for_policy_actions,
    test_real_hardware_connect_reuses_existing_camera_session,
    test_real_run_requires_two_camera_sources,
    test_safe_action_preserves_piper_degree_scale,
    test_stage1_image_batch_outputs_channels_first,
    test_stage1_uses_piper_training_image_slots,
)


def main() -> None:
    for test in (
        test_keyboard_state_machine,
        test_random_parity_sample_keeps_full_future_action_chunk,
        test_keyboard_listener_is_optional_without_display,
        test_keyboard_listener_prefers_current_tty,
        test_episode_replay_is_episode_counted,
        test_episode_countdown_sleeps_once_per_second,
        test_learner_waits_for_two_complete_episodes,
        test_fake_hardware_only_returns_follower_action,
        test_real_hardware_requires_existing_calibration_before_connect,
        test_real_hardware_reuses_sync_pool_for_policy_actions,
        test_real_hardware_connect_reuses_existing_camera_session,
        test_safe_action_preserves_piper_degree_scale,
        test_stage1_image_batch_outputs_channels_first,
        test_stage1_uses_piper_training_image_slots,
        test_real_run_requires_two_camera_sources,
        test_actor_mode_persists_without_learner,
        test_actor_runtime_infers_once_per_action_chunk,
        test_actor_runtime_refills_vla_actions_below_low_watermark,
        test_actor_runtime_caches_generated_vla_actions,
        test_actor_runtime_smooths_automatic_actions,
    ):
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
