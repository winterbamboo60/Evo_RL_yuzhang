# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import MethodType, SimpleNamespace

import pytest

from lerobot.robots.piper_follower.piper_follower import PiperFollower
from lerobot.scripts import lerobot_record as record


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("bimanual", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
def test_home_joints_and_open_grippers_in_two_seconds(monkeypatch, bimanual, reverse, passthrough, reset):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(record.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(record.time, "sleep", lambda dt: setattr(clock, "now", clock.now + dt))
    keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]

    def make_arm():
        arm = SimpleNamespace(
            action_features=dict.fromkeys(keys, float),
            config=SimpleNamespace(calibration_scale=1000),
            calibration={
                key: SimpleNamespace(
                    homing_offset=20000 if key == "gripper.pos" else 1000,
                    range_min=0 if key == "gripper.pos" else -180000,
                    range_max=100000 if key == "gripper.pos" else 180000,
                    drive_mode=int(reverse),
                )
                for key in keys
            },
            _use_uncalibrated_passthrough=lambda: passthrough,
        )
        arm._from_calibration_units = MethodType(PiperFollower._from_calibration_units, arm)
        arm._offset_to_target = MethodType(PiperFollower._offset_to_target, arm)
        return arm

    if bimanual:
        robot = SimpleNamespace(name="bi_piper_follower", left_arm=make_arm(), right_arm=make_arm())
        arms = [("left_", robot.left_arm), ("right_", robot.right_arm)]
    else:
        robot = make_arm()
        robot.name = "piper_follower"
        arms = [("", robot)]
    positions = {prefix + key: 40.0 if key == "gripper.pos" else 10.0 for prefix, _ in arms for key in keys}
    initial = positions.copy()
    robot.get_observation = lambda: {**positions, "camera": object()}
    sent = []

    def send_action(action):
        assert action.keys() == positions.keys()
        for prefix, arm in arms:
            for key in keys:
                value = action[prefix + key]
                positions[prefix + key] = value if passthrough else arm._offset_to_target(key, value)
        sent.append((clock.now, positions.copy()))

    robot.send_action = send_action
    if reset:
        # Reset must neither read nor command the hardware leader.
        record._reset_evorl_arms(robot, None, None, None)
    else:
        record._home_piper_followers(robot, duration_s=2.0)

    assert sent[0][1] == initial
    assert sent[-1][0] == pytest.approx(2.0)
    for prefix, _ in arms:
        for key in keys:
            expected = (101.1 if passthrough else 100.0) if key == "gripper.pos" else (0.0 if passthrough else 1.0)
            assert positions[prefix + key] == pytest.approx(expected)
            values = [pose[prefix + key] for _, pose in sent]
            assert values == sorted(values, reverse=key != "gripper.pos")


def test_home_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration must be positive"):
        record._home_piper_followers(None, duration_s=0.0)
