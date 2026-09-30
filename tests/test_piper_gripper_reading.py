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

from types import SimpleNamespace

import pytest

from lerobot.teleoperators.piper_leader.piper_leader import PiperLeader


@pytest.mark.parametrize(
    "joint_ctrl,gripper_ctrl,manual,prefer_ctrl,expected",
    [
        (None, 65.0, None, True, 65.0),
        ({"joint_1.pos": 1.0}, 65.0, None, True, 65.0),
        (None, None, None, True, 12.0),
        (None, 65.0, True, True, 12.0),
        (None, 65.0, None, False, 12.0),
    ],
)
def test_gripper_source_is_independent_of_joint_fallback(
    joint_ctrl, gripper_ctrl, manual, prefer_ctrl, expected
):
    leader = PiperLeader.__new__(PiperLeader)
    leader.config = SimpleNamespace(prefer_ctrl_messages=prefer_ctrl, fallback_to_feedback=True)
    leader._manual_control_enabled = manual
    leader._wait_for_fresh_feedback_if_needed = lambda: None
    leader._read_joint_from_ctrl = lambda: joint_ctrl
    leader._read_joint_from_feedback = lambda: {"joint_1.pos": 2.0}
    leader._read_gripper_from_ctrl = lambda: gripper_ctrl
    leader._read_gripper_from_feedback = lambda: 12.0
    leader.arm = SimpleNamespace(
        GetArmJointMsgs=lambda: SimpleNamespace(time_stamp=1.0),
        GetArmGripperMsgs=lambda: SimpleNamespace(time_stamp=1.0),
    )

    action = leader._read_raw_action()

    assert action["gripper.pos"] == expected
    assert action["joint_1.pos"] == (1.0 if joint_ctrl is not None and not manual and prefer_ctrl else 2.0)


def test_read_only_connect_does_not_require_control_frames(monkeypatch):
    leader = PiperLeader.__new__(PiperLeader)
    leader.id = "test_leader"
    leader._is_connected = False
    leader.config = SimpleNamespace(
        startup_sleep_s=0.0, read_only_teaching_mode=True,
        require_calibration=True, port="can0",
    )
    calls = []
    leader.arm = SimpleNamespace(ConnectPort=lambda: calls.append("connect"))
    monkeypatch.setattr(PiperLeader, "is_calibrated", property(lambda self: True))

    leader.connect()

    assert leader.is_connected
    assert calls == ["connect"]


def test_missing_gripper_feedback_is_not_a_closed_gripper():
    leader = PiperLeader.__new__(PiperLeader)
    leader.arm = SimpleNamespace(GetArmGripperMsgs=lambda: SimpleNamespace(
        time_stamp=0.0, gripper_state=SimpleNamespace(grippers_angle=0)
    ))
    assert leader._read_gripper_from_feedback() is None


def test_teaching_gripper_stays_open_until_first_real_sample():
    leader = PiperLeader.__new__(PiperLeader)
    leader.config = SimpleNamespace(
        prefer_ctrl_messages=True, fallback_to_feedback=True, read_only_teaching_mode=True,
    )
    leader.calibration = {"gripper.pos": SimpleNamespace(range_max=100000)}
    leader._from_calibration_units = lambda value: value / 1000
    leader._manual_control_enabled = None
    leader._read_joint_from_ctrl = lambda: {"joint_1.pos": 1.0}
    leader._read_gripper_from_ctrl = lambda: None
    leader._read_gripper_from_feedback = lambda: None
    leader.arm = SimpleNamespace(GetArmGripperMsgs=lambda: SimpleNamespace(time_stamp=0.0))
    assert leader._read_raw_action()["gripper.pos"] == 100.0
    leader._read_gripper_from_ctrl = lambda: 42.0
    assert leader._read_raw_action()["gripper.pos"] == 42.0
    leader._read_gripper_from_ctrl = lambda: None
    assert leader._read_raw_action()["gripper.pos"] == 42.0
    # A received zero is a legitimate close command and must remain distinguishable from missing data.
    leader._read_gripper_from_ctrl = lambda: 0.0
    assert leader._read_raw_action()["gripper.pos"] == 0.0
