# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "ACTConfig": ".act.configuration_act",
    "DiffusionConfig": ".diffusion.configuration_diffusion",
    "Evo1Config": ".evo1.configuration_evo1",
    "GrootConfig": ".groot.configuration_groot",
    "PI0Config": ".pi0.configuration_pi0",
    "PI0FastConfig": ".pi0_fast.configuration_pi0_fast",
    "PI05Config": ".pi05.configuration_pi05",
    "SmolVLAConfig": ".smolvla.configuration_smolvla",
    "SmolVLANewLineProcessor": ".smolvla.processor_smolvla",
    "TDMPCConfig": ".tdmpc.configuration_tdmpc",
    "VQBeTConfig": ".vqbet.configuration_vqbet",
    "WallXConfig": ".wall_x.configuration_wall_x",
    "XVLAConfig": ".xvla.configuration_xvla",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_ATTRS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_ATTRS])
