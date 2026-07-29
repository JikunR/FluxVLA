# Copyright 2026 Limx Dynamics
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

from typing import Mapping

WAM_TRAINING_MODES = ('forward', 'idm', 'policy', 'joint', 'vgm_policy')
WAM_MODE_TO_ID = {mode: index for index, mode in enumerate(WAM_TRAINING_MODES)}


def wam_mode_to_id(mode: str) -> int:
    try:
        return WAM_MODE_TO_ID[mode]
    except KeyError as exc:
        raise ValueError(f'Unknown WAM mode: {mode!r}.') from exc


def normalize_wam_mode_probs(
        mode_probs: Mapping[str, float] | None) -> dict[str, float]:
    if mode_probs is None:
        mode_probs = {
            mode: 1.0
            for mode in WAM_TRAINING_MODES if mode != 'vgm_policy'
        }

    unknown = set(mode_probs) - set(WAM_TRAINING_MODES)
    if unknown:
        raise ValueError(f'Unknown WAM mode(s): {sorted(unknown)}')

    probs = {
        mode: float(mode_probs.get(mode, 0.0))
        for mode in WAM_TRAINING_MODES
    }
    if any(value < 0.0 for value in probs.values()):
        raise ValueError(f'`mode_probs` must be non-negative, got {probs}')
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError('At least one `mode_probs` entry must be positive.')
    return {mode: value / total for mode, value in probs.items()}
