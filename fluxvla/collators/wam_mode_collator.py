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

import random
from typing import Dict, Optional

import torch

from fluxvla.engines import COLLATORS
from fluxvla.wam_modes import (WAM_TRAINING_MODES, normalize_wam_mode_probs,
                               wam_mode_to_id)
from .dict_collator import DictCollator


@COLLATORS.register_module()
class WAMModeCollator(DictCollator):
    """Dict collator that assigns one WAM/FastWAM training mode to a batch."""

    def __init__(
        self,
        keys=None,
        meta_keys=None,
        mode: str = 'batch',
        mode_probs: Optional[Dict[str, float]] = None,
        output_key: str = 'training_mode',
    ) -> None:
        super().__init__(keys=keys, meta_keys=meta_keys)
        if mode != 'batch' and mode not in WAM_TRAINING_MODES:
            raise ValueError(f'Unknown WAM training mode: {mode!r}.')
        self.mode = mode
        self.mode_probs = normalize_wam_mode_probs(mode_probs)
        self.output_key = output_key

    def _choose_mode(self) -> str:
        if self.mode != 'batch':
            return self.mode
        modes = list(WAM_TRAINING_MODES)
        probs = [self.mode_probs[mode] for mode in modes]
        return random.choices(modes, weights=probs, k=1)[0]

    def __call__(self, batch):
        collated = super().__call__(batch)
        mode = self._choose_mode()
        collated[self.output_key] = torch.full(
            (len(batch), ),
            wam_mode_to_id(mode),
            dtype=torch.long,
        )
        return collated
