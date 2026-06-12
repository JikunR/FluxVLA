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
"""Collator for Cosmos3-Nano batches.

Handles the mixed bag of:
* Regular tensors / ndarrays that should be stacked.
* Variable-length ``text_token_ids`` that must be padded.
* ``sequence_plan`` dataclass objects that must be kept as a plain list.
* Meta strings / dicts that should remain as lists.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import torch

from fluxvla.engines import COLLATORS


@COLLATORS.register_module()
class Cosmos3Collator:
    """Collate function for Cosmos3-Nano training batches.

    Args:
        tensor_keys (List[str]): Keys whose values are ``np.ndarray`` or
            ``torch.Tensor`` and should be stacked along dim 0.
        sequence_keys (List[str]): Keys whose values are 1-D sequences of
            integers (``text_token_ids``).  Each sample may have a different
            length; they will be **right-padded** to the longest sequence in
            the batch with ``pad_id`` and stacked.
        list_keys (List[str]): Keys whose values should be collected into a
            plain Python list (e.g., ``sequence_plan``, ``task_description``).
            Defaults to ``['sequence_plan']`` because ``SequencePlan`` is a
            dataclass that cannot be stacked by the auto-detect fallback.
        meta_keys (List[str]): Alias for ``list_keys`` – kept for API
            consistency with ``DictCollator``.
        pad_id (int): Padding token ID used for ``sequence_keys``.
            Defaults to 0.
    """

    def __init__(
        self,
        tensor_keys: Optional[List[str]] = None,
        sequence_keys: Optional[List[str]] = None,
        list_keys: Optional[List[str]] = None,
        meta_keys: Optional[List[str]] = None,
        pad_id: int = 0,
    ) -> None:
        self.tensor_keys: List[str] = tensor_keys or []
        self.sequence_keys: List[str] = sequence_keys or ['text_token_ids']
        # sequence_plan is a non-stackable dataclass; always keep as list.
        default_list_keys = ['sequence_plan']
        self.list_keys: List[str] = list(
            set(default_list_keys + (list_keys or []) + (meta_keys or [])))
        self.pad_id = pad_id

    # ------------------------------------------------------------------
    def __call__(self, batch: List[Dict]) -> Dict:
        if not batch:
            return {}

        all_keys = batch[0].keys()
        collated: Dict = {}

        for key in all_keys:
            values = [sample[key] for sample in batch]
            first = values[0]

            if key in self.tensor_keys:
                # Stack numpy arrays or tensors to [B, ...]
                if isinstance(first, np.ndarray):
                    collated[key] = torch.from_numpy(np.stack(values))
                elif isinstance(first, torch.Tensor):
                    collated[key] = torch.stack(values)
                else:
                    raise TypeError(
                        f"Cosmos3Collator: key '{key}' is in tensor_keys "
                        f'but got unsupported type {type(first).__name__}')

            elif key in self.sequence_keys:
                # Pad variable-length integer sequences to the longest in the batch  # noqa: E501
                max_len = max(len(v) for v in values)
                padded = np.full((len(values), max_len),
                                 self.pad_id,
                                 dtype=np.int64)
                for i, v in enumerate(values):
                    if isinstance(v, np.ndarray):
                        padded[i, :len(v)] = v
                    else:
                        padded[i, :len(v)] = np.asarray(v, dtype=np.int64)
                collated[key] = torch.from_numpy(padded)

            elif key in self.list_keys:
                # Keep as list (sequence_plan objects, strings, dicts, etc.)
                collated[key] = values

            else:
                # Auto-detect fallback: try to stack, otherwise keep as list
                try:
                    if isinstance(first, np.ndarray):
                        collated[key] = torch.from_numpy(np.stack(values))
                    elif isinstance(first, torch.Tensor):
                        collated[key] = torch.stack(values)
                    elif isinstance(first, (int, float)):
                        collated[key] = torch.tensor(values)
                    else:
                        collated[key] = values
                except Exception:
                    collated[key] = values

        return collated
