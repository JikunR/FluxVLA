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
"""Transforms for Cosmos3-Nano VLA.

Two transforms:

* ``ProcessCosmos3Prompt`` — tokenise task description text with the
  Qwen3-VL chat template (text-only, no image placeholders).
* ``BuildCosmos3Sequence`` — pad actions to ``max_action_dim``, attach
  ``domain_id`` and ``raw_action_dim``, and build the ``SequencePlan`` expected
  by ``Cosmos3VLA.forward()``.
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional

import numpy as np

import fluxvla.models.third_party_models.cosmos3 as _c3  # noqa: F401
from fluxvla.engines import TRANSFORMS

# ---------------------------------------------------------------------------
# ProcessCosmos3Prompt
# ---------------------------------------------------------------------------


@TRANSFORMS.register_module()
class ProcessCosmos3Prompt:
    """Tokenise task description for Cosmos3-Nano using the Qwen3-VL chat
    template (pure text, no image/video placeholder tokens).

    Args:
        qwen3_vl_model_path (str): Path to the Qwen3-VL model directory (or
            HuggingFace hub name) used to load the tokenizer.
        max_len (int): Maximum token length.  Tokens beyond this limit are
            truncated (right-side).  Defaults to 4096.
        use_system_prompt (bool): Whether to prepend a system prompt.
            Defaults to False.
        cfg_dropout_rate (float): Probability of replacing the caption with an
            empty string (classifier-free guidance training).  Defaults to 0.0.
    """

    def __init__(
        self,
        qwen3_vl_model_path: str,
        max_len: int = 4096,
        use_system_prompt: bool = False,
        cfg_dropout_rate: float = 0.0,
    ) -> None:
        self.qwen3_vl_model_path = qwen3_vl_model_path
        self.max_len = max_len
        self.use_system_prompt = use_system_prompt
        self.cfg_dropout_rate = cfg_dropout_rate

        # Lazy-initialised to avoid heavy imports at registry load time.
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.qwen3_vl_model_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def __call__(self, data: Dict) -> Dict:
        tokenizer = self._get_tokenizer()

        caption = data.get('task_description', '')
        if not isinstance(caption, str):
            caption = str(caption)

        # CFG dropout: replace with empty caption with probability
        # cfg_dropout_rate
        if self.cfg_dropout_rate > 0.0 and random.random(
        ) < self.cfg_dropout_rate:
            caption = ''

        # Build chat-template messages (no image/video tokens)
        conversations = []
        if self.use_system_prompt:
            conversations.append({
                'role':
                'system',
                'content':
                'You are a helpful robot assistant.',
            })
        conversations.append({'role': 'user', 'content': caption})

        try:
            token_ids = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                # add_vision_id=False is specific to processors that
                # expose it (Qwen3VLProcessor).  For a plain tokenizer
                # the kwarg is harmless or absent – use a try/except.
                add_vision_id=False,
                return_dict=False,
            )
        except TypeError:
            # Fallback for tokenizers that don't accept add_vision_id
            token_ids = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )

        # Truncate to max_len
        token_ids = token_ids[:self.max_len]

        data['text_token_ids'] = np.array(token_ids, dtype=np.int64)
        return data


# ---------------------------------------------------------------------------
# BuildCosmos3Sequence
# ---------------------------------------------------------------------------


@TRANSFORMS.register_module()
class BuildCosmos3Sequence:
    """Finalize cosmos3-nano-compatible batch fields.

    Responsibilities:
    * Pad action tensor to ``max_action_dim`` dimensions.
    * Record ``raw_action_dim`` (the original, unpadded dimensionality) for
      loss masking inside the model.
    * Map ``embodiment_id`` to a cosmos3 ``domain_id`` integer via
      ``embodiment_to_domain_id``.
    * Construct a ``SequencePlan`` describing conditioning / generation roles
      of video frames and action tokens.
    * Optionally populate ``conditioning_fps``.

    Args:
        max_action_dim (int): Target padded action dimension.
        embodiment_to_domain_id (Dict[int, int]): Mapping from the integer
            ``embodiment_id`` in the batch to a cosmos3 domain ID.  Unknown
            IDs default to 0 (``no_action``).
        mode (str): One of ``"policy"`` (action fully denoised, no
            conditioning action frames), ``"forward_dynamics"`` (all action
            frames are conditioning, video frames are generated), or
            ``"joint"`` (randomly select one of the three modes each call).
        frame_window_size (int): Total number of video frames in the window
            (including conditioning frames).  Used to set
            ``condition_frame_indexes_vision``.
        num_conditioning_vision_frames (int): Number of leading video frames
            treated as conditioning (clean) input.  Defaults to 1.
        conditioning_fps (float): Frame-rate to record in the output dict.
            Defaults to 15.0.
    """

    # cosmos3 ModelMode strings (mirrors cosmos_framework.inference.args.ModelMode)  # noqa: E501
    MODE_POLICY = 'policy'
    MODE_FORWARD_DYNAMICS = 'forward_dynamics'
    MODE_INVERSE_DYNAMICS = 'inverse_dynamics'
    MODE_JOINT = 'joint'

    def __init__(
        self,
        max_action_dim: int = 64,
        embodiment_to_domain_id: Optional[Dict[int, int]] = None,
        mode: str = 'policy',
        frame_window_size: int = 5,
        num_conditioning_vision_frames: int = 1,
        conditioning_fps: float = 15.0,
    ) -> None:
        self.max_action_dim = max_action_dim
        self.embodiment_to_domain_id = embodiment_to_domain_id or {}
        self.mode = mode
        self.frame_window_size = frame_window_size
        self.num_conditioning_vision_frames = num_conditioning_vision_frames
        self.conditioning_fps = conditioning_fps

        assert mode in (
            self.MODE_POLICY,
            self.MODE_FORWARD_DYNAMICS,
            self.MODE_INVERSE_DYNAMICS,
            self.MODE_JOINT,
        ), f'Unknown mode: {mode!r}'

    def _choose_mode(self) -> str:
        if self.mode == self.MODE_JOINT:
            return random.choice([
                self.MODE_POLICY,
                self.MODE_FORWARD_DYNAMICS,
                self.MODE_INVERSE_DYNAMICS,
            ])
        return self.mode

    def __call__(self, data: Dict) -> Dict:
        # ----------------------------------------------------------------
        # 1. Determine action dimension and pad
        # ----------------------------------------------------------------
        action_key = 'actions'
        action = data.get(action_key)
        if action is None:
            action_key = 'action'
            action = data.get(action_key)

        if action is not None:
            if isinstance(action, np.ndarray):
                raw_action_dim = action.shape[-1]
                pad_width = self.max_action_dim - raw_action_dim
                if pad_width > 0:
                    pad_shape = list(action.shape)
                    pad_shape[-1] = pad_width
                    action = np.concatenate(
                        [action,
                         np.zeros(pad_shape, dtype=action.dtype)],
                        axis=-1,
                    )
                data[action_key] = action
            else:
                import torch  # noqa: F401
                raw_action_dim = action.shape[-1]
                pad_width = self.max_action_dim - raw_action_dim
                if pad_width > 0:
                    import torch.nn.functional as F
                    action = F.pad(action, (0, pad_width))
                    data[action_key] = action
            data['raw_action_dim'] = np.array(raw_action_dim, dtype=np.int64)
        else:
            data['raw_action_dim'] = np.array(
                self.max_action_dim, dtype=np.int64)

        # ----------------------------------------------------------------
        # 2. Map embodiment_id → domain_id
        # ----------------------------------------------------------------
        embodiment_id = data.get('embodiment_ids')
        if embodiment_id is not None:
            eid = int(embodiment_id) if not hasattr(
                embodiment_id, 'item') else embodiment_id.item()
            domain_id = self.embodiment_to_domain_id.get(eid, 0)
        else:
            domain_id = 0
        data['domain_id'] = np.array(domain_id, dtype=np.int64)

        # ----------------------------------------------------------------
        # 3. Build SequencePlan
        #
        # Import lazily so that the registry can load without cosmos-framework
        # installed (e.g., during a plain import of fluxvla).
        # ----------------------------------------------------------------
        try:
            from _c3.data.vfm.sequence_packing import SequencePlan
        except ImportError:
            # Fall back to a minimal placeholder dataclass when
            # cosmos-framework is not installed.
            from dataclasses import dataclass
            from dataclasses import field as dc_field

            @dataclass
            class SequencePlan:  # type: ignore[no-redef]
                has_text: bool = True
                has_vision: bool = True
                has_action: bool = False
                has_sound: bool = False
                condition_frame_indexes_vision: List[int] = dc_field(
                    default_factory=list)
                condition_frame_indexes_action: List[int] = dc_field(
                    default_factory=list)

        effective_mode = self._choose_mode()

        cond_vision: List[int] = list(
            range(self.num_conditioning_vision_frames))
        has_action = action is not None

        if effective_mode == self.MODE_POLICY:
            # All vision frames conditioned; action fully generated.
            cond_action: List[int] = []
            cond_vision = list(range(self.frame_window_size))
        elif effective_mode == self.MODE_FORWARD_DYNAMICS:
            # Leading vision frames + all action conditioned; future video generated.  # noqa: E501
            cond_action = list(range(
                action.shape[0])) if action is not None else []
            cond_vision = list(range(self.num_conditioning_vision_frames))
        else:  # inverse_dynamics
            # First + last vision frame conditioned; action generated.
            cond_action = []
            cond_vision = [0, self.frame_window_size - 1]

        seq_plan = SequencePlan(
            has_text=True,
            has_vision=True,
            has_action=has_action,
            has_sound=False,
            condition_frame_indexes_vision=cond_vision,
            condition_frame_indexes_action=cond_action,
        )
        data['sequence_plan'] = seq_plan
        data['conditioning_fps'] = np.array(
            self.conditioning_fps, dtype=np.float32)

        return data
