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

import copy as _copy
import importlib.util as _importlib_util
import os as _os

_base_path = _os.path.join(
    _os.getcwd(), 'configs/dreamzero/dreamzero_hud04_full_finetune.py')
_spec = _importlib_util.spec_from_file_location(
    '_dreamzero_hud04_full_finetune', _base_path)
_base = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

_frame_window_size = 5
_image_size = 64

model = _copy.deepcopy(_base.model)
train_dataloader = _copy.deepcopy(_base.train_dataloader)
runner = _copy.deepcopy(_base.runner)

model.update(
    frame_window_size=_frame_window_size,
    pretrained_name_or_path=None,  # skip loading full ckpt; arch changed
    use_cache=True,
)
model['vla_head'].update(
    num_frames=_frame_window_size,
    num_frame_per_block=1,
    frame_seqlen=32,
    hidden_size=256,
    dit_dim=512,
    dit_ffn_dim=2048,
    dit_num_heads=8,
    dit_num_layers=4,
    num_inference_steps=4,
    max_chunk_size=1,
)

train_dataloader.update(
    per_device_batch_size=1,
    per_device_num_workers=0,
)

runner.pop('max_epochs', None)
runner.update(
    max_steps=20,
    save_iter_interval=20,
    max_keep_ckpts=1,
    sharding_strategy='no-shard',  # single-GPU: no FSDP sharding needed
)
runner['metric'].update(
    active_trackers=('jsonl', ),
    grad_accumulation_steps=1,
)


def _patch_debug_data_cfg(node):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if current.get('type') == 'ResizeImages':
                current.update(height=_image_size, width=_image_size)
            if current.get('type') == 'PrepareVideo':
                current['frame_window_size'] = _frame_window_size
            if 'frame_window_size' in current:
                current['frame_window_size'] = _frame_window_size
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


_patch_debug_data_cfg(train_dataloader)

del _base, _base_path, _copy, _importlib_util, _os, _spec
del _patch_debug_data_cfg
