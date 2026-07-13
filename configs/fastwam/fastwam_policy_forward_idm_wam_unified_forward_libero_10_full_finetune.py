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
#
# Ablation: FastWAM with WAM-style unified training forward while keeping
# FastWAM internal step mode sampling.

_base_ = './fastwam_policy_forward_idm_per_rank_libero_10_full_finetune.py'

model = dict(
    vla_head=dict(
        sample_mode_in_forward=True,
        unified_training_forward=True,
    ), )
