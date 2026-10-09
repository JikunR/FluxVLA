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

from fluxvla.engines.runners.serving.policy import resolve_report_result_root


def test_report_result_root_defaults_next_to_checkpoints(tmp_path):
    checkpoint = tmp_path / 'run' / 'checkpoints' / 'step-10.safetensors'

    result_root = resolve_report_result_root(None, checkpoint)

    assert result_root == tmp_path / 'run'


def test_relative_report_result_root_uses_checkpoint_run_dir(tmp_path):
    checkpoint = tmp_path / 'run' / 'checkpoints' / 'step-10.safetensors'

    result_root = resolve_report_result_root('reports', checkpoint)

    assert result_root == tmp_path / 'run' / 'reports'
