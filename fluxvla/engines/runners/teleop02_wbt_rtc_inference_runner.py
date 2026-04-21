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

import time

import torch

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .aloha_inference_runner import resample_remaining
from .teleop02_wbt_inference_runner import Teleop02WbtInferenceRunner

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class Teleop02WbtRTCInferenceRunner(Teleop02WbtInferenceRunner):
    """Teleop02 WBT inference runner with RTC prefix conditioning.

    Extends Teleop02WbtInferenceRunner by adding RTC support to
    _predict_action, which conditions the model on previously predicted
    actions for smoother trajectory stitching across inference chunks.

    Args:
        rtc_config (dict, optional): RTC configuration dict. Expected keys:
            - enabled (bool): Whether RTC is active.
            - method (str): 'prefix' or 'guidance'.
            - prefix_len (int, optional): Number of prefix steps. If None,
              estimated from last inference time.
    """

    def __init__(self, rtc_config: dict = None, *args, **kwargs):
        self.rtc_config = rtc_config
        super().__init__(*args, **kwargs)

    def run(self,
            initial_instruction:
            str = 'pour water into the cup'):
        """Run inference loop with mode selected by RTC config.

        If RTC guidance uses VJP, run under no_grad and let guidance internals
        enable gradients only where needed; otherwise use inference_mode.
        """
        overwatch.info('Starting Teleop02 WBT RTC inference runner')

        use_vjp = (
            self.rtc_config and self.rtc_config.get('enabled', False)
            and self.rtc_config.get('method', 'prefix') == 'guidance'
            and self.rtc_config.get('use_vjp', False))
        mode_context = torch.no_grad if use_vjp else torch.inference_mode

        with mode_context():
            self._warmup_model(initial_instruction)
            while self._running:
                self._run_episode(initial_instruction)

    def _predict_action(self, inputs):
        """Predict with RTC prefix conditioning.

        If RTC is enabled and a previous action context exists, injects the
        remaining portion of the previous trajectory as a prefix to guide
        the current prediction for smoother chunk transitions.
        """
        prev = self._prev_ctx
        ctx = self._action_ctx

        ctx.inference_start = time.time()

        if (prev is not None and self.rtc_config
                and self.rtc_config.get('enabled', False)):
            offset = (ctx.inference_start - prev.action_timestamp) / self._dt
            remaining = resample_remaining(prev.raw_actions[0], offset)[None]
            # Use configured prefix_len, or estimate from last inference time
            prefix_len = self.rtc_config.get('prefix_len')
            if prefix_len is None:
                prefix_len = int(prev.inference_elapsed * self.publish_rate)
            prefix_len = min(prefix_len, remaining.shape[1])
            print(f"prefix_len: {prefix_len}, remaining.shape[1]: {remaining.shape[1]}")
            if prefix_len > 0:
                inputs['prev_actions'] = torch.from_numpy(remaining).to(
                    device=inputs['states'].device,
                    dtype=inputs['states'].dtype)
                inputs['prefix_len'] = prefix_len
                inputs['rtc_config'] = self.rtc_config

        raw_action = self.vla.predict_action(**inputs)

        ctx.inference_elapsed = time.time() - ctx.inference_start
        print(f'Inference time: {ctx.inference_elapsed:.3f}s')
        test_prefix_len = int(ctx.inference_elapsed * self.publish_rate)
        print(f'Using prefix_len={inputs.get("prefix_len", "N/A")} (test estimate: {test_prefix_len})')
        ctx.raw_actions = raw_action.cpu().numpy()
        return raw_action
