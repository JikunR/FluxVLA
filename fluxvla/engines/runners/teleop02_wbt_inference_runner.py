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

import signal
import time
from collections import deque
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

from ..utils.root import RUNNERS
from .aloha_inference_runner import resample_remaining
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class Teleop02WbtInferenceRunner(BaseInferenceRunner):
    """Runner for Teleop02 WBT (whole-body tracking) loco-mani inference.

    Uses mros middleware instead of rospy. Sends joint-level commands
    plus base_link pose via /teleop_cmd_WBT, matching the WBT action
    space (42-dim).

    The robot has:
        - 1 head camera
        - 33-dim state (31 joints + 2 hand_closed)
        - 42-dim action (31 joint q + 9 base_pose + 2 hand_closed)
    """

    def __init__(self,
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 *args,
                 **kwargs):
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon

        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = ['head']

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'Teleop02WbtOperator',
                'head_rgb_topic': '/head/color/image_raw/compressed',
                'joint_state_topic': '/joint/state',
                'finger_state_topic': '/brainco1/hand/state',
                'finger_cmd_topic': '/brainco1/hand/cmd',
                'teleop_wbt_topic': '/teleop_cmd_WBT',
                'cmd_vel_topic': '/sdk_cmd_vel_vla',
            }

        if 'task_descriptions' not in kwargs or \
                kwargs['task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'pour water into the cup',
            }

        super().__init__(*args, **kwargs)

        self._running = True
        self._dt = 1.0 / self.publish_rate
        self._model_warmed_up = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle SIGINT for graceful shutdown."""
        print('\nShutdown requested...')
        self._running = False

    def run(self, initial_instruction='pour water into the cup'):
        """Main inference loop using time-based rate control.

        Replaces rospy-based loop with pure Python time.sleep.

        Args:
            initial_instruction (str): Default task instruction.
        """
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Starting Teleop02 WBT inference runner')

        with torch.inference_mode():
            self._warmup_model(initial_instruction)
            while self._running:
                self._run_episode(initial_instruction)

    def _warmup_model(self, instruction: str):
        """Run one dummy inference before waiting for real observations."""
        if self._model_warmed_up:
            return

        warmup_start = time.perf_counter()
        print('[warmup] Starting dummy model warmup...', flush=True)

        dummy_obs = {'qpos': np.zeros(33, dtype=np.float32)}
        for camera_name in self.camera_names:
            dummy_obs[camera_name] = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_obs['task_description'] = instruction

        dataset_start = time.perf_counter()
        inputs = self.dataset(dummy_obs)
        print(
            f'[warmup] dataset_transform='
            f'{(time.perf_counter() - dataset_start) * 1000.0:.3f} ms',
            flush=True)

        predict_start = time.perf_counter()
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision):
            _ = self.vla.predict_action(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f'[warmup] predict_action='
            f'{(time.perf_counter() - predict_start) * 1000.0:.3f} ms',
            flush=True)

        self._model_warmed_up = True
        print(
            f'[warmup] Dummy model warmup completed in '
            f'{(time.perf_counter() - warmup_start) * 1000.0:.3f} ms; '
            f'action discarded. Now waiting for real observation data.',
            flush=True)

    def _run_episode(self, default_instruction):
        """Run a single episode without rospy dependency.

        Args:
            default_instruction (str): Default task instruction.
        """
        t = 0

        print ("run episode()")

        while t < self.max_publish_step and self._running:
            instructions = self._get_user_task_instruction(
                default_instruction)
            self._prev_ctx = None
            for instruction in instructions:
                if not self._running:
                    break
                self._action_ctx = SimpleNamespace()
                self._action_ctx.instruction = instruction
                chunk_start = time.perf_counter()
                print(
                    f'[timing] chunk_start step={t} '
                    f'instruction={instruction!r}',
                    flush=True)

                stage_start = time.perf_counter()
                inputs = self._preprocess(instruction)
                print(
                    f'[timing] preprocess_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                stage_start = time.perf_counter()
                with torch.autocast(
                        'cuda',
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision):
                    raw_action = self._predict_action(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(
                    f'[timing] predict_action_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                stage_start = time.perf_counter()
                actions = self._postprocess_actions(raw_action)
                print(
                    f'[timing] postprocess_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                stage_start = time.perf_counter()
                self._execute_actions(actions, None)
                print(
                    f'[timing] execute_actions_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                self._prev_ctx = self._action_ctx
                t += self.action_chunk
                print(
                    f'[timing] chunk_total='
                    f'{(time.perf_counter() - chunk_start) * 1000.0:.3f} ms',
                    flush=True)
                print(f'Published Step {t}')

    def get_ros_observation(self):
        """Get observation from Teleop02WbtOperator via mros.

        Polls operator.get_frame() until data is available.

        Returns:
            tuple: (head_img_rgb, state_33d)
        """
        while self._running:
            get_frame_start = time.perf_counter()
            result = self.ros_operator.get_frame()
            get_frame_elapsed_ms = (
                time.perf_counter() - get_frame_start) * 1000.0
            print(
                f'get_frame() elapsed: {get_frame_elapsed_ms:.3f} ms, '
                f'valid: {result is not False}',
                flush=True)
            if result is not False:
                return result
            time.sleep(0.01)
        return None

    def update_observation_window(self) -> Dict:
        """Update observation window with latest sensor data.

        Returns:
            Dict: Latest observation with 'qpos' (33d) and 'head' image.
        """
        if self.observation_window is None:
            window_init_start = time.perf_counter()
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)
            print(
                f'[timing] observation_window_init='
                f'{(time.perf_counter() - window_init_start) * 1000.0:.3f} ms',
                flush=True)

        stage_start = time.perf_counter()
        result = self.get_ros_observation()
        print(
            f'[timing] get_ros_observation_total='
            f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
            flush=True)
        if result is None:
            return self.observation_window[-1]

        head_img, state = result

        # Apply JPEG compression to match training conditions
        stage_start = time.perf_counter()
        head_img = self._apply_jpeg_compression(head_img)
        print(
            f'[timing] jpeg_compression='
            f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
            flush=True)

        observation = {
            'qpos': state,
            self.camera_names[0]: head_img,  # 'head'
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _preprocess(self, instruction: str) -> dict:
        """Observe environment and build model inputs with timing logs."""
        stage_start = time.perf_counter()
        obs = self.update_observation_window()
        print(
            f'[timing] update_observation_window='
            f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
            flush=True)

        obs['task_description'] = instruction

        stage_start = time.perf_counter()
        inputs = self.dataset(obs)
        print(
            f'[timing] dataset_transform='
            f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
            flush=True)
        return inputs

    def _predict_action(self, inputs):
        """Run model inference with timing instrumentation."""
        self._action_ctx.inference_start = time.time()
        predict_start = time.perf_counter()
        raw_action = self.vla.predict_action(**inputs)
        print(
            f'[timing] vla_predict_call_returned='
            f'{(time.perf_counter() - predict_start) * 1000.0:.3f} ms',
            flush=True)
        return raw_action

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute actions (sync or async).

        In async mode, skips steps that elapsed during inference.

        Args:
            actions (np.ndarray): Array of denormalized 42-dim actions.
            rate: Unused (kept for interface compatibility).
        """
        if not self._running:
            return

        ctx = self._action_ctx

        if self.async_execution and self._prev_ctx is not None:
            ctx.action_timestamp = ctx.inference_start
            offset = (time.time() - ctx.action_timestamp) / self._dt
            actions = resample_remaining(actions, offset)
        else:
            ctx.action_timestamp = time.time()
            if self.execute_horizon is not None:
                actions = actions[:self.execute_horizon]

        self.ros_operator.execute_trajectory(
            actions,
            dt=self._dt,
            async_exec=self.async_execution,
            running_flag_fn=lambda: self._running)

        if self.async_execution and self.execute_horizon is not None:
            time.sleep(self.execute_horizon * self._dt)

    def _move_to_prepare_pose(self):
        """No-op for Teleop02 WBT (teleop-controlled robot)."""
        pass

    def cleanup(self):
        """Clean up resources."""
        print('Cleaning up Teleop02WbtInferenceRunner')
        self._running = False

        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()

        super().cleanup()
