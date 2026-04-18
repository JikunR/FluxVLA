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

import unittest
from collections import deque
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeRegistry:

    def register_module(self):

        def decorator(cls):
            return cls

        return decorator


def _install_fake_package_modules():
    for name in [
            'fluxvla',
            'fluxvla.engines',
            'fluxvla.engines.utils',
            'fluxvla.engines.operators',
            'fluxvla.engines.runners',
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    root_module = types.ModuleType('fluxvla.engines.utils.root')
    root_module.OPERATORS = _FakeRegistry()
    root_module.RUNNERS = _FakeRegistry()
    sys.modules['fluxvla.engines.utils.root'] = root_module


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_operator_class():
    _install_fake_package_modules()
    module = _load_module(
        'fluxvla.engines.operators.teleop02_wbt_operator',
        'fluxvla/engines/operators/teleop02_wbt_operator.py')
    return module.Teleop02WbtOperator


def _load_runner_class():
    _install_fake_package_modules()

    torch_module = types.ModuleType('torch')
    torch_module.inference_mode = nullcontext
    torch_module.autocast = lambda *args, **kwargs: nullcontext()
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: False, synchronize=lambda: None)
    sys.modules['torch'] = torch_module

    aloha_module = types.ModuleType(
        'fluxvla.engines.runners.aloha_inference_runner')
    aloha_module.resample_remaining = lambda actions, offset: actions
    sys.modules[
        'fluxvla.engines.runners.aloha_inference_runner'] = aloha_module

    base_module = types.ModuleType(
        'fluxvla.engines.runners.base_inference_runner')

    class BaseInferenceRunner:
        pass

    base_module.BaseInferenceRunner = BaseInferenceRunner
    sys.modules[
        'fluxvla.engines.runners.base_inference_runner'] = base_module

    module = _load_module(
        'fluxvla.engines.runners.teleop02_wbt_inference_runner',
        'fluxvla/engines/runners/teleop02_wbt_inference_runner.py')
    return module.Teleop02WbtInferenceRunner


class _FakeSubscriber:

    def __init__(self, msg):
        self.msg = msg

    def readMsgRT(self):
        return self.msg


class _FakePublisher:

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _rot6d_from_euler_zyx(yaw, pitch, roll):
    matrix = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    return matrix[:2].reshape(-1)


def _install_fake_mros_messages():
    for name in [
            'mros',
            'mros.controller_msgs',
            'mros.controller_msgs.msg',
            'mros.std_msgs',
            'mros.std_msgs.msg',
            'mros.teleop_msgs',
            'mros.teleop_msgs.msg',
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))

    class JointCmd:
        pass

    class Float32Array:
        pass

    class TeleopMsg:

        def __init__(self):
            self.header = SimpleNamespace(frame_id='')
            self.world = SimpleNamespace(
                orientation=SimpleNamespace(w=0.0))
            self.joint_cmd = None
            self.anchors = []

    class KeyPoint:

        def __init__(self):
            self.name = ''
            self.pose = SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=0.0, w=1.0))

    sys.modules['mros.controller_msgs.msg'].JointCmd = JointCmd
    sys.modules['mros.std_msgs.msg'].Float32Array = Float32Array
    sys.modules['mros.teleop_msgs.msg'].TeleopMsg = TeleopMsg
    sys.modules['mros.teleop_msgs.msg'].KeyPoint = KeyPoint


class TestTeleop02WbtOperator(unittest.TestCase):

    def test_get_frame_returns_head_and_left_wrist_images(self):
        Teleop02WbtOperator = _load_operator_class()
        operator = Teleop02WbtOperator.__new__(Teleop02WbtOperator)
        operator.color_subscriber = _FakeSubscriber(
            SimpleNamespace(data=b'head'))
        operator.left_wrist_color_subscriber = _FakeSubscriber(
            SimpleNamespace(data=b'left_wrist'))
        operator.joint_state_subscriber = _FakeSubscriber(
            SimpleNamespace(q=np.arange(31, dtype=np.float32)))
        operator.finger_state_subscriber = _FakeSubscriber(
            SimpleNamespace(data=np.zeros(12, dtype=np.float32)))
        operator.last_finger_state = np.zeros(12, dtype=np.float32)
        operator.last_finger_cmd = np.zeros(14, dtype=np.float32)

        head_bgr = np.full((2, 2, 3), [1, 2, 3], dtype=np.uint8)
        left_wrist_bgr = np.full((2, 2, 3), [4, 5, 6], dtype=np.uint8)

        def decode(msg):
            return head_bgr if msg.data == b'head' else left_wrist_bgr

        operator._compressed_msg_to_numpy = decode

        head_img, left_wrist_img, state = operator.get_frame(timeout=0.01)

        np.testing.assert_array_equal(head_img, head_bgr[:, :, ::-1])
        np.testing.assert_array_equal(
            left_wrist_img, left_wrist_bgr[:, :, ::-1])
        np.testing.assert_array_equal(state[:31], np.arange(31))
        np.testing.assert_array_equal(state[31:], np.array([0.0, 0.0]))

    def test_send_action_accumulates_xy_yaw_and_keeps_absolute_z_pitch_roll(
            self):
        _install_fake_mros_messages()
        Teleop02WbtOperator = _load_operator_class()
        operator = Teleop02WbtOperator.__new__(Teleop02WbtOperator)
        operator._accum_base_pos = np.array([0.0, 0.0, 0.9])
        operator._accum_base_rot = Rotation.identity()
        operator.teleop_wbt_publisher = _FakePublisher()
        operator.finger_publisher = _FakePublisher()
        operator.last_finger_cmd = np.zeros(14, dtype=np.float32)

        first_action = np.zeros(42, dtype=np.float64)
        first_action[31:34] = [1.0, 2.0, 0.7]
        first_action[34:40] = _rot6d_from_euler_zyx(0.1, 0.2, 0.3)

        second_action = np.zeros(42, dtype=np.float64)
        second_action[31:34] = [0.5, -1.0, 0.8]
        second_action[34:40] = _rot6d_from_euler_zyx(-0.04, -0.2, 0.1)

        operator.send_action(first_action)
        operator.send_action(second_action)

        first_anchor = operator.teleop_wbt_publisher.messages[0].anchors[0]
        second_anchor = operator.teleop_wbt_publisher.messages[1].anchors[0]

        np.testing.assert_allclose(
            [
                first_anchor.pose.position.x,
                first_anchor.pose.position.y,
                first_anchor.pose.position.z,
            ],
            [1.0, 2.0, 0.7])
        np.testing.assert_allclose(
            [
                second_anchor.pose.position.x,
                second_anchor.pose.position.y,
                second_anchor.pose.position.z,
            ],
            [1.5, 1.0, 0.8])

        first_rot = Rotation.from_quat([
            first_anchor.pose.orientation.x,
            first_anchor.pose.orientation.y,
            first_anchor.pose.orientation.z,
            first_anchor.pose.orientation.w,
        ])
        second_rot = Rotation.from_quat([
            second_anchor.pose.orientation.x,
            second_anchor.pose.orientation.y,
            second_anchor.pose.orientation.z,
            second_anchor.pose.orientation.w,
        ])
        np.testing.assert_allclose(
            first_rot.as_euler('ZYX'), [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(
            second_rot.as_euler('ZYX'), [0.06, -0.2, 0.1], atol=1e-6)


class TestTeleop02WbtInferenceRunner(unittest.TestCase):

    def test_update_observation_window_includes_left_wrist_camera(self):
        Teleop02WbtInferenceRunner = _load_runner_class()
        runner = Teleop02WbtInferenceRunner.__new__(
            Teleop02WbtInferenceRunner)
        runner.observation_window = None
        runner.camera_names = ['head', 'left_wrist']
        runner._apply_jpeg_compression = lambda img: img

        head_img = np.zeros((2, 2, 3), dtype=np.uint8)
        left_wrist_img = np.ones((2, 2, 3), dtype=np.uint8)
        state = np.arange(33, dtype=np.float32)
        runner.get_ros_observation = lambda: (
            head_img, left_wrist_img, state)

        observation = Teleop02WbtInferenceRunner.update_observation_window(
            runner)

        self.assertIsInstance(runner.observation_window, deque)
        np.testing.assert_array_equal(observation['qpos'], state)
        np.testing.assert_array_equal(observation['head'], head_img)
        np.testing.assert_array_equal(
            observation['left_wrist'], left_wrist_img)

    def test_wbt_jpeg_compression_wraps_base_bgr_api_as_rgb(self):
        Teleop02WbtInferenceRunner = _load_runner_class()
        runner = Teleop02WbtInferenceRunner.__new__(
            Teleop02WbtInferenceRunner)

        rgb_img = np.array([[[10, 20, 30]]], dtype=np.uint8)
        bgr_after_jpeg = np.array([[[1, 2, 3]]], dtype=np.uint8)
        base_inputs = []

        def fake_base_jpeg(img):
            base_inputs.append(img.copy())
            return bgr_after_jpeg

        runner._apply_jpeg_compression = fake_base_jpeg

        compressed_rgb = runner._apply_jpeg_compression_rgb(rgb_img)

        np.testing.assert_array_equal(
            base_inputs[0], np.array([[[30, 20, 10]]], dtype=np.uint8))
        np.testing.assert_array_equal(
            compressed_rgb, np.array([[[3, 2, 1]]], dtype=np.uint8))

    def test_write_debug_jpeg_image_converts_rgb_to_bgr_for_cv2(self):
        Teleop02WbtInferenceRunner = _load_runner_class()
        runner = Teleop02WbtInferenceRunner.__new__(
            Teleop02WbtInferenceRunner)
        runner_module = sys.modules[
            'fluxvla.engines.runners.teleop02_wbt_inference_runner']

        writes = []
        original_imwrite = runner_module.cv2.imwrite
        runner_module.cv2.imwrite = lambda path, img: writes.append(
            (path, img.copy()))
        try:
            rgb_img = np.array([[[10, 20, 30]]], dtype=np.uint8)
            runner._write_debug_jpeg_image('/tmp/debug.png', rgb_img)
        finally:
            runner_module.cv2.imwrite = original_imwrite

        self.assertEqual(writes[0][0], '/tmp/debug.png')
        np.testing.assert_array_equal(
            writes[0][1], np.array([[[30, 20, 10]]], dtype=np.uint8))

    def test_update_observation_window_dumps_post_jpeg_debug_images(self):
        Teleop02WbtInferenceRunner = _load_runner_class()
        runner = Teleop02WbtInferenceRunner.__new__(
            Teleop02WbtInferenceRunner)
        runner.observation_window = None
        runner.camera_names = ['head', 'left_wrist']
        runner.debug_jpeg_dump_dir = 'debug_jpeg'
        runner.debug_jpeg_dump_max_frames = 1
        runner._debug_jpeg_dump_count = 0

        head_img = np.zeros((2, 2, 3), dtype=np.uint8)
        left_wrist_img = np.ones((2, 2, 3), dtype=np.uint8)
        post_jpeg_head = np.full((2, 2, 3), 10, dtype=np.uint8)
        post_jpeg_left_wrist = np.full((2, 2, 3), 20, dtype=np.uint8)
        state = np.arange(33, dtype=np.float32)
        runner.get_ros_observation = lambda: (
            head_img, left_wrist_img, state)
        runner._apply_jpeg_compression_rgb = lambda img: (
            post_jpeg_head if img is head_img else post_jpeg_left_wrist)

        writes = []
        runner._write_debug_jpeg_image = lambda path, img: writes.append(
            (path, img.copy()))

        Teleop02WbtInferenceRunner.update_observation_window(runner)
        Teleop02WbtInferenceRunner.update_observation_window(runner)

        self.assertEqual(len(writes), 2)
        self.assertTrue(writes[0][0].endswith('frame_000000_head.png'))
        self.assertTrue(
            writes[1][0].endswith('frame_000000_left_wrist.png'))
        np.testing.assert_array_equal(writes[0][1], post_jpeg_head)
        np.testing.assert_array_equal(writes[1][1], post_jpeg_left_wrist)
        self.assertEqual(runner._debug_jpeg_dump_count, 1)


if __name__ == '__main__':
    unittest.main()
