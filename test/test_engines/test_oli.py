import sys
import types
from types import SimpleNamespace

import numpy as np


def test_oli_operator_registered():
    import fluxvla.engines.operators  # noqa: F401
    from fluxvla.engines.utils.root import OPERATORS
    assert OPERATORS.get('OliOperator') is not None


def test_rot6d_identity_to_unit_quat():
    from fluxvla.engines.operators.oli_operator import _rot6d_to_quat_xyzw
    rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    quat = _rot6d_to_quat_xyzw(rot6d)
    # identity rotation -> [0, 0, 0, 1] (xyzw)
    assert np.allclose(quat, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_oli_runner_registered_and_subclass():
    import fluxvla.engines.runners  # noqa: F401
    from fluxvla.engines.runners.base_inference_runner import \
        BaseInferenceRunner
    from fluxvla.engines.runners.oli_inference_runner import OliInferenceRunner
    from fluxvla.engines.utils.root import RUNNERS
    assert RUNNERS.get('OliInferenceRunner') is not None
    assert issubclass(OliInferenceRunner, BaseInferenceRunner)


def test_oli_config_loads():
    import os

    from mmengine import Config
    path = os.path.join('configs', 'gr00t',
                        'gr00t_eagle_3b_oli_full_finetune.py')
    cfg = Config.fromfile(path)
    assert cfg.inference.type == 'OliInferenceRunner'
    assert cfg.inference.operator.type == 'OliOperator'
    assert cfg.inference.denormalize_action.action_dim == 42
    assert cfg.inference.publish_rate == 30


def test_is_degenerate_rot6d():
    from fluxvla.engines.operators.oli_operator import _is_degenerate_rot6d
    assert _is_degenerate_rot6d([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) is True
    assert _is_degenerate_rot6d([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) is True
    assert _is_degenerate_rot6d([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) is False


def test_rot6d_quat_known_rotations():
    from fluxvla.engines.operators.oli_operator import _rot6d_to_quat_xyzw

    # 90 degrees about z
    q = _rot6d_to_quat_xyzw([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert np.allclose(
        np.abs(q), [0.0, 0.0, 0.70710678, 0.70710678], atol=1e-3)
    # 180 degrees about x (exercises non-trace branch)
    q = _rot6d_to_quat_xyzw([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    assert np.allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-3)


def test_mros_oli_operator_publishes_wbt_teleop_message(monkeypatch):
    from fluxvla.engines.operators.oli_operator import MrosOliOperator

    class JointCmd:
        pass

    class Float32Array:
        pass

    class TeleopMsg:

        def __init__(self):
            self.header = SimpleNamespace(frame_id='')
            self.world = SimpleNamespace(
                orientation=SimpleNamespace(w=0.0))

    class KeyPoint:

        def __init__(self):
            self.name = ''
            self.pose = SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(
                    x=0.0, y=0.0, z=0.0, w=1.0))

    mros_module = types.ModuleType('mros')
    controller_module = types.ModuleType('mros.controller_msgs')
    controller_msg_module = types.ModuleType('mros.controller_msgs.msg')
    std_module = types.ModuleType('mros.std_msgs')
    std_msg_module = types.ModuleType('mros.std_msgs.msg')
    teleop_module = types.ModuleType('mros.teleop_msgs')
    teleop_msg_module = types.ModuleType('mros.teleop_msgs.msg')
    controller_msg_module.JointCmd = JointCmd
    std_msg_module.Float32Array = Float32Array
    teleop_msg_module.TeleopMsg = TeleopMsg
    teleop_msg_module.KeyPoint = KeyPoint
    monkeypatch.setitem(sys.modules, 'mros', mros_module)
    monkeypatch.setitem(sys.modules, 'mros.controller_msgs', controller_module)
    monkeypatch.setitem(
        sys.modules, 'mros.controller_msgs.msg', controller_msg_module)
    monkeypatch.setitem(sys.modules, 'mros.std_msgs', std_module)
    monkeypatch.setitem(sys.modules, 'mros.std_msgs.msg', std_msg_module)
    monkeypatch.setitem(sys.modules, 'mros.teleop_msgs', teleop_module)
    monkeypatch.setitem(sys.modules, 'mros.teleop_msgs.msg', teleop_msg_module)

    class Publisher:

        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    operator = MrosOliOperator.__new__(MrosOliOperator)
    operator._accum_base_pos = np.array([0.0, 0.0, 0.9])
    operator._accum_base_yaw = 0.0
    operator.last_finger_cmd = np.zeros(14, dtype=np.float32)
    operator.teleop_wbt_publisher = Publisher()
    operator.finger_publisher = Publisher()

    first = np.zeros(42, dtype=np.float64)
    first[31:34] = [1.0, 0.0, 0.7]
    first[34:40] = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    first[40] = 1.0
    second = np.zeros(42, dtype=np.float64)
    second[31:34] = [1.0, 0.0, 0.8]
    second[34:40] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    operator.send_action(first)
    operator.send_action(second)

    first_msg, second_msg = operator.teleop_wbt_publisher.messages
    first_anchor = first_msg.anchors[0]
    second_anchor = second_msg.anchors[0]
    assert first_msg.header.frame_id == 'world'
    assert first_msg.joint_cmd.names is not None
    assert len(first_msg.joint_cmd.q) == 31
    assert first_anchor.name == 'base_link'
    assert np.allclose(
        [first_anchor.pose.position.x, first_anchor.pose.position.y,
         first_anchor.pose.position.z],
        [1.0, 0.0, 0.7])
    assert np.allclose(
        [second_anchor.pose.position.x, second_anchor.pose.position.y,
         second_anchor.pose.position.z],
        [1.0, 1.0, 0.8],
        atol=1e-6)
    assert operator.finger_publisher.messages[0].data[12:14] == [2.0, 2.0]


def test_wam_config_uses_standard_mros_teleop_topics():
    import os

    from mmengine import Config
    path = os.path.join(
        'configs', 'wam', 'wam_hud04_basket_t5_full_finetune.py')
    cfg = Config.fromfile(path)
    assert cfg.inference.operator.type == 'MrosOliOperator'
    assert cfg.inference.operator.teleop_wbt_topic == '/teleop_cmd_WBT'
    assert cfg.inference.operator.finger_cmd_topic == '/brainco1/hand/cmd_vla'
