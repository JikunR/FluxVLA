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

import cv2
import numpy as np

from fluxvla.engines.utils.root import OPERATORS

# 31 joint names in teleop_cmd_WBT order
STATE_JOINT_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'head_yaw_joint',
    'head_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_yaw_joint',
    'left_wrist_pitch_joint',
    'left_wrist_roll_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_yaw_joint',
    'right_wrist_pitch_joint',
    'right_wrist_roll_joint',
]

# Default joint stiffness / damping
DEFAULT_KP = 140.0
DEFAULT_KD = 4.0


def _rot6d_to_quat_xyzw(rot6d):
    """Convert 6D rotation (Zhou et al.) to quaternion [qx,qy,qz,qw].

    Uses numpy Gram-Schmidt, consistent with the data collection
    pipeline in create_wbt_lerobot_dataset.py.

    Args:
        rot6d (np.ndarray): (6,) array of 6D rotation.

    Returns:
        np.ndarray: (4,) quaternion in [qx, qy, qz, qw] order.
    """
    from scipy.spatial.transform import Rotation

    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[:3], rot6d[3:6]

    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)

    mat = np.stack([b1, b2, b3], axis=-2)  # (3, 3)
    return Rotation.from_matrix(mat).as_quat()  # xyzw


@OPERATORS.register_module()
class Teleop02WbtOperator:
    """Teleop02 WBT (whole-body tracking) operator using mros middleware.

    Sends joint-level commands + base_link pose via /teleop_cmd_WBT,
    matching the message format recorded in the WBT water task bags.

    The robot has 31 joint positions, 2 hand closed flags (state = 33d),
    and 42-dim actions:
        [0:31]  joint position commands (q)
        [31:34] base_link position (xyz)
        [34:40] base_link rotation (rot6d)
        [40]    left_hand_closed
        [41]    right_hand_closed
    """

    def __init__(
            self,
            head_rgb_topic='/head/color/image_raw/compressed',
            joint_state_topic='/joint/state',
            finger_state_topic='/brainco1/hand/state',
            finger_cmd_topic='/brainco1/hand/cmd_vla',
            teleop_wbt_topic='/teleop_cmd_WBT',
            cmd_vel_topic='/sdk_cmd_vel_vla'):
        """Initialize Teleop02WbtOperator with mros topics.

        Args:
            head_rgb_topic (str): Topic for head camera compressed image.
            joint_state_topic (str): Topic for joint state feedback.
            finger_state_topic (str): Topic for finger state feedback.
            finger_cmd_topic (str): Topic for finger commands.
            teleop_wbt_topic (str): Topic for WBT teleop commands
                (TeleopMsg with joint_cmd + base_link anchor).
            cmd_vel_topic (str): Topic for base velocity commands.
        """
        self.head_rgb_topic = head_rgb_topic
        self.joint_state_topic = joint_state_topic
        self.finger_state_topic = finger_state_topic
        self.finger_cmd_topic = finger_cmd_topic
        self.teleop_wbt_topic = teleop_wbt_topic
        self.cmd_vel_topic = cmd_vel_topic

        self.last_finger_state = np.zeros(12, dtype=np.float32)
        self.last_finger_cmd = np.zeros(14, dtype=np.float32)

        self._init_mros()

    def _init_mros(self):
        """Initialize mros node, subscribers, and publishers."""
        import mros
        from mros.controller_msgs.msg import JointState
        from mros.sensor_msgs.msg import CompressedImage
        from mros.std_msgs.msg import Float32Array
        from mros.teleop_msgs.msg import TeleopMsg

        mros.init('FluxVLATeleop02WbtNode')

        # Subscribers
        self.color_subscriber = mros.subscribe(
            self.head_rgb_topic, CompressedImage, None)
        self.joint_state_subscriber = mros.subscribe(
            self.joint_state_topic, JointState, None)
        self.finger_state_subscriber = mros.subscribe(
            self.finger_state_topic, Float32Array, None)

        # Publishers
        self.teleop_wbt_publisher = mros.advertise(
            self.teleop_wbt_topic, TeleopMsg, None)
        self.finger_publisher = mros.advertise(
            self.finger_cmd_topic, Float32Array, queue_size=10)

    def _compressed_msg_to_numpy(self, msg):
        """Decode CompressedImage (jpeg/png) to numpy BGR image.

        Args:
            msg: mros CompressedImage message.

        Returns:
            np.ndarray or None: BGR image array, or None on failure.
        """
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f'Failed to decode compressed image: {e}')
            return None

    def get_frame(self, timeout=0.5):
        """Get synchronized observation from all sensors.

        Reads head camera image, joint states, and finger states.
        Returns combined 33-dim state vector (31 joints + 2 hand_closed).

        Args:
            timeout (float): Maximum wait time in seconds for each
                sensor reading. Defaults to 0.5.

        Returns:
            tuple or False: If successful, returns
                (head_img_rgb, joint_state_33d). If failed, returns False.
        """
        # (1) Read head image
        head_rgb = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            head_rgb = self.color_subscriber.readMsgRT()
            if head_rgb is not None:
                break
            time.sleep(0.005)
        if head_rgb is None:
            return False

        head_img = self._compressed_msg_to_numpy(head_rgb)
        if head_img is None:
            return False
        # Convert BGR to RGB
        head_img = head_img[:, :, ::-1].copy()

        # (2) Read joint state
        joint_state_msg = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            joint_state_msg = self.joint_state_subscriber.readMsgRT()
            if joint_state_msg is not None:
                break
            time.sleep(0.005)
        if joint_state_msg is None:
            print('No joint state message received')
            return False

        joint_state = np.asarray(joint_state_msg.q, dtype=np.float32)
        if joint_state.size < 31:
            print(f'Joint state size {joint_state.size} < 31')
            return False
        joint_state = joint_state[:31]

        # (3) Read finger state
        finger_state_msg = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            finger_state_msg = self.finger_state_subscriber.readMsgRT()
            if finger_state_msg is not None:
                break
            time.sleep(0.005)

        if finger_state_msg is not None:
            finger_state = np.asarray(
                finger_state_msg.data, dtype=np.float32)
            if finger_state.size >= 12:
                self.last_finger_state = finger_state[:12].copy()

        # Compute hand closed flags from last finger cmd
        left_cmd_avg = float(np.mean(self.last_finger_cmd[0:12:2]))
        right_cmd_avg = float(np.mean(self.last_finger_cmd[1:12:2]))
        left_hand_closed = 1.0 if left_cmd_avg > 20 else 0.0
        right_hand_closed = 1.0 if right_cmd_avg > 20 else 0.0

        # Combine into 33-dim state
        state = np.concatenate([
            joint_state,
            np.array([left_hand_closed, right_hand_closed],
                     dtype=np.float32)
        ])

        return (head_img, state)

    def _make_keypoint(self, name, pos=(0.0, 0.0, 0.0),
                       quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
        """Construct a TeleopMsg KeyPoint.

        Args:
            name (str): Keypoint name.
            pos (tuple): (x, y, z) position.
            quat_xyzw (tuple): (qx, qy, qz, qw) quaternion.

        Returns:
            KeyPoint: Constructed keypoint message.
        """
        from mros.teleop_msgs.msg import KeyPoint

        kp = KeyPoint()
        kp.name = name
        kp.pose.position.x = float(pos[0])
        kp.pose.position.y = float(pos[1])
        kp.pose.position.z = float(pos[2])
        kp.pose.orientation.x = float(quat_xyzw[0])
        kp.pose.orientation.y = float(quat_xyzw[1])
        kp.pose.orientation.z = float(quat_xyzw[2])
        kp.pose.orientation.w = float(quat_xyzw[3])
        return kp

    def send_action(self, action):
        """Send a 42-dim WBT action to the robot.

        Action layout:
            [0:31]  joint position commands (q) -> joint_cmd
            [31:34] base_link position (xyz)    -> anchors[0]
            [34:40] base_link rotation (rot6d)  -> anchors[0]
            [40]    left_hand_closed
            [41]    right_hand_closed

        Publishes a TeleopMsg to /teleop_cmd_WBT containing:
            - joint_cmd: JointCmd with 31 joint position targets
            - anchors[0]: base_link KeyPoint (xyz + quaternion)
        Also publishes finger commands to the finger_cmd_topic.

        Args:
            action (np.ndarray): 42-dim action vector.
        """
        from mros.controller_msgs.msg import JointCmd
        from mros.std_msgs.msg import Float32Array
        from mros.teleop_msgs.msg import TeleopMsg

        action = np.asarray(action, dtype=np.float64)

        # Parse 42-dim action
        joint_cmd_q = action[0:31]
        base_pos = action[31:34]
        base_rot6d = action[34:40]
        left_closed = float(action[40])
        right_closed = float(action[41])

        # Convert base rotation from 6D to quaternion
        base_quat_xyzw = _rot6d_to_quat_xyzw(base_rot6d)

        # Construct TeleopMsg
        teleop_msg = TeleopMsg()
        teleop_msg.header.frame_id = 'world'
        teleop_msg.world.orientation.w = 1.0

        # Joint command
        jcmd = JointCmd()
        jcmd.names = list(STATE_JOINT_NAMES)
        jcmd.q = joint_cmd_q.astype(np.float32).tolist()
        jcmd.v = [0.0] * 31
        jcmd.tau = [0.0] * 31
        jcmd.kp = [DEFAULT_KP] * 31
        jcmd.kd = [DEFAULT_KD] * 31
        jcmd.mode = [0] * 31
        jcmd.na = 31
        teleop_msg.joint_cmd = jcmd

        # Base link anchor (anchors[0])
        teleop_msg.anchors = [
            self._make_keypoint('base_link', base_pos, base_quat_xyzw),
        ]
        self.teleop_wbt_publisher.publish(teleop_msg)

        # Send finger commands (14 dims: 12 finger + 2 force levels)
        left_val = 100.0 if left_closed >= 0.5 else 0.0
        right_val = 100.0 if right_closed >= 0.5 else 0.0
        finger_cmd = [0.0] * 14
        for i in range(0, 12, 2):
            finger_cmd[i] = left_val
        for i in range(1, 12, 2):
            finger_cmd[i] = right_val

        # Thumb aux always closed for better grasping
        finger_cmd[2] = 100.0
        finger_cmd[3] = 100.0

        # Last 2 dims: force level (use level 3)
        finger_cmd[12] = 3.0
        finger_cmd[13] = 3.0

        self.last_finger_cmd = np.array(finger_cmd, dtype=np.float32)
        finger_msg = Float32Array()
        finger_msg.data = finger_cmd
        self.finger_publisher.publish(finger_msg)
