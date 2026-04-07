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
import torch

from fluxvla.engines.utils.root import OPERATORS


def _rotation_6d_to_matrix(d6):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization (Zhou et al., CVPR 2019).
    Identical to pytorch3d's rotation_6d_to_matrix implementation
    used in lerobot's Teleop02.

    Args:
        d6 (torch.Tensor): (..., 6) tensor of 6D rotations.

    Returns:
        torch.Tensor: (..., 3, 3) rotation matrices.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@OPERATORS.register_module()
class Teleop02Operator:
    """Teleop02 humanoid robot operator using mros middleware.

    Handles communication with the Teleop02 (HUD04) loco-mani robot
    including head camera, joint states, finger states, teleop commands,
    and base velocity via mros topics.

    The robot has 31 joint positions, 2 hand closed flags (state = 33d),
    and 41-dim actions (4 body part poses as xyz+rot6d, base commands,
    and hand closed flags).
    """

    def __init__(
            self,
            head_rgb_topic='/head/color/image_raw/compressed',
            joint_state_topic='/joint/state',
            finger_state_topic='/brainco1/hand/state',
            finger_cmd_topic='/brainco1/hand/cmd',
            teleop_cmd_topic='/teleop_cmd',
            cmd_vel_topic='/sdk_cmd_vel'):
        """Initialize Teleop02Operator with mros topics.

        Args:
            head_rgb_topic (str): Topic for head camera compressed image.
            joint_state_topic (str): Topic for joint state feedback.
            finger_state_topic (str): Topic for finger state feedback.
            finger_cmd_topic (str): Topic for finger commands.
            teleop_cmd_topic (str): Topic for teleop pose commands.
            cmd_vel_topic (str): Topic for base velocity commands.
        """
        self.head_rgb_topic = head_rgb_topic
        self.joint_state_topic = joint_state_topic
        self.finger_state_topic = finger_state_topic
        self.finger_cmd_topic = finger_cmd_topic
        self.teleop_cmd_topic = teleop_cmd_topic
        self.cmd_vel_topic = cmd_vel_topic

        self.last_finger_state = np.zeros(12, dtype=np.float32)
        self.last_finger_cmd = np.zeros(12, dtype=np.float32)

        self._init_mros()

    def _init_mros(self):
        """Initialize mros node, subscribers, and publishers."""
        import mros
        from mros.controller_msgs.msg import JointState
        from mros.geometry_msgs.msg import Twist
        from mros.sensor_msgs.msg import CompressedImage
        from mros.std_msgs.msg import Float32Array
        from mros.teleop_msgs.msg import TeleopMsg

        mros.init('FluxVLATeleop02Node')

        # Subscribers
        self.color_subscriber = mros.subscribe(
            self.head_rgb_topic, CompressedImage, None)
        self.joint_state_subscriber = mros.subscribe(
            self.joint_state_topic, JointState, None)
        self.finger_state_subscriber = mros.subscribe(
            self.finger_state_topic, Float32Array, None)

        # Publishers
        self.teleop_cmd_publisher = mros.advertise(
            self.teleop_cmd_topic, TeleopMsg, None)
        self.cmd_vel_publisher = mros.advertise(
            self.cmd_vel_topic, Twist, None)
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
                (head_img_bgr, joint_state_33d). If failed, returns False.
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
        left_cmd_avg = float(np.mean(self.last_finger_cmd[0::2]))
        right_cmd_avg = float(np.mean(self.last_finger_cmd[1::2]))
        left_hand_closed = 1.0 if left_cmd_avg > 20 else 0.0
        right_hand_closed = 1.0 if right_cmd_avg > 20 else 0.0

        # Combine into 33-dim state
        state = np.concatenate([
            joint_state,
            np.array([left_hand_closed, right_hand_closed],
                     dtype=np.float32)
        ])

        return (head_img, state)

    def _rot6d_to_xyzw(self, rot6d):
        """Convert 6D rotation representation to quaternion [qx,qy,qz,qw].

        Args:
            rot6d (np.ndarray): (6,) array of 6D rotation.

        Returns:
            np.ndarray: (4,) quaternion in [qx, qy, qz, qw] order.
        """
        from scipy.spatial.transform import Rotation

        rot6d_tensor = torch.from_numpy(
            np.asarray(rot6d, dtype=np.float32)).unsqueeze(0)
        mat = _rotation_6d_to_matrix(rot6d_tensor).squeeze(0).numpy()
        quat_xyzw = Rotation.from_matrix(mat).as_quat()
        return quat_xyzw

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
        """Send a 41-dim action to the robot.

        Action layout:
            [0:3]   left_hand xyz
            [3:9]   left_hand rot6d
            [9:12]  right_hand xyz
            [12:18] right_hand rot6d
            [18:21] head xyz
            [21:27] head rot6d
            [27:30] torso xyz
            [30:36] torso rot6d
            [36]    base_cmd_z
            [37]    base_vel_discrete_x
            [38]    base_ang_vel_discrete_z
            [39]    left_hand_closed
            [40]    right_hand_closed

        Args:
            action (np.ndarray): 41-dim action vector.
        """
        from mros.geometry_msgs.msg import Twist
        from mros.std_msgs.msg import Float32Array
        from mros.teleop_msgs.msg import TeleopMsg

        action = np.asarray(action, dtype=np.float64)

        # Parse poses
        left_hand_pos = action[0:3]
        left_hand_quat = self._rot6d_to_xyzw(action[3:9])

        right_hand_pos = action[9:12]
        right_hand_quat = self._rot6d_to_xyzw(action[12:18])

        head_pos = action[18:21]
        head_quat = self._rot6d_to_xyzw(action[21:27])

        torso_pos = action[27:30]
        torso_quat = self._rot6d_to_xyzw(action[30:36])

        base_cmd_z = float(action[36])
        base_vel_discrete_x = float(action[37])
        base_ang_vel_discrete_z = float(action[38])

        # Construct TeleopMsg
        teleop_msg = TeleopMsg()
        teleop_msg.header.frame_id = 'pico_headset'
        teleop_msg.world.orientation.w = 1.0

        teleop_msg.anchors = [
            self._make_keypoint(
                'left_hand', left_hand_pos, left_hand_quat),
            self._make_keypoint(
                'right_hand', right_hand_pos, right_hand_quat),
            self._make_keypoint('head', head_pos, head_quat),
            self._make_keypoint('torso', torso_pos, torso_quat),
            self._make_keypoint('base', (0.0, 0.0, base_cmd_z)),
        ]
        self.teleop_cmd_publisher.publish(teleop_msg)

        # Send base velocity
        twist_msg = Twist()
        twist_msg.linear.x = base_vel_discrete_x
        twist_msg.angular.z = base_ang_vel_discrete_z
        self.cmd_vel_publisher.publish(twist_msg)

        # Send finger commands
        left_closed = float(action[39])
        right_closed = float(action[40])
        left_val = 100.0 if left_closed >= 0.5 else 0.0
        right_val = 100.0 if right_closed >= 0.5 else 0.0
        finger_cmd = [0.0] * 12
        for i in range(0, 12, 2):
            finger_cmd[i] = left_val
        for i in range(1, 12, 2):
            finger_cmd[i] = right_val
        self.last_finger_cmd = np.array(finger_cmd, dtype=np.float32)

        finger_msg = Float32Array()
        finger_msg.data = finger_cmd
        self.finger_publisher.publish(finger_msg)
