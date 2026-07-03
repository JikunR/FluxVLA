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

import json
from typing import Dict, List, Optional

import numpy as np
import torch

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.eval_utils import quat2axisangle
from fluxvla.engines.utils.robot_utils import (invert_gripper_action,
                                               normalize_gripper_action)


def _axisangle_to_rotmat(axisangle: np.ndarray) -> np.ndarray:
    axisangle = np.asarray(axisangle, dtype=np.float32)
    angle = np.linalg.norm(axisangle, axis=-1, keepdims=True)
    axis = axisangle / np.maximum(angle, 1e-8)
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    k_mat = np.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        axis=-1,
    ).reshape(axisangle.shape[:-1] + (3, 3))
    eye = np.eye(3, dtype=np.float32)
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    return eye + sin * k_mat + (1.0 - cos) * np.matmul(k_mat, k_mat)


def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / np.maximum(
        np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    return np.asarray(
        rotmat,
        dtype=np.float32)[..., :2, :].reshape(rotmat.shape[:-2] + (6, ))


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def _rotmat_to_axisangle(rotmat: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat, dtype=np.float32)
    cos = (np.trace(rotmat, axis1=-2, axis2=-1) - 1.0) * 0.5
    angle = np.arccos(np.clip(cos, -1.0, 1.0))
    axis = np.stack(
        (
            rotmat[..., 2, 1] - rotmat[..., 1, 2],
            rotmat[..., 0, 2] - rotmat[..., 2, 0],
            rotmat[..., 1, 0] - rotmat[..., 0, 1],
        ),
        axis=-1,
    )
    axis = axis / np.maximum(2.0 * np.sin(angle)[..., None], 1e-8)
    return axis * angle[..., None]


def _libero_state_to_rot6d(values: np.ndarray,
                           include_gripper: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] < 6:
        raise ValueError('LIBERO pose conversion expects xyz+axis-angle, got '
                         f'shape {values.shape}.')
    pos = values[..., :3]
    rot6d = _rotmat_to_rot6d(_axisangle_to_rotmat(values[..., 3:6]))
    parts = [pos, rot6d]
    if include_gripper:
        gripper = values[..., 6:]
        if gripper.shape[-1] == 0:
            gripper = np.zeros(values.shape[:-1] + (1, ), dtype=np.float32)
        else:
            gripper = gripper[..., :1]
        parts.append(gripper.astype(np.float32))
    return np.concatenate(parts, axis=-1).astype(np.float32)


@TRANSFORMS.register_module()
class Normalize:
    """Normalize the data using provided statistics.
    This transform normalizes the data by subtracting
    the mean and dividing by the standard deviation.
    Supports different normalization types: 'mean_std',
        'quantile', or 'min_max'.

    Args:
        norm_stats (List): List of normalization statistics,
            where each element is a dictionary  containing
            'mean', 'std', 'q01', 'q99', 'min', and 'max' for each feature.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
    """

    def __init__(self,
                 norm_stats: List,
                 norm_type: str = 'mean_std',
                 strict: bool = False):
        self.norm_stats = norm_stats
        self.norm_type = norm_type
        self.strict = strict

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            return data
        for key, value in data.items():
            if key in self.norm_stats.keys():
                if self.norm_type == 'quantile':
                    data[key] = self._normalize_quantile(
                        value, self.norm_stats[key])
                elif self.norm_type == 'min_max':
                    data[key] = self._normalize_min_max(
                        value, self.norm_stats[key])
                else:  # norm_type == 'mean_std'
                    data[key] = self._normalize(value, self.norm_stats[key])
        return data

    def _normalize(self, x, stats: Dict):
        return (x - torch.tensor(stats['mean'])) / (
            torch.tensor(stats['std']) + 1e-6)

    def _normalize_quantile(self, x, stats: torch.tensor):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        return (x - torch.tensor(stats['q01'])) / (torch.tensor(
            stats['q99']) - torch.tensor(stats['q01']) + 1e-6) * 2.0 - 1.0

    def _normalize_min_max(self, x, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        return (x - torch.tensor(stats['min'])) / (torch.tensor(
            stats['max']) - torch.tensor(stats['min']) + 1e-6) * 2.0 - 1.0


@TRANSFORMS.register_module()
class DenormalizeLiberoAction:
    """Denormalize the data using provided statistics.
    This transform reverses the normalization done using
    mean/std, quantiles, or min_max.

    Args:
        norm_stats (str or Dict): Normalization statistics,
            which can be a JSON string or a dictionary
            containing 'mean', 'std', 'q01', 'q99', 'min', and 'max' for each
            feature. If a string, it should be a JSON representation
            of the normalization statistics.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
        denorm_action (bool): If True, denormalize the action.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
        normalize_gripper_action (bool): If True, normalize
            the gripper action. This is useful for tasks
            where the gripper action is part of the state
            and needs to be denormalized.
        invert_gripper_action (bool): If True, invert the
            gripper action. This is useful for tasks where
            the gripper action is represented in a way that
            requires inversion (e.g., opening vs. closing).
            This is useful for tasks where the gripper action
            is represented in a way that requires inversion
            (e.g., opening vs. closing).
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = None,
                 norm_type: str = 'mean_std',
                 strict: bool = False,
                 denorm_action: bool = True,
                 normalize_gripper_action: bool = True,
                 invert_gripper_action: bool = True,
                 action_norm_mask: List[bool] = None):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.strict = strict
        self.denorm_action = denorm_action
        self.normalize_gripper_action = normalize_gripper_action
        self.invert_gripper_action = invert_gripper_action
        self.action_norm_mask = action_norm_mask

    def __call__(self, data: Dict) -> Dict:
        """Denormalize the data using the provided statistics.
        This method denormalizes the action in the data
        if the `denorm_action` flag is set to True.
        It retrieves the normalization statistics based on
        the `task_suite_name` from the data and applies
        the appropriate denormalization method.  # noqa: E501

        Args:
            data (Dict): The data to be denormalized, which should
                contain keys that match the keys in `norm_stats`.
        """
        if self.norm_stats is not None and self.denorm_action:
            norm_stats_key = data.get('norm_stats_key')
            norm_stats = self.norm_stats[norm_stats_key]
            action = data.get('action', None)
            assert action is not None, \
                f'Action is not found in the data: {data.keys()}'
            if self.norm_type == 'quantile':
                action = self._denormalize_quantile(action,
                                                    norm_stats['action'])
            elif self.norm_type == 'min_max':
                action = self._denormalize_min_max(action,
                                                   norm_stats['action'])
            else:  # norm_type == 'mean_std'
                action = self._denormalize(action, norm_stats['action'])
        if self.normalize_gripper_action:
            action = normalize_gripper_action(action, binarize=True)
        if self.invert_gripper_action:
            action = invert_gripper_action(action)

        if self.action_dim is not None:
            action = action[:self.action_dim]
        return action

    def _denormalize(self, normalized_action: np.ndarray, stats: Dict):
        assert 'mean' in stats and stats['mean'] is not None
        assert 'std' in stats and stats['std'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]

        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(stats['mean'], dtype=bool)
        action = np.where(
            mask,
            normalized_action * np.array(stats['std']) +
            np.array(stats['mean']), normalized_action)
        return action

    def _denormalize_quantile(self, normalized_action: np.ndarray,
                              stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.action_norm_mask is not None:
            mask = np.array(self.action_norm_mask)
        else:
            mask = np.ones_like(stats['q01'], dtype=bool)  # noqa: E501
        action_high = np.array(stats['q99'])
        action_low = np.array(stats['q01'])
        mask = np.array(mask)
        action = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) +
            action_low,  # noqa: E501
            normalized_action,
        )
        return action

    def _denormalize_min_max(self, normalized_action: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.action_norm_mask is not None:
            mask = np.array(self.action_norm_mask)
        else:
            mask = np.ones_like(stats['min'], dtype=bool)
        action_high = np.array(stats['max'])
        action_low = np.array(stats['min'])
        mask = np.array(mask)
        action = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) +
            action_low,
            normalized_action,
        )
        return action


@TRANSFORMS.register_module()
class DenormalizePrivateAction(DenormalizeLiberoAction):
    """Denormalize the data using provided statistics.
    This transform reverses the normalization done using
    mean/std, quantiles, or min_max.

    Args:
        norm_stats (str or Dict): Normalization statistics,
            which can be a JSON string or a dictionary
            containing 'mean', 'std', 'q01', 'q99', 'min', and 'max' for each
            feature. If a string, it should be a JSON representation
            of the normalization statistics.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
        denorm_action (bool): If True, denormalize the action.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
        normalize_gripper_action (bool): If True, normalize
            the gripper action. This is useful for tasks
            where the gripper action is part of the state
            and needs to be denormalized.
        invert_gripper_action (bool): If True, invert the
            gripper action. This is useful for tasks where
            the gripper action is represented in a way that
            requires inversion (e.g., opening vs. closing).
            This is useful for tasks where the gripper action
            is represented in a way that requires inversion
            (e.g., opening vs. closing).
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = None,
                 norm_type: str = 'mean_std',
                 strict: bool = False,
                 denorm_action: bool = True,
                 normalize_gripper_action: bool = True,
                 invert_gripper_action: bool = True,
                 action_norm_mask: List[bool] = None):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.strict = strict
        self.denorm_action = denorm_action
        self.action_norm_mask = action_norm_mask

    def __call__(self, data: Dict) -> Dict:
        """Denormalize the data using the provided statistics.
        This method denormalizes the action in the data
        if the `denorm_action` flag is set to True.
        It retrieves the normalization statistics based on
        the `task_suite_name` from the data and applies
        the appropriate denormalization method.  # noqa: E501

        Args:
            data (Dict): The data to be denormalized, which should
                contain keys that match the keys in `norm_stats`.
        """
        if self.norm_stats is not None and self.denorm_action:
            norm_stats = self.norm_stats['private']
            action = data.get('action', None)[0]
            assert action is not None, \
                f'Action is not found in the data: {data.keys()}'
            if self.norm_type == 'quantile':
                action = self._denormalize_quantile(action,
                                                    norm_stats['action'])
            elif self.norm_type == 'min_max':
                action = self._denormalize_min_max(action,
                                                   norm_stats['action'])
            else:  # norm_type == 'mean_std'
                action = self._denormalize(action, norm_stats['action'])
        return action


@TRANSFORMS.register_module()
class LiberoDeltaActionToGoalRot6D:
    """Convert LIBERO OSC deltas to absolute controller goals.

    The input ``states`` keeps the observed absolute proprio pose, while
    ``actions`` is the recorded 7D normalized OSC_POSE command. The output
    uses absolute controller goals for pose and preserves the recorded gripper
    command in the last channel: ``xyz(3) + rot6d(6) + gripper(1)``.
    """

    def __init__(self,
                 state_key: str = 'states',
                 action_key: str = 'actions',
                 state_window_key: str = 'state_window',
                 pos_scale: float = 0.05,
                 rot_scale: float = 0.5) -> None:
        self.state_key = state_key
        self.action_key = action_key
        self.state_window_key = state_window_key
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)

    def __call__(self, data: Dict) -> Dict:
        state = np.asarray(data[self.state_key], dtype=np.float32)
        if self.state_window_key not in data:
            raise KeyError(
                f'{type(self).__name__} requires `{self.state_window_key}` '
                'aligned with the action window.')
        state_window = np.asarray(
            data[self.state_window_key], dtype=np.float32)
        actions = np.asarray(data[self.action_key], dtype=np.float32)
        if state.shape[-1] < 6:
            raise ValueError('LIBERO state must contain xyz+axis-angle, got '
                             f'shape {state.shape}.')
        if state_window.shape[:-1] != actions.shape[:-1]:
            raise ValueError(
                f'LIBERO state_window shape {state_window.shape} must align '
                f'with actions shape {actions.shape}.')
        if state_window.shape[-1] < 6:
            raise ValueError(
                'LIBERO state_window must contain xyz+axis-angle, got '
                f'shape {state_window.shape}.')
        if actions.shape[-1] < 7:
            raise ValueError(
                'LIBERO action must contain 7D OSC_POSE command, got '
                f'shape {actions.shape}.')

        current_pos = state_window[..., :3]
        current_rot = _axisangle_to_rotmat(state_window[..., 3:6])
        target_pos = current_pos + actions[..., :3] * self.pos_scale
        delta_rot = _axisangle_to_rotmat(actions[..., 3:6] * self.rot_scale)
        target_rot = np.matmul(delta_rot, current_rot)
        target = np.concatenate(
            (
                target_pos,
                _rotmat_to_rot6d(target_rot),
                actions[..., 6:7],
            ),
            axis=-1,
        ).astype(np.float32)

        data[self.state_key] = _libero_state_to_rot6d(state)
        data[self.action_key] = target
        return data


@TRANSFORMS.register_module()
class LiberoProprioRot6DFromInputs:
    """Build eval-time LIBERO state in xyz+rot6d+gripper layout."""

    def __init__(self,
                 state_dim: int = None,
                 pos_key: str = 'robot0_eef_pos',
                 quat_key: str = 'robot0_eef_quat',
                 gripper_key: str = 'robot0_gripper_qpos',
                 out_key: str = 'states') -> None:
        self.state_dim = state_dim
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key
        self.out_key = out_key

    def __call__(self, data: Dict) -> Dict:
        pos = np.asarray(data[self.pos_key], dtype=np.float32)
        quat = np.asarray(data[self.quat_key], dtype=np.float32)
        gripper = np.asarray(
            data[self.gripper_key], dtype=np.float32).reshape(-1)[:1]
        state = np.concatenate(
            (pos, _rotmat_to_rot6d(_quat_xyzw_to_rotmat(quat)), gripper),
            axis=-1,
        ).astype(np.float32)
        if self.state_dim is not None:
            padded = np.zeros((self.state_dim, ), dtype=np.float32)
            padded[:state.shape[0]] = state
            state = padded
        out = dict(data)
        out[self.out_key] = state
        return out


@TRANSFORMS.register_module()
class DenormalizeLiberoRot6DAction:
    """Convert absolute xyz+rot6d+gripper target to LIBERO OSC_POSE action."""

    def __init__(self,
                 pos_scale: float = 0.05,
                 rot_scale: float = 0.5,
                 action_dim: int = 10,
                 env_action_dim: int = 7,
                 pos_key: str = 'robot0_eef_pos',
                 quat_key: str = 'robot0_eef_quat',
                 gripper_key: str = 'robot0_gripper_qpos',
                 norm_stats=None) -> None:
        del norm_stats
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.action_dim = int(action_dim)
        self.env_action_dim = int(env_action_dim)
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key

    def __call__(self, data: Dict) -> np.ndarray:
        action = np.asarray(data['action'], dtype=np.float32)
        if action.shape[-1] < self.action_dim:
            raise ValueError(
                f'Expected rot6d action dim >= {self.action_dim}, got '
                f'{action.shape[-1]}.')
        target = action[..., :self.action_dim]
        current_pos = self._required_array(data, self.pos_key)
        current_quat = self._required_array(data, self.quat_key)
        self._required_array(data, self.gripper_key)

        delta_pos = (target[:3] - current_pos) / self.pos_scale
        current_rot = _quat_xyzw_to_rotmat(current_quat)
        target_rot = _rot6d_to_rotmat(target[3:9])
        delta_rot = np.matmul(target_rot, np.swapaxes(current_rot, -1, -2))
        delta_axisangle = _rotmat_to_axisangle(delta_rot) / self.rot_scale

        gripper_action = np.array([target[9]], dtype=np.float32)
        gripper_action = normalize_gripper_action(
            gripper_action, binarize=True)
        gripper_action = invert_gripper_action(gripper_action)
        env_action = np.concatenate(
            (delta_pos, delta_axisangle, gripper_action),
            axis=-1,
        ).astype(np.float32)
        return np.clip(env_action[:self.env_action_dim], -1.0, 1.0)

    @staticmethod
    def _required_array(data: Dict, key: str) -> np.ndarray:
        value = data.get(key)
        if value is None:
            raise KeyError(
                f'DenormalizeLiberoRot6DAction requires `{key}` from the '
                'current LIBERO observation.')
        return np.asarray(value, dtype=np.float32)


@TRANSFORMS.register_module()
class NormalizeStatesAndActions:
    """Normalize states and actions in the data.
    This transform normalizes the state and action
    dimensions in the data to match the specified
    action dimension. It pads the state and action
    dimensions to the specified action dimension.

    Args:
        action_dim (int): The dimension to which the state
            and action should be normalized.
        pad_value (float): The value to use for padding.
            Defaults to 0.0.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', 'min_max', or 'none'.
            Defaults to 'mean_std'.
        state_norm_type (str): Optional normalization type for states.
            Defaults to `norm_type`.
        action_norm_type (str): Optional normalization type for actions.
            Defaults to `norm_type`.
        clip_norm (bool): Whether to clip min_max/quantile normalized values
            to [-1, 1]. Defaults to False.
        normalize_states (bool): Whether to normalize states before optional
            padding/truncation. Defaults to True.
        state_key (str | None): The key in the data dictionary
            that contains the state information.
        action_key (str | None): The key in the data dictionary
            that contains the action information. If None, actions are skipped.
    """

    def __init__(self,
                 state_key: Optional[str],
                 action_key: Optional[str],
                 action_dim: int = None,
                 state_dim: int = None,
                 norm_type: str = 'mean_std',
                 state_norm_type: str = None,
                 action_norm_type: str = None,
                 pad_value: float = 0.0,
                 action_norm_mask: List[bool] = None,
                 clip_norm: bool = False,
                 normalize_states: bool = True,
                 *args,
                 **kwargs):
        self.state_key = state_key
        self.action_key = action_key
        self.norm_type = norm_type
        self.state_norm_type = state_norm_type or norm_type
        self.action_norm_type = action_norm_type or norm_type
        self.pad_value = pad_value
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.clip_norm = clip_norm
        self.normalize_states = normalize_states
        if action_norm_mask is not None:
            assert len(action_norm_mask) == action_dim, \
                f'Action norm mask must be of length {action_dim}'
            self.action_norm_mask = action_norm_mask
        else:
            self.action_norm_mask = None

    def __call__(self, data: Dict) -> Dict:
        states = np.asarray(data['states'], dtype=np.float32)
        actions = None
        if self.action_key is not None and 'actions' in data:
            actions = np.asarray(data['actions'], dtype=np.float32)

        needs_state_stats = (
            self.normalize_states and self.state_norm_type != 'none')
        needs_action_stats = (
            actions is not None and self.action_norm_type != 'none')
        if needs_state_stats or needs_action_stats:
            assert 'stats' in data, "Input data must contain 'stats' key"

        if needs_state_stats:
            state_stats = data['stats'][self.state_key]
            states = self._normalize_by_type(states, state_stats,
                                             self.state_norm_type)
        data['states'] = states

        if actions is not None:
            action_stats = None
            if needs_action_stats:
                action_stats = data['stats'][self.action_key]
            actions = self._normalize_by_type(actions, action_stats,
                                              self.action_norm_type,
                                              self.action_norm_mask)
            data['actions'] = actions
        if self.state_dim is not None:
            data['states'] = self._pad_or_truncate_last_dim(
                states, self.state_dim)
        if self.action_dim is not None and actions is not None:
            data['actions'] = self._pad_or_truncate_last_dim(
                actions, self.action_dim)
        return data

    def _pad_or_truncate_last_dim(self, values: np.ndarray,
                                  target_dim: int) -> np.ndarray:
        current_dim = values.shape[-1]
        if current_dim >= target_dim:
            return values[..., :target_dim]
        padded_shape = (*values.shape[:-1], target_dim)
        padded = np.full(padded_shape, self.pad_value, dtype=values.dtype)
        padded[..., :current_dim] = values
        return padded

    def _normalize_by_type(self,
                           x,
                           stats: Dict,
                           norm_type: str,
                           norm_mask: List[bool] = None):
        if norm_type == 'none':
            return x
        if norm_type == 'quantile':
            return self._normalize_quantile(x, stats, norm_mask)
        if norm_type == 'min_max':
            return self._normalize_min_max(x, stats, norm_mask)
        return self._normalize(x, stats, norm_mask)

    def _normalize(self, x, stats: Dict, norm_mask: List[bool] = None):
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        return np.where(norm_mask, (x - np.array(stats['mean'])) /
                        (np.array(stats['std']) + 1e-6), x)

    def _normalize_quantile(self,
                            x,
                            stats: torch.tensor,
                            norm_mask: List[bool] = None):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        normalized = (
            (x - np.array(stats['q01'])) /
            (np.array(stats['q99']) - np.array(stats['q01']) + 1e-6) * 2.0 -
            1.0)
        if self.clip_norm:
            normalized = np.clip(normalized, -1, 1)
        return np.where(norm_mask, normalized, x)

    def _normalize_min_max(self, x, stats: Dict, norm_mask: List[bool] = None):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        normalized = (
            (x - np.array(stats['min'])) /
            (np.array(stats['max']) - np.array(stats['min']) + 1e-6) * 2.0 -
            1.0)
        if self.clip_norm:
            normalized = np.clip(normalized, -1, 1)
        return np.where(norm_mask, normalized, x)


@TRANSFORMS.register_module()
class LiberoProprioFromInputs:
    """Build and normalize Libero proprio state from inputs.

    Reads `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`,
    converts quaternion to axis-angle, concatenates into a
    state vector, and normalizes using `norm_stats[task_suite_name +
    '_no_noops']['proprio']`.

    Expects `task_suite_name` to be present in the input dict.

    Args:
        norm_stats (str | Dict): Path to JSON or dict of normalization stats.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'quantile'.
        pos_key (str): Key for end-effector position.
        quat_key (str): Key for end-effector quaternion.
        gripper_key (str): Key for gripper position.
        out_key (str): Output key for normalized state (default 'states').
    """

    def __init__(self,
                 norm_type: str = 'quantile',
                 state_dim: int = None,
                 pos_key: str = 'robot0_eef_pos',
                 quat_key: str = 'robot0_eef_quat',
                 gripper_key: str = 'robot0_gripper_qpos',
                 stat_key: str = 'proprio',
                 out_key: str = 'states') -> None:
        self.norm_type = norm_type
        self.state_dim = state_dim
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key
        self.out_key = out_key
        self.stat_key = stat_key

    def __call__(self, data: Dict) -> Dict:
        assert self.pos_key in data and self.quat_key in \
            data and self.gripper_key in data, \
            f'Missing proprio keys in data: {self.pos_key}, {self.quat_key}, {self.gripper_key}'  # noqa: E501
        robot0_eef_pos = np.asarray(data[self.pos_key])
        robot0_eef_quat = np.asarray(data[self.quat_key])
        robot0_gripper_qpos = np.asarray(data[self.gripper_key])

        state = np.concatenate((
            robot0_eef_pos,
            quat2axisangle(robot0_eef_quat),
            robot0_gripper_qpos,
        ))

        stats = data['norm_stats'][self.stat_key]
        if self.norm_type == 'quantile':
            state = self._normalize_quantile(state, stats)
        elif self.norm_type == 'min_max':
            state = self._normalize_min_max(state, stats)
        else:  # norm_type == 'mean_std'
            state = self._normalize(state, stats)

        out = dict(data)
        if self.state_dim is not None:
            out[self.out_key] = np.zeros((self.state_dim))
            out[self.out_key][:state.shape[0]] = state
        else:
            out[self.out_key] = state
        return out

    def _normalize(self, normalized_states: np.ndarray, stats: Dict):
        assert 'mean' in stats and stats['mean'] is not None
        assert 'std' in stats and stats['std'] is not None
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(stats['mean'], dtype=bool)
        # Keep eval-time mean/std normalization consistent with training:
        # (x - mean) / (std + eps), without clipping.
        states = np.where(
            mask,
            (normalized_states - np.array(stats['mean'])) /
            (np.array(stats['std']) + 1e-6),
            normalized_states,
        )
        return states

    def _normalize_quantile(self, normalized_states: np.ndarray, stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None
        state_high = np.array(stats['q99'])
        state_low = np.array(stats['q01'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states

    def _normalize_min_max(self, normalized_states: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        state_high = np.array(stats['max'])
        state_low = np.array(stats['min'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states
