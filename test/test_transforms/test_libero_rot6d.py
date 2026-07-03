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

import numpy as np

from fluxvla.transforms.normalize import (DenormalizeLiberoRot6DAction,
                                          LiberoDeltaActionToGoalRot6D,
                                          LiberoProprioRot6DFromInputs,
                                          NormalizeStatesAndActions)
from fluxvla.transforms.transform_cosmos3 import BuildCosmos3Sequence


def test_libero_rot6d_prepend_state_sequence():
    current = np.array(
        [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 0.04, -0.04],
        dtype=np.float32,
    )
    raw_actions = np.tile(
        np.array(
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
            dtype=np.float32,
        ),
        (16, 1),
    )
    data = dict(
        states=current,
        state_window=np.tile(current, (16, 1)),
        actions=raw_actions,
        embodiment_ids=np.array(5, dtype=np.int32),
    )

    data = LiberoDeltaActionToGoalRot6D()(data)
    current_rot6d = data['states'].copy()
    goal_rot6d = data['actions'][0].copy()
    assert current_rot6d.shape == (10, )
    assert goal_rot6d.shape == (10, )
    assert current_rot6d[-1] == np.float32(0.04)

    normalizer = NormalizeStatesAndActions(
        state_key='observation.state',
        action_key='observation.state',
        action_dim=64,
        state_dim=64,
        norm_type='none',
    )
    data = normalizer(data)

    sequence_builder = BuildCosmos3Sequence(
        raw_action_dim=10,
        mode='policy',
        frame_window_size=17,
        prepend_state_to_action=True,
        conditioning_fps=20.0,
    )
    data = sequence_builder(data)

    assert data['actions'].shape == (17, 64)
    np.testing.assert_allclose(data['actions'][0, :10], current_rot6d)
    np.testing.assert_allclose(data['actions'][1, :10], goal_rot6d)
    assert int(data['raw_action_dim']) == 10
    assert data['sequence_plan'].condition_frame_indexes_action == [0]

    restored_inputs = dict(
        action=goal_rot6d,
        robot0_eef_pos=current[:3],
        robot0_eef_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        robot0_gripper_qpos=current[6:],
    )
    restored = DenormalizeLiberoRot6DAction()(restored_inputs)
    expected_env_action = raw_actions[0].copy()
    expected_env_action[-1] = -1.0
    np.testing.assert_allclose(restored, expected_env_action, atol=1e-5)


def test_libero_rot6d_eval_action_conversion():
    obs = dict(
        robot0_eef_pos=np.array([0.10, 0.20, 0.30], dtype=np.float32),
        robot0_eef_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        robot0_gripper_qpos=np.array([0.04, -0.04], dtype=np.float32),
    )
    state = LiberoProprioRot6DFromInputs(state_dim=64)(obs)['states']
    assert state.shape == (64, )
    assert state[9] == np.float32(0.04)

    target = LiberoDeltaActionToGoalRot6D()({
        'states':
        np.array(
            [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 0.04, -0.04],
            dtype=np.float32,
        ),
        'state_window':
        np.array(
            [[0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 0.04, -0.04]],
            dtype=np.float32,
        ),
        'actions':
        np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]], dtype=np.float32),
    })['actions'][0]
    action = DenormalizeLiberoRot6DAction()({**obs, 'action': target})

    assert action.shape == (7, )
    np.testing.assert_allclose(
        action[:3], np.array([0.2, 0.0, 0.0]), rtol=1e-5)
    np.testing.assert_allclose(
        action[3:6], np.array([0.0, 0.0, 0.5]), atol=1e-5)
    assert action[-1] == -1.0
