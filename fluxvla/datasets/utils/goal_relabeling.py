"""
goal_relabeling.py

Provides goal relabeling logic for behavior cloning (BC) use-cases where
rewards and next_observations are not needed. Each function modifies the
trajectory's "task" field.
"""

from typing import Dict

import tensorflow as tf

from .data_utils import tree_merge


def uniform(traj: Dict) -> Dict:
    """
    Relabels each step in a trajectory using a uniformly sampled future goal.

    For each timestep i, this function samples a future timestep j ∈ [i+1, T)
    and uses the observation at j as the new goal for timestep i. The new
    goal is added to the `task` dict of the trajectory.

    Args:
        traj (Dict): A trajectory dictionary with "observation" and "task"
            keys.

    Returns:
        Dict: The trajectory with updated "task" dict containing relabeled
            goals.
    """
    traj_len = tf.shape(tf.nest.flatten(traj['observation'])[0])[0]

    # Generate random floats in [0, 1) for each timestep
    rand = tf.random.uniform([traj_len])

    # Lower and upper bounds for sampling
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)

    # Sample future indices uniformly from [i+1, traj_len)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # Avoid out-of-bounds due to float precision
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # Gather relabeled goal observations
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs),
                                 traj['observation'])

    # Merge relabeled goals into the task dict
    traj['task'] = tree_merge(traj['task'], goal)

    return traj
