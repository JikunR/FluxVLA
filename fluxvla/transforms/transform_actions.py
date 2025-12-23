from typing import Dict, List

import numpy as np

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class ProcessLiberoActions:

    def __init__(self, mask: List[bool] = None) -> None:
        """ProcessLiberoActions is a transform
        that modifies the actions in the data
        by subtracting the state values from
        the actions based on a mask.

        Args:
            mask (List[bool], optional): A list
                indicating which dimensions
                of the state should be subtracted from
                the actions.
                If None, no subtraction is performed.
        """
        self.mask = np.asarray(mask, dtype=bool)

    def __call__(self, data: Dict) -> Dict:
        if 'actions' not in data or self.mask is None:
            return data

        states, actions = data['states'], data['actions']
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, states[..., :dims], 0), axis=-2)
        data['actions'] = actions

        return data
