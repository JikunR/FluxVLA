import torch

from fluxvla.engines.runners.serving.policy import FluxVLAPolicy


class _Policy:

    def __init__(self):
        self.seeds = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def predict_action(self, **batch):
        self.seeds.append(batch['seed'])
        return torch.zeros(2, 3)


def test_policy_receives_episode_seed():
    model = _Policy()
    policy = FluxVLAPolicy(
        model,
        dataset=lambda observation: {},
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=True,
    )

    actions, _ = policy.predict_action({}, '', policy_seed=17)

    assert actions.shape == (2, 3)
    assert model.seeds == [17]
