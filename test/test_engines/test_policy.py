import numpy as np
import pytest
import torch

from fluxvla.engines.runners.serving.policy import (
    FluxVLAPolicy, discover_cuda_worker_devices, resolve_inference_devices)


class _Dataset:

    def __init__(self):
        self.outputs = []

    def __call__(self, observation):
        output = dict(observation)
        self.outputs.append(output)
        return output


class _Model:

    def __init__(self):
        self.batches = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def predict_action(self, **batch):
        self.batches.append(batch)
        return torch.ones((1, 2, 3), dtype=torch.float32)


def test_policy_resets_model_history_for_every_request():
    dataset = _Dataset()
    model = _Model()
    policy = FluxVLAPolicy(
        vla=model,
        dataset=dataset,
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=True,
    )

    first, _ = policy.predict({'states': np.asarray([1.0])}, '', seed=1)
    second, _ = policy.predict({'states': np.asarray([2.0])}, '', seed=2)

    assert first.shape == (2, 3)
    assert second.shape == (2, 3)
    assert [batch['reset_history'] for batch in model.batches] == [True, True]


def test_discovers_all_visible_cuda_devices():
    devices = discover_cuda_worker_devices(
        environ={'CUDA_VISIBLE_DEVICES': '3,7,GPU-test'})

    assert devices == ('3', '7', 'GPU-test')


def test_platform_worker_count_limits_visible_cuda_devices():
    devices = discover_cuda_worker_devices(environ={
        'CUDA_VISIBLE_DEVICES': '3,7,9',
        'NPROC_PER_NODE': '2',
    })

    assert devices == ('3', '7')


def test_platform_gpu_count_supports_unmasked_cloud_nodes():
    devices = discover_cuda_worker_devices(environ={'MLP_WORKER_GPU': '3'})

    assert devices == ('0', '1', '2')


def test_rejects_worker_count_larger_than_visible_devices():
    with pytest.raises(ValueError, match='only 2 CUDA devices are visible'):
        discover_cuda_worker_devices(
            worker_count=3, environ={'CUDA_VISIBLE_DEVICES': '4,5'})


def test_explicit_worker_devices_override_auto_discovery():
    devices = resolve_inference_devices(
        worker_devices=['cuda:2', 'cpu'], num_workers=None)

    assert devices == ('2', 'cpu')
