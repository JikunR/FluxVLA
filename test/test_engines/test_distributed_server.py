import socket
import threading
import time
import uuid
from unittest.mock import Mock

import numpy as np
import pytest

from fluxvla.engines.runners.serving.distributed_server import \
    ServerSupervisor, StatelessZMQCoordinator
from fluxvla.engines.runners.serving.zmq_protocol import (decode_frame,
                                                          decode_header,
                                                          empty_payload,
                                                          encode_frame,
                                                          encode_header)
from scripts.distributed_zmq_server import _normalize_cluster_environment


def _free_endpoint():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return 'tcp://127.0.0.1:{}'.format(probe.getsockname()[1])


def _request(endpoint, header, payload):
    import zmq

    context = zmq.Context()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.LINGER, 0)
    client.setsockopt(zmq.RCVTIMEO, 3000)
    client.connect(endpoint)
    try:
        client.send_multipart([header, payload])
        frames = client.recv_multipart()
        return decode_header(frames[0]), decode_frame(frames[1])
    finally:
        client.close(linger=0)
        context.term()


class _Reporter:

    def process_event(self, **kwargs):
        return {'accepted': True}


def test_server_cli_normalizes_pai_cluster_environment():
    environ = {
        'MLP_ROLE_INDEX': '1',
        'MLP_WORKER_NUM': '3',
        'MLP_WORKER_0_HOST': 'coordinator',
        'MLP_WORKER_0_PORT': '23456',
    }

    topology = _normalize_cluster_environment(environ)

    assert topology == (1, 3, 'coordinator')
    assert environ['RANK'] == '1'
    assert environ['WORLD_SIZE'] == '3'
    assert environ['MASTER_ADDR'] == 'coordinator'
    assert environ['MASTER_PORT'] == '23456'


class _RejectedReporter:

    def process_event(self, **kwargs):
        return {'accepted': False, 'error': 'invalid event'}


def test_run_end_stops_server():
    header = {
        'event_type': 'run_end',
        'request_id': 'request',
        'run_session_id': 'run',
        'sequence': 1,
    }
    coordinator = StatelessZMQCoordinator(
        frontend_bind=_free_endpoint(),
        backend_bind=_free_endpoint(),
        control_bind=_free_endpoint(),
        evaluation_reporter=_Reporter(),
    )
    coordinator._send_frontend = Mock()
    coordinator._begin_shutdown = Mock()
    coordinator._handle_report((), header, encode_frame({}))
    coordinator._begin_shutdown.assert_called_once_with()


def test_rejected_report_stops_server():
    coordinator = StatelessZMQCoordinator(
        frontend_bind=_free_endpoint(),
        backend_bind=_free_endpoint(),
        control_bind=_free_endpoint(),
        evaluation_reporter=_RejectedReporter(),
    )
    coordinator._send_frontend = Mock()
    coordinator._begin_shutdown = Mock()

    coordinator._handle_report(
        (), {
            'event_type': 'episode_start',
            'request_id': 'request',
            'run_session_id': 'run',
            'sequence': 1,
        }, encode_frame({}))

    coordinator._begin_shutdown.assert_called_once_with()


def test_one_model_worker_serves_two_concurrent_clients():
    import zmq

    frontend = _free_endpoint()
    backend = _free_endpoint()
    control = _free_endpoint()
    deployment_id = uuid.uuid4().hex
    coordinator = StatelessZMQCoordinator(
        frontend_bind=frontend,
        backend_bind=backend,
        control_bind=control,
        deployment_id=deployment_id,
        request_timeout_s=3.0,
    )
    coordinator_thread = threading.Thread(target=coordinator.run, daemon=True)
    coordinator_thread.start()

    worker_done = threading.Event()

    def model_worker():
        context = zmq.Context()
        worker = context.socket(zmq.DEALER)
        worker.setsockopt(zmq.IDENTITY, b'worker:0')
        worker.setsockopt(zmq.LINGER, 0)
        worker.connect(backend)
        common = {
            'deployment_id': deployment_id,
            'worker_id': 'worker',
            'supervisor_id': 'supervisor',
            'node_id': 'node',
            'device': 'cuda:0',
        }
        worker.send_multipart([
            encode_header('register_worker', **common),
            empty_payload(),
        ])
        worker.send_multipart([
            encode_header('ready', **common),
            empty_payload(),
        ])
        try:
            for _ in range(2):
                frames = worker.recv_multipart()
                header = decode_header(frames[0])
                observation = decode_frame(frames[1])
                time.sleep(0.05)
                worker.send_multipart([
                    encode_header(
                        'predict_result',
                        **common,
                        request_id=header['request_id'],
                        inference_time_s=0.05,
                    ),
                    encode_frame({
                        'actions':
                        np.asarray([[observation['value']]], dtype=np.float32),
                    }),
                ])
        finally:
            worker_done.set()
            worker.close(linger=0)
            context.term()

    worker_thread = threading.Thread(target=model_worker, daemon=True)
    worker_thread.start()
    deadline = time.monotonic() + 3.0
    while coordinator.ready_workers != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert coordinator.ready_workers == 1

    responses = {}

    def send_prediction(value):
        request_id = f'request-{value}'
        responses[value] = _request(
            frontend,
            encode_header(
                'predict_action',
                deployment_id=deployment_id,
                request_id=request_id,
                seed=value,
            ),
            encode_frame({'value': value}),
        )

    clients = [
        threading.Thread(target=send_prediction, args=(value, ))
        for value in (1, 2)
    ]
    for client in clients:
        client.start()
    for client in clients:
        client.join(timeout=3.0)

    coordinator.close()
    coordinator_thread.join(timeout=3.0)
    worker_thread.join(timeout=3.0)
    assert worker_done.is_set()
    assert not coordinator_thread.is_alive()
    assert set(responses) == {1, 2}
    for value, (header, payload) in responses.items():
        assert header['ok'] is True
        assert header['request_id'] == f'request-{value}'
        np.testing.assert_array_equal(payload['actions'],
                                      np.asarray([[value]], dtype=np.float32))


def test_coordinator_aggregates_heterogeneous_supervisor_capacity():
    coordinator = StatelessZMQCoordinator(
        frontend_bind=_free_endpoint(),
        backend_bind=_free_endpoint(),
        control_bind=_free_endpoint(),
        expected_supervisors=2,
        deployment_id='deployment',
    )
    coordinator._control = Mock()
    coordinator._control.recv_multipart.side_effect = [
        [
            b'supervisor-a',
            encode_header(
                'register_supervisor',
                deployment_id='deployment',
                supervisor_id='supervisor-a',
                worker_count=1,
            ),
        ],
        [
            b'supervisor-b',
            encode_header(
                'register_supervisor',
                deployment_id='deployment',
                supervisor_id='supervisor-b',
                worker_count=2,
            ),
        ],
    ]

    coordinator._receive_control()
    coordinator._receive_control()

    assert coordinator.expected_workers == 3
    for worker_id in range(3):
        coordinator._register_worker(
            f'worker-{worker_id}'.encode(), {
                'worker_id': f'worker-{worker_id}',
                'supervisor_id': 'supervisor-a',
                'node_id': 'node',
                'device': str(worker_id),
            }, 'ready')
    assert coordinator.status == 'starting'
    coordinator._workers[b'worker-1'].supervisor_id = 'supervisor-b'
    coordinator._workers[b'worker-2'].supervisor_id = 'supervisor-b'
    assert coordinator.status == 'ready'


def test_supervisor_fails_instead_of_restarting_worker(tmp_path):
    supervisor = ServerSupervisor(
        control_endpoint=_free_endpoint(),
        backend_endpoint=_free_endpoint(),
        config_path=str(tmp_path / 'config.py'),
        ckpt_path=str(tmp_path / 'checkpoint'),
        devices=('0', ),
        supervisor_id='supervisor',
        node_id='node',
        deployment_id='deployment',
    )
    process = Mock(pid=123, poll=Mock(return_value=17))
    supervisor._processes['node-gpu-0'] = process

    with pytest.raises(RuntimeError, match='exited with code 17'):
        supervisor._monitor_workers(Mock())

    assert supervisor._processes == {}
