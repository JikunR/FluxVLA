import os
import socket
import threading
import time
import uuid
from unittest.mock import Mock

import numpy as np

from fluxvla.engines.runners.serving.distributed_server import \
    StatelessZMQCoordinator, _start_coordinator_process
from fluxvla.engines.runners.serving.zmq_protocol import (decode_frame,
                                                          decode_header,
                                                          empty_payload,
                                                          encode_frame,
                                                          encode_header)
from scripts.distributed_zmq_server import _torchrun_environment


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


def test_server_cli_reads_torchrun_environment(monkeypatch):
    monkeypatch.setenv('RANK', '2')
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setenv('MASTER_ADDR', 'coordinator')

    assert _torchrun_environment() == (2, 4, 1, 'coordinator')


def test_server_coordinator_is_outside_process_group(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    inherited = {}
    process = Mock()
    process.start.side_effect = lambda: inherited.update(os.environ)

    _start_coordinator_process(process)

    assert 'WORLD_SIZE' not in inherited
    assert os.environ['WORLD_SIZE'] == '4'


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
    deployment_id = uuid.uuid4().hex
    coordinator = StatelessZMQCoordinator(
        frontend_bind=frontend,
        backend_bind=backend,
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


def test_coordinator_waits_for_all_workers():
    coordinator = StatelessZMQCoordinator(
        frontend_bind=_free_endpoint(),
        backend_bind=_free_endpoint(),
        expected_workers=3,
        deployment_id='deployment',
    )
    for worker_id in range(3):
        coordinator._register_worker(
            f'worker-{worker_id}'.encode(), {
                'worker_id': f'worker-{worker_id}',
                'node_id': 'node',
                'device': str(worker_id),
            }, 'ready')
    assert coordinator.status == 'ready'
