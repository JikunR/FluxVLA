"""Stateless, multi-node ZMQ inference service for FluxThemis."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import signal
import socket as network_socket
import subprocess
import sys
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .policy import (build_policy_from_config, config_get,
                     cuda_visibility_token, get_server_config, require_mapping,
                     resolve_checkpoint_path, resolve_report_config_path,
                     resolve_report_result_root)
from .zmq_protocol import (decode_frame, decode_header, empty_payload,
                           encode_frame, encode_header)


@dataclass
class _PendingRequest:
    route: tuple[bytes, ...]
    header: dict[str, Any]
    payload: bytes
    deadline: float


@dataclass
class _ModelWorkerState:
    identity: bytes
    worker_id: str
    supervisor_id: str
    node_id: str
    device: str
    generation: int
    status: str
    last_seen: float
    request_id: str | None = None


@dataclass
class _SupervisorState:
    identity: bytes
    supervisor_id: str
    node_id: str
    last_seen: float


class StatelessZMQCoordinator:
    """Route independent prediction requests to the next idle model worker."""

    def __init__(self,
                 frontend_bind: str,
                 backend_bind: str,
                 control_bind: str,
                 evaluation_reporter: Any = None,
                 minimum_ready_workers: int = 1,
                 expected_workers: int | None = None,
                 max_pending_requests: int = 128,
                 request_timeout_s: float = 120.0,
                 startup_timeout_s: float = 900.0,
                 heartbeat_timeout_s: float = 30.0,
                 manifest_path: str | os.PathLike | None = None,
                 advertised_endpoint: str | None = None,
                 deployment_id: str | None = None,
                 config_fingerprint: str = '',
                 checkpoint_fingerprint: str = '',
                 exit_after_run: bool = False) -> None:
        self.frontend_bind = _endpoint(frontend_bind, 'frontend_bind')
        self.backend_bind = _endpoint(backend_bind, 'backend_bind')
        self.control_bind = _endpoint(control_bind, 'control_bind')
        self.evaluation_reporter = evaluation_reporter
        self.minimum_ready_workers = _positive_int(minimum_ready_workers,
                                                   'minimum_ready_workers')
        self.expected_workers = (None if
                                 expected_workers is None else _positive_int(
                                     expected_workers, 'expected_workers'))
        if (self.expected_workers is not None
                and self.minimum_ready_workers > self.expected_workers):
            raise ValueError('minimum_ready_workers cannot exceed '
                             'expected_workers')
        self.max_pending_requests = _positive_int(max_pending_requests,
                                                  'max_pending_requests')
        self.request_timeout_s = _positive_float(request_timeout_s,
                                                 'request_timeout_s')
        self.startup_timeout_s = _positive_float(startup_timeout_s,
                                                 'startup_timeout_s')
        self.heartbeat_timeout_s = _positive_float(heartbeat_timeout_s,
                                                   'heartbeat_timeout_s')
        self.manifest_path = (None if manifest_path is None else
                              Path(manifest_path).expanduser().resolve())
        self.advertised_endpoint = (advertised_endpoint or frontend_bind)
        self.deployment_id = deployment_id or uuid.uuid4().hex
        self.config_fingerprint = str(config_fingerprint)
        self.checkpoint_fingerprint = str(checkpoint_fingerprint)
        self.exit_after_run = bool(exit_after_run)

        self._workers: dict[bytes, _ModelWorkerState] = {}
        self._workers_by_id: dict[str, bytes] = {}
        self._idle_workers: deque[bytes] = deque()
        self._supervisors: dict[str, _SupervisorState] = {}
        self._shutdown_completed: set[str] = set()
        self._pending: deque[_PendingRequest] = deque()
        self._requests: dict[str, _PendingRequest] = {}
        self._running = False
        self._shutdown_started = False
        self._shutdown_deadline: float | None = None
        self._frontend = None
        self._backend = None
        self._control = None
        self._context = None

    @property
    def ready_workers(self) -> int:
        return sum(worker.status in {'idle', 'busy'}
                   for worker in self._workers.values())

    @property
    def status(self) -> str:
        if self._shutdown_started:
            return 'stopping'
        if self.ready_workers >= self.minimum_ready_workers:
            return 'ready'
        return 'starting'

    def run(self) -> None:
        import zmq

        self._context = zmq.Context()
        self._frontend = self._context.socket(zmq.ROUTER)
        self._backend = self._context.socket(zmq.ROUTER)
        self._control = self._context.socket(zmq.ROUTER)
        for current in (self._frontend, self._backend, self._control):
            current.setsockopt(zmq.LINGER, 0)
            current.setsockopt(zmq.SNDHWM, self.max_pending_requests * 2)
            current.setsockopt(zmq.RCVHWM, self.max_pending_requests * 2)
        self._frontend.bind(self.frontend_bind)
        self._backend.bind(self.backend_bind)
        self._control.bind(self.control_bind)
        poller = zmq.Poller()
        poller.register(self._frontend, zmq.POLLIN)
        poller.register(self._backend, zmq.POLLIN)
        poller.register(self._control, zmq.POLLIN)
        self._running = True
        self._install_signal_handlers()
        self._write_manifest()
        print(
            '[FluxVLA] ZMQ coordinator listening '
            f'frontend={self.frontend_bind} backend={self.backend_bind} '
            f'control={self.control_bind}',
            flush=True)
        try:
            while self._running:
                events = dict(poller.poll(timeout=200))
                if self._frontend in events:
                    self._receive_frontend()
                if self._backend in events:
                    self._receive_backend()
                if self._control in events:
                    self._receive_control()
                self._expire_requests()
                self._expire_workers()
                self._dispatch_pending()
                supervisors_stopped = (
                    not self._supervisors
                    or self._shutdown_completed >= set(self._supervisors))
                if (self._shutdown_started and not self._requests
                        and (supervisors_stopped or self._shutdown_expired())):
                    self._running = False
        finally:
            self._begin_shutdown()
            self._write_manifest()
            for current in (self._frontend, self._backend, self._control):
                if current is not None:
                    current.close(linger=0)
            if self._context is not None:
                self._context.term()

    def close(self) -> None:
        self._running = False

    def _receive_frontend(self) -> None:
        frames = self._frontend.recv_multipart()
        if len(frames) < 4 or frames[-3] != b'':
            return
        route = tuple(frames[:-2])
        try:
            header = decode_header(frames[-2])
            payload = frames[-1]
            message_type = header['type']
            if message_type in {'ping', 'status'}:
                self._reply_status(route, header)
            elif message_type == 'predict_action':
                self._validate_deployment(header)
                self._queue_prediction(route, header, payload)
            elif message_type == 'report_event':
                self._validate_deployment(header)
                self._handle_report(route, header, payload)
            elif message_type == 'kill':
                self._validate_deployment(header)
                self._send_frontend(
                    route,
                    encode_header(
                        'kill_response',
                        ok=True,
                        request_id=str(header.get('request_id', ''))),
                    empty_payload())
                self._begin_shutdown()
            else:
                raise ValueError(f'Unknown endpoint: {message_type}')
        except Exception as exc:
            self._send_frontend_error(
                route,
                request_id=_header_string(locals().get('header'),
                                          'request_id'),
                error=exc)

    def _reply_status(self, route: tuple[bytes, ...],
                      request: Mapping[str, Any]) -> None:
        payload = {
            'status': self.status,
            'deployment_id': self.deployment_id,
            'server_mode': 'stateless',
            'evaluation_reporting': self.evaluation_reporter is not None,
            'workers_ready': self.ready_workers,
            'workers_expected': self.expected_workers,
            'minimum_ready_workers': self.minimum_ready_workers,
            'pending_requests': len(self._pending),
        }
        self._send_frontend(
            route,
            encode_header(
                'status_response',
                ok=True,
                request_id=str(request.get('request_id', '')),
            ), encode_frame(payload))

    def _queue_prediction(self, route: tuple[bytes, ...],
                          header: dict[str, Any], payload: bytes) -> None:
        request_id = _required_string(header, 'request_id')
        if request_id in self._requests:
            raise ValueError(f'Duplicate active request_id {request_id!r}')
        if len(self._pending) >= self.max_pending_requests:
            self._send_frontend_error(
                route,
                request_id,
                RuntimeError('prediction queue is full'),
                error_type='overloaded')
            return
        requested_deadline_ms = header.get('deadline_ms')
        timeout_s = self.request_timeout_s
        if requested_deadline_ms is not None:
            if (isinstance(requested_deadline_ms, bool)
                    or not isinstance(requested_deadline_ms, (int, float))
                    or requested_deadline_ms <= 0):
                raise ValueError('deadline_ms must be a positive number')
            timeout_s = min(timeout_s, float(requested_deadline_ms) / 1000.0)
        pending = _PendingRequest(
            route=route,
            header=header,
            payload=payload,
            deadline=time.monotonic() + timeout_s,
        )
        self._pending.append(pending)
        self._requests[request_id] = pending

    def _handle_report(self, route: tuple[bytes, ...],
                       header: Mapping[str, Any], payload: bytes) -> None:
        if self.evaluation_reporter is None:
            raise RuntimeError('FluxVLA evaluation reporting is disabled')
        body = decode_frame(payload)
        if not isinstance(body, Mapping):
            raise TypeError('report_event payload must be a mapping')
        result = self.evaluation_reporter.process_event(
            event_type=_required_string(header, 'event_type'),
            request_id=_required_string(header, 'request_id'),
            run_session_id=_required_string(header, 'run_id'),
            sequence=_required_int(header, 'sequence'),
            payload=dict(body),
        )
        if not isinstance(result, Mapping):
            raise TypeError('evaluation reporter response must be a mapping')
        self._send_frontend(
            route,
            encode_header(
                'report_response',
                ok=True,
                request_id=str(header.get('request_id', '')),
            ), encode_frame(dict(result)))
        if (self.exit_after_run and header.get('event_type') == 'run_end'
                and bool(result.get('accepted', False))):
            self._begin_shutdown()

    def _receive_backend(self) -> None:
        frames = self._backend.recv_multipart()
        if len(frames) < 2:
            return
        identity = frames[0]
        try:
            header = decode_header(frames[1])
            self._validate_deployment(header)
            payload = frames[2] if len(frames) > 2 else empty_payload()
            message_type = header['type']
            if message_type in {'register_worker', 'loading', 'ready'}:
                self._register_worker(identity, header, message_type)
            elif message_type == 'heartbeat':
                worker = self._require_worker(identity, header)
                worker.last_seen = time.monotonic()
            elif message_type in {'predict_result', 'predict_error'}:
                self._finish_prediction(identity, header, payload)
            elif message_type == 'drained':
                worker = self._require_worker(identity, header)
                worker.status = 'draining'
                worker.last_seen = time.monotonic()
            else:
                raise ValueError(
                    f'Unknown model-worker message {message_type!r}')
        except Exception as exc:
            print(
                f'[FluxVLA] rejected model-worker message: {exc}',
                file=sys.stderr,
                flush=True)

    def _register_worker(self, identity: bytes, header: Mapping[str, Any],
                         status: str) -> None:
        worker_id = _required_string(header, 'worker_id')
        generation = _required_int(header, 'generation', minimum=0)
        previous_identity = self._workers_by_id.get(worker_id)
        if previous_identity is not None and previous_identity != identity:
            previous = self._workers.get(previous_identity)
            if previous is not None and generation < previous.generation:
                return
            if previous is not None:
                self._fail_worker_request(previous,
                                          'model worker was replaced')
                previous.status = 'dead'
        worker = self._workers.get(identity)
        worker_status = 'starting' if status == 'register_worker' else status
        if worker is None:
            worker = _ModelWorkerState(
                identity=identity,
                worker_id=worker_id,
                supervisor_id=str(header.get('supervisor_id', '')),
                node_id=str(header.get('node_id', '')),
                device=str(header.get('device', '')),
                generation=generation,
                status=worker_status,
                last_seen=time.monotonic(),
            )
            self._workers[identity] = worker
        elif generation < worker.generation:
            return
        elif worker.status == 'dead' and generation == worker.generation:
            return
        else:
            worker.generation = generation
            worker.status = worker_status
            worker.last_seen = time.monotonic()
        self._workers_by_id[worker_id] = identity
        if status == 'ready':
            worker.status = 'idle'
            self._idle_workers.append(identity)
            print(
                f'[FluxVLA] model worker ready id={worker_id} '
                f'node={worker.node_id} device={worker.device} '
                f'generation={generation}',
                flush=True)
        self._write_manifest()

    def _finish_prediction(self, identity: bytes, header: Mapping[str, Any],
                           payload: bytes) -> None:
        worker = self._require_worker(identity, header)
        request_id = _required_string(header, 'request_id')
        if worker.request_id != request_id:
            return
        pending = self._requests.pop(request_id, None)
        worker.request_id = None
        worker.status = 'idle'
        worker.last_seen = time.monotonic()
        self._idle_workers.append(identity)
        if pending is None:
            return
        if header['type'] == 'predict_error':
            error = str(header.get('error') or 'model worker failed')
            self._send_frontend_error(
                pending.route,
                request_id,
                RuntimeError(error),
                error_type='inference_error')
            return
        response_header = dict(header)
        response_header.pop('worker_id', None)
        response_header.pop('generation', None)
        response_header['type'] = 'predict_response'
        response_header['ok'] = True
        self._send_frontend(pending.route, encode_frame(response_header),
                            payload)

    def _receive_control(self) -> None:
        frames = self._control.recv_multipart()
        if len(frames) < 2:
            return
        identity = frames[0]
        try:
            header = decode_header(frames[1])
            self._validate_deployment(header)
            message_type = header['type']
            supervisor_id = _required_string(header, 'supervisor_id')
            if message_type in {'register_supervisor', 'heartbeat'}:
                self._supervisors[supervisor_id] = _SupervisorState(
                    identity=identity,
                    supervisor_id=supervisor_id,
                    node_id=str(header.get('node_id', '')),
                    last_seen=time.monotonic(),
                )
            elif message_type == 'worker_exit':
                self._supervisors[supervisor_id].last_seen = time.monotonic()
                worker_id = _required_string(header, 'worker_id')
                worker_identity = self._workers_by_id.get(worker_id)
                worker = self._workers.get(worker_identity)
                if (worker is not None and int(header.get('generation', -1)) >=
                        worker.generation):
                    self._fail_worker_request(worker,
                                              'model worker process exited')
                    worker.status = 'dead'
                    self._write_manifest()
            elif message_type in {'worker_started', 'command_completed'}:
                state = self._supervisors.get(supervisor_id)
                if state is not None:
                    state.last_seen = time.monotonic()
            elif message_type == 'shutdown_completed':
                self._shutdown_completed.add(supervisor_id)
            else:
                raise ValueError(
                    f'Unknown server-supervisor message {message_type!r}')
        except Exception as exc:
            print(
                f'[FluxVLA] rejected supervisor message: {exc}',
                file=sys.stderr,
                flush=True)

    def _dispatch_pending(self) -> None:
        while self._pending:
            identity = self._next_idle_worker()
            if identity is None:
                return
            pending = self._pending.popleft()
            request_id = str(pending.header['request_id'])
            if request_id not in self._requests:
                continue
            if pending.deadline <= time.monotonic():
                self._requests.pop(request_id, None)
                self._send_frontend_error(
                    pending.route,
                    request_id,
                    TimeoutError('prediction expired while queued'),
                    error_type='timeout')
                continue
            worker = self._workers[identity]
            worker.status = 'busy'
            worker.request_id = request_id
            worker.last_seen = time.monotonic()
            header = dict(pending.header)
            header['type'] = 'predict'
            header['worker_id'] = worker.worker_id
            header['generation'] = worker.generation
            self._backend.send_multipart(
                [identity, encode_frame(header), pending.payload])

    def _next_idle_worker(self) -> bytes | None:
        while self._idle_workers:
            identity = self._idle_workers.popleft()
            worker = self._workers.get(identity)
            if worker is not None and worker.status == 'idle':
                return identity
        return None

    def _expire_requests(self) -> None:
        now = time.monotonic()
        if self._pending:
            retained = deque()
            while self._pending:
                pending = self._pending.popleft()
                request_id = str(pending.header['request_id'])
                if pending.deadline > now:
                    retained.append(pending)
                    continue
                self._requests.pop(request_id, None)
                self._send_frontend_error(
                    pending.route,
                    request_id,
                    TimeoutError('prediction expired while queued'),
                    error_type='timeout')
            self._pending = retained
        for worker in list(self._workers.values()):
            if worker.status != 'busy' or worker.request_id is None:
                continue
            pending = self._requests.get(worker.request_id)
            if pending is not None and pending.deadline <= now:
                self._fail_worker_request(worker, 'model inference timed out',
                                          'timeout')
                worker.status = 'dead'
                self._request_worker_restart(worker)

    def _expire_workers(self) -> None:
        now = time.monotonic()
        for worker in self._workers.values():
            if worker.status in {'busy', 'dead', 'draining'}:
                continue
            timeout_s = (
                self.startup_timeout_s if worker.status
                in {'starting', 'loading'} else self.heartbeat_timeout_s)
            if now - worker.last_seen <= timeout_s:
                continue
            worker.status = 'dead'
            self._request_worker_restart(worker)
            self._write_manifest()

    def _request_worker_restart(self, worker: _ModelWorkerState) -> None:
        supervisor = self._supervisors.get(worker.supervisor_id)
        if supervisor is None:
            return
        self._control.send_multipart([
            supervisor.identity,
            encode_header(
                'restart_worker',
                deployment_id=self.deployment_id,
                command_id=uuid.uuid4().hex,
                supervisor_id=worker.supervisor_id,
                worker_id=worker.worker_id,
                expected_generation=worker.generation,
            ),
            empty_payload(),
        ])

    def _fail_worker_request(self,
                             worker: _ModelWorkerState,
                             message: str,
                             error_type: str = 'worker_unavailable') -> None:
        request_id = worker.request_id
        worker.request_id = None
        if request_id is None:
            return
        pending = self._requests.pop(request_id, None)
        if pending is not None:
            self._send_frontend_error(
                pending.route,
                request_id,
                RuntimeError(message),
                error_type=error_type)

    def _require_worker(self, identity: bytes,
                        header: Mapping[str, Any]) -> _ModelWorkerState:
        worker = self._workers.get(identity)
        if worker is None:
            raise RuntimeError(
                'model worker must register before sending data')
        generation = _required_int(header, 'generation', minimum=0)
        if generation != worker.generation:
            raise RuntimeError('stale model worker generation')
        worker_id = _required_string(header, 'worker_id')
        if worker_id != worker.worker_id:
            raise RuntimeError('model worker identity mismatch')
        return worker

    def _validate_deployment(self, header: Mapping[str, Any]) -> None:
        if header.get('deployment_id') != self.deployment_id:
            raise RuntimeError('ZMQ deployment_id mismatch')

    def _send_frontend_error(self,
                             route: tuple[bytes, ...],
                             request_id: str,
                             error: Exception,
                             error_type: str = 'protocol_error') -> None:
        self._send_frontend(
            route,
            encode_header(
                'error',
                ok=False,
                request_id=request_id,
                error_type=error_type,
                error=f'{type(error).__name__}: {error}',
            ), empty_payload())

    def _send_frontend(self, route: tuple[bytes, ...], header: bytes,
                       payload: bytes) -> None:
        self._frontend.send_multipart([*route, header, payload])

    def _begin_shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._shutdown_deadline = time.monotonic() + 10.0
        for supervisor in self._supervisors.values():
            self._control.send_multipart([
                supervisor.identity,
                encode_header(
                    'shutdown_node',
                    deployment_id=self.deployment_id,
                    command_id=uuid.uuid4().hex,
                    supervisor_id=supervisor.supervisor_id,
                ),
                empty_payload(),
            ])

    def _shutdown_expired(self) -> bool:
        return (self._shutdown_deadline is not None
                and time.monotonic() >= self._shutdown_deadline)

    def _write_manifest(self) -> None:
        if self.manifest_path is None:
            return
        workers_ready = self.ready_workers
        value = {
            'protocol_version': 1,
            'deployment_id': self.deployment_id,
            'endpoint': self.advertised_endpoint,
            'workers_expected': self.expected_workers,
            'workers_ready': workers_ready,
            'minimum_ready_workers': self.minimum_ready_workers,
            'server_mode': 'stateless',
            'evaluation_reporting': self.evaluation_reporter is not None,
            'status': self.status,
            'config_fingerprint': self.config_fingerprint,
            'checkpoint_fingerprint': self.checkpoint_fingerprint,
            'updated_at': time.time(),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix +
                                                   '.tmp')
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
        os.replace(temporary, self.manifest_path)

    def _install_signal_handlers(self) -> None:

        def stop(_signum, _frame):
            self._begin_shutdown()

        if threading_available_for_signals():
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)


class ServerSupervisor:
    """Own and restart all model-worker subprocesses on one host."""

    def __init__(
            self,
            control_endpoint: str,
            backend_endpoint: str,
            config_path: str,
            ckpt_path: str,
            devices: Sequence[str],
            supervisor_id: str,
            node_id: str,
            deployment_id: str,
            cfg_options: Mapping[str, Any] | None = None,
            heartbeat_interval_s: float = 5.0,
            shutdown_timeout_s: float = 20.0,
            coordinator_process: multiprocessing.Process | None = None
    ) -> None:
        self.control_endpoint = _endpoint(control_endpoint, 'control_endpoint')
        self.backend_endpoint = _endpoint(backend_endpoint, 'backend_endpoint')
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        self.devices = tuple(devices)
        if not self.devices:
            raise ValueError('devices cannot be empty')
        self.supervisor_id = _nonempty(supervisor_id, 'supervisor_id')
        self.node_id = _nonempty(node_id, 'node_id')
        self.deployment_id = _nonempty(deployment_id, 'deployment_id')
        self.cfg_options = dict(cfg_options or {})
        self.heartbeat_interval_s = _positive_float(heartbeat_interval_s,
                                                    'heartbeat_interval_s')
        self.shutdown_timeout_s = _positive_float(shutdown_timeout_s,
                                                  'shutdown_timeout_s')
        self.coordinator_process = coordinator_process
        self._processes: dict[str, subprocess.Popen] = {}
        self._generations: dict[str, int] = {}
        self._shutting_down = False

    def run(self) -> None:
        import zmq

        context = zmq.Context()
        control = context.socket(zmq.DEALER)
        control.setsockopt(zmq.IDENTITY, self.supervisor_id.encode())
        control.setsockopt(zmq.LINGER, 0)
        control.connect(self.control_endpoint)
        poller = zmq.Poller()
        poller.register(control, zmq.POLLIN)
        self._send_control(control, 'register_supervisor')
        for slot, device in enumerate(self.devices):
            worker_id = f'{self.node_id}-gpu-{slot}'
            self._generations[worker_id] = 0
            self._start_worker(control, worker_id, device)
        next_heartbeat = 0.0
        try:
            while not self._shutting_down:
                now = time.monotonic()
                if now >= next_heartbeat:
                    self._send_control(
                        control,
                        'heartbeat',
                        inventory=self._inventory(),
                    )
                    next_heartbeat = now + self.heartbeat_interval_s
                events = dict(poller.poll(timeout=200))
                if control in events:
                    header = decode_header(control.recv_multipart()[0])
                    self._handle_command(control, header)
                if (not self._shutting_down
                        and self.coordinator_process is not None
                        and self.coordinator_process.exitcode is not None):
                    if self.coordinator_process.exitcode == 0:
                        self._shutting_down = True
                    else:
                        raise RuntimeError(
                            'server coordinator exited with code '
                            f'{self.coordinator_process.exitcode}')
                self._monitor_workers(control)
        finally:
            self._shutting_down = True
            self._stop_all_workers()
            try:
                self._send_control(
                    control, 'shutdown_completed', inventory=self._inventory())
            except Exception:
                pass
            control.close(linger=0)
            context.term()

    def _start_worker(self, control: Any, worker_id: str, device: str) -> None:
        generation = self._generations[worker_id]
        command = [
            sys.executable,
            '-m',
            'fluxvla.engines.runners.serving.distributed_server',
            'model-worker',
            '--config',
            self.config_path,
            '--ckpt-path',
            self.ckpt_path,
            '--backend-endpoint',
            self.backend_endpoint,
            '--worker-id',
            worker_id,
            '--supervisor-id',
            self.supervisor_id,
            '--node-id',
            self.node_id,
            '--generation',
            str(generation),
            '--deployment-id',
            self.deployment_id,
            '--device',
            'cuda:0' if cuda_visibility_token(device) is not None else device,
            '--parent-pid',
            str(os.getpid()),
            '--cfg-options-json',
            json.dumps(self.cfg_options),
        ]
        environment = os.environ.copy()
        physical_device = cuda_visibility_token(device)
        if physical_device is not None:
            environment['CUDA_VISIBLE_DEVICES'] = physical_device
        process = subprocess.Popen(
            command, env=environment, start_new_session=True)
        self._processes[worker_id] = process
        self._send_control(
            control,
            'worker_started',
            worker_id=worker_id,
            generation=generation,
            process_id=process.pid,
            device=device,
        )

    def _monitor_workers(self, control: Any) -> None:
        for worker_id, process in list(self._processes.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            generation = self._generations[worker_id]
            self._send_control(
                control,
                'worker_exit',
                worker_id=worker_id,
                generation=generation,
                process_id=process.pid,
                exit_code=exit_code,
            )
            del self._processes[worker_id]
            if not self._shutting_down:
                self._generations[worker_id] = generation + 1
                device = self.devices[int(worker_id.rsplit('-', 1)[1])]
                self._start_worker(control, worker_id, device)

    def _handle_command(self, control: Any, header: Mapping[str, Any]) -> None:
        message_type = header['type']
        if message_type == 'restart_worker':
            worker_id = _required_string(header, 'worker_id')
            expected = _required_int(header, 'expected_generation', minimum=0)
            current = self._generations.get(worker_id)
            if current == expected:
                process = self._processes.pop(worker_id, None)
                if process is not None:
                    _terminate_process(process, self.shutdown_timeout_s)
                self._generations[worker_id] = current + 1
                device = self.devices[int(worker_id.rsplit('-', 1)[1])]
                self._start_worker(control, worker_id, device)
            self._send_control(
                control,
                'command_completed',
                command_id=str(header.get('command_id', '')),
                worker_id=worker_id,
            )
        elif message_type == 'shutdown_node':
            self._shutting_down = True
        elif message_type == 'drain_worker':
            worker_id = _required_string(header, 'worker_id')
            process = self._processes.get(worker_id)
            if process is not None:
                process.send_signal(signal.SIGTERM)
        else:
            raise ValueError(f'Unknown supervisor command {message_type!r}')

    def _stop_all_workers(self) -> None:
        processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + self.shutdown_timeout_s
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        self._processes.clear()

    def _inventory(self) -> list[dict[str, Any]]:
        return [{
            'worker_id': worker_id,
            'generation': self._generations[worker_id],
            'process_id': process.pid,
            'running': process.poll() is None,
        } for worker_id, process in sorted(self._processes.items())]

    def _send_control(self, control: Any, message_type: str,
                      **values: Any) -> None:
        control.send_multipart([
            encode_header(
                message_type,
                deployment_id=self.deployment_id,
                supervisor_id=self.supervisor_id,
                node_id=self.node_id,
                **values,
            ),
            empty_payload(),
        ])


def run_model_worker(config_path: str,
                     ckpt_path: str,
                     backend_endpoint: str,
                     worker_id: str,
                     supervisor_id: str,
                     node_id: str,
                     generation: int,
                     deployment_id: str,
                     device: str,
                     cfg_options: Mapping[str, Any] | None = None,
                     parent_pid: int | None = None,
                     heartbeat_interval_s: float = 5.0) -> None:
    """Load one model replica and serve independent prediction requests."""
    import zmq
    from mmengine import Config

    _configure_child_lifetime(parent_pid)
    from fluxvla.engines.utils.torch_utils import \
        configure_inference_attention_defaults
    configure_inference_attention_defaults()

    context = zmq.Context()
    backend = context.socket(zmq.DEALER)
    identity = f'{worker_id}:{generation}'.encode()
    backend.setsockopt(zmq.IDENTITY, identity)
    backend.setsockopt(zmq.LINGER, 0)
    backend.connect(_endpoint(backend_endpoint, 'backend_endpoint'))

    common = {
        'deployment_id': deployment_id,
        'worker_id': worker_id,
        'supervisor_id': supervisor_id,
        'node_id': node_id,
        'device': device,
        'generation': generation,
    }
    backend.send_multipart([
        encode_header('register_worker', **common),
        empty_payload(),
    ])
    backend.send_multipart([
        encode_header('loading', **common),
        empty_payload(),
    ])
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(dict(cfg_options))
    policy = build_policy_from_config(cfg, ckpt_path=ckpt_path, device=device)
    backend.send_multipart([
        encode_header('ready', **common),
        empty_payload(),
    ])
    print(
        f'[FluxVLA] model worker ready id={worker_id} device={device} '
        f'generation={generation}',
        flush=True)
    poller = zmq.Poller()
    poller.register(backend, zmq.POLLIN)
    running = True
    next_heartbeat = time.monotonic() + heartbeat_interval_s
    while running:
        timeout_ms = max(1, int((next_heartbeat - time.monotonic()) * 1000))
        events = dict(poller.poll(timeout=min(timeout_ms, 1000)))
        if backend in events:
            frames = backend.recv_multipart()
            header = decode_header(frames[0])
            payload = frames[1] if len(frames) > 1 else empty_payload()
            if header['type'] == 'predict':
                request_id = _required_string(header, 'request_id')
                try:
                    observation = decode_frame(payload)
                    if not isinstance(observation, Mapping):
                        raise TypeError('prediction payload must be a mapping')
                    actions, inference_time_s = policy.predict(
                        observation,
                        unnorm_key=str(header.get('unnorm_key', '')),
                        seed=_required_int(header, 'seed'),
                    )
                    actions = np.asarray(actions, dtype=np.float32)
                    backend.send_multipart([
                        encode_header(
                            'predict_result',
                            **common,
                            request_id=request_id,
                            action_horizon=int(actions.shape[0]),
                            action_dim=int(actions.shape[1]),
                            denormalized=True,
                            inference_time_s=float(inference_time_s),
                        ),
                        encode_frame({'actions': actions}),
                    ])
                except Exception as exc:
                    backend.send_multipart([
                        encode_header(
                            'predict_error',
                            **common,
                            request_id=request_id,
                            error=f'{type(exc).__name__}: {exc}',
                        ),
                        empty_payload(),
                    ])
            elif header['type'] in {'shutdown_worker', 'drain_worker'}:
                backend.send_multipart([
                    encode_header('drained', **common),
                    empty_payload(),
                ])
                running = False
        if time.monotonic() >= next_heartbeat:
            backend.send_multipart([
                encode_header('heartbeat', **common),
                empty_payload(),
            ])
            next_heartbeat = time.monotonic() + heartbeat_interval_s
    backend.close(linger=0)
    context.term()


def build_evaluation_reporter_from_config(
        cfg: Any,
        ckpt_path: str,
        config_path: str | os.PathLike | None = None) -> Any:
    """Build the authoritative reporter without loading a model replica."""
    themis_cfg = require_mapping(config_get(cfg, 'themis'), 'config.themis')
    server_cfg = get_server_config(themis_cfg)
    reporting_cfg = dict(
        require_mapping(
            server_cfg.get('evaluation_reporting', {}),
            'themis.server.evaluation_reporting'))
    if not bool(reporting_cfg.get('enabled', False)):
        return None
    section_name = server_cfg.get('dataset_section')
    if section_name not in {'eval', 'inference'}:
        raise ValueError('themis.server.dataset_section must be `eval` or '
                         '`inference`')
    section_cfg = require_mapping(
        config_get(cfg, section_name), f'config.{section_name}')
    resolved_ckpt = resolve_checkpoint_path(
        ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    resolved_config_path = resolve_report_config_path(cfg, config_path)
    result_root = resolve_report_result_root(
        reporting_cfg.get('result_output_dir'), resolved_ckpt)
    reporter_eval_source = config_get(cfg, 'eval', section_cfg)
    reporter_eval_config = copy.deepcopy(
        dict(require_mapping(reporter_eval_source, 'config.eval metadata')))
    runner_cfg = themis_cfg.get('runner')
    if isinstance(runner_cfg, Mapping):
        if 'task_ids' in runner_cfg:
            reporter_eval_config.setdefault('task_ids', runner_cfg['task_ids'])
        if 'episodes_per_task' in runner_cfg:
            reporter_eval_config.setdefault('num_trials_per_task',
                                            runner_cfg['episodes_per_task'])
        if 'episodes_per_task_overrides' in runner_cfg:
            reporter_eval_config.setdefault(
                'num_trials_per_task_overrides',
                runner_cfg['episodes_per_task_overrides'])
        if 'run_name' in runner_cfg:
            reporter_eval_config.setdefault('run_name', runner_cfg['run_name'])
    reporter_eval_config.setdefault('dataset_section', section_name)
    reporter_eval_config.setdefault('result_gpu_id',
                                    reporting_cfg.get('result_gpu_id', 0))

    from .evaluation_reporter import FluxVLAEvaluationReporter
    return FluxVLAEvaluationReporter(
        result_root=result_root,
        config_path=resolved_config_path,
        ckpt_path=resolved_ckpt,
        eval_config=reporter_eval_config,
        logger=None,
        feishu=reporting_cfg.get('feishu'),
        report_kind=reporting_cfg.get('report_kind'),
    )


def load_server_config(config_path: str,
                       cfg_options: Mapping[str, Any] | None = None) -> Any:
    from mmengine import Config
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(dict(cfg_options))
    return cfg


def server_fingerprints(config_path: str, ckpt_path: str) -> tuple[str, str]:
    config_bytes = Path(config_path).read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    checkpoint = Path(ckpt_path).expanduser().resolve()
    stat = checkpoint.stat()
    checkpoint_value = f'{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}'
    checkpoint_hash = hashlib.sha256(checkpoint_value.encode()).hexdigest()
    return config_hash, checkpoint_hash


def launch_server_task(*,
                       config_path: str,
                       ckpt_path: str,
                       devices: Sequence[str],
                       frontend_bind: str,
                       backend_bind: str,
                       control_bind: str,
                       backend_endpoint: str,
                       control_endpoint: str,
                       advertised_endpoint: str,
                       manifest_path: str | None,
                       minimum_ready_workers: int,
                       max_pending_requests: int,
                       request_timeout_s: float,
                       startup_timeout_s: float,
                       heartbeat_timeout_s: float,
                       exit_after_run: bool,
                       cfg_options: Mapping[str, Any] | None = None,
                       node_rank: int | None = None,
                       world_size: int | None = None,
                       deployment_id: str | None = None) -> None:
    """Run one torchrun rank as a node-local server supervisor."""
    node_rank = int(
        os.environ.get('RANK', '0') if node_rank is None else node_rank)
    world_size = int(
        os.environ.get('WORLD_SIZE', '1') if world_size is None else world_size
    )
    if node_rank < 0 or node_rank >= world_size:
        raise ValueError('node_rank must be in [0, world_size)')
    node_id = f'{network_socket.gethostname()}-rank-{node_rank}'
    if deployment_id is None:
        deployment_id = (
            os.environ.get('FLUXVLA_DEPLOYMENT_ID')
            or os.environ.get('TORCHELASTIC_RUN_ID'))
    if not deployment_id:
        stable = (f'{os.environ.get("MASTER_ADDR", "127.0.0.1")}:'
                  f'{os.environ.get("MASTER_PORT", "29500")}:'
                  f'{Path(config_path).resolve()}:{Path(ckpt_path).resolve()}')
        deployment_id = hashlib.sha256(stable.encode()).hexdigest()[:32]

    coordinator_process = None
    if node_rank == 0:
        context = multiprocessing.get_context('spawn')
        coordinator_process = context.Process(
            target=_coordinator_process_main,
            kwargs={
                'config_path': config_path,
                'ckpt_path': ckpt_path,
                'cfg_options': dict(cfg_options or {}),
                'frontend_bind': frontend_bind,
                'backend_bind': backend_bind,
                'control_bind': control_bind,
                'advertised_endpoint': advertised_endpoint,
                'manifest_path': manifest_path,
                'minimum_ready_workers': minimum_ready_workers,
                'expected_workers': world_size * len(devices),
                'max_pending_requests': max_pending_requests,
                'request_timeout_s': request_timeout_s,
                'startup_timeout_s': startup_timeout_s,
                'heartbeat_timeout_s': heartbeat_timeout_s,
                'deployment_id': deployment_id,
                'exit_after_run': exit_after_run,
                'parent_pid': os.getpid(),
            },
            name='fluxvla-zmq-coordinator',
        )
        coordinator_process.start()

    supervisor = ServerSupervisor(
        control_endpoint=control_endpoint,
        backend_endpoint=backend_endpoint,
        config_path=config_path,
        ckpt_path=ckpt_path,
        devices=devices,
        supervisor_id=f'server-supervisor-{node_id}',
        node_id=node_id,
        deployment_id=deployment_id,
        cfg_options=cfg_options,
        coordinator_process=coordinator_process,
    )
    try:
        supervisor.run()
    finally:
        if coordinator_process is not None:
            coordinator_process.join(timeout=15.0)
            if coordinator_process.is_alive():
                coordinator_process.terminate()
                coordinator_process.join(timeout=5.0)
            if coordinator_process.exitcode not in {0, -signal.SIGTERM}:
                raise RuntimeError('server coordinator exited with code '
                                   f'{coordinator_process.exitcode}')


def _coordinator_process_main(**values: Any) -> None:
    config_path = values.pop('config_path')
    ckpt_path = values.pop('ckpt_path')
    cfg_options = values.pop('cfg_options')
    parent_pid = values.pop('parent_pid')
    _configure_child_lifetime(parent_pid)
    cfg = load_server_config(config_path, cfg_options)
    reporter = build_evaluation_reporter_from_config(
        cfg, ckpt_path=ckpt_path, config_path=config_path)
    config_hash, checkpoint_hash = server_fingerprints(config_path, ckpt_path)
    coordinator = StatelessZMQCoordinator(
        evaluation_reporter=reporter,
        config_fingerprint=config_hash,
        checkpoint_fingerprint=checkpoint_hash,
        **values,
    )
    coordinator.run()


def _configure_child_lifetime(parent_pid: int | None) -> None:
    if sys.platform != 'linux' or parent_pid is None:
        return
    import ctypes
    expected = int(parent_pid)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected:
        os.kill(os.getpid(), signal.SIGKILL)


def _terminate_process(process: subprocess.Popen, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _endpoint(value: Any, name: str) -> str:
    value = _nonempty(value, name)
    if not value.startswith(('tcp://', 'ipc://', 'inproc://')):
        raise ValueError(f'{name} must be a complete ZMQ endpoint')
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f'{name} must be a positive number')
    return float(value)


def _required_string(header: Mapping[str, Any], name: str) -> str:
    return _nonempty(header.get(name), f'header.{name}')


def _required_int(header: Mapping[str, Any],
                  name: str,
                  minimum: int | None = None) -> int:
    value = header.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'header.{name} must be an integer')
    if minimum is not None and value < minimum:
        raise ValueError(f'header.{name} must be >= {minimum}')
    return int(value)


def _header_string(header: Any, name: str) -> str:
    if not isinstance(header, Mapping):
        return ''
    value = header.get(name, '')
    return value if isinstance(value, str) else ''


def threading_available_for_signals() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


def _parse_worker_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--backend-endpoint', required=True)
    parser.add_argument('--worker-id', required=True)
    parser.add_argument('--supervisor-id', required=True)
    parser.add_argument('--node-id', required=True)
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--deployment-id', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--parent-pid', type=int, default=None)
    parser.add_argument('--cfg-options-json', default='{}')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != 'model-worker':
        raise SystemExit('distributed_server only exposes model-worker '
                         'internally; use scripts/distributed_zmq_server.py')
    args = _parse_worker_args(argv[1:])
    run_model_worker(
        config_path=args.config,
        ckpt_path=args.ckpt_path,
        backend_endpoint=args.backend_endpoint,
        worker_id=args.worker_id,
        supervisor_id=args.supervisor_id,
        node_id=args.node_id,
        generation=args.generation,
        deployment_id=args.deployment_id,
        device=args.device,
        cfg_options=json.loads(args.cfg_options_json),
        parent_pid=args.parent_pid,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

__all__ = [
    'ServerSupervisor',
    'StatelessZMQCoordinator',
    'build_evaluation_reporter_from_config',
    'launch_server_task',
    'load_server_config',
    'run_model_worker',
    'server_fingerprints',
]
