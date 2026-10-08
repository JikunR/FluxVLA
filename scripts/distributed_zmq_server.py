#!/usr/bin/env python
"""Launch a single- or multi-node stateless FluxVLA ZMQ server."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_FRONTEND_BIND = 'tcp://0.0.0.0:15555'
DEFAULT_MODEL_WORKER_BIND = 'tcp://0.0.0.0:15556'
DEFAULT_SUPERVISOR_BIND = 'tcp://0.0.0.0:15557'
DEFAULT_MINIMUM_READY_WORKERS = 1
DEFAULT_MAX_PENDING_REQUESTS = 128
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_STARTUP_TIMEOUT_S = 900.0
DEFAULT_HEARTBEAT_TIMEOUT_S = 30.0


def parse_args(argv=None):
    from mmengine import DictAction

    parser = argparse.ArgumentParser(
        description='Launch the stateless FluxVLA ZMQ inference service.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', default=None)
    worker_group = parser.add_mutually_exclusive_group()
    worker_group.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of local model workers on every node.')
    worker_group.add_argument(
        '--devices',
        type=_parse_devices,
        default=None,
        help='Override auto-detected local GPUs, for example 0,1.')
    networking = parser.add_argument_group('advanced networking')
    networking.add_argument(
        '--frontend-bind',
        default=DEFAULT_FRONTEND_BIND,
        help=f'Public client-facing bind address (default: '
        f'{DEFAULT_FRONTEND_BIND}).')
    networking.add_argument(
        '--backend-bind',
        default=DEFAULT_MODEL_WORKER_BIND,
        help=f'Internal model-worker bind address (default: '
        f'{DEFAULT_MODEL_WORKER_BIND}).')
    networking.add_argument(
        '--control-bind',
        default=DEFAULT_SUPERVISOR_BIND,
        help=f'Internal supervisor bind address (default: '
        f'{DEFAULT_SUPERVISOR_BIND}).')
    networking.add_argument(
        '--backend-endpoint',
        default=None,
        help='Address model workers connect to; defaults to the backend port '
        'on MASTER_ADDR.')
    networking.add_argument(
        '--control-endpoint',
        default=None,
        help='Address supervisors connect to; defaults to the control port '
        'on MASTER_ADDR.')
    networking.add_argument(
        '--advertise-endpoint',
        default=None,
        help='Public client endpoint; defaults to the frontend port on '
        'MASTER_ADDR.')
    networking.add_argument(
        '--server-manifest',
        default=None,
        help='Optional shared JSON manifest written for client discovery.')
    reliability = parser.add_argument_group('advanced reliability')
    reliability.add_argument(
        '--minimum-ready-workers',
        type=int,
        default=DEFAULT_MINIMUM_READY_WORKERS)
    reliability.add_argument(
        '--max-pending-requests',
        type=int,
        default=DEFAULT_MAX_PENDING_REQUESTS)
    reliability.add_argument(
        '--request-timeout-s', type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    reliability.add_argument(
        '--startup-timeout-s', type=float, default=DEFAULT_STARTUP_TIMEOUT_S)
    reliability.add_argument(
        '--heartbeat-timeout-s',
        type=float,
        default=DEFAULT_HEARTBEAT_TIMEOUT_S)
    parser.add_argument('--exit-after-run', action='store_true')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None)
    return parser.parse_args(argv)


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(',') if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError('--devices cannot be empty')
    if len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError('--devices cannot contain duplicates')
    return devices


def _connect_endpoint(bind: str, host: str) -> str:
    parsed = urlsplit(bind)
    if parsed.scheme != 'tcp' or parsed.port is None:
        raise ValueError('distributed endpoints must use tcp://host:port')
    return 'tcp://{}:{}'.format(host, parsed.port)


def main(argv=None) -> int:
    from mmengine import Config

    from fluxvla.engines.runners.serving.distributed_server import \
        launch_server_task
    from fluxvla.engines.runners.serving.policy import (
        config_get, get_server_config, require_mapping,
        resolve_checkpoint_path, resolve_inference_devices)

    args = parse_args(argv)
    config_path = str(Path(args.config).expanduser().resolve(strict=True))
    cfg = Config.fromfile(config_path)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    themis_cfg = require_mapping(config_get(cfg, 'themis'), 'config.themis')
    server_cfg = get_server_config(themis_cfg)
    section_name = server_cfg.get('dataset_section', 'eval')
    section_cfg = require_mapping(
        config_get(cfg, section_name), f'config.{section_name}')
    checkpoint = resolve_checkpoint_path(
        args.ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    devices = resolve_inference_devices(
        worker_devices=args.devices, num_workers=args.num_workers)
    rank = os.environ.get('RANK', '0')
    device_list = ','.join(devices)
    print(
        f'[FluxVLA] rank={rank} starting {len(devices)} local model workers '
        f'on {device_list}',
        flush=True)
    frontend_bind = args.frontend_bind
    backend_bind = args.backend_bind
    control_bind = args.control_bind
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    backend_endpoint = (
        args.backend_endpoint or _connect_endpoint(backend_bind, master_addr))
    control_endpoint = (
        args.control_endpoint or _connect_endpoint(control_bind, master_addr))
    advertised_endpoint = (
        args.advertise_endpoint
        or _connect_endpoint(frontend_bind, master_addr))
    manifest_path = args.server_manifest

    launch_server_task(
        config_path=config_path,
        ckpt_path=str(checkpoint),
        devices=devices,
        frontend_bind=frontend_bind,
        backend_bind=backend_bind,
        control_bind=control_bind,
        backend_endpoint=backend_endpoint,
        control_endpoint=control_endpoint,
        advertised_endpoint=advertised_endpoint,
        manifest_path=manifest_path,
        minimum_ready_workers=args.minimum_ready_workers,
        max_pending_requests=args.max_pending_requests,
        request_timeout_s=args.request_timeout_s,
        startup_timeout_s=args.startup_timeout_s,
        heartbeat_timeout_s=args.heartbeat_timeout_s,
        exit_after_run=args.exit_after_run,
        cfg_options=args.cfg_options,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
