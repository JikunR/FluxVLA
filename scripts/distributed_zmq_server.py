#!/usr/bin/env python
"""Launch a single- or multi-node stateless FluxVLA ZMQ server."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit


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
    parser.add_argument('--frontend-bind', default=None)
    parser.add_argument('--backend-bind', default=None)
    parser.add_argument('--control-bind', default=None)
    parser.add_argument('--backend-endpoint', default=None)
    parser.add_argument('--control-endpoint', default=None)
    parser.add_argument('--advertise-endpoint', default=None)
    parser.add_argument('--server-manifest', default=None)
    parser.add_argument('--minimum-ready-workers', type=int, default=None)
    parser.add_argument('--max-pending-requests', type=int, default=None)
    parser.add_argument('--request-timeout-s', type=float, default=None)
    parser.add_argument('--startup-timeout-s', type=float, default=None)
    parser.add_argument('--heartbeat-timeout-s', type=float, default=None)
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
    return f'tcp://{host}:{parsed.port}'


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
    section_name = server_cfg.get('dataset_section')
    section_cfg = require_mapping(
        config_get(cfg, section_name), f'config.{section_name}')
    checkpoint = resolve_checkpoint_path(
        args.ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    devices = resolve_inference_devices(
        server_cfg,
        worker_devices=args.devices,
        num_workers=args.num_workers,
    )
    print(
        f'[FluxVLA] rank={os.environ.get("RANK", "0")} starting '
        f'{len(devices)} local model workers on {",".join(devices)}',
        flush=True)
    distributed_cfg = require_mapping(
        server_cfg.get('distributed', {}), 'themis.server.distributed')
    workers_cfg = require_mapping(
        server_cfg.get('workers', {}), 'themis.server.workers')
    frontend_bind = (
        args.frontend_bind or distributed_cfg.get('frontend_bind')
        or 'tcp://0.0.0.0:5555')
    backend_bind = (
        args.backend_bind or distributed_cfg.get('model_worker_bind')
        or 'tcp://0.0.0.0:5556')
    control_bind = (
        args.control_bind or distributed_cfg.get('supervisor_bind')
        or 'tcp://0.0.0.0:5557')
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    backend_endpoint = (
        args.backend_endpoint or _connect_endpoint(backend_bind, master_addr))
    control_endpoint = (
        args.control_endpoint or _connect_endpoint(control_bind, master_addr))
    advertised_endpoint = (
        args.advertise_endpoint or distributed_cfg.get('advertise_endpoint')
        or _connect_endpoint(frontend_bind, master_addr))
    manifest_path = args.server_manifest or distributed_cfg.get(
        'manifest_path')

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
        minimum_ready_workers=(
            args.minimum_ready_workers if args.minimum_ready_workers
            is not None else int(server_cfg.get('minimum_ready_workers', 1))),
        max_pending_requests=(
            args.max_pending_requests if args.max_pending_requests is not None
            else int(server_cfg.get('max_pending_requests', 128))),
        request_timeout_s=(args.request_timeout_s
                           if args.request_timeout_s is not None else float(
                               workers_cfg.get('request_timeout_s', 120.0))),
        startup_timeout_s=(args.startup_timeout_s
                           if args.startup_timeout_s is not None else float(
                               workers_cfg.get('startup_timeout_s', 900.0))),
        heartbeat_timeout_s=(
            args.heartbeat_timeout_s if args.heartbeat_timeout_s is not None
            else float(workers_cfg.get('heartbeat_timeout_s', 30.0))),
        exit_after_run=args.exit_after_run,
        cfg_options=args.cfg_options,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
