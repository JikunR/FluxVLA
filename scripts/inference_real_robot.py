import argparse

from mmengine import Config

from fluxvla.engines import build_runner_from_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='Path to the checkpoint file.')
    args = parser.parse_args()
    return args


def inference(args, cfg):
    runner = build_runner_from_cfg(cfg.inference_runner)
    print(runner)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.inference.ckpt_path = args.ckpt_path
    cfg.inference.cfg = cfg
    inference_runner = build_runner_from_cfg(cfg.inference)
    inference_runner.run_setup()

    inference_runner.run()
