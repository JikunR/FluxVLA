import argparse

from mmengine import Config, DictAction

from fluxvla.engines import build_runner_from_cfg, initialize_overwatch

overwatch = initialize_overwatch(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model with the given configuration.')
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
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=  # noqa: E251
        'override some settings in the used config, the key-value pair in xxx=yyy format'  # noqa: E501
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.eval.cfg = cfg
    cfg.eval.ckpt_path = args.ckpt_path
    if hasattr(cfg.eval,
               'processor') and not hasattr(cfg.eval.processor, 'model_path'):
        cfg.eval.processor.model_path = args.ckpt_path
    eval_runner = build_runner_from_cfg(cfg.eval)
    eval_runner.run_setup()
    eval_runner.run()
