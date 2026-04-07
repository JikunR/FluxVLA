# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect
import sys

# Workaround for PyTorch 2.6 + Python 3.10 inspect.getsourcefile crash
# when non-standard frames (e.g. from mros C extensions) are on the stack.
_original_getsourcefile = inspect.getsourcefile


def _safe_getsourcefile(obj):
    try:
        return _original_getsourcefile(obj)
    except TypeError:
        return None


inspect.getsourcefile = _safe_getsourcefile

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
