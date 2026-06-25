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
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List

import torch

from fluxvla.models.backbones.vlms.wan22_loader import build_wan_text_encoder
from fluxvla.models.third_party_models.fastwam.modules.wan_video_text_encoder import \
    HuggingfaceTokenizer  # noqa: E501

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')


def _read_unique_prompts(dataset_dirs: List[str],
                         prompt_template: str) -> List[str]:
    prompts: List[str] = []
    seen = set()
    total = 0
    for dataset_dir in dataset_dirs:
        tasks_path = Path(dataset_dir) / 'meta' / 'tasks.jsonl'
        if not tasks_path.exists():
            raise FileNotFoundError(f'Missing tasks file: {tasks_path}')
        with tasks_path.open('r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if 'task' not in record:
                    raise KeyError(
                        f'Missing `task` field at {tasks_path}:{line_idx}')
                prompt = prompt_template.format(
                    task=str(record['task']),
                    task_description=str(record['task']),
                )
                total += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
    print(f'[INFO] Loaded {total} task rows, '
          f'deduplicated to {len(prompts)} prompts.')
    return prompts


def _atomic_save(payload: Dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / \
        f'.{output_path.name}.tmp.{uuid.uuid4().hex}'
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def main() -> None:
    repo_root = Path(os.environ.get('FLUXVLA_ROOT', Path.cwd())).expanduser()
    default_checkpoint_root = repo_root / 'checkpoints' / 'Wan2.2-TI2V-5B'
    default_cache_dir = Path(
        os.environ.get(
            'WAM_TEXT_CACHE_DIR',
            '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/text_embeds_cache/libero',  # noqa: E501
        ))

    parser = argparse.ArgumentParser(
        description='Precompute Wan/T5 text context caches for WAM training.')
    parser.add_argument(
        '--dataset-dir',
        action='append',
        required=True,
        help='Dataset root containing meta/tasks.jsonl. Repeatable.')
    parser.add_argument(
        '--cache-dir',
        default=str(default_cache_dir),
        help='Output cache directory.')
    parser.add_argument(
        '--checkpoint-root',
        default=str(default_checkpoint_root),
        help='Wan2.2 checkpoint root containing the T5 encoder shards.')
    parser.add_argument(
        '--tokenizer-path',
        default=None,
        help='Tokenizer path. Defaults to <checkpoint-root>/google/umt5-xxl.')
    parser.add_argument('--context-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--enc-id', default='wan22ti2v5b')
    parser.add_argument('--prompt-template', default=DEFAULT_PROMPT)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser()
    tokenizer_path = (
        Path(args.tokenizer_path).expanduser() if args.tokenizer_path
        is not None else checkpoint_root / 'google' / 'umt5-xxl')
    cache_dir = Path(args.cache_dir).expanduser()
    prompts = _read_unique_prompts(args.dataset_dir, args.prompt_template)
    if not prompts:
        print('[WARN] No prompts found; nothing to do.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_encoder = build_wan_text_encoder(
        checkpoint_root=str(checkpoint_root),
        device=device,
        torch_dtype=torch.bfloat16,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_path),
        seq_len=int(args.context_len),
        clean='whitespace',
    )

    written = 0
    skipped = 0
    with torch.no_grad():
        for start in range(0, len(prompts), int(args.batch_size)):
            batch = prompts[start:start + int(args.batch_size)]
            ids, mask = tokenizer(
                batch, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            for idx, prompt in enumerate(batch):
                hashed = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                cache_path = cache_dir / (
                    f'{hashed}.t5_len{int(args.context_len)}.'
                    f'{args.enc_id}.pt')
                if cache_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                payload = {
                    'context':
                    context[idx].detach().cpu().to(
                        torch.bfloat16).contiguous(),
                    'mask':
                    mask[idx].detach().cpu().to(torch.bool).contiguous(),
                }
                _atomic_save(payload, cache_path)
                written += 1

    print(f'[INFO] Wrote {written} caches to {cache_dir} '
          f'(skipped={skipped}).')


if __name__ == '__main__':
    main()
