"""Fix HuggingFace metadata in parquet files: replace _type 'List' with 'Sequence'.

Usage:
    python scripts/fix_parquet_metadata.py /path/to/dataset/root
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def fix_parquet_file(path: Path) -> bool:
    table = pq.read_table(path)
    meta = table.schema.metadata
    if meta is None or b'huggingface' not in meta:
        return False

    hf_raw = meta[b'huggingface'].decode('utf-8')
    if '"_type": "List"' not in hf_raw:
        return False

    hf_fixed = hf_raw.replace('"_type": "List"', '"_type": "Sequence"')
    new_meta = dict(meta)
    new_meta[b'huggingface'] = hf_fixed.encode('utf-8')
    new_table = table.replace_schema_metadata(new_meta)
    pq.write_table(new_table, path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, help='Dataset root path')
    args = parser.parse_args()

    root = Path(args.data_root)
    data_dir = root / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f'{data_dir} not found')

    files = sorted(data_dir.rglob('*.parquet'))
    print(f'Found {len(files)} parquet files')

    fixed = 0
    for f in files:
        if fix_parquet_file(f):
            fixed += 1
    print(f'Fixed {fixed}/{len(files)} files')


if __name__ == '__main__':
    main()
