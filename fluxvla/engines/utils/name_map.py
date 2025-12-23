import torch
from torch.distributed.fsdp import StateDictType


def str_to_dtype(s: str):
    mapping = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'bool': torch.bool
    }
    s = s.lower()
    if s in mapping:
        return mapping[s]
    else:
        raise ValueError(f'Unsupported dtype string: {s}')


def state_dict_type_map(s: str):
    if s == 'full_state_dict':
        return StateDictType.FULL_STATE_DICT
    if s == 'local_state_dict':
        return StateDictType.LOCAL_STATE_DICT
    if s == 'sharded_state_dict':
        return StateDictType.SHARDED_STATE_DICT
    assert False, f'Unsupported state dict type: {s}'
