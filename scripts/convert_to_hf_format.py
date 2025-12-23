"""
convert_openvla_weights_to_hf.py

Utility script for converting full OpenVLA VLA weights
(from this repository, in the default "Prismatic" format) to
the HuggingFace "AutoClasses" (e.g., those defined in
`prismatic.extern.hf_*`) for "native" use in `transformers``
via `trust_remote_code = True`.

Theoretically, these changes should be fully compatible
with directly merging the models into `transformers` down the
line, with first-class support.

Usage:
    python vla-scripts/extern/convert_openvla_weights_to_hf.py \
        --openvla_model_path_or_id <PATH TO PRISMATIC TRAINING RUN DIR> \
        --output_hf_model_local_path <OUTPUT DIR FOR CONVERTED CHECKPOINT>
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from mmengine import Config
from timm.models.vision_transformer import LayerScale
from transformers import AutoTokenizer

from fluxvla.engines.utils.hf_utils import (OpenVLAConfig,
                                            OpenVLAForActionPrediction,
                                            PrismaticImageProcessor,
                                            PrismaticProcessor)


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
        '--model-path-or-id',
        type=str,
        required=True,
        help='Path to the model directory or ID.',
    )
    parser.add_argument(
        '--output-hf-model-local-path',
        type=str,
        required=True,
        help='Local path to save the converted HuggingFace model.',
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        required=True,
        help='HuggingFace token for authentication.',
    )
    args = parser.parse_args()
    return args


# HF Transformers overwrites parameters with names containing
# `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109  # noqa: E501
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960  # noqa: E501
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Conversion Constants ===
PROJECTOR_KEY_MAPPING = {
    'projector.0.weight': 'projector.fc1.weight',
    'projector.0.bias': 'projector.fc1.bias',
    'projector.2.weight': 'projector.fc2.weight',
    'projector.2.bias': 'projector.fc2.bias',
    'projector.4.weight': 'projector.fc3.weight',
    'projector.4.bias': 'projector.fc3.bias',
}


def remap_state_dicts_for_hf(
    prismatic_vision_backbone_state_dict: Dict[str, torch.Tensor],
    projector_state_dict: Dict[str, torch.Tensor],
    llm_backbone_state_dict: Dict[str, torch.Tensor],
    use_fused_vision_backbone: bool = False,
) -> Dict[str, torch.Tensor]:
    """Remap the state dictionaries from the Prismatic format to the
        HuggingFace format.
    Args:
        prismatic_vision_backbone_state_dict (Dict[str, torch.Tensor]):
            The state dict for the vision backbone.
        projector_state_dict (Dict[str, torch.Tensor]): The state dict for
            the projector.
        llm_backbone_state_dict (Dict[str, torch.Tensor]): The state dict
            for the LLM backbone.
        use_fused_vision_backbone (bool): Whether to use the fused
            vision backbone.
    Returns:
        Dict[str, torch.Tensor]: The remapped state dict for HuggingFace.
    """
    hf_state_dict = {}

    # Iterate through Projector =>> use `PROJECTOR_KEY_MAPPING`
    for key, value in projector_state_dict.items():
        hf_state_dict[PROJECTOR_KEY_MAPPING[key]] = value

    # Iterate through LLM Backbone =>> replace `llm.` with `language_model.`
    for key, value in llm_backbone_state_dict.items():
        hf_state_dict[key.replace('llm.', 'language_model.')] = value

    # Iterate through Vision Backbone =>> add "vision_backbone." prefix
    if not use_fused_vision_backbone:
        for key, value in prismatic_vision_backbone_state_dict.items():
            hf_state_dict[key.replace(
                'featurizer.',
                'vision_backbone.featurizer.')] = value  # noqa: E501
    else:
        # Note =>> Assumes that backbones are always DINO + SigLIP...
        for key, value in prismatic_vision_backbone_state_dict.items():
            if key.startswith('dino_featurizer'):
                if key.endswith('.gamma'):
                    # Handle `LayerScale gamma` =>> DINOv2 only!
                    key = key.replace('.gamma', '.scale_factor')
                hf_state_dict[key.replace(
                    'dino_featurizer.', 'vision_backbone.featurizer.')] = value
            elif key.startswith('siglip_featurizer'):
                hf_state_dict[key.replace(
                    'siglip_featurizer.',
                    'vision_backbone.fused_featurizer.')] = value

    return hf_state_dict


def convert_openvla_weights_to_hf(args) -> None:
    print(
        f'[*] Converting OpenVLA Model `{args.model_path_or_id}` to HF Transformers Format'  # noqa: E501
    )
    torch.set_default_dtype(torch.bfloat16)
    config = Config.fromfile(args.config)

    # Get `config.json`, 'dataset_statistics.json' and
    # `checkpoint_pt` -- mirrors logic in `prismatic.models.load.py`
    if os.path.isdir(args.model_path_or_id):
        print(
            f'[*] Loading from Local Path `{(run_dir := Path(args.model_path_or_id))}`'  # noqa: E501
        )
        config_json = os.path.join(run_dir, 'config.json')
        checkpoint_pt = os.path.join(run_dir, 'checkpoints',
                                     'latest-checkpoint.pt')  # noqa: E501
        dataset_statistics_json = run_dir / 'dataset_statistics.json'

        assert os.path.exists(config_json),\
            f'Missing `config.json` for `{run_dir = }`'
        assert os.path.exists(checkpoint_pt),\
            f'Missing checkpoint for `{run_dir = }`'  # noqa: E501
        assert os.path.exists(dataset_statistics_json),\
            f'Missing `dataset_statistics.json` for `{run_dir = }`'
    else:
        print(
            f'[*] Downloading Prismatic Checkpoint from HF Hub :: `TRI-ML/{args.model_path_or_id}`'  # noqa: E501
        )
        config_json = hf_hub_download(
            'openvla/openvla-dev',
            f'{args.model_path_or_id}/config.json')  # noqa: E501
        checkpoint_pt = hf_hub_download(
            'openvla/openvla-dev',
            f'{args.model_path_or_id}/checkpoints/latest-checkpoint.pt'
        )  # noqa: E501
        dataset_statistics_json = hf_hub_download(
            'openvla/openvla-dev',
            f'{args.model_path_or_id}/dataset_statistics.json')  # noqa: E501

    # Load Normalization Statistics
    with open(dataset_statistics_json, 'r') as f:
        norm_stats = json.load(f)

    # Create HF OpenVLAConfig (`transformers.PretrainedConfig`)
    hf_config = OpenVLAConfig(
        vision_backbone_id=config['model']['vision_backbone']
        ['vision_backbone_id'],
        llm_backbone_id=config['model']['llm_backbone']['llm_backbone_id'],
        arch_specifier=config['model']['arch_specifier'],
        image_resize_strategy=config['model']['vision_backbone']
        ['image_resize_strategy'],
        llm_max_length=config['model']['llm_backbone']['llm_max_length'],
        torch_dtype=torch.bfloat16,
        norm_stats=norm_stats,
    )

    # Instantiate & Add Pad to Tokenizer =>> following
    # `prismatic.models.materialize.get_llm_backbone_and_tokenizer`
    #   TODO (siddk) :: Implement batched generation -- in which
    # case this should set `padding_side = "left"`!
    print('[*] Instantiating and Patching Tokenizer, LLM Config')
    tokenizer = AutoTokenizer.from_pretrained(
        hf_config.hf_llm_id,
        model_max_length=hf_config.llm_max_length,
        token=args.hf_token,
        padding_side='right')
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.init_kwargs.pop(
        'add_prefix_space',
        None)  # Pop to prevent unnecessary warning on reload...
    assert tokenizer.pad_token_id == hf_config.pad_token_id, 'Incorrect Pad Token ID!'  # noqa: E501
    assert len(
        tokenizer
    ) > hf_config.text_config.vocab_size, 'Tokenizer vocabulary must be larger than LLM vocabulary!'  # noqa: E501

    # Patch LLM Config in `hf_config` with vocab_size
    # (+ `hf_config.pad_to_multiple_of`), pad_token_id + validate
    hf_config.text_config.vocab_size += hf_config.pad_to_multiple_of
    hf_config.text_config.pad_token_id = hf_config.pad_token_id
    hf_config.text_config.torch_dtype = torch.bfloat16
    assert hf_config.text_config.use_cache, 'LLM config `use_cache` should be True for inference (set default)!'  # noqa: E501

    # Create Vision Backbone & Transform =>> following
    # `prismatic.models.materialize.get_vision_backbone_and_transform`
    #   =>> Deviates a bit from existing code; as such, explicitly
    # tested in `tests/test_image_transforms.py`
    print(
        '[*] Loading TIMM Vision Backbone(s) and Image Transform(s) =>> Initializing PrismaticImageProcessor'  # noqa: E501
    )
    input_sizes, interpolations, means, stds = [], [], [], []
    for idx, timm_model_id in enumerate(hf_config.timm_model_ids):
        if idx == 0:
            timm_vision_backbone = timm.create_model(
                timm_model_id,
                pretrained=True,
                num_classes=0,
                img_size=hf_config.image_sizes[idx],
                act_layer=hf_config.timm_override_act_layers[idx],
                pretrained_cfg={
                    'file':
                    '/limx_jeff/tos/limx_mani_checkpoints/open_source/huggingface/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
                })
        if idx == 1:
            timm_vision_backbone = timm.create_model(
                timm_model_id,
                pretrained=True,
                num_classes=0,
                img_size=hf_config.image_sizes[idx],
                act_layer=hf_config.timm_override_act_layers[idx],
                pretrained_cfg={
                    'file':
                    '/limx_jeff/tos/limx_mani_checkpoints/open_source/huggingface/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501
                })
        # timm_vision_backbone = timm.create_model(
        #     timm_model_id,
        #     pretrained=True,
        #     num_classes=0,
        #     img_size=hf_config.image_sizes[idx],
        #     act_layer=hf_config.timm_override_act_layers[idx],
        # )

        # Get Per-Backbone Image Processing
        data_cfg = timm.data.resolve_model_data_config(timm_vision_backbone)
        input_sizes.append(
            (3, hf_config.image_sizes[idx], hf_config.image_sizes[idx]))
        interpolations.append(data_cfg['interpolation'])
        means.append(data_cfg['mean'])
        stds.append(data_cfg['std'])

        # Patch `LayerScale` because of HF annoying `fix_key` overwrite...
        for module in timm_vision_backbone.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

    # Create PrismaticImageProcessor (`transformers.ImageProcessingMixin`)
    hf_image_processor = PrismaticImageProcessor(
        use_fused_vision_backbone=hf_config.use_fused_vision_backbone,
        image_resize_strategy=hf_config.image_resize_strategy,
        input_sizes=input_sizes,
        interpolations=interpolations,
        means=means,
        stds=stds,
    )

    # Create top-level PrismaticProcessor (`transformers.ProcessorMixin`  # noqa: E501
    # =>> enables registry w/ AutoProcessor)
    print(
        '[*] Creating PrismaticProcessor Instance from Tokenizer and PrismaticImageProcessor'  # noqa: E501
    )
    hf_processor = PrismaticProcessor(
        image_processor=hf_image_processor, tokenizer=tokenizer)

    # Load Prismatic Model State Dictionary (in preparation for conversion)
    print('[*] Loading Prismatic VLM State Dictionary from Checkpoint')
    model_state_dict = torch.load(checkpoint_pt, map_location='cpu')['model']
    assert ('downsampler' not in model_state_dict) or (len(
        model_state_dict['downsampler']) == 0), 'Downsampler?'
    assert all([
        k in model_state_dict
        for k in ['vision_backbone', 'projector', 'llm_backbone']
    ]), 'Missing keys!'

    # Convert
    print('[*] Running Conversion')
    converted_state_dict = remap_state_dicts_for_hf(
        model_state_dict['vision_backbone'],
        model_state_dict['projector'],
        model_state_dict['llm_backbone'],
        use_fused_vision_backbone=hf_config.
        use_fused_vision_backbone,  # noqa: E501
    )

    # Create PrismaticForConditionalGeneration =>> Note that we can't
    # initialize on `meta` device because TIMM
    print(
        '[*] Building (Randomly Initialized) Model =>> OpenVLAForActionPrediction'  # noqa: E501
    )
    hf_model = OpenVLAForActionPrediction(hf_config)
    hf_model.load_state_dict(converted_state_dict, strict=True, assign=True)

    # Cast Model to BF16 before Saving
    hf_model.to(torch.bfloat16)

    # Save Pretrained Versions to Local Path
    print('[*] Saving Model & Processor to Local Path')
    hf_model.save_pretrained(
        args.output_hf_model_local_path, max_shard_size='7GB')
    hf_image_processor.save_pretrained(args.output_hf_model_local_path)
    hf_processor.save_pretrained(args.output_hf_model_local_path)

    os.makedirs(args.output_hf_model_local_path, exist_ok=True)

    # Copy `dataset_statistics.json` File to Converted Checkpoint Directory
    output_dataset_statistics_json = os.path.join(
        args.output_hf_model_local_path, 'dataset_statistics.json')
    shutil.copyfile(dataset_statistics_json, output_dataset_statistics_json)

    print(
        f'[*] Saving Complete! Saved converted checkpoint to: {args.output_hf_model_local_path}'  # noqa: E501
    )

    ################################################################
    # Optional: Push Model to Hugging Face Hub
    ################################################################

    # # Register AutoClasses
    # OpenVLAConfig.register_for_auto_class()
    # PrismaticImageProcessor.register_for_auto_class("AutoImageProcessor")
    # PrismaticProcessor.register_for_auto_class("AutoProcessor")
    # OpenVLAForActionPrediction.register_for_auto_class("AutoModelForVision2Seq")

    # # Push to HF Hub
    # print("[*] Pushing Model & Processor to HF Hub")
    # hf_config.push_to_hub(cfg.output_hf_model_hub_path)
    # hf_model.push_to_hub(cfg.output_hf_model_hub_path, max_shard_size="7GB")
    # hf_image_processor.push_to_hub(cfg.output_hf_model_hub_path)
    # hf_processor.push_to_hub(cfg.output_hf_model_hub_path)


if __name__ == '__main__':
    args = parse_args()
    convert_openvla_weights_to_hf(args)
