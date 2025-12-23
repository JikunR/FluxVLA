import gc
import unittest

import numpy as np
import pytest
import torch

from fluxvla.engines import build_vision_backbone_from_cfg


class TestHFCausalVisionBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='DinoSigLIPViTBackbone',
            vision_backbone_id='dinosiglip-vit-so-224px',
            dino_config=dict(
                model_id='dino',
                file=  # noqa: E251
                '/limx/tos/limx_mani_checkpoints/open_source/huggingface/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
            ),
            siglip_config=dict(
                model_id='siglip_224',
                file=  # noqa: E251
                '/limx/tos/limx_mani_checkpoints/open_source/huggingface/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501
            ))
        self.siglip_vit = build_vision_backbone_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_siglip_vit_forward(self):
        input_dino = torch.from_numpy(
            np.load(
                'test/data/models/vision_backbones/input_dino.npy')).cuda()

        input_siglip = torch.from_numpy(
            np.load(
                'test/data/models/vision_backbones/input_siglip.npy')).cuda()
        pixel_values = torch.cat([input_dino, input_siglip], dim=1)
        output = self.siglip_vit(pixel_values)
        expected_output = torch.from_numpy(
            np.load(
                'test/data/models/vision_backbones/output_dinosiglipvit.npy')
        ).cuda()
        assert torch.allclose(
            output[:, ::10, ::10], expected_output, rtol=1e-3, atol=1e-1)
