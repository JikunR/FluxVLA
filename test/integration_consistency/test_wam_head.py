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

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mmengine import Config

from fluxvla.collators import WAMModeCollator
from fluxvla.engines import HEADS
from fluxvla.models.backbones.vlms.wan22_text_backbone import Wan22TextBackbone
from fluxvla.models.heads.wam_head import WAMHead
from fluxvla.models.third_party_models.fastwam.modules.mot import MoT
from fluxvla.models.vlas.wam_vla import WAMVLA
from fluxvla.transforms.transform_prompts import LoadCachedTextEmbedding
from fluxvla.wam_modes import (WAM_MODE_TO_ID, WAM_TRAINING_MODES,
                               normalize_wam_mode_probs, wam_mode_to_id)


class _FakeVideoExpert(nn.Module):

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame,
                                  device):
        del video_tokens_per_frame
        return torch.eye(video_seq_len, dtype=torch.bool, device=device)


class _FakeActionExpert(nn.Module):
    action_dim = 7


class _FakeBuiltExpert(nn.Module):

    def __init__(self,
                 config,
                 device='cpu',
                 torch_dtype=torch.float32,
                 skip_load_from_pretrain=False):
        super().__init__()
        self.config_arg = config
        self.device_arg = device
        self.torch_dtype_arg = torch_dtype
        self.skip_load_from_pretrain = skip_load_from_pretrain
        self.blocks = nn.ModuleList()
        self.num_heads = 2
        self.attn_head_dim = 4


class _FakeMoT(nn.Module):

    def __init__(self, video_expert, action_expert):
        super().__init__()
        self.mixtures = {
            'video': video_expert,
            'action': action_expert,
        }


class _BareMoTExpert(nn.Module):

    def __init__(self, num_layers=0, num_heads=2, attn_head_dim=4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim


class _FakeVideoLatentCodec(nn.Module):

    def __init__(self, device, torch_dtype, skip_load_from_pretrain):
        super().__init__()
        self.device_arg = device
        self.torch_dtype_arg = torch_dtype
        self.skip_load_from_pretrain = skip_load_from_pretrain
        self.temporal_downsample_factor = 4


class _FakeContextEncoder(nn.Module):

    def __init__(self, device, torch_dtype, skip_load_from_pretrain):
        super().__init__()
        self.device_arg = device
        self.torch_dtype_arg = torch_dtype
        self.skip_load_from_pretrain = skip_load_from_pretrain

    def forward(self, ids, mask):
        return torch.ones(ids.shape[0], ids.shape[1], 8, device=ids.device)


class _FakeForwardBackbone(nn.Module):

    def __init__(self, skip_load=False):
        super().__init__()
        self.skip_load = skip_load
        self.last_images_shape = None
        self.last_img_masks = None

    def forward(self,
                images,
                lang_tokens,
                img_masks,
                lang_masks=None,
                image_grid_thw=None):
        del image_grid_thw
        self.last_images_shape = None if images is None else tuple(
            images.shape)
        self.last_img_masks = None if img_masks is None else img_masks.cpu()
        batch_size = lang_tokens.shape[0]
        context_len = lang_tokens.shape[1] + 2
        context = torch.ones(
            batch_size, context_len, 8, device=lang_tokens.device)
        additive_mask = torch.zeros(
            batch_size, 1, context_len, context_len, device=lang_tokens.device)
        return context, additive_mask, None


def _build_fake_video_latent_codec(device='cpu',
                                   torch_dtype=torch.float32,
                                   skip_load_from_pretrain=False):
    return _FakeVideoLatentCodec(device, torch_dtype, skip_load_from_pretrain)


def _build_fake_context_encoder(device='cpu',
                                torch_dtype=torch.float32,
                                skip_load_from_pretrain=False):
    return _FakeContextEncoder(device, torch_dtype, skip_load_from_pretrain)


def _build_fake_video_expert(config,
                             device='cpu',
                             torch_dtype=torch.float32,
                             skip_load_from_pretrain=False):
    return _FakeBuiltExpert(config, device, torch_dtype,
                            skip_load_from_pretrain)


def _build_fake_action_expert(config,
                              device='cpu',
                              torch_dtype=torch.float32,
                              skip_load_from_pretrain=False):
    expert = _FakeBuiltExpert(config, device, torch_dtype,
                              skip_load_from_pretrain)
    expert.action_dim = 7
    return expert


def _make_head(**kwargs):
    video_expert = _FakeVideoExpert()
    action_expert = _FakeActionExpert()
    mot = _FakeMoT(video_expert, action_expert)
    return WAMHead(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        text_dim=8,
        device='cpu',
        torch_dtype=torch.float32,
        **kwargs,
    )


class WAMHeadTest(unittest.TestCase):

    def test_registry(self):
        self.assertIs(HEADS.get('WAMHead'), WAMHead)

    def test_wam_mode_ids_are_shared(self):
        self.assertEqual(WAM_TRAINING_MODES,
                         ('forward', 'idm', 'policy', 'joint'))
        self.assertEqual(
            [wam_mode_to_id(mode) for mode in WAM_TRAINING_MODES],
            [0, 1, 2, 3],
        )

    def test_mot_accepts_generic_expert_names(self):
        mot = MoT({
            'observations': _BareMoTExpert(),
            'controls': _BareMoTExpert(),
        })
        self.assertEqual(mot.expert_order, ['observations', 'controls'])

    def test_mot_still_validates_expert_shape_compatibility(self):
        with self.assertRaises(ValueError):
            MoT({
                'observations': _BareMoTExpert(num_heads=2),
                'controls': _BareMoTExpert(num_heads=4),
            })

    def test_training_mode_probs_default_and_validation(self):
        self.assertEqual(
            normalize_wam_mode_probs(None),
            {
                'forward': 0.25,
                'idm': 0.25,
                'policy': 0.25,
                'joint': 0.25,
            },
        )
        self.assertEqual(
            normalize_wam_mode_probs({
                'forward': 0.0,
                'idm': 2.0,
                'policy': 0.0,
                'joint': 0.0,
            }),
            {
                'forward': 0.0,
                'idm': 1.0,
                'policy': 0.0,
                'joint': 0.0,
            },
        )
        with self.assertRaises(ValueError):
            normalize_wam_mode_probs({'bad': 1.0})
        with self.assertRaises(ValueError):
            normalize_wam_mode_probs({'policy': -1.0})
        with self.assertRaises(ValueError):
            normalize_wam_mode_probs({'policy': 0.0})

    def test_mode_collator_honors_degenerate_distribution(self):
        collator = WAMModeCollator(
            keys=['x'],
            mode='batch',
            mode_probs={
                'forward': 0.0,
                'idm': 0.0,
                'policy': 1.0,
                'joint': 0.0,
            },
        )
        batch = [{
            'x': np.array([index], dtype=np.float32)
        } for index in range(4)]
        output = collator(batch)
        self.assertEqual(output['training_mode'].tolist(),
                         [WAM_MODE_TO_ID['policy']] * 4)

    def test_mode_collator_accepts_explicit_joint(self):
        collator = WAMModeCollator(keys=['x'], mode='joint')
        batch = [{'x': np.array([0.0], dtype=np.float32)} for _ in range(3)]
        output = collator(batch)
        self.assertEqual(output['training_mode'].tolist(),
                         [WAM_MODE_TO_ID['joint']] * 3)

    def test_head_accepts_batched_training_mode_ids(self):
        ids = WAMHead._prepare_training_mode_ids(
            training_mode=np.array([0, 1, 2, 3], dtype=np.int64),
            batch_size=4,
            device=torch.device('cpu'),
        )
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])

    def test_forward_attention_mask_allows_actions_to_drive_future_video(self):
        head = _make_head()
        mask = head._build_forward_attention_mask(
            video_seq_len=6,
            action_seq_len=3,
            video_tokens_per_frame=2,
            device=torch.device('cpu'),
        )
        self.assertTrue(mask[:6, :6].diag().all())
        self.assertTrue(mask[6:, 6:].all())
        self.assertFalse(mask[:2, 6:].any())
        self.assertTrue(mask[2:6, 6:].all())
        self.assertFalse(mask[6:, :6].any())

    def test_idm_attention_mask_allows_actions_to_read_full_video(self):
        head = _make_head()
        mask = head._build_idm_attention_mask(
            video_seq_len=6,
            action_seq_len=3,
            video_tokens_per_frame=2,
            device=torch.device('cpu'),
        )
        self.assertTrue(mask[:6, :6].diag().all())
        self.assertTrue(mask[6:, 6:].all())
        self.assertFalse(mask[:6, 6:].any())
        self.assertTrue(mask[6:, :6].all())

    def test_policy_attention_mask_reads_first_frame_only(self):
        head = _make_head()
        mask = head._build_policy_attention_mask(
            video_seq_len=6,
            action_seq_len=3,
            video_tokens_per_frame=2,
            device=torch.device('cpu'),
        )
        self.assertTrue(mask[:6, :6].diag().all())
        self.assertTrue(mask[6:, 6:].all())
        self.assertFalse(mask[:6, 6:].any())
        self.assertTrue(mask[6:, :2].all())
        self.assertFalse(mask[6:, 2:6].any())

    def test_joint_attention_mask_allows_bidirectional_generation(self):
        head = _make_head()
        mask = head._build_joint_attention_mask(
            video_seq_len=6,
            action_seq_len=3,
            video_tokens_per_frame=2,
            device=torch.device('cpu'),
        )
        self.assertTrue(mask[:6, :6].diag().all())
        self.assertTrue(mask[6:, 6:].all())
        self.assertFalse(mask[:2, 6:].any())
        self.assertTrue(mask[2:6, 6:].all())
        self.assertTrue(mask[6:, :6].all())

    def test_training_attention_mask_keeps_policy_and_joint_distinct(self):
        head = _make_head()
        policy_mask = head._build_training_attention_mask(
            mode_ids=torch.tensor([WAM_MODE_TO_ID['policy']] * 2),
            video_seq_len=6,
            action_seq_len=3,
            video_tokens_per_frame=2,
            device=torch.device('cpu'),
        )
        joint_mask = head._build_training_attention_mask(
            mode_ids=torch.tensor([WAM_MODE_TO_ID['joint']] * 2),
            video_seq_len=6,
            action_seq_len=3,
            video_tokens_per_frame=2,
            device=torch.device('cpu'),
        )
        self.assertEqual(tuple(policy_mask.shape), (9, 9))
        self.assertEqual(tuple(joint_mask.shape), (9, 9))
        self.assertTrue(policy_mask[6:, :2].all())
        self.assertFalse(policy_mask[6:, 2:6].any())
        self.assertTrue(joint_mask[6:, :6].all())


class Wan22TextBackboneTest(unittest.TestCase):

    def test_backbone_builds_its_context_encoder(self):
        backbone = Wan22TextBackbone(
            context_encoder=dict(type=_build_fake_context_encoder),
            device='cpu',
            torch_dtype='float32',
            skip_load=True,
        )
        self.assertIsInstance(backbone.context_encoder, _FakeContextEncoder)
        self.assertTrue(backbone.context_encoder.skip_load_from_pretrain)
        self.assertEqual(backbone.context_encoder.torch_dtype_arg,
                         torch.float32)

    def test_backbone_uses_forward_contract(self):
        backbone = Wan22TextBackbone(
            context_encoder=dict(type=_build_fake_context_encoder),
            device='cpu',
            torch_dtype='float32',
            skip_load=True,
        )
        context, context_mask = backbone(
            images=torch.zeros(2, 1, 3, 8, 16),
            lang_tokens=torch.ones(2, 4, dtype=torch.long),
            img_masks=torch.ones(2, 1, dtype=torch.bool),
            lang_masks=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        )
        self.assertEqual(tuple(context.shape), (2, 4, 8))
        self.assertEqual(tuple(context_mask.shape), (2, 4))
        self.assertTrue(context_mask.all())


class LoadCachedTextEmbeddingTest(unittest.TestCase):

    def test_loads_fastwam_compatible_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = 'pick up the blue mug'
            prompt = LoadCachedTextEmbedding.DEFAULT_PROMPT.format(task=task)
            hashed = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            cache_path = Path(tmp_dir) / \
                f'{hashed}.t5_len4.wan22ti2v5b.pt'
            torch.save(
                {
                    'context': torch.arange(12, dtype=torch.float32).reshape(
                        4, 3),
                    'mask': torch.tensor([True, True, False, False]),
                },
                cache_path,
            )

            out = LoadCachedTextEmbedding(
                cache_dir=tmp_dir,
                context_len=4,
            )({
                'task_description': task
            })

        self.assertEqual(tuple(out['context'].shape), (4, 3))
        self.assertTrue(torch.equal(out['context'][2:], torch.zeros(2, 3)))
        self.assertEqual(out['context_mask'].tolist(),
                         [True, True, True, True])


class WAMVLATest(unittest.TestCase):

    def test_vla_builds_backbone_as_a_top_level_module(self):
        model = WAMVLA(
            vlm_backbone=dict(
                type='Wan22TextBackbone',
                context_encoder=dict(type=_build_fake_context_encoder),
            ),
            video_latent_codec=dict(type=_build_fake_video_latent_codec),
            vla_head=dict(
                type='WAMHead',
                video_expert=dict(
                    type=_build_fake_video_expert,
                    config=dict(text_dim=8),
                ),
                action_expert=dict(
                    type=_build_fake_action_expert,
                    config=dict(),
                ),
            ),
            proprio_dim=8,
            skip_load=True,
        )
        self.assertIsInstance(model.vlm_backbone, Wan22TextBackbone)
        self.assertIsInstance(model.video_latent_codec, _FakeVideoLatentCodec)
        self.assertIsInstance(model.vla_head, WAMHead)
        self.assertIsInstance(model.vla_head.mot, MoT)
        self.assertTrue(model.video_latent_codec.skip_load_from_pretrain)
        self.assertTrue(model.vla_head.video_expert.skip_load_from_pretrain)

    def test_vla_uses_vlm_forward_for_context(self):
        model = WAMVLA(
            vlm_backbone=dict(type=_FakeForwardBackbone),
            video_latent_codec=dict(type=_build_fake_video_latent_codec),
            vla_head=dict(
                type='WAMHead',
                video_expert=dict(
                    type=_build_fake_video_expert,
                    config=dict(text_dim=8),
                ),
                action_expert=dict(
                    type=_build_fake_action_expert,
                    config=dict(),
                ),
            ),
            proprio_dim=8,
            skip_load=True,
        )
        context, context_mask = model._encode_vlm_context(
            images=torch.zeros(2, 3, 9, 8, 16),
            lang_tokens=torch.ones(2, 4, dtype=torch.long),
            lang_masks=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
        )
        self.assertTrue(model.vlm_backbone.skip_load)
        self.assertEqual(model.vlm_backbone.last_images_shape,
                         (2, 1, 3, 8, 16))
        self.assertEqual(model.vlm_backbone.last_img_masks.tolist(),
                         [[True], [True]])
        self.assertEqual(tuple(context.shape), (2, 6, 8))
        self.assertEqual(context_mask.tolist(), [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ])

    def test_vla_can_train_from_cached_context_without_backbone(self):
        model = WAMVLA(
            vlm_backbone=None,
            video_latent_codec=dict(type=_build_fake_video_latent_codec),
            vla_head=dict(
                type='WAMHead',
                video_expert=dict(
                    type=_build_fake_video_expert,
                    config=dict(text_dim=8),
                ),
                action_expert=dict(
                    type=_build_fake_action_expert,
                    config=dict(),
                ),
            ),
            proprio_dim=8,
            skip_load=True,
        )
        self.assertIsNone(model.vlm_backbone)
        self.assertEqual(model.all_module_keys,
                         ['video_latent_codec', 'vla_head'])


class WAMConfigTest(unittest.TestCase):

    def test_canonical_configs_are_wam_only(self):
        for path, suite in (
            ('configs/wam/wam_libero_object_full_finetune.py',
             'libero_object'),
            ('configs/wam/wam_libero_10_full_finetune.py', 'libero_10'),
        ):
            cfg = Config.fromfile(path)
            self.assertEqual(cfg.model.type, 'WAMVLA')
            self.assertTrue(cfg.model.mot_checkpoint_mixed_attn)
            self.assertIsNone(cfg.model.vlm_backbone)
            self.assertEqual(cfg.inference_model.vlm_backbone.type,
                             'Wan22TextBackbone')
            self.assertNotIn('model_id', cfg.inference_model.vlm_backbone)
            self.assertNotIn('tokenizer_model_id',
                             cfg.inference_model.vlm_backbone)
            self.assertNotIn('tokenizer_max_len',
                             cfg.inference_model.vlm_backbone)
            self.assertNotIn('load_text_encoder',
                             cfg.inference_model.vlm_backbone)
            self.assertNotIn('context_encoder',
                             cfg.inference_model.vlm_backbone)
            self.assertNotIn('video_latent_codec',
                             cfg.inference_model.vlm_backbone)
            self.assertIn('checkpoint_root', cfg.inference_model.vlm_backbone)
            self.assertEqual(cfg.model.video_latent_codec.type, 'Wan22VAE')
            self.assertIn('checkpoint_root', cfg.model.video_latent_codec)
            self.assertEqual(cfg.model.vla_head.type, 'WAMHead')
            self.assertEqual(cfg.model.vla_head.video_expert.type,
                             'WanVideoDiT')
            self.assertIn('checkpoint_root', cfg.model.vla_head.video_expert)
            self.assertIn('config', cfg.model.vla_head.video_expert)
            self.assertTrue(cfg.model.vla_head.video_expert.config.
                            use_gradient_checkpointing)
            self.assertEqual(cfg.model.vla_head.action_expert.type,
                             'ActionDiT')
            self.assertIn('pretrained_path', cfg.model.vla_head.action_expert)
            self.assertIn('config', cfg.model.vla_head.action_expert)
            self.assertTrue(cfg.model.vla_head.action_expert.config.
                            use_gradient_checkpointing)
            self.assertNotIn('video_dit_config', cfg.model.vla_head)
            self.assertNotIn('action_dit_config', cfg.model.vla_head)
            self.assertNotIn('action_dit_pretrained_path', cfg.model.vla_head)
            self.assertNotIn('skip_dit_load_from_pretrain', cfg.model.vla_head)
            self.assertNotIn('mode_probs', cfg.model.vla_head)
            self.assertEqual(cfg.model.vla_head.loss.lambda_joint_video, 1.0)
            self.assertEqual(cfg.model.vla_head.loss.lambda_joint_action, 1.0)
            self.assertEqual(cfg.eval.task_suite_name, suite)
            self.assertEqual(cfg.eval.model_family, 'wam')

            transforms = [
                transform.type for transform in
                cfg.train_dataloader.dataset.datasets.transforms
            ]
            self.assertNotIn('LiberoPromptFromInputs', transforms)
            self.assertIn('LoadCachedTextEmbedding', transforms)
            self.assertNotIn('SampleWAMTrainingMode', transforms)
            self.assertEqual(cfg.runner.collator.type, 'WAMModeCollator')
            self.assertEqual(cfg.runner.collator.mode, 'batch')
            self.assertEqual(cfg.runner.collator.mode_probs.joint, 1.0)

            collator_keys = set(cfg.runner.collator['keys'])
            self.assertIn('context', collator_keys)
            self.assertIn('context_mask', collator_keys)
            self.assertIn('training_mode', collator_keys)
            self.assertNotIn('lang_tokens', collator_keys)
            self.assertNotIn('lang_masks', collator_keys)
            self.assertNotIn('allowed_missing_key_prefixes', cfg.eval)

            eval_prompt_transforms = [
                transform for transform in cfg.eval.dataset.transforms
                if transform.type == 'LiberoPromptFromInputs'
            ]
            self.assertEqual(len(eval_prompt_transforms), 1)
            self.assertIn("robot's point of view",
                          eval_prompt_transforms[0].prompt_template)

            config_text = Path(path).read_text()
            if suite == 'libero_10':
                self.assertNotIn('_base_', config_text)

    def test_qwen3vl_wam_config_switches_context_backbone_only(self):
        path = 'configs/wam/wam_qwen3vl_0_6b_libero_10_full_finetune.py'
        cfg = Config.fromfile(path)
        self.assertEqual(cfg.model.type, 'WAMVLA')
        self.assertEqual(cfg.model.vlm_backbone.type, 'Qwen3VL')
        self.assertEqual(cfg.model.vlm_backbone.vlm_backbone_id,
                         'qwen3_0.6b_vl_pt')
        self.assertTrue(cfg.model.vlm_backbone.use_projection)
        self.assertEqual(cfg.model.vlm_backbone.projection_output_dim, 4096)
        self.assertEqual(cfg.model.video_latent_codec.type, 'Wan22VAE')
        self.assertEqual(cfg.model.vla_head.video_expert.config.text_dim, 4096)

        config_text = Path(path).read_text()
        self.assertIn('QWEN3VL_0_6B_PATH', config_text)
        self.assertNotIn('Wan22TextBackbone', config_text)


if __name__ == '__main__':
    unittest.main()
