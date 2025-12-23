import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from fluxvla.engines import build_collator_from_cfg, build_dataset_from_cfg


class TestRLDSDataset(unittest.TestCase):

    def setUp(self):
        self.cfg = dict(
            type='RLDSDataset',
            data_root_dir=  # noqa: E251
            '/limx/tos/users/wenhao/libero_rlds/',
            load_camera_views=['primary', 'primary'],
            data_mix=[('libero_10_no_noops', 1.0)],
            batch_transform=dict(
                type='RLDSBatchTransform',
                load_camera_views=['image_primary', 'image_primary'],
                base_tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'openvla/openvla-7b-finetuned-libero-10',  # noqa: E501
                    # special_tokens={'pad_token': '<PAD>'}
                ),
                prompter=dict(
                    type='PurePrompter',
                    model_family='openvla',
                ),
                max_len=180,
                with_labels=False,
                img_transform=dict(
                    type='TransformImage',
                    image_resize_strategy='resize-naive',
                    input_sizes=[[3, 224, 224], [3, 224, 224]],
                    means=[[123.515625, 116.04492188, 103.59375],
                           [128, 128, 128]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [128, 128, 128]],
                ),
            ),
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=9,
                skip_unlabeled=True,
                goal_relabeling_strategy='uniform',
            ),
            frame_transform_kwargs=dict(
                resize_size=(224, 224),
                num_parallel_calls=16,
            ),
            shuffle_buffer_size=100000,
            train=True,
            load_proprio=True,
            image_aug=False,
            action_proprio_normalization_type='bounds_q99')
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        vla_dataset = build_dataset_from_cfg(self.cfg)
        collator = dict(type='NestedCollator')
        self.dataloader = DataLoader(
            vla_dataset,
            batch_size=4,
            sampler=None,
            collate_fn=build_collator_from_cfg(collator),
            num_workers=  # noqa: E251
            0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!  # noqa: E501
        )

    def test_dataset_iter(self):
        for _, batch in enumerate(self.dataloader):
            self.assertEqual(batch['images'].shape,
                             torch.Size([4, 6, 224, 224]))
            self.assertEqual(batch['lang_tokens'].shape, torch.Size([4, 180]))
            self.assertEqual(batch['actions'].shape, torch.Size([4, 10, 7]))
            self.assertEqual(batch['states'].shape, torch.Size([4, 8]))
            self.assertTrue(
                batch['images'].shape == torch.Size([4, 6, 224, 224]))
            self.assertTrue(batch['lang_tokens'].shape == torch.Size([4, 180]))
            self.assertTrue(batch['actions'].shape == torch.Size([4, 10, 7]))
            self.assertTrue(batch['states'].shape == torch.Size([4, 8]))
            break
