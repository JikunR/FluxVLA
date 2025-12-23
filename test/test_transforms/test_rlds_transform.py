import unittest

import numpy as np

from fluxvla.engines.utils import build_transform_from_cfg


class TestRLDSBatchTransform(unittest.TestCase):

    def setUp(self):
        self.cfg = dict(
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
                means=[[123.515625, 116.04492188, 103.59375], [128, 128, 128]],
                stds=[[58.27148438, 57.02636719, 57.27539062], [128, 128,
                                                                128]],
            ))
        self.transform = build_transform_from_cfg(self.cfg)

    def test_rlds_batch_transform(self):
        input_data = np.load(
            'test/data/transforms/rlds_transform/rlds_transform_input.npy',
            allow_pickle=True).item()

        output = self.transform(input_data)
        imgs_target = np.load('test/data/transforms/rlds_transform/images.npy')
        img_masks_target = np.load(
            'test/data/transforms/rlds_transform/img_masks.npy')
        lang_masks_target = np.load(
            'test/data/transforms/rlds_transform/lang_masks.npy')
        lang_tokens_target = np.load(
            'test/data/transforms/rlds_transform/lang_tokens.npy')
        states_target = np.load(
            'test/data/transforms/rlds_transform/states.npy')
        actions_target = np.load(
            'test/data/transforms/rlds_transform/actions.npy')
        self.assertTrue(
            np.allclose(output['images'][:, :10, :10], imgs_target))
        self.assertTrue(np.allclose(output['img_masks'], img_masks_target))
        self.assertTrue(np.allclose(output['lang_masks'], lang_masks_target))
        self.assertTrue(np.allclose(output['lang_tokens'], lang_tokens_target))
        self.assertTrue(np.allclose(output['states'], states_target))
        self.assertTrue(np.allclose(output['actions'], actions_target))
