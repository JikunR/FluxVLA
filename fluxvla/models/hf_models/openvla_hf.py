import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
import transformers
from PIL import Image
from timm.models.vision_transformer import LayerScale
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from transformers import (AutoModelForCausalLM, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizerBase)
from transformers.image_processing_utils import (BatchFeature,
                                                 ImageProcessingMixin)
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import (PaddingStrategy,
                                             PreTokenizedInput, TextInput,
                                             TruncationStrategy)
from transformers.utils import TensorType

logger = logging.getLogger(__name__)

# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    'clip-vit-l': [224],
    'siglip-vit-so400m': [224],
    'dinov2-vit-l': [224],
    'in1k-vit-l': [224],
    'clip-vit-l-336px': [336],
    'siglip-vit-so400m-384px': [384],
    'dinoclip-vit-l-336px': [336, 336],
    'dinosiglip-vit-so-224px': [224, 224],
    'dinosiglip-vit-so-384px': [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    'clip-vit-l': ['vit_large_patch14_clip_224.openai'],
    'clip-vit-l-336px': ['vit_large_patch14_clip_336.openai'],
    'dinov2-vit-l': ['vit_large_patch14_reg4_dinov2.lvd142m'],
    'in1k-vit-l': ['vit_large_patch16_224.augreg_in21k_ft_in1k'],
    'siglip-vit-so400m': ['vit_so400m_patch14_siglip_224'],
    'siglip-vit-so400m-384px': ['vit_so400m_patch14_siglip_384'],
    'dinoclip-vit-l-336px': [
        'vit_large_patch14_reg4_dinov2.lvd142m',
        'vit_large_patch14_clip_336.openai'
    ],
    'dinosiglip-vit-so-224px':
    ['vit_large_patch14_reg4_dinov2.lvd142m', 'vit_so400m_patch14_siglip_224'],
    'dinosiglip-vit-so-384px':
    ['vit_large_patch14_reg4_dinov2.lvd142m', 'vit_so400m_patch14_siglip_384'],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    'clip-vit-l': ['quick_gelu'],
    'clip-vit-l-336px': ['quick_gelu'],
    'dinov2-vit-l': [None],
    'in1k-vit-l': [None],
    'siglip-vit-so400m': [None],
    'siglip-vit-so400m-384px': [None],
    'dinoclip-vit-l-336px': [None, 'quick_gelu'],
    'dinosiglip-vit-so-224px': [None, None],
    'dinosiglip-vit-so-384px': [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    'llama2-7b-pure':
    '/limx/tos/limx_mani_checkpoints/open_source/huggingface/Llama-2-7b-hf',  # noqa: E501
    'llama2-13b-pure': 'meta-llama/Llama-2-13b-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'vicuna-v15-7b': 'lmsys/vicuna-7b-v1.5',
    'vicuna-v15-13b': 'lmsys/vicuna-13b-v1.5',
    'mistral-v0.1-7b-pure': 'mistralai/Mistral-7B-v0.1',
    'mistral-v0.1-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
    'phi-2-3b': 'microsoft/phi-2',
}
LLM_BACKBONE_TO_HF_METACLASS = {
    'llama2-7b-pure': 'llama',
    'llama2-13b-pure': 'llama',
    'llama2-7b-chat': 'llama',
    'llama2-13b-chat': 'llama',
    'vicuna-v15-7b': 'llama',
    'vicuna-v15-13b': 'llama',
    'mistral-v0.1-7b-pure': 'mistral',
    'mistral-v0.1-7b-instruct': 'mistral',
    'phi-2-3b': 'phi',
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


def letterbox_pad_transform(
        image: Image.Image, padding_fill_value: Tuple[int, int,
                                                      int]) -> Image.Image:
    """Given a PIL.Image, pad to square by adding a
    symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(
        image, padding, fill=padding_fill_value, padding_mode='constant')


class PrismaticConfig(PretrainedConfig):
    model_type: str = 'prismatic'
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = 'siglip-vit-so400m',
        llm_backbone_id: str = 'vicuna-v15-7b',
        arch_specifier: str = 'no-align+gelu-mlp',
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = 'letterbox',
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(
                f'Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }'  # noqa: E501
            )

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(
                f'LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }'  # noqa: E501
            )

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>>
        # supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None else any(
                self.vision_backbone_id.startswith(v)
                for v in ['dinoclip', 'dinosiglip']))

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[
            self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[
            self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[
            self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id,\
            pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a
        # `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](
                **text_config) if text_config is not None else CONFIG_MAPPING[
                    LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]())

        # Dispatch **kwargs to super() =>> note that `pad_token_id`
        # collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAConfig(PrismaticConfig):
    model_type: str = 'openvla'

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[
            str, List[float]]]]]] = None,  # noqa: E501
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)


class PrismaticImageProcessor(ImageProcessingMixin):
    """ Prismatic Image Processor for multimodal VLMs.
    This class implements an image processor that can handle
    different image sizes and transformations for the vision
    backbone. It supports both standard and fused vision
    backbones, and can handle different image resize strategies
    (e.g., letterbox, resize-naive, resize-crop).

    Args:
        use_fused_vision_backbone (bool): Whether to use a
            fused vision backbone (two vision backbones
            concatenated together).
        image_resize_strategy (str): The image resize strategy
            to use. Options are 'letterbox', 'resize-naive',
            and 'resize-crop'.
        input_sizes (Optional[List[Tuple[int, int, int]]]): List of
            input sizes for the vision backbone. Defaults to None,
            which will set it to [(3, 224, 224)].
        interpolations (Optional[List[str]]): List of interpolation
            methods for resizing. Defaults to None, which will set it
            to ['bicubic'].
        means (Optional[List[Tuple[float, float, float]]]): List of
            means for normalization. Defaults to None, which will set
            it to [(0.5, 0.5, 0.5)].
        stds (Optional[List[Tuple[float, float, float]]]): List of
            standard deviations for normalization. Defaults to None,
            which will set it to [(0.5, 0.5, 0.5)].
    """
    model_input_names: ClassVar[List[str]] = ['pixel_values']

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = 'letterbox',
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:

        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # Handle `None` default values
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        # TIMM `data_cfg` Parameters
        self.input_sizes, self.interpolations, self.means, self.stds = \
            input_sizes, interpolations, means, stds

        # Grab torchvision transforms via TIMM =>> need to parse
        # for specific "functional" transform values!
        self.tvf_resize_params, self.tvf_crop_params,\
            self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=  # noqa: E251
                1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)  # noqa: E501
                crop_mode=  # noqa: E251
                'center',  # Default crop mode -- no-op when `crop_pct == 1.0`  # noqa: E501
                is_training=  # noqa: E251
                False,  # No image augmentations when loading the transform!
            )

            # [Validation] Ensure appropriate transform structure, expected sizes  # noqa: E501
            if not (isinstance(transform, Compose) and
                    (len(transform.transforms) == 4)  # noqa: E131
                    and isinstance(transform.transforms[0], Resize)
                    and isinstance(transform.transforms[1], CenterCrop)
                    and isinstance(transform.transforms[2], ToTensor)
                    and isinstance(transform.transforms[3], Normalize) and
                    (transform.transforms[0].size == self.input_sizes[idx][-1])
                    and (transform.transforms[1].size
                         == self.input_sizes[idx][-2:])):  # noqa: E129
                raise ValueError(
                    f'Unexpected TIMM image transformation structure/sizes: `{transform}`'  # noqa: E501
                )

            # HF Image Processors *must* be JSON-serializable; as such,
            # cannot have torchvision. as an attribute.
            #   => Instead, we're going to parse the transform and call
            # "torchvision.transforms.functional" (`tvf`)
            resize_t, crop_t, norm_t = transform.transforms[
                0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append({
                'size':
                resize_t.size,
                'interpolation':
                TVF.pil_modes_mapping[resize_t.interpolation],
                'max_size':
                None,
                'antialias':
                True,
            })
            self.tvf_crop_params.append({'output_size': crop_t.size})
            self.tvf_normalize_params.append({
                'mean':
                norm_t.mean.float().numpy().tolist(),
                'std':
                norm_t.std.float().numpy().tolist(),
                'inplace':
                False,
            })
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == 'resize-naive':
                self.tvf_resize_params[idx]['size'] = (resize_t.size,
                                                       resize_t.size)
            elif self.image_resize_strategy == 'letterbox':
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple(
                    [int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == 'resize-crop':
                pass
            else:
                raise ValueError(
                    f'Image resize strategy `{self.image_resize_strategy}` is not supported!'  # noqa: E501
                )

        # Dispatch **kwargs to super()
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """Apply the image transformation to a single PIL.Image.Image instance.

        Args:
            img (PIL.Image.Image): The image to transform.

        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # [Contract] Fused Backbones expect "channel-stacked"
        # inputs; we'll unpack on the model side!
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t,
                                      **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # [Contract] `imgs_t` is a list of Tensors of shape [3, input_size,
        # input_size]; stack along dim = 0
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """ Preprocess a single image or a list of images into a
        `BatchFeature` instance.

        Args:
            images (Union[Image.Image, List[Image.Image]]): A single
            image or a list of images to preprocess.
            return_tensors (Optional[Union[str, TensorType]]): The
            type of tensors to return. Defaults to None.
        Returns:
            BatchFeature: A BatchFeature instance containing
                the preprocessed images.
        """
        if not isinstance(images, list):
            images = [images]

        # Apply `self.img_transform` to each image (will return list
        # of torch.Tensors); stack into "batched" Tensor
        pixel_values = torch.stack(
            [self.apply_transform(img.convert('RGB')) for img in images])

        # Return BatchFeature =>> note that for compatibility, constructor
        # expects Dict[str, np.ndarray], so we convert
        return BatchFeature(
            data={'pixel_values': pixel_values.float().numpy()},
            tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]],
                 **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class PrismaticVisionBackbone(nn.Module):
    """ Prismatic Vision Backbone for multimodal VLMs.
    This class implements a vision backbone that can either
    use a single vision backbone or a fused vision backbone
    (two vision backbones concatenated together).
    It supports both standard and fused vision backbones,
    and can handle different image sizes and activation layers
    for each backbone.

    Args:
        use_fused_vision_backbone (bool): Whether to use a
            fused vision backbone (two vision backbones
            concatenated together).
        image_sizes (List[int]): List of image sizes for
            the vision backbones. If `use_fused_vision_backbone`
            is True, this should contain two sizes.
        timm_model_ids (List[str]): List of TIMM model IDs
            for the vision backbones. If `use_fused_vision_backbone`
            is True, this should contain two model IDs.
        timm_override_act_layers (List[Optional[str]]): List of
            activation layer overrides for the TIMM models.
            If `use_fused_vision_backbone` is True, this should
            contain two activation layer overrides.
        @note: The `image_sizes`, `timm_model_ids`, and
            `timm_override_act_layers` parameters are lists to
            support multiple input sizes and models for different
            vision backbones.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones,
        # create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of
        # the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return
        # the **SECOND-TO-LAST** layer patches!
        assert len(
            timm_model_ids
        ) <= 2, 'Prismatic models only support up to 2 (fused) vision backbones!'  # noqa: E501
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(
                self.featurizer.get_intermediate_layers,
                n={len(self.featurizer.blocks) - 2}))
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(
                    self.fused_featurizer.get_intermediate_layers,
                    n={len(self.fused_featurizer.blocks) - 2}))
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and
        # `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """ Forward pass for the PrismaticVisionBackbone.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape
                [batch_size, 2 * 3, resolution, resolution] if
                `use_fused_vision_backbone` is True, otherwise
                [batch_size, 3, resolution, resolution].
        """
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution,
        # resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(
            img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    """ Prismatic Projector for multimodal VLMs.
    This class implements a projector that maps the
    vision backbone features to the LLM dimension.
    It supports both standard and fused vision backbones,
    and can handle different MLP architectures based on
    the `use_fused_vision_backbone` flag.

    Args:
        use_fused_vision_backbone (bool): Whether to use a
            fused vision backbone (two vision backbones
            concatenated together).
        vision_dim (int): The dimension of the vision features.
        llm_dim (int): The dimension of the LLM features.
        @note: The `vision_dim` and `llm_dim` parameters are
            used to define the input and output dimensions
            of the projector MLP layers.
    """

    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int,
                 llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different
        # MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(
                self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(
                initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for outputs of causal language models
        with past key values.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned
            when `labels` is provided): Language modeling
            loss (for next-token prediction) of the
            language model.
        logits (`torch.FloatTensor` of shape `(batch_size,
            sequence_length, config.vocab_size)`): Prediction
            scores of the language model, typically the output
            of a linear layer applied to the hidden states.
        past_key_values (`Tuple[Tuple[torch.FloatTensor]]`,
            *optional*): Tuple of `torch.FloatTensor` of
            length `config.n_layers`, with each tuple
            having two tensors of shape `(batch_size,
            num_heads, sequence_length, embed_size_per_head)`
            and `(batch_size, num_heads, sequence_length,
            embed_size_per_head)` containing the key and
            value projections of the attention blocks.
        hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of
            the embeddings, and one for the output of each layer)
            of the language model, each of shape `(batch_size,
            sequence_length, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of attention weights of shape `(batch_size,
            num_heads, sequence_length, sequence_length)`.
        projector_features (`torch.FloatTensor`, *optional*):
            Projected features from the vision backbone, of
            shape `(batch_size, sequence_length, llm_dim)`.
            Only returned if `output_projector_states` is
            set to `True` in the configuration.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = 'model'
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ['PrismaticProjector']
    _skip_keys_device_placement: str = 'past_key_values'
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for
        # training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct;
        # if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, 'initializer_range') else
            self.config.text_config.initializer_range)

        if hasattr(module, 'class_embedding'):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):

    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config`
        # Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError(
                'Missing config field `use_fused_vision_backbone`')

        if timm.__version__ not in {'0.9.10', '0.9.11', '0.9.12', '0.9.16'}:
            raise NotImplementedError(
                'TIMM Version must be >= 0.9.10 and < '
                '1.0.0 (breaking);'
                'please raise a GitHub Issue '
                'if you urgently need support for latest TIMM'
                'versions.')

        if (transformers.__version__ != '4.40.1') or (tokenizers.__version__ !=
                                                      '0.19.1'):
            logger.warning(
                f'Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got '  # noqa: E501
                f'`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; '  # noqa: E501
                f'there might be inference-time regressions due to dependency changes. If in doubt, please'  # noqa: E501
                f'use the above versions.')

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes,
            config.timm_model_ids, config.timm_override_act_layers)

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation=config._attn_implementation)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()`
        # and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights(
        )  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
            self,
            new_num_tokens: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        lang_tokens: Optional[torch.LongTensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """ Forward pass for the PrismaticForConditionalGeneration model.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Indices of input sequence tokens in the vocabulary.
                If `pixel_values` is provided, this should be the
                text input sequence.
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding
                token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            pixel_values (`torch.FloatTensor`, *optional*):
                Pixel values of the input images. If provided,
                this should be a tensor of shape
                `(batch_size, num_channels, height, width)`.
            labels (`torch.LongTensor`, *optional*):
                Labels for computing the language modeling loss.
                Indices should be in `[-100, 0, ..., config.vocab_size]`.
                Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens
                with labels in `[0, ..., config.vocab_size]`.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Optionally, instead of passing `input_ids`, the
                user can choose to directly pass an embedded
                representation. This is useful if you want to
                modify the embeddings before feeding them to the
                model.
            past_key_values (`List[torch.FloatTensor]`, *optional*):
                Pre-computed key and value hidden states of the
                attention blocks. Can be used to speed up decoding.
                If provided, it should be a list of tuples, where
                each tuple contains two tensors of shape
                `(batch_size, num_heads, sequence_length,
                embed_size_per_head)` for keys and values.
            use_cache (`bool`, *optional*):
                Whether to use the past key and value hidden states
                to speed up decoding. If set to `True`, the model
                will return the past key and value hidden states
                for future use.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions weights of the
                attention blocks. If set to `True`, the model will
                return a tuple containing the attention weights
                for each attention block.
            output_hidden_states (`bool`, *optional*):
                Whether to return the hidden states of the model.
                If set to `True`, the model will return a tuple
                containing the hidden states for each layer.
            output_projector_features (`bool`, *optional*):
                Whether to return the projected features from the
                vision backbone. If set to `True`, the model will
                return a tensor of shape `(batch_size, sequence_length,
                llm_dim)`.
            return_dict (`bool`, *optional*):
                Whether to return the output as a `ModelOutput`
                instance or a plain tuple. If set to `True`, the
                model will return a `PrismaticCausalLMOutputWithPast`
                instance, otherwise it will return a tuple of
                `(loss, logits, past_key_values, hidden_states,
                attentions, projector_features)`.

        Returns:
            `PrismaticCausalLMOutputWithPast` or `Tuple`:
                The output of the model, either as a
                `PrismaticCausalLMOutputWithPast` instance or a
                tuple containing the loss, logits, past key values,
                hidden states, attentions, and projector features.

        """
        output_attentions = output_attentions if\
            output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        output_projector_features = output_projector_features if\
            output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else\
            self.config.use_return_dict

        # Respect `use_cache` only if not training (even if\
        # `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the
        # following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1)
        # and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) \
        # and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`)
        # =>> requires `past_keys_values` ===
        if lang_tokens.shape[1] == 1:
            assert lang_tokens.shape[
                0] == 1, 'Generation is only currently supported for batch size of 1!'  # noqa: E501
            assert past_key_values is not None, 'You must provide `past_key_values` during cached generation!'  # noqa: E501
            assert labels is None, 'Unexpected key `labels` provided during cached generation!'  # noqa: E501

            language_model_output = self.language_model(
                input_ids=lang_tokens,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (lang_tokens is not None) and (
                inputs_embeds is
                None), 'Missing `input_ids` in language-only forward!'
            assert past_key_values is None, 'Unexpected key `past_key_values` provided during language-only forward!'  # noqa: E501

            language_model_output = self.language_model(
                input_ids=lang_tokens,
                attention_mask=lang_masks,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (lang_tokens.shape[0]
              == pixel_values.shape[0]) or (inputs_embeds.shape[0]
                                            == pixel_values.shape[0]):
            assert past_key_values is None, 'Unexpected key `past_key_values` provided during language-only forward!'  # noqa: E501

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if lang_masks is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0],
                     projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=lang_masks.dtype,
                    device=lang_masks.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(lang_tokens)

            # Build Multimodal Embeddings & Attention Mask =>>
            # Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat([
                input_embeddings[:, :1, :], projected_patch_embeddings,
                input_embeddings[:, 1:, :]
            ],
                                              dim=1)
            multimodal_attention_mask = None
            if lang_masks is not None:
                multimodal_attention_mask = torch.cat([
                    lang_masks[:, :1], projected_patch_attention_mask,
                    lang_masks[:, 1:]
                ],
                                                      dim=1)

            # Build Labels (if specified) =>> Ignore Labels
            # for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0],
                     projected_patch_embeddings.shape[1]),
                    fill_value=self.config.pad_token_id,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat(
                    [labels[:, :1], projected_patch_labels, labels[:, 1:]],
                    dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (lang_tokens.shape[0] != pixel_values.shape[0]) or (
                inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError(
                'Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!'  # noqa: E501
            )

        else:
            raise ValueError(
                'Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n'  # noqa: E501
                f'=> `input_ids` = {lang_tokens is not None}\n'
                f'=> `attention_mask` = {lang_masks is not None}\n'
                f'=> `pixel_values` = {pixel_values is not None}\n'
                f'=> `labels` = {labels is not None}\n'
                f'=> `input_embeds` = {inputs_embeds is not None}\n'
                f'=> `past_key_values` = {past_key_values is not None}\n'
                f'=> `use_cache` = {use_cache}')

        # Unpack `language_model_output` and return
        # PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings
                                              is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """ Prepare inputs for generation.
        Args:
            input_ids (`torch.Tensor`, *optional*):
                The input token IDs for the language model.
            past_key_values (`List[torch.FloatTensor]`, *optional*):
                Pre-computed key and value hidden states of the
                attention blocks. Can be used to speed up decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Optionally, instead of passing `input_ids`, the
                user can choose to directly pass an embedded
                representation. This is useful if you want to
                modify the embeddings before feeding them to the
                model.
            pixel_values (`torch.FloatTensor`, *optional*):
                Pixel values of the input images. If provided,
                this should be a tensor of shape
                `(batch_size, num_channels, height, width)`.
            attention_mask (`torch.Tensor`, *optional*):
                Mask to avoid performing attention on padding
                token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the
                prepared inputs for generation. The dictionary
                includes the following keys:
                - `input_ids`: The input token IDs for the
                    language model.
                - `attention_mask`: The attention mask for the
                    input tokens.
                - `pixel_values`: The pixel values of the input
                    images.
                - `past_key_values`: The pre-computed key and value
                    hidden states.
        """
        if ((input_ids is not None) and
            (input_ids.shape[0] > 1)) or ((inputs_embeds is not None) and
                                          (inputs_embeds.shape[0] > 1)):
            raise ValueError(
                'Generation with batch size > 1 is not currently supported!')

        # Handle `past_key_values` (cache) =>> assume `input_ids`
        # just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the
        # 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'input_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update({
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
        })

        return model_inputs

    # Defer to Language Model (all handle this differently, with
    # different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    """ OpenVLAForActionPrediction is a model for action prediction
    based on the OpenVLA architecture. It extends the
    `PrismaticForConditionalGeneration` class and implements
    the `predict_action()` method to generate actions based on
    input token IDs and unnormalized action statistics.
    It uses the `OpenVLAConfig` configuration class to define
    the model's architecture and parameters.

    Args:
        config (`OpenVLAConfig`): The configuration class
            for the OpenVLA model, which contains
            the model's architecture and parameters.
    """
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size -\
            self.config.pad_to_multiple_of

    def predict_action(self,
                       input_ids: Optional[torch.LongTensor] = None,
                       unnorm_key: Optional[str] = None,
                       **kwargs: str) -> np.ndarray:
        """ Predict actions from the model given input_ids and
        unnormalized key for action statistics.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                The input token IDs for the language model.
            unnorm_key (`str`, *optional*):
                The key for the unnormalized action statistics
                to be used for de-normalizing the predicted actions.
            **kwargs: Additional keyword arguments to pass to
                the `generate()` method.

        Returns:
            np.ndarray: The predicted actions as a NumPy array,
                de-normalized based on the action statistics
                for the specified `unnorm_key`.
        """
        # If the special empty token ('') does not already appear after
        # the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs
        # seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(
                    torch.Tensor([29871]).long(), dim=0).to(input_ids.device)),
                dim=1)

        # Run VLA inference
        generated_ids = self.generate(
            input_ids,
            max_new_tokens=self.get_action_dim(unnorm_key),
            **kwargs)

        # Extract predicted action tokens and translate into
        # (normalized) continuous actions
        predicted_action_token_ids = generated_ids[
            0, -self.get_action_dim(unnorm_key):].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get(
            'mask', np.ones_like(action_norm_stats['q01'], dtype=bool))
        action_high, action_low = np.array(action_norm_stats['q99']), np.array(
            action_norm_stats['q01'])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) +
            action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]],
                          unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f'Your model was trained on more than one dataset, '
                f'please pass a `unnorm_key` from the following options to choose the statistics '  # noqa: E501
                f'used for un-normalizing actions: {norm_stats.keys()}')
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f'The `unnorm_key` you chose is not in the set of available dataset statistics, '  # noqa: E501
            f'please choose from: {norm_stats.keys()}')
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]['action']['q01'])

    def get_action_stats(self,
                         unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]['action']


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py  # noqa: E501
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ['image_processor', 'tokenizer']
    image_processor_class: str = 'AutoImageProcessor'
    tokenizer_class: str = 'AutoTokenizer'

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput],
                    List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """ Preprocess a (batch of) text(s) and image(s) for a Prismatic VLM;
        applies the image processor and tokenizer, returning a BatchFeature.

        Args:
            text (Union[TextInput, PreTokenizedInput, List[TextInput],
                List[PreTokenizedInput]]):
                A single text input or a list of text inputs to preprocess.
            images (Union[Image.Image, List[Image.Image]]): A single
                PIL image or a list of PIL images to preprocess.
            padding (Union[bool, str, PaddingStrategy]): Whether to
                pad the text inputs; defaults to False (no padding).
            truncation (Optional[Union[bool, str, TruncationStrategy]]):
                Whether to truncate the text inputs; defaults to None
                (no truncation).
            max_length (Optional[int]): Maximum length for truncation;
                defaults to None (no maximum length).
            return_tensors (Optional[Union[str, TensorType]]): The
                type of tensors to return; defaults to
                TensorType.PYTORCH.
        """
        pixel_values = self.image_processor(
            images, return_tensors=return_tensors)['pixel_values']
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length)

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError(
                'Batch is malformed; expected same number of images and text inputs!'  # noqa: E501
            )

        return BatchFeature(data={**text_inputs, 'pixel_values': pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===  # noqa: E501
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor,
                         Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor,
                         Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names))
