from transformers import (Dinov2Config, SiglipVisionConfig, SiglipVisionModel,
                          ViTForImageClassification)

VISION_BACKBONE_CONFIGS = dict(
    dino=dict(
        model_id='vit_large_patch14_reg4_dinov2',
        config=Dinov2Config,
        model_cls=ViTForImageClassification,
    ),
    siglip_224=dict(
        model_id='vit_so400m_patch14_siglip_224',
        config=SiglipVisionConfig,
        model_cls=SiglipVisionModel,
    ),
    siglip_384=dict(
        model_id='vit_so400m_patch14_siglip_384',
        config=SiglipVisionConfig,
        model_cls=SiglipVisionModel,
    ),
)
