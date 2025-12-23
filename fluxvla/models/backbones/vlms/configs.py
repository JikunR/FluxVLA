from transformers import (PaliGemmaConfig, PaliGemmaForConditionalGeneration,
                          Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration)

VLM_BACKBONE_CONFIGS = dict(
    paligemma_3b_pt_224=dict(
        model_id='paligemma-3b-pt-224',
        config=PaliGemmaConfig,
        model_cls=PaliGemmaForConditionalGeneration,
    ),
    qwen2_5_3b_vl_pt_224=dict(
        model_id='qwen2-5-vl_pt_224',
        config=Qwen2_5_VLConfig,
        model_cls=Qwen2_5_VLForConditionalGeneration,
    ))
