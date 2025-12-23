from .openvla_hf import (OpenVLAConfig, OpenVLAForActionPrediction,
                         PrismaticImageProcessor, PrismaticProcessor)

register_dict = dict(
    OpenVLA=dict(
        name='openvla',
        model_class=OpenVLAForActionPrediction,
        config_class=OpenVLAConfig,
        processor_class=PrismaticProcessor,
        image_processor_class=PrismaticImageProcessor))
