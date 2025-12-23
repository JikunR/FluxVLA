from transformers import AutoProcessor

from ..utils import PROCESSORS


@PROCESSORS.register_module()
class PretrainedProcessor:

    def __init__(self, model_path: str, trust_remote_code: bool = True):
        """Load pretrained processor from the specified model path.

        Args:
            model_path (str): Path of the model to load the processor from.
            trust_remote_code (bool, optional): Whether to trust remote code.
                Defaults to True.
        """
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

    def __call__(self, *args, **kwds):
        return self.processor(*args, **kwds)

    def save_pretrained(self, save_directory: str):
        """Save the processor to the specified directory.

        Args:
            save_directory (str): Directory where the processor will be saved.
        """
        self.processor.save_pretrained(save_directory)
