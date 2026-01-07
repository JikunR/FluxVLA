import math
import os
from typing import Dict

import torch
import torch.distributed as dist
import tqdm
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoImageProcessor, AutoModelForVision2Seq,
                          AutoProcessor, BitsAndBytesConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines.utils.overwatch import initialize_overwatch
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class DDPHFFinetuneRunner:
    """
    DDPHFFinetuneRunner is a runner for fine-tuning HuggingFace models
    using Distributed Data Parallel (DDP) training. It is designed to
    handle the setup and execution of training loops for Vision-Language
    Models (VLMs) such as Pi0, Paligemma, and Libero.

    This runner supports both epoch-based and iteration-based training modes,
    and handles RLDS datasets which may have infinite iterators.

    Args:
        cfg (Dict): Configuration dictionary containing model and training
            parameters.
        args: Command-line arguments or additional parameters.
        learning_rate (float): Learning rate for the optimizer.
        collator (Dict): Collator configuration for batching data.
        sampler (Dict): Sampler configuration for data loading.
        grad_accumulation_steps (int): Number of steps for gradient
            accumulation.
        max_epochs (int, optional): Maximum number of training epochs
            (for epoch-based training). Cannot be used with max_steps.
        max_steps (int, optional): Maximum number of training steps
            (for iteration-based training). Cannot be used with max_epochs.
        save_epoch_interval (int): Frequency of saving model checkpoints
            per epoch. Default is 1.
        save_iter_interval (int): Frequency of saving model checkpoints
            per iteration. Default is 10000.
        save_latest_checkpoint_only (bool): If True, only the latest
            checkpoint is saved, overwriting previous ones.
            Default is False.
        trust_remote_code (bool): Whether to trust remote code when
            loading models.
            Default is True.
    """

    def __init__(self,
                 cfg: Dict,
                 args,
                 learning_rate: float,
                 collator: Dict,
                 sampler: Dict,
                 grad_accumulation_steps: int,
                 max_epochs: int = None,
                 max_steps: int = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 save_latest_checkpoint_only: bool = False,
                 trust_remote_code: bool = True):
        from ..utils.builder import (build_collator_from_cfg,
                                     build_processor_from_cfg)

        # Validate training mode configuration
        # Only one of max_epochs or max_steps should be specified
        assert not (max_epochs is not None and max_steps is not None), \
            'Only one of max_epochs or max_steps should be specified!'
        assert not (max_epochs is None and max_steps is None), \
            'Either max_epochs or max_steps should be specified!'

        # Store configuration
        self.cfg = cfg
        self.args = args
        self.learning_rate = learning_rate
        self.collator = build_collator_from_cfg(collator)
        self.sampler = sampler
        self.quantization_config = None

        # Create work directory if it doesn't exist
        os.makedirs(args.work_dir, exist_ok=True)

        # Setup quantization configuration if enabled
        if cfg.model.use_quantization:
            assert cfg.model.use_lora, \
                'Quantized training only supported for LoRA fine-tuning!'
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4')

        # Training configuration
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.save_epoch_interval = save_epoch_interval
        self.save_iter_interval = save_iter_interval
        self.save_latest_checkpoint_only = save_latest_checkpoint_only  # noqa: E501

        # Distributed training setup
        self.wandb_mode = os.environ.get('WANDB_MODE', 'online')
        self.distributed_state = overwatch.distributed_state
        self.device_id = overwatch.local_rank()
        self.processor = build_processor_from_cfg(cfg.processor)
        self.trust_remote_code = trust_remote_code

        # Determine training mode: 'epoch_based' or 'step_based'
        self.training_mode = 'step_based' if max_steps is not None else 'epoch_based'  # noqa: E501

        # Initialize epoch tracking variables (important for RLDS datasets)
        self.current_epoch = 0
        self.steps_in_current_epoch = 0
        # Will be determined dynamically at runtime for RLDS datasets
        self.actual_steps_per_epoch = None
        self.global_step = 0

    def run_setup(self, *args, **kwargs):
        """Initialize model, optimizer, and distributed training setup.

        This method performs the following setup tasks:
        1. Sets up CUDA device
        2. Registers custom model components if needed
        3. Loads the model with appropriate configurations
        4. Sets up LoRA if enabled
        5. Initializes the optimizer
        6. Wraps the model with DDP for distributed training
        """
        from fluxvla.models.hf_models.configs import register_dict

        # Set CUDA device and clear cache
        torch.cuda.set_device(device_id := self.device_id)
        torch.cuda.empty_cache()

        # Initialize model configuration if custom config class is specified
        config = None
        if 'config_class' in register_dict.get(self.cfg['model']['type'], {}):
            config = register_dict[
                self.cfg['model']['type']]['config_class'].from_pretrained(
                    self.cfg.model.model_path,
                    trust_remote_code=self.trust_remote_code)

        # Register custom image processor if specified
        if 'image_processor_class' in register_dict.get(
                self.cfg['model']['type'], {}):
            AutoImageProcessor.register(
                register_dict[self.cfg['model']['type']]['name'],
                register_dict[self.cfg['model']
                              ['type']]['image_processor_class'])

        # Register custom processor if specified
        if 'processor_class' in register_dict.get(self.cfg['model']['type'],
                                                  {}):
            AutoProcessor.register(
                register_dict[self.cfg['model']['type']]['config_class'],
                register_dict[self.cfg['model']['type']]['processor_class'])

        # Load the model with custom class if specified, otherwise use default
        if 'model_class' in register_dict.get(self.cfg['model']['type'], {}):
            self.vla = register_dict[
                self.cfg['model']['type']]['model_class'].from_pretrained(
                    self.cfg.model.model_path,
                    trust_remote_code=self.trust_remote_code,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True)
        else:
            # Fallback to default AutoModelForVision2Seq
            self.vla = AutoModelForVision2Seq.from_pretrained(
                self.cfg.model.model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True)

        torch.cuda.set_device(
            device_id := self.distributed_state.local_process_index)

        # Prepare model for quantized training if enabled
        if self.cfg.model.use_quantization:
            self.vla = prepare_model_for_kbit_training(self.vla)

        # Setup LoRA if enabled
        if self.cfg.model.use_lora:
            # Use configured lora_alpha, default to lora_rank if not specified
            lora_alpha = getattr(self.cfg.model, 'lora_alpha',
                                 self.cfg.model.lora_rank)
            # Get modules_to_save for full fine-tuning of specific modules
            modules_to_save = getattr(self.cfg.model, 'modules_to_save', None)
            lora_config = LoraConfig(
                r=self.cfg.model.lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=self.cfg.model.lora_dropout,
                target_modules=self.cfg.model.lora_target_modules,
                modules_to_save=modules_to_save,
                init_lora_weights='gaussian',
            )
            self.vla = get_peft_model(self.vla, lora_config)
            self.vla.print_trainable_parameters()

        # Initialize optimizer with only trainable parameters
        trainable_params = [
            param for param in self.vla.parameters() if param.requires_grad
        ]
        self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
        torch.cuda.empty_cache()

        # Move model to device if not using quantization
        if not self.cfg.model.use_quantization:
            self.vla = self.vla.to(device_id)

        # Wrap model with DDP for distributed training
        self.vla = DDP(
            self.vla,
            device_ids=[device_id],
            find_unused_parameters=True,
            gradient_as_bucket_view=True)

        # Log training setup information
        overwatch.info(
            'DDP =>> Finalized Training Setup:\n'
            f'|-> Training Mode = {self.training_mode}\n'
            f'|-> {"Max Epochs = " + str(self.max_epochs) if self.training_mode == "epoch_based" else "Max Steps = " + str(self.max_steps)}\n'  # noqa: E501
            f'|-> Global (Effective) Batch Size = {self.cfg.train_dataloader.per_device_batch_size * overwatch.distributed_state.num_processes}\n'  # noqa: E501
            f'|-> Per-Device Batch Size = {self.cfg.train_dataloader.per_device_batch_size}\n'  # noqa: E501
            f'|-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n'  # noqa: E501
            f'|-> Distributed World Size = {overwatch.world_size()}\n'  # noqa: E501
        )

    def _estimate_steps_per_epoch(self, dataset, sampler):
        """Estimate the number of steps per epoch.

        This is particularly important for RLDS datasets which may have
        infinite iterators or unknown lengths.

        Args:
            dataset: The dataset object
            sampler: The data sampler (may be DistributedSampler)

        Returns:
            int or None: Estimated steps per epoch, or None if unknown
        """
        if sampler is not None:
            # When using DistributedSampler
            # it handles the dataset partitioning
            return len(sampler)
        else:
            try:
                dataset_len = len(dataset)
                global_batch_size = self.cfg.train_dataloader.per_device_batch_size * overwatch.world_size(  # noqa: E501
                )
                return math.ceil(dataset_len / global_batch_size)
            except (TypeError, AttributeError):
                # Dataset has no length (infinite iterator like RLDS)
                return None

    def _should_save_checkpoint(self) -> bool:
        """Determine if a checkpoint should be saved at the current step/epoch.

        Returns:
            bool: True if checkpoint should be saved
        """
        if self.training_mode == 'step_based':
            # Save based on iteration intervals
            return (self.global_step > 0
                    and self.global_step % self.save_iter_interval == 0)
        else:  # epoch_based
            # Save at the end of specified epoch intervals
            return (self.current_epoch > 0
                    and self.current_epoch % self.save_epoch_interval == 0)

    def _should_terminate(self) -> bool:
        """Check if training should terminate.

        Returns:
            bool: True if training should stop
        """
        if self.training_mode == 'step_based':
            return self.global_step >= self.max_steps
        else:  # epoch_based
            return self.current_epoch >= self.max_epochs

    def run(self, vla_dataset):
        """Execute the training loop for the Vision-Language Model.

        This method supports both epoch-based and iteration-based training,
        and properly handles RLDS datasets with potentially infinite iterators.

        Key features:
        - Handles both finite and infinite datasets (RLDS)
        - Supports gradient accumulation
        - Implements proper checkpoint saving for both training modes
        - Manages distributed training synchronization

        Args:
            vla_dataset: The dataset for training

        Returns:
            str: Path to the final saved checkpoint
        """
        # Setup distributed sampler if needed
        if self.sampler == 'distributed':
            sampler = torch.utils.data.distributed.DistributedSampler(
                vla_dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                drop_last=False  # Don't drop incomplete batches
            )
        else:
            sampler = None

        # Initialize DataLoader
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.cfg.train_dataloader.per_device_batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=0,
        )

        # Save dataset statistics (only on main process)
        from fluxvla.datasets.utils import save_dataset_statistics
        if self.distributed_state.is_main_process:
            save_dataset_statistics(vla_dataset.dataset_statistics,
                                    self.args.work_dir)

        # Estimate steps per epoch (may be None for infinite datasets)
        estimated_steps_per_epoch = self._estimate_steps_per_epoch(
            vla_dataset, sampler)

        # Log training information
        if overwatch.is_rank_zero():
            try:
                dataset_len = len(vla_dataset)
                overwatch.info(f'Dataset length: {dataset_len}')
            except (TypeError, AttributeError):
                overwatch.info(
                    'Dataset length: unknown (infinite iterator/RLDS)')

            overwatch.info(f'Training mode: {self.training_mode}')
            overwatch.info(
                f'Estimated steps per epoch: {estimated_steps_per_epoch}')

            if self.training_mode == 'epoch_based':
                overwatch.info(f'Max epochs: {self.max_epochs}')
                if estimated_steps_per_epoch:
                    overwatch.info(
                        f'Expected total steps: {self.max_epochs * estimated_steps_per_epoch}'  # noqa: E501
                    )
            else:
                overwatch.info(f'Max steps: {self.max_steps}')

        # Calculate total steps for progress bar
        if self.training_mode == 'step_based':
            total_steps = self.max_steps
        else:
            # For epoch_based training with RLDS, use estimate or placeholder
            if estimated_steps_per_epoch is not None:
                total_steps = self.max_epochs * estimated_steps_per_epoch
            else:
                # Unknown dataset size, use conservative estimate
                # Will be updated dynamically
                total_steps = self.max_epochs * 100

        # Set model to training mode
        self.vla.train()
        self.optimizer.zero_grad()

        # Initialize tracking variables
        self.global_step = 0
        accumulated_loss = 0
        checkpoint_path = self.args.work_dir

        # Main training loop
        with tqdm.tqdm(
                total=total_steps,
                desc='Training',
                disable=not overwatch.is_rank_zero()) as progress:

            while not self._should_terminate():
                # Set epoch for distributed
                # sampler (important for proper shuffling)
                if self.training_mode == 'epoch_based' and sampler is not None:
                    sampler.set_epoch(self.current_epoch)

                # Create iterator for current epoch
                dataloader_iter = iter(dataloader)
                epoch_step_count = 0

                # Determine target steps for this epoch
                target_steps_per_epoch = None
                if self.actual_steps_per_epoch is not None:
                    # Use previously determined actual steps
                    target_steps_per_epoch = self.actual_steps_per_epoch
                elif estimated_steps_per_epoch is not None:
                    # Use estimated steps
                    target_steps_per_epoch = estimated_steps_per_epoch

                # Inner loop for processing batches
                while True:
                    try:
                        # Get next batch
                        batch = next(dataloader_iter)
                        epoch_step_count += 1
                    except StopIteration:
                        # End of dataloader iteration (rare for RLDS)
                        if self.training_mode == 'epoch_based':
                            overwatch.info(
                                f'Epoch {self.current_epoch} completed after {epoch_step_count} steps'  # noqa: E501
                            )
                        break

                    # Forward pass with mixed precision
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output: CausalLMOutputWithPast = self.vla(**batch)
                        loss = output.loss

                    # Scale loss for gradient accumulation
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()
                    accumulated_loss += loss.item()

                    # Perform optimizer step after gradient accumulation
                    if (self.global_step +
                            1) % self.grad_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Log metrics
                        avg_loss = accumulated_loss / self.grad_accumulation_steps  # noqa: E501
                        if self.global_step % 100 == 0 and overwatch.is_rank_zero(  # noqa: E501
                        ):
                            overwatch.info(f'[Epoch {self.current_epoch}] '
                                           f'Step {self.global_step}: '
                                           f'Loss = {avg_loss:.4f}')
                        accumulated_loss = 0

                    self.global_step += 1
                    self.steps_in_current_epoch += 1
                    progress.update(1)

                    # Check for checkpoint saving (step-based mode)
                    if self.training_mode == 'step_based' and self._should_save_checkpoint(  # noqa: E501
                    ):
                        checkpoint_path = self._save_checkpoint(
                            epoch=self.current_epoch,
                            global_step=self.global_step,
                            is_epoch_checkpoint=False)

                    # Check termination condition for step-based training
                    if self.training_mode == 'step_based' and self._should_terminate(  # noqa: E501
                    ):
                        return checkpoint_path

                    # For epoch-based training, check if epoch should end
                    if self.training_mode == 'epoch_based':
                        epoch_should_end = False

                        if target_steps_per_epoch is not None:
                            # We know the target epoch size
                            epoch_should_end = epoch_step_count >= target_steps_per_epoch  # noqa: E501

                        if epoch_should_end:
                            # Record actual steps per
                            # epoch on first epoch completion
                            if self.actual_steps_per_epoch is None:
                                self.actual_steps_per_epoch = epoch_step_count
                                # Update progress bar total with actual steps
                                new_total = self.max_epochs * self.actual_steps_per_epoch  # noqa: E501
                                progress.total = new_total
                                progress.refresh()
                                if overwatch.is_rank_zero():
                                    overwatch.info(
                                        f'Determined actual steps per epoch: {self.actual_steps_per_epoch}'  # noqa: E501
                                    )
                            break

                # Handle epoch completion
                if self.training_mode == 'epoch_based':
                    self.current_epoch += 1
                    self.steps_in_current_epoch = 0

                    # Save checkpoint at epoch intervals
                    if self._should_save_checkpoint():
                        checkpoint_path = self._save_checkpoint(
                            epoch=self.current_epoch,
                            global_step=self.global_step,
                            is_epoch_checkpoint=True)

        # Return final checkpoint path
        return checkpoint_path

    def _save_checkpoint(self, epoch, global_step, is_epoch_checkpoint):
        """Save model checkpoint with proper handling
            for LoRA and distributed training.

        This method:
        1. Saves processor and model state
        2. Merges LoRA weights if applicable
        3. Handles checkpoint naming based on training mode
        4. Manages checkpoint retention policy

        Args:
            epoch (int): Current epoch number
            global_step (int): Current global training step
            is_epoch_checkpoint (bool): Whether this is
                an epoch-based checkpoint

        Returns:
            str: Path to the saved checkpoint directory
        """
        checkpoint_dir = self.args.work_dir

        # Only main process saves the checkpoint
        if self.distributed_state.is_main_process:
            checkpoint_name = f'epoch_{epoch}' if is_epoch_checkpoint else f'step_{global_step}'  # noqa: E501
            overwatch.info(f'Saving Model Checkpoint: {checkpoint_name}')

            # Save processor and base model
            self.processor.save_pretrained(self.args.work_dir)
            self.vla.module.save_pretrained(self.args.work_dir)

        # Synchronize all processes before proceeding
        dist.barrier()

        # Handle LoRA model merging if applicable
        if self.cfg.model.use_lora:
            # Load the base model
            base_vla = AutoModelForVision2Seq.from_pretrained(
                self.cfg.model.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code)

            # Load and merge LoRA weights
            merged_vla = PeftModel.from_pretrained(base_vla,
                                                   self.args.work_dir)
            merged_vla = merged_vla.merge_and_unload()

            # Save the merged model
            if self.distributed_state.is_main_process:
                if self.save_latest_checkpoint_only:
                    # Overwrite previous checkpoint
                    merged_vla.save_pretrained(self.args.work_dir)
                    overwatch.info(
                        f'Saved latest checkpoint at: {self.args.work_dir}')
                else:
                    # Create separate checkpoint
                    # directory with descriptive name
                    if is_epoch_checkpoint:
                        checkpoint_name = f'epoch_{epoch:03d}_chkpt'
                    else:
                        checkpoint_name = f'step_{global_step:06d}_chkpt'

                    checkpoint_dir = os.path.join(self.args.work_dir,
                                                  checkpoint_name)
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # Save dataset statistics along with checkpoint
                    from fluxvla.datasets.utils import save_dataset_statistics
                    if hasattr(self, 'current_dataset_stats'):
                        save_dataset_statistics(self.current_dataset_stats,
                                                checkpoint_dir)

                    # Save processor and model
                    self.processor.save_pretrained(checkpoint_dir)
                    merged_vla.save_pretrained(checkpoint_dir)

                    overwatch.info(f'Saved checkpoint at: {checkpoint_dir}')

        # Final synchronization
        dist.barrier()

        return checkpoint_dir
