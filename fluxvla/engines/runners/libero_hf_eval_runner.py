import json
import os
import time
from typing import Dict

import numpy as np
import torch
import tqdm
from libero.libero import benchmark

from fluxvla.engines.utils.eval_utils import (get_libero_dummy_action,
                                              get_libero_env, get_libero_image,
                                              get_vla_action, quat2axisangle,
                                              save_rollout_video)
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.robot_utils import (invert_gripper_action,
                                               normalize_gripper_action)
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils.root import RUNNERS


@RUNNERS.register_module()
class HFLiberoEvalRunner:
    """Runner for evaluating models using Hugging Face Transformers.
    This class sets up the evaluation environment, loads the model,
    and runs the evaluation process.

    Args:
        cfg (Dict): Configuration dictionary containing model and
            evaluation settings.
        processor (Dict): Configuration dictionary for the processor.
        seed (int): Random seed for reproducibility.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family for evaluation.
        task_suite_name (str): Name of the task suite for evaluation.
        trust_remote_code (bool): Whether to trust remote code when
            loading the model. Defaults to True.
        resize_size (int): Size to which images will be resized.
            Defaults to 224.
        num_trials_per_task (int): Number of trials to run per task.
            Defaults to 50.
        num_steps_wait (int): Number of steps to wait before starting
            the evaluation. Defaults to 10.
        attn_implementation (str): Attention implementation to use.
            Defaults to 'flash_attention_2'.
        torch_dtype (str): Data type for PyTorch tensors. Defaults to 'bf16'.
        load_in_8bit (bool): Whether to load the model in 8-bit precision.
            Defaults to False.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
            Defaults to False.
        low_cpu_mem_usage (bool): Whether to use low CPU memory usage
            when loading the model. Defaults to True.
    """

    def __init__(self,
                 cfg: Dict,
                 processor: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_suite_name: str,
                 trust_remote_code: bool = True,
                 resize_size: int = 224,
                 num_trials_per_task: int = 50,
                 num_steps_wait: int = 10,
                 attn_implementation: str = 'flash_attention_2',
                 torch_dtype: str = 'bf16',
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 low_cpu_mem_usage: bool = True,
                 center_crop: bool = True):
        from fluxvla.engines import build_processor_from_cfg
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.model_family = model_family
        self.task_suite_name = task_suite_name
        self.trust_remote_code = trust_remote_code
        self.processor = build_processor_from_cfg(processor)
        self.resize_size = resize_size
        self.num_trials_per_task = num_trials_per_task
        self.num_steps_wait = num_steps_wait
        self.attn_implementation = attn_implementation
        self.torch_dtype = str_to_dtype(torch_dtype)  # noqa: E501
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.center_crop = center_crop

    def run_setup(self):
        """Set up the evaluation environment and model."""
        set_seed_everywhere(self.seed)
        from transformers import (AutoConfig, AutoImageProcessor,
                                  AutoModelForVision2Seq, AutoProcessor)

        from fluxvla.models.hf_models.configs import register_dict

        # Register model components
        AutoConfig.register(
            register_dict[self.cfg['model']['type']]['name'],
            register_dict[self.cfg['model']['type']]['config_class'])
        AutoImageProcessor.register(
            register_dict[self.cfg['model']['type']]['config_class'],
            register_dict[self.cfg['model']['type']]['image_processor_class'])
        AutoProcessor.register(
            register_dict[self.cfg['model']['type']]['config_class'],
            register_dict[self.cfg['model']['type']]['processor_class'])
        AutoModelForVision2Seq.register(
            register_dict[self.cfg['model']['type']]['config_class'],
            register_dict[self.cfg['model']['type']]['model_class'])

        # Load the model
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.ckpt_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True)
        self.vla.eval()
        self.vla.to('cuda:0')
        dataset_statistics_path = os.path.join(self.ckpt_path,
                                               'dataset_statistics.json')

        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, 'r') as f:
                norm_stats = json.load(f)
            self.vla.norm_stats = norm_stats
        else:
            print(
                'WARNING: No local dataset_statistics.json file found for current checkpoint.\n'  # noqa: E501
                'You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.'  # noqa: E501
                'Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.'  # noqa: E501
            )

    def run(self):
        """Run the evaluation process."""
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f'Task suite: {self.task_suite_name}')
        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = f'EVAL-{self.task_suite_name}-{self.model_family}-{data_time}'  # noqa: E501
        local_log_filepath = os.path.join(self.ckpt_path, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        total_episodes, total_successes = 0, 0
        unnorm_key = self.task_suite_name
        if self.model_family == 'openvla':
            # In some cases, the key must be manually modified (e.g. after
            # training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            if unnorm_key not in self.vla.norm_stats and f'{unnorm_key}_no_noops' in self.vla.norm_stats:  # noqa: E501
                unnorm_key = f'{unnorm_key}_no_noops'
            assert unnorm_key in self.vla.norm_stats, f'Action un-norm key {unnorm_key} not found in VLA `norm_stats`!'  # noqa: E501
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, resolution=256)
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(self.num_trials_per_task)):
                print(f'\nTask: {task_description}')
                log_file.write(f'\nTask: {task_description}\n')

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                if self.task_suite_name == 'libero_spatial':
                    max_steps = 220  # longest training demo has 193 steps
                elif self.task_suite_name == 'libero_object':
                    max_steps = 280  # longest training demo has 254 steps
                elif self.task_suite_name == 'libero_goal':
                    max_steps = 300  # longest training demo has 270 steps
                elif self.task_suite_name == 'libero_10':
                    max_steps = 520  # longest training demo has 505 steps
                elif self.task_suite_name == 'libero_90':
                    max_steps = 400  # longest training demo has 373 steps

                print(f'Starting episode {task_episodes+1}...')

                log_file.write(f'Starting episode {task_episodes+1}...\n')
                while t < max_steps + self.num_steps_wait:
                    # IMPORTANT: Do nothing for the first
                    # few timesteps
                    # because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < self.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action())
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, self.resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        'full_image':
                        img,
                        'state':
                        np.concatenate((obs['robot0_eef_pos'],
                                        quat2axisangle(obs['robot0_eef_quat']),
                                        obs['robot0_gripper_qpos'])),
                    }
                    action = get_vla_action(
                        self.vla,
                        self.processor,
                        self.ckpt_path,
                        observation,
                        task_description,
                        unnorm_key,
                        device=self.vla.device,
                        center_crop=self.center_crop)
                    action = normalize_gripper_action(action, binarize=True)
                    if self.model_family == 'openvla':
                        action = invert_gripper_action(action)
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                    # except Exception as e:
                    #     print(f'Error during action prediction: {e}')
                    #     log_file.write(f'Caught exception: {e}\n')
                    #     action = get_libero_dummy_action()

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    work_dir=self.ckpt_path,
                    log_file=log_file)

                # Log current results
                print(f'Success: {done}')
                print(f'# episodes completed so far: {total_episodes}')
                print(
                    f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)'  # noqa: E501
                )
                log_file.write(f'Success: {done}\n')
                log_file.write(
                    f'# episodes completed so far: {total_episodes}\n')
                log_file.write(
                    f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n'  # noqa: E501
                )
                log_file.flush()
