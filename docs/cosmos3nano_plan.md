# Plan: Add cosmos3-nano Training & Inference to FluxVLA

## Context

FluxVLA currently supports GR00T (Eagle backbone + FlowMatchingHead) and DreamZero (WanBackbone + CausalWanModel DiT) on robot data. The goal is to add **cosmos3-nano** — a Qwen3-VL-8B-based MoT (Mixture of Transformers) world foundation model from `/root/projects/cosmos-framework` — for training and inference on robot data (same LeRobot v2.1 parquet format as HUD04/ALOHA). The data pipeline specifically follows `DROIDLeRobotDataset`'s approach (video + action + domain_id + SequencePlan) while reusing FluxVLA's `ParquetDataset` infrastructure.

**Key architectural difference from existing models**: cosmos3-nano encodes video through a Wan2.2 VAE *during the model forward pass* (not in the dataset), uses full sequence-packing (`PackedSequence`), and performs *joint* video+action flow matching through the same Transformer backbone (no separate VLM backbone + action head split).

______________________________________________________________________

## Architecture Overview

```
ParquetDataset (LeRobot v2.1 parquet)
  → ProcessParquetInputs  (video frames, states, actions — existing)
  → ParquetPrompter        (task description — existing)
  → ProcessCosmos3NanoPrompt  [NEW] — Qwen3VL text-only tokenization
  → ResizeImages           (256px — existing, but pick cosmos3 resolution)
  → SimpleNormalizeImages  ([-1,1] for VAE — existing)
  → NormalizeStatesAndActions (state/action normalization — existing)
  → BuildCosmos3NanoSequence  [NEW] — set domain_id, pad action→64d, build SequencePlan
→ DistributedRepeatingDataset
→ FSDPTrainRunner → DictCollator (keys updated)
→ Cosmos3NanoVLA.forward()        [NEW VLA class]
    ├── vae.encode(video) → latents
    ├── pack_input_sequence() → PackedSequence
    ├── RectifiedFlow.get_interpolation() → add noise to action + video latents
    ├── Cosmos3VFMNetwork.forward(PackedSequence) → preds_action, preds_vision
    └── compute_flow_matching_loss(preds_action, target) → loss dict
```

______________________________________________________________________

## Files to Create

### 1. `fluxvla/models/vlas/cosmos3nano_vla.py` — `Cosmos3NanoVLA`

**Class**: `@VLAS.register_module() class Cosmos3NanoVLA(BaseVLA)`

**Constructor params**:

- `pretrained_name_or_path: str` — cosmos3-nano checkpoint path
- `max_action_dim: int = 64` — padded action dim
- `action_loss_weight: float = 10.0` — from Nano config
- `num_inference_steps: int = 20`
- `shift: dict = {"256": 3, "480": 5}` — resolution-dependent RF shift
- `train_time_action_distribution: str = "logitnormal"` — action timestep sampler
- `independent_action_schedule: bool = True`
- `resolution: str = "256"` — target video resolution for packing

**Initialization** (mirrors OmniMoTModel.set_up\_\*):

```python
self.net = Cosmos3VFMNetwork.from_pretrained(...)   # from cosmos_framework
self.vae = VideoTokenizerInterface.from_pretrained(...)  # Wan2.2 VAE
self.qwen_tokenizer = create_qwen2_tokenizer_with_download(...)
self.rectified_flow_action = RectifiedFlow(train_time_distribution="logitnormal", ...)
self.rectified_flow_vision = RectifiedFlow(train_time_distribution="waver", ...)
self.special_tokens = add_special_tokens(self.qwen_tokenizer)
```

**`forward(self, video, text_token_ids, actions, domain_id, sequence_plan, raw_action_dim, ...)` — training**:

1. VAE encode: `x0_latents = [self.vae.encode(v) for v in video]`  — `[C=48, T//4, H//16, W//16]`
2. Build `GenerationDataClean(x0_tokens_vision=x0_latents, x0_tokens_action=actions, ...)`
3. Sample timesteps: `t_action = self.rectified_flow_action.sample_train_time(B)`; `t_vision = ...`
4. Add noise: `gen_noised = self._add_noise(gen_clean, t_vision, t_action)`
5. Pack: `packed_seq = pack_input_sequence(sequence_plan, text_token_ids, gen_clean, t_action, self.special_tokens, ...)`  — import from `cosmos_framework.data.vfm.sequence_packing`
6. Forward: `out = self.net(packed_seq, fps_vision=..., fps_action=...)`
7. Loss: `action_loss = compute_flow_matching_loss(out["preds_action"], gen_noised.vt_target_action, ..., raw_action_dim=raw_action_dim)`; optional `vision_loss`
8. Return: `{"loss": action_loss + vision_loss, "action_loss": action_loss.detach(), ...}`

**`predict_action(self, video, text_token_ids, domain_id, ...)` — inference** (policy mode):

1. VAE encode current frame(s): `latents = self.vae.encode(video)`
2. `seq_plan = SequencePlan(has_vision=True, has_action=True, condition_frame_indexes_vision=[0, ..., T-1], condition_frame_indexes_action=[])`  — all video frames conditioned, action fully denoised
3. Build `gen_clean` with clean video latents + zero action
4. Sample action from noise with `FixedStepSampler` / `UniPCSampler` (N=20 steps):
   - Build `packed_seq` with noisy action at current step
   - `out = self.net(packed_seq)`
   - Update action via sampler step
5. Unpack action: `action = out["preds_action"][0][:, :raw_action_dim]`
6. Return `action` `[T_act, D]`

**`get_fsdp_wrapping_policy()`**: wrap `Qwen3VLDecoderLayer` and `DomainAwareLinear` at minimum.

**`freeze_backbones(cfg)`**: optionally freeze `self.net.language_model` (VLM layers), keep `moe_gen` layers, `time_embedder`, `vae2llm`, `llm2vae`, `action2llm`, `llm2action` trainable (follows cosmos nano optimizer `keys_to_select`).

______________________________________________________________________

### 2. `fluxvla/transforms/transform_cosmos3nano.py` — Two new transforms

#### `ProcessCosmos3NanoPrompt`

- **Purpose**: Tokenize task description text for cosmos3-nano (Qwen3VL chat template, text-only, no image placeholder tokens — unlike Eagle's `ProcessPromptsWithImage`)
- **Input**: `data['task_description']` (str)
- **Output**: `data['text_token_ids']` (np.int64 array, truncated to `max_len=4096`)
- **Implementation**: calls `cosmos_framework.model.vfm.vlm.qwen3_vl.utils.tokenize_caption(caption, tokenizer, add_vision_id=False)`; tokenizer initialized from `qwen3_vl_model_path`
- **Registered as**: `@TRANSFORMS.register_module()`

#### `BuildCosmos3NanoSequence`

- **Purpose**: Finalize cosmos3-compatible batch fields from normalized data
- **Params**: `max_action_dim=64`, `embodiment_to_domain_id: dict`, `mode="policy"` (or `"joint"` for random sampling of all three modes)
- **Input**: data with `actions [T, D]`, `embodiment_ids`, optional `proprio`
- **Output**:
  - `data['actions']` padded to `[T, max_action_dim]` via `pad_action_to_max_dim()`
  - `data['domain_id']` = `embodiment_to_domain_id[embodiment_id]`
  - `data['raw_action_dim']` = original action dim (for loss masking)
  - `data['sequence_plan']` = `SequencePlan(has_vision=True, has_action=True, condition_frame_indexes_vision=[0], condition_frame_indexes_action=[])` for policy mode
  - `data['conditioning_fps']` = dataset fps
- **Registered as**: `@TRANSFORMS.register_module()`

______________________________________________________________________

### 3. `fluxvla/collators/cosmos3nano_collator.py` — `Cosmos3NanoCollator`

- **Purpose**: Handle `SequencePlan` objects (dataclass, not tensor-collatable) and variable-length `text_token_ids`
- **Inherits from**: basic collator logic
- **`__call__(batch)`**:
  - Tensors: `video [B, C, T, H, W]`, `actions [B, T_act, 64]`, `domain_id [B]`, `raw_action_dim [B]`
  - Text: pad `text_token_ids` to `max_len` → `[B, L]`
  - List pass-through: `sequence_plan` (list of SequencePlan objects)
  - Meta: `task_description`, `stats`, `info`
- **Registered as**: `@COLLATORS.register_module()`

______________________________________________________________________

### 4. `fluxvla/engines/runners/cosmos3nano_inference_runner.py` — `Cosmos3NanoInferenceRunner`

Follows `AlohaInferenceRunner` pattern:

- `run_setup()`: load model, build inference dataset (`Cosmos3NanoInferenceDataset`)
- `run()`: main inference loop (ROS observation → predict_action → execute)
- `_predict_action(inputs)`: calls `vla.predict_action(**inputs)`
- `_postprocess_actions(raw_action)`: `DenormalizePrivateAction` → clip → threshold

New class `Cosmos3NanoInferenceDataset` (follows `PrivateInferenceDataset` pattern):

- Runs `ProcessCosmos3NanoPrompt` + `ResizeImages` + `SimpleNormalizeImages` + `NormalizeStatesAndActions` + `BuildCosmos3NanoSequence`
- Returns batch dict ready for `Cosmos3NanoVLA.predict_action()`

**Registered as**: `@RUNNERS.register_module()` and `@DATASETS.register_module()`.

______________________________________________________________________

### 5. `configs/cosmos3nano/cosmos3nano_hud04_full_finetune.py` — Training config

```python
_ckpt_root = './checkpoints'
_cosmos3_nano_ckpt = _ckpt_root + '/Cosmos3-Nano'   # path to cosmos3-nano weights
_qwen3_vl_path = _cosmos3_nano_ckpt + '/qwen3_vl'

_action_horizon = 24
_frame_window_size = 5    # 1 conditioning + 4 future frames (start small, expand later)
_image_height = 256
_image_width = 256

model = dict(
    type='Cosmos3NanoVLA',
    pretrained_name_or_path=_cosmos3_nano_ckpt,
    max_action_dim=64,
    action_loss_weight=10.0,
    resolution='256',
    num_inference_steps=20,
    freeze_vlm_layers=False,   # full finetune; set True for head-only
)

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={'observation.state': ['proprio'], 'action': ['action']},
        statistic_keys=['observation.state', 'timestamp', 'action'],
        datasets=dict(
            basket=[dict(
                type='ParquetDataset',
                data_root_path=[...same HUD04 paths as dreamzero_hud04_full_finetune.py...],
                transforms=[
                    dict(type='ProcessParquetInputs',
                         parquet_keys=['observation.state', 'timestamp', 'actions', 'info', 'stats', 'action_masks'],
                         video_keys=['observation.images.head', 'observation.images.left_wrist'],
                         name_mappings={'observation.state': ['states'], 'actions': ['actions']},
                         embodiment_id=0),
                    dict(type='ParquetPrompter', use_conversation=False),
                    dict(type='ProcessCosmos3NanoPrompt',
                         qwen3_vl_model_path=_qwen3_vl_path,
                         max_len=4096),
                    dict(type='ResizeImages', height=_image_height, width=_image_width),
                    dict(type='SimpleNormalizeImages'),   # [-1, 1] for VAE input
                    dict(type='NormalizeStatesAndActions',
                         action_dim=64, state_dim=64,
                         state_key='proprio', action_key='action',
                         norm_type='mean_std'),
                    dict(type='BuildCosmos3NanoSequence',
                         max_action_dim=64,
                         embodiment_to_domain_id={0: 8, 1: 8},   # map to droid_lerobot domain
                         mode='policy',
                         frame_window_size=_frame_window_size),
                    dict(type='PrepareVideo', num_views=2, frame_window_size=_frame_window_size),
                ],
                action_window_size=_action_horizon,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                frame_window_size=_frame_window_size,
            )],
            # candy=[...similar...]
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=0.1,
    collator=dict(
        type='Cosmos3NanoCollator',
        tensor_keys=['video', 'actions', 'domain_id', 'raw_action_dim'],
        sequence_keys=['text_token_ids'],
        list_keys=['sequence_plan'],
        meta_keys=['task_description', 'stats', 'info', 'timestamp'],
    ),
    sampler=None,
    metric=dict(type='VLAMetric', active_trackers=('jsonl', 'wandb'), ...),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.05,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
)

inference = dict(
    type='Cosmos3NanoInferenceRunner',
    seed=7,
    task_descriptions={...},
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='Cosmos3NanoInferenceDataset',
        qwen3_vl_model_path=_qwen3_vl_path,
        img_keys=['head', 'left_wrist'],
        embodiment_id=0,
        domain_id=8,
        max_action_dim=64,
        transforms=[...resize, normalize, build_sequence...],
    ),
    denormalize_action=dict(type='DenormalizePrivateAction', norm_type='mean_std', action_dim=52),
    operator=dict(type='Teleop02WbtOperator', ...),
)
```

Also add: `configs/cosmos3nano/cosmos3nano_hud04_debug_single_gpu.py` for rapid iteration (fewer frames, smaller resolution).

______________________________________________________________________

## Files to Modify

### `fluxvla/models/vlas/__init__.py`

Add (with `try/except ImportError` like DreamZeroVLA):

```python
try:
    from .cosmos3nano_vla import Cosmos3NanoVLA  # noqa: F401
except ImportError:
    pass
```

### `fluxvla/transforms/__init__.py`

Add:

```python
from .transform_cosmos3nano import *
```

### `fluxvla/collators/__init__.py`

Add:

```python
from .cosmos3nano_collator import Cosmos3NanoCollator  # noqa: F401
```

### `fluxvla/engines/runners/__init__.py` (or equivalent)

Add import for `Cosmos3NanoInferenceRunner`.

### `requirements.txt`

Add:

```
cosmos-framework @ file:///root/projects/cosmos-framework[train]
```

(or install as editable dep; cosmos-framework requires `lerobot`, `qwen-vl-utils`, `flash-attn` from its `[train]` extras)

______________________________________________________________________

## Critical Implementation Details

### PackedSequence Construction

`pack_input_sequence()` from cosmos-framework expects:

- `sequence_plans: list[SequencePlan]` — one per batch sample
- `input_text_indexes: list[list[int]]` — tokenized text IDs per sample
- `gen_data_clean: GenerationDataClean` — with video latents + action tokens
- `input_timesteps: torch.Tensor` — per-sample diffusion timesteps
- `special_tokens: dict` — `{start_of_generation, end_of_generation}`

Import directly:

```python
from cosmos_framework.data.vfm.sequence_packing import pack_input_sequence, SequencePlan
from cosmos_framework.model.vfm.utils.data_and_condition import GenerationDataClean, GenerationDataNoised
from cosmos_framework.model.vfm.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.vfm.diffusion.rectified_flow import RectifiedFlow, TrainTimeSampler
```

### Video Format for VAE

- Dataset outputs `video: [N_views, C, T, H, W]` float32 in `[-1, 1]` (after `SimpleNormalizeImages`)
- Multiple views are stacked vertically (same as DROIDLeRobotDataset `concat_view`): `torch.cat([views], dim=-2)`
- VAE encodes: `[C=3, T, H_concat, W]` → `[C=48, T//4, H//16, W//16]`
- For `frame_window_size=5`: T=5, H=256, W=512 (2 views stacked horizontally) → latent `[48, 2, 16, 32]`

### Action Normalization Chain

cosmos3-nano uses **quantile normalization** for DROID; for HUD04 we use `mean_std` (matching DreamZero HUD04). The `NormalizeStatesAndActions` in FluxVLA handles this. The normalized action is then **padded to 64 dims** by `BuildCosmos3NanoSequence`.

### Domain ID Mapping

HUD04 data uses `embodiment_id=0` (basket) and `embodiment_id=1` (candy). Map to `domain_id=8` (droid_lerobot's ID, which shares a common 10D franka-like action space). This can be refined if HUD04 needs a distinct domain ID registered in `EMBODIMENT_TO_DOMAIN_ID`.

### Flow Matching Loss Masking

`compute_flow_matching_loss()` uses `raw_action_dim` to mask out the padded dimensions of the action. Since HUD04 has `action_dim=52`, only the first 52 dims contribute to the loss.

### Checkpoint Loading

cosmos3-nano uses HuggingFace-style `from_pretrained()`. The existing FluxVLA `load_checkpoint_into_model()` / streaming safetensors path may not apply. Cosmos3NanoVLA should override `load_pretrained()` to call `Cosmos3VFMNetwork.from_pretrained(path)`.

______________________________________________________________________

## Reused Existing Code

| Component                               | Reused From                                                  |
| --------------------------------------- | ------------------------------------------------------------ |
| `ParquetDataset`                        | `fluxvla/datasets/parquet_dataset.py`                        |
| `DistributedRepeatingDataset`           | `fluxvla/datasets/dataset_wrapper.py`                        |
| `ProcessParquetInputs`                  | `fluxvla/transforms/transform_inputs.py`                     |
| `ParquetPrompter`                       | `fluxvla/transforms/prompters.py`                            |
| `ResizeImages`, `SimpleNormalizeImages` | `fluxvla/transforms/transform_images.py`                     |
| `NormalizeStatesAndActions`             | `fluxvla/transforms/normalize.py`                            |
| `PrepareVideo`                          | `fluxvla/transforms/transform_images.py`                     |
| `DenormalizePrivateAction`              | `fluxvla/transforms/normalize.py`                            |
| `FSDPTrainRunner`                       | `fluxvla/engines/runners/fsdp_train_runner.py`               |
| `BaseVLA`                               | `fluxvla/models/vlas/base_vla.py`                            |
| `compute_flow_matching_loss`            | `cosmos_framework/model/vfm/algorithm/loss/flow_matching.py` |
| `pack_input_sequence`, `SequencePlan`   | `cosmos_framework/data/vfm/sequence_packing.py`              |
| `RectifiedFlow`, `TrainTimeSampler`     | `cosmos_framework/model/vfm/diffusion/rectified_flow.py`     |
| `Cosmos3VFMNetwork`                     | `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`      |
| `DomainAwareLinear`                     | `cosmos_framework/model/vfm/mot/domain_aware_linear.py`      |
| `pad_action_to_max_dim`                 | `cosmos_framework/data/vfm/action/transforms.py`             |
| `tokenize_caption`                      | `cosmos_framework/model/vfm/vlm/qwen3_vl/utils.py`           |

______________________________________________________________________

## Verification

### Step 1: Unit test — data pipeline

```bash
python -c "
from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.transforms.transform_cosmos3nano import ProcessCosmos3NanoPrompt, BuildCosmos3NanoSequence
# Build dataset with new transforms, print one sample's keys/shapes
"
```

Expected: `text_token_ids [L]`, `video [N*C, T, H, W]`, `actions [T_act, 64]`, `domain_id scalar`, `sequence_plan SequencePlan`

### Step 2: Unit test — model forward

```bash
python -c "
from fluxvla.models.vlas.cosmos3nano_vla import Cosmos3NanoVLA
model = Cosmos3NanoVLA(pretrained_name_or_path='./checkpoints/Cosmos3-Nano', ...)
batch = {...}  # mock batch
loss = model(**batch)
print(loss)
"
```

### Step 3: Debug single-GPU training

```bash
torchrun --nproc-per-node=1 scripts/train.py \
  --config configs/cosmos3nano/cosmos3nano_hud04_debug_single_gpu.py
```

Check: loss decreases over a few steps, no OOM, action_loss logged.

### Step 4: Full training run

```bash
torchrun --nproc-per-node=8 scripts/train.py \
  --config configs/cosmos3nano/cosmos3nano_hud04_full_finetune.py
```

### Step 5: Inference check

Adapt `tools/check_dreamzero_predictions.py` → `tools/check_cosmos3nano_predictions.py`:

```bash
python tools/check_cosmos3nano_predictions.py \
  --config configs/cosmos3nano/cosmos3nano_hud04_full_finetune.py \
  --checkpoint ./work_dirs/.../checkpoint_epoch_2 \
  --num-samples 20
```

Verify predicted actions overlap with ground truth actions in CSV/NPZ output.
