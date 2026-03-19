# Real-Time Chunking (RTC)

Real-Time Chunking (RTC) uses the previously predicted action sequence as a known prefix to constrain the current denoising process, reducing inconsistencies between consecutive action chunks and improving smoothness under asynchronous execution.

> **Reference**: [Physical Intelligence Kinetix — Real-Time Chunking](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)

## Overview

Standard action chunking independently generates a full action sequence at each inference step, which can cause jitter between adjacent chunks. RTC addresses this with two key ideas:

1. **Training**: Randomly simulate execution delays so the model learns to predict future actions given a known prefix of already-executed actions.
2. **Inference**: Lock the not-yet-finished portion of the previous prediction as a prefix in the current denoising process, ensuring coherence between new and historical actions.

FluxVLA supports two RTC inference modes:

| Mode       | Requires RTC Training | Mechanism                                                                                                        |
| ---------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `prefix`   | Yes                   | Locks prefix positions to clean time during denoising, directly splicing known actions                           |
| `guidance` | No                    | Corrects the velocity field to align the denoising trajectory with the known prefix (no special training needed) |

## Configuration

RTC is enabled by layering two config sections on top of any existing training config. Below are the templates and parameter descriptions.

### Training Config (`rtc_training_config`)

Add to `model.vla_head`:

```python
model = dict(
    vla_head=dict(
        rtc_training_config=dict(
            enabled=True,        # Enable RTC training
            max_delay=7,         # Max simulated delay steps
            distribution='exponential',  # Delay sampling distribution:
                                         #   'exponential' — favors small delays (recommended)
                                         #   'uniform'     — uniform distribution
        )))
```

**How it works**: During training, a delay value `d ∈ [0, max_delay)` is randomly sampled per batch element. The first `d` steps of the action sequence have their time set to clean (known), and these positions are masked out from the loss.

### Inference Config (`inference`)

RTC inference uses `AlohaRTCInferenceRunner` (inherits from `AlohaInferenceRunner`) with a dedicated `rtc_config`. Add at the top level of the config file:

```python
inference = dict(
    type='AlohaRTCInferenceRunner',  # Use RTC runner instead of AlohaInferenceRunner
    async_execution=True,   # Enable async execution (inference and execution run in parallel)
    execute_horizon=10,     # Number of action steps to execute per cycle
    rtc_config=dict(
        enabled=True,
        method='prefix',    # RTC mode:
                            #   'prefix'   — requires RTC training, locks prefix (recommended)
                            #   'guidance'  — no training needed, velocity field guidance
        prefix_len=5,       # Prefix length (number of locked known action steps)

        # ---- guidance mode only ----
        # decay_end=10,           # Position where guidance weight decays to 0
        # schedule='exp',         # Decay schedule: 'exp', 'linear', 'ones', 'zeros'
        # max_guidance_weight=5.0, # Upper bound for guidance weight
        # use_vjp=False,          # Use VJP correction (more precise but slower)
    ))
```

### Full Example: Enabling RTC on an Existing Config

Using PI0.5 + ALOHA as an example:

```python
_base_ = './pi05_paligemma_aloha_full_finetune.py'

# Training: add RTC noise conditioning
model = dict(
    vla_head=dict(
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',
        )))

# RTC can be fine-tuned on a pretrained checkpoint, e.g. 1 epoch
runner = dict(max_epochs=1)

# Inference: use RTC runner with async execution + prefix RTC
inference = dict(
    type='AlohaRTCInferenceRunner',
    async_execution=True,
    execute_horizon=10,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=5,
    ))
```

For GR00T or other models, simply replace `_base_` with the corresponding base training config. The RTC parameters are model-agnostic.

## Testing

Use `scripts/test_rtc.py` to visualize the RTC denoising process:

```bash
python scripts/test_rtc.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --checkpoint /path/to/checkpoint.pt \
    --prefix_len 5 \
    --output_dir work_dirs/rtc_test
```

## Supported Models

| Model                       | RTC Training | Prefix Inference | Guidance Inference |
| --------------------------- | ------------ | ---------------- | ------------------ |
| FlowMatchingHead (GR00T)    | ✅           | ✅               | ✅                 |
| PI0FlowMatching (PI0/PI0.5) | ✅           | ✅               | ✅                 |
