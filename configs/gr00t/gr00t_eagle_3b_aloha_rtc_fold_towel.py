# RTC config for ALOHA fold towel task.
# Training: rtc_training_config in vla_head.
# Inference: AlohaRTCInferenceRunner with rtc_config and postprocessing.

_base_ = './gr00t_eagle_3b_aloha_full_finetune_fold_towel.py'

# --- Training: add RTC noise conditioning ---
model = dict(
    vla_head=dict(
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',
        )))

runner = dict(max_epochs=1)

# --- Inference: async + prefix RTC ---
inference = dict(
    type='AlohaRTCInferenceRunner',
    async_execution=True,
    horizon_config=dict(dynamic=True, max_horizon=28),
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=5,
    ),
    postprocess_config=dict(
        enabled=True,
        method='joint_mpc',
        mode='tracking',
        num_stitch=6,
        max_velocity=5.0,
        max_acceleration=20.0,
        max_jerk=20.0,
        tracking_weight=3.0,
    ))
