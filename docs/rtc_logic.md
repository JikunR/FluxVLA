# RTC (Real-Time Chunking) 训练与推理逻辑

基于配置文件 `configs/gr00t/gr00t_hud04_rtc_full_finetune.py` 的完整 RTC 逻辑说明。

参考：[Physical Intelligence Kinetix — simulated_delay](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)

---

## 1. 背景：为什么需要 RTC

标准 action chunking 推理模式下，模型每次生成一个完整的动作序列（chunk），执行完再生成下一个。存在两个问题：

1. **chunk 边界不连续**：相邻 chunk 之间没有信息传递，轨迹拼接处可能出现突变。
2. **推理延迟浪费**：模型推理期间机器人要么停下来等，要么盲目执行旧动作。

RTC 通过让模型在生成新 chunk 时**感知上一轮已执行的动作**，实现滚动推理（rolling inference），使得 chunk 之间平滑过渡。

---

## 2. 当前配置总览

```python
# ============ 训练配置（model.vla_head）============
rtc_training_config=dict(
    enabled=True,
    max_delay=7,                # 最大延迟步数（exclusive），即 delay ∈ [0, 6]
    distribution='exponential', # 指数分布，偏好小延迟
)

# ============ 推理配置（inference）============
rtc_config=dict(
    enabled=True,
    method='prefix',            # 前缀替换法
    prefix_len=None,            # 动态估算（基于上一轮推理耗时）
)
```

关键模型参数：
- `num_steps = 32`：每个 chunk 包含 32 步动作
- `action_dim = 64`（内部），`ori_action_dim = 42`（实际输出维度）
- `num_inference_timesteps = 4`：去噪迭代步数
- `publish_rate = 30`（默认）：动作发布频率 30Hz
- `async_execution = True`：异步执行（推理和执行并行）

---

## 3. 训练时 RTC 逻辑

**目标**：让模型学会"看到前几步已知动作后，接着生成后续动作"的能力。

### 3.1 代码入口

`FlowMatchingHead.forward()`（`fluxvla/models/heads/flow_matching_head.py:281`）

### 3.2 流程

#### Step 1: 采样随机噪声时间

```python
t_scalar = self.sample_time(B, device, dtype)  # shape: (B,), 值域 [0, 1]
```

Flow Matching 中 `t=0` 表示纯噪声，`t=1` 表示干净数据。

#### Step 2: 采样随机延迟

```python
# fluxvla/engines/utils/rtc_training.py:28
delays = sample_training_delay(
    batch_size=B,
    max_delay=7,              # 你的配置
    distribution='exponential',
    device=device
)  # shape: (B,), 每个样本一个整数 delay ∈ [0, 6]
```

指数分布的权重：`w = exp([6, 5, 4, 3, 2, 1, 0])`，归一化后小延迟（delay=0）概率最高，大延迟（delay=6）概率最低。符合实际推理场景（大多数时候推理很快，前缀较短）。

#### Step 3: 修改 per-position 时间和 loss mask

```python
# fluxvla/engines/utils/rtc_training.py:64
t, action_masks = apply_rtc_time_conditioning(
    t_scalar, action_masks, delays, T=32, clean_time=1.0
)  # t: (B, T), action_masks: (B, T)
```

对于每个样本，假设 `delay=3`：

| 位置 | 0 | 1 | 2 | 3 | 4 | ... | 31 |
|------|---|---|---|---|---|-----|-----|
| **时间 t** | 1.0 | 1.0 | 1.0 | t_scalar | t_scalar | ... | t_scalar |
| **action_mask** | 0.0 | 0.0 | 0.0 | 原值 | 原值 | ... | 原值 |

- 前 3 步（delay 位置）：`t=1.0`（clean_time）
- 其余位置：使用原始采样时间

#### Step 4: 构造带噪轨迹

```python
noisy_trajectory = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * actions
```

对于前缀位置（`t=1.0`）：

```
x = (1 - 1.0) * noise + 1.0 * actions = actions  # 前缀位置直接是真实动作
```

对于其余位置（`t=t_scalar`）：正常的噪声插值。

#### Step 5: 计算 loss

```python
velocity = actions - noise
loss = F.mse_loss(pred, velocity, reduction='none') * action_masks.unsqueeze(-1)
loss = loss.sum() / (action_masks.sum() * action_dim)
```

前缀位置的 `action_mask=0.0`，**不参与 loss 计算**。模型只需学习预测非前缀位置的速度场。

### 3.3 训练时 RTC 示意图

```
样本 i:  delay=3, t_scalar=0.6

位置:        [0]    [1]    [2]    [3]    [4]   ...  [31]
             ───前缀（已知）───    ───────需要预测─────────
时间 t:      1.0    1.0    1.0    0.6    0.6   ...   0.6
输入 x_t:    a_gt   a_gt   a_gt   noisy  noisy ...  noisy
loss mask:    0      0      0      1      1    ...    1
             ↑不计算loss          ↑正常计算loss
```

**效果**：模型学会了——当 chunk 开头几步是"干净的已知动作"时，利用这些信息来更好地预测后续动作。

---

## 4. 推理时 RTC 逻辑

**目标**：利用上一轮已预测的动作片段作为前缀，约束当前 chunk 的生成，实现平滑衔接。

### 4.1 代码入口

`Teleop02WbtRTCInferenceRunner._predict_action()`（`teleop02_wbt_rtc_inference_runner.py:68`）

### 4.2 主循环时序

```
           ┌─ preprocess ─┬─ predict_action ─┬─ postprocess ─┬─ execute_actions ─┐
chunk 0:   │   获取观测     │   模型推理        │  反归一化      │  发送到机器人      │
           └──────────────┴──────────────────┴───────────────┴──────────────────┘
           ↓ _prev_ctx = _action_ctx
           ┌─ preprocess ─┬─ predict_action（带 RTC）─┬─ postprocess ─┬── ...
chunk 1:   │   获取观测     │  注入 prev_actions       │               │
           └──────────────┴─────────────────────────┴───────────────┴──
```

每轮结束时 `self._prev_ctx = self._action_ctx`（第 211 行），将当前上下文传递给下一轮。

### 4.3 prefix 构造流程

#### Step 1: 计算时间偏移

```python
offset = (ctx.inference_start - prev.action_timestamp) / self._dt
```

- `prev.action_timestamp`：上一轮动作开始执行的时间戳
- `ctx.inference_start`：当前轮推理开始的时间戳
- `self._dt = 1.0 / publish_rate`：单步动作的时间间隔

`offset` 表示从上一轮开始执行到现在经过了多少步（浮点数）。

#### Step 2: 截取剩余轨迹

```python
remaining = resample_remaining(prev.raw_actions[0], offset)[None]
# prev.raw_actions shape: (1, 32, action_dim)
# prev.raw_actions[0] shape: (32, action_dim)
# remaining shape: (1, M, action_dim)，M = 32 - int(offset)
```

`resample_remaining` 通过线性插值从上一轮轨迹中截取尚未执行完的部分（处理浮点偏移的对齐）。

#### Step 3: 确定 prefix_len

当前配置 `prefix_len=None`，使用**动态估算**：

```python
prefix_len = int(prev.inference_elapsed * self.publish_rate)
```

即：上一轮模型推理花了多长时间，这段时间内机器人大约执行了多少步。

然后做边界保护：

```python
prefix_len = min(prefix_len, remaining.shape[1])  # 不能超过剩余轨迹长度
```

#### Step 4: 注入模型输入

```python
inputs['prev_actions'] = remaining     # (1, M, action_dim)
inputs['prefix_len'] = prefix_len
inputs['rtc_config'] = self.rtc_config # 传给 head 选择 prefix 或 guidance 方法
```

### 4.4 模型端 prefix 去噪（`_predict_action_prefix_rtc`）

`FlowMatchingInferenceHead._predict_action_prefix_rtc()`（`flow_matching_head.py:417`）

```python
def _predict_action_prefix_rtc(self, actions, denoise, batch_size,
                                device, dt, prev_actions, prefix_len):
    for t in range(self.num_inference_timesteps):  # 4 步去噪
        t_cont = t / float(self.num_inference_timesteps)
        t_discretized = int(t_cont * self.num_timestep_buckets)
        t_global = torch.full((batch_size,), fill_value=t_discretized, device=device)

        # ① 每步去噪前：强制替换前缀位置为已知动作
        actions[:, :prefix_len] = prev_actions[:, :prefix_len]

        # ② 构造 per-position 时间编码
        t_enc = torch.full((batch_size, num_steps), fill_value=t_discretized, ...)
        t_enc[:, :prefix_len] = self.num_timestep_buckets  # 前缀位置 = "干净"时间

        # ③ 去噪
        v = denoise(actions, t_global, t_enc)
        actions = actions + dt * v

    # ④ 最终再次强制替换前缀
    actions[:, :prefix_len] = prev_actions[:, :prefix_len]
    return actions
```

核心机制：

| 操作 | 前缀位置（0 ~ prefix_len-1） | 其余位置（prefix_len ~ 31） |
|------|------------------------------|---------------------------|
| 动作值 | 强制替换为 `prev_actions` | 正常去噪生成 |
| 时间编码 | `num_timestep_buckets`（最大值，表示"已完全干净"） | 当前去噪步的时间 |
| 去噪过程中 | 每步都被覆盖回已知值 | 逐步从噪声走向干净 |

**关键点**：前缀位置的时间编码 `t_enc = num_timestep_buckets` 告诉模型"这些位置是已知的干净动作"，与训练时 `t=1.0`（clean_time）的语义一致。模型据此调整其余位置的预测，使新生成部分自然衔接前缀。

### 4.5 推理时 RTC 示意图

```
chunk 0 (第一次推理，无 RTC):
  模型输出: [a0, a1, a2, ..., a31]
  开始执行，同时记录 action_timestamp 和 inference_elapsed

chunk 1 (第二次推理，带 RTC):
  假设 offset=10 (上轮已执行了 ~10 步), inference_elapsed=0.1s, publish_rate=30
  → remaining = [a10', a11', ..., a31']  (上轮剩余 22 步，经重采样对齐)
  → prefix_len = int(0.1 * 30) = 3

  去噪过程 (4 步迭代):
  ┌───────────────────────────────────────────────────────────────┐
  │ 位置:    [0]     [1]     [2]     [3]    [4]   ...   [31]     │
  │          ────前缀（锁定）────   ──────去噪生成──────────        │
  │ 动作:    a10'    a11'    a12'    noise→ noise→ ... noise→     │
  │ t_enc:   max     max     max     当前t  当前t  ... 当前t      │
  │ 每步去噪后: 强制覆盖回 a10',a11',a12'                          │
  └───────────────────────────────────────────────────────────────┘

  最终输出: [a10', a11', a12', b3, b4, ..., b31]
           ↑与上轮尾部平滑衔接    ↑新生成的动作
```

---

## 5. 训练与推理的对应关系

| 维度 | 训练时 | 推理时 |
|------|--------|--------|
| **配置位置** | `model.vla_head.rtc_training_config` | `inference.rtc_config` |
| **前缀来源** | 真实动作（ground truth） | 上一轮模型预测的剩余轨迹 |
| **前缀长度** | 随机采样 `delay ∈ [0, 6]`（指数分布） | 动态估算 `int(inference_elapsed * publish_rate)` |
| **前缀识别方式** | per-position time 设为 `clean_time=1.0` | per-position t_enc 设为 `num_timestep_buckets` |
| **非前缀位置** | 正常 flow matching 训练 | 正常去噪迭代 |
| **loss/output** | 前缀位置 loss mask=0 | 前缀位置强制覆盖为已知值 |
| **目的** | 学会利用已知前缀生成连续轨迹 | 利用已执行动作平滑拼接 chunk |

**训练-推理一致性**：训练时通过随机延迟模拟各种前缀长度场景，推理时前缀长度由实际推理延迟决定。两者使用相同的"前缀位置标记为干净"语义，确保模型的训练分布覆盖推理时遇到的情况。

---

## 6. 配置建议

### `max_delay` 选择

建议 `max_delay` 覆盖推理时可能出现的最大 `prefix_len`。当前配置 `max_delay=7`，推理时：

```
prefix_len ≈ inference_elapsed × publish_rate
```

如果推理耗时约 0.1~0.2s，publish_rate=30Hz，则 `prefix_len ≈ 3~6`，在 `max_delay=7` 的训练覆盖范围内。

### `prefix_len=None` vs 固定值

- `None`（当前配置）：根据上一轮实际推理耗时动态计算，适应性强。
- 固定值（如 `prefix_len=5`）：更稳定可预测，适合推理耗时波动小的场景。
