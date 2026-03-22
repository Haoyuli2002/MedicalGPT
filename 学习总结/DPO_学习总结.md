# DPO（Direct Preference Optimization）学习总结

---

## 一、训练参数总结

### 1. 基础训练参数

| 参数 | 含义 | 典型值 | 说明 |
|------|------|--------|------|
| `per_device_train_batch_size` | 每个 GPU 每次前向传播处理的样本数 | 1~4 | 受 GPU 显存限制 |
| `gradient_accumulation_steps` | 累积多少个 mini-batch 的梯度后再执行一次参数更新 | 4~8 | 有效 batch size = per_device × accumulation × GPU数 |
| `max_steps` | 最大 optimizer step 数（优先级高于 epoch） | 100~200 | 设了就覆盖 `num_train_epochs` |
| `num_train_epochs` | 训练轮数 | 1~3 | `max_steps=-1` 时才生效 |
| `learning_rate` | 学习率 | 2e-5 ~ 5e-4 | DPO 通常用较小学习率 |
| `warmup_steps` | 学习率预热步数 | 50~100 | 前 N 步线性增大学习率 |
| `weight_decay` | 权重衰减（L2 正则化） | 0.01~0.05 | 防止过拟合 |
| `lr_scheduler_type` | 学习率调度策略 | `cosine` | 余弦退火 |

### 2. 序列长度参数

| 参数 | 含义 | 典型值 | 说明 |
|------|------|--------|------|
| `max_source_length` | 输入（prompt）的最大 token 数 | 256~1024 | 超过则截断 |
| `max_target_length` | 输出（response）的最大 token 数 | 256~512 | 超过则截断 |

> **注意**：source + target 不能超过模型的最大上下文长度（如 LLaMA2=4096）。长度越大，显存消耗越大。

### 3. LoRA 参数

| 参数 | 含义 | 典型值 | 说明 |
|------|------|--------|------|
| `target_modules` | LoRA 插入到哪些线性层 | `all` | `all` = 所有线性层；也可指定 `q_proj,v_proj` 等 |
| `lora_rank` (r) | 低秩分解的秩（瓶颈维度） | 8~16 | 越大表达能力越强，参数越多 |
| `lora_alpha` (α) | 缩放因子，实际缩放 = α/r | 16~32 | 通常设为 2×rank |
| `lora_dropout` | LoRA 层的 Dropout 率 | 0.05~0.1 | 防止过拟合 |

#### LoRA 原理简述

```
原始权重 W (冻结) → 输出 = W·x + (α/r) × A·B·x
                                    ↑ 只训练 A 和 B
A: [d_out × r], B: [r × d_in]，参数量远小于原始 W
```

#### 常见 target_modules（以 LLaMA 为例）

| 模块 | 层名 |
|------|------|
| Attention Q/K/V/O | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| MLP | `gate_proj`, `up_proj`, `down_proj` |

### 4. 精度参数：FP16 vs BF16

| 特性 | FP32 | FP16 | BF16 |
|------|------|------|------|
| 总位数 | 32 | 16 | 16 |
| 内存占用 | 4 字节 | 2 字节 | 2 字节 |
| 指数位 / 尾数位 | 8 / 23 | 5 / 10 | 8 / 7 |
| 数值范围 | ±3.4×10³⁸ | ±6.5×10⁴ | ±3.4×10³⁸ |
| 精度 | 高 | 中 | 低 |
| 训练稳定性 | 最稳定 | 需 loss scaling | 稳定（范围大） |

**选择指南**：

| 硬件 | 推荐 |
|------|------|
| NVIDIA A100/H100 | `--bf16 True` |
| NVIDIA T4/V100 | `--fp16 True` |
| Apple M系列 (MPS) | `--bf16 False --fp16 False`（用 FP32） |

### 5. 日志与保存参数

| 参数 | 含义 | 说明 |
|------|------|------|
| `logging_steps` | 每 N 步记录一次 loss | optimizer step，不是前向传播次数 |
| `eval_steps` | 每 N 步评估一次 | 需配合 `eval_strategy=steps` |
| `save_steps` | 每 N 步保存一次 checkpoint | — |
| `report_to` | 日志输出目标 | `tensorboard` 或 `wandb` |

---

## 二、DPO 核心原理

### 1. 为什么需要 DPO？

传统 RLHF 流程：**SFT → 训练奖励模型(RM) → PPO 强化学习**

| 问题 | 说明 |
|------|------|
| 需要单独训练奖励模型 | 额外的训练成本 |
| PPO 训练不稳定 | 超参数敏感，容易崩溃 |
| 需要 4 个模型 | policy、reference、reward、value，显存需求大 |

**DPO 直接从偏好数据优化，跳过 RM 和 PPO，只需要 2 个模型（策略模型 + 参考模型）。**

### 2. 数据格式

每条数据是一个**偏好对**：

```json
{
  "question": "什么是糖尿病？",
  "response_chosen": "糖尿病是一种以高血糖为特征的代谢性疾病...",
  "response_rejected": "糖尿病就是血糖高，少吃糖就行了。"
}
```

### 3. 核心公式

#### DPO 损失函数

$$L_{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \cdot \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

其中：
- $\pi_\theta$ = 策略模型（正在训练的模型）
- $\pi_{ref}$ = 参考模型（冻结的 SFT 模型）
- $y_w$ = chosen（人类偏好的好回复）
- $y_l$ = rejected（人类不偏好的差回复）
- $\beta$ = 温度参数（控制偏离参考模型的程度）
- $\sigma$ = sigmoid 函数

#### 隐式奖励

$$r(x, y) = \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

### 4. 概率计算方法（Teacher Forcing）

$\log \pi_\theta(y|x)$ 的计算方式是**对 response 部分每个 token 的条件 log 概率求和**：

$$\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_1, \ldots, y_{t-1})$$

**具体步骤**：

```
1. 将 prompt + response 拼接成完整序列作为输入
2. 做一次前向传播，得到每个位置的 logits
3. 对 response 部分的每个位置：
   - 取前一个位置的 logits → softmax → 得到概率分布
   - 从中取出"真实下一个 token"的 log 概率
4. 将 response 部分所有位置的 log 概率求和
```

```
输入:   [什么] [是] [糖尿病] [？]  [糖尿] [病]  [是]  [一种]
         ↓     ↓     ↓       ↓     ↓      ↓     ↓      ↓
模型预测: ...   ...   ...    ...   [糖尿] [病]  [是]  [一种] [...]
                                    ↑      ↑     ↑      ↑
                              只取 response 部分的 log P(真实token)
```

> **注意**：这里是 Teacher Forcing，模型被"强迫"看到真实的 token 序列，而不是自己生成。我们只是在"问"模型：你觉得这个 token 出现在这里的概率有多大？

### 5. 完整更新逻辑

对于一个偏好对 (prompt, chosen, rejected)：

```python
# Step 1: 4 次前向传播
log_pi_chosen   = sum_log_probs(model_θ,   prompt + chosen)    # 策略模型对 chosen
log_pi_rejected = sum_log_probs(model_θ,   prompt + rejected)  # 策略模型对 rejected
log_ref_chosen  = sum_log_probs(model_ref, prompt + chosen)    # 参考模型对 chosen
log_ref_rejected = sum_log_probs(model_ref, prompt + rejected) # 参考模型对 rejected

# Step 2: 计算隐式奖励
reward_chosen  = β × (log_pi_chosen - log_ref_chosen)
reward_rejected = β × (log_pi_rejected - log_ref_rejected)

# Step 3: 计算 DPO Loss
loss = -log(sigmoid(reward_chosen - reward_rejected))

# Step 4: 反向传播，更新 θ（只更新策略模型，参考模型冻结）
loss.backward()
optimizer.step()
```

### 6. 评估指标

| 指标 | 公式 | 含义 | 期望趋势 |
|------|------|------|---------|
| `eval/loss` | DPO Loss | 整体损失 | ↓ 下降 |
| `eval/rewards/chosen` | β × (log π_θ(chosen) - log π_ref(chosen)) | 模型对好回复的偏好分 | ↑ 上升 |
| `eval/rewards/rejected` | β × (log π_θ(rejected) - log π_ref(rejected)) | 模型对差回复的偏好分 | ↓ 下降 |
| `eval/rewards/accuracies` | P(reward_chosen > reward_rejected) | 偏好对排序准确率 | ↑ 趋近 1.0 |
| `eval/rewards/margins` | reward_chosen - reward_rejected | 好坏回复的奖励差距 | ↑ 增大 |

### 7. DPO vs PPO 对比

| 维度 | PPO (RLHF) | DPO |
|------|-----------|-----|
| 是否需要奖励模型 | ✅ 需要 | ❌ 不需要 |
| 训练稳定性 | 较差 | 较好 |
| 内存需求 | 高（4 个模型） | 低（2 个模型） |
| 实现复杂度 | 高 | 低 |
| 数据需求 | 在线采样 | 离线偏好数据 |

---

## 三、训练 Pipeline

```
Stage 1: PT（增量预训练，可选）
    ↓
Stage 2: SFT（有监督微调）→ 得到 SFT 模型
    ↓
Stage 3: DPO（直接偏好优化）→ 得到对齐后的模型
    ↓
推理测试
```

每个阶段使用 LoRA 训练后，需要用 `merge_peft_adapter.py` 将 adapter 权重合并回 base model，作为下一阶段的输入。