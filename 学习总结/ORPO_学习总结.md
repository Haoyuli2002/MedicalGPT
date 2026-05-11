# ORPO (Odds Ratio Preference Optimization) 学习总结

> 论文：[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

---

## 1. 概述

ORPO 是一种**单步偏好优化**方法，将 SFT（有监督微调）和偏好对齐合并为一个统一的损失函数。

### 核心优势
- ❌ **不需要参考模型**（ref_model）
- ❌ **不需要单独的 SFT 阶段**
- ✅ 一步同时完成 SFT + 偏好对齐
- ✅ 缓解灾难性遗忘问题

### 训练路径对比

```
传统路径:  Base Model → SFT → DPO (需要 ref_model)
ORPO路径:  Base Model → ORPO (一步到位)
```

---

## 2. 数学公式

### 2.1 基础定义 - 序列概率

模型对整个序列的生成概率（取 token-level log prob 的平均后再 exp，即几何平均）：

$$P_\theta(y|x) = \exp\left(\frac{1}{|y|} \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t})\right)$$

### 2.2 Odds（比值/赔率）

定义序列 y 相对于 prompt x 的 odds：

$$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$$

**直觉理解**：odds 衡量的是"模型生成这个回复的倾向 vs 不生成的倾向"。

### 2.3 Odds Ratio（比值比）

Chosen ($y_w$) 和 Rejected ($y_l$) 的比值比：

$$OR_\theta(y_w, y_l | x) = \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}$$

**直觉理解**：OR > 1 表示模型更偏好 chosen；OR < 1 表示模型更偏好 rejected。

### 2.4 Log Odds Ratio

取对数方便计算：

$$\log OR_\theta(y_w, y_l|x) = \underbrace{\log P_\theta(y_w|x) - \log(1 - P_\theta(y_w|x))}_{\log \text{odds}(y_w)} - \underbrace{\left[\log P_\theta(y_l|x) - \log(1 - P_\theta(y_l|x))\right]}_{\log \text{odds}(y_l)}$$

### 2.5 ORPO 总损失函数 ⭐

$$\boxed{\mathcal{L}_{ORPO} = \underbrace{-\frac{1}{|y_w|}\sum_{t=1}^{|y_w|} \log P_\theta(y_{w,t} | x, y_{w,<t})}_{\text{SFT Loss: 在 chosen 上的 NLL}} + \lambda \cdot \underbrace{\left(-\log \sigma\left(\log OR_\theta(y_w, y_l|x)\right)\right)}_{\text{Odds Ratio Alignment Loss}}}$$

简写：

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT}(y_w) + \lambda \cdot \mathcal{L}_{OR}$$

其中：
- $\mathcal{L}_{SFT}$：标准交叉熵损失，只在 chosen 回复上计算
- $\mathcal{L}_{OR}$：Odds Ratio 对齐损失
- $\lambda$（代码中为 `beta`）：平衡两个损失的权重，默认 **0.1**
- $\sigma$：sigmoid 函数

---

## 3. 两个 Loss 的作用

| 组成部分 | 作用 | 类比 |
|---------|------|------|
| $\mathcal{L}_{SFT}$ | 让模型学会生成好的回复 | 教学生写好作文 |
| $\mathcal{L}_{OR}$ | 让模型区分好坏回复，增大 chosen 和 rejected 的差距 | 教学生判断哪篇作文更好 |

**核心洞察**：
- 单独的 SFT 只增加 chosen 的概率，但不保证 rejected 的概率会降低
- $\mathcal{L}_{OR}$ 显式地拉大两者的差距
- 两者同时优化避免了分阶段训练的灾难性遗忘

---

## 4. 与其他方法的对比

| 特性 | DPO | ORPO | RLHF (PPO) |
|------|-----|------|------------|
| 需要 ref_model | ✅ | ❌ | ✅ |
| 需要先 SFT | ✅ | ❌ | ✅ |
| 需要 Reward Model | ❌ | ❌ | ✅ |
| 训练步骤 | 2步 (SFT→DPO) | 1步 | 3步 (SFT→RM→PPO) |
| 灾难性遗忘风险 | 中等 | 低 | 中等 |
| 计算开销 | 中（需存储 ref_model） | 低 | 高（4个模型） |
| 数据格式 | prompt/chosen/rejected | prompt/chosen/rejected | prompt + reward signal |

### DPO 损失 vs ORPO 损失

**DPO Loss**：
$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta \cdot \left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$$

**ORPO Loss**：
$$\mathcal{L}_{ORPO} = \mathcal{L}_{NLL}(y_w) + \lambda \cdot \left(-\log\sigma\left(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\right)\right)$$

**关键区别**：
- DPO 用 ref_model 的比值来约束（防止偏离太远）
- ORPO 用 odds ratio 直接比较，不需要 ref_model
- ORPO 额外包含 SFT loss，所以一步搞定

---

## 5. MedicalGPT 中的实现

### 5.1 代码结构

```python
# orpo_training.py

from trl import ORPOConfig, ORPOTrainer

# 配置
training_args = ORPOConfig(
    beta=0.1,                    # λ 参数，平衡 SFT 和 OR loss
    max_length=full_max_length,
    max_prompt_length=max_source_length,
    ...
)

# 训练器 - 注意没有 ref_model！
trainer = ORPOTrainer(
    model,                       # 只需要一个模型
    args=training_args,
    train_dataset=train_dataset, # 包含 prompt/chosen/rejected
    processing_class=tokenizer,
    peft_config=peft_config,     # 可选 LoRA
)
```

### 5.2 数据格式

```json
{
  "system": "You are a helpful assistant.",
  "history": [["之前的问题", "之前的回答"]],
  "question": "当前问题",
  "response_chosen": "优质回复（人类偏好的）",
  "response_rejected": "较差回复（人类不偏好的）"
}
```

经过 `return_prompt_and_responses()` 处理后变为：
```python
{
    "prompt": "完整的 prompt（含 system + history + question）",
    "chosen": "优质回复",
    "rejected": "较差回复"
}
```

### 5.3 运行命令

```bash
# run_orpo.sh
CUDA_VISIBLE_DEVICES=0 python orpo_training.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \   # 可以直接用 base model！
    --train_file_dir ./data/reward \
    --do_train \
    --do_eval \
    --use_peft True \
    --orpo_beta 0.1 \
    --max_source_length 2048 \
    --max_target_length 512 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-4 \
    --output_dir outputs-orpo-v1
```

---

## 6. 为什么 ORPO 不需要 Reference Model？

### DPO 为什么需要 ref_model？
DPO 的 loss 是通过 KL 约束派生的，需要 ref_model 来防止 policy 偏离太远：
$$\text{DPO reward} = \beta \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

### ORPO 的替代方案
ORPO 使用 **odds** 本身作为正则化：
- $\text{odds} = \frac{P}{1-P}$
- 当 P 接近 1 时，odds → ∞，但 log odds 增长缓慢
- 这自然地防止了概率坍缩到 0 或 1
- odds ratio 本身就包含了"相对比较"的信息，不需要额外的基准

**直觉**：odds 的数学性质自带"软约束"，不让模型过度自信。

---

## 7. ORPO 的超参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `beta` (λ) | OR loss 的权重 | 0.1 |
| `max_length` | prompt + response 的最大长度 | 2048-4096 |
| `max_prompt_length` | prompt 的最大长度 | 1024-2048 |
| `learning_rate` | 学习率 | 5e-5 ~ 5e-4 |

**beta 的影响**：
- beta 太小 → 偏好对齐不够强，模型可能不区分好坏
- beta 太大 → SFT 部分被压制，模型可能不会好好生成
- 0.1 是论文推荐的默认值

---

## 8. 适用场景

### ORPO 适合：
- ✅ 想一步到位完成训练（不想分 SFT 和 DPO 两阶段）
- ✅ 计算资源有限（不需要存储 ref_model）
- ✅ 担心灾难性遗忘
- ✅ 从 base model 直接训练（如 Qwen2.5-0.5B）

### ORPO 可能不如 DPO 的场景：
- ⚠️ 已经有高质量的 SFT 模型，只想做微调对齐
- ⚠️ 需要精细控制 SFT 和对齐的训练节奏
- ⚠️ 数据量非常大，分阶段训练更灵活

---

## 9. 总结

```
ORPO = SFT + Odds Ratio Alignment (一步到位)

Loss = NLL_on_chosen + λ * (-log σ(log(odds_chosen / odds_rejected)))

优点：不需要 ref_model，不需要先 SFT，训练简单
缺点：灵活性相对较低，不能单独调整 SFT 和对齐的训练
```

---

## 参考资料

- 论文：[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- TRL 文档：[ORPOTrainer](https://huggingface.co/docs/trl/main/en/orpo_trainer)
- MedicalGPT 实现：`orpo_training.py`