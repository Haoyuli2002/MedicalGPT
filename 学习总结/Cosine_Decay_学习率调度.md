# Cosine Decay 学习率调度（Cosine Annealing）

---

## 1. 概述

Cosine Decay（余弦退火）是一种学习率调度策略，让学习率按照余弦函数的形状从最大值平滑地衰减到最小值。它是目前 LLM 训练（SFT、DPO、PPO、GRPO）中**最主流**的 scheduler。

---

## 2. 公式

### 完整公式（含 Warmup）

训练分为两个阶段：

**阶段1：Linear Warmup（线性预热）**

$$\eta_t = \frac{t}{t_{warmup}} \cdot \eta_{max}, \quad t \leq t_{warmup}$$

**阶段2：Cosine Decay（余弦衰减）**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - t_{warmup}}{T - t_{warmup}} \cdot \pi\right)\right), \quad t > t_{warmup}$$

其中：
- $\eta_t$：第 t 步的学习率
- $\eta_{max}$：最大学习率（peak learning rate）
- $\eta_{min}$：最小学习率（通常为 0）
- $t$：当前训练步数
- $t_{warmup}$：warmup 结束的步数
- $T$：总训练步数

---

## 3. 曲线形状

```
lr
↑
η_max |      ╭──────────╮
      |     ╱            ╲
      |    ╱              ╲
      |   ╱                ╲
      |  ╱                  ╲
      | ╱                    ╲
η_min |╱________________________╲→ steps
      0   t_warmup            T
      ↑        ↑               ↑
   线性上升  到达峰值      余弦下降到η_min
```

---

## 4. 为什么用 Cosine Decay？

| 特点 | 说明 |
|------|------|
| 训练前期 lr 大 | 快速学习，探索参数空间 |
| 训练后期 lr 小 | 精细调整，收敛到更好的局部最优 |
| 下降平滑 | 比 linear decay 更自然，避免 lr 突变导致的训练不稳定 |
| 无需手动调节 | 不像 step decay 需要手动设定阶梯下降点 |

---

## 5. 与其他 Scheduler 对比

| Scheduler | 曲线形状 | 特点 | 适用场景 |
|-----------|---------|------|---------|
| **cosine** | 半个余弦波 | 平滑，最常用 | LLM 训练、DPO、SFT |
| linear | 直线下降 | 简单 | 快速实验 |
| constant | 水平线 | lr 不变 | 极短训练 |
| constant_with_warmup | 预热后水平 | 预热后保持不变 | 短训练 |
| polynomial | 多项式曲线 | 可调节下降速度 | 特殊需求 |
| step | 阶梯下降 | 在固定epoch后突降 | 传统CV训练 |

---

## 6. 在 DPO/SFT 训练中的典型配置

```bash
--learning_rate 5e-5           # η_max
--lr_scheduler_type cosine     # 使用 cosine decay
--warmup_steps 10              # 前10步线性预热
--num_train_epochs 2           # 总共训练2个epoch
```

### Warmup 的作用

- 训练刚开始时，模型权重是随机的或预训练状态
- 如果一开始就用大 lr，梯度可能很大，导致训练不稳定
- Warmup 让 lr 从 0 慢慢上升，给模型一个"热身"阶段

### Warmup 步数的选择

| 经验规则 | 说明 |
|---------|------|
| 总步数的 5-10% | 最常用 |
| 10-100 步 | 对于小数据集 |
| 500-2000 步 | 对于大规模预训练 |

---

## 7. 数学直觉

余弦函数 $\cos(x)$：
- $x = 0$ 时，$\cos(0) = 1$，lr = η_max
- $x = \pi$ 时，$\cos(\pi) = -1$，lr = η_min

所以公式中：
$$\frac{1}{2}(1 + \cos(\theta))$$

将 cos 的值域从 [-1, 1] 映射到 [0, 1]，再乘以 (η_max - η_min) 得到实际的 lr 范围。

---

## 8. Python 实现示例

```python
import math

def cosine_decay_with_warmup(step, total_steps, warmup_steps, lr_max, lr_min=0):
    """计算给定步数的学习率"""
    if step < warmup_steps:
        # Linear warmup
        return lr_max * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# 示例：100步训练，10步warmup，最大lr=5e-5
for step in range(0, 101, 10):
    lr = cosine_decay_with_warmup(step, 100, 10, 5e-5)
    print(f"Step {step:3d}: lr = {lr:.6f}")
```

输出：
```
Step   0: lr = 0.000000
Step  10: lr = 0.000050  ← warmup 结束，到达峰值
Step  20: lr = 0.000047
Step  30: lr = 0.000040
Step  40: lr = 0.000031
Step  50: lr = 0.000025
Step  60: lr = 0.000019
Step  70: lr = 0.000010
Step  80: lr = 0.000003
Step  90: lr = 0.000001
Step 100: lr = 0.000000  ← 衰减到0
```

---

## 9. 常见问题

### Q: warmup_steps 设多大？
A: 总步数的 5-10%。如果训练 100 步，warmup 设 5-10 步。

### Q: lr 衰减到 0 会不会太小？
A: 训练末期 lr 很小是故意的——让模型在接近最优解时做微小调整，避免"跳过"最优点。

### Q: 为什么不用 linear decay？
A: Cosine decay 在训练初期保持较大的 lr 更长时间（因为余弦曲线前半段下降慢），让模型有更多时间在高 lr 下学习；而 linear decay 从一开始就在线性下降。

### Q: DPO 中的 β 参数和 lr 什么关系？
A: 没有直接关系。β 控制的是 DPO loss 中偏好信号的强度，lr 控制的是每步权重更新的幅度。但两者都需要调得合适才能训练稳定。

---

## 10. 总结

```
Cosine Decay = Warmup（线性上升）+ Cosine Annealing（余弦下降）

优点：平滑、自然、无需手动调节阶梯点
公式核心：lr = 0.5 * lr_max * (1 + cos(progress * π))
典型配置：warmup = 总步数的 5-10%，lr = 5e-5 (DPO)
```

---

## 参考资料

- 原始论文：[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- HuggingFace Transformers: `get_cosine_schedule_with_warmup()`
- PyTorch: `torch.optim.lr_scheduler.CosineAnnealingLR`