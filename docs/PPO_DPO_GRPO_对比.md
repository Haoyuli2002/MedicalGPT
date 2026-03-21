# PPO vs DPO vs GRPO 对比

## 目录

- [一、一句话总结](#一一句话总结)
- [二、核心维度对比](#二核心维度对比)
- [三、目标函数对比](#三目标函数对比)
- [四、训练流程对比](#四训练流程对比)
- [五、每步计算量对比](#五每步计算量对比)
- [六、同一任务的三种处理方式](#六同一任务的三种处理方式)
- [七、选择建议](#七选择建议)

---

## 一、一句话总结

| 方法 | 核心思路 |
|------|---------|
| **PPO** | 训练奖励模型打分 → 策略生成回复 → RM 评估 → GAE 优势 → PPO-Clip 更新 |
| **DPO** | 直接从偏好对学习，将 RM 隐含在策略的 log 概率差中 |
| **GRPO** | 每个 prompt 生成一组回复 → 规则打分 → 组内归一化 → PPO-Clip 更新 |

---

## 二、核心维度对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **奖励来源** | 学习的 Reward Model | 隐式（log prob 差） | 规则函数（可验证） |
| **需要 RM** | ✅ 需要单独训练 | ❌ | ❌ |
| **模型数量** | 4（policy + ref + RM + value） | 2（policy + ref） | 2（policy + ref） |
| **数据格式** | prompt（RM 用偏好对） | {question, chosen, rejected} | {question, answer} |
| **优势估计** | GAE + value model | 不需要 | 组内均值/标准差归一化 |
| **是否在线生成** | ✅ 1 个/prompt | ❌ 离线数据 | ✅ N 个/prompt |
| **显存需求** | 高（4 模型） | 低（2 模型） | 中（2 模型 + 多次生成） |
| **训练稳定性** | 差（超参敏感） | 好 | 好 |
| **实现复杂度** | 高 | 低 | 中 |
| **训练速度** | 慢 | 快 | 中 |
| **适用任务** | 通用 | 通用（偏好对齐） | 可验证任务（数学、代码） |
| **R1 推理风格** | 可以但复杂 | ❌ 不适合 | ✅ 天然支持 |

---

## 三、目标函数对比

### PPO

$$L_{PPO} = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{old}}\hat{A}_t^{GAE},\;\text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1{-}\epsilon, 1{+}\epsilon\right)\hat{A}_t^{GAE}\right)\right]$$

- $\hat{A}_t^{GAE}$ = GAE 优势（per-token，依赖 value model）
- 奖励来自学习的 RM
- 另有 value loss 更新 value model

### DPO

$$L_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

- 无需优势估计
- 隐式奖励 = $\beta \cdot (\log\pi_\theta - \log\pi_{ref})$
- 直接最大化 chosen 相对 rejected 的概率差

### GRPO

$$L_{GRPO} = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|o_i|}\sum_{t}\left(\min\left(\frac{\pi_\theta}{\pi_{old}}\hat{A}_i,\;\text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1{-}\epsilon, 1{+}\epsilon\right)\hat{A}_i\right) - \beta \cdot D_{KL}\right)\right]$$

- $\hat{A}_i = (r_i - \mu) / \sigma$（组内归一化，per-sequence）
- 奖励来自规则函数
- 除以 $|o_i|$ 做长度归一化

### 公式关键区别

| | PPO | DPO | GRPO |
|---|---|---|---|
| 优势来源 | GAE（value model） | 隐含在公式中 | 组内归一化 |
| 优势粒度 | per-token | per-sequence | per-sequence |
| Clip 机制 | ✅ | ❌ | ✅ |
| KL 约束方式 | 奖励中减 KL | σ 函数自然约束 | 损失中减 KL |

---

## 四、训练流程对比

### PPO（三阶段）

```
前置: SFT 训练 → RM 训练
                    ↓
Step 1: policy 根据 prompt 生成 1 个 response
Step 2: RM 打分 → reward
Step 3: value model 估计 V(s) → GAE 计算 per-token 优势
Step 4: ref_policy 计算 KL 惩罚
Step 5: PPO-Clip 更新 policy + MSE 更新 value model
```

### DPO（一阶段）

```
前置: SFT 训练
        ↓
Step 1: 从数据集取 (prompt, chosen, rejected)
Step 2: policy 计算 log π_θ(chosen) 和 log π_θ(rejected)
Step 3: ref 计算 log π_ref(chosen) 和 log π_ref(rejected)
Step 4: 计算 DPO loss → 反向传播更新 policy
```

### GRPO（一阶段）

```
前置: SFT 训练
        ↓
Step 1: policy 根据 prompt 生成 N 个 response
Step 2: 规则函数对每个 response 打分 → r_1, ..., r_N
Step 3: 组内归一化 → Â_1, ..., Â_N
Step 4: ref 计算 KL 惩罚
Step 5: PPO-Clip 更新 policy
```

---

## 五、每步计算量对比

| 操作 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 自回归生成 | 1 次 | 0 | N 次（如 4 次） |
| Policy 前向 | 1 次 | 2 次（chosen + rejected） | N 次 |
| Ref 前向 | 1 次 | 2 次 | N 次 |
| RM 前向 | 1 次 | 0 | 0（规则函数，几乎无开销） |
| Value 前向 | 1 次 | 0 | 0 |
| 反向传播 | 2 次（policy + value） | 1 次（policy） | 1 次（policy） |
| **总前向** | 4 次 | 4 次 | 2N 次 |
| **总反向** | 2 次 | 1 次 | 1 次 |
| **瓶颈** | 生成 + 4 模型 | 无生成，速度最快 | N 次生成 |

---

## 六、同一任务的三种处理方式

### 任务：让模型学会回答"57+29=?"

### PPO 的处理

```
数据: {"prompt": "57+29=?"}

1. RM 已用偏好对训练好（知道正确答案好于错误答案）
2. Policy 生成: "57+29=86"
3. RM 打分: 0.85（较高分）
4. Value model 估计基线: V=0.6
5. GAE 优势: 正值（表现好于预期）
6. 更新: 增大生成 "86" 的概率
```

### DPO 的处理

```
数据: {
  "question": "57+29=?",
  "chosen": "57+29=86",
  "rejected": "57+29=84"
}

1. Policy:  log π_θ(chosen)=-5.2,  log π_θ(rejected)=-4.8
2. Ref:     log π_ref(chosen)=-5.5, log π_ref(rejected)=-5.0
3. 隐式奖励差 = β×[(−5.2+5.5) − (−4.8+5.0)] = β×[0.3 − 0.2] = 0.1β
4. Loss = -log σ(0.1β)
5. 梯度推动: 增大 chosen 概率，减小 rejected 概率
```

### GRPO 的处理

```
数据: {"question": "57+29=?", "answer": "86"}

1. Policy 生成 4 个回复:
   回复1: <think>7+9=16,进1,5+2+1=8</think><answer>86</answer>  → 奖励 2.0
   回复2: <think>57+29=86</think><answer>86</answer>             → 奖励 2.0
   回复3: <think>57+29=84</think><answer>84</answer>             → 奖励 1.0
   回复4: 答案是86                                               → 奖励 0.0

2. 组内归一化: μ=1.25, σ=0.829
   Â = [+0.90, +0.90, -0.30, -1.51]

3. 更新: 鼓励回复1/2，轻微抑制回复3，强烈抑制回复4
```

### 关键差异总结

| | PPO | DPO | GRPO |
|---|---|---|---|
| 需要的数据 | prompt | prompt + chosen + rejected | prompt + answer |
| 回复来源 | 在线生成 1 个 | 离线数据（已有） | 在线生成 N 个 |
| 判断好坏的方式 | RM 打分 | 人工标注偏好对 | 规则验证 |
| 基线 | value model | ref model | 组内均值 |

---

## 七、选择建议

### 选 DPO 当：

- ✅ 有高质量的人工偏好标注数据
- ✅ 任务偏主观（对话风格、安全性、有用性）
- ✅ 计算资源有限
- ✅ 想要简单稳定的训练

### 选 PPO 当：

- ✅ 需要在线探索和精细控制
- ✅ 有足够的 GPU 显存（4 个模型）
- ✅ 任务复杂，需要持续适应
- ✅ 有能力调参（PPO 对超参敏感）

### 选 GRPO 当：

- ✅ 任务结果可自动验证（数学、代码、事实问答）
- ✅ 想训练 R1 风格的推理模型（think → answer）
- ✅ 不想训练奖励模型
- ✅ 显存有限但需要在线学习

### 组合使用

实际项目中可以组合：

```
SFT → DPO（快速对齐安全性和有用性）→ GRPO（增强数学/推理能力）
```

或者：

```
SFT → RM → PPO（通用对齐）→ GRPO（专项推理增强）