# PPO（Proximal Policy Optimization）学习总结

---

## 一、PPO 完整 Pipeline

```
Stage 1: SFT（有监督微调）
    → 得到 SFT 模型（作为 policy 和 ref_policy 的初始化）
    ↓
Stage 2: Reward Model 训练
    → 用偏好数据训练奖励模型（学会给回复打分）
    ↓
Stage 3: PPO 强化学习
    → 用奖励模型的反馈优化策略模型
    ↓
推理测试
```

**对比 DPO**：DPO 跳过了 Stage 2（不需要奖励模型），直接从偏好数据优化。

---

## 二、Stage 2：奖励模型（Reward Model）训练

### 1. 模型架构

奖励模型使用 `AutoModelForSequenceClassification`（num_labels=1）：
- 基于预训练语言模型（如 Qwen2.5-0.5B）
- 在最后一层加一个**分类头（score head）**，输出一个**标量分数**
- 输入一个 prompt + response，输出一个奖励值 r ∈ ℝ

```
输入: "什么是糖尿病？糖尿病是一种以高血糖为特征的代谢性疾病..."
      ↓
[CausalLM backbone] → hidden states
      ↓
[Score Head (Linear)] → 标量: 3.7 (奖励分数)
```

### 2. 数据格式

与 DPO 相同，使用偏好对数据：

```json
{
  "system": "",
  "history": [],
  "question": "什么是糖尿病？",
  "response_chosen": "糖尿病是一种以高血糖为特征的代谢性疾病...",
  "response_rejected": "糖尿病就是血糖高，少吃糖就行了。"
}
```

### 3. 损失函数：Pairwise LogLoss（InstructGPT）

$$L_{RM} = -\mathbb{E}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

代码实现（`reward_modeling.py`）：

```python
def compute_loss(self, model, inputs):
    rewards_chosen = model(input_ids=inputs["input_ids_chosen"],
                           attention_mask=inputs["attention_mask_chosen"])[0]
    rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                             attention_mask=inputs["attention_mask_rejected"])[0]
    loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
    return loss
```

**直觉**：训练奖励模型让 chosen 的分数高于 rejected 的分数。

### 4. RM 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name_or_path` | Qwen/Qwen2.5-0.5B-Instruct | 基础模型 |
| `per_device_train_batch_size` | 4 | 每 GPU batch size |
| `gradient_accumulation_steps` | 8 | 梯度累积 |
| `num_train_epochs` | 1 | 训练轮数 |
| `learning_rate` | 2e-5 | 学习率 |
| `max_source_length` | 1024 | prompt 最大长度 |
| `max_target_length` | 256 | response 最大长度 |
| `use_peft` | True | 使用 LoRA |
| `lora_rank` | 8 | LoRA 秩 |
| `lora_alpha` | 16 | LoRA 缩放因子 |
| `output_dir` | outputs-rm-qwen-v1 | 输出目录 |

---

## 三、Stage 3：PPO 训练

### 1. 四个模型

PPO 训练需要同时维护 **4 个模型**，这是 PPO 显存需求大的主要原因：

| 模型 | 类型 | 是否训练 | 作用 |
|------|------|---------|------|
| **Policy（策略模型）** | CausalLM | ✅ 训练 | 生成回复，是我们要优化的目标 |
| **Ref Policy（参考模型）** | CausalLM | ❌ 冻结 | SFT 模型的副本，提供 KL 约束基准 |
| **Reward Model（奖励模型）** | SequenceClassification | ❌ 冻结 | 给生成的回复打分 |
| **Value Model（价值模型）** | SequenceClassification | ✅ 训练 | 估计状态价值 V(s)，用于计算优势函数 |

> **注意**：使用 LoRA 时，ref_policy 可以为 None（TRL 会自动通过禁用 adapter 来模拟参考模型），节省显存。

代码（`ppo_training.py`）：

```python
value_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.reward_model_path, num_labels=1)    # 价值模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    training_args.reward_model_path, num_labels=1)    # 奖励模型
policy = AutoModelForCausalLM.from_pretrained(
    training_args.sft_model_path)                      # 策略模型

# 使用 LoRA 时不需要单独的 ref_policy
if peft_config is None:
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path)                  # 参考模型
else:
    ref_policy = None  # LoRA 模式下自动处理
```

### 2. PPO 训练循环

```
对每个 batch 的 prompt:
    ┌─────────────────────────────────────────────────────┐
    │ Step 1: 生成（Rollout）                               │
    │   policy 模型根据 prompt 自回归生成回复 response        │
    │   （这里是真正的"生成"，不是 Teacher Forcing！）         │
    ├─────────────────────────────────────────────────────┤
    │ Step 2: 评估（Evaluation）                            │
    │   reward_model 给 (prompt, response) 打分 → reward    │
    │   value_model 估计每个 token 位置的价值 → V(s)         │
    │   ref_policy 计算 log π_ref(response|prompt)          │
    │   policy 计算 log π_θ(response|prompt)                │
    ├─────────────────────────────────────────────────────┤
    │ Step 3: 计算优势（Advantage Estimation）               │
    │   用 GAE 计算每个 token 的优势 A_t                     │
    │   reward 中加入 KL 惩罚防止偏离太远                     │
    ├─────────────────────────────────────────────────────┤
    │ Step 4: PPO 更新                                      │
    │   用 PPO-Clip 目标函数更新 policy 和 value_model       │
    └─────────────────────────────────────────────────────┘
```

### 3. 核心公式

#### （1）KL 惩罚奖励

在原始奖励上减去 KL 散度惩罚，防止策略偏离参考模型太远：

$$r_{total}(x, y_t) = r_{RM}(x, y) - \beta \cdot \text{KL}[\pi_\theta \| \pi_{ref}]$$

其中 per-token KL：

$$\text{KL}_t = \log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{ref}(y_t|x, y_{<t})$$

#### （2）广义优势估计（GAE, Generalized Advantage Estimation）

##### 为什么需要优势函数？

在强化学习中，我们想知道一个动作"好不好"。但"好不好"是相对的：
- **绝对奖励 r_t**：这个 token 获得了多少奖励
- **价值 V(s_t)**：在这个状态下，**平均**能获得多少未来总奖励

**优势 = 实际表现 - 平均表现**：

$$A_t = Q(s_t, a_t) - V(s_t)$$

- $A_t > 0$：这个动作比平均好 → 增大这个动作的概率
- $A_t < 0$：这个动作比平均差 → 减小这个动作的概率
- $A_t = 0$：这个动作和平均一样 → 不变

##### 从 TD 误差到 GAE

**TD 误差（Temporal Difference Error）**是最基本的优势估计：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

其中：
- $r_t$ = 在时间步 t 获得的即时奖励
- $\gamma V(s_{t+1})$ = 下一步的折扣价值（"未来还能拿多少"）
- $V(s_t)$ = 当前状态的价值估计（"本来期望拿多少"）
- $r_t + \gamma V(s_{t+1})$ = 实际表现（即时奖励 + 未来价值）
- $V(s_t)$ = 期望表现

**直觉**：$\delta_t$ 回答的是"这一步的结果比我预期的好多少？"

##### GAE：多步 TD 的加权平均

单步 TD 误差 $\delta_t$ 虽然方差低但偏差高（因为依赖 V 的估计精度）。多步估计偏差低但方差高。GAE 通过**指数加权平均**取折中：

$$\hat{A}_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

展开来看：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

- 近处的 TD 误差（$\delta_t$）权重大 → 稳定
- 远处的 TD 误差（$\delta_{t+k}$）权重指数衰减 → 减少噪声

##### 两个关键参数

| 参数 | 含义 | 范围 | 效果 |
|------|------|------|------|
| $\gamma$（折扣因子） | 未来奖励的重要程度 | 0~1（通常 0.99） | 越大越看重长期回报 |
| $\lambda$（GAE 参数） | 偏差-方差权衡 | 0~1（通常 0.95） | 越大方差越高但偏差越低 |

**λ 的极端情况**：
- $\lambda = 0$：$\hat{A}_t = \delta_t$（只看一步 TD，低方差高偏差）
- $\lambda = 1$：$\hat{A}_t = \sum \gamma^l \delta_{t+l}$（蒙特卡洛回报，高方差低偏差）
- $\lambda = 0.95$：折中，实践中效果最好

##### 在 LLM PPO 中的应用

在 LLM 的 PPO 训练中，把生成回复的每个 token 看作一个时间步：

```
状态 s_t = (prompt, y_1, y_2, ..., y_{t-1})  ← 已生成的 token 序列
动作 a_t = y_t                                ← 生成的下一个 token
奖励 r_t:
  - 中间 token (t < T): r_t = -β × KL_t      ← 只有 KL 惩罚
  - 最后一个 token (t = T): r_T = RM_score - β × KL_T  ← RM 打分 + KL 惩罚
```

**奖励分配的关键**：奖励模型只在最后一个 token 给出整体评分，中间 token 的奖励只有 KL 惩罚。GAE 通过价值函数将最终奖励"传播"回每个 token 位置。

```
生成序列:  [糖尿] [病]  [是]  [一种] [代谢] [性]  [疾病] [。]
奖励:       -KL   -KL   -KL   -KL   -KL   -KL   -KL   RM_score-KL
价值:       V_1   V_2   V_3   V_4   V_5   V_6   V_7   V_8
TD误差:     δ_1   δ_2   δ_3   δ_4   δ_5   δ_6   δ_7   δ_8
GAE优势:    Â_1   Â_2   Â_3   Â_4   Â_5   Â_6   Â_7   Â_8
                                                         ↑
                                              RM 给了高分，δ_8 大
                                              GAE 把这个信号向前传播
                                              让前面 token 的优势也提高
```

##### GAE 计算伪代码

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: [r_0, r_1, ..., r_T]  每个 token 的奖励
    values:  [V_0, V_1, ..., V_T, V_{T+1}]  每个位置的价值估计
    """
    T = len(rewards)
    advantages = [0] * T
    last_gae = 0
    
    # 从后往前计算（递推公式）
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]  # TD 误差
        last_gae = delta + gamma * lam * last_gae               # GAE 递推
        advantages[t] = last_gae
    
    returns = [advantages[t] + values[t] for t in range(T)]  # 目标回报
    return advantages, returns
```

**递推公式**（实际实现中从后往前算，更高效）：

$$\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}$$

#### （3）PPO-Clip 目标函数

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中：

$$r_t(\theta) = \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{old}}(y_t|x, y_{<t})}$$

- $r_t(\theta)$ = 新旧策略的概率比率
- $\epsilon$ = clip 范围（通常 0.2）
- $\hat{A}_t$ = GAE 计算的优势
- clip 机制**限制每次更新的幅度**，保证训练稳定

#### （4）Value Loss

$$L^{V} = \mathbb{E}\left[(V_\theta(s_t) - V_{target}(s_t))^2\right]$$

### 4. PPO 与 DPO 概率计算的关键区别

| 方面 | DPO | PPO |
|------|-----|-----|
| 回复来源 | 数据集中的**固定** chosen/rejected | 策略模型**在线生成** |
| 概率计算方式 | Teacher Forcing（喂入完整序列） | 先生成，再评估生成序列的概率 |
| 是否需要生成 | ❌ 不需要 | ✅ 每个 step 都要生成 |
| 奖励来源 | 隐式（log prob 之差） | 显式（奖励模型打分） |

---

## 四、PPO 特有参数

| 参数 | 含义 | 典型值 | 说明 |
|------|------|--------|------|
| `sft_model_path` | SFT 模型路径 | — | 用于初始化 policy 和 ref_policy |
| `reward_model_path` | 奖励模型路径 | — | 用于初始化 reward_model 和 value_model |
| `total_episodes` | 总训练 episode 数 | 30000 | 控制总训练量 |
| `response_length` | 生成回复的最大 token 数 | 1000 | 策略模型生成时的最大长度 |
| `missing_eos_penalty` | 缺失 EOS 惩罚 | 1.0 | 如果生成没有正常结束（没产生 EOS），施加惩罚 |
| `per_device_train_batch_size` | 每 GPU batch size | 1 | PPO 显存需求大，通常设得很小 |
| `gradient_accumulation_steps` | 梯度累积 | 4 | 补偿小 batch size |
| `gradient_checkpointing` | 梯度检查点 | True | 用时间换空间 |

---

## 五、PPO vs DPO 详细对比

| 维度 | PPO (RLHF) | DPO |
|------|-----------|-----|
| **需要奖励模型** | ✅ 需要单独训练 | ❌ 不需要 |
| **模型数量** | 4 个（policy + ref + reward + value） | 2 个（policy + ref） |
| **显存需求** | 高（4 个模型） | 低（2 个模型） |
| **训练稳定性** | 较差（超参敏感） | 较好（类似 SFT） |
| **实现复杂度** | 高 | 低 |
| **数据类型** | 在线生成 + 奖励打分 | 离线偏好对 |
| **是否需要生成** | ✅ 每步都要生成 | ❌ 不需要 |
| **训练速度** | 慢（生成 + 多模型前向） | 快（只需前向传播） |
| **理论对齐效果** | 更灵活（在线探索） | 受限于离线数据质量 |
| **适合场景** | 需要在线探索、精细控制 | 快速对齐、资源有限 |

### 计算量对比（每个训练 step）

| 操作 | PPO | DPO |
|------|-----|-----|
| 自回归生成 | ✅ 1 次（policy 生成回复） | ❌ 无 |
| 前向传播 | 4 次（policy + ref + reward + value） | 4 次（policy×2 + ref×2） |
| 反向传播 | 2 次（policy + value） | 1 次（policy） |
| **总计** | 1 次生成 + 4 次前向 + 2 次反向 | 4 次前向 + 1 次反向 |

---

## 六、奖励模型训练 Pipeline 总结

```
偏好数据 (chosen/rejected pairs)
    ↓
tokenize: prompt + chosen → input_ids_chosen
          prompt + rejected → input_ids_rejected
    ↓
模型前向：reward_chosen  = RM(input_ids_chosen)   → 标量
          reward_rejected = RM(input_ids_rejected) → 标量
    ↓
Loss = -log σ(reward_chosen - reward_rejected)
    ↓
反向传播，更新 RM 参数
```

训练目标：让 RM 学会 **chosen 分数 > rejected 分数**。

---

## 七、PPO 训练 Pipeline 总结

```
prompt (从训练集采样)
    ↓
policy 生成回复 response（自回归生成）
    ↓
reward_model 打分：r = RM(prompt, response)
    ↓
value_model 估计价值：V(s_t) for each token position
    ↓
ref_policy 计算参考概率：log π_ref(response|prompt)
    ↓
计算 KL 惩罚 + GAE 优势
    ↓
PPO-Clip 更新 policy 参数
Value Loss 更新 value_model 参数