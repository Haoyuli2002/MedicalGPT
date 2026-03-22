# TRL Trainers 详解

本项目使用 [TRL（Transformer Reinforcement Learning）](https://github.com/huggingface/trl) 库的 Trainer 进行 RLHF 训练。本文档详细讲解三个 Trainer 的 API、内部机制和核心参数。

## 目录

- [一、概览](#一概览)
- [二、DPOTrainer](#二dpotrainer)
- [三、PPOTrainer](#三ppotrainer)
- [四、GRPOTrainer](#四grpotrainer)
- [五、三者核心差异总结](#五三者核心差异总结)

---

## 一、概览

| 训练脚本 | TRL Trainer | TRL Config | 用途 |
|----------|-------------|------------|------|
| `dpo_training.py` | `DPOTrainer` | `DPOConfig` | 直接偏好优化 |
| `ppo_training.py` | `PPOTrainer` | `PPOConfig` | 近端策略优化 |
| `grpo_training.py` | `GRPOTrainer` | `GRPOConfig` | 组相对策略优化 |

本项目的 `.py` 脚本**不是从零实现**算法，而是：
1. 数据准备 → 加载/处理为 TRL 要求的格式
2. 模型配置 → 加载基础模型 + LoRA/QLoRA
3. 参数设置 → 配置 Config 对象
4. 初始化 Trainer → 传入模型、数据、配置
5. 调用 `trainer.train()` → 核心逻辑全在 TRL 内部

---

## 二、DPOTrainer

### 2.1 初始化签名

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model,                    # 要训练的 policy 模型
    ref_model=ref_model,      # 冻结的参考模型（None 时自动复制 model）
    args=DPOConfig(...),      # 训练配置
    train_dataset=dataset,    # 训练集
    eval_dataset=eval_dataset,# 验证集
    tokenizer=tokenizer,      # 分词器
    peft_config=peft_config,  # LoRA 配置（可选）
)
```

### 2.2 DPOConfig 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | 0.1 | **最核心参数**。KL 惩罚系数，控制 policy 偏离 ref 的程度 |
| `loss_type` | `"sigmoid"` | 损失类型。`"sigmoid"` = 标准 DPO，还有 `"hinge"`, `"ipo"`, `"kto_pair"` 等变体 |
| `label_smoothing` | 0.0 | 标签平滑，>0 可防止过拟合偏好对 |
| `max_length` | None | chosen/rejected 的总最大长度 |
| `max_prompt_length` | None | prompt 部分最大长度 |
| `max_target_length` | None | response 部分最大长度 |
| `generate_during_eval` | False | 验证时是否生成回复（方便查看效果） |
| `precompute_ref_log_probs` | False | 是否预先计算 ref 的 log prob（省显存） |

### 2.3 train_step 内部流程

```
train_step(batch):
    1. policy 计算: log π_θ(chosen), log π_θ(rejected)
    2. ref 计算:    log π_ref(chosen), log π_ref(rejected)   [@torch.no_grad]
    3. 计算 log ratio:
       chosen_logratios  = log π_θ(chosen)  - log π_ref(chosen)
       rejected_logratios = log π_θ(rejected) - log π_ref(rejected)
    4. loss = -log σ(β × (chosen_logratios - rejected_logratios))
    5. 反向传播，只更新 policy
```

### 2.4 β 的影响

```
β 大 (0.5+):  policy 必须紧跟 ref，保守更新，不容易崩但学得慢
β 小 (0.01):  policy 可大幅偏离 ref，激进更新，学得快但可能不稳定
β = 0.1:      本项目设置，常见默认值
```

### 2.5 loss_type 变体

| loss_type | 公式 | 特点 |
|-----------|------|------|
| `"sigmoid"` | $-\log\sigma(\beta \cdot \Delta)$ | 标准 DPO |
| `"hinge"` | $\max(0, 1 - \beta \cdot \Delta)$ | SVM 风格，对已学好的样本不再施加梯度 |
| `"ipo"` | $(\Delta - 1/(2\beta))^2$ | 回归风格，目标是固定的 margin |
| `"kto_pair"` | KTO 损失 | 不需要成对，只需知道单个回复好/差 |

其中 $\Delta = \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$

---

## 三、PPOTrainer

### 3.1 初始化签名

```python
from trl import PPOTrainer, PPOConfig

trainer = PPOTrainer(
    args=PPOConfig(...),           # 训练配置
    processing_class=tokenizer,    # 分词器
    policy=policy,                 # 策略模型（被训练）
    ref_policy=ref_policy,         # 参考模型（冻结）
    reward_model=reward_model,     # 奖励模型（冻结）
    value_model=value_model,       # 价值模型（被训练）
    train_dataset=dataset,         # 训练集
)
```

**注意：PPOTrainer 需要传入 4 个模型**。

### 3.2 PPOConfig 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **生成相关** | | |
| `response_length` | 53 | 生成回复的最大 token 数 |
| `temperature` | 0.7 | 生成温度 |
| `missing_eos_penalty` | None | 回复没有 EOS 时的惩罚分 |
| **PPO 算法** | | |
| `num_ppo_epochs` | 4 | 每批数据的 PPO 更新轮数 |
| `cliprange` | 0.2 | PPO clip 范围 ε |
| `kl_coef` | 0.05 | KL 惩罚系数（加在奖励上） |
| `whiten_rewards` | False | 是否白化奖励 |
| **Value Model** | | |
| `vf_coef` | 0.1 | value loss 的权重 |
| `cliprange_value` | 0.2 | value function 的 clip 范围 |
| **GAE** | | |
| `gamma` | 1.0 | 折扣因子（文本生成一般用 1.0） |
| `lam` | 0.95 | GAE 的 λ 参数 |
| **训练控制** | | |
| `total_episodes` | None | 总训练 episode 数 |

### 3.3 train_step 内部流程

```
train_step(batch):
    ── 阶段 1: 生成 ──
    1. policy 根据 prompt 自回归生成 response（采样）
    
    ── 阶段 2: 打分 ──
    2. reward_model 对 (prompt + response) 打分 → reward
    3. 如果 response 没有 EOS → reward -= missing_eos_penalty
    4. reward -= kl_coef × KL(π_θ || π_ref)   [per-token KL]
    
    ── 阶段 3: 优势估计 ──
    5. value_model 估计每个 token 位置的 V(s)
    6. GAE 计算 per-token 优势:
       δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
       Â_t = Σ_k (γλ)^k × δ_{t+k}
    
    ── 阶段 4: 更新 ──
    7. PPO-Clip 更新 policy:
       ratio = π_θ(a|s) / π_old(a|s)
       L_clip = min(ratio × Â, clip(ratio, 1-ε, 1+ε) × Â)
    
    8. MSE 更新 value_model:
       L_value = (V_θ(s) - V_target)²
    
    9. 总 loss = -L_clip + vf_coef × L_value
```

### 3.4 关键参数调参指南

```
kl_coef:
  大 (0.2)  → policy 被拉住，不敢偏离 ref → 安全但慢
  小 (0.01) → policy 自由探索 → 快但可能 reward hacking
  推荐: 0.05 (默认)

cliprange:
  大 (0.3) → 允许大步更新 → 学得快但可能不稳定
  小 (0.1) → 保守更新 → 稳定但慢
  推荐: 0.2 (默认)

num_ppo_epochs:
  大 (8)  → 每批数据反复利用 → 样本效率高但可能过拟合
  小 (1)  → 只用一次 → 稳定但样本效率低
  推荐: 4 (默认)

gamma=1.0, lam=0.95:
  文本生成标准设置，一般不需要改
```

---

## 四、GRPOTrainer

### 4.1 初始化签名

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model,                    # policy 模型（ref 自动从初始权重复制）
    reward_funcs=[reward_fn1, ...], # 奖励函数列表（Python 函数！）
    args=GRPOConfig(...),           # 训练配置
    train_dataset=dataset,          # 训练集
    peft_config=peft_config,        # LoRA 配置（可选）
)
```

**独特之处：`reward_funcs` 是 Python 函数列表，不是模型。**

### 4.2 GRPOConfig 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **GRPO 特有** | | |
| `num_generations` | 8 | **最核心参数**。每个 prompt 生成几个回复 |
| `beta` | 0.04 | KL 惩罚系数 |
| `epsilon` | 0.2 | PPO clip 范围 |
| **生成相关** | | |
| `max_prompt_length` | 256 | prompt 最大长度 |
| `max_completion_length` | 256 | 生成回复最大长度 |
| `temperature` | 0.9 | 生成温度（较高以获得多样性） |
| **训练** | | |
| `num_ppo_epochs` | 1 | 每批数据的更新轮数 |
| **vLLM 加速** | | |
| `use_vllm` | False | 是否用 vLLM 加速生成 |
| `vllm_gpu_utilization` | 0.5 | vLLM GPU 使用率 |

### 4.3 train_step 内部流程

```
train_step(batch):
    ── 阶段 1: 批量生成 ──
    1. 对每个 prompt，policy 生成 N 个 response（num_generations=N）
    
    ── 阶段 2: 规则打分 ──
    2. 对每个 response，调用所有 reward_funcs:
       for fn in reward_funcs:
           scores = fn(completions, answer=answer, ...)
       total_reward[i] = sum(all scores for response i)
    
    ── 阶段 3: 组内归一化 ──
    3. 对同一 prompt 的 N 个回复:
       μ = mean(total_rewards)
       σ = std(total_rewards)
       Â_i = (r_i - μ) / σ
    
    ── 阶段 4: 更新 ──
    4. ref_model 计算 per-token KL 惩罚
    5. PPO-Clip 更新:
       ratio = π_θ / π_old
       L = min(ratio × Â, clip(ratio, 1-ε, 1+ε) × Â) - β × KL
       # Â 是 sequence-level，所有 token 共享
       # 除以 response 长度做归一化
    6. 只更新 policy（无 value model）
```

### 4.4 reward_funcs 签名

```python
def my_reward_func(completions, **kwargs):
    """
    Args:
        completions: List[List[Dict]]
            completions[i][0]["content"] → 第 i 个回复的文本
        **kwargs: 数据集中的其他字段
            如 answer, question 等

    Returns:
        List[float]: 每个回复的奖励分
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        reward = ...  # 你的打分逻辑
        rewards.append(reward)
    return rewards
```

本项目定义了两个奖励函数：

```python
# 1. 答案正确性（0 或 1）
def accuracy_reward(completions, answer, **kwargs):
    # 从 <answer> 标签提取答案，用 math_verify 验证
    ...

# 2. 格式正确性（0 或 1）
def format_reward(completions, **kwargs):
    # 检查是否符合 <think>...</think><answer>...</answer>
    ...
```

### 4.5 num_generations 的影响

```
N 大 (16): 优势估计更准确，但生成开销 16×
N 小 (2):  估计噪声大，但快
N = 4:     本项目设置，平衡精度和速度

极端情况：
N = 1 → 只有一个回复，std=0，无法归一化 → 不可用
N = 2 → 只比两个回复，非常嘈杂
N ≥ 4 → 通常足够得到有意义的组内排名
```

### 4.6 beta (KL 系数) 对比

```
DPO  的 beta = 0.1    → 在 loss 函数中，较大的约束
PPO  的 kl_coef = 0.05 → 在 reward 中减去，中等约束
GRPO 的 beta = 0.001  → 在 loss 中减去，极小约束 → 允许大幅探索

为什么 GRPO 的 beta 这么小？
→ GRPO 的 clip 机制已经限制了更新幅度
→ 组内归一化本身有正则化效果
→ 数学/推理任务需要更大的探索空间
```

---

## 五、三者核心差异总结

### 5.1 API 层面

| | DPOTrainer | PPOTrainer | GRPOTrainer |
|---|---|---|---|
| **输入模型数** | 2 (policy + ref) | 4 (policy + ref + RM + value) | 1 (policy，ref 自动创建) |
| **奖励来源** | 隐式（log prob 差） | `reward_model` 对象 | `reward_funcs` 函数列表 |
| **数据格式** | chosen + rejected | prompt only | question + answer |
| **生成行为** | 不生成 | 生成 1 个/prompt | 生成 N 个/prompt |

### 5.2 算法层面

| | DPOTrainer | PPOTrainer | GRPOTrainer |
|---|---|---|---|
| **核心参数** | `beta` | `kl_coef`, `cliprange`, `gamma`, `lam` | `num_generations`, `beta` |
| **优势估计** | 无（隐式） | GAE (per-token) | 组内归一化 (per-sequence) |
| **Clip 机制** | 无 | PPO-Clip | PPO-Clip |
| **Value Model** | 无 | 有（需训练） | 无 |
| **反向传播** | 1 次（policy） | 2 次（policy + value） | 1 次（policy） |

### 5.3 计算开销

| 操作 | DPOTrainer | PPOTrainer | GRPOTrainer |
|------|-----------|-----------|-------------|
| 自回归生成 | 0 | 1 次 | N 次 |
| Policy 前向 | 2 次 | 1 次 | N 次 |
| Ref 前向 | 2 次 | 1 次 | N 次 |
| RM/规则 前向 | 0 | 1 次 | ≈0（规则函数） |
| Value 前向 | 0 | 1 次 | 0 |
| **总前向** | 4 次 | 4 次 | 2N 次 |
| **总反向** | 1 次 | 2 次 | 1 次 |
| **瓶颈** | 无生成，最快 | 4 模型内存 | N 次生成 |

### 5.4 本项目的参数对比

| 参数 | DPO | PPO | GRPO |
|------|-----|-----|------|
| `learning_rate` | 2e-5 | 3e-6 | 5e-7 |
| `beta` / `kl_coef` | 0.1 | 0.05 | 0.001 |
| `batch_size` | 4 | 1 | 4 |
| `grad_accum` | 8 | 4 | 1 |
| `lora_r` | 8 | 8 | 16 |
| `lora_alpha` | 16 | 16 | 32 |
| `gradient_checkpointing` | True | True | False |
| 量化 | 无 | 无 | QLoRA 4bit |

**趋势解读**：
- **学习率**：DPO(2e-5) > PPO(3e-6) > GRPO(5e-7) — 强化学习方法需要更小的学习率
- **KL 系数**：DPO(0.1) > PPO(0.05) > GRPO(0.001) — GRPO 需要更多探索空间
- **LoRA rank**：GRPO(16) > DPO/PPO(8) — 更强的 adapter 补偿 4bit 量化损失