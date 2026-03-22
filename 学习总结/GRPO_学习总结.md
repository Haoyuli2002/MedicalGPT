# GRPO（Group Relative Policy Optimization）学习总结

## 目录

- [一、核心思想](#一核心思想)
- [二、训练 Pipeline](#二训练-pipeline)
- [三、奖励函数](#三奖励函数)
  - [3.1 accuracy_reward](#31-accuracy_reward)
  - [3.2 format_reward](#32-format_reward)
- [四、Group Relative 优势估计](#四group-relative-优势估计)
- [五、目标函数](#五目标函数)
- [六、具体例子](#六具体例子)
- [七、数据格式与训练参数](#七数据格式与训练参数)
- [八、GRPO vs PPO vs DPO](#八grpo-vs-ppo-vs-dpo)

---

## 一、核心思想

GRPO（来自 DeepSeek-R1）的核心创新：**用组内比较替代 value model**。

- 对每个 prompt 生成**一组**回复（如 4 个）
- 用规则函数（非学习的 RM）给每个回复打分
- 用组内均值/标准差归一化奖励 → 得到优势
- 无需 value model、无需 reward model → 只需 2 个模型（policy + ref）

```
PPO:  需要 4 个模型（policy + ref + reward_model + value_model）
DPO:  需要 2 个模型（policy + ref），但需要偏好对数据
GRPO: 需要 2 个模型（policy + ref），只需 {question, answer} 数据
```

---

## 二、训练 Pipeline

```
每个训练 step:
  1. 采样一个 prompt
  2. Policy 生成 N 个回复（num_generations=4）
  3. 奖励函数对每个回复打分
  4. 组内归一化 → 优势 Â_i
  5. PPO-Clip 风格更新 policy（带 KL 惩罚）
```

与 PPO 的关键区别：
- PPO 每个 prompt 生成 1 个回复，用 GAE + value model 估计优势
- GRPO 每个 prompt 生成 N 个回复，用组内统计量估计优势

---

## 三、奖励函数

GRPO 使用**基于规则的奖励函数**（不是学习的 RM），这是它能去掉 reward model 的关键。

### 3.1 accuracy_reward

验证模型回答是否正确（0 或 1）：

```python
def accuracy_reward(completions, answer, **kwargs):
    for content, sol in zip(contents, answer):
        gold_parsed = parse(sol)                    # 解析标准答案
        answer_parsed = parse(extract_answer(content))  # 从 <answer> 标签提取并解析
        reward = float(verify(answer_parsed, gold_parsed))  # 数学验证
    return rewards
```

- 用 `math_verify` 库进行数学等价性验证
- 从 `<answer>` 标签中提取答案
- 正确 → 1.0，错误 → 0.0

### 3.2 format_reward

检查输出格式是否正确（0 或 1）：

```python
def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think><answer>.*?</answer>$"
    matches = [re.match(pattern, content) for content in contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards
```

要求格式：`<think>推理过程</think><answer>答案</answer>`

**总奖励 = accuracy_reward + format_reward**（最高 2.0）。

---

## 四、Group Relative 优势估计

这是 GRPO 的核心创新，替代 PPO 中的 GAE + value model。

### 公式

对一个 prompt 生成 N 个回复，奖励为 $r_1, r_2, ..., r_N$：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_N)}{\text{std}(r_1, ..., r_N)}$$

### 直觉

- 不关心绝对分数，只关心**组内相对排名**
- 奖励高于组内平均 → 优势 > 0 → 增大概率
- 奖励低于组内平均 → 优势 < 0 → 减小概率
- 标准差归一化确保优势的量级一致

### 与 PPO GAE 的对比

| | PPO (GAE) | GRPO (Group Relative) |
|---|---|---|
| 需要 value model | ✅ | ❌ |
| 基线来源 | V(s) 价值函数 | 组内均值 μ |
| 计算粒度 | per-token | per-sequence |
| 额外模型开销 | 高（训练 value model） | 无 |

---

## 五、目标函数

GRPO 使用 PPO-Clip 风格的目标函数，加上 KL 惩罚：

$$L_{GRPO} = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(r_t^{(i)}(\theta)\hat{A}_i,\;\text{clip}(r_t^{(i)}(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_i\right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})\right)\right]$$

其中：

$$r_t^{(i)}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$$

注意：
- $\hat{A}_i$ 是 **sequence-level** 的优势（同一回复的所有 token 共享）
- 除以 $|o_i|$（回复长度）做长度归一化
- β 是 KL 惩罚系数（`run_grpo.sh` 中设为 0.001）

---

## 六、具体例子

### 训练数据

```json
{"question": "57+29=?", "answer": "86"}
```

### Step 1: 生成 4 个回复

```
回复 1: <think>57+29，7+9=16进1，5+2+1=8，所以86</think><answer>86</answer>
回复 2: <think>57加29等于86</think><answer>86</answer>
回复 3: <think>57+29=84吧</think><answer>84</answer>
回复 4: 57+29等于86
```

### Step 2: 打分

| 回复 | accuracy | format | 总奖励 |
|------|----------|--------|--------|
| 回复 1 | 1.0 ✅ | 1.0 ✅ | **2.0** |
| 回复 2 | 1.0 ✅ | 1.0 ✅ | **2.0** |
| 回复 3 | 0.0 ❌ | 1.0 ✅ | **1.0** |
| 回复 4 | 0.0 ❌ | 0.0 ❌ | **0.0** |

### Step 3: 组内归一化

```
μ = (2.0 + 2.0 + 1.0 + 0.0) / 4 = 1.25
σ = std([2.0, 2.0, 1.0, 0.0]) = 0.829

回复 1: Â = (2.0 - 1.25) / 0.829 = +0.90  ← 增大概率
回复 2: Â = (2.0 - 1.25) / 0.829 = +0.90  ← 增大概率
回复 3: Â = (1.0 - 1.25) / 0.829 = -0.30  ← 轻微减小
回复 4: Â = (0.0 - 1.25) / 0.829 = -1.51  ← 大幅减小
```

### Step 4: 更新

- 回复 1、2 → 答对且格式对 → **鼓励**
- 回复 3 → 答错但格式对 → **轻微抑制**（格式功劳被答错抵消）
- 回复 4 → 答错且格式错 → **强烈抑制**

模型学到：**先用 `<think>` 推理，再用 `<answer>` 给出正确答案**。

---

## 七、数据格式与训练参数

### 数据格式

只需 `question` 和 `answer`，无需偏好对：

```json
{"question": "肛门病变可能是什么疾病的症状?", "answer": "食管克罗恩病"}
{"question": "1+2=?", "answer": "3"}
```

### System Prompt（R1 风格）

```
The assistant first thinks about the reasoning process in the mind and then 
provides the user with the answer. The reasoning process and answer are 
enclosed within <think> </think> and <answer> </answer> tags.
```

### 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_generations` | 4 | 每个 prompt 生成的回复数 |
| `beta` | 0.001 | KL 惩罚系数（很小，允许较大探索） |
| `learning_rate` | 5e-7 | 学习率（比 DPO/SFT 更小） |
| `lr_scheduler_type` | cosine | 余弦退火 |
| `max_prompt_length` | 16384 | prompt 最大长度 |
| `max_completion_length` | 512 | 生成回复最大长度 |
| `per_device_train_batch_size` | 4 | 每 GPU batch size |
| `use_peft` / `qlora` | True | QLoRA 4bit 量化训练 |
| `lora_r` / `lora_alpha` | 16 / 32 | LoRA 配置 |
| `gradient_checkpointing` | False | GRPO + LoRA 时需关闭 |

---

## 八、GRPO vs PPO vs DPO

| 维度 | GRPO | PPO | DPO |
|------|------|-----|-----|
| **奖励来源** | 规则函数 | 学习的 RM | 隐式（偏好对） |
| **优势估计** | 组内归一化 | GAE + value model | 不需要 |
| **模型数量** | 2（policy + ref） | 4 | 2（policy + ref） |
| **显存需求** | 中（多次生成） | 高（4 模型） | 低 |
| **数据格式** | {question, answer} | prompt only | {question, chosen, rejected} |
| **每步生成** | N 个/prompt | 1 个/prompt | 0 |
| **适用场景** | 可验证任务（数学、代码） | 通用 | 通用 |
| **训练稳定性** | 好 | 差（超参敏感） | 好 |
| **R1 风格推理** | ✅ 天然支持 | 可以但复杂 | ❌ 不适合 |

### GRPO 的局限

- **依赖可验证的奖励函数**：需要能自动判断对错（数学、代码、事实问答）
- **不适合开放式任务**：如创意写作、对话风格——难以定义规则奖励
- **生成开销**：每个 prompt 需要生成 N 个回复，推理成本高于 DPO