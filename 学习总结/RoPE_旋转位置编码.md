# RoPE（Rotary Position Embedding，旋转位置编码）

> 论文：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

## 1. 一句话概括

RoPE 通过**旋转向量**的方式将位置信息编码到 token 的表示中，使得两个 token 之间的注意力分数**只取决于它们的相对距离**，而不是绝对位置。

---

## 2. 为什么需要位置编码？

Transformer 的自注意力机制是**排列不变的**（permutation invariant）——如果不加位置编码，"猫吃鱼"和"鱼吃猫"对模型来说没有区别。位置编码让模型知道每个 token 在序列中的位置。

### 位置编码的演进

| 方法 | 代表模型 | 类型 | 特点 |
|------|---------|------|------|
| Sinusoidal PE | 原始 Transformer | 绝对位置 | 固定公式，不学习 |
| Learned PE | BERT, GPT-2 | 绝对位置 | 可学习，但固定长度 |
| Relative PE | T5, Transformer-XL | 相对位置 | 只看相对距离 |
| ALiBi | BLOOM | 相对位置 | 在注意力分数上加偏置 |
| **RoPE** | LLaMA, Qwen, Mistral | 相对位置 | 旋转 Q/K 向量 |

---

## 3. RoPE 的核心思想

传统位置编码是**加法**（加在 embedding 上）。RoPE 是**乘法**（对 Query 和 Key 做旋转）。

关键直觉：
- 把 token 的向量表示看作复平面上的点
- 根据 token 的位置，旋转这个点一定的角度
- 两个 token 的注意力分数 = 旋转后的内积 = 只取决于旋转角度之差 = 相对位置

---

## 4. 数学公式

### 4.1 2D 情况（最简单的理解）

对于一个 2 维向量 $\mathbf{q} = (q_0, q_1)$，位置为 $m$ 时，RoPE 将其旋转角度 $m\theta$：

$$f(\mathbf{q}, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

这就是一个标准的 2D 旋转矩阵！

### 4.2 高维情况

对于 $d$ 维向量（$d$ 为偶数），将向量分成 $d/2$ 对，每对独立旋转：

$$R_m = \begin{pmatrix} R_m^{(1)} & & \\ & R_m^{(2)} & \\ & & \ddots \\ & & & R_m^{(d/2)} \end{pmatrix}$$

其中每个 2×2 块为：

$$R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

### 4.3 频率参数

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1$$

- $i=0$ 时：$\theta_0 = 1$（高频，变化快）
- $i=d/2-1$ 时：$\theta_{d/2-1} \approx 0$（低频，变化慢）

不同维度对应不同频率，类似傅里叶变换的多频率分解。

### 4.4 应用到注意力机制

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(R_m Q)(R_n K)^T}{\sqrt{d}}\right) V$$

其中 $R_m Q$ 和 $R_n K$ 分别是位置 $m$ 和 $n$ 的旋转后的 Query 和 Key。

---

## 5. 核心性质：相对位置编码

RoPE 的关键性质：

$$\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = \langle R_{m-n} \mathbf{q}, \mathbf{k} \rangle$$

**证明直觉**：旋转矩阵是正交矩阵，内积中两个旋转可以合并为一个相对旋转。

这意味着：位置 $m$ 的 query 和位置 $n$ 的 key 的注意力分数只取决于 $m - n$（相对距离），实现了相对位置编码。

---

## 6. 为什么 RoPE 这么流行？

| 优点 | 说明 |
|------|------|
| ✅ 相对位置编码 | 注意力只看相对距离，泛化性好 |
| ✅ 无需额外参数 | 不用学习位置嵌入，直接用数学公式 |
| ✅ 可扩展到更长序列 | 配合插值方法可外推到超长上下文 |
| ✅ 计算高效 | 只是元素级乘法和加法，开销极小 |
| ✅ 远距离衰减 | 高频分量使远距离 token 的注意力自然衰减 |

---

## 7. 使用 RoPE 的主流模型

几乎所有新一代开源 LLM 都使用 RoPE：

- LLaMA / LLaMA 2 / LLaMA 3
- Qwen / Qwen 2 / Qwen 2.5
- Mistral / Mixtral
- DeepSeek / DeepSeek-V3
- Yi
- InternLM
- Baichuan 2

---

## 8. RoPE Scaling（扩展上下文长度）

RoPE 的一个限制是：训练时用的最大位置决定了模型能处理的最大序列长度。为了扩展上下文窗口，有以下方法：

### 8.1 Linear Scaling（线性插值）

$$\theta_i' = \theta_i / s$$

其中 $s$ 是缩放因子。相当于把位置"压缩"，让模型以为序列没那么长。

- 优点：简单有效
- 缺点：会损失精度

### 8.2 Dynamic NTK-Aware Scaling

$$\theta_i' = (10000 \cdot s)^{-2i/d}$$

动态调整频率基数，在不同长度下自适应。

- 优点：不需要微调就能外推
- 缺点：外推太远仍会退化

### 8.3 YaRN（Yet another RoPE extensioN）

结合 NTK scaling + 注意力缩放 + 温度调整，目前最先进的 RoPE 扩展方法。

### 在 MedicalGPT 中的使用

```python
# supervised_finetuning.py 中的参数
--rope_scaling linear   # 线性插值
--rope_scaling dynamic  # 动态NTK
```

---

## 9. 代码实现（简化版）

```python
import torch

def rotary_embedding(x, seq_len, dim, base=10000):
    """对输入 x 应用 RoPE"""
    # 计算频率
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算位置角度
    positions = torch.arange(seq_len).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq_len, dim/2]
    
    # 构造 cos 和 sin
    cos = torch.cos(angles)  # [seq_len, dim/2]
    sin = torch.sin(angles)  # [seq_len, dim/2]
    
    # 将 x 分成两半
    x1 = x[..., 0::2]  # 偶数维度
    x2 = x[..., 1::2]  # 奇数维度
    
    # 应用旋转
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    # 交错合并
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

---

## 10. RoPE vs 其他位置编码

| 特性 | Sinusoidal | Learned | ALiBi | RoPE |
|------|-----------|---------|-------|------|
| 类型 | 绝对 | 绝对 | 相对 | 相对 |
| 参数 | 0 | O(L×d) | 0 | 0 |
| 长度外推 | 差 | 不支持 | 好 | 好（+scaling） |
| 计算开销 | 低 | 低 | 低 | 低 |
| 主流程度 | 已淘汰 | BERT时代 | BLOOM | **当前主流** |

---

## 11. 常见问题

### Q: RoPE 只作用于 Q 和 K，不作用于 V？
A: 对。因为位置信息通过 attention score（Q·K 的内积）传递，V 不需要额外的位置编码。

### Q: 为什么用 10000 作为基数？
A: 这是经验选择。较大的基数使低频分量变化更慢，能捕捉更长距离的位置关系。一些模型（如 CodeLlama）使用 1000000 作为基数来支持超长上下文。

### Q: RoPE 和 Attention Mask 有什么关系？
A: 没有直接关系。RoPE 编码位置信息，Attention Mask 控制哪些 token 能被看到（因果 mask 确保自回归）。两者独立工作。

---

## 12. 总结

```
RoPE = 对 Q 和 K 向量按位置旋转

核心公式：f(q, m) = R_m · q （旋转矩阵 × 向量）
核心性质：<R_m·q, R_n·k> 只取决于 m-n（相对位置）
频率设计：θ_i = 10000^(-2i/d)（多频率分解）

优点：无参数、相对位置、计算高效、可扩展
地位：几乎所有主流开源 LLM 的标配
```

---

## 参考资料

- 原始论文：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Extending Context Window via RoPE Scaling](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [博客: Understanding RoPE](https://blog.eleuther.ai/rotary-embeddings/)