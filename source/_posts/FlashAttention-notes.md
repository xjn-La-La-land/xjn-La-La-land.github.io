---
title: FlashAttention 系列论文精读（FA1～FA4）
date: 2026-06-19 12:00:00
tags: 
  - CS-notes
  - GPU
  - AI
comments: false
excerpt: 从 Online Softmax 出发，系统解读 FlashAttention 系列论文（FA1～FA4），涵盖算法推导、Recomputation、IO 复杂度分析，以及 FA3 针对 Hopper 的 Warp Specialization、FA4 针对 Blackwell 的 TMEM/流水线重设计。
mathjax: true
---







## 写在前面

> 我们从 online softmax 开始，系统地解读 FlashAttention 系列论文（FA1 ~ FA4）。
>
> 
>
> 从 FlashAttention 的演化路径来看，有这样一个明显的趋势，**算法创新和硬件细节的耦合越来越紧**。
>
> 如果说 FA1/FA2 还是 NeurIPS 风格的算法论文，那 FA3/FA4 就更像是 SOSP / ASPLOS 风格的系统论文。
>
> * FA1 通过 online softmax + tiling 让中间矩阵不落回 HBM，本质上是**把一个数学上等价的算法重新组织，让它符合 GPU memory hierarchy 的特性**；
> * FA2 在算法上减少 non-matmul 操作，并且加入 warp 级别的 partition 改进，涉及到 warp scheduling；
> * FA3 用上 Hopper 的 warp specialization（producer/consumer pipeline）+ wgmma + TMA，可以说是面向 Hopper 平台的 arch-specific 优化；
> * FA4 用上 Blackwell 的第五代 Tensor Core + Tensor Memory，更是面向 Blackwell 平台的 arch-specific 优化；
>
> 从 FA1 到 FA4，硬件感知视角越来越强烈，尤其是 FA3, FA4，出发点就是面向 NVIDIA 新一代架构的变化，在 FA 算法上进行优化和适配。





## [Online Softmax](https://arxiv.org/abs/1805.02867)

### 回顾标准 Softmax

* 对于一行注意力分数 $x \in \mathbb{R}^N$（即 $QK^\top$ 的一行），标准 softmax 为：
  $$
  \text{softmax}(x)i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
  $$

* 但是存在数值稳定性问题：**直接计算 $e^{x_i}$ 容易溢出**。

* 标准做法是减去最大值：
  $$
  m(x) := \max_i x_i
  $$

  $$
  \text{softmax}(x)i = \frac{e^{x_i - m(x)}}{\sum_{j=1}^{N} e^{x_j - m(x)}}
  $$

* 论文将这三个量分别定义为：

  > $$m(x) := \max_i x_i$$
  >
  > $$f(x) := \begin{bmatrix} e^{x_1 - m(x)} , \cdots , e^{x_N - m(x)} \end{bmatrix}$$
  >
  > $$\ell(x) := \sum_i f(x)_i = \sum_i e^{x_i - m(x)}$$
  >
  > $$\text{softmax}(x) = \frac{f(x)}{\ell(x)}$$

* 问题所在：计算 $m(x)$ 必须先扫描全部 $N$ 个元素，无法分块——这正是标准实现必须把整个 $N \times N$ 矩阵加载进内存的原因。

### 分块分解公式

* 论文的核心数学：把序列切成两块，证明可以用两块的局部统计量合并出全局结果。

  > 设 $x = \begin{bmatrix} x^{(1)} , x^{(2)} \end{bmatrix} \in \mathbb{R}^{2B}$，其中 $x^{(1)}, x^{(2)} \in \mathbb{R}^B$。
  >
  > 1) 合并最大值：$$m(x) = \max\left(m(x^{(1)}),\ m(x^{(2)})\right)$$ ——这是显然的，全局最大值等于两段最大值取最大。
  >
  > 2) 合并指数向量 $f$。
  >
  >    对第一段元素，
  >
  >    - $$f(x)_i = e^{x_i^{(1)} - m(x)}$$ （注意 $m(x)$ 是全局最大）
  >    - $f(x^{(1)})_i = e^{x_i^{(1)} - m(x^{(1)})}$
  >    - 因此 $$ f(x)_i = e^{x_i^{(1)} - m(x^{(1)})} \cdot e^{m(x^{(1)}) - m(x)} = f(x^{(1)})_i \cdot e^{m(x^{(1)}) - m(x)}$$
  >
  >    对第二段同理，所以：
  >
  >    $$f(x) = \begin{bmatrix} e^{m(x^{(1)}) - m(x)} \cdot f(x^{(1)}) , e^{m(x^{(2)}) - m(x)} \cdot f(x^{(2)}) \end{bmatrix}$$
  >
  > 3. 合并归一化因子 $\ell$
  >
  >    对 $f(x)$ 求和：
  >    $$
  >    \boxed{\ell(x) = e^{m(x^{(1)}) - m(x)} \cdot \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \cdot \ell(x^{(2)})}
  >    $$
  >    **这就是关键公式：只需要两段各自的 $(m, \ell)$，就能合并出全局 $(m, \ell)$**，无需看到全部数据。

### 推广到多块——增量更新

* 将上述二段分解推广到 $T$ 块，每次处理一个新块时做增量更新：
* 处理完前 $t$ 块后，维护：
    - $m^{(t)}$：前 $t$ 块的全局最大值
    - $\ell^{(t)}$：前 $t$ 块的（重新对齐的）指数和

* 当第 $t+1$ 块到来时，设这块的局部量为 $\tilde{m} = \max(x^{(t+1)})$，$\tilde{\ell} = \sum_i e^{x_i^{(t+1)} - \tilde{m}}$：
  * $$m^{(t+1)} = \max\left(m^{(t)},\ \tilde{m}\right)$$
  * $$\ell^{(t+1)} = e^{m^{(t)} - m^{(t+1)}} \cdot \ell^{(t)} + e^{\tilde{m} - m^{(t+1)}} \cdot \tilde{\ell}$$
* 处理完全部 $T$ 块后，最终 $m^{(T)}$ 和 $\ell^{(T)}$ 与一次性计算全局 softmax 完全等价。

### 总结：Online Softmax 的本质

Online Softmax 的精妙之处在于：$e^{m^{\text{old}} - m^{\text{new}}}$ 这个修正因子，把"当时的局部最大值"和"全局最大值"的差异精确抹平，使得逐块计算与全局计算数学等价。



---



## [FlashAttention](https://arxiv.org/abs/2205.14135)

### Forward Pass

#### Standard Forward Propergation

##### 单 head

单个 head 的自注意力计算公式为
$$
\boxed{O = \text{softmax}(QK^\top) V}
$$
其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度，$d$ 是这个 head 的维度（标准场景下常常是 $N \gg d$）。

把它展开，就是三步：
$$
S=QK^\top \in \mathbb{R}^{N\times N},
$$

$$
P=\text{softmax}(S)\in \mathbb{R}^{N\times N},
$$

$$
O=PV \in \mathbb{R}^{N\times d}.
$$



<img src="/images/flash-attention/image-20260329143829014.png" alt="image-20260329143829014" style="zoom:33%;" />

> 矩阵 $$Q$$ 的第 $$i$$ 行 $$q_i$$ 表示第 $i$ 个 token 的 query 向量；
>
> 矩阵 $$K$$ 的第 $$j$$ 行 $$k_j$$ 表示第 $j$ 个 token 的 key 向量；
>
> 矩阵 $$S$$ 的 $s_{ij} = q_i⋅k_j$ 表示“第 $i$ 个位置去关注第 $j$​ 个位置”时的匹配分数，第 $i$ 行
> $$
> [s_{i1},s_{i2},\dots,s_{iN}]
> $$
> 就是位置 $i$ 对所有位置的注意力打分。

<img src="/images/flash-attention/image-20260329143857647.png" alt="image-20260329143857647" style="zoom:33%;" />

> 对 $S$ 的每一行做 Softmax 归一化 $p_i =\text{softmax}(s_i)$，得到：对于固定的 query，它应该如何在所有的 key/value 位置分配注意力权重。

<img src="/images/flash-attention/image-20260329143930669.png" alt="image-20260329143930669" style="zoom:33%;" />

> 矩阵 $V$ 的第 $j$ 行 $v_j$ 表示第 $j$ 个位置真正被读取、被聚合的 value 向量；
>
> 而输出矩阵 $O$ 的第 $i$ 行
> $$
> o_i=\sum_{j=1}^N p_{ij}v_j
> $$
> 对所有 token 的 value 向量使用 token $i$ 的注意力权重加权求和，表示“第 $i$ 个 token 在看完所有可见位置后，融合上下文得到的输出向量”。

##### 多 head

对于多头自注意力，输入通常先是 $$X\in\mathbb{R}^{N\times d_{\text{model}}}$$

然后通过三组线性投影得到总的
$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$
其中如果有 $h$ 个 head、每个 head 的维度是 $d$，通常满足 $d_{\text{model}}=h\cdot d$

于是投影后的总矩阵可以看成
$$
Q,K,V\in\mathbb{R}^{N\times (h d)}
$$
再把它们 reshape 成
$$
Q,K,V\in\mathbb{R}^{h\times N\times d}
$$
此时第 $m$ 个 head 会单独做一遍单头注意力：
$$
O^{(m)}=\mathrm{softmax}\!\bigl(Q^{(m)}(K^{(m)})^\top\bigr)V^{(m)},
\qquad
O^{(m)}\in\mathbb{R}^{N\times d}.
$$
也就是说，每个 head 都有自己独立的 $Q^{(m)},K^{(m)},V^{(m)}$，各自产生自己的注意力分布和输出。所有 head 算完以后，把它们在特征维拼接起来：
$$
O_{\text{cat}}\in\mathbb{R}^{N\times (h d)},
$$
再经过一个输出投影
$$
Y=O_{\text{cat}}W_O,\qquad Y\in\mathbb{R}^{N\times d_{\text{model}}}.
$$
因此，多头注意力相对于单头的变化，本质上不是公式变了，而是把一个大的表示空间切成多个 head，让每个 head 在自己的子空间里独立做 $\mathrm{softmax}(QK^\top)V$，再把结果拼回去。这样不同 head 可以学习不同类型的相关性，比如有的 head 更关注局部依赖，有的 head 更关注长程依赖，有的 head 更偏语义关系，有的 head 更偏位置关系。



---

#### 优化目标

标准实现里，通常会先算出 $S$，写入 HBM；再读回 $S$ 做 softmax 得到 $P$，再写入 HBM；最后再读回 $P$ 与 $V$ 相乘得到 $O$，并写回 HBM。

- $d$ 是“小维度”，通常几十到一百多
- **$N$** 是“大维度”，从几百一路到几万都有可能
- attention 难做、显存容易爆，主要就是因为会形成一个 $N\times N$ 的 score matrix

**FlashAttention 的目标是，不产生 $N×N$ 的中间矩阵，直接在分块的过程中得到最终的 $O$。**



#### 关键观察：输出 $O$ 的行与列独立

注意力输出 $O$ 的第 $i$ 行只取决于 $Q$ 的第 $i$ 行：
$$
o_i = \text{softmax}(q_i K^\top) V = \frac{\sum_j e^{s_{ij} - m_i} v_j}{\sum_j e^{s_{ij} - m_i}}
$$
其中 $s_{ij} = q_i \cdot k_j$，$m_i = \max_j s_{ij}$。这意味着**可以按行分块处理** $Q$，每段 $Q$ 对应的输出行可以独立计算。

#### 循环结构

FlashAttention 算子循环结构示意图

<img src="/images/flash-attention/绘图.png" alt="绘图" style="zoom:12%;" />

FlashAttention 的双重循环（Algorithm 1: FlashAttention Forward Pass）：

```
外层循环：遍历 K/V 的分块 j = 1, 2, ..., Tc
      内层循环：遍历 Q 的分块 i = 1, 2, ..., Tr
          在 SRAM 内：用 Qi, Kj, Vj 更新 Oi 的部分结果 
```

形象地看，就是在遍历注意力矩阵的所有 Block $(i, j)$，每次处理一个 Block 的计算，但从不把整个 $N×N$ 矩阵存下来：

```
注意力矩阵 (按 Q 行分块 × K 列分块)：
           K块1   K块2   K块3
  Q块1  [ S_11 | S_12 | S_13 ]  → 更新 O块1
  Q块2  [ S_21 | S_22 | S_23 ]  → 更新 O块2
  Q块3  [ S_31 | S_32 | S_33 ]  → 更新 O块3
```

对每个 $Q$ 块 $i$，需要依次看完所有 $K$ 块 $j=1,2,3,...$，才能完成这行的 softmax 归一化。

#### 输出 $O$ 的增量更新推导

对于 $Q$ 的第 $i$ 个分块，在处理完 $K/V$ 的第 $j$ 个分块后，如何更新输出 $O_i$？

定义符号：处理完前 $t$ 个 $K/V$ 块后，维护的状态是：
  - $m_i^{(t)}$：前 $t$ 块的全局最大值
  - $\ell_i^{(t)}$：前 $t$ 块的（对齐到 $m_i^{(t)}$ 的）指数和，$$\ell_i^{(t)} = \sum_{k\in 块 1..t}e^{s_{ik} - m_i^{(t)}}$$
  - $O_i^{(t)}$：基于前 $t$ 块的当前最佳估计输出，即"假设 $K$ 只有前 $t$ 块，softmax 归一化后的输出"。

$$
O_i^{(t)} = \frac{\sum_{k \in \text{块}1..t} e^{s_{ik} - m_i^{(t)}} \cdot v_k}{\ell_i^{(t)}}
$$

当第 $t+1$ 块到来时，设这块的局部统计量为：

* 第 $t+1$ 块内的行最大值：$$\tilde{m} = \max_{k\in 块 t+1} s_{ik} $$
* 第 $t+1$ 块内的指数和：$$\tilde{\ell} = \sum_{k\in 块 k+1} e^{s_{ik} - \tilde{m}}$$
* 未归一化的局部权重，行向量：$$\tilde{P} = \exp(S_{i,t+1} - \tilde{m})$$

步骤一：更新全局最大值和归一化因子（Online Softmax 公式）：

* $$m_i^{(t+1)} = \max(m_i^{(t)},\ \tilde{m})$$
* $$\ell_i^{(t+1)} = e^{m_i^{(t)} - m_i^{(t+1)}} \cdot \ell_i^{(t)} + e^{\tilde{m} - m_i^{(t+1)}} \cdot \tilde{\ell}$$

步骤二：更新输出 $O_i$。
$$
O_i^{(t+1)} = \frac{\overbrace{\sum_{k \in 块1..t} e^{s_{ik} - m_i^{(t+1)}} v_k}^{\text{旧块的贡献}} + \overbrace{\sum_{k \in 块t+1} e^{s_{ik} - m_i^{(t+1)}}
  v_k}^{\text{新块的贡献}}}{\ell_i^{(t+1)}}
$$
分别处理两项：

1. 旧块贡献：从 $O_i^{(t)}$ 的定义出发

   $$\sum_{k \in 块1..t} e^{s_{ik} - m_i^{(t+1)}} v_k = e^{m_i^{(t)} - m_i^{(t+1)}} \cdot \underbrace{\sum_{k \in 块1..t} e^{s_{ik} - m_i^{(t)}} v_k}_{= \ell_i^{(t)} \cdot
     O_i^{(t)}}= e^{m_i^{(t)} - m_i^{(t+1)}} \cdot \ell_i^{(t)} \cdot O_i^{(t)}$$

2. 新块贡献：

   $$\sum_{k \in 块t+1} e^{s_{ik} - m_i^{(t+1)}} v_k = e^{\tilde{m} - m_i^{(t+1)}} \cdot \underbrace{\sum_{k\in 块 t+1} e^{s_{ik} - \tilde{m}} v_k}_{= \tilde{P} \cdot V_{t+1}}= e^{\tilde{m} - m_i^{(t+1)}} \cdot \tilde{P} \cdot V_{t+1}$$

3. 合并：

$$
\boxed{O_i^{(t+1)} = \frac{1}{\ell_i^{(t+1)}} \left( e^{m_i^{(t)} - m_i^{(t+1)}} \cdot \ell_i^{(t)} \cdot O_i^{(t)} + e^{\tilde{m} - m_i^{(t+1)}} \cdot \tilde{P} \cdot
   V_{t+1} \right)}
$$

​	这正是 Algorithm 1 第 12 行的公式，用矩阵形式写就是：

$$
O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}\left(\text{diag}(\ell_i) e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij}
  V_j\right)
$$

#### 整体算法

> 初始化：$O_i = 0$，$ℓ_i = 0$，$m_i = -∞$
>
> 对每个 $K/V$ 块 $j$：
>
> 1. 计算局部得分：$S_{ij} = Q_i @ K_j^\top$ ← 在 SRAM 内
> 2. 计算局部统计：
>    * $\tilde{m} = \text{rowmax}(S_{ij})$
>    * $\tilde{P} = e^{S_{ij} - \tilde{m}}$
>    * $\tilde{\ell} = \text{rowsum}(\tilde{P})$
> 3. 更新全局统计：
>    * $m_{new} = \max(m_i, \tilde{m})$
>    * $$\ell_{new} = e^{m_i - m_{new}} * \ell_i + e^{\tilde{m} - m_{new}} * \tilde{\ell}$$
> 4. 更新输出：$$O_i = (1/\ell_{new}) * (e^{m_i - m_{new}}* \ell_i * O_i + e^{\tilde{m} - m_{new}} * \tilde{P} @ V_j)$$
> 5. $m_i ← m_{new}$，$\ell_i ← \ell_{new}$
>
> 遍历完所有 $K/V$ 块后，$O_i$ 就是正确结果。





---



### Backward Pass

#### Standard Attention Backward Propergation

先看标准 Attention 的反向传播（Algorithm 3）：

> 输入：$Q$, $K$, $V$, $dO \in R^{N×d}$，还有前向保存的 $P \in R^{N×N}$
>
> 1. $dV = P^\top \cdot dO$              → 需要读 $P$（N×N）
> 2. $dP = dO \cdot V^\top$              → 产生 $dP$（N×N），写入 HBM
> 3. $dS_{ij} = P_{ij}(dP_{ij} - \sum_l P_{il} dP_{il})$  → 需要同时读 $P$ 和 $dP$（各 N×N）
> 4. $dQ = dS \cdot K$
> 5. $dK = dS^\top \cdot Q$

问题的根源：步骤 1 和 3 都依赖 $P \in \mathbb{R}^{N \times N}$，必须在前向时保存，占用 $O(N^2)$ 显存。

#### 推导梯度公式

设损失函数为标量 $\phi$，输出梯度为 $dO = \frac{\partial \phi}{\partial O}$，目标是求 $dQ, dK, dV$。

1. 求 $dV$

   由 $O = PV$，链式法则得：$$dV = P^\top dO \quad \Rightarrow \quad dv_j = \sum_i P_{ij} \cdot do_i = \sum_i \frac{e^{q_i^\top k_j}}{L_i} do_i$$

   其中 $L_i = \sum_j e^{q_i^\top k_j}$ 是第 $i$ 行的归一化因子（即论文的 $\ell_i$）。

   **关键：只要已知 $L_i$，就可以逐块累加，无需存 $P$。**

2. 求 $dS$（softmax 的反向传播）

   由 $O = PV$ 得 $dP = dO \cdot V^\top$，即 $dP_{ij} = do_i^\top v_j$。

   Softmax 的 Jacobian 为 $\text{diag}(P_{i:}) - P_{i:}P_{i:}^\top$，所以：$$dS_{i:} = P_{i:} \circ dP_{i:} - (P_{i:}^\top dP_{i:}) \cdot P_{i:}$$

   定义一个标量 $D_i$（论文 Eq. 4 的核心）：
   $$
   \boxed{D_i = P_{i:}^\top dP_{i:} = \sum_j P_{ij} dP_{ij} = \sum_j \frac{e^{q_i^\top k_j}}{L_i} \cdot do_i^\top v_j = do_i^\top \underbrace{\sum_j \frac{e^{q_i^\top k_j}}{L_i}
      v_j}_{= o_i} = do_i^\top o_i}
   $$
   这是这一推导的精华：$D_i$ 本来需要对长度 $N$ 的向量 $P_{i:}$ 和 $dP_{i:}$ 做点积（$N$ 个元素），但论文发现它等价于 $dO$ 和 $O$ 对应行的点积——这两个向量长度只有 $d$（远小于 $N$），可以直接在 SRAM 里算。

   于是 $dS$ 化简为：$$dS_{ij} = P_{ij}(dP_{ij} - D_i) =P_{ij}(dP_{ij} - do_i^\top o_i)$$

3. 求 $dQ$ 和 $dK$

   由 $S_{ij} = q_i^\top k_j$，反向得：

   $$dq_i = \sum_j dS_{ij} k_j = \sum_j P_{ij}(dP_{ij} - D_i) k_j$$

   $$dk_j = \sum_i dS_{ij} q_i = \sum_i P_{ij}(dP_{ij} - D_i) q_i$$

   两个公式都是逐块累加的形式，无需全局的 $P$ 矩阵，只需能在当前块内重新计算 $P_{ij}$。

#### Recomputation——如何在反向时重建 P

前向传播结束后，FlashAttention 只保存：

| 保存内容             | 大小                  | 用途                                                  |
| -------------------- | --------------------- | ----------------------------------------------------- |
| $O$（输出）          | $N \times d$          | 计算 $D_i = dO_i^\top O_i$                            |
| $\ell$（归一化因子） | $N$                   | 重建 $P_{ij}$                                         |
| $m$（行最大值）      | $N$                   | 重建 $P_{ij}$                                         |
| $Q, K$（输入）       | $2 \times N \times d$ | 重新计算 $S_{ij}$，进而算 $P_{ij}$；以及累加 $dQ, dK$ |
| $V$（输入）          | $N \times d$          | 计算 $dP_{ij} = dO_i V_j^\top$，以及累加 $dV$         |
| PRNG 状态 $R$        | 常数                  | 复现 dropout mask                                     |

不保存 $S$ 和 $P$（各 $N \times N$）。

在反向传播时，Algorithm 4 第 11-13 行对每个 $(i,j)$ 块现场重算：

```
S_ij = τ · Q_i @ K_j.T               ← 用已有的 Q_i, K_j 重算
P_ij = diag(ℓ_i)⁻¹ exp(S_ij - m_i)   ← 用保存的 ℓ_i, m_i 归一化
```

这就是 Recomputation：**用计算换存储，在 SRAM 里临时生成需要的 $P_{ij}$ 块，用完即丢。**



#### 整体算法

```
外层循环 j（遍历 K/V 块）：
      加载 K_j, V_j 到 SRAM，初始化 dK̃_j = 0, dṼ_j = 0

      内层循环 i（遍历 Q/O 块）：
          加载 Q_i, O_i, dO_i, ℓ_i, m_i 到 SRAM

          [Recompute] S_ij = τ Q_i K_jᵀ
          [Recompute] P_ij = diag(ℓ_i)⁻¹ exp(S_ij - m_i)
          [Recompute] dropout mask Z_ij（用 PRNG 状态 R 复现）

          dṼ_j += P_ij_dropped.T @ dO_i          ← 累加 dV

          dP_ij = dO_i @ V_j.T
          D_i = rowsum(dO_i ◦ O_i)               ← d 维点积，无需 N 维
          dS_ij = P_ij ◦ (dP_ij - D_i)           ← softmax 反向

          dQ_i += τ · dS_ij @ K_j                ← 累加 dQ，写回 HBM
          dK̃_j += τ · dS_ij.T @ Q_i              ← 累加 dK

      写 dK_j ← dK̃_j，dV_j ← dṼ_j 到 HBM
```

#### 与普通 Gradient Checkpointing 的对比

|          | 普通 Gradient Checkpointing  | FlashAttention Recomputation             |
| -------- | ---------------------------- | ---------------------------------------- |
| 思路     | 不存激活，反向时整个前向重跑 | 只存 $m, \ell, O$，反向时块内重算 $S, P$ |
| 内存     | $O(N)$（同 FlashAttention）  | $O(N)$                                   |
| 速度     | 慢，相当于跑了两次前向       | 快，因为省去了大量 HBM 读写              |
| 关键区别 | 速度换内存（trade-off）      | 内存和速度同时改善                       |

普通 Checkpointing 是以速度换内存；FlashAttention 的 Recomputation 额外增加的 FLOP 代价，被节省的 HBM 访问量超额补偿了，所以反而更快。



### IO 复杂度

#### Standard Attention vs FlashAttention 的 HBM 访问量

Standard Attention（Algorithm 0），三步各自读写 HBM：

| 步骤                    | 读入                   | 写出         | HBM 访问量         |
| ----------------------- | ---------------------- | ------------ | ------------------ |
| $S = QK^\top$           | $Q, K$（各 $Nd$）      | $S$（$N^2$） | $\Theta(Nd + N^2)$ |
| $P = \text{softmax}(S)$ | $S$（$N^2$）           | $P$（$N^2$） | $\Theta(N^2)$      |
| $O = PV$                | $P, V$（各 $N^2, Nd$） | $O$（$Nd$）  | $\Theta(Nd + N^2)$ |
| 合计                    |                        |              | $\Theta(Nd + N^2)$ |

FlashAttention 的证明思路：

* 外层循环遍历 $T_c$ 个 $K/V$ 块，**每块 $K/V$ 只从 HBM 加载一次**。内层遍历 $Q$ 时，每轮外循环要把全部 $Q$ 和 $O$ 都过一遍，所以 HBM 访问量是：
  $$
  \Theta(Nd + Nd \cdot T_c) = \Theta(Nd \cdot T_c)
  $$
  



* SRAM 的约束决定了块大小——$K/V$ 块（$B_c \times d$）、$Q/O$ 块（$B_r \times d$）、以及局部得分矩阵 $S_{ij}$（$B_r \times B_c$）都要装进 SRAM（大小 $M$），因此：
  $$
  B_c = \Theta\left(\frac{M}{d}\right) \quad \Rightarrow \quad T_c = \frac{N}{B_c} = \Theta\left(\frac{Nd}{M}\right)
  $$
  代入得：
  $$
  \Theta(Nd \cdot T_c) = \Theta!\left(\frac{N^2 d^2}{M}\right)
  $$

对比结论：

$$
\underbrace{\Theta(Nd + N^2)}_{\text{标准 Attention}} \quad \text{vs} \quad \underbrace{\Theta\!\left(\frac{N^2 d^2}{M}\right)}_{\text{FlashAttention}}
$$


以 A100 的典型参数（$d = 64$，$M \approx 192\text{KB} \approx 49152$ 个 float32 元素）代入，两者之比约为：$$\frac{N^2}{N^2 d^2 / M} = \frac{M}{d^2} = \frac{49152}{4096} \approx 12\times$$

#### FlashAttention 已是最优下界

* 命题：不存在任何精确 Attention 算法，对**所有** $M \in [d, Nd]$ 的 HBM 访问量都低于 $o\left(\frac{N^2 d^2}{M}\right)$。

* 证明（反证法）：

  > 假设存在这样的算法，对所有 $M$ 的 HBM 访问量为 $o\left(\frac{N^2 d^2}{M}\right)$。
  >
  > 取 $M = \Theta(Nd)$（SRAM 能装下所有输入的情形）代入：
  >
  > $$o\left(\frac{N^2 d^2}{Nd}\right) = o(Nd)$$
  >
  > 但是，Q、K、V、O 各自大小都是 $Nd$，它们一开始就在 HBM 里。任何算法至少要把输入读一遍、把输出写一遍，所以 HBM 访问量的下界是 $\Omega(Nd)$。
  >
  > 这与"访问量 $o(Nd)$"矛盾。

* 结论：FlashAttention 的 $\Theta\left(\frac{N^2 d^2}{M}\right)$ 复杂度是渐进最优的——不存在更好的精确 Attention 算法。

### Causal attention adaptation

Causal attention 要求第 $i$ 个 query 只能 attend 到位置 $j \leq i$ 的 key，对应的掩码矩阵是下三角矩阵：
$$
S_{ij} = \begin{cases} q_i^\top k_j & j \leq i \\ -\infty & j > i \end{cases}
$$
注意力矩阵被分成三类块：把 Q 和 K 都切成 $T$ 块，注意力矩阵变成 $T \times T$ 个小块。以 $4 \times 4$ 块为例：

```
K块:    1      2      3      4
Q块1 [ ▓▓▓▓ | XXXX | XXXX | XXXX ]
Q块2 [ ████ | ▓▓▓▓ | XXXX | XXXX ]
Q块3 [ ████ | ████ | ▓▓▓▓ | XXXX ]
Q块4 [ ████ | ████ | ████ | ▓▓▓▓ ]
████ = 完整有效块（全部 j ≤ i）
▓▓▓▓ = 对角块（部分有效，需要 mask）
XXXX = 完全无效块（全部 j > i，全是 -∞）
```

三类块的处理方式：

| 类型                 | 条件              | 处理方式               |
| -------------------- | ----------------- | ---------------------- |
| 完全无效块（上三角） | K块序号 > Q块序号 | 直接跳过，对输出无贡献 |
| 对角块               | K块序号 = Q块序号 | 计算时施加 causal mask |
| 完整有效块（下三角） | K块序号 < Q块序号 | 正常计算，无需 mask    |

    外层循环：遍历 K/V 的分块 j = 1, 2, ..., Tc
          内层循环：遍历 Q 的分块 i = 1, 2, ..., Tr
          		On Chip:
              i < j:  Q块在 K块左侧 → 全部 -∞ → 跳过
              i = j:  对角块 → 施加 causal mask（STAGE 2, on-band）
              i > j:  Q块在 K块右侧 → 全部有效 → 正常算（STAGE 1, off-band）





---



## [FlashAttention2](https://arxiv.org/abs/2307.08691)

FlashAttention2 针对 FlashAttention 原始算法的一些瓶颈进行了系统的性能优化，这些优化让 FA2 算子的速度达到了 FA1 的 2x，在 A100 上跑到硬件峰值算力的 50%~73%，接近 GEMM 的利用率。



### Forward Pass

#### 减少 Non-Matmul FLOPs

GPU 上的 matmul 操作由于 Tensor core 的加速，相比 non-matmal 操作会“便宜”很多。A100 上 matmul 吞吐是 non-matmul 的 **16×**（312 TFLOPs vs 19.5 TFLOPs），减少非矩阵乘运算收益显著。

在算法层面是如何减少 Non-Matmul FLOPs 的？这里有两个 trick：

1. 原始 FA 算法在更新 `O_i` 时，需要进行两次缩放（`ℓ_i(j-1)`和`1/ℓ_i(j)`，忽略最大值的缩放）：

   ```
   O(j) = ℓ(j)⁻¹ · (ℓ(j-1) e^{m(j-1)-m(j)} · O(j-1) + exp(S(j) - m(j)) · V(j))
   ```

   $$
   \boxed{O_i^{(t+1)} = \frac{1}{\ell_i^{(t+1)}} \left( e^{m_i^{(t)} - m_i^{(t+1)}} \cdot \ell_i^{(t)} \cdot O_i^{(t)} + e^{\tilde{m} - m_i^{(t+1)}} \cdot \tilde{P} \cdot
             V_{t+1} \right)}
   $$
   
    
   
   FA2 改为维护一个**未归一化的中间量** `Õ`，只在最后一步做一次除法：
   
    ```markdown
    Õ(j) = e^{m(j-1)-m(j)} · Õ(j-1) + exp(S(j) - m(j)) · V(j)
    O    = ℓ(Tc)⁻¹ · Õ(Tc)
    ```



2. 反向传播只需保存 **log_sum_exp** `L = m + log(ℓ)`，而不是分别保存 max `m` 和归一化因子 `ℓ`，保存量从 $2N$ 压缩到 $N$。为什么合并是等价的？反向传播需要重建 attention weight $P_{ij}$。用 $m$ 和 $\ell$ 分开的写法是：
   $$
   P_{ij} = \frac{\exp(S_{ij} \cdot \text{scale} - m_i)}{\ell_i}
   $$
   等价地：
   $$
   P_{i j}=\exp (S_{i j} \cdot \text { scale }-\underbrace{\left(m_{i}+\log \ell_{i}\right)}_{\mathrm{LSE}_{i}})
   $$
   这样只需要 LSE 一个量就能直接算出 $P_{ij}$，不需要 $m$ 和 $\ell$ 分开存。



#### 交换循环顺序，增加 Grid 并行度

* 原始 FA 算法中，**O_i、m_i、ℓ_i 是跨外层 j 循环的累积量**，第 j 次迭代的 `m_i(j)` 依赖 `m_i(j-1)`，`O_i(j)` 依赖 `O_i(j-1)`。因此 K/V tile 不能跨 thread block 并行计算。所以只能在 batch 和 head 维度做 thread block 级别的并行。当序列很长时，batch size 通常很小，导致 SM 利用率低。
* FA2 把前向传播的循环顺序对调，外层 i 的每次迭代维护的是**自己私有的** `O_i, m_i, ℓ_i`，不同 i 之间**完全没有数据依赖**。每个 thread block 认领一个 Q tile，从头到尾独立跑完内层循环，最后写回自己的 O tile ——天然并行。



#### Warp 级工作划分：从 Split-K 到 Split-Q

我们假设一个 thread block 负责处理一个 Q tile（`Br` × `d`），需要遍历所有 K/V tile 完成 attention 计算。block 内有 4 个 warp。

##### FA1 的 Split-K 方案

把 **K tile 沿 `Bc` 方向切分**给 4 个 warp，Q 由所有 warp 共享访问：

```
K^T (d × Bc)
┌────┬────┬────┬────┐
│ W1 │ W2 │ W3 │ W4 │  ← 每个 warp 负责 K^T 的 1/4 列
└────┴────┴────┴────┘
Q (Br × d)
┌────────┐
│Warp 1-4│  ← 所有 warp 都要读完整的 Q
|        |
└────────┘
```

每个 warp 算出一个**列切片**，拼起来才是完整的 `S = QK^T (Br × Bc)`。而 Softmax 每一行的 max 和 sum 需要看到该行**所有列**的值，所以 warp 之间**必须先通信，再算 softmax**：

```
Step 1: 各 warp 算局部 max → 写入 shared memory
Step 2: __syncthreads()
Step 3: 读回全局 max，各 warp 算局部 exp sum → 写入 shared memory  
Step 4: __syncthreads()
Step 5: 读回全局 sum，各 warp 完成 softmax 归一化
```

同样的，计算`O = P·V`，**V tile 也是沿着 `Bc` 方向切分**给 4 个 warp。每个 warp 计算：

```
P_k (Br×Bc/4) @ V_k (Bc/4 × d) → O_k (Br × d), k=1,2,3,4
```

最后各 warp 的 O 切片也要**规约到 shared memory** 累加得到完整输出。

##### FA2 的 Split-Q 方案

把 **Q 沿行方向切分**给 4 个 warp，K、V 由所有 warp 共享访问：

```
Q (Br × d)
┌─────────────────────┐
│        Warp 1       │  ← Br/4 行
├─────────────────────┤
│        Warp 2       │
├─────────────────────┤
│        Warp 3       │
├─────────────────────┤
│        Warp 4       │
└─────────────────────┘
K^T (d × Bc)
┌─────────────────────┐
│      Warp 1-4       │  ← 所有 warp 共享完整的 K、V
└─────────────────────┘
```

每个 warp 负责**完整的若干行**，softmax 所需的 max 和 sum 在单个 warp 内部就能算出来，不需要通信。并且各 warp 写的是**不同的内存区域**，不需要 sync，不需要规约！

```
Warp 1: Q1 (Br/4 × d) @ K^T (d × Bc) → S1 (Br/4 × Bc)  # 完整的行切片
Warp 2: Q2 (Br/4 × d) @ K^T          → S2 (Br/4 × Bc)
...

Warp 1 独立完成：
  m1 = rowmax(S1)          # 只看自己的行，不需要别人的数据
  ℓ1 = rowsum(exp(S1 - m1))
  O1 = exp(S1 - m1) @ V    # V 从 shared memory 读，但只读不写
  # 写回自己负责的 O 行
```

Split-Q 的 warp 分配策略更优，本质上因为**softmax 是行方向的，输出 O 也是行独立的**。这也和 FA2 在 thread block 级别"外循环走 Q"的设计一脉相承——**行独立性**从 block 级到 warp 级都是核心设计原则。



---

完整的 FA2 前向算法框架

```
输入：Q, K, V ∈ R^{N×d}（存于 HBM），块大小 Br, Bc
输出：O ∈ R^{N×d}，logsumexp L ∈ R^N（存于 HBM）
分块参数：
  Tr = ⌈N/Br⌉    # Q/O 的块数
  Tc = ⌈N/Bc⌉    # K/V 的块数
───────────────────────────────────────────────────────────────────
for i = 1 to Tr:                          # 外层：遍历 Q 块，每次迭代独立，可分配到不同 thread block

    load Q_i (Br×d) : HBM → SRAM
		# 初始化片上滚动状态
    O_i = zeros(Br, d)                    # 未归一化的累积输出
    ℓ_i = zeros(Br)                       # 累积 exp sum
    m_i = -inf · ones(Br)                 # 当前已知的行 max

    for j = 1 to Tc:                      # 内层：遍历 K/V 块

        load K_j (Bc×d), V_j (Bc×d) : HBM → SRAM
        S_ij = Q_i @ K_j^T                # (Br×d) @ (d×Bc) → (Br×Bc)，片上完成，不写 HBM

        # ── Online softmax ──
        m_i_new = max(m_i, rowmax(S_ij))                     # (Br,)，逐行取历史 max 与当前块 max 的较大值
        P̃_ij = exp(S_ij - m_i_new)                           # (Br×Bc)，pointwise
        ℓ_i = exp(m_i - m_i_new) * ℓ_i + rowsum(P̃_ij)        # (Br,)
        
        O_i = diag(exp(m_i - m_i_new)) @ O_i + P̃_ij @ V_j    # 更新累积输出（FA2 关键：O_i 保持未归一化）
        m_i = m_i_new                                        # 滚动更新 max

    end for   # 内层循环结束，K/V 全部遍历完毕

    O_i = diag(ℓ_i)^{-1} @ O_i            # (Br×d) 循环结束后统一做一次归一化
    L_i = m_i + log(ℓ_i)                  # (Br,)  保存 logsumexp 供 backward 使用
    write O_i, L_i → HBM                  # 写回 HBM（整个内层循环期间 O_i 都在片上，仅此一次写回）

end for   # 外层循环结束
return O, L
```



### Backward Pass

```
输入：Q, K, V, O, dO ∈ R^{N×d}，L ∈ R^N（均在 HBM）
输出：dQ, dK, dV ∈ R^{N×d}
─────────────────────────────────────────────────────────────────
# 预处理：计算 softmax backward 辅助量
D = rowsum(dO ⊙ O) ∈ R^N               # 逐行点积，写入 HBM

# 初始化输出梯度
dQ = zeros(N, d)
for j = 1 to Tc:                       # 外层：遍历 K/V 块

    load K_j, V_j : HBM → SRAM         # K/V tile 驻留 SRAM 整个内层循环
    dK_j = zeros(Bc, d)                # 片上累积，内层循环结束后一次写回
    dV_j = zeros(Bc, d)

    for i = 1 to Tr:                   # 内层：遍历 Q 块

        load Q_i, O_i, dO_i, L_i, D_i : HBM → SRAM
        
        [Recompute] S_ij = Q_i @ K_j^T     # (Br×Bc) 注意力分数
        [Recompute] P_ij = exp(S_ij - L_i) # (Br×Bc) 归一化概率，FA2改进：直接用 L，不需要分别存 m 和 ℓ

        dV_j += P_ij^T @ dO_i          # (Bc×d) 累加 dV

        dP_ij = dO_i @ V_j^T           # (Br×Bc)
        dS_ij = P_ij ⊙ (dP_ij - D_i)   # (Br×Bc)，softmax backward

        # 更新 dQ（需要读-改-写 HBM，跨 j 迭代累积）
        load dQ_i : HBM → SRAM
        dQ_i += dS_ij @ K_j            # (Br×d) 累加 dQ
        write dQ_i : SRAM → HBM        # atomic add 处理并发写冲突

        dK_j += dS_ij^T @ Q_i          # (Bc×d) 累加 dK

    end for
    write dK_j, dV_j : SRAM → HBM      # 内层循环结束，一次写回

end for
return dQ, dK, dV
```

需要注意，Backward 仍然是 K/V 方向在外层循环。这是因为 Backward 在 j 方向有 dK, dV 两个矩阵要累加，在 i 方向有 dQ 矩阵要累加，总有一个方向的分块累加无法常驻在片上 SRAM。“**两害相权取其轻**”，让 dK, dV 两个矩阵在 SRAM 上累加，只留下 dQ 一个矩阵用 atomic add 写入 HBM，是更优的选择。

|              | Forward              | Backward                                                  |
| ------------ | -------------------- | --------------------------------------------------------- |
| 外层循环     | Q（各块独立）        | K/V（dK/dV 片上累积）                                     |
| 内层循环     | K/V                  | Q                                                         |
| 跨迭代写冲突 | 无                   | dQ 需要 atomic add                                        |
| 片上驻留     | $Q_i$                | $K_j$, $V_j$                                              |
| 矩阵乘次数   | 2（$QK^\top$, $PV$） | 5（$QK^\top$, $PV^\top$, $dOV^\top$, $dSK$, $dS^\top Q$） |
| 保存的中间量 | L（log_sum_exp）     | 重算 S 和 P，不存                                         |





---



## [FlashAttention3](http://arxiv.org/abs/2407.08608)

不同于 FlashAttention2 在算法上的进一步优化，FlashAttention3 在算法上与 FA2 是一样的。但是 FA3 与 NVIDIA 新一代的 Hopper 架构结合更加紧密，围绕着 Hopper 引入的两个关键特性——**异步执行**和**低精度计算**，进行了 Arch-specific 的优化。

在 H100 上，FA3 的前向性能达到了 FA2 的1.5-2.0×，FP16 精度下达到了 740 TFLOPS（75% 硬件吞吐利用率），FP8 精度下接近 1.2 PFLOPS

### NVIDIA Hopper architecture

**Hopper 是 NVIDIA GPU 架构史上最具范式转换意义的一次升级之一**，重要性可能仅次于 Volta（首次引入 Tensor Core）。

为什么说 Hopper 是一次“范式转变”？因为 GPU 编程的心智模型从 Volta → Turing → Ampere 时代的**SPMD（同质化并行）+ 同步执行**，变成了 Hopper 时代的**异质化协作 + 异步执行**。

Hopper 架构的变化让整个软件栈需要重新适配：

* **CUTLASS 3.x **完全重写，引入 CuTe（基于 layout algebra 的全新抽象），第一次把"异步流水线"作为一等公民；
* **cuBLAS / cuDNN** 内核全部重新设计才能用上 Hopper 特性；
* **TensorRT** FP8 支持需要重写量化通路；
* **PyTorch** 中 FlashAttention、scaled_dot_product_attention 等核心 op 在 Hopper 上的实现是另一套代码路径；

FlashAttention-3 其实是最直接的例子——FA2 的算法在 Hopper 上跑不到 35% 利用率，必须从算法层重新设计才能拿到 75%。

#### Memory Hierarchy

| Hardware Level                      | Parallel Agent       | Data Locale | Capacity @ Bandwidth（Hopper H100 SXM5） |
| ----------------------------------- | -------------------- | ----------- | ---------------------------------------- |
| Chip                                | Grid                 | GMEM        | 80 GiB @ 3.35 TB/s                       |
| GPC (*Graphics Processing Cluster*) | Threadblock Clusters | L2          | 50 MiB @ 12 TB/s                         |
| SM (*Streaming Multi-processor*)    | Threadblock (CTA)    | SMEM        | 228 KiB per SM, 31TB/s per GPU           |
| Thread                              | Thread               | RMEM        | 256 KiB per SM                           |

SMEM 是 scratchpad 存储，硬件不会自动填充，软件必须显式 load。并且 SMEM **是高度 banked 的**（32 个 bank，4-byte stride），bank conflict 是 SMEM 性能调优的核心问题。

L2 则是软件透明的 cache。Hopper 中的 L2 容量达到了 50MB，已经可以装下很多算子的整个 working set（功能有点类似于 CPU 多核共享的 L3？）

#### Thread Hierarchy

Hopper 的层级是六层（从细到粗）：

```
Thread → Warp → Warpgroup → CTA(threadblock) → Cluster → Grid
  1      32     128         ≤2048              ≤8 CTA    全 GPU
```

CTA 内的所有 Warp 在同一 SM 上执行，通过 SMEM 来共享数据，协作完成计算。

Cluster 内的所有 CTA 在同一 GPC 上执行，每个 CTA 调度到一个 SM，通过 GPC 上的 SM-to-SM network，实现跨 CTA 共享和协作。

**Hopper 架构引入了 Warpgroup 和 Cluster 这两层**，

* Warpgroup 的出现是因为 **WGMMA 指令需要 4 个 warp 协同执行，集体提供操作数和接收输出**，这 4 个 warp 构成一个连续的 warpgroup；
* Cluster 的出现是因为 Hopper 在硬件上实现了 SM-to-SM network，让 GPC 内的多个 SM 之间可以互相访问对方的 SMEM。Cluster 把跨 CTA 的协作能力（SM-to-SM network、multicast TMA）暴露给编程模型，让算法能跨 CTA 协作而不付出 GMEM 代价。

#### 核心：三大升级

##### 异步执行

**核心思想**：把数据搬运和 MMA 计算从 CUDA cores 上"剥离"出去，用专用硬件单元承担，让多条 pipe 并行工作。

两个核心硬件单元：

* **TMA (Tensor Memory Accelerator)**
  * Per SM 的专用 DMA 引擎，负责 GMEM ↔ SMEM 的多维 tensor tile 拷贝；
  * 只需要一个 thread 发起 TMA 指令 (`cp.async.shared.global`)，整个 tile 的拷贝完全交给硬件**异步**执行，这个 thread 不需要阻塞，可以执行后面的指令；
  * 通过 tensor descriptor（`CUtensorMap`）抽象多维地址，硬件自动算 offset；
  * 通过**硬件 mbarrier** 完成同步，使用数据时通过 `mbarrier.test_wait`/`mbarrier.try_wait`  查询完成状态；
  * 还支持 multicast（一次 load 分发到 cluster 内多个 SM）
* **异步 WGMMA (Warp Group MMA)**
  * Warpgroup 粒度的 Tensor Core 指令（`wgmma.mma_async`），由 4 个 warp 协同发射；
  * Hopper 的 Tensor Core 第一次变成**异步**——发射指令后立即返回，warp 继续执行后续代码，结果稍后才到累加器。
  * 操作数可以由 RMEM 提供，也可以直接来自 SMEM；输出还是累加到 RMEM；
  * 通过 `wgmma.fence`/`wgmma.commit_group`/`wgmma.wait_group` 显式管理数据依赖；

> TMA 和 WGMMA 的“异步”，本质上是将**同步责任从硬件转移到软件**，**让 warp 内的指令调度不再被 SM 硬件的顺序发射限制。**
>
> * Ampere 中的 mma 指令 `mma.sync` 是同步的：
>
>   ```
>   mma.sync D, A, B, C;    ← 发射，warp 卡住
>   add R0, D, R1;          ← 必须等 D 写回才能执行
>   ```
>
>   `mma.sync` 之后，后面任何**依赖 D 的指令**都会被硬件 scoreboard 阻塞，直到 D 真正写回寄存器。此时 warp scheduler 会把这个 warp 标记为 stall，切到别的 warp。（并行性只能靠 **warp 切换**）
>
> * Hopper 中的 mma 指令 `wgmma.mma_async` 是异步的：
>
>   ```
>   wgmma.mma_async D, A, B, C;    ← 发射，warp 立即前进
>   add R0, R10, R20;              ← 立即执行（不依赖 D）
>   fmul R1, R11, R21;             ← 立即执行
>   ...
>   wgmma.wait_group 0;            ← 这里才显式等
>   add R2, D, R3;                 ← 现在 D 可用
>   ```
>
>   **D 寄存器在硬件上是 in-flight 状态**——硬件不再用 scoreboard 自动追踪 D 的依赖。
>
>   使用结果前需要显式同步，如果在 wait 之前读 D，行为未定义——同步责任从硬件转移到软件。
>
>   这样单个 warp 自己就能驱动多个 pipe 同时工作，不需要切换。

##### Warp-specialization

异步硬件单元有了，但要真正榨干它们，需要编程模型层面的配合。**Hopper 引入了几个机制让 warp-specialization 真正可行：**

* `setmaxnreg`指令，能够进行动态寄存器重分配

  这打破了 Ampere 时代"所有 warp 必须用同样多寄存器"的约束。在 producer-consumer 模式下：

  - Producer warpgroup（只发 TMA）：调用 `setmaxnreg.dec`，用极少寄存器
  - Consumer warpgroup（做 WGMMA + softmax）：调用 `setmaxnreg.inc`，拿到 producer 让出来的寄存器，能放更大的累加器和 pipeline buffer

* 硬件 mbarrier：一个 64-bit SMEM 状态机

  支持跨 thread / warpgroup / CTA / cluster 同步，TMA 完成自动 arrive。这套机制让 producer-consumer 之间的同步**几乎零开销**，不需要软件维护 spinlock。

> **从 Ampere 上的"同质化 warp + scheduler" 到 Hopper 上的 "Warp-specialization"，为什么？**
>
> 一个“表层”的原因也许是 Warp-specialization 让 Warp 资源的分配更加灵活，比如寄存器数量，可以根据不同的负载动态分配；
>
> 深层的问题在于**编译器优化空间**。这一点 NVIDIA 在 CudaDMA[Bauer et al. 2011] 那篇老论文里就讲过：当一个 warp 既要做 load 又要做 compute 时，编译器很难做出最优的指令调度；但如果显式划分 producer/consumer 角色，每个 warp 的指令流变得简单且单一职责，编译器能生成更优的 schedule，硬件 scheduler 也更容易让独立 pipe 真正并行。



##### 低精度数据格式

Hopper 的 WGMMA 指令支持 FP8 Tensor Core，相比 FP16/BF16 提供 **2× 单 SM 吞吐**。但用好 FP8 WGMMA 必须理解其**操作数 layout 约束**。

| 精度          | SMEM 中支持的 layout    |
| ------------- | ----------------------- |
| FP16 WGMMA    | mn-major **或** k-major |
| **FP8 WGMMA** | **仅 k-major**          |

在 attention 这类**两个连续 GEMM 融合在单 kernel** 的场景下（比如 $S = QK^\top$ 之后接 $O = PV$），第一个 WGMMA 的 **FP32 累加器**（即作为下一步 P 矩阵）的寄存器布局，与第二个 WGMMA 对 **FP8 操作数 A** 的布局期望**不一致**。这正是 FA3 要处理的核心工程问题——通过 **byte permute + in-kernel transpose** 来桥接两个 WGMMA 之间的 layout 不匹配。



---



### Forward Pass

要理解 FA3 的 Forward Pass，核心其实就是两点：**warp-specialization** 和 **software pipelining**。

实际上 FA3 底层算法和 FA2 是一样的，但是 FA3 通过这两个手段来实现**显式的 hardware pipe overlap**：

* warp-specialization 让 TMA pipe 与其他计算单元的 pipe overlap；
* software pipelining 则让 MMA pipe 与 non-matmul 运算（softmax and friends，涉及到 MUFU pipe, FP32 pipe 等）overlap.

#### warp-specialization

CTA 内 warpgroup 分为 **Producer warpgroup** 和 **Consumer warpgroup** 两类角色：

* **Producer warpgroup **只负责发起 TMA load 指令，不参与计算；
* **Consumer warpgroup** 只做 WGMMA 和 softmax，不参与 load。

Producer 与 Consumer 之间通过 s-stage 的循环 SMEM 缓冲来配合。

<img src="/images/flash-attention/绘图 1-7479842.png" alt="绘图 1" style="zoom:25%;" />

* Producer 可以提前几个 iteration 发起对该轮 $K_j$/$V_j$ 的 TMA load，TMA 异步完成后会自动设置对应的 mbarrier 通知 Consumer 数据就绪，数据存放在循环 SMEM 缓冲中。

* Consumer 等待该轮的$K_j$/$V_j$ 就绪后，发起 WGMMA，WGMMA 异步完成后，Consumer 也设置对应的 mbarrier 通知 Producer 缓冲区已释放。

* 这就是最经典的 "async_load + mma" pattern。stage-s 的深度取决于 TMA 的延迟，假设 $T_{\text{TMA}}$ 是单次 TMA load 的端到端延迟，$T_{\text{consumer\_iter}}$是consumer iteration 的吞吐时间，
  $$
  s \geq \left\lceil \frac{T_{\text{TMA}}}{T_{\text{consumer\_iter}}} \right\rceil
  $$



另外，Producer thread 的寄存器需求非常低，可以降低 Producer thread 的寄存器数量，增加 Consumer thread 的寄存器数量，实现 **CTA 内寄存器重分配**，缓解 Consumer 的寄存器压力，放下更大的 S/P tile。



#### software pipelining

Consumer warpgroup 的负载相比 Producer warpgroup 要复杂很多，GEMM 跑在 MMA pipe 上，softmax and friends 跑在 MUFU pipe + FP32 pipe + ALU pipe 上。而且更重要的是，由于 MUFU/FP32 的吞吐远低于 MMA，softmax and friends 的计算时间和 GEMM 的时间是相同量级的，也就是说，Consumer pipe 不能只考虑 bound on MMA。

因为 GEMM 和 softmax and friends 跑在不同的硬件 pipe 上，FA3 在 Consumer 的循环中引入了软件流水，让它们可以 overlap 起来。**软件流水又可以通过两种形式实现**：**Spatial Pineline** 和 **Temporal Pineline**，分别在空间和时间上进行进行 overlap。

首先我们将 Consumer 的一次 iteration 分为 3 段：`GEMM0`($QK^\top$), `softmax&friends`($P=\text{softmax}(S)$), `GEMM1`($PV$)。

1. **Spatial Pineline**

   FA3 论文中把它称为 **Pingpong scheduling**，实际上就是同时有两个 Consumer warpgroup 在运行。让 WG1 的 `GEMM1[j]`+`GEMM0[j+1]` 与 WG2 的 `softmax[j]` 并行执行，WG1 和 WG2 通过  `bar.sync` 指令同步流水线阶段。

   ![image-20260504121813300](/images/flash-attention/image-20260504121813300.png)

   Spatial Pineline 的好处是 WG1 和 WG2 可以共享 SMEM 中的 K/V tile，从而减小 L2 的流量；坏处是寄存器中需要一直保持两个 GEMM1 的输出累加 O tile，同时还有一个 S tile 和一个 P tile，寄存器压力很大。

   当然上面的图是理想化的，实际上 GEMM0 + GEMM1 总时长 ≠ softmax 时长。

2. **Temporal Pineline**

   FA3 论文中把它称为“Intra-warpgroup overlapping GEMMs and softmax”，在循环中做了 **2-stage 流水，让 GEMM1[j] 与 softmax[j+1] 的计算 overlap**。

   <img src="/images/flash-attention/image-20260504125840671.png" alt="image-20260504125840671" style="zoom:37%;" />

   Temporal Pineline 在寄存器中只需要保持一个 GEMM1 的输出累加 O tile。不过代价是 `GEMM1[j-1]` → `correction[j]` → `GEMM1[j]`  的 RAW 依赖导致这三步需要串行执行，`correction[j]` 的延迟被暴露。

   上面的图也是理想化的，实际上 softmax 时间受到 MUFU 吞吐的限制会更长，而 correction 的时间会更短。

   论文中还设计了一个 3-stage 版本的流水线，把 `GEMM1[j-1]` 和 `GEMM0[j+1]` 共同藏在 `softmax[j]` 的 shadow 中。理论上这样可以让 MMA pipe 全程不空闲。

   <img src="/images/flash-attention/image-20260504154913658.png" alt="image-20260504154913658" style="zoom:37%;" />

   但是实际上 NVCC/ptxas 没生成预期的双 WGMMA 重叠调度，会退化为 2-stage 的效果。

#### miscellaneous

对比 FA2，**其中还有一些硬件原语的升级：**

* 数据搬运升级为 TMA，不需要每个 thread 自己算地址，Producer 的寄存器需求很低。

* WGMMA 的操作数可以直接从 SMEM 读，省掉 SMEM↔RMEM 的中转。

* 同步原语升级为 mbarrier，带来更精细的同步控制 + 更低的同步开销 + 与 TMA 的天然集成。

  > **FA2 的同步原语**：
  >
  > - `cp.async.commit_group` + `cp.async.wait_group N`
  > - `__syncthreads()`（CTA 级 barrier）
  > - `__syncwarp()`（warp 级 barrier）
  >
  > **FA3 的同步原语**：
  >
  > - `mbarrier.init` / `mbarrier.arrive` / `mbarrier.wait` / `mbarrier.expect_tx`（producer-consumer 跨 warpgroup 同步、TMA 完成通知）
  > - `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group`（wgmma 异步依赖管理）
  > - 仍可用 `bar.sync` named barrier（在 pingpong scheduling 中使用）



单个 CTA 视角的下的 Forward Pass

```
输入：Q_i ∈ R^{Br×d}, K, V ∈ R^{N×d}（存于 HBM），块大小 Br, Bc
输出：O_i ∈ R^{Br×d}，logsumexp L ∈ R^Br（存于 HBM）
分块参数：
  Tr = ⌈N/Br⌉    # Q/O 的块数
  Tc = ⌈N/Bc⌉    # K/V 的块数
───────────────────────────────────────────────────────────────────
初始化 pipeline 对象：
		- 管理 s-stage circular buffer (smem_K[s], smem_V[s])
   	- 为 K, V 的每个 stage 配 mbarrier (bar_K_full[s], bar_V_full[s], bar_K_empty[s], bar_V_empty[s])
   	- 为 Q 配单独的 mbarrier (bar_Q)

if in producer warpgroup:
		deallocate predetermined number of registers
		issue TMA load Q_i (HBM → SMEM)
		for j = 0 to Tc - 1:
				wait for (j%s)_th stage of the buffer to be consumed
				issue TMA load K_j, V_j (HBM → smem_K/smem_V)
		end for
		
if in consumer warpgroup:
		reallocate predetermined number of registers
		On-chip, initialize O_i = zeros(Br, d), ℓ_i = zeros(Br), m_i = -inf · ones(Br)
		
		# Prologue
		wait for Q_i to be loaded in SMEM
		wait for K_0 to be loaded in smem_K
		issue WGMMA0: S_i0 = Q_i @ K_0^T
		wait for WGMMA0 to finish
		compute P_i0 = softmax(S_i0), update m_i/ℓ_i
		
		# Main Loop(2-stage sw pipeline)
		for j = 1 to Tc - 1:
				wait for K_j to be loaded in smem_K
				issue WGMMA0: S_ij = Q_i @ K_j^T (SS-GEMM)
				
				wait for V_{j-1} to be loaded in smem_V
				issue WGMMA1: O_i += P_{i,j-1} @ V_{j-1} (RS-GEMM)
				
				wait for WGMMA0 to finish
				compute P_ij = softmax(S_ij), update m_i/ℓ_i
				wait for WGMMA1 to finish
				rescale O_i
				
				release (j%s)_th stage of the buffer for the producer
		end for
		
		# epilogue
		wait for V_{Tc-1} to be loaded in smem_V
		issue WGMMA1: O_i += P_{i, Tc-1} @ V_{Tc-1} (RS-GEMM)
		wait for WGMMA1 to finish
		
		O_i = diag(ℓ_i)^{-1} @ O_i  # O_i 归一化
    L_i = m_i + log(ℓ_i)        # m_i 和 ℓ_i 融合为 LSE
    write O_i, L_i → HBM
```



我们将硬件原语细化到 PTX 指令，Producer 和 Consumer 的 Forward Pass 分别长这样：

```
========== Producer Warpgroup ==========
setmaxnreg.dec...  # 降低最大寄存器数量

# leader thread 发起 Q_i 的 TMA load
mbarrier.expect_tx [bar_Q], Q_BYTES
cp.async.bulk.tensor.2d ... [smem_Q], [tmap_Q, i], [bar_Q]    # issue TMA load Q_i

# --- 主循环 ---
phase = 0
for j = 0 to Tc - 1:
		s_idx = j % s
		mbarrier.try_wait.parity [bar_K_empty[s_idx], phase                       # 等 buffer slot 被 consumer 用完
    mbarrier.expect_tx [bar_K_full[s_idx]], K_BYTES
    cp.async.bulk.tensor.2d... [smem_K[s_idx]], [tmap_K, j], [bar_K_full[s_idx]]  # issue TMA load K_j
    
    mbarrier.try_wait.parity [bar_V_empty[s_idx], phase                           # 等 buffer slot 被 consumer 用完
    mbarrier.expect_tx [bar_V_full[s_idx]], V_BYTES
    cp.async.bulk.tensor.2d... [smem_V[s_idx]], [tmap_V, j], [bar_V_full[s_idx]]  # issue TMA load V_j

   	if ((j+1) % s == 0): phase ^= 1
```

```
========== Consumer Warpgroup ==========
setmaxnreg.inc...  # 申请更多寄存器

# --- 初始化累加值 ---
O_i = 0 ∈ R^(Br × d)
l_i = 0, m_i = -∞ ∈ R^Br

# --- Prologue ---
phase_K = 0, phase_V = 0
mbarrier.try_wait.parity [bar_Q], 0                 # wait for Q_i to be loaded in SMEM
mbarrier.try_wait.parity [bar_K_full[0]], phase_K   # wait for K_0 to be loaded in smem_K

wgmma.fence
S_cur = Q_i × K_0^T  (SS-GEMM)   # issue WGMMA0[0]
wgmma.commit_group
wgmma.wait_group 0               # commit and wait for WGMMA0[0] to finish
mbarrier.arrive [bar_K_empty[0]] # release smem_K[0]

m_i_old = m_i                    # compute online_softmax[0]
m_i = max(m_i_old, rowmax(S_cur))
P_cur = exp(S_cur - m_i)
l_i = exp(m_i_old - m_i) * l_i + rowsum(P_cur)

# --- 主循环：j 从 1 到 T_c - 1，2-stage pipeline ---
for j = 1 to T_c - 1:
		s_cur = j % s
    s_prev = (j-1) % s
    
    mbarrier.try_wait.parity [bar_K_full[s_cur]], phase_K    # wait for K_j to be loaded in smem_K
    wgmma.fence
    S_next = Q_i × K_j^T   (SS-GEMM)                         # issue WGMMA0[j]
    wgmma.commit_group                                       # commit but not wait
    
    mbarrier.try_wait.parity [bar_V_full[s_prev]], phase_V   # wait for V_{j-1} to be loaded in smem_V
    wgmma.fence
    O_i += P_cur × V_{j-1}  (RS-GEMM, A 来自寄存器)            # issue WGMMA1[j-1]
    wgmma.commit_group                                       # commit but not wait
    
    wgmma.wait_group 1     (允许 1 个 group 还在 flight)       # 等待 WGMMA0[j] 完成（WGMMA1[j-1] 仍在运行）
    
    m_i_old = m_i                                            # compute softmax[j] (与 WGMMA1[j-1] overlap)
		m_i = max(m_i_old, rowmax(S_next))
   	P_next = exp(S_next - m_i)
   	l_i = exp(m_i_old - m_i) × l_i + rowsum(P_next)
   	
   	wgmma.wait_group 0     (允许 0 个 group 还在 flight)       # 等待 WGMMA1[j-1] 完成
   	O_i *= diag(exp(m_i_old - m_i))                          # rescale O_i
   	
   	mbarrier.arrive [bar_K_empty[s_cur]]                     # release smem_K[j]
    mbarrier.arrive [bar_V_empty[s_prev]]                    # release smem_V[j-1]
    
    P_cur = P_next, S_cur = S_next
    if ((j+1) % s == 0): phase_K ^= 1, phase_V ^= 1
    
# --- Epilogue ---
mbarrier.try_wait.parity [bar_V_full[(Tc-1) % s]], phase_V   # wait for V_{Tc-1} to be loaded in smem_V
wgmma.fence
O_i += P_cur × V_{Tc-1}   (RS-GEMM)                          # issue WGMMA1[Tc-1]
wgmma.commit_group
wgmma.wait_group 0                                           # commmit and wait

mbarrier.arrive [bar_V_empty[(Tc-1) % s]]

# (Br×d) 循环结束后统一做一次归一化
O_i = diag(l_i)^(-1) × O_i
L_i = m_i + log(l_i)
write O_i, L_i → HBM
```



---

### FP8 支持

FP8 Attention 中的精度分是这样的：只有 Q, K, V 输入和 O 输出是 FP8，中间数值敏感的计算（累加、softmax、scale）都是 FP32。

```
Q_i, K_j (FP8) ──┐
              WGMMA0 ─→ S (FP32 累加器)
                        │
                        × s_q × s_k  (反量化还原量级，FP32)
                        │
                        online softmax (FP32)
                        │
                        量化 + cvt (FP32 → FP8) + byte permute
                        │
                        P (FP8) ──┐
V_j (FP8) ───→ transpose ─────  WGMMA1 ──→ O (FP32 累加器)
                                           │
                                           × s_p × s_v （反量化还原量级，FP32）
                                           │
                                           rescale by ℓ
                                           │
                                           O (FP8) → HBM
```

#### layout transformations

FP8 Attention 麻烦的地方在于需要进行 **layout transformation**，这是由于 WGMMA 指令对于 FP8 数据的 layout 要求。具体来说，有两个矩阵需要进行  **layout transformation**：

* Q, K, V 在 head dimension 连续，而 FP8 WGMMA 要求 SMEM 中的操作数是 k-major layout。$Q_i$ 和 $K_j^\top$ 原始 layout 已经是 k-major 了，但是 $V_j$ 需要转置一下才是 k-major。FA3 将转置操作融合在 kernel 中（**in-kernel transpose**），使用 **LDSM** 和 **STSM** 指令，**LDSM** 将 $V_j$ 从 SMEM copy 到寄存器，**STSM** 再将再 $V_j$  copy 回 SMEM，在此过程中完成转置操作。

* 我们知道 Attention 的两条 MMA 是 back-to-back 的（GEMM0 的输出累加矩阵 $S_{ij}$ 经过标量运算得到 GEMM1 的输入 A 矩阵 $P_{ij}$）。这两个矩阵不仅精度需要转换（FP32 精度到 FP8 精度），而且麻烦的是 FP8 WGMMA 规定的 operand A layout 与 accumulator layout 也不一样（具体可以参考 PTX ISA）。FA3 通过 byte permute 指令（`prmt.b32`）对寄存器数据进行重排，以很小的开销完成了 $S_{ij}$ layout 到 $P_{ij}$ layout 的转换。

  > 经过 byte permute 之后的 $P$ 其实还有一个问题，它的**列位置被置换**了（比如原 col 0,1,8,9 现在被当作 col 0,1,2,3 看待）。
  > 那 $PV$ 的结果会错——除非 **$V$ 的行也做对应的置换**。
  >
  > 这里用到了**矩阵乘的一个性质**：
  > $$
  > (P \cdot V)_{ij} = \sum_k P_{ik} \cdot V_{kj}
  > $$
  > 如果 $P$ 的列从 [0,1,2,...,N-1] 重排为 [σ(0), σ(1), ..., σ(N-1)]，那么 $V$ 的行也按同样的 σ 重排，乘积就保持不变：
  > $$
  > \sum_k P_{i, \sigma(k)} \cdot V_{\sigma(k), j} = \sum_k P_{ik} \cdot V_{kj}
  > $$
  > **所以其实在 transpose $V_j$ 时，还会进行 $V_j$ 行的 shuffle 来配合 $P_{ij}$ 列的 shuffle.** 这一块具体细节可以看论文。



FA3 通过 **byte permute + in-kernel transpose** 来高效桥接两个 WGMMA 之间的 layout 不匹配。

#### Accuracy

除此之外，在数值精度上，FP8 (e4m3) 只有 3 bit 尾数 + 4 bit 指数，极容易上溢/下溢，需要配合 scaling  factor 来正确表示数值。

```
原始 FP32/FP16 张量 X
        ├── 计算 scale s（可能多个，对应 per-tensor / per-block / per-row）
        ▼
量化后的 FP8 张量 Q = round(X / s) ← 存储/传输/计算用这个
        ├── 矩阵乘
        ▼
反量化回高精度: X' = Q × s
```

由于 LLM 中存在 outlier feature/activation（少数元素的幅值是其他元素的几十甚至几百倍），per-tensor scaling 误差会很大。FA3 采用两个优化：

- **Block quantization**：每个 `Br`×`d` 或 `Bc`×`d` 的 block 单独存一个 scale。因为 FA 本来就按 block 操作，scale 直接乘到 $S$ 上几乎零开销。
- **Incoherent processing**：在量化前用一个**随机正交矩阵** $M$ 乘 $Q$ 和 $K$。因为 $(QM)(KM)^\top = QK^\top$，attention 输出不变，但 outlier 被"摊薄"成多个值的随机和。$M$ 取 Hadamard × 随机 ±1 对角阵的形式，$O(d \log d)$ 复杂度。



---

### Backward Pass

FA3 的 Backward 继承了 Forward 的 Warp specialization 和 GEMM-softmax pipelining 思想，具体优化思路如下：

1. 把 CTA 内 warp 分成 **3 类**，**用一个独立 warp 异步处理 atomic，让 consumer 的 GEMM pipeline 不被打断。**
   * Producer warpgroup：发 TMA load ($K_j$, $V_j$, $Q_i$, $dO_i$)
   * Consumer warpgroups：做 GEMM 和 softmax 反向
   * dQ-writer warp：专门把 $dQ_i$ atomic add 到 GMEM
2. **GEMM-softmax pipelining**：和 forward 一样可以做 intra-warpgroup 流水
3. **预处理 kernel**：FA2 延续下来的优化，提前计算 $D = \text{rowsum}(dO ⊙ O)$ ，避免 mainloop 中重复计算

Backward 关键流程：

```
预处理 kernel: D = rowsum(dO ⊙ O)，存 GMEM       ← 单独的 kernel

Backward main kernel (每 CTA 处理一个 K_j, V_j):

Producer:
    TMA load K_j, V_j（外层 once）
    循环 i: TMA load Q_i, dO_i 到 s-stage circular buffer

Consumer (mainloop, 每个 i):
    1. S_ij = Q_i @ K_j^T               (SS-GEMM)
    2. dP_ij = dO_i @ V_j^T             (SS-GEMM)
    3. P_ij = exp(S_ij - L_i)           ← 用前向存的 L 重算 P
    4. dS_ij = P_ij ⊙ (dP_ij - D_i)     ← 用预算的 D
    5. dV_j += P_ij^T @ dO_i            (RS-GEMM, 累加到本 CTA 的 dV_j)
    6. dK_j += dS_ij^T @ Q_i            (RS-GEMM, 累加到本 CTA 的 dK_j)
    7. dQ_local = dS_ij @ K_j           (SS-GEMM)
    8. 写 dQ_local 到 SMEM, 通知 dQ-writer
    
dQ-writer (并行):
    从 SMEM 读 dQ_local
    用 semaphore 做 atomic add 到 GMEM 的 dQ_i

Epilogue: 把累加好的 dV_j, dK_j 写回 HBM
```





---



## [FlashAttention4](https://arxiv.org/abs/2507.05326)

FlashAttention4 同样延续了 FlashAttention3 的硬件感知视角，针对 NVIDIA 新一代 Blackwell 架构的**非对称硬件扩展**（Tensor Core 翻倍而 SMEM 带宽和 MUFU 几乎不变，导致 bottleneck 转移），**重新优化设计了流水线**。同时在算法层面上引入**软件模拟的指数运算**和 **conditional softmax rescaling**，以缓解 MUFU 吞吐瓶颈，以及减少 non-matmul 操作。

在 B200 上，FA4 前向算子的性能是 cuDNN 9.13 的 1.1-1.3×（由于 Blackwell 对 wgmma 指令没有向前兼容，FA3 在 B200 上跑不起来，所以无法和前代比较）。

### NVIDIA Blackwell architecture

**Blackwell 是 Hopper 范式的深化与补全。**

如果说 Hopper 的第四代 Tensor Core 是"**部分异步**"，Blackwell 上的第五代 Tensor Core 就是更进一步做到了**“完全异步”**。Blackwell 架构把"**warp-specialized 异步执行**"的范式补完了。

与 FA3 一样，FA4 的出发点同样还是 Blackwell 的架构变化。

#### Tensor Memory

Blackwell 最大的架构变化就是引入了 **tensor momory（TMEM）**。TMEM 可以理解为每个 tensor core 私有的 scratchpad，用于存放 MMA 指令的输入和累加输出，专门服务于 MMA pipe。

同样都是 SM 的片上 SRAM，我们可以对比一下 RF/TMEM/SMEM 的定位：

| 维度                  | RF（Registers）                           | TMEM（Tensor Memory）                                        | SMEM（Shared Memory）                          |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| 容量                  | 256 KB/SM (64K × 32bit regs)              | 256 KB/SM                                                    | 228 KB/SM                                      |
| 访问粒度              | per-thread，线程私有                      | **warp，lane-to-row 映射**                                   | CTA 内任意线程任意地址                         |
| 访问延迟              | ~1 cycle                                  | ~similar to SMEM                                             | ~20-30 cycle                                   |
| 谁能写                | ALU/LSU/任何指令                          | Tensor Core 指令<br />- **`tegen5.mma`的累加输出**<br />- `tcgen05.st` TMEM ↔ 寄存器<br />- `tcgen5.cp` TMEM ↔ SMEM<br />- ... | LSU、TMA、`stmatrix` 等                        |
| 谁能读                | 所有指令                                  | Tensor Core 指令<br />- **`tegen5.mma`的 operand A 输入**<br />- `tcgen05.ld ` TMEM ↔ 寄存器<br />- ... | LSU、TMA、**`tegen5.mma` 的 operand A/B 输入** |
| 与 Tensor Core 的关系 | Hopper 上存放 MMA operand A & accumulator | **Blackwell 上存放 MMA operand A & accumulator**             | **MMA 的 operand A/B 的来源**                  |

---

为什么 Blackwell 要添加 TMEM 这一块 SRAM？

1. 最直接的一点，**缓解 Register Pressure**。Tensor Core 的 A/D 矩阵直接从 TMEM 读写，不再需要分配寄存器。这样终于缓解了 Hopper 时代 FA3 内核极高的寄存器压力，也可以支持更大的 tile 尺寸。

2. **更友好的 back-to-back GEMM 支持**

   * FA3 中 GEMM0 是 SS 模式，GEMM1 是 RS 模式，中间结果 P 分布在不同线程的寄存器上，需要进行 in-kernel layout 转换；
   * FA4 中 GEMM0 是 SS 模式，GEMM1 是 TS 模式，中间结果 P 放在 TMEM 中，**accumulator 和 operand A 的 TMEM layout 是兼容的**，不需要额外的 layout transformation。

3. **MMA 累加值可以跨 thread 共享，不再与 SIMT 寄存器绑定。**这是 Blackwell 的 tensor core **做到完全异步的关键**，让 MMA 的发射者和消费者完全解耦，通过 TMEM 来协作。这样其实形成了两层 Producer-Consumer 模型：

   * TMA warpgroup 搬运数据，MMA warpgroup 使用数据计算 GEMM，数据在 SMEM 上传递；
   * MMA warpgroup 计算 GEMM，softmax warpgroup 使用 GEMM 结果，数据在 TMEM 上传递。

   正是这个特性，FA4 可以把 GEMM 和 softmax 拆分到不同的 warpgroup。

4. **MMA 只需要单线程发射。**Blackwell 上 MMA 和 TMA 一样是单线程发射，不需要 warpgroup 协同发射。

---

**TMEM 的物理组织形式是一个 128 rows × 512 cols 的二维阵列**，每个 cell 是 32 bit。

* 总容量 = 128 × 512 × 4 B = 256 KB
* 128 rows 对应一个 warpgroup 的 128 个线程。一个 warp 对应 TMEM 的 32 行，**thread i 只能访问第 (warp_id × 32 + i) 行，不能跨行访问**。这就是 TMEM 的 **lane-to-row 映射**，是硬件约束。
* 512 cols，每列 32 bit
* 分配粒度：**32 列一组**（即 128 × 32 × 4 B = **16 KB 一个 "sub-partition" / "column group"**），所以一个 TMEM 可以分成 16 个 16 KB 的块管理，用 `tcgen05.alloc` / `tcgen05.dealloc` 显式管理。

---

> **第五代 Tensor Core 把 GPU 的执行模型从"warpgroup 协同操作 RF" 推向 "SM 内多种异步引擎（TC、TMA、MUFU、CpAsync）通过 TMEM/SMEM 解耦协作"**。



#### 非对称硬件扩展

从 Ampere 到 Hopper，再到 Blackwell，硬件资源扩展的一个关键趋势是：**tensor core 的吞吐量扩展速度快于其他功能单元。**

| 单元                  | Ampere(A100) | Hopper(H100) | Blackwell(B200) | A→H      | H→B      |
| --------------------- | ------------ | ------------ | --------------- | -------- | -------- |
| FP16/BF16 Tensor Core | 312 TFLOPS   | 989 TFLOPS   | 2250 TFLOPS     | **3.2×** | **2.3×** |
| FP8 Tensor Core       | 不支持       | 1979 TFLOPS  | 4500 TFLOPS     | 新增     | 2.3×     |
| FP4 Tensor Core       | 不支持       | 不支持       | 9000 TFLOPS     | —        | 新增     |
| HBM 带宽              | 2 TB/s       | 3.35 TB/s    | 8 TB/s          | 1.7×     | 2.4×     |
| SMEM 带宽/SM          | 128 B/clock  | 128 B/clock  | 128 B/clock     | 1×       | 1×       |
| MUFU exp/SM           | 16 ops/clock | 16 ops/clock | 16 ops/clock    | 1×       | 1×       |
| RF/SM                 | 256 KB       | 256 KB       | 256 KB          | 1×       | 1×       |

下面这个是三代架构 Tensor Core（MMA指令）的变化，每一代架构上 Tensor Core 的尺寸都在成倍增加，支持的精度也越来越多，并且逐渐从同步变成异步。

| 架构      | MMA 指令          | 发射                       | 输入        | 输出 | 同步性                                            |
| --------- | ----------------- | -------------------------- | ----------- | ---- | ------------------------------------------------- |
| Ampere    | `mma.sync`        | warp 协同（32 线程）       | RF          | RF   | **同步**（指令完成才能继续）                      |
| Hopper    | `wgmma.mma_async` | warpgroup 协同（128 线程） | RF / SMEM   | RF   | **半异步**（异步发射，但输出在 RF 强迫一定耦合）  |
| Blackwell | `tcgen05.mma`     | **单线程**                 | SMEM / TMEM | TMEM | **全异步**（输出脱离 RF，发射者和消费者完全解耦） |

| 架构      | Tensor Core 吞吐(ops/clock) | MMA 尺寸 (M×N×K)                                             | 数据精度支持                                                 |
| --------- | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ampere    | FP16/BF16: 512              | FP16/BF16: `16×8×16`                                         | FP64, TF32, FP16, BF16, INT8, INT4                           |
| Hopper    | FP16/BF16: 4096             | FP16/BF16: `64×N×16`，`N ∈ {8, 16, ..., 256}`<br />FP8: `64×N×32` | FP64, TF32, FP16, BF16, **FP8 (E4M3/E5M2)**, INT8            |
| Blackwell | FP16/BF16: 8192             | FP16/BF16: `128×N×16`，`N ∈ {16, 32, ..., 256}`<br />FP8: `128×N×32`<br />**FP4: `128×N×64`**<br />（2-CTA 模式下 M 可达 256） | FP64, TF32, FP16, BF16, FP8, **FP6 (E2M3/E3M2)**, **FP4 (E2M1)**, **MXFP8/MXFP6/MXFP4 (microscaling)** |

论文对 Blackwell 架构（B200）进行了 roofline 分析，表明 **Attention 的性能瓶颈从矩阵乘法本身转移到了 shared memory 访存流量，以及 softmax 这类非矩阵乘操作上。**具体过程如下。



> 假设沿着 Q 和 K 的 seq_len 维度划分出的 tile 形状为 `M × N`，head dimension 为 `d`。每条 MMA 指令作用在大小为 `128 × 128` 的 tile 上，MMA 输入精度为 BF16/FP16，每个元素占 2 字节。
>
> 
>
> 前向传播的一次迭代，执行两次矩阵乘加（MMA）操作：
>
> * `QK^T`：由 `M × d` 和 `d × N` 的输入计算得到 `M × N` 的输出；
> * `PV`：由 `M × N` 和 `N × d` 的输入计算得到 `M × d` 的输出。
>
> 每次 MMA 需要 `2MNd` 次浮点操作。tensor core 吞吐量为每周期 `8192 FLOPs`（这个数值可以从 B200 理论最大 FLOPS 推导得到：`2.25 PFLOPS / 1850 MHz clock speed / 148 SMs = 8192 ops / clock / SM`），总计算时间为：
> $$
> T_{\mathrm{MMA}} = \frac{4MNd}{8192}\ \text{cycles}
> $$
> GEMM0 是 shared-shared（SS）模式，每条 MMA 会从 SMEM 读取 `128 × d` 的 Q tille 和 `d × 128` 的 K tile，计算 `M × N` 的输出总共需要 $\lceil M/128 \rceil \times \lceil N/128 \rceil$ 条 MMA 指令。所以 GEMM0 的 SMEM 读取量为：
> $$
> \lceil M/128 \rceil \times \lceil N/128 \rceil \times (128d + 128d) \times 2\text{ B}
> $$
> GEMM1 是 tensor-shared（TS）模式，每条 MMA 都会从 SMEM 中读取 `N × 128` 的 V tile，计算 `M × d` 的输出需要 $\lceil M/128 \rceil \times \lceil d/128 \rceil$ 条 MMA 指令。所以 GEMM1 的 SMEM 读取量为：
> $$
> \lceil M/128 \rceil \times \lceil d/128 \rceil \times 128N \times 2 \text{ B}
> $$
> SMEM 带宽为每周期 128 字节，则 SMEM 读取时间为：
> $$
> T_{\mathrm{smem}}
> =
> \frac{
> 2\lceil M/128 \rceil \lceil N/128 \rceil 256d
> +
> 2\lceil M/128 \rceil \lceil d/128 \rceil 128N
> }{128}
> $$
> 假设 `M`、`N`、`d` 都是 128 的倍数，可化简为：
> $$
> T_{\mathrm{smem}}
> =
> \frac{3MNd}{8192}
> \ \text{cycles}
> $$
> 前向传播还需要对 `M × N` 个值执行指数运算，对应 attention 矩阵 `S`。Exp 单元吞吐量为每周期 16 次，其所需时间为：
> $$
> T_{\mathrm{exp}} = \frac{MN}{16}\ \text{cycles}
> $$



经过计算，在不同配置下，三种硬件 pipe 需要的时间：

| 资源             | `M = N = d = 128` | `M=256, N=d=128` |
| ---------------- | ----------------- | ---------------- |
| MMA compute      | **1024**          | **2048**         |
| Shared memory    | 768               | 1536             |
| Exponential unit | **1024**          | **2048**         |

`d=128`的时候，光是 exp unit 的计算时间就已经和 tensor core 的计算时间一样长了，所以 softmax and friends 一定会成为主要瓶颈。

---



从 Ampere 到 Hopper，再到 Blackwell，Tensor Core 每代都翻倍（甚至超过翻倍），HBM 带宽勉强跟上，但 SMEM 带宽、MUFU、RF 容量**三代都没变**。

FA4 的核心出发点就是在**缓解"非对称扩展"造成的失衡**。Hopper 时代 MMA 强了 3.2 倍，这种失衡还没那么严重，软件勉强还能用 RF + 同步 softmax 跟上。到了Blackwell 时代 MMA 又强了 2.3 倍，**软件层已经无法靠传统手段跟上**。

下面我们来看 FA4 的具体优化手段。







### Forward Pass

**FA4 前向 Kernel 的分块大小为 `Br = Bc = 128`**，这与硬件规模是对应的：

* Tensor Core 尺寸变成 `128×N×K`，TMEM 容量是 128 rows × 512 cols × 4B，`Br = 128` 可以让 warpgroup 中 128 个线程每个线程在 softmax 阶段处理一行数据，**不需要跨线程规约操作**（softmax 本身就是按行操作的）。
* 每个线程要在寄存器中保存 S/P tile 的一行，也就是 `Bc` 个 FP32/FP16 元素。`Bc = 128` 主要受到寄存器数量、TMEM 容量的约束。

#### What exactly are "softmax & friends"?

首先，有必要更明确地划分一下 "softmax and friends" 具体包含哪些标量操作。

```python
# one iteration of non-causual attention forward pass
S_ij = Q_i @ K_j^T # GEMM0
# ------------------------
# scale = 1/sqrt(d) * 1/ln(2)
m_ij = max(m_i, row_max(S_ij) * scale)  # FP32 max 归约 + FP32 mul + FP32 max
S_ij = S_ij * scale - m_ij              # FP32 FMA
P̃_ij = exp2(S_ij)                       # exp2 近似
l_ij = row_sum(P̃_ij)                    # FP32 add 规约
# -- update m_i and l_i --
alpha = exp2(m_i - m_ij)                # FP32 sub + exp2 近似
l_i = l_i * alpha + l_ij                # FP32 FMA
m_i = m_ij                              # reg copy
# -- rescale output accumulator --
O_i = O_i * alpha                       # FP32 mul
# ------------------------
O_i += P̃_ij @ V_j # GEMM1
```

这些标量操作会在对应的硬件 pipe 上运行，不同的硬件 pipe 在物理上是独立的，可以并行执行操作。

| Pipe                                 | 处理的指令                                                   | 吞吐量              |
| ------------------------------------ | ------------------------------------------------------------ | ------------------- |
| **FMA pipe / FP32 pipe / FFMA pipe** | FP32 的 `add/mul/fma`，包括 `max/min`、`fma.f32`             | ～128 ops/clock/SM  |
| **ALU pipe / INT pipe / XU pipe**    | 整数 add/sub/mul、bitwise (and/or/xor)、shift、比较、type conversion 的一部分 | ～128 ops/clock/SM  |
| **MUFU / SFU / XU**                  | exp2, log2, rcp, rsqrt, sin, cos                             | ～16 ops/clock/SM   |
| **Tensor Core**                      | MMA                                                          | ～8192 ops/clock/SM |
| **LSU**                              | load/store（包括 SMEM、TMEM、GMEM）                          |                     |

"softmax and friends" 的每一步操作，对应到 PTX 指令以及硬件 pipe。我们还可以根据硬件 pipe 的吞吐量，估算出每一步操作的延迟。

| 步骤                    | Hardware Pipe | 操作数量（per thread）         | 估计 cycles             |
| ----------------------- | ------------- | ------------------------------ | ----------------------- |
| TMEM load S             | LSU           | 1×`tcgen05.ld`（LDTM）         | 几十 cycle 延迟，可隐藏 |
| "row_max"               | FMA           | 127×`max.f32`                  | ~127                    |
| "scale + max with m_i"  | FMA           | 1×`mul.f32`+1×`max.f32`        | ~2（可忽略）            |
| "S * scale - m_ij"      | FMA           | 128×`fma.f32`                  | ~128                    |
| **exp2**                | **MUFU**      | 128×`ex2.approx.f32`           | **~1024**               |
| "row_sum"               | FMA           | 127×`add.f32`                  | ~127                    |
| alpha = exp2(m_i - m_i) | MUFU          | 1×`sub.f32`+1×`ex2.approx.f32` | 几 cycle（可忽略）      |
| update l_i              | FMA           | 1×`fma.f32`                    | ~1（可忽略）            |
| **rescale O**           | LSU + FMA     | d×`mul.f32` + LDTM & STTM      | ~d + 几十 cycle 延迟    |
| FP32→BF16               | CVT           | 64×`cvt.rn.bf16.f32`           | ~64                     |
| TMEM store P            | LSU           | 1×`tcgen05.st`（STTM）         | 几十 cycle 延迟，可隐藏 |

我们发现：“softmax and friends” 的大部分标量计算是在 FP32 pipe 上进行的（除了 exp2 计算在 MUFU 上）。但 FP32 pipe 的吞吐远高于 MUFU（128 ops/clock vs 16 ops/clock），所以 **MUFU 上的 exp2 成为了绝对的关键路径**（~1024 cycle，相当于 2 个 GEMM 的时间）。当然 FP32 pipe 的负担也不轻，几个开销比较大的操作都是对 S/P tile 的 element wise 操作（"row_max"，"scale & sub"，"row_sum"，FP32→BF16）。

接下来我们会看到，FA4 还会往 FP32 pipe 里塞 exp2 仿真，让 **FP32 pipe 也接近吞吐极限**。

---

#### new pipeline

FA4 采用了 **Warp specialization** 和 **Pingpong scheduling**（我们将它归类为 **Spatial Pineline** 实现的软件流水） 两种方式，来让不同的硬件 pipe overlap 执行。

**Warp specialization** 上面，由于 TMEM 的存在，Consumer 的工作现在可以拆分给多个 warpgroup 协作完成：

* TMA warpgroup 负责发起异步的 `cp.async.bulk.tensor` 指令；

* MMA warpgroup 负责发起异步的 `tcgen05.mma` 指令；
* Softmax warpgroup 负责计算 online softmax，包括计算 $P=\text{softmax}(S)$ 以及更新 $m_i$, $\ell_i$；
* Correction warpgroup 负责 rescale $O_i$.

**由于 Bottleneck 的转移，FA4 流水线的核心就是希望让 TMA/MMA/Correction warpgroup 的工作都隐藏在 Softmax warpgroup 的延迟中，让 MUFU pipe 没有空闲。**



FA4 是通过 **Spatial Pineline** 来实现的 GEMM 和 softmax 的 overlap。**每个 threadblock 会同时计算两个相邻的 Q/O tile**，当其中一个 tile 执行 GEMM 时，另一个 tile 则执行 softmax。（其实 FA3 就已经这样做了，只不过 FA4 把 rescale O 分给单独的 Correction warpgroup 执行，让 rescale O 也隐藏在 softmax中，从关键路径上解耦了）



![image-20260505213330940](/images/flash-attention/image-20260505213330940.png)

这是 warpgroup 视角下的软件流水线，一个 driver warpgroup + 两个 softmax warpgroup + 一个 correction warpgroup 的组合。

* TMA 和 GEMM 之间同样还是 Producer-Consumer 模型，不过 TMA warpgroup 和 MMA warpgroup 被合并成了一个 warpgroup（**driver WG**），由 driver warpgroup 完成所有的 TMA 和 MMA 指令的发射。

* 两个 O tile 的 rescale 合并在一个 correction warpgroup 中完成。这里能够让 rescale[j] 隐藏在 softmax[j] 中的关键在于，rescale[j] 的工作其实只需要得到 scale 因子 alpha 就可以开始，**所以当 softmax[j] 先计算出 alpha 之后， rescale[j] 就可以启动了，这样 rescale[j] 与 softmax[j] 的 exp 计算就可以 overlap。**

下面这个硬件 pipe 视角的流水线会更加清晰一些：

![image-20260505214524570](/images/flash-attention/image-20260505214524570.png)

GEMM/softmax/rescale 的数据都是通过 TMEM 来传递的：

* GEMM0[j] 在 MMA pipe 上计算出 S，写入 TMEM；然后 softmax[j] 启动，将 S 从 TMEM 读入寄存器；
* softmax[j] 在 FP32 pipe 上计算出 "row_max"，得到 alpha，将 alpha 写入 TMEM；然后 rescale[j] 启动，将 alpha 和 O 从 TMEM 读入寄存器；
* softmax[j] 在 MUFU pipe 上计算出 P，写入 TMEM；rescale[j] 在 FP32 pipe 上对 O 缩放，写入 TMEM；这两个都完成后，GEMM1[j] 启动，从 TMEM 读 P 和 O；

这样 TMEM 中的空间分配就是这样的：两个 O tile 需要一直保持，占据大约一半的空间；剩下的空间分配给两个 S tile 和 alpha 向量，P tile 与对应的 S tile 重叠。



最后，为了降低 softmax warpgroup 的寄存器压力，FA4 将 P 的存储分阶段进行：前四分之三的数据先被存储一次，并触发对应的 MMA 操作；最后四分之一的数据则单独存储。



---

#### Software emulation exp2



NV GPU 上的指数运算在 MUFU pipe 中进行，通过`ex2.approx.f32`一条指令完成。每个 SM 的 sub-partition 有 32 个 FFMA lane（对应 warp 32 lane），但 **MUFU 只配了 4 个 lane**，这导致 MUFU pipe 的吞吐只有 FP32 pipe 的 1/8。（B300 上已经把 MUFU 资源翻倍了，吞吐达到了 32 ops/clock/SM）



为了尽量提高 exp2 计算的吞吐，FA4 在算法上走两条路来计算 exp2：**一部分元素通过 MUFU pipe 硬件计算 exp2，一部分元素使用多项式近似来软件模拟计算 exp2。**

那如何通过软件模拟计算 exp2 呢？这个算法其实很早就出现了，利用了 IEEE 754 浮点数编码本身的结构。



> **核心思路：把 2^x 拆成两半**
>
> 要算 $2^x$，关键观察是 IEEE 754 单精度浮点数 (FP32) 的结构本身就是**以 2 为底的指数表示**：
> $$
> \text{FP32}: \text{value} = (-1)^s \cdot 2^{e-127} \cdot 1.m
> $$
> 其中 $e$ 是 8-bit 指数字段，$m$ 是 23-bit 尾数。也就是说，**FP32 的二进制位本身就编码了一个 2 的整数次幂乘上一个 [1, 2) 区间的尾数**。
>
> 我们把 $x$ 拆成整数部分和小数部分：$x = \lfloor x \rfloor + x_{\text{frac}}, \quad x_{\text{frac}} \in [0, 1)$
>
> 那么：$2^x = 2^{\lfloor x \rfloor} \cdot 2^{x_{\text{frac}}}$
>
> 这两个部分各自有非常便宜的算法：
>
> - $2^{\lfloor x \rfloor}$ ：$\lfloor x \rfloor$ 是个整数，乘 $2^{\lfloor x \rfloor}$ **就是把这个整数加到 FP32 的指数字段** —— 整数加法就可以完成，甚至不需要浮点运算
> - $2^{x_{\text{frac}}}$ ：$x_{\text{frac}} \in [0, 1)$，结果落在 $[1, 2)$ 区间。**这个区间很短，多项式逼近精度相对较高** —— 4-5 条 FMA 就可以满足精度要求
>
> ---
>
> **完整算法步骤拆解**
>
> 1. `x = max(x, -127)`
>
>    这样能避免指数结果过小下溢。因为 FP32 能表示的最小正规数是 $2^{-126}$，如果 `x < -127`，$2^x$ 已经接近或低于 FP32 的下溢边界。
>
> 2. 计算 $\lfloor x \rfloor$：将 $2^{23} + 2^{22}$ 加到 `x` 上，使小数位被压入尾数字段中，然后再以向下取整模式将 $2^{23} + 2^{22}$ 减回去。这里利用了尾数 23-bit 长度的限制。
>
> 3. 计算小数部分：$x_{\mathrm{frac}} = x - \lfloor x \rfloor$
>
> 4. 多项式进行求值计算 $2^{x_{\mathrm{frac}}}$
>
>    在 $[0, 1)$ 上用一个 4-5 次多项式逼近 $2^{x_{\text{frac}}}$：
>    $$
>    2^{x_{\text{frac}}} \approx p_0 + p_1 \cdot x_{\text{frac}} + p_2 \cdot x_{\text{frac}}^2 + \dots + p_n \cdot x_{\text{frac}}^n
>    $$
>    多项式可以用 **Horner 形式**求值，让所有运算变成 FMA：
>    $$
>    p(x_{\text{frac}}) = ((((p_n \cdot x_{\text{frac}} + p_{n-1}) \cdot x_{\text{frac}} + p_{n-2}) \cdot x_{\text{frac}} + \dots) \cdot x_{\text{frac}} + p_0)
>    $$
>    每一层都是一个 `fma.f32 r, x_frac, r, p_i`，**n 次多项式正好用 n 条 FMA**。
>
>    论文给出的精度数据：
>
>    | Degree            | FP32 max rel err | BF16 max rel err |
>    | ----------------- | ---------------- | ---------------- |
>    | 3                 | 8.77e-5          | 3.90e-3          |
>    | 4                 | 3.05e-6          | 3.89e-3          |
>    | 5                 | 1.44e-7          | 3.89e-3          |
>    | Hardware MUFU.EX2 | 1.41e-7          | 3.89e-3          |
>
>    注意，转换到 BF16 精度上，**degree 3 之后所有版本的 BF16 误差都一样**，因为 BF16 本身的量化误差（~3.9e-3，对应 7-bit 尾数）已经掩盖了多项式逼近误差。
>
> 5. 合并整数部分和小数部分：只要**把 $\lfloor x \rfloor$ 加到 $2^{x_{\text{frac}}}$ 的指数字段上**。



多项式模拟计算 exp2 占用的是 FFMA 和 ALU pipe，这两个 pipe 的吞吐量比较高。但是，软件模拟的代价是：需要额外的寄存器来保存中间值和系数，会增加寄存器占用和带宽消耗。并且 FFMA pipe **不是只跑 emulation exp2**，softmax & friends 的其他标量运算基本上都要在 FFMA pipe 上运行，已经占用了 FFMA pipe 相当一部分吞吐，留给 emulation exp2 的余量有限。

可以简单估算一下，上面表格统计出来的 softmax & friends 分工，假设把 25% 的 exp2 用软件模拟。

* MUFU pipe 时间 ～ 1024 × 75% = 768 cycles
* FFMA pipe 原本计算时间 ～ 500 cycles。假设一次 emulation exp2 计算消耗 8 条 FMA 指令，那么 FFMA pipe 计算 25% 的 exp2 时间 ～ (128 × 128 × 25% × 8) ÷ 128 = 256 cycles。总共就是 756 cycles
* 所以 **FA4 对 exp2 进行软件模拟的比例约在 10–25%**



---



####  $O_i$ & $\ell_i$ Conditional rescaling

经典 FlashAttention 的 online softmax 维护两个统计量：

- $m_i$：第 $i$ 行到目前为止所有处理过的 block 的 row max
- $\ell_i$：第 $i$ 行用 $m_i$ 归一化后的指数和

每次 max 跳变，rescale 因子 $e^{m_i^{(j-1)} - m_i^{(j)}}$ 会把**累加器 $O$ 和 $\ell$ 同步缩小**到新基准 $m_i^{(j)}$ 下的尺度。

softmax 减去 row max 是为了保证 e 指数运算数值稳定（直接计算 $e^{x}$ 很容易在 FP32 上溢出），而 online softmax 对 $O$ 和 $\ell$ 进行 rescale 的同样也是为了保持指数的数值稳定。只要 $O_{\text{final}}$ 和 $\ell_{\text{final}}$ 共用同一个最大值 $m^*$​，最后一步归一化时：
$$
\frac{O_{\text{final}}}{\ell_{\text{final}}} = \frac{\sum_j e^{S_{ij} - m^*} V_j}{\sum_j e^{S_{ij} - m^*}}
$$
**$m^*$ 在分子分母同时出现，会被精确消掉**。所以从数学上看，$m^*$ 取什么值都不影响最终结果——它只是一个"内部的参考点"，用于保证每一个元素的 exp 指数都不超过 0。

但其实没有必要控制这么严格——如果 exp 指数不是很大，在 FP32 安全范围内，**就不需要更新 $m_i$，也不需要对 $O_i$ 和 $\ell_i$ 进行 rescale**。这就是 FA4 的又一个优化：$O_i$ & $\ell_i$ 的 confitional rescaling。

---

**Conditional rescaling 的逻辑**



我们可以在 rescaling 中容忍一定的“松弛量”：当
$$
m_i^{(j)} - m_i^{(j-1)} \leq \tau
$$
时，**我们跳过对 $m_i$ 的更新，并继续使用 $m_i^{(j-1)}$。只有当最大值的变化超过阈值时才进行 rescale。**

其中 $\tau$ 是一个阈值，通常设为 $\log_2(256)=8.0$，对应的 rescaling factor 为 256.0。

FA 算法中的更新逻辑变为：
$$
O_i^{(j)} =
\begin{cases}
\alpha * O_i^{(j-1)} + e^{S_{ij}-m_i^{(j)}}V_j, & \text{if } m_i^{(j)} - m_i^{(j-1)} > \tau \\
O_i^{(j-1)} + e^{S_{ij}-m_i^{(j-1)}}V_j, & \text{otherwise}
\end{cases}
$$

$$
\ell_i^{(j)} =
\begin{cases}
\alpha * \ell_i^{(j-1)} + \sum_k e^{S_{i,k} - m_i^{(j)}}, & \text{if } m_i^{(j)} - m_i^{(j-1)} > \tau \\
\ell_i^{(j-1)} + \sum_k e^{S_{i,k} - m_i^{(j-1)}}, & \text{otherwise}
\end{cases}
$$

```
# main loop
for j = 1 to Tc:                      # 内层：遍历 K/V 块
    load K_j (Bc×d), V_j (Bc×d) : HBM → SRAM
    S_ij = Q_i @ K_j^T                # GEMM0

    m_i_new = max(m_i, rowmax(S_ij))
    if m_i_new - m_i > τ:
    		P̃_ij = exp(S_ij - m_i_new)
    		alpha = exp(m_i - m_i_new)
    		ℓ_i = alpha * ℓ_i + rowsum(P̃_ij)
    		m_i = m_i_new
    		O_i = alpha * O_i
    else:
    		P̃_ij = exp(S_ij - m_i)
    		ℓ_i = ℓ_i + rowsum(P̃_ij)

    O_i = O_i + P̃_ij @ V_j            # GEMM1

end for   # 内层循环结束，K/V 全部遍历完毕

O_i = O_i / ℓ_i                       # 循环结束后统一做一次归一化
L_i = m_i + log(ℓ_i)                  # 保存 logsumexp 供 backward 使用
write O_i, L_i → HBM                  # 写回 HBM
```

Conditional rescaling 的实际收益取决于 "$m$ 跳变 > $\tau$ 的频率"。直觉上，处理顺序是按 K block 流式扫的，前几个 block 容易找到新的 max（每次都会触发 rescale），但越往后，$m$ 越接近全局 max，新 block 触发跳变的概率越低。

具体地，假设 $S$ 的元素近似独立分布，处理到第 $j$ 个 block 时已经看了 $jN$ 个值，新 block 出现一个比当前 max 大 8 的值，概率会随 $j$ 快速下降。所以**整个 attention 计算中，rescale 触发率可能只有 5-10%**，绝大多数迭代都跳过 rescale。

这正是 FA4 conditional rescaling 的核心收益：让 **90% 以上的迭代跳过 rescale，大大节省了 correction WG 的工作量**。



> 实际中，为避免 warp divergence，只要 warp 中任何一个线程需要 rescale，就对整个 warp 做 rescale。













---



### Backward Pass

TODO











---



## 写在后面

> 未来的 Kernel 优化工作会有什么样的趋势？
>
> 
>
> * **Trend 1：编程模型层的"民主化"** 
>
>   CuTe、Triton、cuTile 这一批工作的共同目标，是**把硬件感知的复杂度从 "个人英雄主义" 转移到 "DSL/编译器" 上**。FA1 那种工作以前需要 Tri Dao 这样的人手写 CUDA 几个月，现在 Triton 上几百行能逼近 80% 性能；FA4 利用 CuTe-DSL 来实现，没有了复杂的 C++ template，大幅缩减了编译时间，并且降低了 GPU kernel 的上手门槛。
>
> * **Trend 2：但顶级 sota 仍然需要 hand-tuning**
>
>   DSL 永远会有 10-20% 的性能差距追不上 hand-written assembly。这部分由厂商内部团队（cuBLAS/cuDNN）和少数顶级开源项目（CUTLASS 的 collective mainloop、ThunderKittens 的某些 kernel）填补。**学术界很难在这个层次上竞争**——不是因为信息不对称，而是工程量和持续投入不对称。
>
> * **Trend 3：算法-硬件 co-design 的窗口期**
>
>   真正有学术空间的，反而是**当一个新硬件特性出现、但厂商还没来得及把所有 algorithm 都重新做一遍的窗口期**。FA1 在 2022 年出现，是因为 Volta/Ampere 的 SMEM 已经存在好几年了，但学术界一直没人想清楚 attention 该怎么用它；Hopper 的 TMA + warp specialization 出来后，FA3、FlashDecoding、各种 prefill/decoding 优化又是一波；Blackwell 的 TMEM 出来，FA4、新的 GEMM kernel、新的 fused op 又会是一波。
