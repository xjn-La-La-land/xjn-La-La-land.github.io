---
title: DeepSeek-V4.1-Flash 架构调研
date: 2026-09-13 21:22:04
updated: 2026-09-15 10:46:06
tags: LLM推理
comments: false
excerpt: DeepSeek-V4.1-Flash 架构解析：CED（Causal-Encoder+Decoder）结构、SWA Bounded Replay、CSA2 的 Full/Reindex/Reuse 三种模式、Hierarchical Sparse Indexer，以及 mHC、Engram、DSpark 等高效架构扩展。
mathjax: true
---

## 官方技术报告

<iframe src="/html/deepseek-v4-flash/deepseek-v4-flash-manual.html" loading="lazy" title="DeepSeek-V4.1-Flash 架构与推理手册" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/deepseek-v4-flash/deepseek-v4-flash-manual.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>

## 组会 slides

<iframe src="/html/deepseek-v4-flash/deepseek-v4-flash-slides.html" loading="lazy" title="DeepSeek V4.1 Flash 架构调研（组会 slides）" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/deepseek-v4-flash/deepseek-v4-flash-slides.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>

## 整体架构

![这张图片展示了DeepSeek-V4.1-Flash的整体架构，主要包含左侧的因果编码器（Causal Encoder）和右侧的解码器（Decoder）两部分，二者通过CED连接。因果编码器共40层，划分成各20层的结构，前2层采用SWA（滑动窗口注意力），其余层使用CSA2。解码器中部分注意力模块带有“Reindex”“Reuse”等标识，模型还包含Engram、Single-Pass mHC、DSpark等配套模块，最终通过分层稀疏索引器输出候选池，呈现了该模型的各核心组成模块及连接关系。](figure-01.png)

- 40 层网络划分为各含 20 层的 causal encoder 和 decoder (Causal-Encoder+Decoder: CED)
- 所有层的 FFN 均使用标准 DeepSeekMoE
- encoder 的前两层只使用 SWA（Sliding-Window Attention），其余各层使用 Compressed Sparse Attention 2（CSA2）+ SWA
- 模型还包含 Single-Pass mHC、Engram、DSpark 和 Hierarchical Sparse Indexer

---

## CED: Causal-Encoder+Decoder 结构

### CED 的计算流程

- <strong>Prefill：</strong>输入 → 20 层 causal encoder → 构建 global KV，仅末尾 128 个 token 继续经过 20 层 decoder、近似重建 SWA KV → 预测首个输出 token；
- <strong>Decode：</strong>新 token → 20 层 encoder → 20 层 decoder（读取 global KV + 本层 SWA KV）→ 预测下一个 token。

---

### CED 的收益在哪里？

CED 改变 decoder global KV 的来源：由 encoder 最终 hidden state 投影，而不是由各 decoder layer 自身的 hidden state 生成。

- 普通 decoder-only Transformer 为了准备每一层的历史 KV，通常必须让全部 prompt token 跑过全部层。
- CED 中，decoder 的 global KV 在得到 encoder 最终 hidden state 后就可以构建，不必让全部 prompt token 再跑完后 20 层。

---

### CED 为什么能这样做？

- Decoder-only 结构做 Prefill，所有输入 token 都需要经过每一层计算得到每一层的 hidden state，从而为后续生成准备 global KV cache；
- 我们只要预测下一个 token（只需要最后一个位置的 logits），前面所有历史位置的深层表示更接近于 next-token 的预测，作用有限。
- CED 结构可以理解是在调整两个部分的分工：

| 工作 | CED 的安排 |
|-|-|
| 为历史 token 构建全局可读取的表示 | 主要由 encoder 完成 |
| 基于历史信息完成当前 next-token 预测 | 仍使用完整的 encoder + decoder |

- 现有结果支持这套方案在已测条件下具有较好的能力与效率，但不构成相对普通 decoder-only 的严格无损保证。

---

## Attention

| 位置 | 层数 | Attention 配置 |
|-|-|-|
| Encoder 第 1–2 层 | 2 | **只有 SWA** |
| Encoder 第 3–20 层 | 18 | **SWA + CSA2 global attention**，global KV 压缩率为 2 |
| Decoder 第 21–40 层 | 20 | **SWA + CSA2 global attention**，global KV 压缩率为 1 |

对于这些 CSA2 层，本层 Q 在一次主 attention 中联合读取：

```Plaintext
本层 Q
   ├── 读取本层 SWA KV：最近的局部窗口
   └── 读取选中的 global KV：提供更长范围的历史信息
```

这里有两个不同的缓存规则：

- SWA KV：每层自己生成、自己维护，来自该层的 hidden state，窗口大小为 128 tokens。
- Global KV：按 CSA2 配置生成或跨层复用。

---

## SWA: Sliding-Window Attention

SWA（Sliding-Window Attention）限制每个 token 只关注最近的一段窗口，用更少的计算和缓存，换取更有限的直接访问范围。

每一层的 SWA KV 直接由本层 hidden state 生成 => Prefill 需要为每一层计算 SWA KV: [B, H, 128, d_h]，这个过程叫做 SWA Replay  
SWA Replay 恢复一个 128-token 的窗口，需要计算的序列长度不止 128。

> 因为 SWA KV 来自各层自己的 hidden state，而这些 hidden state 又依赖前一层的 SWA 计算。依赖范围会沿层数向前扩展。  
> 精确重建 $L$ 层的 SWA KV 需要重放 $L\times n_{win}$ 个 token。

```Plaintext
SWA window n_win = 4 (current token + 3 previous tokens)

                                  History               |  Target
Position              0..2     3..5     6..8     9..11  | 12..14
Input                [0..2]   [3..5]   [6..8]   [9..11] |[12..14]
                                        v               |
                                                        |
Layer 1 (SWA)                                           |
  Q                          [==========================|========]
  KV                [===================================|========]
                                        v               |
  H1                          [3..5]   [6..8]   [9..11] |[12..14]
                                                        |
                                        v               |
Layer 2 (SWA)                                           |
  Q                                   [=================|========]
  KV                         [==========================|========]
                                        v               |
  H2                                   [6..8]   [9..11] |[12..14]
                                                        |
                                                 v      |
Layer 3 (SWA)                                           |
  Q                                            [========|========]
  KV                                  [=================|========]
                                                 v      |
  H3                                            [9..11] |[12..14]
                                                        |
                                                        |    v
Layer 4 (SWA)                                           |
  Q                                                     |[=======]
  KV                                           [========|========]
                                                        |    v
Output                                                  |[12..14]
```

每向前追溯一层，需要的 hidden state 范围就向左扩展 $W-1=3$ 个位置。

---

### SWA Bounded Replay

Bounded Replay 就是在这个依赖上做了截断：原本应当读取、但位于 12～14 之前的 decoder SWA KV 被省略了，因此得到的状态是近似的。

```Plaintext
W = 4
Replay range = [11, 14]

Position          0 ........ 10 | 11   12   13   14
                                |
Decoder input     not replayed  |[11] [12] [13] [14]
                                |         v
                                |
Layer 1 (SWA)                   |
  Q                             |[===============]
  KV                            |[===============]
                                |         v
  H1                            |[11] [12] [13] [14]
                                |
                                |         v
Layer 2 (SWA)                   |
  Q                             |[===============]
  KV                            |[===============]
                                |         v
  H2                            |[11] [12] [13] [14]
                                |
                                |         v
Layer 3 (SWA)                   |
  Q                             |[===============]
  KV                            |[===============]
                                |         v
  H3                            |[11] [12] [13] [14]
                                |
                                |         v
Layer 4 (SWA)                   |
  Q                             |[===============]
  KV                            |[===============]
                                |         v
  H4                            |[11] [12] [13] [14]
                                |
Needed outputs                  |     [12] [13] [14]
```

理论依据：实际模型对 SWA 的一些远处依赖并不敏感，SWA 的实际有效感受野远小于理论上的 $n_{win} \times \frac{L}{2}$.

---

## CSA2: Compressed Sparse Attention 2

> 长上下文需要同时控制 KV cache 存储量和 attention 计算量。这些成本可以沿三个具有乘性关系的维度降低：

- Entry 大小维度: GQA 减少 KV head 数量；MLA 在多个 head 之间共享较小的 latent。
- Seq_len 维度: 每 $m$ 个 token 压缩为一个 entry，例如 DeepSeek-V4 中的 CSA。
- 层维度: 部分层复用其他层的 cache 和选择结果，不再各自维护；也可以将部分层替换为更高效的层。

---

每个 CSA2 层都被静态指定为 Full、Reindex 或 Reuse 三种模式之一。三种模式都会计算本层 query 和 SWA KV，再结合选中的 main KV entry 生成新的 attention output；区别在于如何获取 main KV、indexer K 和 Top-K 索引。

![图片展示了CSA2的三种模式：Full、Reindex和Reuse。Full模式下，Core Attention、Selected Main KV、Indexer Q、Indexer K、Top-K Indices等组件参与；Reindex模式中，Indexer Q和Indexer K被标注为黄色；Reuse模式下，Main KV和Top-K Indices被标注为黄色。三种模式均计算本层query和SWA KV，结合选中的main KV entry生成新的attention output，但获取main KV、indexer K和Top-K索引的方式不同。](figure-02.png)

### Full Mode

我们先详细拆解 Full Mode 的计算。以单个输入 token 来看：$x_t\in\mathbb R^{5120}$

#### 1. 生成 Main Q

Main Q 使用两级 projection：

$$r_t=\operatorname{Norm}_Q(x_tW_{QA})\in\mathbb R^{1280}$$

$$q_{t,h}=r_tW_{QB,h}\in\mathbb R^{512},\qquad h=1,\ldots,64$$

这里 $r_t$ 是中间的 query latent；后面的 Indexer Q 也会使用它。

#### 2. 生成 SWA KV

SWA KV 由本层输入直接生成：

$$u_i=\operatorname{Norm}_{SWA}(x_iW_{SWA})\in\mathbb R^{512}$$

对于位置 $t$，局部窗口为：

$$\mathcal W_t=\{\max(1,t-127),\ldots,t\}$$

因此，SWA 分支提供的是 $\{u_i:i\in\mathcal W_t\}$。  
每条 SWA KV entry 是一份 512 维 latent，同时作为 key 和 value，并由多个 query heads 共享。

#### 3. 生成 Main KV：把一组 token 压缩成一条 entry

用 $g_i$ 表示生成 global KV 的输入：

- Encoder 层的输入：来自该层的输入 hidden state。
- Decoder 层的输入：来自 encoder 最终 hidden state，这是 CED 的规定。

先生成待聚合的向量和 gate logits：

$$a_i=g_iW_C \in\mathbb R^{512},\qquad z_i=g_iW_G \in\mathbb R^{512}$$

随后，每个 channel 独立地在组内做 softmax：

$$\alpha_{i,c} = \frac{\exp(z_{i,c})} {\sum_{p\in\mathcal B_j}\exp(z_{p,c})}$$

$$\widetilde c_{j,c} = \sum_{i\in\mathcal B_j}\alpha_{i,c}a_{i,c}, \qquad c_j=\operatorname{Norm}_{C}(\widetilde c_j)$$

$c_j$ 就是图中的 Main KV entry

V4.1-Flash 的具体设置是：

| 位置 | 压缩率 | 含义 |
|-|-|-|
| Encoder 的 CSA2 | $m=2$ | 每两个 token 生成一条 Main KV |
| Decoder 的 CSA2 | $m=1$ | 每个 token 一条 Main KV，不做序列维聚合 |

当 $m=1$ 时，Main KV 计算退化为：

$$c_i=\operatorname{Norm}_{C}(g_iW_C)$$

不需要计算压缩 gate。

#### 4. 生成 Indexer Q

Indexer 使用更小的维度：32 个 query heads，每个 head 128 维。  
Indexer Q 来自前面的 query latent：

$$q^I_{t,h}=r_tW^I_{Q,h}\in\mathbb R^{128}$$

#### 5. 生成 Indexer K

Indexer K 由已经生成的 Main KV 投影得到：

$$k^I_j=\operatorname{Norm}_{I}(c_jW^I_K)\in\mathbb R^{128}$$

在 32 个 Indexer Q head 之间共享。

#### 6. 给 Main KV 打分

从 $x_t$ 生成各个 indexer head 的组合权重：

$$w_t=x_tW^I_w\in\mathbb R^{32}$$

然后计算位置 $t$ 对 Main KV entry $j$ 的检索分数：

$$\boxed{ s_{t,j} = \frac{1}{\sqrt{128}\sqrt{32}} \sum_{h=1}^{32} w_{t,h}\, \operatorname{ReLU} \left(q^I_{t,h}(k^I_j)^\top\right) }$$

这个分数用于筛选位置，还不是最终 attention 权重。

#### 7. Top-K 选择 Main KV

首先必须先满足 causality：一个压缩分组只有在其最后一个 token 已经可见时，才能被当前 query 读取。  
从可见KV中选出得分最高的最多 512 条：

$$\mathcal I_t = \operatorname{TopK}_{j:\,jm\le t}(s_{t,j},512)$$

这里得到的是每个 query 位置的一组共享索引，供该位置的 64 个 Main Q heads 使用。

#### 8. 拼接两类 KV

把局部 SWA KV 和选出的 Main KV 拼起来：

$$B_t = \operatorname{Concat} \left( [u_i]_{i\in\mathcal W_t}, [c_j]_{j\in\mathcal I_t} \right)$$

$B_t$ 最多包含：

$$128+512=640$$

条 KV entry，每条 512 维。

#### 9. 执行 Core Attention

然后，每个 main head 使用自己的 Main Q，重新计算真正的 attention 权重：

$$p_{t,h} = \operatorname{softmax} \left( \frac{q_{t,h}B_t^\top}{\sqrt{512}} \right)$$

$$o_{t,h}=p_{t,h}B_t$$

最后合并各个 head，经 output projection 得到图顶端的 Attention Output：

$$y_t=\operatorname{OutputProjection}\left(\operatorname{Concat}_{h=1}^{64}o_{t,h}\right)$$

### Reindex Mode

复用前面最近一层可用的 main KV 及对应的 indexer K。

### Reuse Mode

本层复用前面最近可用的 main KV，以及前面某个 Full 或 Reindex 模式层针对这份 main KV 计算出的最新 Top-K 索引。

---

## Hierarchical Sparse Indexer

CSA 的 Indexer 需要计算 $Q@K^\top$，在极长上下文下，这部分开销仍是主要计算瓶颈。  
Hierarchical Sparse Indexer 就是为了减少 Decoder 中多个 Indexer 反复扫描长上下文的开销。  
关键观察：在 decoder 中，可以直接使用较浅层 indexer 的信息，限制较深层 indexer 需要考虑的候选位置，而无需增加额外 state。  
核心机制是：第一个 Full CSA 的 Indexer 建立候选池，后面的 ReIndex CSA 的 Indexer 只在池内重新筛选。

![该图展示了Hierarchical Sparse Indexer的工作流程，清晰呈现了其分层稀疏索引的核心机制。第一个全模式索引器会对全部位置计算分数，生成候选池，后续重索引模式的多个索引器均基于这个共享候选池计算分数，无需重新扫描全部位置。顶部均标注有Top-512，对应最终要筛选出的目标数量，整个流程通过这种分层设计，将后续索引器的单查询开销从随上下文长度线性增长转为常数，以此降低Decoder处理长上下文的开销，这一架构正是为实现上述优化而设计的。](figure-03.png)

```Plaintext
全部可见 global KV
        |
        v
第一个 Full Indexer
        +--> Top-512：本层 attention 使用
        |
        +--> 候选池：最多 16,384 个位置
                       |
                       +--> Reindex 层 A --> 自己的 Top-512
                       +--> Reindex 层 B --> 自己的 Top-512
                       +--> ...
```

候选池大小固定时，后续 indexer 的单 query 开销由随上下文长度线性增长变为常数。

---

### CSA vs Full Attention: 收益在哪里？

1. 主 attention 只精细处理一小部分历史  
对于接近 1M token 的上下文：

|  | Full Attention | CSA2 |
|-|-|-|
| 当前 Q 参与主 attention 的 KV | 接近 100 万条 | 最多 128 条 SWA KV + 512 条选中的 global KV |
| 如何获得长程信息 | 对全部历史直接计算 attention | 先用 Indexer 筛选，再读取选中的 global KV |
| 最近的信息 | 与其他历史一起处理 | SWA 保留直接的局部访问 |

因此，主 attention 中的 $QK^\top$、softmax 和加权求和规模大幅缩小。（FlashAttention 可以降低 Full Attention 的中间存储和数据搬运开销；CSA2 则进一步减少了实际参与主 attention 的 KV 条目。）

---

1. 全局扫描仍然存在，但交给更小、可复用的 Indexer  
你可能会想到：Indexer 不还是要扫描历史吗？  
是的，但它比主 attention 更轻：

- Main Q：64 heads × 512 维。
- Indexer Q：32 heads × 128 维，主要计算检索分数。
- Reuse 层直接沿用 Top-K，不重新索引。
- Decoder 的 Hierarchical Sparse Indexer 让后续 Reindex 层只搜索最多 16,384 个候选位置。  
所以它把工作分成了：较便宜的筛选，加上对少量位置的精细 attention。

---

1. 跨层共享和压缩，减少 KV 存储  
CSA2 的存储收益主要来自：

- 跨层共享：多个 layer 复用同一份 main KV 和 indexer K，减少重复缓存。
- 序列压缩：Encoder 中每两个 token 聚合成一条 global KV；Decoder 的压缩率为 1。
- 配套的 FP4 缓存格式：进一步减少每条 main KV 的字节数。

更小的缓存可以容纳更多并发请求，也能降低 prefix cache 加载、卸载和迁移的成本。这正是 Agent 长上下文场景中的重要收益。

---

那么，代价呢？  
Full Attention 允许每个 head 直接给全部历史位置分配权重；CSA2 加入了压缩、共享和筛选约束：

- 压缩可能损失细节；
- Indexer 可能漏掉重要位置；
- 跨层复用 KV 或 Top-K 会减少各层独立选择的自由度。

---

## 高效架构扩展

---

### mHC

mHC（“流形约束的超连接”） 将普通 Transformer 的单条 residual stream 扩展为多条。  
核心机制：多条 stream 保存信息，一条混合输入参与计算  
普通 residual connection 可以写成：

$$x_{\ell+1}=x_\ell+F_\ell(x_\ell)$$

每次计算产生的新信息，都加到同一条 hidden state 中。  
DeepSeek-V4.1 mHC 为每个 token 保留 4 条 residual stream，每条仍是 $d=5120$ 维：

$$X_\ell=\begin{bmatrix}x_{\ell,1}\\x_{\ell,2}\\x_{\ell,3}\\x_{\ell,4}\end{bmatrix}\in\mathbb R^{4\times d}$$

mHC 的计算公式是：

$$X_{\ell+1} = \underbrace{B_\ell X_\ell}_{\text{传递旧信息}} + \underbrace{C_\ell F_\ell(A_\ell X_\ell)}_{\text{计算并写入新信息}}$$

$F_\ell$ 可以是 attention 或 MoE。  
这些系数由当前 token 的 residual 数据动态预测：

$$(A_\ell,B_\ell,C_\ell)=\mathcal H_\ell(X_\ell)$$

A、B、C 系数的含义

| 系数 | 形状 | 控制什么 | 直观理解 |
|-|-|-|-|
| $A_\ell$ | $1\times 4$ | 从各条 stream 读入多少信息 | **怎么读** |
| $B_\ell$ | $4\times 4$ | 旧 stream 之间怎样混合和传递 | **怎么保留、怎么交换** |
| $C_\ell$ | $4\times1$ | 新结果向各条 stream 写入多少 | **怎么写回** |

---

### Single-Pass mHC

DeepSeek-V4.1 将当前输入使用的 $A_\ell$，换成前一个子层生成的 $A_{\ell-1}$：

$$\boxed{ X_{\ell+1} = B_\ell X_\ell+ C_\ell F_\ell(A_{\ell-1}X_\ell) }$$

当前仍计算 $(A_\ell,B_\ell,C_\ell)$，但把 $A_\ell$ 留给下一子层。这里“前一个”沿网络深度计数，不是上一个 token。  
这样，当前输入混合无需等待本次系数预测完成，更容易与 residual 更新等操作融合，减少 activation 的重复读写。多 stream、B 的残差混合和 C 的写回机制都保留，改变的是输入混合系数的使用时机。

---

### Engram

Engram 是一种 conditional memory：将近期 token 组合映射到大型、可训练的 embedding 表，以少量查表访问提供额外特征，再通过 context-aware gate 融入模型。

- **核心机制：**

```Plain Text
最近的 2 / 3 / 4 个 token
→ 多头 hash 确定地址
→ 查表读取 embedding
→ 拼接并 projection
→ 根据当前 hidden state 计算 gate
→ 将结果加入 residual stream
```

- <strong>查什么：</strong>以当前 token 结尾的连续 token 组合，即 token n-gram。例如 [I, like, this, book] 中，当前位置的 2-gram 是 [this, book]。
- <strong>读出什么：</strong>训练得到的 embedding vector。查表地址由 token ID 确定，不需要扫描整张表。
- <strong>怎样使用：</strong>查表特征经过 projection 得到 key/value，当前 hidden state 与 key 的匹配程度决定 gate，再把 value 加入 residual stream。

对第 $r$ 条 mHC residual stream，可以简写为：

$$\boxed{x'_{t,r}=x_{t,r}+g_{t,r}v_t}$$

其中，$v_t$ 来自查表特征，$g_{t,r}$ 根据当前上下文计算。相同的 token 组合可以查到相同表项，但在不同上下文中的使用强度不同。

> “MoE 按输入选择少量专家参与计算，Engram 则按 token 组合读取少量参数记忆。V4.1 放了两个 Engram 模块，虽然总表很大，每次只查少量行，并用当前 hidden state 决定这些记忆要用多少。”

---

### DSpark

DeepSeek-V4.1-Flash 使用 DSpark 进行 speculative decoding。

![这张图片展示了DeepSeek-V4.1-Flash使用的DSpark推测解码方案的核心优化思路与技术细节，整体围绕“高效解码”展开，包含INHERITED SPIKE、REPAIR THE CHAIN, NOT THE TOPOLOGY、PACK REAL TOKENS, THEN VERIFY ONCE等多个核心模块。图中明确标注了DSpark相关的关键机制，如轻量串行修正（Lightweight serial correction）、单链无树拓展、全局边际成本（Global marginal cost），以及硬件成本模型、匹配处理器的前缀调度器等优化内容，直观呈现了该方案通过精准采样、并行计算、冗余验证优化等手段降低解码成本、提升效率的设计逻辑，契合架构调研中对DSpark高效推测解码的说明。](figure-04.png)

Drafter 一次 forward 可并行计算 5 个 draft 位置的 base logits，同时由轻量的 Markov head 建模 draft token 之间的依赖。  
Drafter 包含 3 个 Transformer block，sliding attention window 为 128 个 token.  
DSpark 在预训练结束后的专门阶段引入。在该阶段，backbone 保持冻结，只训练 DSpark。