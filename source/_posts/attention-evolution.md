---
title: LLM Attention 架构近期演进
date: 2026-09-16 14:52:01
updated: 2026-09-22 00:20:03
tags: LLM推理
comments: false
excerpt: 梳理 Full / Linear / Sparse / Compressed / SWA 五大类 Attention 的近期演进，配合 DeepSeek、Qwen、Kimi、MiniMax 四家技术报告的详细解读（内嵌 HTML 完整文档）。
mathjax: true
---

<iframe src="/html/attention-evolution/attention-overview.html" loading="lazy" title="Attention 演进汇总" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/attention-evolution/attention-overview.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>

## Overview

### Full Attention

Full 指的是**每个 query token 会访问全部因果可见的历史 token**。Full Attention 有一些简单变体：

GQA/MQA 描述 **KV entry 如何在 query heads 之间共享**

- **GQA** (Grouped-Query Attention)：每组 query heads 共享相同的 KV。
- **MQA** (Multi-Query Attention)：所有 query heads 共享相同的 KV。

**Gated** Attention 对输出加上了一次门控


### MLA（Multi-head Latent Attention）

**MLA** 用低维的 latent 向量压缩表示 KV entry，减少了 KV cache 的 `bytes/entry` 存储开销。（没有减少 FLOPS）

**Gated** MLA 同样对输出增加一次门控


### Linear Attention

Linear 指的是**计算量 FLOPS 随序列长度线性增长**（同时用**固定大小**的递推状态保存历史信息）

- **Lightning Attention**：以输入无关的衰减系数递推状态，不做 delta-rule 误差校正：

$$S_t=\rho S_{t-1}+k_t^\top v_t$$

- **GDN** (Gated DeltaNet)：用输入相关的标量衰减系数递推状态，并按预测误差执行 delta-rule 校正:

$$S_t^-=\alpha_tS_{t-1},\\S_t=S_t^-+\beta_tk_t^\top(v_t-k_tS_t^-)$$

- **KDA** (Kimi Delta Attention)：将 GDN 的标量衰减扩展为 key feature 逐维衰减，再执行相同的 delta-rule 误差校正：

$$S_t^-=\operatorname{diag}(\alpha_t)S_{t-1},\\S_t=S_t^-+\beta_tk_t^\top(v_t-k_tS_t^-)$$


### Sparse Attention

Sparse 指的是**每个 query token 只选择部分历史 token 访问**，减少了 core attention 的 FLOPS。（没有减少 KV cache 存储开销）

- **DSA** (DeepSeek Sparse Attention)：MLA 低秩压缩 + MQA Indexer 选择 Top-K KV entry
- **CSA/CSA2** (Compressed Sparse Attention)：Compressor 在序列长度方向压缩，MLA 在 entry 方向压缩，MQA Indexer 选择压缩过的 KV entry
- **QSA** (Qwen Sparse Attention)：没有 MLA，MQA Indexer 按 block 粒度选择 KV entry
- **MSA** (MiniMax Sparse Attention)：同样 MQA Indexer 按 block 粒度选择 KV entry


### Compressed Attention

Compressed 指的是**将多个 token 的 KV entry 合并为一个压缩向量表示**，减少 KV cache `entries/layer` 的存储开销，同时减少 FLOPS；

- **CSA/CSA2** (Compressed Sparse Attention)：相邻几个 token 加权聚合得到一个 compressed entry，在 compressed entry 上运行 DSA
- **HCA** (Heavily Compressed Attention)：同样的压缩，但是压缩率远大于 CSA，在 compressed entry 上运行 Full Attention


### SWA (Sliding Window Attention)

每个 query token 只访问局部窗口中的历史 token，保留附近 token 的精确细节，更远的 token 信息需要跨层传递。


---


## DeepSeek 技术报告详解

<iframe src="/html/attention-evolution/deepseek-attention.html" loading="lazy" title="DeepSeek Attention 机制演进" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/attention-evolution/deepseek-attention.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>

> 之前已经调研过 DeepSeek V4.1 Flash 的架构，详细分析了 CSA2(Compressed Sparse Attention) 机制，所以我们这里主要补充前面几代的 attention 演进  
> 技术报告中 attention 机制完整的内容已经整理在 html 文档中了，这里只是想做一些重点强调和补充

### MLA (Multi-Head Latent Attention)

MLA 最早由 DeepSeek-V2 提出，V3 沿用并继续优化了它。

![图片展示了Multi-Head Latent Attention (MLA)架构。输入hidden_state h_t经投影后得到低维latent c_t^Q和c_t^KV，再通过不同权重矩阵恢复各head的Q、K和V。其中，c_t^Q经concatenate和RoPE处理后，与c_t^KV的K部分concatenate，共同参与Multi-Head Attention计算，最终输出hidden_state u_t。该图直观呈现了MLA中latent的生成与应用过程，与上下文对MLA架构的介绍相呼应。](figure-01.png)

MLA 的核心是每个历史 token 的 K/V 不直接以完整多头形式保存，而是以一个低维 latent 表示。"latent" 可以理解为一种“潜在表示“。  
展开来看的话，它在细节上还是有很多不一样的地方。

#### latent: K/V 的低秩联合压缩

首先 latent 的三个特点：

- 低维：latent 的维度小于所有 head 的完整 K/V
- 共享：所有 attention head 从同一个 latent 生成自己的 K/V
- 不可直接解释：它不是某个具体 head 的 key 或 value，而是模型学习出的压缩表示

MLA 把输入 hidden_state 投影到低维的 latent $c_t^{KV}$ 和 $c_t^Q$

$$c_t^{KV}=W^{DKV}h_t, \qquad c_t^Q=W^{DQ}h_t$$

再从共享 latent 恢复各个 head 的 Q, K 和 V

$$[q_{t,1}^{C};\ldots;q_{t,n_h}^{C}]=W^{UQ}c_t^Q$$

$$[k_{t,1}^{C};\ldots;k_{t,n_h}^{C}]=W^{UK}c_t^{KV}$$

$$[v_{t,1}^{C};\ldots;v_{t,n_h}^{C}]=W^{UV}c_t^{KV}$$

#### KV projection absorption

这时候我们会想到，MLA 将 hidden_state 先压缩为 latent(down)，然后展开为完整的 Q/K/V(up)。每次 decode 都需要将保存的历史 latent KV 展开为完整 K/V 的投影，这一步的开销会线性增长。  
省了存储开销，但是 no free lunch，需要更多的计算来弥补。

MLA projection absorption 通过一个简单的数学变化，消掉了投影的主要开销。  
我们先考没有 RoPE 的情况：当前位置 $t$ 对于历史位置 $s$ 的 attention score 可以写成（省略 head 下标）：

$$(q_t^C)^\mathsf T k_s^C=(q_t^C)^\mathsf T W^{UK}c_s^{KV}=\underbrace{\left((W^{UK})^\mathsf T q_t^C\right)^\mathsf T}_{\text{当前 query 只需计算一次}}c_s^{KV}.$$

这就是 key projection absorption：把原本作用于所有历史 latent 的 $W^{UK}$，移到当前 query 上。之后直接和缓存的 $c_s^{KV}$ 做点积。  
它的好处在 decode 时尤其明显。假设已经有 $N$ 个历史 token，现在只生成一个新 token：

- 只缓存 latent、但不做 absorption：需要把 $N$ 个历史 latent 展开成各 head 的 K，再与当前 Q 计算。
- 缓存展开后的 K：可以避免重复展开，但需要保存并读取庞大的多头 K cache。
- 做 absorption：只变换当前 token 的 Q，随后直接读取 $N$ 个小 latent，不需要展开或缓存完整 K。

#### decoupled RoPE

标准的 Full Attention 在输入 hidden state 经过 QKV 投影之后，QK 还要经过 RoPE (Rotary Positional Embadding)，将位置信息嵌入。

decoupled RoPE 解决的是，如何同时保留 RoPE 的位置建模能力，以及 MLA 的 projection absorption，使推理时只读压缩 latent，不必展开各 head 的完整 K/V。

> 上面 projection absorption 的推导，没有考虑对 QK 计算 RoPE 的步骤。加上之后，上面的矩阵变换就不成立了。

decoupled RoPE 是说，Q 和 K 都会分开生成两部分分量（content 分量和 positional 分量），只让较小的 positional 分量承载 RoPE，然后最终拼接在一起。

对于 Q，content Q 和 positional Q 都从 latent $c_t^Q$ 投影生成。

$$[q_{t,1}^{C};\ldots;q_{t,n_h}^{C}]=W^{UQ}c_t^Q,\qquad[q_{t,1}^{R};\ldots;q_{t,n_h}^{R}]=\operatorname{RoPE}(W^{QR}c_t^Q),$$

然后拼接成最终进行 Full-attention 计算的 Q。

$$q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}].$$

K 也是类似的路径，不过 positional K 直接从输入 hidden_state 投影，并且在所有 K Head 之间共享。

$$[k_{t,1}^{C};\ldots;k_{t,n_h}^{C}]=W^{UK}c_t^{KV}, \qquad k_t^R=\operatorname{RoPE}(W^{KR}h_t)$$

然后拼接成最终进行 Full-attention 计算的 K。

$$k_{t,i}=[k_{t,i}^{C};k_t^R]$$

---

### DSA(DeepSeek Sparse Attention)

DSA 在 DeepSeek V3.2 上引入和使用。

![图片展示了DSA（DeepSeek Sparse Attention）架构。输入隐藏状态ht与输出隐藏状态ut通过Multi-Query Attention（Core Attention）处理。其中，QKV做一次Full-Attention，计算量不变。Sparse Attention通过Top-k Selector筛选相关Token，Lightning Indexer对k、w进行部分RoPE操作。该图与上下文紧密相关，直观呈现了DSA架构中多查询注意力机制、Top-k选择器及Lightning索引器等关键组件及其数据流向，帮助理解DSA架构的核心思想。](figure-02.png)

DSA 延续了 MLA 的 latent + decoupled RoPE 结构，然后加入了稀疏选择机制。

> V3.2 从 V3.1-Terminus checkpoint 继续训练，需要继承原有 MLA 参数和架构。

MLA 仍然需要对 QKV 做一次 Full-Attention，计算量并没有变化（其实压缩+展开的投影开销还增加了一点计算）。Sparse Attention 的核心思想是，每个 Query Token 其实并不需要看到所有的历史 Token，可以忽略掉那些毫不相干的 Token，只看到最相关的那些 Token。但是，首先需要对每个 Query Token 筛选出它的 Top-K 个最相关的 Token 出来，然后在做 Core Attention。

#### Lightning indexer

Query token 先进行低秩压缩，再生成少量的 indexer query head:

$$c_t^Q=h_tW^{DQ},\qquad[q_{t,1}^{I};\ldots;q_{t,n_h^I}^{I}]=c_t^QW^{UQ}.$$

对 query token $h_t\in\mathbb{R}^d$ 和历史 token $h_s\in\mathbb{R}^d$，MQA indexer（indexer K 被所有 indexer Q head 共享）计算一次 "QK-dot"，然后对各 head 加权求和得到一个相关性分数：

$$I_{t,s}=\sum_{j=1}^{H^I}w_{t,j}^{I}\cdot\operatorname{ReLU}\!\left((q_{t,j}^{I})^{\mathsf T}k_s^{I}\right).$$

Indexer 的计算量仍然会随着长度线性增长，但是 indexer head 数较少，并可用 FP8 实现，因此它比主 attention 轻得多。

对所有历史位置得到 $I_{t,:}$ 后，只保留 Top-K 对应的 KV entry：

$$\mathcal{S}_t=\{s\mid I_{t,s}\in\operatorname{TopK}(I_{t,:})\}$$

对这些 KV entry, 再展开成完整的 K/V.

$$k_{s,i} = \left[ W_i^{UK}c_s^{KV}; k_s^R \right], \qquad v_{s,i}=W_i^{UV}c_s^{KV}$$

#### Sparse Attention 如何计算？

最后的 Core Attention 是一个 Sparse MQA。 MQA 的好处在于同一个 query token 的所有 Q heads 共享同一组 Top-K 位置。在 MLA 语境下，可以理解为 Q 需要经过压缩(down)，再展开多个 head(down); 而 KV 只经过压缩(down)。

不同 query 的选择可以表示为：

```
原始 KV cache：一份共享存储

query t₀ 的 indices：[1, 5, 9, 12, ...]
query t₁ 的 indices：[0, 5, 8, 15, ...]
query t₂ 的 indices：[2, 6, 9, 14, ...]
```

官方 FlashMLA sparse kernel 接收原始 KV cache 和每个 query 的 Top-K 索引，根据这些索引，从 KV cache gather 对应行到 shared memory。kernel 计算流程：

1. 读取当前 query 的一小段 Top-K 索引。
2. 根据这些索引，从 KV cache gather 对应行到 shared memory。
3. 让该 query 的一组 Q heads 与这块 KV 执行 QK、online softmax 和 PV。
4. 继续处理下一块，累积输出。

---

### CSA(Compressed Sparse Attention)

CSA 继承了 DSA 的稀疏选择机制，并且 KV 压缩/共享的方向不仅在 $n_h \times d_h$ 维度，还在 seq_len 维度。

![该图片展示了Shared Key-Value Multi-Query Attention的架构，核心为CSA（Compressed Sparse Attention）相关组件。输入包含KV Token的隐藏状态与Query Token的隐藏状态，左侧流程中，KV Token的隐藏状态经Token-Level Compressor生成Compressed KV Entries，再经Top-k Selector选出Selected Compressed KV Entries，结合Sliding Window KV Entries通过Concatenation完成处理。右侧Lightning Indexer模块内，Query Token的隐藏状态生成Indexer Queries，Token-Level Compressor处理相关内容得到Compressed Indexer Keys，结合Multi-Query Attention输出的Index Scores，共同完成索引相关计算，最终生成的处理结果反馈至Shared Key-Value Multi-Query Attention模块，体现了CSA架构压缩KV、处理序列维度的设计特点。](figure-03.png)

#### Token-level Compressor

Compressor 把每 $m$ 个 token 的 KV 合并成一个 compressed KV entry. 它的压缩并不是把 $m$ 个 token 简单平均，而是让相邻两段 token 投影交错重叠，然后逐 channel 分别做 softmax 以及加权求和。

##### 1. 计算内容投影和压缩权重

输入 hidden states：

$$H=[h_0,h_1,\ldots,h_{n-1}]\in\mathbb{R}^{n\times d}.$$

CSA 先从每个 hidden state 产生两套候选 KV 表示和两套 compression logits：

$$C^a=HW^{aKV},\qquad C^b=HW^{bKV},\\Z^a=HW^{aZ},\qquad Z^b=HW^{bZ}.$$

$c$ 是 compressed KV entry 的维度。其中：

- $C^a,C^b\in\mathbb{R}^{n\times c}$：两套 content 投影；
- $Z^a,Z^b\in\mathbb{R}^{n\times c}$：对应的压缩权重 logits；

##### 2. 拼接内容矩阵和打分矩阵

我们举一个简单的例子。假设 $m=2$，有 8 个 token：$h_0,h_1,h_2,h_3,h_4,h_5,h_6,h_7$。每两个 token 是一个压缩块。

compressed entry 把当**前块的 a 分量和前一个块的 b 分量合并起来。比如对第二个压缩块，内容矩阵**为：

$$X =\begin{bmatrix}C_0^b \\ C_1^b \\ C_2^a \\ C_3^a\end{bmatrix}\in \mathbb{R}^{8 \times c}.$$

这一块对应的打分矩阵就是把 logits 加上可学习的 bias $B^a$, $B^b$：

$$L=\begin{bmatrix}Z_0^b+B_0^b\\Z_1^b+B_1^b\\Z_2^a+B_2^a\\Z_3^a+B_3^a\end{bmatrix}\in\mathbb{R}^{8\times c}.$$

其中 $B^a,B^b\in\mathbb{R}^{2\times c}$.

##### 3. 每个 channel 独立做 softmax + 加权求和

对 channel $r$，先在这 8 行之间做权重归一化：

$$A_{p,r}=\frac{\exp(L_{p,r})}{\displaystyle\sum_{\ell=0}^{3}\exp(L_{\ell,r})},\qquad p=0,\ldots,3.$$

然后对内容做加权求和：

$$\boxed{C_1^{\mathrm{Comp}}[r] = \sum_{p=0}^{3} A_{p,r}X_{p,r}}$$

对整个压缩快，写成向量形式就是：

$$\boxed{C_1^{\mathrm{Comp}}=\sum_{u=0}^{1}A_{u,:}\odot C_{2+u}^{a} +\sum_{u=0}^{1}A_{4+u,:}\odot C_u^{b}}$$

最终的效果就是将 8 个输入 token $H\in\mathbb{R}^{8\times d}$ 压缩成 4 个 compressed KV entry $C^{\mathrm{Comp}}\in \mathbb{R}^{4\times c}$。

#### Lightning Indexer

CSA 的 indexer 结构和 DSA 是相同的，但是它会在 Compressor 压缩过的 K 上进行计算：

- Query token 还是先低秩压缩，再生成多个 indexer query head:

$$c_t^Q=h_tW^{DQ},\qquad[q_{t,1}^{I};\ldots;q_{t,n_h^I}^{I}]=c_t^QW^{IUQ}.$$

- 历史 token 经过 Compressor 生成 compressed indexer keys $K^{I\mathrm{Comp}}$，$m$ 个 token 压缩成一条 compressed indexer K entry.

Indexer 结构还是一个 MQA Indexer，同样是经过一次 "QK-dot"，然后对各 head 进行加权求和，得到每一个 Query token 对每一个**压缩块历史位置**的相关性打分。

#### Shared-KV MQA

CSA 的 core attention 也是 MQA，所有的 Q head 共享同一份 KV entry；更进一步的共享是 shared-KV，也就是**同一个 entry 同时承担 key 和 value 的作用**。

---

### HCA (Heavily Compressed Attention)

Heavily Compressed Attention（HCA）与 CSA 使用相似的 compressor 和 shared-KV MQA，但走向另一个取舍点：使用远大于 $m$ 的压缩率 $m'$，**不再执行 sparse Top-K**。

它以极少的 compressed entry 提供粗粒度的全局覆盖，CSA 则保留更细的、query-dependent 的选择能力；V4 在不同层交错使用二者。

![HCA：极高压缩率的 global KV 与未压缩 SWA KV](figure-04.png)

#### HCA compressor

HCA 从 hidden states 生成一组 KV entry 与 compression logits：

$$C=HW^{KV},\qquad Z=HW^Z.$$

每个不重叠的 $m'$ token 块独立归一化：

$$S_{m'i:m'(i+1)-1}=\operatorname{Softmax}_{\mathrm{row}}\left(Z_{m'i:m'(i+1)-1}+B\right),$$

$$C_i^{\mathrm{Comp}}=\sum_{j=m'i}^{m'(i+1)-1}S_j\odot C_j.$$

输出序列长度缩小到约 $1/m'$。  
随后使用与 CSA 相同的低秩 query、shared-KV MQA 和 grouped output projection：

$$o_{t,i}=\operatorname{CoreAttn}\left(q_{t,i},C^{\mathrm{Comp}},C^{\mathrm{Comp}}\right).$$

#### 5.2 CSA/HCA 的共同补充设计

##### **Query 与 KV entry normalization**

在 core attention 前，对每个 query head 和唯一的 compressed KV head 分别执行 RMSNorm，防止 attention logit 爆炸并提高训练稳定性。

##### **Partial RoPE**

Query、KV entry 和 core-attention output 只有最后 64 个维度使用 RoPE。由于同一 compressed KV 同时作为 key 和 value，直接加权得到的 output 会携带 absolute position；V4 再对每个输出的最后 64 维施加位置 $-i$ 的 RoPE，使输出重新表达 query 与 KV entry 之间的相对距离。

##### **额外 SWA 分支**

**为严格保持因果性，CSA/HCA 的 query 只能关注已经完成的 compressed block，因此看不到当前 block 内的其他 token**；同时语言模型通常更依赖最近 token。每个 query 因而还读取最近 $n_{\mathrm{win}}$ 个未压缩 SWA KV，并与 compressed KV 拼接后共同执行 core attention。

##### **Attention sink**

每个 head 有可学习 sink logit $z_h'$，其指数项加入 softmax 分母：

$$s_{h,i,j}=\frac{\exp(z_{h,i,j})}{\sum_k\exp(z_{h,i,k})+\exp(z_h')}.$$

这允许一个 query head 分配给真实 KV entry 的总注意力小于 1，甚至接近 0，而不必把概率质量强行分配给不相关位置。

---


## Qwen 技术报告详解

<iframe src="/html/attention-evolution/qwen-attention.html" loading="lazy" title="Qwen Attention 机制演进" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/attention-evolution/qwen-attention.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>


---


## Kimi 技术报告详解

<iframe src="/html/attention-evolution/kimi-attention.html" loading="lazy" title="Kimi Attention 机制演进" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/attention-evolution/kimi-attention.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>


---


## MiniMax 技术报告详解

<iframe src="/html/attention-evolution/minimax-attention.html" loading="lazy" title="MiniMax Attention 机制演进" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/attention-evolution/minimax-attention.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>