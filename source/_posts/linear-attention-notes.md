---
title: Linear Attention 调研笔记
date: 2026-09-07 10:07:23
updated: 2026-09-16 16:38:32
tags: LLM推理
comments: false
excerpt: 从 vanilla linear attention 到 DeltaNet、GDN、KDA 的算法脉络，chunkwise 实现推导，FLA GDN kernel 解析，以及 RTX 5090 上的 prefill 性能实测。
mathjax: true
---

> 更新时间：2026-09-12  
> 
> 范围：先理解算法，再核对当前开源模型中的实际用法。


```Plain Text
linear attention
├── vanilla linear attention: 固定大小历史状态的累加
├── DeltaNet                : 历史状态的“纠错”机制
├── Gated DeltaNet          : 历史状态的“遗忘”机制
└── Kimi Delta Attention    : 更细粒度的“遗忘”机制
```


## 从标准 attention 到固定状态

标准 causal attention 在位置 $t$ 读取所有历史 $k_i,v_i$：

$$y_t = \sum_{i\le t}\operatorname{softmax}(q_tk_i^\top)v_i.$$

![这张图展示了Linear Attention相关计算的存储与数据流设计，涉及SRAM和HBM两种存储。图中橙色区块代表SRAM，绿色区块代表HBM，存在内外两层循环结构，其中Inner Loop沿水平方向、Outer Loop沿垂直方向，用于数据的迭代处理。查询Q_i（橙色）、键K_j（橙色）和值V_j（橙色）存储在SRAM中，数据通过箭头指示的方向在SRAM和HBM间流转，深色、紫色方块则对应注意力计算过程中的数据，体现了减少存储开销的优化思路，与上下文介绍的Linear Attention降低计算量、优化存储的主题相契合。](figure-01.png)

Full attention 需要保留精确的逐 token KV，但计算量随上下文长度平方增长 $\propto O(N^2d）$，KV cache 存储开销随上下文长度线性增长。

> FlashAttention 通过分块和减少 HBM 读写降低实际开销，无需存储完整的注意力分数矩阵，但完整 attention 的二次计算量和不断增长的历史 KV 仍然存在。


**Linear attention 说，可以将历史 KV 汇总到一个大小固定的状态中，用这份状态参与后续计算。** 每当新 token 到来，模型将它的信息写入状态，再用当前 query 从状态中读取所需内容。这样，每一步都只需访问固定大小的记忆，避免反复遍历不断增长的历史 KV。

从自回归推理的角度看，这种计算已经具有 **RNN** 的形式：根据上一步状态和当前输入更新状态，再产生输出。历史信息通过状态传递给后续位置。

![这张图围绕线性注意力对应的类RNN计算逻辑展开，包含RNN展开的序列计算结构、内部单元计算示意图，以及对应说明文本与图例。左侧和中部的展开计算模块标注了步骤t=1、t=2等的计算环节，显示输入token、输入嵌入、循环状态更新、输出线性层等内容，同时用箭头体现计算流向，还标注了跨时间的依赖关系，且明确普通RNN会累积长期错误的问题；右侧展示了RNN cell的内部计算结构，含输入、缩放、权重运算等环节；图例和说明文本则分别对各模块功能、信号传递规则等进行解释，辅助说明线性注意力将历史KV汇总为固定状态、以RNN形式完成计算的核心逻辑。](figure-02.png)

![图片是一张表格，对比了RNN和Transformer在历史信息表示、时间步依赖、训练时的序列并行、长距离信息路径及主要瓶颈等方面的特点。RNN用固定大小的历史表示，时间步依赖性强，训练时需按时间顺序进行，长距离信息路径复杂，主要瓶颈是长程依赖与梯度传播困难。Transformer保留各token表示，可直接读取整个前缀的K/V，同一层内各位置可直接交互，同层中任意两个可见token间通常只需一次Attention，主要瓶颈是Attention计算/显存随序列长度呈二次增长。](figure-03.png)


---

### vanilla linear attention: [Transformers are RNN](https://arxiv.org/abs/2006.16236)

> **把相似度写成特征映射的内积，并利用结合律先累计历史**，替换掉对注意力矩阵 $QK^T$的 softmax 指数运算，使历史 KV 的贡献能够预先累积。


1. **将 softmax attention 写成归一化的相似度加权**

展开前文的 softmax，可以得到：

$$y_t= \frac{ \sum_{i\le t}\exp(q_tk_i^\top/\sqrt{d_k})\,v_i }{ \sum_{i\le t}\exp(q_tk_i^\top/\sqrt{d_k}) }.$$

分子根据 query 与各个 key 的相似度累加 value，分母将这些相似度归一化。论文保留这种加权平均的形式，将指数相似度替换为特征映射后的内积：

$$\operatorname{sim}(q_t,k_i)=\phi(q_t)\phi(k_i)^\top.$$

论文实验采用逐元素的 $\phi(x)=\operatorname{ELU}(x)+1$，使映射后的特征为正。这定义了新的注意力算子，与原来的 softmax attention **并不完全等价**。

1. **利用结合律，将历史信息的累积与当前 query 分开**

替换相似度后：

$$y_t= \frac{ \sum_{i\le t} \phi(q_t)\phi(k_i)^\top v_i }{ \sum_{i\le t} \phi(q_t)\phi(k_i)^\top }.$$

对固定位置 $t$，$\phi(q_t)$ 与求和下标 $i$ 无关，因此可以移到求和外：

$$y_t= \frac{ \phi(q_t)\left(\sum_{i\le t}\phi(k_i)^\top v_i\right) }{ \phi(q_t)\left(\sum_{i\le t}\phi(k_i)^\top\right) }.$$

**括号中的两项都只依赖历史 K/V，与当前 query 无关。** 它们可以提前累积，供不同 query 读取。

1. **用两个状态保存这些累积量**

定义：

<callout emoji="🏖️">
$$S_t=\sum_{i\le t}\phi(k_i)^\top v_i, \qquad z_t=\sum_{i\le t}\phi(k_i)^\top.$$
</callout>

其中，$S_t$ 累积 key 与 value 的关联，$z_t$ 用于计算读出时的归一化分母。如果特征映射后的维度为 $m$，value 维度为 $d_v$，则：

$$S_t\in\mathbb R^{m\times d_v}, \qquad z_t\in\mathbb R^{m\times1}.$$

**两个状态的大小都与历史长度无关**。当前 query 的输出为：

<callout emoji="🏖️">
$$y_t=\frac{\phi(q_t)S_t}{\phi(q_t)z_t}.$$
</callout>

1. **将累积量改写为逐 token 的递推**

从零状态开始，每次只需加入当前 token 的贡献：

<callout emoji="✍️">
$$\boxed{ \begin{aligned} S_0&=0,\qquad z_0=0,\\ S_t&=S_{t-1}+\phi(k_t)^\top v_t,\\ z_t&=z_{t-1}+\phi(k_t)^\top,\\ y_t&=\frac{\phi(q_t)S_t}{\phi(q_t)z_t}. \end{aligned} }$$
</callout>

K/V 负责写入，query 负责读取。先更新状态、再计算输出，对应前文允许访问当前位置的因果范围 $i\le t$。推理时，历史信息通过 $S_t,z_t$ 传递到下一步，无需保留完整的历史 KV 序列。


### 先写一个最小实现

整体流程：

![图片展示了Linear Attention的单头实现流程。首先，将当前token投影与特征映射，得到query、value、key。然后，query、key通过ELU函数加1后，与value进行外积运算，累加到固定状态。接着，query读出并归一化，与更新后的状态相除。该图与上下文紧密相关，直观呈现了从标准attention到固定状态的最小实现过程，是理解Linear Attention工作原理的关键示例。](figure-04.png)

```Python
def linear_recurrence(q, k, v, eps=1e-6):
    # 原始投影结果：q, k: [T, dk]；v: [T, dv]
    q, k = phi(q), phi(k)

    S = np.zeros((v.shape[1], k.shape[1]), dtype=np.float32) # [dv, dk]
    z = np.zeros(k.shape[1], dtype=np.float32)               # [dk]
    out = np.empty_like(v)

    for t, (qt, kt, vt) in enumerate(zip(q, k, v)):
        S += np.outer(vt, kt)               # 累积 KV 关联
        z += kt                             # 累积归一化状态
        out[t] = (S @ qt) / (z @ qt + eps)  # 更新后读取

    return out
```

  
  

---

## DeltaNet：把状态累加改成误差修正

> - [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174): 提出用 delta rule 替代简单累加，纠正记忆中已有的 key–value 关联。
> - [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484): 解决 DeltaNet 沿序列维并行训练的问题，使其能够用于更大规模的语言模型。


vanilla linear attention 的缺陷：

原始 linear attention 将不同 token 的 key–value 关联不断累加到状态中。当它们的 key 相同或相近、而 value 不同时，这些内容会在状态中叠加，读出时可能相互干扰。它没有根据已有记忆调整本次写入的机制（纠错）。

<strong>DeltaNet 在写入前增加了一次读取：先检查状态对当前 key 的预测，再根据预测与目标 value 的差异修改状态。</strong>这样，模型能够修正已有的关联。


1. **读取当前 key 已有的预测**

更新前，用当前 key 读取旧状态：

$$\hat v_t=k_tS_{t-1}.$$

$\hat v_t$ 表示：按照当前记忆，这个 key 应该对应什么 value。随后计算它与当前目标 value 的差异：

$$e_t=v_t-\hat v_t.$$

这里使用 **key** 检查即将修改的关联；当前 query 留到状态更新后，用于产生 attention 输出。

1. **将误差写回状态**

DeltaNet 用误差构造本次写入量：

$$S_t=S_{t-1}+\beta_t k_t^\top e_t.$$

将误差展开：

$$S_t= S_{t-1} -\underbrace{\beta_t k_t^\top\hat v_t}_{\text{移除部分旧预测}} +\underbrace{\beta_t k_t^\top v_t}_{\text{写入当前目标}}.$$

因此，当状态已经能准确预测当前 value 时，误差接近零，更新也接近零；预测偏差越大，需要修正的内容越多。

这可以理解为一次在线 least-squares/fast-weight 更新：如果当前 key 地址已经能读出正确 value，更新很小；如果预测错了，就用误差纠正旧关联。

1. **用更新后的状态回答当前 query**

状态更新完成后，计算单个 head 的输出：

$$y_t=q_tS_t.$$

整个递推可以合并为：

<callout emoji="📍">
$$\boxed{ \begin{aligned} S_0&=0,\\ \hat v_t&=k_tS_{t-1},\\ e_t&=v_t-\hat v_t,\\ S_t&=S_{t-1}+\beta_t k_t^\top e_t,\\ y_t&=q_tS_t. \end{aligned} }$$
</callout>


```Python
def delta_recurrence(q, k, v, beta, eps=1e-6):
    # q, k: [T, dk]；v: [T, dv]，均为原始投影结果。
    # beta: [T]，每步更新强度，通常由模型通过 sigmoid 产生。
    q, k = silu_l2(q, eps), silu_l2(k, eps)
    S = np.zeros((v.shape[1], k.shape[1]), dtype=np.float32)  # [dv, dk]
    out = np.empty_like(v)

    for t, (qt, kt, vt, bt) in enumerate(zip(q, k, v, beta)):
        v_hat = S @ kt                 # 用 key 读取旧预测
        error = vt - v_hat
        S += bt * np.outer(error, kt)  # 写入误差修正
        out[t] = S @ qt                # 用 query 读取更新后的状态

    return out
```


---

## [Gated DeltaNet](https://arxiv.org/abs/2412.06464)：衰减 + delta 写入

DeltaNet 能沿当前 key 修正已有的关联，但它的修改范围受到当前 key 的限制。如果上下文发生变化，大量旧信息已经失去作用，模型缺少直接减弱整份旧状态的控制。

Gated DeltaNet 为此加入一个由输入决定的衰减门：**先减弱旧状态，再执行 delta 更新。** 它将 Mamba-2 式的状态衰减与 DeltaNet 的定向修正结合起来，可以理解为“增加了整体遗忘的能力”。


1. **先衰减旧状态**

当前 token 为每个 head 产生一个标量 $\alpha_t$，控制旧状态的保留比例：

$$\widetilde S_{t-1}=\alpha_tS_{t-1}.$$

$\alpha_t$ 接近 1 时，大部分历史得以保留；接近 0 时，旧状态被大幅减弱。同一个 head 内，整张状态矩阵使用相同的衰减比例。

1. **从衰减后的状态读取旧预测**

用当前 key 读取经过衰减的记忆，再计算它与当前 value 的差异：

$$\hat v_t=k_t\widetilde S_{t-1}, \qquad e_t=v_t-\hat v_t.$$

**误差基于衰减后的状态计算。** 衰减已经改变了记忆中的内容，本次 delta 更新要修正的是衰减之后的预测。

1. **写入修正，再产生输出**

与 DeltaNet 一样，$\beta_t$ 控制误差修正的强度：

$$S_t=\widetilde S_{t-1}+\beta_t k_t^\top e_t.$$

随后，用当前 query 读取更新后的状态：

$$y_t=q_tS_t.$$

将以上步骤合并，得到：

<callout emoji="🌅">
$$\boxed{ \begin{aligned} S_0&=0,\\ \widetilde S_{t-1}&=\alpha_tS_{t-1},\\ e_t&=v_t-k_t\widetilde S_{t-1},\\ S_t&=\widetilde S_{t-1}+\beta_t k_t^\top e_t,\\ y_t&=q_tS_t. \end{aligned} }$$
</callout>

等价的单步更新式为：

$$S_t= \alpha_tS_{t-1} +\beta_t k_t^\top \left(v_t-\alpha_t k_tS_{t-1}\right).$$

这里的 $q_t,k_t,v_t$ 表示进入递推核心的向量，已经完成对应的投影、特征处理和归一化。

这里几个参数的含义：

- $α_t$：forget/decay gate，控制旧记忆保留多少；
- $β_t$：write gate，控制当前误差写入多少；
- $S_t$：每个 head 的固定大小矩阵状态。


```Python
def gated_delta_recurrence(q, k, v, alpha, beta, eps=1e-6):
    # q, k: [T, dk]；v: [T, dv]，均为原始投影结果。
    # alpha, beta: [T]，门控函数产生的保留比例和更新强度。

    q, k = silu_l2(q, eps), silu_l2(k, eps)
    S = np.zeros((v.shape[1], k.shape[1]), dtype=np.float32)  # [dv, dk]
    out = np.empty_like(v)

    for t, (qt, kt, vt, at, bt) in enumerate(zip(q, k, v, alpha, beta)):
        S *= at                        # 先衰减整个旧状态
        v_hat = S @ kt                 # 从衰减后的状态读取旧预测
        error = vt - v_hat
        S += bt * np.outer(error, kt)  # 写入误差修正
        out[t] = S @ qt                # 用 query 读取更新后的状态

    return out
```


---

## [Kimi Delta Attention](https://arxiv.org/abs/2510.26692)（KDA）

> [technical report](https://yzhang.site/assets/pubs/techreport/2025/kda.pdf)

Gated DeltaNet 已经能够同时进行整体遗忘和 delta 修正，但它在同一个 head 内只使用一个衰减率。整张状态矩阵一起缩放，意味着不同通道中的信息只能按相同的比例保留。

Kimi Delta Attention（KDA）进一步细化了这一控制：**为同一个 head 内的每个 key 特征通道产生单独的衰减率。** 它保留 GDN 的 delta 更新方式，让不同通道能够以不同速度遗忘历史。论文摘要明确称 KDA “extends Gated DeltaNet with a finer-grained gating mechanism”。


1. **将标量衰减门扩展为向量**

GDN 为每个 token、每个 head 产生一个标量 $\alpha_t$。KDA 将其扩展为：

$$\boldsymbol{\alpha}_t = [\alpha_{t,1},\ldots,\alpha_{t,d_k}] \in[0,1]^{d_k}.$$

更新强度 $\beta_t$ 仍是每个 head 的标量。两者均由当前输入通过可学习的门控计算产生。**沿 key 特征维度进行通道级衰减，配合标量 $\beta_t$ 控制 delta 修正。**

1. **按通道衰减旧状态**

将衰减向量写成对角矩阵：

$$D_t=\operatorname{Diag}(\boldsymbol{\alpha}_t),$$

则旧状态的衰减为：

$$\widetilde S_{t-1}=D_tS_{t-1}.$$

在当前 $S:[d_k,d_v]$ 的布局下，每一行对应一个 key 特征通道，因此：

$$\widetilde S_{t-1}[j,:] = \alpha_{t,j}S_{t-1}[j,:].$$

例如，某个通道的衰减率接近 1，可以较完整地保留历史；另一个通道的衰减率接近 0，则会大幅减弱该通道的旧内容。这些通道是学习到的特征坐标，并不直接对应某一条独立事实。

1. **从衰减后的状态计算误差，再执行 delta 更新**

后续步骤与 GDN 相同：

$$\hat v_t=k_t\widetilde S_{t-1}, \qquad e_t=v_t-\hat v_t,$$

$$S_t=\widetilde S_{t-1}+\beta_tk_t^\top e_t, \qquad y_t=q_tS_t.$$

从零状态开始，完整的单步递推为：

<callout emoji="🏖️">
$$\boxed{ \begin{aligned} S_0&=0,\\ D_t&=\operatorname{Diag}(\boldsymbol{\alpha}_t),\\ \widetilde S_{t-1}&=D_tS_{t-1},\\ e_t&=v_t-k_t\widetilde S_{t-1},\\ S_t&=\widetilde S_{t-1}+\beta_tk_t^\top e_t,\\ y_t&=q_tS_t. \end{aligned} }$$
</callout>

这里的 Q/K/V 指进入递推核心、完成前置处理后的向量。展开更新式，可得到与论文式（1）对应的形式：

$$S_t= \left(I-\beta_tk_t^\top k_t\right)D_tS_{t-1} +\beta_tk_t^\top v_t.$$

顺序仍是先衰减，再纠错。$D_t$ 与 $I-\beta_tk_t^\top k_t$ 一般不能交换位置。


```Python
def kda_recurrence(q, k, v, alpha, beta, eps=1e-6):
    # 沿用前例：q/k 在函数内做 SiLU + L2；v 直接作为写入目标。
    # q, k: [T, dk]；v: [T, dv]
    # alpha: [T, dk]，逐 key 通道的保留比例；beta: [T]。

    q, k = silu_l2(q, eps), silu_l2(k, eps)
    S = np.zeros((v.shape[1], k.shape[1]), dtype=np.float32)  # [dv, dk]
    out = np.empty_like(v)

    for t, (qt, kt, vt, at, bt) in enumerate(zip(q, k, v, alpha, beta)):
        S *= at[None, :]               # 每一列对应一个 key 通道
        v_hat = S @ kt                 # 从衰减后的状态读取旧预测
        error = vt - v_hat
        S += bt * np.outer(error, kt)  # 写入误差修正
        out[t] = S @ qt                # 读取更新后的状态

    return out
```


---


> 下面我们来看 Linear Attention 的 sota 实现，主要是 GDN 和 KDA kernel


## 从逐 token 算法到 Chunkwise 实现

Linear Attention 的算法看起来是一个“逐 token”串行的过程，但是它对历史状态的更新具有线性结构，可以展开递推，将多个 token 的计算合并为矩阵运算。（类比先行进位加法器）

在 prefill 中，当前层的整段输入已经给定，可以提前得到各个位置的 Q/K/V。将序列划分为若干个 chunk 后，每个 chunk 的输出可以拆成两部分：**读取进入该块的历史状态，以及计算当前块内的贡献。**

我们下面以 Gated DeltaNet 为例，推导 Chunkwise 计算公式。

> 接下来全是数学公式的推导，慎入！


1. **将每一步的纠错写成状态增量**

只考虑一个 head，沿用前文的行向量约定：

$$q_t,k_t\in\mathbb R^{1\times d_k},\qquad v_t\in\mathbb R^{1\times d_v},\qquad S_t\in\mathbb R^{d_k\times d_v}.$$

这里的 Q/K/V 已完成进入递推核心之前的处理。根据上面的算法，GDN 先衰减旧状态，再根据当前 key 的预测误差修正状态：

$$S_t=\alpha_tS_{t-1} +\beta_tk_t^\top\left(v_t-\alpha_tk_tS_{t-1}\right).$$

把经过 $\beta_t$ 缩放的误差记为 $r_t$：

<callout emoji="✍️">
$$\boxed{ \begin{aligned} r_t&=\beta_t\left(v_t-\alpha_tk_tS_{t-1}\right),\\ S_t&=\alpha_tS_{t-1}+k_t^\top r_t,\\ y_t&=q_tS_t. \end{aligned} }$$
</callout>

<strong>$r_t$ 是本次写入的修正向量。</strong>它依赖旧状态，因此不能直接将所有 $r_t$ 当成已知输入；Chunkwise 需要先处理这部分依赖。


1. **展开状态，分开块外历史与块内写入**

取一个包含 $C$ 个 token 的 chunk，块内位置编号为 $1,\ldots,C$，进入该块的状态记为 $S_{\mathrm{in}}$。

定义**累计衰减**：

$$\gamma_i=\prod_{t=1}^{i}\alpha_t.$$

它表示进入该块之前的历史，到位置 $i$ 时还保留多少。对于块内位置 $j$ 的写入，定义：

$$D_{ij}= \begin{cases} \displaystyle\prod_{t=j+1}^{i}\alpha_t,&j\le i,\\ 0,&j>i. \end{cases}$$

其中 $D_{ii}=1$：当前位置的写入尚未经历后续衰减。位置 $j$ 的写入到达位置 $i$ 时，则需要经过 $j+1,\ldots,i$ 的衰减。

展开状态递推：

<callout emoji="📚">
$$\boxed{ S_i= \underbrace{\gamma_iS_{\mathrm{in}}}_{\text{衰减后的块外历史}} + \underbrace{\sum_{j=1}^{i}D_{ij}k_j^\top r_j}_{\text{衰减后的块内写入}} }$$
</callout>

接下来，将 $S_{i-1}$ 的展开式代入 $r_i$：

<callout emoji="📍">
$$\begin{aligned}r_i&=\beta_i\left(v_i-\alpha_i k_iS_{i-1}\right)\\&\Rightarrow\boxed{r_i=\underbrace{\beta_i v_i}_{\text{当前 value}}-\underbrace{\beta_i\gamma_i k_iS_{\mathrm{in}}}_{\text{块外历史对当前 key 的预测}}-\underbrace{\sum_{j<i}\beta_iD_{ij}(k_i k_j^\top)r_j}_{\text{块内前序写入对当前 key 的预测}}}\end{aligned}$$
</callout>

**这样，对整张历史状态的依赖，就被改写为对块内前序修正向量 $r_j$ 的依赖。其系数只由当前块的 K 和门控参数决定。**


1. **将块内纠错关系整理成下三角方程组**

按行堆叠块内向量：

$$Q,K\in\mathbb R^{C\times d_k},\qquad V,R\in\mathbb R^{C\times d_v}.$$

<strong>其中，$R$ 的第 $i$ 行就是 $r_i$。</strong>定义严格下三角矩阵 $L$：

$$L_{ij}= \begin{cases} \beta_iD_{ij}(k_i k_j^\top),&j<i,\\ 0,&j\ge i. \end{cases}$$

将上一节的 $r_i$ 递推展开式的求和项移到左边：

$$\boxed{ r_i+\sum_{j<i}\beta_iD_{ij}(k_i k_j^\top)r_j = \beta_i v_i-\beta_i\gamma_i k_iS_{\mathrm{in}}. }$$

这个式子写成矩阵形式，等价于：

$$\boxed{ (I+L)R = \operatorname{Diag}(\beta)V - \operatorname{Diag}(\beta\odot\gamma)KS_{\mathrm{in}} }$$

$\odot$ 表示逐元素乘法。

由于 $L$ 严格下三角，$I+L$ 的对角线全为 1，这个方程组可以进行三角求解。

右侧只有 $S_{\mathrm{in}}$ 来自块外，因此可以先求出两个**与输入状态无关的矩阵** $U,W$：

$$\begin{aligned} (I+L)U&=\operatorname{Diag}(\beta)V,\\ (I+L)W&=\operatorname{Diag}(\beta\odot\gamma)K. \end{aligned}$$

其中 $U\in\mathbb R^{C\times d_v}$，$W\in\mathbb R^{C\times d_k}$。等输入状态确定后，整块修正量为：

$$\boxed{R=U-WS_{\mathrm{in}}.} \tag{1}$$

$U,W$ 汇总了块内的纠错关系，**不同 chunk 可以分别提前计算。**

**下三角求解仍然包含行与行之间的依赖。这个改写将依赖集中到块内的小矩阵运算中，使后续涉及整张状态的计算能够组织为矩阵乘法。**


1. **用矩阵乘法计算输出和最终状态**

将展开后的状态代入 $y_i=q_iS_i$：

$$y_i =\gamma_iq_iS_{\mathrm{in}} +\sum_{j\le i}D_{ij}(q_i k_j^\top)r_j.$$

整块输出为：

$$\boxed{ Y= \underbrace{\operatorname{Diag}(\gamma)QS_{\mathrm{in}}}_{\text{读取块外历史}} + \underbrace{\left((QK^\top)\odot D\right)R}_{\text{读取块内修正}} }\tag{2}$$

**矩阵 $D$ 同时表达了衰减和因果范围**：**上三角为零，当前位置只能读取本块中不晚于自己的写入。**

块结束时，传给下一个 chunk 的状态是：

$$\boxed{ S_{\mathrm{out}} = \gamma_C S_{\mathrm{in}} + K^\top \operatorname{Diag}(D_{C1},\ldots,D_{CC})R. }\tag{3}$$

由此，一个 chunk 的计算可以分为：**先处理块内的衰减和纠错关系，得到 $U,W$；输入状态确定后，计算 $R$、输出和最终状态。计算过程无需逐个构造块内所有 $S_i$。**


1. **Chunkwise 并行的范围与开销**

不同 chunk 的累计衰减、块内矩阵以及 $U,W$ 可以并行准备。块边界的状态通常沿序列传递；各块的输入状态确定后，输出计算也可以并行展开。

这种方式引入了 $C\times C$ 的块内矩阵、三角运算和临时存储，同时将大量计算组织成矩阵乘法。固定 chunk 大小 $C$ 后，每块的计算规模固定，总工作量仍随序列长度线性增长。

**Chunk 大小会影响块内计算量、寄存器和片上存储需求，也会影响并行度。实际性能取决于这些开销能否被矩阵计算效率和状态复用所抵消。** 


1. **与先行进位加法器的类比**

> 到这里应该很容易看懵了吧！没关系，还记得先行进位加法器吗？我们把结构图画出来，就会发现这里的联系了！

先行进位加法器可以提前计算一组 bit 的生成信号 $G$ 和传播信号 $P$，再根据输入进位得到：

$$c_{\mathrm{out}}=G\lor(P\land c_{\mathrm{in}}).$$

**组内的 $G,P$ 只依赖本组输入，不必等待低位进位到达。**

![图片展示了块内并行块间串行（16位加法器）的计算流程。图中有四个并行块，每个块内有4位，块间通过串行方式连接。每个块内有生成信号$g$和传播信号$p$，如$p_{3-0}$、$g_{3-0}$等，块间有进位信号$c$，如$c_{16}$、$c_{15-13}$等。箭头表示信号传递方向，如$p_{3-0}$到$p_{11-8}$、$g_{3-0}$到$g_{11-8}$等。该图与上下文介绍的先行进位加法器提前计算生成和传播信号，再根据输入进位得到$c_{\\mathrm{out}}$的内容相关，直观呈现了这一计算过程。](figure-05.png)

GDN 的 chunk 也可以提前整理本块对输入状态的作用。将上面 $(1)$$R=U-WS_{\mathrm{in}}$ 代入 $(3)$，可将其归纳为：

$$S_{\mathrm{out}} =A_{\mathrm{chunk}}S_{\mathrm{in}}+B_{\mathrm{chunk}}.$$

$A_{\mathrm{chunk}}$ 描述本块如何衰减、修改已有状态，$B_{\mathrm{chunk}}$ 描述本块产生的状态贡献。**两者都由块内输入决定，可以在输入状态到达之前准备。**

因此，<strong>“块内预处理、块间传递状态”可以类比“组内先行进位、组间传递进位”。</strong>两者都通过提前计算一段操作对输入的作用，缩短逐步传播的依赖链。

![图片展示了GDN Chunkwise并行计算流程。从输入计算、状态准备、块内预处理、序列切片等步骤，逐步进行。输入计算部分，有各项独立输入和块内输入计算；状态准备部分，有块输出、块修正与状态、块间传递状态等；块内预处理部分，有块内输入计算、块内修正与状态、块内状态准备等；序列切片部分，有块内输入计算、块内修正与状态、块内状态准备等。该图与上下文紧密相关，直观呈现了GDN Chunkwise并行计算的各环节及流程。](figure-06.png)


---


## 现有算子实现

![图片展示了推理框架与算子后端的架构图。左侧为推理框架，包含Flash Linear Attention、FlashQLA、TileGym、llama.cpp、GMLM等算子，其中Flash Linear Attention被红色箭头指向。右侧为算子后端，有FreeBasis、SQLang、VLLM、TeraBERT-LLM、FlashRefiner等，其中FreeBasis被蓝色箭头指向。图中还标注了部分算子的实现库，如Flash Linear Attention使用CUDA等。该图与上下文介绍的LA算子实现库相关，直观呈现了算子在推理框架中的位置及后端实现情况。](figure-07.png)

目前 LA 算子的实现库主要有这些：

- [**Flash Linear Attention（FLA）**](https://github.com/fla-org/flash-linear-attention)：覆盖多种线性注意力算法的通用算子库，包括 DeltaNet、GDN 和 KDA。核心实现以 **Triton** 为主，提供分块计算、逐 token 递推及训练所需的反向算子。图中多个框架移植了它的源码；上游 FLA 也接入了 FlashQLA、FlashKDA 等可选后端。
- [**FlashQLA**](https://github.com/QwenLM/FlashQLA)：Qwen 团队基于 **TileLang** 开发的线性注意力算子库，重点优化 **GDN 的 Chunked Prefill**，包含前向和反向实现。通过算子融合、并行策略和数据搬运优化提高效率，也可以通过 FLA 接口调用。
- [**FlashKDA**](https://github.com/MoonshotAI/FlashKDA)：Moonshot AI 针对 **KDA Prefill** 开发的专用算子库，使用 **CUDA C++，结合 CUTLASS / CuTe C++ 模板库**实现融合 kernel。它是独立于 FLA Triton 代码的实现，可作为 FLA 后端；vLLM 则通过源码集成使用其衍生版本。
- [**FlashInfer**](https://github.com/flashinfer-ai/flashinfer)：面向大模型推理服务的综合 GPU 算子库，除常规 Attention 外，也提供 **GDN、KDA 的 Prefill / Decode 算子**。图中相关实现主要使用 **CUDA 和 CuTe DSL**，由推理框架根据模型、阶段及硬件条件选择调用。
- [**TileGym**](https://github.com/NVIDIA/TileGym)：NVIDIA 维护的 CUDA Tile 算子库与示例集合。其中也有独立维护的 **cuTile GDN 实现**，包含分块 Prefill 和递推 Decode，适合研究 cuTile 如何表达和优化线性注意力。（由于偏教学性质，不保证 sota 性能，所以前端框架基本没有接入）
- [**GGML**](https://github.com/ggml-org/ggml)：llama.cpp 使用的底层张量计算库。其线性注意力通过组合计算图或专用融合算子执行，并分别在 **CPU、CUDA、Metal** 等后端实现，支持图中的 GDN / KDA 路径。
- **框架内置 CUDA / CuTe DSL 算子。**


---


## FLA GDN kernel (FreeToken revised version)


本节整理 FLA GDN 的 **prefill 路径**。**FreeToken 的 GDN kernel 基于 SGLang 引入并改造的 FLA Triton 实现，以源码内置的方式接入 FreeToken，并进行了状态管理和运行时适配。**

我们下面分析的 FLA GDN kernel 就是用的 FreeToken 源码对照的。（~~这是因为本人最近花了很多时间熟悉 FreeToken 代码，但是又不想重新去看 FLA~~）

> FreeToken 内置并适配了源自 FLA/SGLang 的 GDN Triton 实现，我花了一整个下午过了一遍源代码。我的评价是，这是我见过的最恶心的 kernel。。。  
> 所以我们下面不可能把里面的实现细节展开来讲，而是保持一个算法宏观的视角，介绍这么复杂的一个 kernel 是怎么拆分的。

**总体来说，FLA 的 GDN 实现将 chunkwise 计算拆成累计衰减、块内纠错变换、状态递推和输出计算；前后还包括输入投影、因果卷积、归一化和输出投影。**


![图片展示了FLA GDN的Chunkwise并行计算流程。从输入X开始，经投影、卷积、归一化等操作，进入Chunkwise计算，分为累计衰减、块内纠错变换、状态递推和输出计算等步骤。每个Chunk独立计算，涉及Diag、GEMM、SGL等操作，最终输出状态和状态递推。该图与文档中对FLA GDN实现的描述相呼应，直观呈现了其计算过程。](figure-08.png)

模型入口是 `Qwen3_5GatedDeltaNet.forward()`，chunk 主调用链位于 `chunk_gated_delta_rule_fwd()`。各步骤不一定对应一个独立的 Triton kernel：投影是 GEMM op，卷积可以走 `sgl_kernel` 或 Triton，部分门控通过 PyTorch element-wise 操作完成。

沿用前文的符号，**chunk 大小记为 $C$，当前核心实现取 $C=64$**。块内公式以一个 V head 为单位，使用它对应的 Q/K head；$N$ 表示本次 batch 的 token 总数。


让 agent 整理了 GDN prefill 路径下，每一个 triton kernel 的操作：

<iframe src="/html/linear-attention/gdn-chunkwise-demo.html" loading="lazy" title="GDN prefill 各 triton kernel 操作整理" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/linear-attention/gdn-chunkwise-demo.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>


下面我们来看每一个 kernel：

1. **输入投影：用 GEMM 生成四路输入**

输入 X 的 shape 为 `[N, hidden_size]`。普通路径把几组投影权重沿输出维合并，一次 GEMM 得到：

$$P=XW_{\mathrm{in}}^\top, \qquad [\mathrm{conv\_in}\mid z\mid b\mid a] =\operatorname{split}(P).$$

`conv_in` 包含 Q/K/V 的原始投影；`a,b` 用于生成状态门控；`z` 绕过递推核心，留给输出端的 gated RMSNorm。这里的 split 沿特征维进行，尚未沿 token 维切 chunk。


1. **因果卷积：读取短历史窗口，融合 SiLU**

`conv_in` 先经过 depthwise causal convolution。每个特征通道单独计算，沿 token 维读取当前值和最近的历史值。以卷积宽度为 4 为例，对一个通道：

$$\mathrm{mixed}_t=\operatorname{SiLU} (w_0x_{t-3}+w_1x_{t-2}+w_2x_{t-1}+w_3x_t).$$

这里 x 是卷积前的投影值。若状态池中已有 `[x1,x2,x3]`，本次新增 `[x4,x5]`，则：

```Plain Text
更新前的窗口：   [x1, x2, x3]
x4 的卷积窗口：[x1, x2, x3, x4]
x5 的卷积窗口：[x2, x3, x4, x5]
更新后的窗口：   [x3, x4, x5]
```

窗口保存在 `pool.conv_states`，新请求缺少的历史补零。不同请求通过序列边界和槽位索引隔离；窗口保存的是原始投影值，而不是 SiLU 后的结果。

卷积与 SiLU 在同一个算子内完成。`mixed` 随后沿特征维拆成 Q/K/V，再展开 head 维。


1. **门控与 Q/K 归一化：element-wise 操作和 head 内归约**

门控分支直接使用投影得到的 a、b：

$$\beta=\sigma(b),\qquad g=-\exp(A_{\log})\operatorname{softplus}(a+dt_{\mathrm{bias}}).$$

其中 g 是逐 token 的 log-decay，$\alpha=\exp(g)$。这些操作按 token、V head 独立计算，不经过因果卷积。


1. **`chunk_local_cumsum`：生成块内累计衰减**

每个 chunk、每个 V head 独立进行前缀和，将逐 token 的 $g_i=\log\alpha_i$ 转成：

$$G_i=\sum_{t=1}^{i}g_t=\log\gamma_i.$$

为了展示边界，假设 $C=4$：

```Plain Text
输入：[g1, g2, g3, g4 | g5, g6, g7, g8]

输出：[g1, 
      g1+g2, 
      g1+g2+g3, 
      g1+g2+g3+g4
     |g5, 
      g5+g6, 
      g5+g6+g7, 
      g5+g6+g7+g8]
```

累计在每个 chunk 开头重新开始。kernel 用 `tl.cumsum` 计算，输出仍以 log 形式保存；后续通过 $\exp(G_i)$ 得到 $\gamma_i$，在 $j\le i$ 的因果范围内通过 $\exp(G_i-G_j)$ 得到 $D_{ij}$。


1. **`chunk_gated_delta_rule_fwd_kkt_solve_kernel`：构造 L 并求逆**

这一 kernel 根据 K、累计衰减和 β，计算：

$$L_{ij}=\beta_i\exp(G_i-G_j)(k_i k_j^\top)\quad(j<i), \qquad A=(I+L)^{-1}.$$

这里 $A\in\mathbb R^{C\times C}$ 是块内逆矩阵，与前文描述状态传播的 $A_{\mathrm{chunk}}$ 不同。

一个 program 负责一个 chunk、一个 V head。64 个 token 分成四组，每组 16 个，形成 16×16 子块；只计算下三角覆盖的 **4 个对角子块和 6 个非对角子块**。点积沿 key 特征维分片累加，再乘衰减、β，并施加严格下三角 mask。

$$L= \begin{bmatrix} L_{00}&0&0&0\\ L_{10}&L_{11}&0&0\\ L_{20}&L_{21}&L_{22}&0\\ L_{30}&L_{31}&L_{32}&L_{33} \end{bmatrix}.$$

求逆也按子块进行，分成两步来做：(记 $J=I+L$)

1. 四个对角子块（diagonal block）$J_{00},J_{11}, J_{22}, J_{33}$，通过逐行递推求逆；（三角矩阵求逆）
2. 剩下的非对角子块（off-diagonal block)，再用矩阵乘法合并。例如：

$$A_{10}=-A_{11}L_{10}A_{00}, \qquad A_{ii}=(I+L_{ii})^{-1}.$$

构造 L 和求逆融合在同一个 kernel 内，省掉中间 L 的显存写回与重读；结果 A 写回后供下一步使用。


1. **`recompute_w_u_fwd`：行缩放加矩阵乘法，生成 U/W**

有了 A，两组下三角方程的解变成：

$$U=A\operatorname{Diag}(\beta)V, \qquad W=A\operatorname{Diag}(\beta\odot\gamma)K.$$

kernel 加载 A 后，直接对 K/V 做广播乘法，将缩放后的 tile 送入 `tl.dot`，无需构造对角矩阵或单独保存缩放结果。

同一个 program 处理一个 chunk、一个 V head 的 U/W，并沿输出特征列分块。例如 $d_k=d_v=128$ 时，U 和 W 各拆成两个 64 列的 tile，共进行四次 64×64 矩阵乘法。这些 tile 写入不同输出列，循环之间无需累加。

1. **`chunk_gated_delta_rule_fwd_h`：带着状态依次处理各个 chunk**

这一阶段对应前文式（1）和式（3）：

$$\begin{aligned}R &= U - WS_{\mathrm{in}}, \\S_{\mathrm{out}} &= \gamma_C S_{\mathrm{in}}+ K^\top \operatorname{Diag}(D_{C,:})R.\end{aligned}$$

并行维度改成“请求 × V head × value 通道片”。一个 program 持有自己负责的状态片，在内部循环中依次处理当前请求的所有 chunk。初始状态从 `pool.recurrent_states` 读取，全部 chunk 结束后将末状态写回同一槽。

每轮先保存入口状态 h，再计算并保存 `v_new=R`；随后给局部 R 乘到块末端的衰减，更新状态。下一轮直接沿用局部状态，不从 h 快照重新加载。h 和未做末端衰减的 R 会交给后面的输出 kernel。


1. **`chunk_fwd_o`：合并块外历史与块内修正**

对应前文式（2），代码另外传入读出缩放 $s=1/\sqrt{d_k}$：

$$Y=s\left[ \operatorname{Diag}(\gamma)QS_{\mathrm{in}} +\left((QK^\top)\odot D\right)R \right].$$

一个 program 处理一个 chunk、一个 V head 的部分输出通道。它分别计算 $QS_{\mathrm{in}}$ 和 $QK^\top$，加入衰减后，将后者乘 R，最后相加、缩放并写出结果。各 chunk 的入口状态已经就绪，因此输出计算可沿 chunk 维并行。

这里的因果 mask **包含对角线**，因为当前 token 的输出读取更新后的状态；构造 L 时则使用严格下三角。不同 value 通道片的 program 会重复计算同一份 $QK^\top$，这是当前分工中一处明确的重复计算。


1. **gated RMSNorm 与输出投影：从各 head 的 Y 回到 hidden states**

各 chunk/head 的输出按 token 位置写入同一个 `[N, Hv, dv]` Tensor。随后逐 token、逐 V head 计算：

$$\widehat Y_{t,h} =\operatorname{RMSNorm}_{d_v}(Y_{t,h}) \odot\operatorname{SiLU}(z_{t,h}).$$

RMSNorm 与 z 的门控乘法融合在一个 Triton kernel 中，归一化只沿该 head 的 $d_v$ 维进行。结果 reshape 为 `[N, Hv*dv]`，再通过输出 GEMM：

$$X_{\mathrm{out}} =\operatorname{reshape}(\widehat Y)W_o^\top \in\mathbb R^{N\times\mathrm{hidden\_size}}.$$

到这里得到 `Qwen3_5GatedDeltaNet.forward()` 的返回值；残差相加由外层 decoder layer 处理。


---


## 公开资料核实的代表模型

| 模型 | 线性/固定状态模块 | 整体架构 | 当前应如何理解 |
|-|-|-|-|
| Qwen3-Next-80B-A3B | Gated DeltaNet | 与 Gated Attention 混合；公开 Transformers 文档明确写出该组合 | GDN 进入主流 MoE 模型的代表案例：[HF Transformers 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_next.md) |
| Kimi Linear 48B-A3B | Kimi Delta Attention (KDA) | 约 3:1 KDA : MLA 的混合设计 | GDN 的细粒度门控后继，重点看有限状态记忆和硬件效率：[论文](https://arxiv.org/abs/2510.26692) |
| Kimi K3 | Kimi Delta Attention (KDA) | KDA 与 Attention Residuals；官方页面称为混合线性注意力架构 | 确认使用 KDA；官方公开页面未在本文写死完整层间比例，拿到权重后再以 `config.json`/源码核实：[Kimi K3 官方介绍](https://www.kimi.com/news/kimi-k3)、[开放模型页](https://www.kimi.ai/ai-models/kimi-k3) |
| NVIDIA Nemotron-H / Nemotron Nano | Mamba-2 selective SSM | Mamba-2 与标准 attention 交错 | 是相邻的固定状态路线，不是 DeltaNet/GDN；适合比较状态大小、scan、长上下文吞吐：[NVIDIA 文档](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/nemotron-h) |
| Jamba 1.5 | Mamba/SSM | SSM、attention、MoE 混合 | 早期重要 hybrid SSM-Transformer 基线，不归入狭义 GDN/KDA：[论文](https://arxiv.org/abs/2408.12570) |

截至本文更新时间，最值得优先深入的是 **Qwen3.6/Qwen3-Next 的 GDN** 和 **Kimi Linear/K3 的 KDA**；Nemotron-H、Jamba 用来建立 SSM 邻域和系统对照。模型是否“用了 linear attention”必须看具体模块定义，不能仅凭“hybrid”“long context”宣传语判断。


---


## 性能测试

我们上面拆解了 FLA 的 GDN Triton 实现（Prefill 路径）。应该可以明显感受到 FLA 相对于 FA 优化的难度（一个是单个 Triton kernel，一个被拆分成多个 Triton kernel，每个 kernel 的并行度都达不到 FA 的程度）。

下面也是在 RTX5090 上测量一下 **FLA, FlashQLA, FlashInfer** 这几个算子库中 GDN Prefill kernel 的性能 benchmark。有意思的是这三个库的 GDN kernel 分别使用 **Triton, TileLang, CUDA+CuTe DSL**，所以算子的真实性能不仅仅靠 kernel 优化，还有一整套编译软件栈的积累。

- 我们测量 **GDN 分块前向算子整体**的耗时，包含多个 kernel，不含 projection、卷积和首次编译。config：`B=1，H=HV=16，K=V=128`，BF16。

| Seq_len | FLA（Triton） | FlashQLA（TileLang） | FlashInfer（CuTe DSL） |
| --- | --- | --- | --- |
| 512 | 53.05 | 35.14 | 63.95 |
| 1024 | 81.30 | 53.77 | 89.26 |
| 2048 | 141.01 | 77.03 | 170.25 |
| 4096 | 282.50 | 159.50 | 218.42 |
| 8192 | 607.36 | 272.13 | 383.92 |


![这张折线图展示了RTX 5090上不同Linear Attention实现方案的GDN预填充延迟数据，测试条件为B=1、Hq=Hk=Hv=16、K=V=128、BF16 Q/K/V，纵轴为GDN预填充运行时长（单位微秒），横轴为序列长度（token数），序列长度范围为512到8192。图中包含FLA（Triton）、FlashQLA（TileLang）、FlashInfer三个方案，每个方案分别提供两种实现的延迟，其中实线、实心标记对应CUDA Graph，虚线、空心标记对应Eager forward。随着序列长度增加，所有方案的预填充延迟均上升，FlashQLA的延迟始终低于另外两种方案，FLA的延迟始终高于其他两种方案。](figure-09.png)

然后我们对比了一下 cuDNN 的 causal attention：

![这张图片是在RTX 5090显卡上，对GDN注意力机制与因果全注意力做预填充时延对比的折线图。图表的横轴为输入序列长度，范围是512到8192个token，纵轴为预填充运行时间，单位为微秒。图中不同线条对应不同实现方式，包含FLA、FlashQLA、FlashInfer及cuDNN相关实现，其中cuDNN全注意力的CUDA Graph实现，随着序列长度增加，运行时增长幅度最大且最终时延最高，在序列长度为8192时达到约1500微秒，其余各类实现的运行时均远低于该数值。](figure-10.png)

<iframe src="/html/linear-attention/gdn-prefill-timing-report.html" loading="lazy" title="GDN prefill timing report" style="width:100%;height:80vh;min-height:560px;border:1px solid #d0d7de;border-radius:8px;background:#fff;"></iframe>
<p style="text-align:right;font-size:0.85em;margin-top:0.2em;"><a href="/html/linear-attention/gdn-prefill-timing-report.html" target="_blank" rel="noopener">在新窗口打开完整页面 ↗</a></p>


这个结果其实有一些反直觉。因为从算法上看，causal attention 比 LA 简洁，LA 多出来那么多并行度低的 non-matmul 操作，还拆成多个 kernel，是怎么做到比 causal attention 更快的？

按理说，对于更长的序列，LA chunk 之间的状态依赖链会更长，延迟会更明显的显现出来成为瓶颈。怎么解释长序列场景下，causal-attention 比 LA 慢这么多？


首先，GPU 利用率更高、kernel 数量更少，并不一定意味着总耗时更短。

我们知道 causal attention 的计算开销$\propto \mathcal{O}(N^2)$，LA/GDN 在固定 head dimension、chunk size 时，计算量随 $N$ 近似线性增长，计算开销可以认为 $\propto \mathcal{O}(N)$。固定 head dimension 和 chunk size，可以粗略写成：

$$T_{\text{Full}}\approx aN^2+L_{\text{Full}}, \qquad T_{\text{GDN}}\approx bN+L_{\text{GDN}}.\tag{4}$$

- Full Attention 没有跨 query 的状态依赖，却要完成更多计算，有限的 SM 需要分多轮处理这些工作。
- GDN 可能受较长的状态依赖链限制；但它仍然能够长序列时胜出，因为 Full Attention 需要处理越来越多的 token 对。

在 RTX5090 这张卡上，seq_len 在 8192 的时候 grid-size 大概率已经等于 SM 数量（persistent kernel），也就是所有的计算任务均匀的摊在所有的硬件单元上。我们不能说这里已经是 compute-bound，但这里的关键是增长速度。

- Full-attention 在 compute-bound 到来之前，也许可以保持线性增长；一旦达到 compute-bound，那么它的延迟就会变成 $\mathcal{O}(N^2)$，会以二次函数的速度增长; 
- Linear Attention 即使到达 compute-bound，它的延迟还是 $\mathcal{O}(N)$.

所以尽管 LA 有更多的 kernel launch 开销，更多的 non-matmul 操作，但是它还是能在长序列下把 Full attention 按在地上摩擦。

> 这就是 $(4)$ 这个式子想表达的意思，N 足够大时，$T_{Full}$ 一定会超过 $T_{GDN}$。

而且更可怕的是这些 "Flash" 库已经把短序列的延迟也优化的相当好了！


> CUDA Graph 能大幅减少逐 kernel 的 CPU 提交开销，但不会消除多 kernel 的全部成本。
> 
> ```Plain Text
> Eager：CPU 分别提交 A、B、C…… → GPU 执行 A、B、C……
> Graph：CPU 提交一次 replay   → GPU 执行 A、B、C……
> ```


---

## 参考资料

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Parallelizing Linear Transformers with the Delta Rule](https://papers.nips.cc/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Qwen3-Next model documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_next.md)
- [Kimi K3 official introduction](https://www.kimi.com/news/kimi-k3)
- [Nemotron-H documentation](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/nemotron-h)
- [Jamba 1.5](https://arxiv.org/abs/2408.12570)