---
title: Hands on FreeToken
date: 2026-09-07 18:38:13
updated: 2026-09-22 11:30:54
tags:
  - CS-notes
  - LLM inference
  - FreeToken
comments: false
excerpt: 跟踪 FreeToken 中请求从 prompt 到 token 的完整执行路径，解析 Scheduler、Engine、前缀缓存、CUDA Graph、Overlap scheduling 与 MoE 权重缓存。
mathjax: true
---

> 这份笔记围绕 FreeToken 的 Scheduler 和 Engine，跟踪请求从输入 prompt 到返回 token 的过程，再展开前缀复用、模型计算和 MoE 优化。模型示例采用 Qwen3.6-35B-A3B 的文本推理路径。
> 
> 讲解先以单 GPU、BF16、`normal_loop()` 为基准，走通首次请求未命中前缀缓存的路径。后续再加入前缀命中、分块 prefill、overlap scheduling 和 CPU/GPU 混合计算；具体模型配置与实验环境随对应章节记录。

## 1. 框架入口与启动

### 1.1 从一次生成请求看进程分工

用户发送一轮对话后，FreeToken 需要把消息转换成模型输入、组织计算，再把生成的 token 转回文本。**在线服务将这些工作分配给几类进程：**

| 组件 | 接收的内容 | 主要工作 |
| --- | --- | --- |
| API server | HTTP 请求中的对话、prompt 和生成参数 | 处理接口协议，向 tokenizer 发送任务，把生成结果返回客户端。 |
| Tokenizer worker | TokenizeMsg | 按需要应用 chat template，将输入编码为 token IDs，发送 UserMsg。 |
| Scheduler worker | 已编码的请求和控制消息 | 管理请求与缓存，选择 batch，调用 Engine，处理采样结果和停止条件。 |
| Detokenizer worker | DetokenizeMsg 中的输出 token | 维护请求的解码状态，将增量文本通过 UserReply 发回 API server。 |

Tokenizer 发给 Scheduler 的 `UserMsg` 包含：请求 ID、CPU 上的一维 token ID 张量和采样参数。

```python
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
```

对于多轮对话，这个张量通常对应本次重新渲染的完整 prompt，包括传入的历史消息。Scheduler 因而有机会将它与之前请求留下的前缀缓存匹配。

这些进程通过 ZeroMQ 队列传递内部消息。单 GPU 时有一个 Scheduler worker；张量并行（Tensor Parallelism，TP）时，每个 rank 有自己的 Scheduler 和 Engine。各 rank 协同执行同一批请求，rank 0 对接 tokenizer 和结果发送。后文先按单 rank 阅读。

- 在线模式入口是 [`launch_server()`](https://github.com/FlashML-org/FreeToken/blob/e05cff83a04b322fc7823678aa2d05c826aad26c/python/freetoken/server/launch.py#L125)。它为各 rank 启动 `_run_scheduler()`，并启动 tokenizer workers 和一个专用 detokenizer worker。每个 Scheduler 完成初始化后调用 `run_forever()`，持续接收和执行请求。
- 离线模式入口是 [`LLM.generate()`](https://github.com/FlashML-org/FreeToken/blob/e05cff83a04b322fc7823678aa2d05c826aad26c/python/freetoken/llm/llm.py#L105)，使用同一套调度逻辑。`LLM` 继承 `Scheduler`，通过本地方法提供输入和收集输出；所有请求结束后，`RequestAllFinished` 使调度循环返回。

![图 1](figure-01.png)

> 图 1：虚线框表示进程边界，实线框表示进程内对象。蓝色箭头标出跨进程消息，灰色箭头标出主要方法调用；PrefillManager 和 DecodeManager 合并展示。图中的 `TableManager.allocate()` 由 prefill 准入过程中的 `PrefillAdder` 调用。

### 1.2 Scheduler 和 Engine 的边界

Scheduler 持有一个 `self.engine`。二者位于同一个 worker 进程中，`self.engine.forward_batch(...)` 是普通的 Python 方法调用。Engine 本身也是 CPU 上的 Python 对象，它持有模型、GPU 张量和 CUDA stream，并向设备提交计算。

Scheduler 负责决定本轮处理哪些请求，并准备计算所需的位置与缓存映射。其中的一些核心结构：

- `TableManager` 管理请求表的空闲行，
- `CacheManager` 管理 KV 页和前缀复用，
- `PrefillManager` 与 `DecodeManager` 管理待处理请求。

Engine 接收已经准备好的 batch，提交模型前向和采样，返回 GPU/CPU token 张量及拷贝完成事件。

有一点需要清楚：Scheduler 能决定哪个请求进入 batch；该请求经过某层 MoE 时需要哪些专家，则由路由结果和 MoE cache 逻辑决定。

### 1.3 启动时建立哪些东西

启动需要两类配置：

- checkpoint 配置描述模型结构，例如层数、head 数量和专家维度。
- 运行参数控制并发上限、缓存预算与 backend。

`EngineConfig.model_config` 根据 checkpoint 的 architecture 选择解析器，将模型配置转成 FreeToken 内部表示。

`Scheduler.init()` 首先创建 Engine。按当前实现，主要顺序如下：

1. 绑定当前 rank 的 GPU，调整配置，创建 Engine stream 和 Context，初始化通信，并记录加载模型前的空闲显存。
2. 创建模型对象并加载权重，建立 CPU expert banks、GPU expert cache，以及所需的 CPU executor。
3. 根据剩余预算和配置计算 KV 页数，分配 KV pool、GDN state pool 和 `page_table`。
4. 创建 Attention/MoE backend 和采样器，建立 dummy request；启用 CUDA Graph 时捕获相应的 decode batch size，适用时预热 prefill 路径。
5. 创建 `TableManager`、`CacheManager`、`PrefillManager`、`DecodeManager` ...。`token_pool` 在 `TableManager` 中创建。worker 随后报告就绪并进入 `run_forever()`。

这些步骤在接收实际推理任务前完成。

`run_forever()` 在 `normal_loop()` 和 `overlap_loop()` 之间选择。本文先解读顺序模式：提交一个 batch，等待并处理该 batch 的结果，再进入下一轮。

> Qwen3.6-35B-A3B 总共有 40 层 decoder layer，其中 30 层使用 Gated DeltaNet（GDN），10 层使用 Full Attention，每层都有 MoE。这三类组件需要保存的内容各不相同：
> 
> - Full Attention 需要历史 token 的 K/V。保存这些张量可以避免每轮重复计算历史 K/V；保存量随处理过的 token 数线性增长。
> - GDN 处理新 token 时，读取旧状态并写入新状态，还需要 causal convolution 的短历史。因此每个请求要保存 recurrent state 和 convolution state。实时状态的形状由模型维度确定，不随序列长度增长；从历史中间位置恢复则需要该位置的 Snapshot 。
> - MoE 根据路由结果选择专家参与计算。专家权重在推理过程中保持不变，可以跨请求使用。BF16 专家权重体积较大，offload/hybrid 路径将完整 routed-expert 权重保存在 CPU 内存，并用 GPU expert cache 容纳其中一部分。

![图 2](figure-02.png)

> 图 2：启动顺序从左到右。`page_table` 在 Engine 的资源初始化阶段创建；Engine 返回后，TableManager 创建 `token_pool`。底部对照标出两者的对象归属。


---


## 2. 运行时资源

### 2.1 资源池与请求的关系

**这里的“资源池”指的是 FreeToken 在 GPU 显存中维护的缓存资源，包含了多种运行时缓存数据。**

Engine 在启动时为资源池分配 GPU 张量。

请求进入运行阶段时，Scheduler 从这些池中分配行号、page 或 slot，并把索引记录到请求上。源码里的“分配一个 KV page”通常就是取出池中的空闲页，并更新相应的分配记录。

资源池会被多个请求使用，请求结束后仍可继续服务下一批请求。Prefix cache 还可能保留结束请求的部分 KV 和 GDN  Snapshot ，以供后续命中。

| 资源 | 保存的内容 | 主要索引 |
| --- | --- | --- |
| token_pool | 请求的 token IDs | 请求行号、逻辑 token 位置。 |
| page_table | 逻辑 token 位置对应的物理 KV slot | 与 token_pool 相同的二维坐标。 |
| KV pool | 各 Full Attention 层的 K/V 张量 | 层号、物理 KV slot，再到 head 和通道。 |
| linear_state_pool | GDN 实时状态与中间快照 | GDN 层号、状态 slot。 |
| Prefix cache | 可复用 token 前缀及其 KV/GDN 索引 | token ID 前缀。 |
| MoE expert cache | GPU 上驻留的专家权重 | (layer_id, expert_id) 映射到 expert slot。 |

### 2.2 `token_pool` 和 `page_table`：同一坐标查两种内容

在本文使用的 Qwen 路径中，这两张表都是 GPU 上的二维 `int32` 张量，形状相同：

```python
[max_running_req + 1, aligned_max_seq_len]
```

- 行号：由 `req.table_idx` 指定，是可回收的运行槽位，独立于标识请求的 `uid`。
- 列号：表示 token 在该请求序列中的逻辑位置。总列数经过 page 和内存对齐处理，可能大于实际的 `max_seq_len`。
- 额外的一行：供 dummy request 使用。

**`token_pool` 保存 token ID**。例如：

```python
token_pool[3, 5] = 109266
```

这表示占用第 3 行的请求，在位置 5 上的 token ID 为 `109266`。prompt token 会被复制到这张表中，后续采样出的 GPU token 也会写回对应位置，作为下一次 decode 的输入。请求另有 CPU 上的 `input_ids`，供结果处理等逻辑使用。

**`page_table` 保存同一位置的 KV 存储索引**：

```python
page_table[3, 5] = 127
```

这表示该 token 的 K/V 使用物理 token slot 127。这里的 `127` 是整数索引，不是字节指针。各 Full Attention 层都有自己的 K/V 存储，使用同一个 slot 编号定位各自那一份数据。

读取同一组二维坐标时：

- `token_pool` 给出本轮的模型输入。
- `page_table` 给出这些输入计算出的 K/V 应写入的位置。

![图 3](figure-03.png)

> 图 3：`TableManager` 分配请求行的 `table_idx`。同一 `[table_idx, pos]` 坐标在 `token_pool` 中取得 token ID，在 `page_table` 中取得物理 KV slot；以 `page_size=16` 为例，slot 127 位于 page 7 的 offset 15。


### 2.3 KV pool：按页分配的历史 K/V

**KV pool 保存实际的 K/V 数值。**FreeToken 按 page 管理分配，每页容纳 `page_size` 个 token，`num_pages` 表示可用页数。因此共享 KV 容量是：

```Plain Text
可用物理 token slots = num_pages × page_size
```

例如，页号和页内偏移都从 0 开始：

- `page_size=1`：一页就是一个 token slot。
- `page_size=16`：物理 slot 127 位于第 7 页、页内偏移 15。

`page_table` 按 token 存储索引，底层按整页分配；同一请求的不同页可以散布在池中。

Qwen 的 Full Attention 使用 `MHAKVCache`。将层号重映射等细节省略后，底层张量的维度是：

```Plain Text
[2, full-attention 层数, 物理页数, page_size, 本 rank 的 KV heads, head_dim]
```

- 第一个维度的 `2` 对应 K 和 V。
- 物理页数包含给 dummy request 使用的保留页；用户可用容量使用 `num_pages` 计算，不计入保留页。
- 层维度只覆盖 Full Attention 层。GDN 层的状态另存于 `linear_state_pool`。

按本文 Qwen 示例，10 个 Full Attention 层、2 个 KV heads、`head_dim=256`，BF16 每个元素占 2 bytes，则每个 token 的 K/V 数据为：

```Plain Text
10 层 × 2（K 和 V）× 2 KV heads × 256 × 2 bytes
= 20,480 bytes
= 20 KiB/token
```

按这个单 token 开销计算：

- 8,192 个 token 对应 160 MiB。
- 32,768 个 token 对应 640 MiB。

这些数值只统计 K/V 数据，不包含两张索引表、GDN 状态、模型权重和其他运行开销。KV pool 的总容量由所有请求及保留的缓存前缀共享。


### 2.4 `linear_state_pool`：实时 GDN 状态与 Snapshot 

**一个 GDN state slot 保存某个序列位置上、当前 rank 所有 GDN 层的状态。**

池中主要有两组 GPU 张量：

```Plain Text
conv_states：
    [GDN 层数, slot 数量, convolution 通道数, kernel_size - 1]

recurrent_states：
    [GDN 层数, slot 数量, value heads, key_head_dim, value_head_dim]
```

- `conv_states` 保存短卷积需要的历史输入，精度随模型 dtype；在本文设置下使用 BF16。
- `recurrent_states` 保存递推状态，FreeToken 默认使用 FP32，可以通过 `FREETOKEN_MAMBA_SSM_DTYPE` 调整。

本地模型有 16 个 key heads、32 个 value heads，key/value head dimension 均为 128，卷积宽度为 4。由此得到 8,192 个 convolution 通道。TP=1 时，一枚 slot 对应的张量及数据量为：

```Plain Text
conv state：      [30, 8192, 3]，BF16，1.40625 MiB
recurrent state： [30, 32, 128, 128]，FP32，60 MiB
合计：            61.40625 MiB/slot
```

启用 Hybrid Radix Cache 后，新请求先取得一个实时 slot 和两个 ping-pong slot：

```python
linear_slot_idx = pool.alloc(1)[0]
ping_pong = tuple(pool.alloc(2))
```

- `linear_slot_idx` 指向实时状态。`pool.conv_states[:, req.linear_slot_idx]` 和 `pool.recurrent_states[:, req.linear_slot_idx]` 随 forward 更新。
- 两个 ping-pong slot 保留中间状态 Snapshot ，没有固定的 prefill/tool-call 分工。

写 Snapshot 和向 Prefix cache 提交 slot 的时机在第 4 章展开。

这些 slot 来自同一个池。slot 0 是 Hybrid 路径的 padding 保留槽；池中其余 slot 既可以由运行请求持有，也可以由 Radix Tree 持有。因此只按“每个运行请求三个 slot”计算整个池的大小会漏掉缓存前缀。

当前初始化公式中，令 `R=max_running_req`、`r=linear_state_cache_ratio`，Hybrid 路径的物理 slot 数为：

```Plain Text
4 × R + max(4, int(r × R)) + 1
```

- `4 × R`：为每个运行请求预留实时状态、两个 Snapshot 缓冲，以及一份运行期间受保护的已提交前缀 Snapshot 。
- `max(4, int(r × R))`：额外缓存容量，使用代码中的 `int` 截断规则。
- `1`：padding 保留槽。

`naive` 路径使用 `R+1` 个槽，以请求 `table_idx` 索引实时状态，不启用这套跨请求 Snapshot 缓存。

GDN 的实时状态大小不随上下文增长，但保存更多历史 Snapshot 仍然需要更多显存。`linear_state_cache_ratio` 控制状态池容量， Snapshot 的生成时机由另一套逻辑决定。

![图 4](figure-04.png)

> 图 4：`LinearStatePool` 的 slot 0 是 padding 保留槽；其余 slot 可由运行请求或 Radix Tree 持有。一枚 slot 横跨当前 rank 的全部 GDN 层，每层包含 `conv state` 和 `recurrent state`。


### 2.5 Prefix cache：记录哪些计算可以复用

前面的 KV pool 和 GDN pool 解决状态存放问题，**Prefix cache 负责查找：新请求开头的一段 token 是否已经计算过，对应的数据还保存在哪里。**

**本文使用的 `HybridRadixCache` 以 token ID 前缀组织 Radix Tree。**树的结构是 Python 对象，节点关联 token 序列和底层资源索引。每个节点中存储：

```python
node._key         # 该节点这一段 token IDs
node._value       # 该段 token 对应的物理 KV 索引
node.mamba_value  # 可选：节点末尾对应的 GDN 状态 slot ID
```

`node.mamba_value` 是一个 slot 编号。被保留的状态张量仍在 `linear_state_pool` 中，K/V 数据也仍在 KV pool 中；Prefix cache 通过引用和持有这些资源实现复用。

两个请求的前缀相同时，KV 和 GDN 的复用方式不同：

- KV：各自 `page_table` 的前缀区间可以指向同一组 KV slots。
- GDN：将匹配到的 Snapshot 复制到新请求的实时 slot。后续递推更新实时 slot，共享 Snapshot 保持不变。

对于这个同时包含 Full Attention 和 GDN 的模型，可恢复位置必须同时有可用的 K/V 和该边界的 GDN 状态。token 前缀匹配得更长，但末尾没有 Snapshot 时，实际命中长度会退回较早的有效 Snapshot 边界。请求结束后，仍由 Prefix cache 持有的页和状态槽也会继续占用池容量，直到它们被回收。


### 2.6 MoE expert cache：模型权重的 GPU 驻留空间

**专家权重属于模型本身。在 offload/hybrid 路径中：**

- **CPU host banks 按层保存完整的 routed-expert 权重。**
- **GPU expert cache 保存其中一部分。一枚 expert slot 容纳某一层、某一个 routed expert 的权重，通常包括 gate、up、down 三个投影。**

按本文 Qwen 示例，单个 BF16 expert 的原始投影尺寸为：

```Plain Text
gate_proj：[512, 2048]
up_proj：  [512, 2048]
down_proj：[2048, 512]

单个 expert = 3 × 512 × 2048 × 2 bytes = 6 MiB
全部 routed experts = 40 层 × 256 experts × 6 MiB = 60 GiB
```

60 GiB 只统计这些 routed-expert 权重，不含 shared expert、Attention/GDN 投影、Embedding、LM Head 等其他模型权重。GPU cache 的实际布局可以合并投影或重新排布；量化格式还会改变存储大小，这里的 6 MiB 仅用于 BF16 容量估算。

缓存维护两个方向的映射：

- `slot_for_id[layer_id, expert_id]` 查询一个专家当前在哪个 GPU slot，`-1` 表示不驻留。
- `id_of_slot` 记录每个 GPU slot 对应的专家身份。

请求需要某个专家时，执行路径据此决定复用已有权重、搬入缺失权重，或在 Hybrid MoE 中分派 CPU 计算。

启用 prefill double buffer 时，两个整层缓冲使用这个 GPU cache 的前 `2 × num_experts` 个 slot。对 256 experts 的 BF16 示例，这部分是 512 个 slot、3 GiB。它属于既有 expert cache 的容量，不能在计算显存总量时再额外加一遍。

专家权重缓存的生命周期不与某个 `Req` 绑定。不同 MoE backend 的执行与搬运策略在第 7 章展开。


### 2.7 容量参数怎样约束请求

同一块 GPU 上，模型驻留权重、MoE cache、KV pool、GDN pool，以及计算临时空间共同占用显存。扩大某个池会减少留给其他部分的空间，因此并发数、上下文长度和专家驻留量需要一起考虑。

请求能否准入，需要分别检查：

- 请求行：`max_running_req` 限制同时占用的运行槽位数，也参与 GDN 状态池定容。
- KV pages：KV pool 要容纳所有请求和保留前缀实际占用的页。
- GDN slots：GDN pool 要容纳运行状态与缓存 Snapshot 。

因此，运行槽位有空余，并不保证新请求所需的其他资源也足够。

启动时的显存规划涉及两个参数：

- `memory_ratio` 以加载模型前记录的空闲显存为预算基准。Engine 在预算中扣除已驻留的数据和状态池开销，再求解 KV 容量。比例之外的 `(1 - memory_ratio)` 余量为 CUDA Graph、激活和临时计算预留空间。
- `moe_cache_auto` 启用后，联合规划 MoE 和 KV 的容量，并为 KV 留出配置的最低容量。

对本文 Qwen 路径，Engine 还将实际单请求长度上限限制为 `min(config.max_seq_len, num_pages × page_size)`。这是一条单请求上限；多个请求仍要竞争共享容量。

Scheduler 会在这些资源约束下选择本轮 batch，第 3 章继续分析准入和调度代码。


## 3. Life of a token：一次请求的执行

### 3.1 调度循环与消息入口

![图 5](figure-05.png)

> 图 5：normal 模式下，一轮循环依次完成本轮 forward 和本轮结果处理；采样 token 的 GPU 副本写入 `token_pool` 供下一轮 decode 使用，CPU 副本经 Detokenizer 转成文本并逐层返回客户端。Prefill 采样得到的第一个 token 也沿同一路径返回。Scheduler 与 Engine 是同一 worker 进程中的逻辑泳道。


`run_forever()` 持续调用 `normal_loop()`。每次迭代先处理新消息，再选择并执行一个 batch，最后处理这个 batch 的结果。一个请求通常需要一次 prefill 和多次 decode，因此会跨越多轮循环。

下面摘出生成请求的主路径，省略缓存重建分支：

```python
def normal_loop(self) -> None:
    blocking = not (
        self.prefill_manager.runnable
        or self.decode_manager.runnable
        or self._pending_rebuild is not None
    )
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    # 此处省略空闲时执行缓存重建的分支。

    forward_input = self._schedule_next_batch()
    ongoing_data = Noneif forward_input is not None:
        self._restore_linear_states(forward_input.batch)
        ongoing_data = (forward_input, self._forward(forward_input))

    self._process_last_data(ongoing_data)
    self._flush_abort_acks()
```

这里的 `ongoing_data` 保存本轮输入和输出，在同一轮交给 `_process_last_data()`。`run_forever()` 的外层已经进入 `engine_stream_ctx`，本轮的 GPU 准备操作、模型计算和结果写回都按 Engine stream 的顺序提交。

> 本章用一个请求贯穿流程：
> 
> - prompt token IDs 为 `[10, 20, 30, 40]`，长度为 4。
> - 输出预算为 3 个 token，资源充足，实际长度上限至少为 7。
> - 使用单 rank、`page_size=1` 和 Hybrid Radix Cache；本次未命中前缀，prefill 预算足够一次处理整个 prompt。
> - 假设采样结果依次为 `50`、`60`、`70`，没有提前命中 EOS 或 stop string。这些 ID 仅用于说明读写位置。


循环入口先判断是否需要等待新消息：

- 没有待 prefill 请求、没有运行中的 decode 请求，也没有待执行重建时，`blocking=True`。
- 只要还有其中一类工作，`blocking=False`，收取当前已到达的消息后继续调度。

单 rank 在线路径实际调用 `_recv_msg_single_rank()`。阻塞模式先做空闲检查，再等到至少一条消息；之后继续取出队列中已经到达的消息。非阻塞模式下，队列为空就返回空列表。这里等待的是 CPU 消息队列，与 CUDA 完成事件无关。

对于普通 `UserMsg`，`_process_one_msg()` 先计算剩余生成空间 `max_output_len = engine.max_seq_len - input_len`：

- 没有生成空间时，返回长度错误。
- 请求的 `max_tokens` 超过剩余空间时，下调输出预算。

有效请求通过 `prefill_manager.add_one_req(msg)` 进入等待队列。

等待队列存放的是 `PendingReq`：

```python
self.pending_list.append(
    PendingReq(req.uid, req.input_ids, req.sampling_params, mm_embeds=req.mm_embeds)
)
```

它先保存输入 token、采样参数和请求 ID。此时请求还没有取得自己的 `table_idx` 或 GDN 状态槽。后续准入成功后，才创建携带运行资源的 `Req`。

同一入口也会处理取消、缓存重建等控制消息。批量消息会被拆开逐条处理；这些控制分支在本章示例中不触发。


### 3.2 请求准入与 batch 准备

![图 6](figure-06.png)

> 图 6：请求 A 和 B 的输入按顺序拼成一个长度为 6 的 batch。`input_tuple` 用两组索引标出每个输入 token 的请求行和逻辑位置；Engine 为每个请求产生一个 next token，再由 `write_tuple` 写回各自的 `token_pool` 行。


`_schedule_next_batch()` 优先尝试组成 prefill batch：

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

- PrefillManager 按顺序遍历 `pending_list`，由本轮的 `PrefillAdder` 判断是否接纳请求。
- 遇到无法接纳的请求时停止遍历；此前已选中的请求仍可组成 batch。
- 只有没有组成 prefill batch 时，才尝试 decode。DecodeManager 将 `running_reqs` 按 `uid` 排序，组成 decode batch。

**因此，prefill-first 会使已有 decode 请求等待新 prompt 的处理；当 prefill 因资源不足而无法组成 batch 时，已有 decode 仍有机会继续运行。**


对于尚未准入的新请求，`PrefillAdder` 依次处理：

1. 检查 TableManager 是否还有空闲请求行。
2. 与 Prefix cache 匹配，取得可复用长度和 cache handle。本例命中长度为 0。
3. 按未缓存 prompt 长度、输出预算和本轮已有预留量检查 KV 容量。可用量包括空闲页和可回收的前缀缓存；锁定命中前缀后还会再次检查。
4. Hybrid GDN 路径检查是否能取得一个实时 slot 和两个 Snapshot  slot；不足时尝试回收可淘汰的 Snapshot 。
5. 分配 `table_idx` 和 GDN slots，确定本次 prefill chunk，将对应 prompt token 复制到 `token_pool`，创建 `Req` 或 `ChunkedReq`。

这时的 KV 检查是准入预算判断，实际的本轮页分配发生在后面的 `_prepare_batch()`。

`prefill_budget` 限制本轮处理的未缓存 token 总数。如果某个 prompt 不能一次处理完，就创建 `ChunkedReq`，将剩余部分留到之后的 prefill。继续处理时沿用请求已经取得的资源。中间 chunk 的输出不会作为用户的生成结果，最后一个 chunk 才进入普通生成路径。

假设分配到 `table_idx=3`。请求建立后：

| 字段 | 初值 | 用途 |
| --- | --- | --- |
| input_ids | [10, 20, 30, 40]，CPU 张量 | 保存 host 侧 token 序列。 |
| table_idx | 3 | 定位 token_pool/page_table 的请求行。 |
| cached_len | 0 | 调度器记录的已处理前缀长度。 |
| device_len | 4 | forward 前的序列长度，也是输入区间的右端点。 |
| extend_len | 4 | device_len - cached_len，本轮需要计算的 token 数。 |
| max_device_len | 7 | prompt 长度加输出预算。 |
| linear_slot_idx | 池分配的一个索引 | 请求的实时 GDN 状态位置。 |

最终得到 `Batch(reqs=[req], phase="prefill")`。


选出请求后，`_schedule_next_batch()` 调用 `_prepare_batch()`，将请求列表转成模型可以使用的输入描述。本文 Qwen 文本路径中的主要操作是：

1. 准备 batch 大小和存储：`graph_runner.pad_batch()` 为可使用 CUDA Graph 的 decode batch 补 dummy request，其余情况下保留原请求数；然后为真实请求本轮的输入位置分配 KV pages，更新 `page_table`。
2. 构造坐标：拼接各请求 `[cached_len, device_len)` 内的 positions，生成读取输入的 `input_tuple` 和写回采样结果的 `write_tuple`。
3. 准备模型输入描述：查询 `page_table[input_tuple]` 得到 K/V 写入位置 `batch.out_loc`，构造 Attention/GDN metadata 和采样参数，返回 `ForwardInput`。

Metadata 告诉计算层各请求的长度、输入边界和状态位置。例如，GDN 需要知道每个请求应读写哪个实时 slot，Attention 需要找到历史 K/V。它们随本批请求准备，模型各层使用这些信息执行计算。

返回结构是：

```python
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D
    write_tuple: Indice2D
```

其中两个 tuple 都包含一对 GPU `int64` 索引张量，逐元素配对成二维坐标。本例第一次 prefill 时可以表示为：

```Plain Text
batch.positions = [0, 1, 2, 3]

input_tuple = (
    [3, 3, 3, 3],  # 每个输入 token 属于请求表的哪一行
    [0, 1, 2, 3],  # 每个输入 token 的逻辑位置
)

write_tuple = (
    [3],           # 真实请求的行号
    [4],           # 本轮采样结果应写入的位置
)
```

- `input_tuple` 每个输入 token 对应一组坐标。它包含所有待计算 token，使用 CUDA Graph padding 时还会包含 dummy 输入。
- `write_tuple` 每个真实请求对应一组坐标，因为每个请求本轮只采样一个 token。

`write_tuple` 记录的是 forward 之前的 `device_len`。已有 token 占用位置 `[0, 4)`，所以下一个 token 写在位置 4。这个坐标需要提前保存，不能等 forward 将长度推进后再用新的 `device_len` 计算。


`normal_loop()` 拿到 `ForwardInput` 后，先调用：

```python
self._restore_linear_states(forward_input.batch)
```

它只在 prefill 请求带有 `mamba_restore_src` 时，将源 Snapshot 复制到请求的实时 `linear_slot_idx`，随后清除恢复标记；没有 GDN pool 或当前是 decode 时直接返回。本例未命中前缀，无需复制，GDN prefill 路径会按新序列初始化状态。


### 3.3 前向执行与请求推进


`Scheduler._forward()` 将输入准备与 Engine 执行连接起来：

```python
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    batch, sample_args, input_mapping, output_mapping = forward_input
    batch.input_ids = self.token_pool[input_mapping]
    if self.toolcall_anchor_id is not None and not batch.is_prefill:
        self.cache_manager. Snapshot _toolcall_anchor(batch.reqs)
    forward_output = self.engine.forward_batch(batch, sample_args)
    self.token_pool[output_mapping] = forward_output.next_tokens_gpu
    self.decode_manager.filter_reqs(forward_input.batch.reqs)
    return forward_output
```

这里的 `input_mapping`、`output_mapping` 分别是解包后的 `input_tuple`、`write_tuple`。

首先，`token_pool[input_mapping]` 将本轮输入收集成一维张量。本例取得 `[10, 20, 30, 40]`。Prefill batch 包含多个请求时，各请求的待计算 token 按顺序拼接；请求之间的边界由 metadata 描述。

![图 7](figure-07.png)

> 图 7：同一请求依次经历 prefill、Decode 1、Decode 2。每列分开显示模型输入区间、`complete_one()` 的 CPU 记账、`token_pool` 写回、`copy_done_event` 完成和 `append_host()`。最后的 `70` 已写回并追加到 CPU 序列，但没有作为下一轮输入；最终 `cached_len=6`，CPU token 序列长度为 7。


`Engine.forward_batch()` 随后按源码顺序完成以下工作：

1. 在 Engine stream 上设置 Context 并提交模型计算：**prefill 走 eager `model.forward()`；decode 在捕获范围内可以 replay，否则走 eager。**若存在 CPU MoE executor，随后检查其错误标记，发现故障则抛出异常。
2. 对真实请求调用 `complete_one()`，推进 CPU 上的长度记账。
3. 取出真实请求对应的 logits，按采样参数产生下一 token，转为 GPU `int32` 张量。
4. 发起 token 的异步 GPU→CPU 拷贝，在同一 stream 上记录完成事件，返回 `ForwardOutput`。

Prefill 虽然处理多个输入 token，LM Head 只选择每个请求最后一个输入位置的 hidden state 来产生 logits，因此本例只采样一个输出 token。采样参数由 `Sampler.prepare()` 提前准备；全 greedy batch 使用 `argmax`，其他情况按温度及 top-k/top-p 等参数采样。

`complete_one()` 更新两个长度字段：

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1
```

第一次 prefill 的记账变化是：

- `cached_len`：从 0 变为 4，记录本轮模型计算覆盖到的前缀边界。
- `device_len`：从 4 变为 5，为本轮即将采样的 token 推进逻辑长度。
- CPU `input_ids`：此时仍只有原来的 4 个 token，尚未追加输出。

`complete_one()` 位于采样调用之前，只更新调度字段，不等待 GPU。模型计算、采样和拷贝之间的依赖由 stream 顺序保证；CPU 要读取结果时，仍须等待完成事件。

`ForwardOutput` 返回三项：

- `next_tokens_gpu`：各真实请求的下一 token，形状为 `[batch.size]`。
- `next_tokens_cpu`：对应的 CPU 结果张量，异步拷贝完成后才能读取。
- `copy_done_event`：记录在 D2H 拷贝之后的 CUDA event。

返回 Scheduler 后，本例的输出 `50` 写入 `token_pool[3, 4]`。接着更新 decode 请求集合：

```python
self.running_reqs = {
    req for req in self.running_reqs.union(reqs) if req.can_decode
}
```

- 新完成 prefill、仍有生成预算的请求进入集合。
- 正在 decode 的请求有剩余预算就保留，预算耗尽则移除。
- `ChunkedReq.can_decode` 恒为 `False`，中间 chunk 不进入 decode 集合。

本例此时还剩 `max_device_len - device_len = 7 - 5 = 2` 个输出 token 的预算，因此进入 `running_reqs`。这一步尚未判断 EOS 或 stop string，停止条件要等 CPU 结果可读后处理。

事件还有一个边界需要区分：`copy_done_event` 在 Engine 内记录，而 `token_pool` 写回在 Engine 返回后才提交。该事件覆盖 D2H 拷贝，不覆盖之后的写回。在 normal 模式中，写回与下一轮输入读取都位于同一个 Engine stream，执行顺序由该 stream 保证。


### 3.4 结果处理、后续 decode 与结束

本轮计算提交后，`normal_loop()` 随即处理 `ongoing_data`：

```python
ongoing_data = (forward_input, forward_output)
self._process_last_data(ongoing_data)
```

它取出 batch、CPU token 和事件：

```python
batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
copy_done.synchronize()
```

**`synchronize()` 在这里阻塞 CPU，直到事件之前的工作完成。**随后第 `i` 个 CPU token 对应 `batch.reqs[i]`，普通请求的处理顺序是：

1. 从 `next_tokens_cpu[i]` 取出 token，通过 `append_host()` 追加到 CPU `input_ids`，检查停止条件，并在需要时记录 tool-call anchor。本例的 host 序列变为 `[10, 20, 30, 40, 50]`。
2. 构造包含请求 ID、token ID、是否结束和结束原因的 `DetokenizeMsg`，再处理资源：结束请求退出 decode 集合；完成 prefill 但还要继续生成的请求调用 `cache_req(finished=False)`。
3. 更新运行统计，通过 `send_result(reply)` 将消息发给 detokenizer。

停止判断包含三个条件：

| 判断 | 检查对象 | 命中结果 |
| --- | --- | --- |
| hit_length | not req.can_decode，判断剩余输出预算是否耗尽 | 预算耗尽，结束原因可为 "length"。 |
| hit_eos | 本轮 token ID 是否在 EOS 集合中，且未设置 ignore_eos | 结束原因为 "stop"。 |
| matched_stop | 生成内容尾部解码后的文本是否含有配置的 stop string | 返回命中的字符串，否则为 None；命中时结束原因为 "stop"。 |

EOS 已命中时不再检查 stop string；EOS 或 stop string 与长度限制同时命中时，结束原因优先记为 `"stop"`。字符串匹配可以跨多个 token，因此要对生成内容的尾部解码，而不是只比较最新的 token ID。


函数还有几条特殊分支：

- 中间 `ChunkedReq` 跳过普通输出处理，不把它的采样结果发送给用户，也不在这里提交中间 chunk 的前缀。
- 已取消请求处理资源回收，终止确认由后面的 `_flush_abort_acks()` 发送。
- overlap 模式中可能出现已结束请求多启动一步的情况，`finished_reqs` 用于跳过额外输出；本章 normal 模式的示例不走这条路径。

Detokenizer 收到 `DetokenizeMsg` 后，按请求维护解码状态，将 token 转成增量文本，再通过 `UserReply` 交给 API server。一个 token 不保证立即产生可发送的完整字符；配置 stop string 时，detokenizer 还会暂存可能属于停止字符串的尾部，并在命中后截去该字符串及其后面的内容。


第二次进入 `normal_loop()` 时，本例已经在 `running_reqs` 中，因此不会阻塞等待新请求。假设没有新的 prefill batch 抢先执行，DecodeManager 选中它继续计算。

- 输入区间由 `[cached_len, device_len) = [4, 5)` 给出，从 `token_pool[3, 4]` 读入上一轮生成的 `50`。
- 本轮为位置 4 准备 KV 存储，使用已有历史状态，模型采样出 `60`。
- `60` 写入位置 5；CPU 结果处理完成后，进入下一轮 decode。

第三轮同样读入 `60`，采样出 `70`。下表用 `(cached_len, device_len, CPU input_ids 长度)` 表示请求状态；初始值为 `(0, 4, 4)`。

| 轮次 | 模型输入 | 采样输出及写入列 | complete_one() 后 | CPU 追加后、资源处理前 | Decode 集合 |
| --- | --- | --- | --- | --- | --- |
| Prefill | [10, 20, 30, 40] | 50，列 4 | (4, 5, 4) | (4, 5, 5) | 加入，剩余预算 2。 |
| Decode 1 | [50] | 60，列 5 | (5, 6, 5) | (5, 6, 6) | 保留，剩余预算 1。 |
| Decode 2 | [60] | 70，列 6 | (6, 7, 6) | (6, 7, 7) | 移除，剩余预算 0。 |

最后一轮的 GPU token 仍然写回，CPU token 仍然追加并返回。区别在于 `max_device_len - device_len = 0`，不再安排下一次 decode。`_process_last_data()` 将结束原因标为 `"length"`，并调用 `_free_req_resources()`：

- 先执行 `cache_req(req, finished=True)`，决定哪些 KV/GDN 资源保留给 Prefix cache、哪些释放。
- 再释放请求的表格行，将 `req.table_idx` 设为 `-1`。

输出序列已经包含 `70`，但模型从未将它作为下一轮输入。因此本例在最后一次 GPU 模型计算完成后，Full Attention 的 KV 覆盖逻辑位置 `[0, 6)`，GDN 实时状态累计覆盖前 6 个 token。最后一个输出 token 位于位置 6，没有对应的输入计算。最终状态对应 `cached_len=6`，与 CPU token 序列长度 7 相差一个 token。

本轮结果处理结束后，`normal_loop()` 调用 `_flush_abort_acks()`；本例没有取消消息，它什么也不做。下一轮若所有队列都已空且没有待执行重建，Scheduler 再次阻塞等待新消息。


---


## 4. 前缀复用与状态生命周期

上一章跟踪了一个未命中缓存的请求。这一章继续看它留下的状态如何被后续请求复用。讨论范围仍是 Qwen 的文本推理路径、Hybrid Radix Cache；示例默认 `page_size=1`。

### 4.1 匹配前缀并恢复状态

前缀复用要求 token ID 从序列开头连续相同。可复用长度还取决于缓存里保留了哪些状态。Full Attention 的 KV 按 token 存储，能够截取前缀；GDN  Snapshot 对应一个确定的序列边界，不能从较晚的 Snapshot 中截出较早的状态。

假设请求 A 留下了前 152 个 token 的 KV，并在长度 128 和 152 处保存了 GDN  Snapshot 。新请求 B 共 160 个 token，与 A 的前 150 个 token 相同：

- token 匹配可以到长度 150。
- 长度 152 的 Snapshot 已经包含 B 不同的后续内容，不能使用。
- 如果长度 128 的 Snapshot 仍在，就复用 `[0, 128)`，重新 prefill `[128, 160)`。其中 `[128, 150)` 虽然 token 相同，仍需重算，以推进 GDN 状态。

**`HybridRadixCache.match_prefix()` 先沿树匹配 token，再沿父节点寻找最深的有效 Snapshot 。**整条已匹配路径都没有 Snapshot 时，返回命中长度 0。

![图 8](figure-08.png)

> 图 8：请求 B 的 token 前缀匹配到长度 150；树在 150 处分裂后，新节点没有 GDN  Snapshot ，长度 152 的原 Snapshot 仍留在后缀且已包含分歧内容。因此匹配结果沿父节点退回长度 128，将 `S128` 复制到 B 的实时 slot，再重新 prefill `[128, 160)`。


调度器接下来使用匹配结果完成准入和恢复：

1. 用 cache handle 保护匹配到的前缀资源，将命中长度写入 `req.cached_len`。
2. 为 B 分配自己的 `table_idx`，将其页表前缀指向缓存中的 KV slots。A、B 可以使用相同的前缀 KV，但各有自己的 token 表行。
3. 为 B 分配新的实时 GDN slot，把树中 Snapshot 的 slot ID 记录为 `mamba_restore_src`。
4. 在 B 的首个 prefill forward 之前，复制状态并清除恢复标记：

```python
pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
req.mamba_restore_src = None
```

这次复制覆盖当前 rank 所有 GDN 层的 conv/recurrent state。之后 B 更新自己的实时状态，缓存中的 Snapshot 保持不变。复制在 Engine stream 上排在模型计算之前；后续 prefill chunk 和 decode 直接使用已经推进的实时状态。

请求的 token 表仍保留完整输入，命中前缀只改变哪些位置需要重新计算。还有一条边界规则：`CacheManager.match_req()` 最多拿 `input_ids[:input_len - 1]` 做匹配，至少留一个输入 token 进入模型，以便产生本次生成所需的 logits。


### 4.2 保存  Snapshot  与提交给 Radix Tree

实时 GDN 状态随 forward 更新。为了保留某个中间边界，FreeToken 将对应状态写入 `mamba_ping_pong` 中的一枚 slot。这个动作完成后， Snapshot 仍归请求持有；只有提交给 Radix Tree，其他请求才能通过前缀匹配找到它。

Prefill 的 Snapshot 边界由 `_build_track_metadata()` 选择。当前 GDN kernel 的 `CHUNK_SIZE=64`，每个请求在本次 forward 内最多选择一个内部边界：

```python
c = (req.extend_len - 1) // CHUNK_SIZE
boundary = req.cached_len + c * CHUNK_SIZE
```

只有 `c >= 1` 才生成这种 Snapshot 。例如从长度 0 开始处理 150 个 token，选中长度 128；处理 64 个 token 时不选内部边界，处理 65 个时才选中长度 64。这个边界相对于本次 extend 的起点计算，不等于把完整序列长度统一向下取整到 64。

各 GDN 层从 kernel 返回的中间 recurrent state 和边界处的 convolution 历史窗口提取状态，通过 `_write_track_ Snapshot ()` 写入选定 slot。forward 结束时，实时 slot 已经推进到本次输入末尾，而 Snapshot 停留在选定边界。

调度器用两个字段跟踪这次写入：

- `mamba_last_track_seqlen` 记录待提交 Snapshot 的前缀长度。
- `mamba_next_track_idx` 选择下一次写入的 ping-pong 位置，写入目标确定后在 0、1 之间切换。

一个长 prompt 可以跨多次 prefill forward，每次符合条件时都会记录 Snapshot 。但是 Scheduler 不会将中间 `ChunkedReq` 的缓存提交给 Radix Tree，因此只有最后一个 chunk 边界记录的 Snapshot 最终成为可复用前缀；如果最后一块太短，没有产生新的内部 Snapshot，就可能没有可提交的 prefill Snapshot。

最后一个 prefill chunk 处理完、请求还要继续生成时，`cache_req(req, finished=False)` 尝试提交 Snapshot。提交长度必须按 KV page 对齐，使 K/V 前缀与 GDN 状态描述同一个边界。Radix Tree 接受这枚 slot 后，请求会补一个新 slot：

```python
if not mamba_exist:
    self.ensure_mamba_slots(1)
    pp = list(req.mamba_ping_pong)
    pp[frozen_idx] = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pp)
```

`mamba_exist=True` 表示树中已经有相同前缀的有效 Snapshot ，此次不接收请求提供的 slot；原 slot 仍可留给请求使用，也无需补位。树接受时则转移所有权，状态张量仍在原来的 pool 位置，并不再次复制一份。若提交时发现重复的 KV，继续运行的请求还需要将页表改指树中保留的 KV，避免之后访问已经归还的重复页。

Decode 中额外的 Snapshot 来自 tool-call anchor。工具调用内容被客户端重新序列化后，新 prompt 可能从工具参数处发生分歧；在工具调用起始标记之后保留状态，可以为仍然相同的那段前缀增加一个恢复点。启用 `special_token_ckpt` 且起始标记能编码为单个 token 时，调度器记录第一个符合条件的标记位置。

保存顺序是：采样出标记并记录位置，下一次 decode 消费该 token，再在后续 forward 前保存状态。`Snapshot_toolcall_anchor()` 要求 `cached_len` 恰好到达该位置、位置按 page 对齐，并且存在可用 Snapshot 槽且没有尚待提交的 Snapshot 。它把实时 GDN 状态复制到选定的 ping-pong slot；请求继续 decode， Snapshot 等到请求结束才提交。

各阶段的保存与提交时机如下：

| 场景 | 状态保存位置 | 向树提交和补位 |
| --- | --- | --- |
| 中间 prefill chunk 产生快照 | 当前选中的 ping-pong slot | 正常续接时不提交，后续可能覆盖。 |
| 最终 prefill chunk 产生快照，且请求继续生成 | 当前选中的 ping-pong slot | 结果处理时尝试提交；树接受后补新 slot。 |
| Decode 到达 tool-call anchor | 实时 slot 复制到选中的 ping-pong slot | 暂不提交，保留到请求结束。 |
| 请求结束 | 有效的待提交快照和最终实时状态 | 尝试交给树；请求不再补快照槽。 |

两个 ping-pong slot 没有固定的 prefill/tool-call 分工。源码能证明轮换与补位的行为；它们在当前所有路径上是否都不可简化，尚未验证。


### 4.3 请求结束与资源回收

第三章的请求结束时会调用 `cache_req(req, finished=True)`。此时 `linear_slot_idx` 保存的是最终已处理输入的 GDN 状态，对应 `cached_len`；刚采样但尚未消费的最后一个输出 token 不在这个状态中。

Hybrid 路径先处理尚未提交的有效 Snapshot ，再处理实时 slot：

1. 若存在有效、page-aligned 的待提交 Snapshot ，先用其前缀 token、KV 索引和 Snapshot  slot 插入 Radix Tree。已有相同 Snapshot 时释放重复 slot，否则由树接管。
2. 当 `cached_len > 0` 且恰好按 page 对齐时，尝试将实时 slot 交给树，作为最终边界的 Snapshot 。长度不对齐时跳过这次实时状态提交，避免把较晚的状态绑定到较短的前缀。
3. 解除旧 cache handle 的保护，归还重复或未保留的资源，释放请求剩余的状态槽；随后 Scheduler 释放 `table_idx`。

第 2 步无需再复制实时状态，因为请求已经不会继续修改它。树接受这枚 slot 时，将其 ID 记入 `node.mamba_value`；如果已有同一边界的有效 Snapshot ，则归还重复的实时 slot。请求持有的引用随后被清除，防止再次释放已经转交给树的 slot。多模态请求在当前实现中不走这条共享前缀发布路径。

保留进树的资源也需要保护和淘汰。`CacheManager.lock()` 在这里增加引用计数：保护命中节点的 GDN  Snapshot ，以及从该节点到根路径上的 KV。`unlock()` 解除对应保护，让资源在没有使用者时重新成为可回收对象。它们维护的是缓存使用关系，不是 Python 线程互斥锁。

KV 和 GDN 各有容量约束，所以淘汰入口不同：

- KV 页不足时，`evict_full()` 从未受保护的叶节点按 LRU 回收，返回 KV 索引及该节点的 GDN slot。内部节点的 KV 仍是后代前缀的一部分，不能随意删除。
- GDN slot 不足时，`evict_mamba()` 可以只移除内部节点的 Snapshot ，保留其 KV 和子节点。若移除的是可回收叶节点，则同时回收其 KV，并清理随后暴露的无 Snapshot 叶节点。

例如，长度 128 的内部节点失去 GDN  Snapshot 后，长度 152 的后代 Snapshot 仍可能有效。新请求若只匹配到长度 150，就不能再从 128 恢复，需要退回更早的有效 Snapshot ；能匹配到 152 的请求则仍可使用后代状态。缓存中存在 K/V，并不保证任意中间位置都能复用。

树本身返回需要释放的索引，CacheManager 再把它们归还给 KV pool 或 GDN pool 的空闲列表。这让后续请求可以重新分配这些资源；底层预先分配的 GPU 张量仍然存在，所以归还 page/slot 不等于显存占用立即下降。


---


## 5. Engine、模型与算子执行

第 3 章把 Engine 当成一个执行接口。本章打开这个接口，看一轮 batch 在模型内部怎样流动。示例继续使用本地 Qwen3.6-35B-A3B 文本配置：40 层中 30 层是 GDN、10 层是 Full Attention，每层都有 top-8 routed MoE 和 gated shared expert。

### 5.1 从 `Engine.forward_batch()` 到 logits

`Engine.forward_batch()` 的输入是已准备好的 `Batch` 和采样参数，输出是 `ForwardOutput`：

```Plain Text
Batch
  → Context.batch
  → CUDA Graph replay 或 model.forward()
  → logits
  → Sampler
  → next_tokens_gpu / next_tokens_cpu / copy_done_event
```

执行顺序可以压缩成四步：

- 临时安装 `Context.batch`。模型层从 Context 读取 `input_ids`、positions、attention metadata 和 GDN metadata。
- Decode 且 batch size 在捕获范围内时，`GraphRunner.replay()` 更新固定 buffer 后重放 CUDA Graph；其他情况执行 eager `model.forward()`。
- 若存在 CPU MoE executor，检查 watchdog 标记，避免 coordinator 失效时继续使用旧的 expert 输出。
- `complete_one()` 更新请求长度后，取真实请求对应的 logits，执行采样，发起异步 D2H 拷贝并记录完成事件。

Prefill 的输入有多个 token，但 LM Head 只选每个请求最后一个输入位置的 hidden state，所以仍然每个请求产生一个采样 token。CUDA Graph 的 padding 行由 `logits[:batch.size]` 丢弃。


### 5.2 Qwen3.6 的模型计算路径

Qwen 的顶层路径是：

![图 9](figure-09.png)

每个 decoder layer 将 residual add 融入下一次 RMSNorm，然后执行 mixer 和 MoE：

![图 10](figure-10.png)

Full Attention 会做 fused QKV projection、Q/K RMSNorm、partial rotary embedding，并通过 Attention backend 读写历史 K/V。attention output 乘 output gate 后经过 output projection。

GDN 会做 fused input projection、causal convolution、gated delta rule、gated RMSNorm 和 output projection。Prefill 使用变长 chunk kernel，decode 使用单 token kernel；两条路径都按请求状态 slot 读写 `linear_state_pool`。

MoE 的层内顺序是：

```Plain Text
hidden
  ├─ router → top-k expert IDs / weights
  ├─ shared expert → sigmoid gate × shared output
  └─ routed experts → routed output
      ↓
  routed output + shared output
```

Qwen 配置是 256 个 routed experts、每 token 选择 8 个。routed expert 的执行由 `ctx.moe_backend` 决定；shared expert 属于模型层本身。


#### Gated DeltaNet

![图 11](figure-11.png)

```Plain Text
Input RMSNorm
  ↓ X [N, 2048]
Fused Input Projection [2048 → 12352]
  ├─ conv_in [N, 8192]
  │    └─ Depthwise Causal Conv1D, width=4 + SiLU
  │         ├─ Q [N, 16, 128]
  │         ├─ K [N, 16, 128]
  │         └─ V [N, 32, 128]
  ├─ Z [N, 32, 128]
  ├─ b [N, 32] → β = sigmoid(b)
  └─ a [N, 32] → g = -exp(A_log)·softplus(a+dt_bias)
                         ↓
               Gated Delta Rule
                         ↓ O [N, 32, 128]
        Per-head RMSNorm₁₂₈(O) × SiLU(Z)
                         ↓ [N, 4096]
              Output Projection [4096 → 2048]
                         ↓
                    Residual Add
```


### 5.3 CUDA stream

CUDA kernel 是在 GPU 上并行执行的一段函数。例如一次模型 forward 中可能出现：

```Plain Text
script:                     |   backend:                 
x = rms_norm(x)             |   RMSNorm kernel
qkv = linear(x)             |   cuBLAS GEMM kernel
q, k = rope(q, k)           |   RoPE kernel
y = attention(q, k, v)      |   FlashInfer Attention kernel
```

一个 PyTorch 算子不一定只对应一个 kernel，也可能发射多个 kernel。

**CUDA stream** 可以理解成 GPU 的命令队列。Python/CPU 负责发布命令，GPU 负责异步执行命令，CUDA stream 是这些命令的有序队列。同一个 stream 中的命令严格按顺序执行。

通过这个命令队列，CPU 和 GPU 实现异步执行。

```python
a = torch.matmul(x, w1)  # CPU 向 stream 提交 GEMM 1
b = torch.relu(a)        # CPU 向 stream 提交 ReLU
c = torch.matmul(b, w2)  # CPU 向 stream 提交 GEMM 2
```

![图 12](figure-12.png)


### 5.4 Eager Execution

Eager 的意思是：Python 每执行到一个 PyTorch 算子，就立即完成 dispatch，并向当前 CUDA stream 提交相应 kernel。

```python
def forward(x):
    x = self.norm(x)       # Python dispatch -> enqueue norm kernel
    x = self.attention(x)  # Python dispatch -> enqueue attention kernels
    x = self.mlp(x)        # Python dispatch -> enqueue GEMM kernels
    return x
```

每次 kernel launch 都会经过类似路径：

```Plain Text
Python
-> PyTorch dispatcher
-> ATen/custom op
-> CUDA runtime/driver
-> GPU command queue
```

单次 launch 开销通常不大，但 decode 每层处理的 token 很少，GPU kernel 本身也很短。40 层累计大量小 kernel 后，CPU launch 和 kernel 之间的空隙会变得明显。


### 5.5 CUDA Graph

CUDA Graph 会把一整段 CUDA 操作序列预先录制下来：

```Plain Text
Kernel A -> Kernel B -> Memcpy C -> Kernel D
```

录制完成后，运行时不再由 Python逐个 launch，而是一次提交整张图：

```python
graph.replay()
```

![图 13](figure-13.png)

CUDA Graph **不会把多个 kernel 融合成一个 kernel**。它只是把多个 launch 预先组织好，减少 CPU/driver 每轮重复提交的成本。

Graph capture 通常要求：数据内容可以变化，但是 buffer 地址和形状不能随便变化。

```Plain Text
kernel 序列固定
tensor shape 固定
tensor 内存地址固定
依赖关系固定
```


---


## 6. Overlap scheduling

`normal_loop()` 在提交当前 batch 后立即处理当前结果；**`overlap_loop()` 把相邻两个 batch 错开：GPU 执行 batch N+1 时，CPU 处理 batch N 的结果。**

![图 14](figure-14.png)

> 图 9：Scheduler host 在准备并启动 batch N+1 后，通过 `copy_done_event(N)` 确认 batch N 的 CPU token 可读，并执行 `_process_last_data(N)`；这段 CPU 结果处理与 Engine stream 上的 `forward(N+1)` 重叠。两次 `wait_stream()` 分别约束 N 结束后的 N+1 准备，以及 drain 中 page table、SWA 映射等 GPU 写入必须排在 N+1 之后。

### 6.1 两条 stream 和一轮重叠

Scheduler 使用 `self.stream` 准备 metadata、分配页和维护表；Engine 使用 `self.engine.stream` 执行模型、采样和 D2H 拷贝。一轮 overlap 的顺序是：

```Plain Text
Scheduler stream：等待上一批 Engine 工作 → 准备 batch N+1
Engine stream：   等待 Scheduler 准备 → 执行 batch N+1
Scheduler stream：等待 batch N+1 Engine 工作 → 处理 batch N 结果
```

`last_data` 是上轮已经启动但尚未 drain 的 batch，`ongoing_data` 是本轮刚启动的 batch。`copy_done_event` 只保证 CPU token 可读；另一个 `wait_stream()` 负责约束 Scheduler 对 page table、SWA 映射和状态的 GPU 写入。


---


## 7. MoE 权重缓存与计算优化

MoE 的权重搬运发生在模型的 MoE layer 和 `OffloadMoeCache` 中。Scheduler 只决定 batch；路由到哪些专家、哪些专家命中 GPU cache，以及缺失专家如何执行，都发生在模型 forward 内。

### 7.1 routed expert 的四种路径

| backend | routed expert 的主要计算位置 | GPU cache 的角色 |
| --- | --- | --- |
| fused | GPU 常驻权重 | 不需要 offload expert cache。 |
| offload | GPU 计算；缺失权重从 CPU host bank 搬到 GPU slot cache | 保存常用 expert，按 LRU 替换。 |
| cpu | CPU executor 计算 decode expert | prefill 仍需要 GPU 层缓冲；decode 不依赖 GPU expert slot cache。 |
| hybrid | GPU 计算 cache hit 和一部分 PCIe fetch miss；CPU 计算剩余 miss | 同时维护 GPU slot cache 和 CPU executor。 |

Qwen3.6 的 routed MoE 先得到 top-k expert IDs 和权重，再执行 shared expert，最后把 routed 输出与 gated shared 输出相加。GPU cache 只管理 routed experts。

### 7.2 Prefill 的双缓冲

Prefill 是长 token extend。对 offload/hybrid 路径，当前层计算需要完整的一层专家权重；如果每层都先同步搬运、再计算，PCIe 传输会暴露在关键路径上。

启用 `moe_prefill_overlap` 后，`OffloadMoeCache` 借用 GPU expert cache 的前 `2 × num_experts` 个 slot 作为两个整层 buffer：

```Plain Text
layer 0 → buffer 0
layer 1 → buffer 1
layer 2 → buffer 0（layer 0 已 release）
```

事件顺序是：

1. 计算 stream 为当前层设置开始拷贝的 fence。
2. copy stream 将下一层专家权重搬入另一 buffer。
3. 计算 stream 等待当前层 buffer ready，再执行专家 GEMM。
4. 当前层 GEMM 完成后记录 release event，允许后续层复用该 buffer。

![图 15](figure-15.png)

> 图 10：Scheduler 只提交 prefill batch；模型内部 copy stream 依次把 layer 0、1、2 搬入 buffer 0、1、0。计算 stream 等待当前层 ready 后执行 expert GEMM，layer 0 的 release event 使 layer 2 可以复用 buffer 0；因此这里的 overlap 是“下一层权重搬运”与“当前层 GEMM”的重叠。


`prefetch_prefill_layer(layer_id + 1)` 负责提前启动下一层搬运；最后一层之后的 prefetch 是 no-op。双缓冲中的权重是本轮 prefill 的临时视图，不是 decode 阶段的持久驻留状态。`moe_prefill_hit_d2d` 可将 cache 中已有的 resident experts D2D gather 到 buffer，只把 miss rows 通过 PCIe 搬入。

历史预热 A/B 测量：在 RTX 5090 上、8K prompt、关闭 decode CUDA Graph 的同一路径中，开启 overlap 为 7058.7 tok/s（1.161 s），关闭为 5018.5 tok/s（1.632 s），差异为 40.7%。这是历史实验结果，不代表当前机器或当前 checkout 的新测量。


### 7.3 Decode cache 与 Hybrid MoE

Decode 每个请求通常只处理一个 token，单层 active experts 很少。GPU cache 用 `slot_for_id` 从 `(layer, expert)` 找 slot，`id_of_slot` 从 slot 找回专家身份；`usage` 选择 GPU victim，Hybrid 路径还用 `expert_recency` 选择本轮应该 fetch 的 miss。

Hybrid decode 的一层可以压缩成：

```Plain Text
fused_topk
  → ensure_experts_hybrid
  → 路由拆成 GPU routes / CPU overflow
  → CPU decode_submit
  → GPU copy_missing + expert GEMM
  → CPU decode_sync
  → gpu_routed + cpu_routed
```

![图 16](figure-16.png)

> 图 11：`ensure_experts_hybrid` 将 route 改写为 GPU slot 或 `-1`。GPU 路径直接使用 cache hit，并通过 `copy_missing` 搬入受 fetch 策略选中的 miss；CPU executor 接住其余 overflow。`decode_submit` 先启动 CPU 工作，使它与 GPU 的 PCIe fetch 和 expert GEMM 重叠，最后在 GPU 上计算 `gpu_routed + cpu_routed`。


缺失专家中只有受 fetch cap 或带宽比例选中的部分进入 GPU cache；剩余 route 的 expert ID 会改成 `-1`，交给 CPU executor。CPU executor 将 activation、IDs 和权重复制到 pinned host buffers，计算完成后再把结果拷回 GPU。GPU 和 CPU 两路结果最后相加。

自动 fetch 使用带宽 profile 计算 fetch fraction；代码默认按 `pcie_overlap / (pcie_overlap + cpu_overlap)` 选择整数 fetch 数，没有 profile 时回退到固定 cap 1。GPU victim 的 `usage` 和 PCIe fetch 的 `expert_recency` 是两套独立的 LRU 信号。

历史 RTX 5090 同路径数据：

| 条件 | 吞吐 |
| --- | --- |
| offload | 86.29 tok/s |
| hybrid auto + overlap + graph | 107.61 tok/s |
| hybrid，关闭 overlap | 88.36 tok/s |
| hybrid，fetch=1 | 95.61 tok/s |
| hybrid，fetch=0 | 63.40 tok/s |
| hybrid，关闭 graph | 26.02 tok/s |

这些数字来自 2026-09-04 的 warmed decode 实验；不同 backend 的 BF16 路径可能产生不同 token，比较重点是吞吐、延迟和 route/stat invariant。已有机制测试为 38 passed，CPU MoE 数值测试为 20 passed。
