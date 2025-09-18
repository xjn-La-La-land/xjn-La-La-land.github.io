---
title: Intel AMX组会报告
date: 2025-09-18 20:55:41
tags: CS-notes
comments: false
excerpt: Intel AMX架构介绍、香山上AMX的移植、AMX GEMM算子的性能优化
mathjax: true
---

## Intel AMX的架构介绍

[Intel AMX](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html)(Advanced Matrix Extension)是Intel Xeon Scalable 处理器核心上的**专用硬件模块**，旨在加速**依赖矩阵操作的深度学习训练和推理工作负载**。

### 基本结构

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDRjODk2MGNkNThhNzZiZDg0NGEzMTIxODliM2M3NzdfeU5zN01SUkI1REEwSkQ4bnVOTkR0MlNZWUtVZGVNaXFfVG9rZW46Q2Via2JVWkE0b2xXdTN4VERkZGNEZU9ybmRnXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

1. Tile寄存器：8个二维矩阵寄存器，每个Tile寄存器最大容量为16行×64B=1KB；、
2. Tile控制寄存器（TILECFG）：存储每个Tile启用的行数和列数；
3. TMUL运算单元：计算Tile矩阵乘的脉动阵列，支持两种数据类型：1）**int8×int8=>int32**；2）**bf16×bf16=>fp32**.
4. 一致性内存接口：AMX的访存通路，load/store Tile数据；

### 指令

12条基础AMX指令

|                 | **指令**                                                     | **操作**                                | **吞吐（Throughput）** | **延迟（Latency）** |
| --------------- | ------------------------------------------------------------ | --------------------------------------- | ---------------------- | ------------------- |
| 配置            | **LDTILECFG**                                                | 将Tile配置从内存加载到TILECFG中         | Not Relevent           | 204                 |
| **STTILECFG**   | 将TILECFG中的Tile配置存储到内存                              | Not Relevent                            | 19                     |                     |
| **TILERELEASE** | 将TILECFG中的Tile配置释放                                    | Not Relevent                            | 13                     |                     |
| 访存            | **TILELOADD**                                                | Load data into tile.(from L1D)          | 8                      | 45                  |
| **TILELOADDT1** | Load data into tile **with hint to optimize data caching**.(from L2) | 33                                      | 48                     |                     |
| **TILESTORED**  | Store tile.（to L1D）                                        | 16                                      | Not Relevent           |                     |
| 运算            | **TDPBSSD**                                                  | int8 tile点积运算，并累加到int32 tile上 | 16                     | 52                  |
| **TDPBSUD**     | int8/uint8 tile点积，并累加到int32 tile上                    |                                         |                        |                     |
| **TDPBUSD**     | uint8/int8 tile点积，并累加到int32 tile上                    |                                         |                        |                     |
| **TDPBUUD**     | uint8 tile点积运算，并累加到int32 tile上                     |                                         |                        |                     |
| **TDPBF16PS**   | bf16 tile点积运算，并累加到fp32 tile上                       |                                         |                        |                     |
| 其他            | **TILEZERO**                                                 | Zero tile.                              | 0                      | 16                  |

### 峰值算力

TMUL单元每16 cycle能计算一次Tile点积累加（一条TDP指令），所以其峰值算力为2MNK/16 (op/cycle)

- INT8：2×16×64×16/16 = 2048 op/cycle
- BF16：2×16×32×16/16 = 1024 op/cycle

### 编程方式

1. AMX指令内联汇编：提供了[Intrinsics接口](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

   1. | 1234567 | TDPBSSDTDPBSUDTDPBUSDTDPBUUDTILELOADDTILELOADDT1TILESTORED | **void** _tile_dpbssd(__tile dst, __tile src1,   __tile src2);**void** _tile_dpbsud(__tile dst, __tile src1,   __tile src2);**void** _tile_dpbusd(__tile dst, __tile src1,   __tile src2);**void** _tile_dpbuud(__tile dst, __tile src1,   __tile src2);**void** _tile_loadd(__tile dst, **const** **void** *base, **int** stride);**void** _tile_stream_loadd(__tile dst, **const** **void** *base, **int** stride);**void** _tile_stored(__tile src, **void** *base, **int** stride); |
      | ------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
      |         |                                                            |                                                              |

2. 矩阵运算算子库：[Intel oneAPI](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-1/base-online-offline.html#BASE-ONLINE-OFFLINE)中的MKL库(Math Kernel Library)提供了很多算子，如GEMM算子：`cblas_gemm_s8u8s32`, `cblas_gemm_s16s16s32`, `cblas_gemm_bf16bf16f32`, `cblas_gemm_f16f16f32`

### RTFM

- [Intel Software Developer’s Manual](https://www.intel.com/content/www/us/en/content-details/782158/intel-64-and-ia-32-architectures-software-developer-s-manual-combined-volumes-1-2a-2b-2c-2d-3a-3b-3c-3d-and-4.html?wapkw=intel 64 and ia-32 architectures software developer's manual&docid=782158) (CHAPTER 19)：AMX架构介绍
- [Intel Optimization Reference Manual](https://www.intel.com/content/www/us/en/content-details/821612/intel-64-and-ia-32-architectures-optimization-reference-manual-volume-1.html) (CHAPTER 20)：AMX矩阵乘性能优化
- [Intel oneAPI Math Kernel Library for C](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/overview.html)：MKL库算子

## XiangShan上的移植实现

在RISC-V XiangShan处理器上移植了INT8类型的TDP，TILELOADDT1和TILESTORED，实现了AMX最基础的运算和访存功能。

我们在CPU后端执行单元中添加了一个模块TMU(Tile Matrix Unit)，用于AMX指令进行独立的运算和访存。

### 控制通路

- 所有AMX指令进入整数发射队列IQ3，发射到TMU中；
- 在IQ3中控制AMX指令顺序发射，从而保证AMX指令顺序执行；
- 内存一致性如何保证？在Dispatch流水级，检查TILELOADDT1/TILESTORED指令和其他访存指令是否同时出现在ROB中，如果出现，则将后面一条指令阻塞在Dispatch级，直到前一条指令提交（并且sbuffer排空）。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjlhMzM5NmNiZjFlY2ZmNjMxNzQyNzAxZDQxYjhkZDlfWEFtaFVsckV3c1EzMFh0bHdqbG5UWWJkRjhoMUZ6OERfVG9rZW46SzBxNGIxREEwb2ZqTnp4RDVOd2NhakRNbkRnXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

### 访存通路

- TILELOADDT1：TMU连接到l2-cache的TileLink总线，发送Get请求读数据；
- TILESTORED：TMU连接到sbuffer，将数据先写入sbuffer，然后由sbuffer管理并写入dcache。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=MDRiN2Y1ZjZhODgzNDk3N2Y5M2IzNDc5NmU0ZjM1NjJfS1FZMFl2dDlwd1hmeTl1MG5CNzZxVjV6TE0zcHBJWGFfVG9rZW46QWFvYWJIcFc3b0ROQUx4YjFXS2NQSXZSbjBiXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

### TMU单元微结构

- Tiles寄存器堆：每个Tile是一个SRAM，16个set，每个set 64B，每周期可以读写一行的数据；
- TDPUnit：16×16个点积累加单元（DPAUnit）构成的脉动阵列，计算tileA × tileB + tileC。可以实现17 cycle的TDP吞吐；
- TLSUnit：用于管理AMX访存请求的循环队列，TILELOADDT1/TILESTORED按行拆分成16个请求进入队列。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=YWQ5ZDAyZjBlMzY5OGYzMDhhZjQ4YjIwYmUwZWFhODBfMjRQMmp4ekcwam56bU44SjQySEJoRGVlbkFkRWh2ZjZfVG9rZW46S2dkSWJZZzFibzQ4b0d4M0N1SmNhM2t5bmNlXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

## GEMM算子的性能分析与优化

Intel oneMKL库的GEMM算子，在我们的测试中，无论是单核还是多核，对AMX的利用率都偏低（单核~50%，多核随核数增加显著降低）。我们尝试手写GEMM算子，看看能不能跑出更高的AMX利用率。如果能达到比较理想的利用率，说明oneMKL库的算子还有优化空间；如果达不到，我们就看看利用率被什么限制住了，以及AMX架构应该怎样设计。

### 单核优化

1. Register Blocking：8个Tile寄存器如何分配？
   1. 1x2 blocking: `1A*2B=>2C`, 3/2 tileload per TDP
   2. 1x4 blocking: `1A*4B=>4C`, 5/4 tileload per TDP
   3. 2x2 blocking: `2A*2B=>4C`, 4/4 tileload per TDP
   4.  由于Tile寄存器数量的限制，2A2B4C的寄存器分配就是最优的方案了。

   5. ![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTVlY2QxMzA5N2JhM2UxZDdhOTA5YTkyYmJhZjNiNmZfNTNUS1dCakVQS1VpSmV4TDFTTHZnbWY1QU1NUzBqcEZfVG9rZW46VFdlWmJtZmRjb0lXSG14dXE2dmN3QWhDbjRnXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

   6.  这样GEMM的三层循环结构如下：

   7. 暂时无法在飞书文档外展示此内容

   8.  在最内层核心循环中：

   9. 对矩阵A，会重复多次访问32×K的一块数据。我们希望A的数据直接驻留在L1-D中，用TILELOADD来加载tileA；
   10. 对矩阵B，整个B的数据会被顺序访问，我们希望B的数据能在L2中驻留，用TILELOADDT1加载tileB；
2. Cache Blocking：矩阵规模增大时，Cache容量不足导致Cache miss增加。所以在GEMM三层循环外再加上一层分块，使得分块大小与L1-D/L2容量匹配。
   1. ![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGJmOTI0N2MyNWEzOWFlYmVjYTg5ODM5MGY3NDU5NGZfVGpQWWJuakxXSndpRWlGNnA4OXFpdWVGZWJnMlpzZzhfVG9rZW46Q3ZFWWJMeXlQb3FNUFJ4elA1cGN2SzNtbmJoXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

   2.  经过实际测试，TN=512, TK=1472是一个比较好的Cache分块大小，而M方向是否分块对性能没有明显影响。此时A分块大小32×TK=46KB，能够将 L1-D(48KB) 充分利用；B分块大小TN×TK=736KB，能保证fit in L2(2MB).
3. Packing：如果tileload加载的矩阵分块每一行在内存中紧密排列，就能获得更好的tileload性能（**Strided tileload is slower than dense/compat tileload**）。我们在缓冲区中，将矩阵A, B的数据按照访问顺序重新排列，使得每一次tileload可以访问连续的1KB数据。
4. Software Prefetch：在GEMM循环中插入软件预取指令，进一步提高单核的AMX利用率。
   1. 将下一个要访问的A和C分块预取到L1D，
   2. 将下一个要访问的B分块预取到L2；

经过上面一系列的性能优化，我们的手写GEMM算子对AMX的利用率在大多数情况下已经比oneMKL算子更好了。

- M↑, N=TN=512, K=TK=1472，利用率能稳定在75%.

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=YTExYTZiYmI3NjA3ZmIxZjQ5ZWRhZmQ3YTJjZjgxZjdfN0RBcE96eFVLRWJldEZTRjdyNFh0b21ab0xVaE9hM0xfVG9rZW46Rlk1MWJuZDVwb1ZVNFZ4eEwwN2NiMDd4bkVlXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

- M↑, N=TN=512, K↑，fit in L3时，利用率能稳定在 70%；overflow L3时，下降到65%。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=YTM0NzVjYzY4MWM0NDE0YmQ4ZDM4OWQwNWRiMmQxMjFfMHRtaHVEdjZTc3hjMHVVRjl1dkNHVHplaTB4N2ZKOFBfVG9rZW46VW9HTGJRZXdXb2o3T294YW1OTmNoU0ZmbkJoXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

- M↑, N↑, K↑，fit in L3时，利用率能稳定在 68%；overflow L3时，下降到65%。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=YzkwZjRmM2JmYWZkYjE1N2EwMjU3ZmQ5YjAyMGRiODRfZlREZ1Bpc0g5SWtKSHFGaG9LemU4djN1QWZSclNnNHZfVG9rZW46QjJjamJPMGZRb0VHU0Z4U0tVeWNxN0dvbkplXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

我们基本清楚了在单核心场景下，面对不同的矩阵规模，限制AMX利用率的各种因素。总结在下表中（表格中AMX的利用率是我们优化后能够达到的最好表现）

| **矩阵规模**          | **矩阵形状**            | **AMX利用率（优化后）**                    | **限制因素**                                 |
| :-------------------- | :---------------------- | :----------------------------------------- | :------------------------------------------- |
| A & B & C fit in L2   | M=256, N=256, K=1472    | ~82%                                       | 数据相关导致的tile资源冲突                   |
| A & B & C fit in L3   | M↑, N=TN=512, K=TK=1472 | ~75%                                       | L3 => L2 加载 A & C 数据的开销，L3带宽       |
| M↑, N=TN=512, K↑      | ~70%                    | L3 => L2 加载 A & B & C 数据的开销，L3带宽 |                                              |
| M↑, N↑, K↑            | ~68%                    | L3 => L2 加载 A & B & C 数据的开销，L3带宽 |                                              |
| A & B & C overflow L3 | M↑, N↑, K↑              | ~65%                                       | DDR => L3 加载 A & B & C 数据的开销，DDR带宽 |

我们发现，AMX的性能（利用率）严重依赖于矩阵的大小与形状，原因在于CPU的缓存系统非常复杂，要充分利用缓存系统的性能，矩阵数据的排列、分布还需要满足一系列的条件。

目前手写的GEMM算子对AMX的利用率在65%~70%，还是有30%的性能损失，主要受到**tile寄存器的数据相关**和**L3带宽**的限制。

### 多核优化

在多核场景下，将GEMM算子适配到多线程上，总体的思路就是**以Cache分块为单位，将计算任务均匀分配给不同的线程**。多核优化的关键在于将计算任务均匀地分配到每个核心，同时尽量保证单个核心的效率。

- 一方面，分块粒度（TM, TN）不能太大，否则很可能每个核计算任务不均匀；
- 另一方面，分块粒度（TM, TN）也不能太小，否则单核心的效率会太低。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=YjUyNDYyZTg3Yzg3MzE1YzMzMDFjZTllZTlhMDUyYzBfWjVFMFdSREd0ZVB1YjQ3S0Y5QkNjNGVkOVVRc016RlpfVG9rZW46SzRHUGJrbU9Kb2N1aHd4TG9lZWNnYjdXblRnXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

### 针对NUMA结构优化

当核数进一步增多，来到多个NUMA结点的场景，我们希望每个node在GEMM循环中只访问自己的本地内存，这就要求矩阵数据在内存中根据NUMA结构进行分布。

具体而言，我们将矩阵B在每个NUMA结点内存中复制一份，而矩阵A, C在M方向拆分，分配到各个NUMA结点上。（类似于用空间换时间？）

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=M2MwYmZlZTczYTAzY2FhOGRjNDc5NWMzOWVjNDkxZWFfN25qOWU5bmRkQnFMcTQzcDZrRTg1VkNENWtYRHQ0QXVfVG9rZW46UkpKWWI1a2ZGb0ozUHl4UFZvOGNpdklvbnZoXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=MTZlNjk0MjYyOGIyMTAyM2RjYjk4MWUzMjk3MDMyMjhfYWlyR0JlVUt2cGkwNlFPaWlSUDZJN1ZjMkxmUVJzRWJfVG9rZW46Sjl2NmJyOVR3b1RtSkJ4ajViRmNkZzBTblZkXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

目前在多核+多NUMA结点下，手写GEMM算子对AMX的利用率在30%~40%。主要受到**DDR/L3的带宽**限制，而且相比单核，多核场景下L3带宽限制会更加严重。

### AMX架构优化方向

- 增加Tile寄存器数量，Register Blocking可以做到更大，让tileload/TDP的比值更小，单核AMX利用率可以进一步提高。
- 多核场景下利用率的降低主要是L3带宽的限制，增大L3到L2的带宽可以提高AMX多核的性能。

## LLM推理加速效果

目前的开源大模型推理框架中，**[llama.cpp](https://github.com/ggml-org/llama.cpp)** 和 **[ktransformers](https://github.com/kvcache-ai/ktransformers)** 已集成实现成了Intel AMX对cpu推理的加速。我们下面就来分析一下两个项目中AMX加速相关的代码，并且跑一下代码，**看看它们对AMX的利用效率如何，以及实际的加速效果是否符合预期。**

### Llamma.cpp

#### 性能实测

使用llama-bench工具来测试模型推理的性能，测试平台是**Xeon PLATINUM 8580** (120核)。在后端配置上，对比测试三组指令集：1) AMX + AVX512 + AVX256；2) AVX512 + AVX256；3) AVX256；

**测试模型选择Q8_0量化类型，**包括：

- Llama-3.2-1B-Instruct-Q8_0.gguf，模型权重1.22 GB
- Llama-3.2-3B-Instruct-Q8_0.gguf，模型权重3.18 GB
- Llama-3.1-8B-Instruct-Q8_0.gguf，模型权重7.95 GB

测试512 tokens的prefilling：

暂时无法在飞书文档外展示此内容

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjA1ZjZkZDQxOTg1N2FiNmM3YTMxMzZlOWFkOWY4NDBfZFNBWWRBbFJpOHdHYkhkOW13b2s5YldSYWVRU1BJRTJfVG9rZW46VUhEdmJVTldSb1ZLdHl4Z0RSYWNNRDlGbkFjXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

在INT8上，AMX的计算部件峰值性能是2048 op/cycle，而AVX512 VNNI是256 op/cycle，**AMX理论性能约为AVX512 VNNI 的8倍。**但是我们测出来AMX的加速性在单核时只有AVX512的2倍左右，而且随着核数增加，性能提升会变缓，在64核时甚至只有AVX512的不到1.5倍，显然AMX的利用率是不太理想的。

#### 性能优化

我们发现llama.cpp的AMX GEMM算子有几个问题可能会影响整体的性能：

1. 对于Q8_0等量化类型，由于BLOCK_K=32，导致A的分块大小是16*32，B的分块大小是8*64，都只使用了一半的tile寄存器。这样相当于一条tdp指令只有1/4的计算是有效的，这会导致AMX利用率明显下降；
2. INT8算子无法直接在Tile寄存器上累加结果，需要频繁的tilestore将结果搬到内存；而且GEMM循环中还插入了对结果的反量化和累加计算，带来额外的开销。
3. 在计算之前，输入A需要从float32转换成Q8_0量化类型，并且这个转换没有做并行优化；

对于量化类型，使用INT8类型运算对AMX的限制很大。我们不妨考虑换一种思路，**使用AMX支持的另一种数据：BF16类型来运算**。使用BF16解决了INT8算子的前两个问题：

- BF16不是量化类型，避免了分块大小受到量化数据的限制，可以充分使用Tile寄存器的容量；
- BF16经过TDP计算之后直接是FP32，结果可以在Tile寄存器上累加；

我们需要**将输入和权重数据都转换成BF16来计算。**具体而言，对于权重矩阵B，我们在重排之前将其反量化成BF16数据；对于输入矩阵A，我们在GEMM循环之前对其进行FP32到BF16的转换。FP32转换成BF16，可以通过AVX向量指令完成。然后，GEMM最内层循环，对K方向的遍历将结果累加在Tile寄存器中，然后直接存到输出矩阵C中。

最后，将上面的手写GEMM算子接入到这里。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=MGM4ODJhMTE5MDdiMDViODEzY2Q0MjQ5ODQ5YmY3YmJfTWZrcnhRQVhOR2dSQzZrYTVhalRYUjFLNkw1TVp1cXhfVG9rZW46STZUY2J4QWdWb0dld014cnNCVGNqdEhabkdjXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

我们的AMX BF16算子相比原始的INT8算子有明显的性能提升。在核数比较少的时候能达到AVX512算子性能的4~5倍。**但是，AMX算子的多核扩展性不好，**AMX的多核加速比明显低于AVX256和AVX512.

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=NmU3NDU5OTVkMTMxOWViYmIwZDFmZjcwZDFmODRjZWNfTmtXM084d1gwcDdKNTJrTnpaWWZHcFdLcWZlMnU3MVRfVG9rZW46UVlYaWJYTlE2b0RjS2h4cUpNTGMxR3Y5bkpoXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)

目前还需要解决几个问题：

1. LLM推理负载中会出现行数为1的输入矩阵，此时GEMM退化成向量×矩阵，需要加入AVX的协同优化；
2. 矩阵规模大部分是M,N=512,512，或者M,N=512,2048这种大小，核数进一步增加，面临计算任务不够分的问题，导致AMX多核扩展性不好**。**

### Ktransformers

#### 性能实测

因为ktransformers不支持在纯CPU环境上运行，所以只简单测试了其中AMX GEMM算子的性能。

![img](https://pcn7jbsuk8nv.feishu.cn/space/api/box/stream/download/asynccode/?code=YmM1MzFmMmE2YzIyN2U0Yzk0MDBjNGM2N2ExN2M0ZmZfODlyM0dzV3JRNlJscEh6NGNvNXI5a3R1OWdsYnFmRHdfVG9rZW46UVA0RWI3OVBHbzNXekt4RE1iY2N2NGlxbmRnXzE3NTgxOTg5NjA6MTc1ODIwMjU2MF9WNA)