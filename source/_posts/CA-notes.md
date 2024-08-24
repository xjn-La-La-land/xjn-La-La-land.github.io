---
title: CA-notes
date: 2024-08-18 10:01:03
tags: CS-notes
comments: false
excerpt: 《计算机体系结构基础》复习笔记
mathjax: true
---


## 什么是计算机体系结构？
* CA是描述计算机各组成部分及其相互关系的一组规则和方法，是程序员所看到的计算机属性。
* CA研究内容包括**指令系统结构(ISA)**和**计算机组织结构**。
* 微体系结构是微处理器的组织结构，并行体系结构是并行计算机的组织结构。
* 冯诺依曼结构的**存储程序和指令驱动执行原理**是现代计算机体系结构的基础。


## 第1章：引言
&emsp;&emsp;计算机系统的四个层次
```makrkdowm
                        |----------|
                        |   应用   |
                        |----------|
                API(C, Java, Javascript...)
                        |----------|
                        |    OS    |
                        |----------|
                ISA(X86, ARM, MIPS, RISCV...)
                        |----------|
                        |   硬件   |
                        |----------|
                          工艺模型
                        |----------|
                        |  晶体管  |
                        |----------|
```

&emsp;&emsp;衡量计算机的指标：**性能，价格，功耗**

###### 性能
* 性能最本质的定义：完成一个任务所需的时间
* 完成一个任务的时间 = 指令数 * CPI * T

> MIPS （百万条指令/秒）
> MFLOPS （百万条浮点指令/秒）

* 处理器微结构研究的主要内容：如何降低 CPI，提高 IPC
* IPC = $\frac{总指令数} {总周期数}$

* 影响 IPC 的因素：处理器微结构设计、编译器、指令功能

###### 价格

* 14nm 开始采用 FinFET工艺，晶圆生产成本大幅提高，单个晶体管的成本不再成指数下降。

###### 功耗

* 动态功耗优化：升级工艺，工艺可以降低电容和电压；
* 静态功耗优化：可以选择低功耗工艺。

> 体系结构设计的基本原则：平衡性，局部性，并行性，虚拟化

## 第2章：指令系统

&emsp;&emsp;计算机系统的层次

```markdown
                    _________________________
                   |         应用软件        |  上层软件
                   |—————————————————————————|
                   | 操作系统、编译器、虚拟机 |  基础软件
                   |—————————————————————————|
                   |         指令系统        |
                   |—————————————————————————|
                   |        微体系结构       |  逻辑硬件
                   |—————————————————————————|
                   |        电路与器件       |  物理硬件
                   |_________________________|

```

&emsp;&emsp;指令系统的设计原则：**兼容性，通用性，高效性，安全性**。

* RISC在某种意义上就是编译技术推动的结果。为使编译器有效地调度指令，**至少需要16个通用寄存器**。

&emsp;&emsp;依据指令长度的不同，指令系统分为：CISC, RISC和VLIW.

> CISC：指令长度可变；
  RISC：指令长度固定；
  VLIW：多条同时执行的指令的组合。其同时执行的条件由编译器指定，无需硬件判断。

* RISC的简化：
    * 简化指令功能，指令的执行周期短；
    * 简化指令编码，使译码变简单；
    * 简化访存类型，访存只通过load/store指令实现；
**运算指令不访存，访存指令不运算！**
**最本质的特征：通过load/store结构简化了指令间关系，CPU只要通过寄存器号的比较就能判断指令之间有没有数据相关性。**

* X86处理器中将CISC指令译码为类RISC的内部操作。

* VLIW可以简化硬件实现，但增加了编译器的设计难度。

&emsp;&emsp;存储管理：**连续实地址，段式存储管理，页式存储管理。**

* 段式存储中每段有不同的起始地址和长度，但段内地址仍需要连续。

* 段页式管理：
```markdown

                |————————————————————————————|
                | 段号  | 虚拟页号 | 页内偏移 |
                |————————————————————————————|
```

&emsp;&emsp;运行级别：**用户态，核心态**

* LoongArch指令系统定义了PLV0 ~ PLV3四个权限等级，由 `CSR.CRMD` 的 PLV 域确定。核心态程序运行在 PLV0 级，用户态程序运行在 PLV3 级。

* LoongArch指令系统运行级别：调试模式，主机模式（包含 Host-PLV0 ~ Host-PLV3），客户机模式（包含 Guest-PLV0 ~ Guest-PLV3）。处理器复位后处于 Host-PLV0 级。

 | 指令集    | 内存空间                               | IO空间                                 |
 |:---------:|:--------------------------------------:|:--------------------------------------:|
 | X86       | 独立的内存空间，使用一般的访存指令访问 | 独立的IO空间，使用专门的in/out指令访问 |
 | MIPS      | Cached                                 | Ucached                                |
 | ARM       | Normal                                 | Device                                 |
 | LoongArch | 一致可缓存                             | 强序非缓存                             |

&emsp;&emsp;MIPS/ARM/LoongArch：内存空间和IO空间映射到同一个系统内存空间，都通过load/store指令访问，区分不同的存储访问类型。


根据指令使用数据的方式，指令系统可分为堆栈型、累加器型和寄存器型。

 | 指令系统类型                    | 特点                                                                                 |
 |:-------------------------------:|:-------------------------------------------------------------------------------------|
 | 堆栈型（零地址指令）            | 操作数都在栈顶，运算指令不需要指定操作数，对栈顶数据进行运算并将结果压回栈顶。       |
 | 累加器型（单地址指令）          | 运算指令一个操作数在指令中指定（在内存中），另一个操作数在累加器中，结果写回累加器。 |
 | 寄存器-存储器型                 | 每个操作数由指令显示指定，操作数为寄存器和内存单元。                                 |
 | 寄存器-寄存器型（load-store型） | 每个操作数由指令显示指定，但除了访存指令外，其他指令的操作数只能是寄存器。           |


* 对齐访问：对该数据的访问起始地址是其数据长度的整数倍。

* 大小尾端
    * 大尾端：最高有效字节在低地址；
    * 小尾端：最高有效字节在高地址；

    ```markdown
    一个 32bit 的 int 类型数据A，在内存中的地址为 addr

    大尾端：                       小尾端：
    |---------|——————————|         |---------|——————————|
    | addr+3  | A[7:0]   |         | addr+3  | A[31:24] |
    |---------|——————————|         |---------|——————————|
    | addr+2  | A[15:8]  |         | addr+2  | A[13:16] |
    |---------|——————————|         |---------|——————————|
    | addr+1  | A[23:16] |         | addr+1  | A[15:8]  |
    |---------|——————————|         |---------|——————————|
    | addr    | A[31:24] |         | addr    | A[7:0]   |
    |---------|——————————|         |---------|——————————|

    ```




* 流程控制语句
    * 辅助控制语句：
    ```
                   C 语言控制流语句            汇编代码
                goto, break, continue    b 无条件跳转指令
                return                   返回值写入，jr $ra(ret)
                if~else                  beqz $t0, jump_loc
                switch~case              通常会被映射为跳转表的形式
                for, while, do~while     blt 条件跳转
    ```


## 第3章：特权指令系统

##### 异常分为6种：
  * 外部事件(不是指令直接引发的异常)，即**中断**
  * 指令执行中的错误(`ine, ale, ade(adef, adel), pif, pil, pis, pme, ppi`)
  * 数据完整性问题(存储器校验错误)
  * 地址转换异常(TLB重填异常)
  * 系统调用和陷入(RISCV: `ecall`; LoongArch: `syscall`)
  * 需要软件修正的运算：常见的是浮点指令导致的异常


&emsp;&emsp;**精确异常**：发生任何异常时，被异常打断的指令之前的指令都执行完，而该指令之后的所有指令都像没执行一样。

&emsp;&emsp;**采用专门的异常返回指令**

<pre>
<code>
  LoongArch: ertn
  RICSCV:    eret
  x86:       iret
</code></pre>

&emsp;&emsp;异常返回指令需要**原子地**完成恢复等级权限、恢复中断使能状态、跳转至异常返回地址等多个操作！

##### 异常处理流程
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-5mvr{border-color:#ffffff;color:#ffffff;font-family:"Trebuchet MS", Helvetica, sans-serif !important;text-align:left;
  vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-5mvr">异常处理准备</th>
    <th class="tg-5mvr">调整CPU的权限等级，关闭全局中断<br>硬件保存异常现场的部分信息<br>记录异常来源</th>
    <th class="tg-5mvr">0 --&gt; CRMD.PLV, 0 --&gt; CRMD.IE<br>CRMD.PLV --&gt; PRMD.PPLV, CRMD.IE --&gt; PRMD.PIR<br>异常一级编号 --&gt; ESTAT.Ecode<br>异常二级编号 --&gt; ESTAT.EsubCode<br>异常指令的机器码 --&gt; BADI<br>引起异常的虚地址 --&gt; BADV</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-5mvr">确定异常来源</td>
    <td class="tg-5mvr">异常处理程序根据异常编号跳转到不同的处理函数入口</td>
    <td class="tg-5mvr">入口页号来自于 EENTRY 寄存器；<br>页内偏移来自于 ECFG.VS 和 异常编号（向量化中断）</td>
  </tr>
  <tr>
    <td class="tg-5mvr">保存执行状态</td>
    <td class="tg-5mvr">由软件将通用寄存器、状态寄存器的值保存到栈中</td>
    <td class="tg-5mvr"></td>
  </tr>
  <tr>
    <td class="tg-5mvr">处理异常</td>
    <td class="tg-5mvr">跳转到对应的异常处理程序异常处理</td>
    <td class="tg-5mvr"></td>
  </tr>
  <tr>
    <td class="tg-5mvr">恢复执行状态并返回</td>
    <td class="tg-5mvr">将栈中保存的异常现场恢复，最后执行异常返回指令 ertn</td>
    <td class="tg-5mvr"></td>
  </tr>
</tbody>
</table>

* 异常嵌套：只有优先级更高的异常才能进行嵌套，低优先级/同优先级的异常只能等待当前异常处理完成。

* 两类异常的出现机会最多：**中断** 和 **缺页异常**

* 两种中断传递机制：**中断线** 和 **消息中断**（通过总线传递中断消息）

* LoongArch的中断优先级：`IPI` > `TI` > `PMI` > `HWI7 ~ HWI0` > `SWI1 ~ SWI0`。

* LoongArch虚拟地址空间大小：对于 PLV0 级来说，LA32架构下虚拟地址空间大小为 $2^{32}$ 字节，LA64架构下虚拟地址空间大小为 $2^{64}$ 字节（但存在空洞）。


* TLB 表项需要记录什么？

<pre>
  |--------------|------|---|--------|---|
  |     VPPN     |  PS  | G |  ASID  | E |
  |     19       |  1   | 1 |   10   | 1 |
  |--------------|------|---|--------|---|
   ---------------|------|-------|------|----|----|----|----|
        PPN0      | RPLV0|  PLV0 | MAT0 | NX0| NR0| D0 | V0 |
        20        |  1   |   2   |  2   | 1  | 1  | 1  | 1  |
   ---------------|------|-------|------|----|----|----|----|
   ---------------|------|-------|------|----|----|----|----|
        PPN1      | RPLV1|  PLV1 | MAT1 | NX1| NR1| D1 | V1 |
        20        |  1   |   2   |  2   | 1  | 1  | 1  | 1  |
   ---------------|------|-------|------|----|----|----|----|

  E: 表示该TLB表项是否存在；
  V: 表示该页表项是有效的且被访问过的。
</pre>


* TLB 地址翻译过程

<pre>
  |————————————————————————————————————————————————————————|
  |使用待查虚地址 vaddr 与 ASID.ASID中的 asid，在 TLB 中查找。|
  |————————————————————————————————————————————————————————|
                             ||
                             ||
                             \/
  |————————————————————————————————————————————————————————|
  | TLB 表项的 E == 1
  | PS = 0, vaddr[31:13] == VPPN; PS = 1, vaddr[31:22] == VPPN[18:9]
  | TLB 表项的 G == 1 或 asid == ASID
  |————————————————————————————————————————————————————————|
                ||hit                           ||miss
                \/                              \/
  |————————————————————————————————|     |————————————————————|
  | 若 V = 0，                     |     | 触发TLB重填异常 tlbr |
  | 触发页无效异常 pif, pil, pis    |     |————————————————————|
  |————————————————————————————————|
                ||
                \/
  |————————————————————————————————|
  | 若访问权限不满足，              |
  | 触发页权限等级不合规异常 ppi     |
  |————————————————————————————————|
                ||
                \/
  |————————————————————————————————|
  | 检查访问类型是否满足，           |
  | load, NR = 1 --> pnr           |
  | store, D = 0 --> pme           |
  | fetch, NX = 1 --> pnx          |
  |————————————————————————————————|
                ||
                \/
  |————————————————————————————————|
  | 取出命中的 物理页框号 PPN        |
  |        和 存储访问类型 MAT      |
  |————————————————————————————————|
</pre>


&emsp;&emsp;C语言伪代码描述一台64位LoongArch机器上的TLB进行访存虚实地址转换的过程（包含TLB地址翻译相关异常的判定过程）
```c
// TLB数据结构定义  
typedef struct TLBEntry {  
    bool v;                 //有效
    bool nr;                //不可读
    bool d;                 //脏
    bool nx;                //不可执行
    bool rplv;              //受限权限等级使能
    unsigned int plv;       //权限等级
    unsigned int mat;       //存储访问类型
    unsigned long long ppn; //物理页号
} TLBEntry;  
typedef struct Access {
    unsigned long long vaddr;
    unsigned int type;               //Inst/Load/Store
}Access;
  
// TLB进行访存虚实地址转换的过程  
void translate(Access access, unsigned long long &paddr) {  
    TLBEntry *tlbEntry = getTLBEntry(access.vaddr);      // 从TLB中获取对应虚地址的条目，包含查STLB、MTLB，选择奇偶页的过程，具体过程略
    if (tlbEntry == NULL){                             // TLB重填异常  
        raiseException(TLBR);
        return;
    } else if (!tlbEntry->valid) {  
        if(access.type == Inst )  raiseException(PIF);                  // 取指操作页无效例外
        else if(access.type == Load) raiseException(PIL);          // load操作页无效例外
        else raiseException(PIS);                                                 // store操作页无效例外
        return;
    }else if (tlbEntry->rplv == 0 && csr.crmd.plv > tlbEntry->plv || tlbEntry->rplv == 1 && csr.crmd.plv != tlbEntry->plv) {
        raiseException(PPI);                                                         // 页权限等级不合规例外
        return;
    }else if(access.type == Inst && tlbEntry.nx == 1) {
        raiseException(PNX);                                                       // 页不可执行例外
        return;
    }else if(access.type == Load && tlbEntry.nr == 1) {
        raiseException(PNX);                                                       // 页不可读例外
        return;
    }else if(access.type == Store && tlbEntry.d == 0) {
        raiseException(PNX);                                                       // 页修改例外
        return;
    }  
    *paddr = tlbEntry->pageFrameAddress + (access.vaddr & PAGE_OFFSET_MASK); // 计算物理地址  
}
```

## 第4章：软硬件协同

&emsp;&emsp;LoongArch ABI

   | 寄存器编号 | 助记符    | 使用约定                                 |
   |:-----------|:----------|:-----------------------------------------|
   | 0          | `zero`    | 总是为0                                  |
   | 1          | `ra`      | 子程序返回地址                           |
   | 2          | `tp`      | Thread Pointer，指向线程私有存储区(pcb?) |
   | 3          | `sp`      | 栈指针                                   |
   | 4~11       | `a0 ~ a7` | 子程序的前8个参数                        |
   | 4~5        | `v0, v1`  | `a0, a1`的别名，用于表示返回值           |
   | 12~20      | `t0 ~ t8` | 不需要保存和恢复的暂存器                 |
   | 21         | Reserved  | 暂时保留不用                             |
   | 22         | `fp`      | Frame Pointer，栈帧指针                  |
   | 23~31      | `s0 ~ s8` | 寄存器变量，子程序使用需要保存和恢复     |

* 函数返回值使用2个寄存器 `a0, a1`:可以用于存放 `128bit` 的 long double类型数据。

* 函数调用约定：
LoongArch参数传递规则：

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-il3a{border-color:#ffffff;color:#ffffff;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-il3a" rowspan="12">整数调用规范<br>提供了8个参数寄存器 a0 ~ a7</th>
    <th class="tg-il3a" rowspan="6">标量</th>
    <th class="tg-il3a" rowspan="2">标量宽度 &lt;= XLEN</th>
    <th class="tg-il3a">有可用的寄存器，则在单个寄存器中传递</th>
  </tr>
  <tr>
    <th class="tg-il3a">没有可用的寄存器，则在栈上传递</th>
  </tr>
  <tr>
    <th class="tg-il3a" rowspan="3">XLEN &lt; 标量宽度 &lt;= 2 * XLEN</th>
    <th class="tg-il3a">有2个可用的寄存器，则低XLEN位在小编号寄存器中，高XLEN位在大编号寄存器中</th>
  </tr>
  <tr>
    <th class="tg-il3a">只有1个可用的寄存器，则低XLEN位在寄存器中，高XLEN位在栈上传递</th>
  </tr>
  <tr>
    <th class="tg-il3a">没有可用的寄存器，则在栈上传递</th>
  </tr>
  <tr>
    <th class="tg-il3a">标量宽度 &gt; 2 * XLEN</th>
    <th class="tg-il3a">通过引用传递，在参数列表中用地址替换</th>
  </tr>
  <tr>
    <th class="tg-il3a" rowspan="6">结构体</th>
    <th class="tg-il3a" rowspan="2">结构体宽度 &lt;= XLEN</th>
    <th class="tg-il3a">有可用的寄存器，则在单个寄存器中传递</th>
  </tr>
  <tr>
    <th class="tg-il3a">没有可用的寄存器，则在栈上传递</th>
  </tr>
  <tr>
    <th class="tg-il3a" rowspan="3">XLEN &lt; 结构体宽度 &lt;= 2 * XLEN</th>
    <th class="tg-il3a">有2个可用的寄存器，则前半部分小编号寄存器中，后半部分在大编号寄存器中</th>
  </tr>
  <tr>
    <th class="tg-il3a">只有1个可用的寄存器，则前半部分在寄存器中，后半部分在栈上传递</th>
  </tr>
  <tr>
    <th class="tg-il3a">没有可用的寄存器，则在栈上传递</th>
  </tr>
  <tr>
    <th class="tg-il3a">结构体宽度 &gt; 2 * XLEN</th>
    <th class="tg-il3a">通过引用传递，并在参数列表中被替换为地址</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-il3a" rowspan="7">浮点调用规范<br>提供了8个浮点参数寄存器 fa0 ~ fa7<br>（可变参数使用整型参数寄存器传递！）</td>
    <td class="tg-il3a" rowspan="3">标量</td>
    <td class="tg-il3a" rowspan="2">标量宽度 &lt;= FLEN</td>
    <td class="tg-il3a">有可用的浮点参数寄存器，通过单个寄存器传递</td>
  </tr>
  <tr>
    <td class="tg-il3a">没有可用的浮点参数寄存器，使用整数参数寄存器传递</td>
  </tr>
  <tr>
    <td class="tg-il3a">标量宽度 &gt; FLEN</td>
    <td class="tg-il3a">使用整数参数寄存器传递</td>
  </tr>
  <tr>
    <td class="tg-il3a" rowspan="4">结构体</td>
    <td class="tg-il3a">结构体只包含一个浮点实数</td>
    <td class="tg-il3a">与一个独立的浮点实数传递方式相同</td>
  </tr>
  <tr>
    <td class="tg-il3a" rowspan="2">结构体只包含2个浮点实数</td>
    <td class="tg-il3a">若2个浮点实数宽度 &lt;= FLEN，且有2个浮点参数寄存器可用，通过寄存器传递</td>
  </tr>
  <tr>
    <td class="tg-il3a">否则通过整数参数寄存器传递</td>
  </tr>
  <tr>
    <td class="tg-il3a">结构体只包含一个整数一个浮点实数</td>
    <td class="tg-il3a">如果可用，可以通过一个整数参数寄存器+一个浮点参数寄存器传递，否则通过整数参数寄存器传递</td>
  </tr>
</tbody></table>

* 进程的虚拟地址空间布局
```markdown
       __________________________
      |                          | 高地址
      + ------------------------ +
      |            栈            |
      + ------------------------ +
      |            \/            |
      |                          |
      + ------------------------ +
      |         动态链接库        |
      + ------------------------ +
      |                          |
      |            /\            |
      + ------------------------ +
      |            堆            |
      + ------------------------ +
      |     未初始化数据(bss)     | 初始化为0      
      + ------------------------ +
      |     未初始化数据(data)    | 从程序文件装入 
      + ------------------------ +
      |       程序代码(text)      |
      + ------------------------ +
      |                          | 低地址
      + ------------------------ +
```

* 使用 `ll` + `sc` 指令实现“测试并设置”
```asm
	la.local $a0, lock
test_and _set:
	ll.w     $v0, $a0, 0
	li       $t0, 0x1
	sc.w     $t0, $a0, 0
	beqz     $t0, test_and _set
```

* 使用 `ll` + `sc` 指令实现自旋锁
```asm
	la.local $a0, lock
selfspin:
	ll.w     $t0, $a0, 0
	bnez     $t0, selfspin
	li       $t1, 0x1
	sc.w     $t1, $a0, 0
	beqz     $t1, selfspin
	<\Critical section>
	st.w     $zero, lock
```


## 第5章：计算机组成原理与结构

* 计算机系统的硬件结构：
	* CPU: 主要包含控制器和运算器；
	* GPU: 图形处理；
	* 北桥: 离 CPU 最近的芯片，主要负责控制显卡、内存与 CPU 之间的数据交换，向上连接处理器，向下连接南桥。
	* 南桥: 主要负责硬盘、键盘以及各种对带宽要求较低的 IO 接口与内存、CPU 之间交换数据。

<pre>
CPU-CPU-北桥-南桥------四片结构
CPU-北桥-南桥----------三片结构（GPU功能被集成到北桥，集成显卡）
CPU-弱北桥-南桥--------三片结构（内存控制器从北桥集成到CPU芯片中）
CPU-南桥---------------两片结构（CPU也被集成到CPU芯片中，PCIE被集成到南桥中，北桥消失）
</pre>

* CPU与IO间的同步：查询方式、中断方式
* CPU与IO间的通信：PIO方式、DMA方式


* 影响 SDRAM 读写速度的因素：**行缓冲局部性**(open page/close page) 和 **Bank 级并行度**。


## 第6章：计算机总线接口技术

* 总线的层次：机械层、电器层、协议层、架构层

* PCI 总线、 DDR 总线等都是传统的并行总线, 而 USB、
SATA、 PCIE、HT等都是串行总线。




## 第7章：计算机系统启动过程分析

* `bios` 的前几条指令
```markdown
	dil         t0, (0x7 << 16)
	csrxchg     zero, t0, 0x4    # 将除 TLB 外的所有例外和中断入口设置为同一个
	dli         t0, 0x1c001000   
	csrwr       t0, 0xc          # 将该例外入口地址 (0xc 号csr寄存器) 设置为 0x1C001000
	dli         t0, 0x1c001000
	csrwr       t0, 0x88         # 将 TLB 重填例外的入口地址 (0x88 号csr寄存器) 也设置为 0x1C001000
	dli         t0, (1 << 2)
	csrxchg     zero, t0, 0x0    # 关闭全局中断
	la          sp, stack
	la          gp, _gp
```

* 第一条指令从一个特定地址取回：LoongArch，`0x1c000000`。从闪存中得到。

* 初始化过程：
<pre>
	Step 1: 调试接口初始化-----串口初始化
	Step 2: TLB 初始化
	Step 3: Cache 初始化
	Step 4: 内存初始化
	Step 5: IO总线初始化
	Step 6: 探测设备、加载驱动
	Step 7: 加载 OS
	Step 8: 唤醒从核
</pre>

* TLB的初始化：**将全部 TLB 表项初始化为无效项**；
<pre>
	可以使用 tlbwr 指令，通过循环将整个 TLB 表项清零；
	也可以使用 invtlb 指令，由硬件完成循环清空操作。
</pre>

* cache的初始化：`cacop` 指令用于初始化cache，将指定 cache 行的 Tag 写为0。

* TLB 和 cache 的初始化都由专门的复位电路完成，不再由低效的 Ucache 程序完成。

* 内存的初始化：首先通过 I2C 总线对外部内存条的 SPD 芯片进行读取，获取内存的配置信息；根据配置信息对内存控制器初始化。

```
内存初始化与cache初始化
+--------------------------- + ----------------------------+
|        内存初始化           |         cache初始化          
+--------------------------- + ----------------------------+
|                相似：不涉及存储的初始数据                   
+--------------------------- + ----------------------------+
|   仅需要对内存控制器初始化   | Cache有专门的硬件控制位表示     
|                            | Cache 块是否有效，需要对硬     
|                            | 件控制位写入初值              
+--------------------------- + ----------------------------+
```

* IO 总线初始化
<pre>
	HT总线，3件事：
	1. 对 IO 总线的访问地址空间进行设置；
	2. 对 IO 设备的 DMA 访问空间进行规定；
	3. 对 HT 总线进行升频。
</pre>

* 总线设备的探测：**PCI协议**
<pre>
	配置空间------------存储设备基本信息，用于设备的探测和发现
	IO 空间-------------用于少量设备寄存器访问（小）
	Memory 空间---------大

	对总线号、设备号、功能号进行枚举，就可以遍历整个总线，检测到哪个地址上存在设备。

	对 PCI 设备的探测和驱动加载是一个递归调用过程, 大致算法如下:
	1) 将初始总线号、初始设备号、初始功能号设为 0。
	2) 使用当前的总线号、设备号、功能号组成一个配置空间地址, 使用该地址, 访问其 0 号寄存器, 检查其设备号。
	3) 如果读出全 1 或全 0, 表示无设备。
	4) 如果该设备为有效设备, 检查每个 BAR 所需的空间大小, 并收集相关信息。
	5) 检测其是否为一个多功能设备, 如果是则将功能号加 1 再重复扫描, 执行第 2 步。
	6) 如果该设备为桥设备, 则给该桥配置一个新的总线号, 再使用该总线号, 从设备号 0、功能号 0 开始递归调用, 执行第 2 步。
	7) 如果设备号非 31, 则设备号加 1, 继续执行第 2 步; 如果设备号为 31, 且总线号为 0, 表示扫描结束, 如果总线号非 0, 则退回上一层递归调用。

	BAR 寄存器
	|——————————————————————————————————————————————————————————|
	|   可写位   |      只读0位      |可预取标志| 64位标志 |IO标志|
	|31        n|n-1               4|3        |2        1|0    |
	|——————————————————————————————————————————————————————————|
</pre>


* 多核启动过程

## 第8章：运算器设计

* IEEE754 标准
<pre>
	float 类型（32 bit）
	|———————————————————————————————————————————|
	| S |   Exp.(阶码)   |    Fraction(尾数)     |
	| 1 |      8         |        23            |
	|———————————————————————————————————————————|

	double 类型（64 bit）
	|———————————————————————————————————————————|
	| S |   Exp.(阶码)   |    Fraction(尾数)     |
	| 1 |      11        |        52            |
	|———————————————————————————————————————————|

	规格化：将实数表示为 1.xxx * 2^(...) 的形式，省略 "1."，尾数为小数点后的部分
	尾数用原码表示，阶码用移码表示！
	移码所用的偏置常量是 2^n -1 ！！
	--> 8位阶码可以表示的范围是 -127 ~ 128, (00000000 表示的是 -127, 11111111 表示的是 128)
</pre>

* IEEE754 的特殊值：

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:#ffffff;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:#ffffff;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-8jgo{border-color:#ffffff;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-8jgo" rowspan="2"></th>
    <th class="tg-8jgo" colspan="4">单精度</th>
    <th class="tg-8jgo" colspan="4">双精度</th>
  </tr>
  <tr>
    <th class="tg-8jgo">符号</th>
    <th class="tg-8jgo">阶码</th>
    <th class="tg-8jgo">尾数</th>
    <th class="tg-8jgo">值</th>
    <th class="tg-8jgo">符号</th>
    <th class="tg-8jgo">阶码</th>
    <th class="tg-8jgo">尾数</th>
    <th class="tg-8jgo">值</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-8jgo">正无穷</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">255</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">+∞</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">2047</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">+∞</td>
  </tr>
  <tr>
    <td class="tg-8jgo">负无穷</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">255</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">-∞</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">2047</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">-∞</td>
  </tr>
  <tr>
    <td class="tg-8jgo">NaN</td>
    <td class="tg-8jgo">0或1</td>
    <td class="tg-8jgo">255</td>
    <td class="tg-8jgo">!=0</td>
    <td class="tg-8jgo">NaN</td>
    <td class="tg-8jgo">0或1</td>
    <td class="tg-8jgo">2047</td>
    <td class="tg-8jgo">!=0</td>
    <td class="tg-8jgo">NaN</td>
  </tr>
  <tr>
    <td class="tg-8jgo">规格化非0正数</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0&lt;e&lt;255</td>
    <td class="tg-8jgo">f</td>
    <td class="tg-8jgo">1.f*2^{e-127}</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0&lt;e&lt;2047</td>
    <td class="tg-8jgo">f</td>
    <td class="tg-8jgo">1.f*2^{e-1023}</td>
  </tr>
  <tr>
    <td class="tg-8jgo">规格化非0负数</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">0&lt;e&lt;255</td>
    <td class="tg-8jgo">f</td>
    <td class="tg-8jgo">-1.f*2^{e-127}</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">0&lt;e&lt;2047</td>
    <td class="tg-8jgo">f</td>
    <td class="tg-8jgo">-1.f*2^{e-1023}</td>
  </tr>
  <tr>
    <td class="tg-8jgo">非规格化非0正数</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">f!=0</td>
    <td class="tg-8jgo">0.f*2^(-126)</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">f!=0</td>
    <td class="tg-8jgo">0.f*2^(-1022)</td>
  </tr>
  <tr>
    <td class="tg-8jgo">非规格化非0负数</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">f!=0</td>
    <td class="tg-8jgo">-0.f*2^(-126)</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">f!=0</td>
    <td class="tg-8jgo">-0.f*2^(-1022)</td>
  </tr>
  <tr>
    <td class="tg-8jgo">正0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
  </tr>
  <tr>
    <td class="tg-8jgo">负0</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">-0</td>
    <td class="tg-8jgo">1</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">0</td>
    <td class="tg-8jgo">-0</td>
  </tr>
</tbody>
</table>

* 门延迟：非门、与非门、或非门是一级门延迟。
* 1位全加器  `A + B + Cin = {Cout, S}`
	* 产生 S 3级门延迟；
	* 产生 Cout 2级门延迟。

* 行波进位加法器
<pre>
	32bit 加法器的延迟怎么计算？
	从A0B0和Cin到C31：2×31=62级门
	从C31到S31：3级门
	从C31到Cout：2级门
	最大延迟：65级门
</pre>

* 先行进位加法器
<pre>
	32bit 加法器
	采用4位一块，块内并行，块间串行：21级门延迟
	采用4位一块，块内并行，块间并行：15级门延迟
</pre>

* Booth 乘法

## 第9章：指令流水线

* 指令间的相关：数据相关（`RAW, WAW, WAR`），控制相关（转移指令），结构相关

* 解决数据相关：阻塞/前递技术
* 解决控制相关：阻塞；改进：在译码级就进行转移条件的判断和转移地址的计算；分支延迟槽


* 流水线处理器如何实现精确异常？

> 1) 任何一级流水发生异常时, 在流水线中记录下发生异常的事件, 直到写回阶段再处理。
> 2) 如果在执行阶段要修改机器状态 (如状态寄存器), 保存下来直到写回阶段再修改。
> 3) 指令的 PC 值随指令流水前进到写回阶段为异常处理专用。
> 4) 将外部中断作为取指的异常处理。
> 5) 指定一个通用寄存器 (或一个专用寄存器) 为异常处理时保存 PC 值专用。
> 6) 当发生异常的指令处在写回阶段时, 保存该指令的 PC 及必需的其他状态, 置取指的 PC 值为异常处理程序入口地址。

* 动态流水线
	* ROB：维护指令的有序提交（乱序变成有序以保证正确）；
	* 保留站：保存处于等待操作数阶段的指令，不让它们阻塞住后面的指令（有序变成乱序以提高性能）；
	* 寄存器重命名技术：用于处理 `WRW, WAR` 冲突。

* 转移预测

<pre>
	BTB 的预测过程: 用取指 PC 与表中各项的 PC 进行比较, 
	如果某项置相等且该项的饱和计数器值指示预测跳转, 
	则取出该项所存的跳转目标并跳转过去。

	BTB表项（在 取指级 就分支预测）
	|-----------------------------------------------|
	|    PC    |    Target     | Counter(饱和计数器) |
	|-----------------------------------------------|


	BHT/PHT：转移历史表
	利用 PC 的低位进行索引，每一项记录上一次执行时跳转的情况。
	1位PHT  2位PHT
</pre>

<code>
	AMAT = HitTime + MissRate * MissPenalty
</code>

## 第10章：并行编程基础

* 多处理器系统可分为**共享存储系统**和**消息传递系统**两类。
* 共享存储变成模型：`Pthreads`和`OpenMP`;
* 消息传递变成模型：`MPI`
* 2个例子：pai的计算和矩阵乘法

* pthread实现：
```c
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#define NUM_THREADS 4 // 假设线程数目为 4
int num_steps = 1000000;
double step = 0.0, sum = 0.0;
pthread_mutex_t mutex;
void *countPI(void *id) {
	int index = (int ) id;
	int start = index*(num_steps/NUM_THREADS);
	int end;
	double x = 0.0, y = 0.0;
	if (index == NUM_THREADS-1)
		end = num_steps;
	else
		end = start+(num_steps/NUM_THREADS);
	for (int i=start; i<end; i++)
	{
		x=(i+0.5)*step;
		y +=4.0/(1.0+x*x);
	}
	pthread_mutex_lock(&mutex);
	sum += y;
	pthread_mutex_unlock(&mutex);
}
int main() {
	int i;
	double pi;
	step = 1.0 / num_steps;
	sum = 0.0;
	pthread_t tids[NUM_THREADS];
	pthread_mutex_init(&mutex, NULL);
	for(i=0; i<NUM_THREADS; i++) {
		pthread_create(&tids[i], NULL, countPI, (void *) i);
	}
	for(i=0; i<NUM_THREADS; i++)
		pthread_join(tids[i], NULL);
	pthread_mutex_destroy(&mutex);
	pi = step*sum;
	printf("pi %1f\n", pi);
	return 0;
}
```

```c
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#define NUM_THREADS 4 // 假设线程数目为 4
#define n 1000
double *A,*B,*C;
void *matrixMult(void *id) { // 计算矩阵乘
	int my_id = (int ) id;
	int i,j,k,start,end;
	// 计算进程负责的部分
	start = my_id*(n/NUM_THREADS);
	if(my_id == NUMTHREADS-1)
		end = n;
	else
		end = start+(n/NUM_THREADS);
	for(i=start;i<end;i++)
		for(j=0;j<n;j++) {
			C[i*n+j] = 0;
			for(k=0;k<n;k++)
				C[i*n+j]+=A[i*n+k]*B[k*n+j];
		}
}
int main() {
	int i,j;
	pthread_t tids[NUM_THREADS];
	// 分配数据空间
	A = (double *)malloc(sizeof(double)*n*n);
	B = (double *)malloc(sizeof(double)*n*n);
	C = (double *)malloc(sizeof(double)*n*n);
	// 初始化数组
	for(i=0;i<n;i++)
		for(j=0;j<n;j++){
			A[i*n+j] = 1.0;
			B[i*n+j] = 1.0;
		}
	for(i=0; i<NUM_THREADS; i++)
		pthread_create(&tids[i], NULL, matrixMult, (void *) i);
	for(i=0; i<NUM_THREADS; i++)
		pthread_join(tids[i], NULL);
	return 0;
}
```

* OpenMP 实现
* OpenMP 程序开始于一个单独的主线程 (Master Thread), 主线程串行执行,遇到一个并行域 (Parallel Region) 开始并行执行。
* 编译制导语句


```c
#include <stdio.h>
#include <omp.h>
int main(){
	int i;
	int num_steps=1000000;
	double x,pi,step,sum=0.0;
	step = 1.0/(double) num_steps;
	#pragma omp parallel for private(i, x), reduction(+:sum)
	for(i=0;i<num_steps;i++)
	{ 
		x=(i+0.5)*step;
		sum = sum+4.0/(1.0+x*x);
	}
	pi = step*sum;
	printf(“pi %1f\n”, pi);
	return 0;
}
```

```c
#include <stdio. h>
#include <omp. h>
#define n 1000
double A[n][n],B[n][n],C[n][n];
int main()
{
	int i,j,k;
	//初始化矩阵 A 和矩阵 B
	for(i=0;i<n;i++)
		for(j=0;j<n;j++) {
			A[i][j] = 1.0;
			B[i][j] = 1.0;
		}
	//并行计算矩阵 C
	#pragma omp parallel for shared(A,B,C) private(i,j,k)
	for(i=0;i<n;i++)
		for(j=0;j<n;j++){
			C[i][j] = 0;
			for(k=0;k<n;k++)
				C[i][j]+=A[i][k]* B[k][j];
		}
	return 0;
}
```

* MPI 实现

* OpenMP 与 MPI 并行编程的对比：
	* OpenMP 是线程的并行，MPI 是进程的并行。OpenMP 只需要在串行程序的基础上添加编译制导指令即可，程序的编写相对 MPI 来说更简单；
	* OpenMP 线程之间的通信通过共享存储来实现；MPL 进程之间的通信通过消息的收发传递来实现。
	* OpenMP 可以不指定线程个数，而由编译器选择合适的并行线程个数，从而对计算任务自动划分；而 MPI 对于计算任务的划分需要程序员完成。

```c
#include <stdio.h>
#include “mpi.h”
#define n 1000
int main(int argc, char **argv)
{
	double *A,*B,*C;
	int i,j,k;
	int ID,num_procs,line;
	MPI_Status status;
	MPI_Init(&argc,&argv); // 初始化 MPI 环境
	MPI_Comm_rank(MPI_COMM_WORLD,&ID); // 获取当前进程号
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); // 获取进程数目
	//分配数据空间
	A = (double *)malloc(sizeof(double)*n*n);
	B = (double *)malloc(sizeof(double)*n*n);
	C = (double *)malloc(sizeof(double)*n*n);
	line = n/num_procs;//按进程数来划分数据
	if(ID==0){ // 节点 0,主进程
		//初始化数组
		for(i=0;i<n;i++)
			for(j=0;j<n;j++){
				A[i*n+j] = 1.0;
				B[i*n+j] = 1.0;
			}
		//将矩阵 A、B 的相应数据发送给从进程
		for(i=1;i<num_procs;i++) {
			MPI_Send(B, n*n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(A+(i-1)*line*n, line*n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
		}
		//接收从进程的计算结果
		for(i=1;i<num_procs;i++)
			MPI_Recv(C+(i-1)*line*n, line*n, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
		//计算剩下的数据
		for(i=(num_procs-1)*line;i<n;i++)
			for(j=0;j<n;j++) {
				C[i*n+j]=0;
				for(k=0;k<n;k++)
					C[i*n+j]+=A[i*n+k]*B[k*n+j];
			}
	}else {
		//其他进程接收数据,计算结果,发送给主进程
		MPI_Recv(B,n*n,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);
		MPI_Recv(A+(ID-1)*line*n,line*n,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&status);
		for(i=(ID-1)*line;i<ID*line;i++)
			for(j=0;j<n;j++) {
				C[i*n+j]=0;
				for(k=0;k<n;k++)
					C[i*n+j]+=A[i*n+k]*B[k*n+j];
			}
		MPI_Send(C+(num_procs-1)*line*n,line*n,MPI_DOUBLE,0,2,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
```


```c
#include “mpi.h”
int main(int argc,char *argv[])
{ 
	int myid,count;
	MPI_Init(&agrc,&argv); /* 启动计算 */
	MPI_Comm_size(MPI_COMM_WORLD,&count); /* 获得进程总数 */
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); /* 获得自己进程号 */
	printf("I am %d of %d\n", myid,count); /* 打印消息 */
	MPI_Finalize(); /* 结束计算 */
}
```

## 第11章：多核处理器结构

<pre>
	SIMD结构
	SMP结构：多核共享存储
	CC-NUMA结构：更多核共享存储
	MPP或机群结构
</pre>

* 多核处理器的访存结构
<pre>
	片上Cache结构：私有、片上共享、片间共享
	UCA结构：集中式共享最后一级cache，所有处理器对二级 cache的访问延迟相同
	NUCA结构：分布式共享

	存储一致性模型：顺序/处理器/弱/释放一致性
	Cache一致性协议：目录/侦听，write-invalidate/write-update
</pre>


* 多核处理器的互连结构
<pre>
	总线、交叉开关、片上网络
	片上网络的路由算法：维序路由，自适应路由
</pre>

## 第12章 性能分析

* Perf 是内置于 Linux 内核源码树中的性能分析工具
* 常见模拟器：SimOS, Simplescalar, GEM5...

## 重点考点

1. PPT翻页的过程：
* 从按下键盘开始

<pre>
    键盘 -------> 南桥芯片（将键盘编码保存在寄存器中）
					 ||
					 ||外部中断信号
					 \/              _________________________
				    CPU ----------->|  ESTAT寄存器的某一位置1   |
                                    |  ECFG寄存器的局部中断使能 |
									|  CRMD寄存器的全局中断使能 |
									+———————————||————————————+
				                                \/
	当这条指令成为       <-----------中断信号附在一条处于译码级的指令上
	ROB 的第一条指令时，             送到重排序缓冲(Re-Order Buffer)
		||
		\/
	取消该指令后面的所有指令 --> 异常处理准备 --> 确定异常来源 --> 保存异常现场
	                                                               |
							   异常处理   <—————————————————————————+
							      ||
					|—————————————\/———————————————————|
					| 读取 ESTAT 寄存器，获取例外原因，  |
			————————| 发现是外部中断例外。               |
		   ||		| 向南桥的中断控制器读终端原因，同时清|
		   ||		| 除南桥的中断位。发现空格键被按下。  |
		   ||		|——————————————————————————————————|
		   \/
	操作系统查找读到的空格是给谁的（有没有进程在阻塞等待键盘输入）
	发现 WPS 在阻塞态，将它唤醒

	WPS 被唤醒后，得到要翻页的请求，在内存中准备好PPT下一页的数据
	调用操作系统中的显示驱动程序（陷入内核态），将要显示的内容送到显存
	GPU 访问显存空间刷新屏幕

	|————————————|--------------|
	|  等待按键   |   用户态     |
	|————————————|--------------|
	|  中断处理   |   核心态     |
	|————————————|--------------|
	|  准备数据   |   用户态     |
	|————————————|--------------|
	|  操作显存   |   核心态     |
	|————————————|--------------|
	|  返回ppt   |    用户态     |
	|————————————|--------------|

	CPU, GPU, DC, 显存, 内存之间的同步和通信过程：
	1. CPU准备需要显示的数据和GPU需要执行的指令，放在内存中，并写GPU寄存器通知GPU开始工作；
	2. GPU通过DMA方式从内存读取指令和数据，并将结果绘制在显存中，完成后通过中断通知CPU任务完成；
	3. DC周期性持续地通过DMA访问显存，并输出到显示接口。
</pre>


2. 指出一段程序共产生多少次 TLB refill 和 TLB invalid 异常（可能包含TLB替换和进程切换）

&emsp;&emsp;在一台 Linux/LoongArch 机器上执行如下程序片段，假设数组a和b的起始地址都是8KB边界对齐的，操作系统仅支持4KB页大小。处理器中的TLB有32项，采用LRU替换算法。...请问执行该程序片段的过程中会发生多少次与TLB地址翻译相关的异常。

```c
void cycle(double *a){
    int i,j;
    double b[65536];
    for(i=0;i<3;i++)
        for(j=0;j<65536;j++)
             a[j] = b[j];
}
```

> 首次访问 a 时会产生重填异常和store无效异常；首次访问b时会产生重填异常和load无效异常。<br>
> $\because$**TLB每项中保存两个连续的虚拟页**<br>
> $\therefore$**每两页产生1个重填异常和2个无效异常**<br>
> 计算：a、b所占页数均为 $(65536 * 8 / 4K)= 128$<br>
> i=0时，总异常数为 $2*(128/2+128/2*2)=384$ 次<br>
> i=1/2时，由于TLB项数不足不命中，故产生重填异常 $2*2*(128/2)=256$ 次<br>
> 总计640次。

3. C语言的不同变量（全局、局部等）映射到地址空间的哪些段？
	* **程序的代码**保存在代码段(text)
	* **初始化的全局和静态变量**保存在数据段(data)
	* **未初始化的全局和静态变量**保存在bss段(初始化为0)
	* **过程调用中的局部变量**保存在栈中
	* **动态分配的内存**在堆中

4. 函数调用、系统调用、中断处理、进程切换都需要上下文切换，请结合LoongArch的ABI说明上述上下文切换时保留现场有什么不同（内容、位置）？

* 6种上下文切换场景的对比

 | 场景               | 上下文切换时保存和恢复的内容                           |
 |:-------------------|:-------------------------------------------------------|
 | 函数调用           | 部分寄存器(包括栈帧相关的 `sp` 和 `fp`，返回地址 `ra`) |
 | 中断和异常         | 全部定点寄存器，异常现场信息，异常相关信息             |
 | 系统调用           | 部分定点寄存器(包括栈帧相关寄存器)，异常现场信息       |
 | 进程切换           | 全部用户态寄存器，页表基址，当前PC                     |
 | 线程切换           | 全部用户态寄存器，TLS，当前PC                          |
 | 虚拟机与宿主机切换 | 虚拟CPU状态(寄存器，必要的特权资源等)                  |

5. 系统中有关IO设备是如何探测到的

&emsp;&emsp;计算机启动时，会对总线号、设备号、功能号进行枚举，就可以遍历整个总线，检测到哪个地址上存在设备。


6. 简单描述从硬盘读一个文件的DMA过程。

<pre>
	Step 1: CPU 设置某内存区域为磁盘 DMA 区域；
	Step 2: CPU 读写磁盘 DMA 控制寄存器，通知磁盘启动 DMA 传输；
	Step 3: 磁盘将目标文件传输到内存；
	Step 4: DMA 控制器向 CPU 发出中断，通知 CPU 数据传输完成；
	Step 5: CPU 收到中断后，OS 唤醒等待的进程。
</pre>

7. 给出一个晶体管电路，给出真值表和等价的逻辑表达式。


8. 结合转移猜测算法，计算一段小程序的命中率。

```c
for(i=0; i<10; i++)
  for(j=0; j<10; j++)
    for(k=0; k<10; k++)
      {...}
```

&emsp;&emsp;计算分别使用一位BHT表和使用两位BHT表进行转移猜测时三重循环分别的转移猜测准确率，假设BHT表的初始值均为0。

* 1位BHT，总在进入和退出时猜错，for i, for j, for k，循环各判断10、100、1000次。

* 2位BHT，首次进入时猜错两次，每次退出时猜错一次，但再次进入时不会猜错。


9. EDA工具：电子设计自动化
10. 摩尔定律：集成电路芯片上所集成的晶体管数量，每18个月就翻一倍

11. 嵌入式基准测试程序如 EEMBC 和桌面基准测试程序在行为特性上有什么差别?

&emsp;&emsp;不同方面的差别：


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-il3a{border-color:#ffffff;color:#ffffff;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-il3a">方面</th>
    <th class="tg-il3a">嵌入式基准测试程序</th>
    <th class="tg-il3a">桌面基准测试程序</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-il3a">硬件平台</td>
    <td class="tg-il3a">嵌入式系统</td>
    <td class="tg-il3a">桌面或服务器系统</td>
  </tr>
  <tr>
    <td class="tg-il3a">系统资源</td>
    <td class="tg-il3a">嵌入式系统的资源有限，因此嵌入式基准测试程序需要更高效的代码和更小的内存占用。</td>
    <td class="tg-il3a">桌面基准测试程序可以使用更多的系统资源，因此可以测试更复杂的应用程序和算法。</td>
  </tr>
  <tr>
    <td class="tg-il3a">测试场景</td>
    <td class="tg-il3a">嵌入式基准测试程序通常测试嵌入式系统的<span style="font-weight:bold">实时性能</span>，如启动速度、响应时间和功耗等</td>
    <td class="tg-il3a">桌面基准测试程序通常测试<span style="font-weight:bold">计算性能</span>，如CPU、GPU和内存的性能</td>
  </tr>
</tbody>
</table>
