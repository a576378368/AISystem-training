# 分布式训练

## 本章学习目标

1. 理解分布式训练的基本概念和必要性
2. 掌握数据并行（DP与DDP）的原理与实现
3. 理解FSDP和ZeRO优化的原理
4. 掌握模型并行（张量并行与流水线并行）的原理
5. 了解混合并行（DP+PP+TP）的概念
6. 理解分布式训练中的通信优化

## 预备知识

- 深度学习训练流程
- PyTorch编程基础
- 计算机网络基础（通信原语）

---

## 1 分布式训练概述

### 1.1 为什么需要分布式训练

深度学习模型正在变得越来越大。让我用一个具体的例子来说明这个问题：

| 模型 | 参数量 | 模型大小(FP32) | 训练硬件需求 |
|------|--------|----------------|--------------|
| ResNet-50 | 2500万 | ~100MB | 单卡可训练 |
| BERT-Base | 1.1亿 | ~440MB | 多卡加速 |
| GPT-3 | 1750亿 | ~700GB | 需要分布式 |
| GPT-4 | 约1万亿 | ~4TB | 超大规模集群 |
| LLaMA-65B | 650亿 | ~260GB | 多卡多机 |

当模型参数量达到数十亿甚至万亿级别时：
1. **单个设备内存放不下**：一张NVIDIA A100 GPU只有80GB HBM，而1750亿的模型需要约700GB
2. **单设备计算太慢**：训练GPT-3使用单卡需要几十年

分布式训练通过多设备协作，解决这两个问题。

### 1.2 分布式训练的基本思想

分布式训练的核心思想是**并行化**：

1. **数据并行**：让多个设备处理不同的数据批次，并行加速
2. **模型并行**：把模型拆分到多个设备，解决内存问题

```
数据并行示意：
设备0: Batch0 ──► [Model] ──► Grad0 ──┐
设备1: Batch1 ──► [Model] ──► Grad1 ──┼──► 平均梯度 ──► 更新模型
设备2: Batch2 ──► [Model] ──► Grad2 ──┤
设备3: Batch3 ──► [Model] ──► Grad3 ──┘

模型并行示意：
        参数分片
      ┌────────────┐
Layer1 │ Layer2   │ Layer3
      ▼           ▼
  设备0        设备1
```

### 1.3 分布式训练的系统架构

典型的分布式训练系统包含：

```
┌─────────────────────────────────────────────────┐
│                   训练集群                        │
│  ┌─────────────┐     ┌─────────────┐            │
│  │   Node 0   │◄───►│   Node 1   │            │
│  │ GPU0 GPU1   │     │ GPU0 GPU1   │            │
│  │   │    │    │     │   │    │    │            │
│  └─────────────┘     └─────────────┘            │
│         ▲                   ▲                    │
│         │      Network     │                    │
│         └─────────────────┘                    │
└─────────────────────────────────────────────────┘
```

关键组件：
- **节点（Node）**：一台物理服务器，包含多块GPU
- **GPU/NPU**：实际的计算设备
- **网络**：节点间通信（InfiniBand/NVLink）
- **集合通信**：AllReduce、Broadcast等

### 1.4 分布式训练的挑战

分布式训练面临三大挑战：

#### 1.4.1 通信带宽

不同设备间的通信带宽差异巨大：

| 通信路径 | 带宽 |
|----------|------|
| NVLink (同节点GPU间) | ~900 GB/s |
| PCIe (GPU间) | ~32 GB/s |
| InfiniBand HDR | ~50 GB/s |
| 以太网 | ~12.5 GB/s |

通信往往成为瓶颈，需要精心设计以减少通信量。

#### 1.4.2 负载均衡

需要确保：
- 各设备计算量相近，避免空闲等待
- 各设备通信量均衡
- 考虑内存限制

#### 1.4.3 收敛性

并行化不能影响模型训练的收敛性：
- 梯度同步的一致性
- 异步训练的梯度陈旧问题

---

## 2 数据并行

数据并行是最简单也最常用的分布式训练方法。

### 2.1 数据并行的基本原理

数据并行的核心思想：
1. 复制模型到所有设备
2. 各设备用不同数据批次前向传播
3. 汇总梯度，更新模型参数
4. 同步更新后的参数

```
        数据分片
    ┌─────────────┐
    │ Batch[0:4]  │──► 设备0: Forward → Backward → Grad0
    │ Batch[4:8]  │──► 设备1: Forward → Backward → Grad1
    │ Batch[8:12] │──► 设备2: Forward → Backward → Grad2
    │ Batch[12:16]│──► 设备3: Forward → Backward → Grad3
    └─────────────┘
            │
            ▼
    AllReduce: Grad_avg = (Grad0 + Grad1 + Grad2 + Grad3) / 4
            │
            ▼
    参数更新: W_new = W_old - lr * Grad_avg
            │
            ▼
    Broadcast: 所有设备获得相同的新参数
```

### 2.2 DP vs DDP

PyTorch提供两种数据并行方式：**DataParallel (DP)** 和 **DistributedDataParallel (DDP)**。

#### 2.2.1 DataParallel (DP)

DP是单进程多线程的实现：

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
# 使用DataParallel包装模型
model = nn.DataParallel(model)

# 输入会自动分到多个GPU
inputs = torch.randn(16, 10)
output = model(inputs)  # 16个样本分到4个GPU
```

**DP的特点**：
- 单进程，使用多线程
- 受Python GIL限制
- 通信成为瓶颈
- 第一个GPU负载更重（汇总梯度）

#### 2.2.2 DistributedDataParallel (DDP)

DDP是多进程的实现：

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")

    # 每个进程创建自己的模型
    model = nn.Linear(10, 5).cuda()
    # 用DDP包装
    model = DDP(model)

    # 训练循环
    for data in dataloader:
        inputs = data.cuda()
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()  # DDP自动处理梯度同步

    dist.destroy_process_group()

if __name__ == "__main__":
    # 启动多个进程
    mp.spawn(main, args=(), nprocs=4)
```

**DDP的优势**：
- 多进程，无GIL限制
- 梯度通过AllReduce同步
- 通信与计算重叠
- 扩展性好

### 2.3 DDP核心原理

#### 2.3.1 Ring AllReduce

DDP使用Ring AllReduce算法进行梯度同步：

```
假设4个GPU形成环形：
GPU0 → GPU1 → GPU2 → GPU3 → GPU0

Reduce Scatter阶段（3步）：
Step 1: GPU0处理chunk0, GPU1处理chunk1, ...
Step 2: 每个GPU接收并累加
Step 3: 最终每GPU有聚合后的部分梯度

AllGather阶段（3步）：
类似地广播最终结果
```

Ring AllReduce的优点：
- 通信量与设备数量无关
- 通信带宽利用率高
- 去中心化，无通信瓶颈

#### 2.3.2 梯度同步时机

DDP的梯度同步时机与backward执行重叠：

```python
# DDP的backward执行流程：
# 1. 本地的backward计算梯度
# 2. 自动触发AllReduce同步梯度
# 3. 通信与计算重叠

output = model(x)
loss = loss_fn(output, target)

# 这里backward会触发梯度同步
loss.backward()  # DDP内部处理AllReduce
```

### 2.4 DDP实现细节

#### 2.4.1 进程组初始化

```python
import os
import torch.distributed as dist

# 环境变量方式
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)

dist.init_process_group(backend="nccl")

# 或者显式初始化
dist.init_process_group(
    backend="nccl",  # 使用NCCL后端（NVIDIA）
    init_method="env://",  # 从环境变量读取
    rank=rank,
    world_size=world_size
)
```

#### 2.4.2 DistributedSampler

Dataloader使用DistributedSampler确保数据不重复：

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4
)
```

#### 2.4.3 保存和加载检查点

```python
# 只在rank 0保存
if dist.get_rank() == 0:
    torch.save(model.state_dict(), checkpoint_path)

# 所有进程同步后加载
dist.barrier()  # 等待所有进程
model.load_state_dict(torch.load(checkpoint_path))
```

### 2.5 数据并行的通信分析

#### 2.5.1 通信量

对于参数量为 $\Psi$ 的模型，一个AllReduce的通信量：
- 数据并行度为 $N$ 时，每个设备通信量约为 $2\Psi$（使用Ring AllReduce）
- 通信量与模型参数量成正比

#### 2.5.2 通信与计算重叠

DDP通过钩子函数实现通信与计算重叠：

```python
# 伪代码示意
def backward_hook(self, grad):
    # 梯度就绪时触发
    # 异步启动AllReduce
    dist.all_reduce(grad)
    return grad
```

### 2.6 数据并行的挑战

#### 2.6.1 内存挑战

每个设备需要存储：
- 模型参数：$\Psi$
- 梯度：$\Psi$
- 优化器状态：$2\Psi$ (Adam)
- 激活值：取决于模型深度和批次大小

使用混合精度训练（FP16）可减少约一半内存。

#### 2.6.2 通信挑战

当GPU数量增加时：
- 通信时间占比增加
- 可能成为瓶颈

解决方案：
- 增加计算量（增大批次）
- 减少通信量（梯度压缩）
- 优化通信算法

---

## 3 FSDP与ZeRO优化

当模型变得非常大时，即使使用数据并行，单个设备的内存也可能不够。**ZeRO（Zero Redundancy Optimizer）** 和 **FSDP（Fully Sharded Data Parallel）** 提供了一种解决方案。

### 3.1 ZeRO的核心思想

ZeRO的核心思想是**分片（Sharding）**：不复制完整的模型状态到每个设备，而是将状态分区存储。

| 优化阶段 | 分片内容 | 内存节省 |
|----------|----------|----------|
| ZeRO-1 | 优化器状态 | 4倍 |
| ZeRO-2 | 优化器状态 + 梯度 | 8倍 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | N倍 |

### 3.2 ZeRO-1: 优化器状态分片

ZeRO-1将优化器状态按设备数量分片：

```
原始：每个设备存储完整的优化器状态（8字节/参数 for Adam）
ZeRO-1：每个设备只存储1/N的优化器状态

Before: [GPU0: O_ful, GPU1: O_ful, GPU2: O_ful, GPU3: O_ful]
After:  [GPU0: O_1/4, GPU1: O_2/4, GPU2: O_3/4, GPU3: O_4/4]
```

通信量不变，但内存节省4倍。

### 3.3 ZeRO-2: 梯度分片

ZeRO-2在ZeRO-1基础上增加梯度分片：

```
Before: [GPU0: G_ful, GPU1: G_ful, GPU2: G_ful, GPU3: G_ful]
After:  [GPU0: G_1/4, GPU1: G_2/4, GPU2: G_3/4, GPU3: G_4/4]
```

使用Reduce-Scatter，每个设备只负责部分梯度的聚合和更新。

内存节省8倍。

### 3.4 ZeRO-3: 参数分片

ZeRO-3将模型参数也分片：

```
每个设备只存储1/N的参数
需要参数时通过All-Gather获取
```

这是最激进的优化，允许训练超出单设备内存的模型。

### 3.5 FSDP实现

FSDP是PyTorch对ZeRO的实现：

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# ZeRO-2 策略
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # 分片梯度+优化器状态
    device_id=torch.cuda.current_device(),
    # 混合精度支持
    mixed_precision={
        "param_dtype": torch.float16,
        "reduce_dtype": torch.float16,
        "buffer_dtype": torch.float16,
    }
)

# ZeRO-3 策略
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 分片参数+梯度+优化器状态
)
```

### 3.6 ZeRO通信分析

| 策略 | 通信量（相对于DP） | 内存节省 |
|------|-------------------|----------|
| DP | 1x | 1x |
| ZeRO-1 | ~1x | 4x |
| ZeRO-2 | ~1x | 8x |
| ZeRO-3 | ~1.5x | N倍 |

ZeRO-3通信量增加约50%，但换来了线性增长的内存节省。

### 3.7 ZeRO-Offload

ZeRO-Offload将部分数据卸载到CPU或NVMe：

```python
from deepspeed.runtime.zero import Stage3

# 卸载优化器状态到CPU
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config={
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",  # 卸载到CPU
            }
        }
    }
)
```

这允许用有限的GPU内存训练超大模型。

---

## 4 模型并行

当模型太大无法放入单个设备时，需要使用**模型并行**。模型并行分为两种：
- **张量并行（Tensor Parallelism）**：将单层的参数分到多个设备
- **流水线并行（Pipeline Parallelism）**：将不同层的计算分到不同设备

### 4.1 张量并行基础

张量并行将单层（如矩阵乘法）的参数矩阵按维度切分：

#### 4.1.1 列切分（Column Parallel）

将权重矩阵按列切分：

```python
# Y = X @ W
# W shape: [input_dim, output_dim]
# 如果切分W为 [input_dim, output_dim/N] x N

W1, W2 = W.split(output_dim // N, dim=1)
Y1 = X @ W1  # 设备0
Y2 = X @ W2  # 设备1
Y = torch.cat([Y1, Y2], dim=1)  # 拼接结果
```

#### 4.1.2 行切分（Row Parallel）

将权重矩阵按行切分：

```python
# Y = X @ W
# W shape: [input_dim, output_dim]
# 如果切分W为 [input_dim/N, output_dim] x N

X1, X2 = X.split(input_dim // N, dim=0)
Y1 = X1 @ W1  # 设备0
Y2 = X2 @ W2  # 设备1
Y = Y1 + Y2  # 求和合并
```

### 4.2 Transformer中的张量并行

现代大模型主要使用Transformer架构，其张量并行实现如下：

#### 4.2.1 MLP层的并行

```python
class ParallelMLP(nn.Module):
    def __init__(self, dim, hidden_dim, n_devices):
        super().__init__()
        # 切分第一个线性层（列切）
        self.fc1 = ColumnParallelLinear(dim, hidden_dim, n_devices)
        # 切分第二个线性层（行切）
        self.fc2 = RowParallelLinear(hidden_dim, dim, n_devices)

    def forward(self, x):
        x = self.fc1(x)  # [B, H/N]
        x = F.gelu(x)
        x = self.fc2(x)  # [B, H]
        return x
```

#### 4.2.2 自注意力层的并行

```python
class ParallelSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, n_devices):
        super().__init__()
        self.n_heads = n_heads
        self.n_devices = n_devices

        # QKV投影，列切
        self.qkv = ColumnParallelLinear(dim, dim * 3, n_devices)

        # 输出投影，行切
        self.proj = RowParallelLinear(dim, dim, n_devices)

    def forward(self, x):
        B, N, C = x.shape

        # QKV分到不同设备
        qkv = self.qkv(x)  # [B, N, 3C/N]
        q, k, v = qkv.split(C // self.n_devices, dim=-1)

        # 计算注意力（需要AllReduce）
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = attn @ v

        # 输出
        out = self.proj(attn)
        return out
```

### 4.3 张量并行的通信

张量并行需要额外的集合通信：

| 操作 | 通信类型 | 通信量 |
|------|----------|--------|
| AllGather (输入) | 集合通信 | $O(seq \times hidden)$ |
| AllReduce (输出) | 集合通信 | $O(seq \times hidden)$ |

通信量与张量并行的设备数成正比。

### 4.4 DeviceMesh实现

PyTorch的DeviceMesh提供了高层API：

```python
from torch.distributed.device_mesh import init_device_mesh

# 创建一个2D设备网格
# 8 GPU: 4路张量并行 x 2路数据并行
mesh = init_device_mesh("cuda", (4, 2))

# 获取子网格
tp_mesh = mesh["tp"]  # 张量并行网格
dp_mesh = mesh["dp"]  # 数据并行网格
```

### 4.5 流水线并行基础

流水线并行将模型按层切分：

```python
# 8层模型分配到4个设备
# 设备0: Layer0-1
# 设备1: Layer2-3
# 设备2: Layer4-5
# 设备3: Layer6-7

class PipelineParallel(nn.Module):
    def __init__(self, n_stages, n_devices):
        super().__init__()
        self.devices = [torch.device(f"cuda:{i}") for i in range(n_devices)]

        # 每个设备有自己的模型分片
        self.stage0 = Stage0().to(self.devices[0])
        self.stage1 = Stage1().to(self.devices[1])
        # ...

    def forward(self, x):
        x = x.to(self.devices[0])
        x = self.stage0(x)
        x = x.to(self.devices[1])
        x = self.stage1(x)
        # ...
```

### 4.6 朴素流水线的问题

朴素流水线会有**气泡（Bubble）**问题：

```
朴素流水线示意（4阶段）：
时间  T1   T2   T3   T4   T5   T6   T7   T8
设备0 [F0  ][F1  ][F2  ][F3  ][idle][idle][idle][idle]
设备1 [idle][F0  ][F1  ][F2  ][F3  ][idle][idle][idle]
设备2 [idle][idle][F0  ][F1  ][F2  ][F3  ][idle][idle]
设备3 [idle][idle][idle][F0  ][F1  ][F2  ][F3  ][idle]

气泡比例 = (N-1) / N（N为阶段数）
```

**气泡造成GPU利用率低**。

### 4.7 微批次（Micro-batch）

为减少气泡，将批次划分为更小的微批次：

```python
# 原始：batch_size=16
# 微批次：num_microbatches=4, micro_batch_size=4

# 流水线调度：
# T1: F0(m0) F1(m0) F2(m0) F3(m0)
# T2: F0(m1) F1(m1) F2(m1) F3(m1)
# T3: F0(m2) F1(m2) F2(m2) F3(m2)
# T4: F0(m3) F1(m3) F2(m3) F3(m3)
```

### 4.8 Gpipe流水线

Gpipe使用同步流水线：

```python
from torch.distributed.pipeline.sync import Pipe

# 将模型切分为多个stage
model = nn.Sequential(
    stage0, stage1, stage2, stage3
)

# 使用Pipe进行流水线并行
pipe_model = Pipe(model, chunks=4)  # 4个微批次

# 执行
output = pipe_model(input)
```

### 4.9 PipeDream流水线

PipeDream使用异步流水线（1F1B策略）：

```python
# One Forward One Backward
# 设备完成一个micro-batch的前向后立即反向
# 而不是等待所有设备完成后再反向
```

PipeDream的优点：
- 气泡更少
- 内存效率更高（无需保存所有激活）

### 4.10 流水线并行的挑战

1. **负载均衡**：每阶段的计算量需要相近
2. **通信开销**：阶段间需要传输中间结果
3. **调度复杂**：需要精心设计调度策略
4. **内存压力**：需要保存或重计算激活

---

## 5 混合并行

混合并行结合多种并行策略，以最佳利用大规模集群。

### 5.1 为什么需要混合并行

单一并行策略的局限性：

| 并行策略 | 优势 | 局限 |
|----------|------|------|
| 数据并行 | 扩展性好 | 内存占用大 |
| 张量并行 | 内存节省 | 通信开销大 |
| 流水线并行 | 内存节省 | 气泡问题 |

**混合并行**取长补短：
- 张量并行：用于单机内多GPU
- 数据并行：用于多机间
- 流水线并行：进一步切分模型

### 5.2 3D混合并行

3D混合并行 = 数据并行 × 张量并行 × 流水线并行

```
8x8 GPU集群的3D并行布局：

      数据并行维度 (8)
    ┌─────────────────────┐
    │ GPU0,0 │ GPU0,1 │ ... │ GPU0,7 │ ← 同一行是数据并行
    ├────────┼────────┼─────┼────────┤
    │ GPU1,0 │ GPU1,1 │ ... │ GPU1,7 │
    ├────────┼────────┼─────┼────────┤
    │  ...   │  ...   │ ... │  ...   │
    ├────────┼────────┼─────┼────────┤
    │ GPU7,0 │ GPU7,1 │ ... │ GPU7,7 │ ← 张量并行在一行内
    └────────┴────────┴─────┴────────┘
              ↑
         流水线并行维度（按行切分）

例如：
- 每个节点8 GPU（8路张量并行）
- 8个节点（8路数据并行）
- 流水线切分模型到4个阶段
```

### 5.3 混合并行的通信

混合并行的通信模式：

| 通信类型 | 范围 | 说明 |
|----------|------|------|
| AllReduce | 张量并行组内 | 梯度平均 |
| AllGather | 流水线阶段间 | 激活传递 |
| AllReduce | 数据并行组间 | 梯度同步 |

**关键优化**：让通信与计算重叠。

### 5.4 混合并行的配置

使用PyTorch实现混合并行：

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel

# 1. 创建2D设备网格
mesh = init_device_mesh("cuda", (8, 8))  # 8路张量并行 x 8路数据并行
tp_mesh = mesh["tp"]
dp_mesh = mesh["dp"]

# 2. 张量并行（单机内）
tp_plan = {
    "attention.q_proj": ColwiseParallel(),
    "attention.k_proj": ColwiseParallel(),
    "attention.v_proj": ColwiseParallel(),
    "attention.out_proj": RowwiseParallel(),
    # ...
}
model = parallelize_module(model, tp_mesh, tp_plan)

# 3. 数据并行（多机间）
model = FSDP(model, device_mesh=dp_mesh)
```

### 5.5 混合并行的挑战

1. **拓扑感知**：需要考虑NVLink、PCIe、InfiniBand的带宽差异
2. **调度复杂**：多维度的并行需要协调
3. **内存管理**：不同层级的内存策略不同
4. **故障恢复**：大规模集群的容错

### 5.6 混合并行的最佳实践

1. **优先张量并行**：利用NVLink高带宽
2. **流水线并行用于跨机**：减少跨机通信
3. **数据并行作为最后手段**：当其他方式不够时
4. **ZeRO叠加**：进一步节省内存

---

## 6 分布式训练实战

### 6.1 环境设置

分布式训练需要正确的环境配置：

```bash
# 单机多卡：设置可见GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py

# 多机多卡：设置各节点信息
# Node 0:
torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    --node_rank=0 train.py

# Node 1:
torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    --node_rank=1 train.py
```

### 6.2 DDP训练模板

完整的DDP训练模板：

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

class Trainer:
    def __init__(self, model, train_loader, lr=0.01):
        # 初始化分布式
        self.setup_distributed()

        # 移动模型到GPU并包装DDP
        self.model = model.cuda()
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def setup_distributed(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            # 设置epoch（确保数据shuffle一致）
            self.train_loader.sampler.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0 and self.rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

    def cleanup(self):
        dist.destroy_process_group()

def main():
    # 创建模型和数据
    model = nn.Linear(784, 10)
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    trainer = Trainer(model, loader)
    trainer.train(n_epochs=10)
    trainer.cleanup()

if __name__ == "__main__":
    mp.spawn(main, args=(), nprocs=4)
```

### 6.3 FSDP训练模板

FSDP训练的完整模板：

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.multiprocessing as mp

# 混合精度配置
mixed_precision = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

def train_fsdp():
    # 初始化
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 创建模型
    model = MyTransformerModel()

    # FSDP包装
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        mixed_precision=mixed_precision,
        auto_wrap_policy=transformer_auto_wrap_policy,
        device_id=local_rank,
    )

    # 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = output.loss
            loss.backward()
            optimizer.step()

    # 清理
    dist.destroy_process_group()
```

### 6.4 梯度累积与分布式训练

梯度累积是扩大有效批次的常用技术：

```python
# 在DDP中使用梯度累积
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

model.train()
optimizer.zero_grad()

for step, (data, target) in enumerate(dataloader):
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 缩放损失
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 6.5 混合精度训练

使用torch.cuda.amp进行混合精度：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = model.cuda()

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    optimizer.zero_grad()

    # 自动混合精度
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # 缩放损失，反向传播
    scaler.scale(loss).backward()

    # 梯度裁剪（需要unscale）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

### 6.6 检查点保存与恢复

分布式训练的检查点需要特别处理：

```python
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, path):
    # 只在rank 0保存
    if dist.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    # 同步所有进程
    dist.barrier()

def load_checkpoint(model, optimizer, path):
    # 所有进程都加载（模型状态会被broadcast）
    checkpoint = torch.load(path, map_location=f'cuda:{local_rank}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 等待所有进程完成
    dist.barrier()
    return checkpoint['epoch']
```

### 6.7 性能调优

#### 6.7.1 CUDA加速

```python
# 启用CUDA加速
torch.backends.cudnn.benchmark = True

# 选择最佳算法
torch.backends.cudnn.deterministic = False
```

#### 6.7.2 数据加载优化

```python
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,           # 多进程数据加载
    pin_memory=True,         # 固定内存加速传输
    persistent_workers=True, # 保持worker进程
    prefetch_factor=2,       # 预取因子
)
```

#### 6.7.3 梯度同步优化

```python
# 使用梯度作为bucket，减少通信次数
model = DDP(model, bucket_cap_mb=25)
```

---

## 7 分布式通信原语

理解分布式通信原语对于优化分布式训练至关重要。

### 7.1 集合通信概述

| 原语 | 功能 | 典型用途 |
|------|------|----------|
| AllReduce | 所有设备求和/平均 | 梯度同步 |
| Broadcast | 一对多广播 | 参数同步 |
| AllGather | 收集所有设备数据 | 张量并行 |
| ReduceScatter | 分散求和 | 流水线并行 |
| Send/Recv | 点对点通信 | Pipeline调度 |

### 7.2 AllReduce

AllReduce是最常用的集合通信原语：

```python
import torch.distributed as dist

# 所有设备求和，结果同步到所有设备
tensor = torch.randn(1024).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# 现在所有设备的tensor值相同（都等于原始值的和）
```

### 7.3 Broadcast

Broadcast用于广播数据：

```python
# 只在rank 0有有效数据
if rank == 0:
    data = torch.randn(1024)
else:
    data = torch.zeros(1024)

# 广播给所有设备
dist.broadcast(data, src=0)

# 现在所有设备都有相同的数据
```

### 7.4 AllGather

AllGather收集所有设备的数据：

```python
# 每个设备有不同的输入
local_tensor = torch.tensor([rank] * 10).cuda()

# 收集所有设备的数据
world_size = dist.get_world_size()
gather_list = [torch.zeros(10).cuda() for _ in range(world_size)]
dist.all_gather(gather_list, local_tensor)

# gather_list[0] = [0,0,...0]
# gather_list[1] = [1,1,...1]
# ...
```

### 7.5 点对点通信

Send和Recv用于更灵活的通信：

```python
# 发送
if rank == 0:
    tensor = torch.randn(1024).cuda()
    dist.send(tensor, dst=1)

# 接收
if rank == 1:
    tensor = torch.zeros(1024).cuda()
    dist.recv(tensor, src=0)
```

### 7.6 通信与计算重叠

关键的优化是让通信与计算重叠：

```python
# 错误的顺序：先通信后计算
loss.backward()  # 计算梯度
dist.all_reduce(grad)  # 通信（等待）
optimizer.step()

# 正确的顺序：通信与计算重叠
loss.backward()  # 计算部分梯度
# 启动异步AllReduce
handle = dist.all_reduce(grad, async_op=True)
# 在通信进行时可以做其他计算
compute_something()
# 等待通信完成
handle.wait()
optimizer.step()
```

---

## 8 分布式训练中的常见问题

### 8.1 梯度不一致

问题：不同设备的梯度值有微小差异

原因：
- 浮点精度
- 计算顺序不同

解决：
```python
# 使用reducers进行同步
dist.all_reduce(grad, op=dist.ReduceOp.SUM)
grad.div_(world_size)
```

### 8.2 死锁

问题：多个进程互相等待

原因：
- 通信调用不匹配
- 同步点过多

解决：
```python
# 使用barrier同步
dist.barrier()

# 确保所有进程的通信调用一致
```

### 8.3 负载不均衡

问题：某些GPU空闲等待

原因：
- 各设备数据量不同
- 模型切分不均匀

解决：
```python
# 使用DistributedSampler均匀分配
sampler = DistributedSampler(dataset, shuffle=True)
```

### 8.4 通信瓶颈

问题：训练速度受通信限制

解决：
- 增加计算量（增大批次）
- 使用梯度压缩
- 选择合适的并行策略
- 利用NVLink等高速互连

---

## 本章小结

本章系统介绍了分布式训练的核心概念与技术。

**核心要点回顾**：

1. **分布式训练的必要性**：大模型参数量巨大，单设备无法容纳和高效训练。

2. **数据并行（DP/DDP）**：
   - 复制模型到多设备，处理不同数据
   - DDP比DP更高效，使用Ring AllReduce同步梯度
   - 通信与计算可重叠

3. **FSDP与ZeRO优化**：
   - ZeRO通过分片减少内存冗余
   - ZeRO-1/2/3分别优化器状态、梯度、参数分片
   - 通信量增加换取内存节省

4. **张量并行**：
   - 将单层参数切分到多设备
   - 适合Transformer等层
   - 通信开销较大

5. **流水线并行**：
   - 将不同层分配到不同设备
   - 使用微批次减少气泡
   - Gpipe和PipeDream是两种主要策略

6. **混合并行**：
   - 结合数据并行、张量并行、流水线并行
   - 3D并行是大模型训练主流
   - 需要考虑拓扑和负载均衡

7. **通信原语**：
   - AllReduce、Broadcast、AllGather等
   - 通信与计算重叠是关键优化

**思考与练习**：

1. 为什么DDP比DP效率更高？
2. ZeRO-3相比数据并行，通信量增加了多少？内存节省了多少？
3. 张量并行和流水线并行各适合什么场景？
4. 为什么微批次能减少流水线的气泡？
5. 混合并行中，为什么通常把张量并行放在单机内，流水线并行放在跨机？
6. 如何调试分布式训练中的死锁问题？
7. 如果网络带宽有限，应该选择哪种并行策略？为什么？

---

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=263809674&bvid=BV1ve411w7DL&cid=925113485&page=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->
