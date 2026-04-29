# AI集群

## 学习目标

1. 理解AI集群的整体架构和组成
2. 掌握计算节点、存储系统、网络互联的关键技术
3. 了解NCCL和RDMA等通信原语
4. 熟悉Kubernetes在GPU调度中的应用
5. 理解分布式训练中的通信优化策略

## 2.1 AI集群架构概述

### 2.1.1 为什么需要AI集群

训练大规模神经网络模型是一项计算密集型任务。以GPT-3为例，其训练据估计需要约3.64 × 10^23次浮点运算（FLOP），使用单张NVIDIA A100 GPU需要大约304年的时间。即便是使用数千张GPU组成集群进行分布式训练，也需要数周甚至数月才能完成。

因此，**AI集群是训练大模型的必要基础设施**。一个设计良好的AI集群需要综合考虑计算能力、内存带宽、网络互联、存储系统等多个方面，任何一个环节的瓶颈都可能限制整体训练效率。

### 2.1.2 AI集群的总体架构

现代AI集群通常采用分层架构设计，主要包括以下几个层次：

**计算层**：由GPU、NPU、TPU等AI加速器组成，是集群的核心计算资源。现代AI服务器通常配备8张或更多加速卡，通过NVLink实现高速互联。

**网络层**：负责计算节点之间的数据传输。AI集群通常使用InfiniBand或RoCE（RDMA over Converged Ethernet）等高速网络，提供100Gbps甚至400Gbps的网络带宽。

**存储层**：提供高速并行文件系统，用于存储训练数据、模型checkpoint等。常用的方案包括Google File System、Parallel Virtual File System等。

**调度层**：负责管理集群资源、调度训练任务。Kubernetes已成为事实上的标准容器编排平台，配合GPU调度插件实现高效的资源管理。

```
┌─────────────────────────────────────────────────────────┐
│                      调度层                              │
│              (Kubernetes, 资源管理, 任务调度)              │
├─────────────────────────────────────────────────────────┤
│                      网络层                              │
│         (InfiniBand / RoCE, 高速交换器, 路由)              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ 计算节点 │  │ 计算节点 │  │ 计算节点 │  │ 计算节点 │    │
│  │  (8 GPU)│  │  (8 GPU)│  │  (8 GPU)│  │  (8 GPU)│    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
├─────────────────────────────────────────────────────────┤
│                      存储层                              │
│            (并行文件系统, 数据存储, Checkpoint)            │
└─────────────────────────────────────────────────────────┘
```

### 2.1.3 AI集群的关键指标

评估一个AI集群的性能，主要关注以下关键指标：

**算力（Computation Power）**：通常以FP64/FP32/FP16/TFLOPS（每秒万亿次浮点运算）来衡量。NVIDIA H100的FP16算力高达1979 TFLOPS。

**内存带宽（Memory Bandwidth）**：决定数据从显存到计算单元的传输速度。H100的内存带宽为3.35 TB/s。

**网络带宽（Network Bandwidth）**：决定节点间数据传输速度。InfiniBand HDR可以提供200 Gbps的单向带宽。

**延迟（Latency）**：影响集合通信效率。RDMA技术可以将网络延迟降低到微秒级。

**可扩展性（Scalability）**：集群规模扩大时，效率损失要尽可能小。

## 2.2 计算节点

### 2.2.1 计算节点架构

计算节点是AI集群的基本单元，通常由以下组件构成：

**AI加速器**：如NVIDIA GPU（V100、A100、H100系列）或华为昇腾NPU。加速器通过PCIe或NVLink与CPU连接。

**主处理器（CPU）**：负责协调计算任务、管理数据流动。现代服务器通常配置两颗高性能CPU。

**本地存储**：用于存放临时数据和启动镜像。通常使用NVMe SSD，容量在TB级别。

**内存（DRAM）**：为主处理器和加速器提供数据缓存。服务器配置通常在TB级别。

**高速互联**：如NVLink（GPU间）、InfiniBand或RoCE（节点间）。

一个典型的8-GPU服务器配置：

| 组件 | 规格 |
|-----|-----|
| CPU | 2 × Intel Xeon Gold / AMD EPYC |
| GPU | 8 × NVIDIA A100/H100 (80GB) |
| 内存 | 2 TB DRAM |
| 本地存储 | 2 × 3.84 TB NVMe SSD |
| 网络 | 200 Gbps InfiniBand HDR |

### 2.2.2 GPU架构与并行计算

GPU（Graphics Processing Unit）最初为图形渲染设计，但其大规模并行计算架构非常适合深度学习工作负载。

**GPU架构特点**：

1. **大量轻量级核心**：GPU拥有数千个CUDA核心，可以同时执行大量并行任务
2. **高带宽显存**：如HBM（High Bandwidth Memory），提供TB/s级的访问带宽
3. **Tensor Core**：专门为矩阵运算优化的硬件单元，大幅加速FP16/BF16计算
4. **统一内存**：支持GPU和CPU共享虚拟地址空间，简化编程

**GPU并行计算模型**：

在GPU上，线程被组织成三层结构：
- **线程（Thread）**：最基本的执行单元
- **线程块（Block）**：一组线程共享共享内存，可以协作执行
- **网格（Grid）**：所有线程块组成，负责执行整个kernel

GPU通过SIMT（Single Instruction Multiple Thread）模式执行，32个线程为一组（ warp）以相同频率执行指令。

### 2.2.3 多GPU互联技术

当训练任务需要跨多个GPU时，GPU之间的互联带宽成为关键因素。

**PCIe**：传统的GPU-CPU连接方式，但带宽有限（PCIe 4.0 x16提供约64 GB/s双向带宽），已成为瓶颈。

**NVLink**：NVIDIA开发的高速互联技术，可以在GPU之间以及GPU-CPU之间提供直接连接。H100支持18条NVLink通道，提供900 GB/s的双向带宽。

**NVSwitch**：连接多个GPU的交换芯片，允许任意GPU之间进行高速通信。DGX H100使用4颗NVSwitch连接8张GPU。

在实际训练中，跨节点通信的开销往往大于节点内通信。因此，在设计分布式训练策略时，应当优先考虑节点内并行，将通信密集的操作限制在节点内部。

## 2.3 存储系统

### 2.3.1 训练任务的存储需求

大规模神经网络训练对存储系统提出了极高的要求：

**海量数据存储**：训练数据集可能达到TB甚至PB级别。例如，LLM训练可能需要数万亿个token的文本数据。

**高吞吐量**：训练过程中需要频繁读取数据样本和模型权重。存储带宽必须足够高以避免成为瓶颈。

**高并发**：数千个计算节点可能同时访问存储系统，要求存储系统能够支持大量并发读取。

**checkpoint保存**：训练过程中需要定期保存模型状态，checkpoint可能达到数百GB。

### 2.3.2 分布式文件系统

为了满足上述需求，AI集群通常部署分布式文件系统：

**并行虚拟文件系统（PVFS）**：一种专门为高性能计算设计的并行文件系统，广泛用于AI和科学计算场景。

**Google File System（GFS）**：Google为海量数据存储开发的分布式文件系统，具有高可靠性和高吞吐量特点。

**Lustre**：开源并行文件系统，具有良好的可扩展性，支持TB/s级的聚合带宽。

**阿里云OSS/Amazon S3**：云端的分布式对象存储服务，提供海量、低成本的存储能力。

### 2.3.3 存储性能优化

为了最大化存储系统效率，通常采用以下优化策略：

**数据预取（Data Prefetching）**：在计算进行的同时异步读取下一批数据，隐藏IO延迟。

**数据缓存（Data Caching）**：将热点数据缓存在本地或分布式缓存层，减少远程访问。

**数据分割（Data Sharding）**：将大文件分散到多个存储节点，实现并行读取。

**检查点优化**：采用增量checkpoint、异步checkpoint等技术，减少checkpoint对训练的影响。

## 2.4 网络互联

### 2.4.1 网络架构的重要性

在分布式训练中，网络是连接各个计算节点的桥梁。网络带宽和延迟直接影响集合通信的效率，从而影响整体训练性能。

**关键影响**：

1. **梯度同步**：数据并行中，各节点需要同步梯度，带宽不足会限制并行效率
2. **模型参数同步**：流水线并行中，不同阶段需要传递中间结果
3. **数据加载**：多个节点同时读取训练数据，对网络带宽需求高

### 2.4.2 主流网络技术

**InfiniBand**：

InfiniBand是一种高性能网络技术，提供高带宽、低延迟的特性。

| 版本 | 带宽 | 延迟 |
|-----|-----|-----|
| FDR | 56 Gbps | ~1微秒 |
| EDR | 100 Gbps | ~0.6微秒 |
| HDR | 200 Gbps | ~0.5微秒 |
| NDR | 400 Gbps | ~0.3微秒 |

InfiniBand支持RDMA（Remote Direct Memory Access），允许直接访问远程内存，无需CPU介入。

**RoCE（RDMA over Converged Ethernet）**：

RoCE将RDMA技术运行在标准以太网上，降低了部署成本。它有两个版本：
- **RoCE v1**：使用Ethernet帧传输InfiniBand数据包
- **RoCE v2**：添加了IP和UDP头部，支持路由

RoCE的优势在于可以复用现有的以太网基础设施，但需要无损网络支持（如PFC流量控制）。

**HTTP/2、TCP等传统网络**：

传统网络协议开销较大，不适合大规模分布式训练场景，通常仅用于管理流量。

### 2.4.3 网络拓扑设计

AI集群的网络拓扑设计对整体性能至关重要。

**传统三层网络**：

- Edge Switch（接入层）：连接服务器
- Aggregation Switch（汇聚层）：聚合多个接入层
- Core Switch（核心层）：连接汇聚层和外部网络

这种拓扑在AI训练场景中可能造成性能瓶颈，因为跨服务器的通信需要经过多跳。

**Fat-Tree拓扑**：

一种多根树结构，每一层都均匀地连接到上一层，提供多路径转发。适合AI集群的高带宽需求。

**DragonFly拓扑**：

一种高度互联的网络拓扑，每个交换机与其他交换机直接连接，减少跳数，提供低延迟通信。

现代AI集群通常采用胖树（Fat-Tree）或DragonFly拓扑，配合InfiniBand或RoCE网络。

## 2.5 通信原语

### 2.5.1 集合通信的概念

在分布式训练中，计算节点之间需要频繁通信以同步数据和梯度。这些通信操作通常由**集合通信原语（Collective Communication Primitives）**来实现。

集合通信与点对点通信不同，它涉及一组节点之间的数据交换。常见的集合通信操作包括：

- **AllReduce**：所有节点贡献数据，最终每个节点得到汇总结果
- **AllGather**：收集所有节点的数据，每个节点得到完整数据
- **Broadcast**：从一个节点向所有节点发送数据
- **ReduceScatter**：对数据进行规约并分散到各节点

### 2.5.2 NCCL通信库

**NCCL**（NVIDIA Collective Communications Library）是NVIDIA开发的集合通信库，专门针对GPU集群优化。

**NCCL的核心功能**：

1. **GPU间通信**：通过NVLink、PCIe等高速互联实现GPU直接通信
2. **集合通信原语**：提供AllReduce、AllGather、Broadcast等操作
3. **拓扑感知**：自动检测网络拓扑，优化通信路径
4. **多GPU和多节点**：支持单节点多GPU和多节点扩展

**NCCL的工作原理**：

NCCL使用**环形通信（Ring Communication）**算法进行AllReduce操作。假设有N个节点，数据被分成N份，每个节点将自己的第i份发送给下一个节点，同时接收上一个节点的第i-1份并累加。

这种环形通信的优势：
- 通信与计算可以重叠
- 适合高带宽网络
- 扩展性好，带宽利用率稳定

**NCCL使用示例**：

```python
import torch
import torch.distributed as dist
import ncclpy

# 初始化分布式环境
dist.init_process_group(backend="nccl")

# 获取本地GPU ID
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 创建 NCCL 通信组
group = dist.new_group([0, 1, 2, 3])

# AllReduce 操作
tensor = torch.randn(1024, 1024).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
```

### 2.5.3 RDMA通信技术

**RDMA**（Remote Direct Memory Access）是一种直接内存访问技术，允许在主机间直接传输数据，无需CPU介入。

**RDMA的优势**：

1. **零拷贝（Zero-Copy）**：数据可以直接从网卡写入内存，无需内核拷贝
2. **低延迟**：绕过操作系统，约1-2微秒
3. **低CPU占用**：不消耗本端和远端CPU资源
4. **高带宽**：充分利用网络带宽

**RDMA的关键概念**：

- **QP（Queue Pair）**：发送队列和接收队列的组合
- **CQ（Completion Queue）**：记录已完成的操作
- **MR（Memory Region）**：已注册的内存区域
- **GID（Global ID）**：标识RDMA设备在全球网络中的位置

**RDMA操作类型**：

1. **单边操作**：Read、Write、Send/Recv
2. **双边操作**：Send/Recv需要远端参与

**NCCL与RDMA的关系**：

NCCL 2.0之后的版本支持通过RoCE和InfiniBand使用RDMA进行数据传输，提供更高效的通信性能。

### 2.5.4 集合通信算法

不同的集合通信算法适用于不同的网络拓扑和通信模式。

**AllReduce算法**：

1. **Ring AllReduce**：适合带宽受限的场景，时间复杂度O(n)
2. **Tree AllReduce**：适合延迟敏感的场景，深度O(log n)
3. **Rabenseifner's算法**：基于树形结构的优化算法

**Ring AllReduce详解**：

假设有N个节点，每个节点有M大小的数据需要求和：

1. **Reduce-Scatter阶段**：进行N-1轮传递，每轮每个节点发送和接收M/N大小的数据
2. **AllGather阶段**：进行N-1轮传递，每轮每个节点发送和接收M/N大小的数据

总通信量约为2 × (N-1) × M/N ≈ 2M，适合大模型训练场景。

**广播算法**：

Broadcast操作通常使用树形结构，从根节点开始逐层向下传播：

1. 根节点将数据发送到下一层的所有节点
2. 每个收到数据的节点继续向下传播
3. 深度O(log n)，总通信量M × (n-1)

## 2.6 调度系统

### 2.6.1 容器与容器编排

**容器（Container）**是一种轻量级的虚拟化技术，将应用程序及其依赖打包在一起，确保应用在任何环境下都能一致运行。

Docker是最流行的容器运行时之一，它的核心概念包括：
- **镜像（Image）**：只读模板，包含应用程序和依赖
- **容器（Container）**：镜像的运行实例
- **仓库（Registry）**：存储和分发镜像的仓库

**容器编排（Container Orchestration）**是管理多容器应用的自动化平台，负责：
- 容器部署和调度
- 服务发现和负载均衡
- 容器扩缩容
- 健康检查和自愈

### 2.6.2 Kubernetes概述

**Kubernetes**（简称K8s）是目前最主流的容器编排平台，最初由Google开发，现在由CNCF维护。

**Kubernetes的核心概念**：

1. **Pod**：Kubernetes的基本调度单元，可以包含一个或多个容器
2. **Node**：工作节点，运行Pod的服务器
3. **Cluster**：一组Node组成的计算集群
4. **Deployment**：声明式地管理Pod副本集
5. **Service**：为一组Pod提供稳定的访问入口

**Kubernetes架构**：

```
┌────────────────────────────────────────────┐
│              Control Plane                 │
│  ┌────────┐ ┌────────┐ ┌────────────────┐ │
│  │ API    │ │Sched-  │ │ Controller     │ │
│  │ Server │ │ uler   │ │ Manager        │ │
│  └────────┘ └────────┘ └────────────────┘ │
└────────────────────────────────────────────┘
                    │
┌────────────────────────────────────────────┐
│              Data Plane                     │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │ Node │  │ Node │  │ Node │  │ Node │   │
│  │ Kubelet + Kube-Proxy + Pods            │ │
│  └──────┘  └──────┘  └──────┘  └──────┘   │
└────────────────────────────────────────────┘
```

### 2.6.3 GPU调度

Kubernetes原生不支持GPU调度，需要通过设备插件（Device Plugin）扩展来实现。

**NVIDIA Device Plugin**：

Kubernetes通过NVIDIA Device Plugin来识别和管理GPU资源。工作流程：

1. 在每个GPU节点上部署Device Plugin DaemonSet
2. Device Plugin向Kubernetes报告节点上的GPU数量
3. 用户提交GPU需求，调度器选择合适的节点
4. Pod调度到节点后，Device Plugin负责GPU分配

**GPU调度示例**：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: train-container
    image: pytorch/pytorch:2.0.1
    resources:
      limits:
        nvidia.com/gpu: 8  # 请求8个GPU
    command: ["python", "train.py"]
```

**GPU调度策略**：

1. **binpack**：优先将Pod调度到GPU密集的节点
2. **spread**：将Pod的GPU分散到不同节点，提高可靠性
3. **node sizing**：根据GPU需求选择最合适的节点大小

### 2.6.4 调度策略优化

在大规模训练场景中，需要精细的调度策略来优化资源利用率。

**gang调度**：

训练任务通常需要所有Worker同时启动才能开始工作。Gang调度确保所有相关的Pod同时被调度，避免部分Worker等待导致的资源浪费。

**优先级调度**：

为不同类型的任务设置优先级：
- 训练任务通常需要高优先级，保证及时完成
- 评估任务可以适当降低优先级
- 批量推理任务可以在低峰期执行

**抢占调度**：

高优先级任务可以抢占低优先级任务的资源，用于紧急训练需求。

**资源配额**：

通过ResourceQuota和LimitRange限制每个团队/项目的资源使用，防止资源争抢。

### 2.6.5 弹性训练调度

弹性训练允许在训练过程中动态添加或移除计算节点，对于大规模训练非常有价值。

**PyTorch Elastic (TorchElastic)**：

TorchElastic是PyTorch提供的弹性训练框架，支持：
- 动态调整Worker数量
- 节点故障自动恢复
- 容错训练

**Kubernetes上的弹性训练**：

结合Kubernetes和TorchElastic：
- 使用Kubernetes Deployment管理训练Pod
- 配置Pod数量范围（min:max）
- TorchElastic处理Worker间的协调

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elastic-training
spec:
  replicas: 8  # 初始副本数
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.1
        resources:
          limits:
            nvidia.com/gpu: 1
```

## 2.7 通信优化

### 2.7.1 通信与计算的重叠

在分布式训练中，如果等到所有计算完成后再进行通信，GPU将长时间处于空闲状态。**通信与计算重叠**是提升训练效率的关键技术。

**重叠的工作原理**：

1. 将计算过程分成多个阶段
2. 在计算当前阶段的同时，异步通信前一个阶段的梯度
3. 当前一阶段计算完成且通信也完成时，进入下一阶段

**PyTorch中的实现**：

PyTorch的DistributedDataParallel（DDP）通过以下机制实现重叠：
- 使用多个bucket存储梯度
- 每个bucket的梯度计算完成后立即触发通信
- 通信与后续bucket的计算并行执行

### 2.7.2 梯度压缩

梯度通信是数据并行中的主要开销之一。**梯度压缩**通过减少通信数据量来加速训练。

**主要压缩方法**：

1. **量化压缩（Quantization）**：
   - 将32位浮点梯度压缩为较低精度（如FP16、INT8）
   - 压缩率高，但可能影响模型精度
   - 需要合适的解压缩策略

2. **稀疏压缩（Sparsification）**：
   - 只传输Top-K最重要的梯度
   - 需要额外传递索引信息
   - 压缩率取决于稀疏度

3. **随机压缩（Random Skipping）**：
   - 随机选择一部分梯度进行传输
   - 类似于Dropout，但需要校正

### 2.7.3 自适应通信

不同训练阶段对通信的需求不同。**自适应通信**根据当前训练状态动态调整通信策略。

**基于梯度的自适应**：

- 当梯度变化较小时，可以减少通信频率
- 当检测到收敛困难时，增加通信精度

**基于拓扑的自适应**：

- 节点内通信使用高速链路（NVLink）
- 跨节点通信使用较慢链路（InfiniBand）
- 优化通信路径，减少跨节点流量

### 2.7.4 通信库配置优化

正确配置通信库可以显著提升性能。

**NCCL配置参数**：

```bash
# 设置NCCL缓存大小
NCCL_BUFFSIZE=1048576

# 启用NCCL调试日志
NCCL_DEBUG=INFO

# 设置网络插件
NCCL_NET_GDR_LEVEL=PIX  # 或 PXD
```

**环境变量优化**：

```bash
# 绑定GPU到NUMA节点
CUDA_VISIBLE_DEVICES=0,1,2,3

# 启用TCP绑定
NCCL_SOCKET_IFNAME=eth0

# 设置TCP端口范围
NCCL_IB_PORTS=1
NCCL_IB_TC=106  # 流量类别
```

## 小结

本节我们详细学习了AI集群的构成和关键技术。

**核心要点**：

1. AI集群采用分层架构，包括计算层、网络层、存储层和调度层，需要各层协同配合才能发挥最佳性能
2. 计算节点通常配备多GPU，通过NVLink实现高速互联，GPU是分布式训练的核心计算资源
3. 分布式文件系统提供高吞吐量、高并发的数据访问能力，是训练海量数据的基础设施
4. 网络互联技术（InfiniBand、RoCE、RDMA）提供了高带宽、低延迟的通信能力
5. NCCL是GPU集合通信的核心库，提供了高效的集合通信原语实现
6. Kubernetes已成为AI集群调度的标准平台，GPU调度通过设备插件实现
7. 通信优化技术（计算通信重叠、梯度压缩等）是提升分布式训练效率的关键

**思考题**：

1. AI集群为什么需要专门的高速网络？传统TCP/IP网络为什么不适用？
2. Ring AllReduce算法相比其他AllReduce算法有什么优势和劣势？
3. NCCL是如何利用网络拓扑信息优化通信路径的？
4. 在设计分布式训练策略时，为什么要优先考虑节点内并行？
5. Kubernetes的gang调度对于训练任务有什么重要意义？