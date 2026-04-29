# 04 推理优化技术

## 学习目标

1. 理解推理优化的核心目标和评价指标
2. 掌握延迟优化的主要技术和策略
3. 了解吞吐量优化的方法，包括并发推理和模型并行
4. 掌握内存优化的核心技术，包括内存复用和显存优化
5. 理解推理优化的综合实践方法

## 4.1 推理优化概述

### 4.1.1 推理优化的目标

推理优化是确保深度学习模型在生产环境中高效运行的关键环节。与模型压缩不同，推理优化更关注**如何在给定硬件和模型条件下最大化推理性能**。

**核心优化目标**

1. **延迟（Latency）**：单个请求从发送到返回结果的端到端时间。对于实时交互应用（如语音助手、自动驾驶），延迟是关键指标。

2. **吞吐量（Throughput）**：单位时间内系统处理的请求数量。对于高并发服务（如推荐系统、搜索排序），吞吐量是关键指标。

3. **资源效率（Resource Efficiency）**：单位资源消耗能够处理的请求数量。资源包括CPU/GPU时间、内存、显存、带宽等。

4. **能效比（Energy Efficiency）**：单位能耗能够处理的请求数量。对于边缘设备和移动端尤为重要。

**延迟与吞吐量的关系**

延迟和吞吐量往往是相互制约的：

- 低延迟系统通常采用**小批次**甚至**单请求**处理，减少等待时间
- 高吞吐量系统通常采用**大批次**处理，通过并行化分摊固定开销

理想情况下，我们希望同时优化延迟和吞吐量。但在实际系统中，需要根据业务场景进行权衡：

- 实时交互应用 → 优化延迟优先
- 批处理服务 → 优化吞吐量优先
- 在线推理服务 → 两者兼顾

### 4.1.2 性能评价指标

**延迟指标**

- **平均延迟（Average Latency）**：所有请求延迟的算术平均值
- **P50延迟（P50 Latency）**：50%请求的延迟低于此值
- **P95延迟（P95 Latency）**：95%请求的延迟低于此值
- **P99延迟（P99 Latency）**：99%请求的延迟低于此值
- **最大延迟（Max Latency）**：所有请求中延迟的最大值

P95/P99延迟称为**尾部延迟（Tail Latency）**，反映了用户体验的"最坏情况"。对于SLA要求高的服务，尾部延迟是关键指标。

**吞吐量指标**

- **QPS（Queries Per Second）**：每秒请求数
- **RPS（Requests Per Second）**：每秒请求数（与QPS类似）
- **TPS（Transactions Per Second）**：每秒事务数
- **FPS（Frames Per Second）**：每秒处理的帧数（用于视频/图像场景）

**资源效率指标**

- **GPU利用率（GPU Utilization）**：GPU计算单元的实际使用百分比
- **显存带宽利用率（Memory Bandwidth Utilization）**：实际显存带宽使用相对于峰值的比例
- **CPU/GPU空闲时间（Idle Time）**：设备等待数据或同步的时间比例

### 4.1.3 性能瓶颈分析

**计算瓶颈（Compute Bound）**

当GPU计算能力不足时，称为计算瓶颈。表现为GPU利用率高，但内存带宽未充分利用。

可能的解决方案：

- 使用更高效的算子实现
- 启用TensorCore等专用计算单元
- 使用更低精度的计算（如FP16替代FP32）

**内存带宽瓶颈（Memory Bound）**

当GPU等待数据传输时，称为内存带宽瓶颈。表现为显存带宽利用率低，GPU经常空闲等待数据。

可能的解决方案：

- 使用更高效的数据布局（如NCHW4）
- 启用内存复用，减少数据传输
- 使用异步计算，隐藏内存访问延迟

**内存容量瓶颈（Memory Capacity Bound）**

当模型或中间结果无法完全放入显存时，称为内存容量瓶颈。表现为频繁的CPU-GPU数据传输。

可能的解决方案：

- 启用模型并行，将模型分布在多卡
- 启用内存复用，优化中间结果生命周期
- 使用更小的数据类型（如INT8替代FP32）

## 4.2 延迟优化

### 4.2.1 批处理优化

**动态批处理（Dynamic Batching）**

动态批处理是平衡延迟和吞吐量的关键技术。核心思想是：

- 当请求较少时，不等待，直接处理当前请求（保持低延迟）
- 当请求较多时，积攒多个请求形成批次处理（提高吞吐量）

**实现策略**

```python
class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_time_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time_ms / 1000.0
        self.requests = []
    
    def add_request(self, request):
        self.requests.append(request)
        
        # 检查是否可以形成批次
        if len(self.requests) >= self.max_batch_size:
            return self._form_batch()
        
        # 检查是否超时
        if self._wait_timeout():
            return self._form_batch()
        
        return None  # 继续等待
    
    def _form_batch(self):
        if not self.requests:
            return None
        batch = self.requests[:self.max_batch_size]
        self.requests = self.requests[self.max_batch_size:]
        return batch
```

**批处理优化策略**

1. **自适应批次大小**：根据当前负载动态调整最大批次大小
2. **优先级调度**：重要请求优先处理，避免高优请求等待
3. **亲和性调度**：将相似请求调度到同一实例，提高缓存命中率

### 4.2.2 异步执行

**同步执行 vs 异步执行**

同步执行模式下，请示必须等待前一个请求完成后才能处理。异步执行允许请求的并行处理。

```
同步执行：
请求1 → 请求2 → 请求3 → ...
         等待     等待

异步执行：
请求1 ──────────→ 返回
请求2 ──→ 返回
请求3 ────→ 返回
```

**CUDA异步操作**

CUDA中的异步操作允许计算和数据传输并行进行：

```cuda
// 异步内存拷贝
cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);

// 异步执行Kernel
my_kernel<<<blocks, threads, 0, stream>>>(d_data);

// 并行进行：CPU准备下一个batch的数据，GPU处理当前batch
```

**流（Stream）管理**

CUDA Stream允许指定操作的执行顺序和依赖关系：

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Stream 1: 执行推理
cudaMemcpyAsync(d_input, h_input1, size, cudaMemcpyHostToDevice, stream1);
infer_kernel<<<..., stream1>>>(d_input, d_output);
cudaMemcpyAsync(h_output1, d_output, size, cudaMemcpyDeviceToHost, stream1);

// Stream 2: 同时进行下一个请求的数据准备
cudaMemcpyAsync(d_input2, h_input2, size, cudaMemcpyHostToDevice, stream2);
infer_kernel<<<..., stream2>>>(d_input2, d_output2);
// ...
```

### 4.2.3 预热与缓存

**模型预热（Warmup）**

首次推理时，由于CPU-GPU数据传输、内核JIT编译、缓存预填充等因素，延迟通常较高。预热通过在服务启动后执行若干次推理来解决这个问题：

```python
def warmup(model, warmup_inputs, num_warmup=10):
    """预热模型"""
    for i in range(num_warmup):
        _ = model(warmup_inputs[i % len(warmup_inputs)])
    
    # 同步，确保所有操作完成
    torch.cuda.synchronize()
```

**计算结果缓存**

对于输入分布相对稳定的场景，可以缓存部分计算结果避免重复计算：

```python
class ResultCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
    
    def get(self, input_hash):
        if input_hash in self.cache:
            self.access_count[input_hash] += 1
            return self.cache[input_hash]
        return None
    
    def put(self, input_hash, result):
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[input_hash] = result
        self.access_count[input_hash] = 1
```

### 4.2.4 流水线并行

**推理流水线**

将推理过程分解为多个阶段，每个阶段由专用资源处理，形成流水线：

```
请求 → 预处理 → 推理 → 后处理 → 响应
         ↓          ↓         ↓
       阶段1      阶段2     阶段3
```

不同阶段的计算特性和资源需求不同，流水线并行可以让各阶段并行处理：

```python
from queue import Queue
from threading import Thread

class InferencePipeline:
    def __init__(self):
        self.preprocess_queue = Queue(maxsize=100)
        self.inference_queue = Queue(maxsize=100)
        self.postprocess_queue = Queue(maxsize=100)
        
        # 启动各阶段线程
        self.preprocess_thread = Thread(target=self._preprocess_loop)
        self.inference_thread = Thread(target=self._inference_loop)
        self.postprocess_thread = Thread(target=self._postprocess_loop)
    
    def _preprocess_loop(self):
        while True:
            request = self.preprocess_queue.get()
            preprocessed = self._preprocess(request)
            self.inference_queue.put(preprocessed)
    
    def _inference_loop(self):
        while True:
            preprocessed = self.inference_queue.get()
            result = self.model(preprocessed)
            self.postprocess_queue.put(result)
    
    def _postprocess_loop(self):
        while True:
            result = self.postprocess_queue.get()
            response = self._postprocess(result)
            # 发送响应...
```

## 4.3 吞吐量优化

### 4.3.1 并发推理

**多实例部署**

部署多个模型实例并行处理请求，是提高吞吐量的基本方法：

```yaml
# Kubernetes Deployment配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 4  # 4个副本
  template:
    spec:
      containers:
      - name: inference
        image: my-inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # 每个副本使用1个GPU
```

**多实例+负载均衡**

```python
import asyncio
from aiohttp import web

class LoadBalancer:
    def __init__(self, instances):
        self.instances = instances
        self.current = 0
    
    async def route(self, request):
        # 轮询选择实例
        instance = self.instances[self.current]
        self.current = (self.current + 1) % len(self.instances)
        return await instance.process(request)
    
    async def health_check(self):
        # 健康检查，移除不健康的实例
        healthy = []
        for inst in self.instances:
            if await inst.is_healthy():
                healthy.append(inst)
        self.instances = healthy
```

### 4.3.2 模型并行

当单个GPU无法容纳整个模型时，需要将模型分布到多个GPU上：

**模型切分策略**

1. **按层切分**：将不同层放到不同GPU
2. **按通道切分**：将同一层的不同通道放到不同GPU
3. **混合切分**：结合前两种策略

**张量并行（Tensor Parallelism）**

张量并行将单个层的参数矩阵切分到多个GPU：

对于矩阵乘法$Y = XA$，其中$A$是$m \times n$的权重矩阵，可以将$A$按列切分为$[A_1, A_2]$，分别放到GPU1和GPU2：

```python
class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_gpus=2):
        super().__init__()
        self.num_gpus = num_gpus
        # 每个GPU只存储out_features/num_gpus列
        self.weight_shards = [
            nn.Parameter(torch.randn(out_features//num_gpus, in_features))
            for _ in range(num_gpus)
        ]
    
    def forward(self, x):
        # 将输入广播到所有GPU
        x = x.cuda()
        # 各GPU并行计算部分结果
        partial_outputs = [
            F.linear(x, weight.cuda(i))
            for i, weight in enumerate(self.weight_shards)
        ]
        # 汇总结果
        return torch.cat(partial_outputs, dim=-1)
```

**流水线并行（Pipeline Parallelism）**

流水线并行将不同层放到不同GPU，形成"生产者-消费者"关系：

```
GPU0: Layer0 → Layer1 → Layer2 →
GPU1:                Layer3 → Layer4 → Layer5 →
GPU2:                             Layer6 → Layer7 → Output
```

为减少流水线气泡，通常使用**微批次（Microbatch）**调度：

```python
class PipelineParallel(nn.Module):
    def __init__(self, stage_devices, stage_layers):
        self.stages = nn.ModuleList()
        for device, num_layers in zip(stage_devices, stage_layers):
            stage = nn.Sequential(*[create_layer() for _ in range(num_layers)])
            self.stages.append(stage.to(device))
    
    def forward(self, x):
        microbatches = x.chunk(self.num_microbatches)
        outputs = []
        
        for mb in microbatches:
            # 顺序流经各阶段
            for stage, device in zip(self.stages, self.stage_devices):
                mb = mb.cuda(device)
                mb = stage(mb)
            outputs.append(mb.cpu())
        
        # 合并微批次结果
        return torch.cat(outputs, dim=0)
```

### 4.3.3 数据并行推理

对于相同模型处理不同输入的场景，可以使用数据并行：

**多输入批量处理**

```python
def batch_inference(model, inputs):
    """将多个输入合并为批次处理"""
    # 确保所有输入形状一致
    batch = torch.stack(inputs)
    
    # 单次前向传播
    outputs = model(batch)
    
    # 拆分结果
    return outputs.unbind()
```

**Prefetch预取**

在GPU处理当前批次时，CPU同时准备下一批次数据：

```python
class PrefetchInference:
    def __init__(self, model, num_streams=2):
        self.model = model
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
    
    def step(self, current_batch, next_batch):
        stream = self.streams[self.current_stream]
        
        with torch.cuda.stream(stream):
            # GPU处理当前批次
            current_output = self.model(current_batch.cuda())
            
            # 准备下一批次数据（异步）
            next_batch_pinned = self._pin_memory(next_batch)
        
        # 切换stream
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        
        return current_output
```

## 4.4 内存优化

### 4.4.1 内存复用

**内存分配策略**

推理过程中的内存分配策略直接影响内存使用效率：

**Inplace操作**

Inplace操作直接在输入的内存位置写入输出，节省内存：

```python
# 普通实现
def relu(x):
    y = torch.clamp(x, min=0)  # 分配新内存
    return y

# Inplace实现
def relu_(x):
    x.clamp_(min=0)  # 直接修改x的内存
    return x
```

**内存池（Memory Pool）**

预先分配一块内存池，复用已分配的内存：

```python
class MemoryPool:
    def __init__(self, total_size, device='cuda'):
        self.pool = torch.empty(total_size, device=device)
        self.allocated = {}
        self.offset = 0
    
    def allocate(self, size):
        if self.offset + size > self.pool.numel():
            self.offset = 0  # 简单策略：环形复用
        offset = self.offset
        self.offset += size
        return self.pool[offset:offset+size]
    
    def reset(self):
        self.offset = 0
```

### 4.4.2 显存优化

**显存复用策略**

GPU显存的分配和释放有较高开销，应尽量复用已分配的显存：

```python
class GPUMemoryPool:
    def __init__(self):
        self.buffers = {}
    
    def get_buffer(self, name, size):
        if name not in self.buffers:
            self.buffers[name] = torch.empty(size, device='cuda')
        elif self.buffers[name].numel() < size:
            self.buffers[name] = torch.empty(size, device='cuda')
        return self.buffers[name]
    
    def release_all(self):
        self.buffers = {}
        torch.cuda.empty_cache()
```

**中间结果复用**

推理过程中的中间结果可以在使用完毕后复用：

```python
class InferenceMemory:
    def __init__(self):
        self.tensor_lifetime = {}
    
    def allocate(self, name, tensor):
        self.tensor_lifetime[name] = tensor
        return tensor
    
    def reuse_if_possible(self, name, size, dtype):
        if name in self.tensor_lifetime:
            t = self.tensor_lifetime[name]
            if t.numel() >= size:
                t = t[:size]
                return t
        return torch.empty(size, dtype=dtype, device='cuda')
```

### 4.4.3 内存优化技术

**梯度检查点（Gradient Checkpointing）**

通过重计算减少显存占用：

```python
# 前向传播时不保存所有中间结果
# 反向传播时重新计算需要的中间结果

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # 只保存少量检查点
        y = checkpoint(self.layer1, x)
        z = checkpoint(self.layer2, y)
        w = self.layer3(z)
        return w
```

**混合精度推理**

使用FP16替代FP32可以减少一半的显存占用：

```python
# 自动混合精度
with torch.cuda.amp.autocast():
    output = model(input)

# 手动FP16推理
model = model.half()  # 转换为FP16
input = input.half()
output = model(input)
```

**内存估算**

在实际部署前，估算模型推理的显存需求：

```python
def estimate_memory(model, input_shape, batch_size=1):
    """估算推理显存占用"""
    input_tensor = torch.randn(batch_size, *input_shape).cuda()
    
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return peak_memory
```

## 4.5 推理优化实践

### 4.5.1 延迟优化实践

**端到端延迟分析**

```python
import time
from contextlib import contextmanager

@contextmanager
def profile(name):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")

# 端到端推理延迟分析
with profile("Total"):
    with profile("Preprocess"):
        preprocessed = preprocess(raw_input)
    
    with profile("CUDA copy"):
        input_gpu = preprocessed.cuda()
    
    with profile("Inference"):
        output = model(input_gpu)
    
    with profile("Postprocess"):
        result = postprocess(output.cpu())
```

**针对性优化**

分析各阶段耗时，针对性优化：

1. **预处理耗时过多** → 使用向量化操作、SIMD
2. **CUDA拷贝耗时过多** → 使用Pinned Memory、异步拷贝
3. **推理耗时过多** → 优化模型结构、使用TensorRT
4. **后处理耗时过多** → 优化算法、使用CUDA加速

### 4.5.2 吞吐量优化实践

**Benchmark工具**

```python
import time
import threading

class ThroughputBenchmark:
    def __init__(self, model, batch_size=1):
        self.model = model
        self.batch_size = batch_size
    
    def run(self, num_requests, num_warmup=10):
        # 预热
        for _ in range(num_warmup):
            self._run_one_batch()
        
        torch.cuda.synchronize()
        
        # 开始计时
        start = time.perf_counter()
        for _ in range(num_requests):
            self._run_one_batch()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        qps = num_requests / elapsed
        print(f"QPS: {qps:.2f}, Total time: {elapsed:.2f}s")
        return qps
    
    def _run_one_batch(self):
        inputs = [torch.randn(3, 224, 224).cuda() for _ in range(self.batch_size)]
        batch = torch.stack(inputs)
        _ = self.model(batch)
```

### 4.5.3 综合优化策略

**分层优化方法**

1. **算法层优化**：选择更高效的模型结构（如MobileNet替代ResNet）
2. **框架层优化**：使用高效的推理引擎（如TensorRT、ONNX Runtime）
3. **算子层优化**：使用算子融合、布局转换等优化
4. **Kernel层优化**：使用高效的Kernel实现（如Winograd、Im2Col）

**典型优化配置**

```python
# TensorRT优化配置示例
import tensorrt as trt

builder = trt.Builder(logger)
network = builder.create_network()
config = builder.create_builder_config()

# 启用FP16
config.set_flag(trt.BuilderFlag.FP16)

# 设置显存上限
config.max_workspace_size = 1 << 30  # 1GB

# 启用CUBLAS CUBMM搜索
config.set_flag(trt.BuilderFlag.CUBLA_ALLOW_FP32_FP16_WILDCARD)
config.set_flag(trt.BuilderFlag.CUBLAS_ALLOW_WILDCARD)

# 构建引擎
engine = builder.build_serialized_network(network, config)
```

**性能监控与告警**

```python
class PerformanceMonitor:
    def __init__(self, sla_latency_ms=100):
        self.latencies = []
        self.sla_latency = sla_latency_ms
    
    def record(self, latency_ms):
        self.latencies.append(latency_ms)
        
        # 计算指标
        avg_latency = sum(self.latencies) / len(self.latencies)
        p99_latency = sorted(self.latencies)[int(len(self.latencies) * 0.99)]
        
        # 检查SLA
        if p99_latency > self.sla_latency:
            self._send_alert(f"P99延迟 {p99_latency:.2f}ms 超过SLA {self.sla_latency}ms")
        
        return {
            'avg_latency': avg_latency,
            'p99_latency': p99_latency,
            'requests': len(self.latencies)
        }
```

## 本章小结

1. **推理优化目标**：核心目标是降低延迟、提高吞吐量和资源效率。延迟和吞吐量往往需要权衡，需要根据业务场景选择优化方向。

2. **延迟优化**：主要技术包括动态批处理（平衡延迟和吞吐）、异步执行（计算与传输并行）、预热与缓存（消除冷启动开销）、流水线并行（各阶段并行处理）。

3. **吞吐量优化**：主要技术包括多实例部署+负载均衡、模型并行（张量并行/流水线并行，将大模型分布到多GPU）、数据并行推理（多输入合并批次处理）、Prefetch预取。

4. **内存优化**：主要技术包括Inplace操作（直接修改输入内存）、内存池（复用已分配内存）、显存复用策略、梯度检查点（重计算换内存）、混合精度推理（FP16替代FP32）。

5. **优化实践**：通过端到端延迟分析定位瓶颈，针对性优化。典型优化流程是：算法层→框架层→算子层→Kernel层逐层优化。

## 思考与练习

1. **概念理解**：解释为什么动态批处理可以同时优化延迟和吞吐量。如果只有静态批次（即固定批次大小），会有什么缺点？

2. **原理分析**：CUDA中的异步执行是如何"隐藏"内存访问延迟的？请分析异步Memcpy和异步Kernel执行的关系。

3. **设计思考**：假设你需要为一个实时视频分析服务设计推理优化方案，输入是视频流，每帧需要独立推理。请从延迟优化和吞吐量优化两个角度分析你会采用哪些技术？

4. **实践分析**：对比张量并行和流水线并行两种模型并行策略的优缺点。什么场景下适合使用张量并行？什么场景下适合使用流水线并行？

5. **内存优化**：解释梯度检查点（Checkpointing）是如何"用计算换内存"的。在什么场景下这种trade-off是值得的？

6. **综合应用**：假设你在部署一个目标检测模型到边缘设备（只有4GB显存），但模型+中间结果需要6GB显存。请设计一个综合使用多种优化技术的方案来满足这个显存约束。
