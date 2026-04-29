# 计算图

## 本章学习目标

1. 理解计算图的概念及其在AI框架中的核心地位
2. 掌握计算图的基本构成：节点（算子）与边（张量）
3. 理解静态计算图与动态计算图的区别与适用场景
4. 掌握计算图的调度与执行机制
5. 了解算子融合的概念与优化效果
6. 理解控制流在计算图中的表示方法

## 预备知识

- 数据结构（图的基本概念）
- 深度学习基础（神经网络前向传播与反向传播）
- Python编程基础

---

## 1 计算图基础

### 1.1 什么是计算图

**计算图（Computational Graph）** 是AI框架用来表达神经网络计算的核心数据结构。简单来说，计算图是一个**有向无环图（DAG, Directed Acyclic Graph）**，其中：

- **节点（Node）**：表示数据（标量、向量、矩阵或张量）或者运算操作
- **边（Edge）**：表示数据在节点之间的流动方向

想象一条流水线工厂：原材料从一端进入，经过一系列机器的加工处理，最终变成成品。计算图描述的就是这个加工流程，只不过这里的"原材料"是数据（张量），"机器"是数学运算。

以一个简单的函数为例：

$$z = (x + y) \times 2$$

其计算图可以表示为：

```
    x ──┐
        ├──► add ──► multiply ──► z
    y ──┘          ▲
                  │
                2 ─┘
```

在这个计算图中：
- $x$ 和 $y$ 是输入节点
- "add" 是加法运算节点
- "multiply" 是乘法运算节点
- 边表示数据的流向

### 1.2 为什么要用计算图

使用计算图有以下几个重要原因：

#### 1.2.1 统一表达各类计算

神经网络本质上是一系列数学运算的组合。计算图提供了一种统一的方式来描述这些运算，无论是：
- 简单的矩阵乘法：$Y = XW^T$
- 复杂的卷积操作：$Y = Conv(X, W)$
- 多分支的网络结构
- 带循环的RNN

都可以用计算图统一表达。

#### 1.2.2 支持自动微分

计算图是自动微分的基础。通过反向遍历计算图，结合链式法则，可以自动计算出每个参数的梯度。这正是深度学习框架（如PyTorch、TensorFlow）的核心功能。

#### 1.2.3 便于优化执行

有了计算图，AI框架可以在执行之前对计算过程进行分析和优化：
- **算子融合**：将多个小算子合并为一个，减少内存访问
- **内存优化**：分析数据依赖，复用中间结果
- **并行优化**：识别无依赖的算子，实现并行执行

#### 1.2.4 简化分布式执行

在大规模训练中，计算图可以方便地进行切分：
- 数据并行：不同设备处理不同数据批次
- 模型并行：不同设备负责不同层的计算
- 流水线并行：将不同层分配到不同设备

### 1.3 计算图的基本构成

#### 1.3.1 张量（Tensor）

**张量**是计算图中的基本数据单位。在数学上，张量是向量和矩阵的推广：

- **0阶张量（标量）**：一个单独的数，如 $3.14$
- **1阶张量（向量）**：一维数组，如 $[1, 2, 3]$
- **2阶张量（矩阵）**：二维数组，如 $\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}$
- **n阶张量**：n维数组

在深度学习中，一张图片可以表示为三维张量（高度、宽度、通道），一批图片表示为四维张量（批量大小、高度、宽度、通道）。

```python
# PyTorch中张量的创建
import torch

# 标量
scalar = torch.tensor(3.14)  # shape: []

# 向量
vector = torch.tensor([1.0, 2.0, 3.0])  # shape: [3]

# 矩阵
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: [2, 2]

# 批量图片（假设3张32x32的彩色图片）
batch = torch.randn(3, 32, 32, 3)  # shape: [N, H, W, C]
```

张量在计算图中的特点：
- 每个张量有自己的形状（shape）和数据类型（dtype）
- 张量可以保存在不同设备上（CPU、GPU、NPU）
- 张量可以是叶子节点（参数）或中间节点（计算结果）

#### 1.3.2 算子（Operator）

**算子**是计算图中的操作节点，表示具体的数学运算。

常见的算子分类：

| 类别 | 算子示例 | 说明 |
|------|----------|------|
| 逐元素运算 | Add, Sub, Mul, Div | 元素对应位置运算 |
| 矩阵运算 | MatMul, Conv | 矩阵/卷积乘法 |
| 激活函数 | ReLU, Sigmoid, Tanh | 非线性变换 |
| 归约运算 | Sum, Mean, Max | 汇总操作 |
| 形状操作 | Reshape, Transpose | 改变张量形状 |
| 索引操作 | Index, Slice | 取张量的一部分 |

```python
# PyTorch中的算子
import torch

x = torch.randn(2, 3)
y = torch.randn(2, 3)

# 逐元素运算
z1 = x + y  # Add
z2 = x * y  # Mul
z3 = torch.relu(x)  # ReLU

# 矩阵运算
w = torch.randn(3, 4)
z4 = torch.matmul(x, w)  # Matrix Multiply

# 归约运算
z5 = x.sum()  # Sum
z6 = x.mean(dim=1)  # Mean along dim=1
```

### 1.4 计算图示例

让我们通过一个具体的神经网络前向传播来理解计算图。

假设有一个简单的两层全连接网络：

```python
import torch

# 定义网络
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)      # Linear1
        x = torch.relu(x)     # ReLU
        x = self.fc2(x)      # Linear2
        return x

# 创建网络和输入
model = SimpleNet()
x = torch.randn(32, 784)  # 批量大小32，784维输入

# 前向传播
output = model(x)
```

对应的计算图大致如下：

```
输入x ──► Linear1 ──► ReLU ──► Linear2 ──► 输出
              ▲                        │
              │                        │
           权重W1                    权重W2
```

### 1.5 计算图与自动微分的关系

计算图和自动微分是密不可分的。在神经网络的训练中：

1. **前向传播**：沿着计算图从输入到输出，计算每个节点的值
2. **反向传播**：从输出开始，沿着计算图反向遍历，计算每个参数对损失的梯度

```python
import torch

# 创建计算图
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([5.0], requires_grad=True)

# 前向传播
z = x * y + torch.log(x)

# 反向传播
z.backward()

print(x.grad)  # dy/dx = y + 1/x = 5 + 0.5 = 5.5
print(y.grad)  # dy/dy = x = 2
```

在这个例子中：
- 前向传播构建了计算图：$z = x \times y + \log(x)$
- 反向传播时，PyTorch沿着计算图反向计算梯度

---

## 2 静态计算图与动态计算图

AI框架根据计算图的构建方式，分为两大阵营：**静态计算图**和**动态计算图**。

### 2.1 静态计算图

**静态计算图（Static Computational Graph）** 的特点是：计算图的结构在执行之前就已经确定，执行过程中不会改变。

#### 2.1.1 工作流程

```python
# TensorFlow 1.x 风格的伪代码
# 1. 定义计算图
x = placeholder("x", shape=[None, 784])
y = placeholder("y", shape=[None, 10])

W1 = variable(tf.random_normal([784, 256]))
b1 = variable(tf.zeros([256]))
h1 = relu(matmul(x, W1) + b1)

W2 = variable(tf.random_normal([256, 10]))
b2 = variable(tf.zeros([10]))
logits = matmul(h1, W2) + b2

# 2. 定义损失和优化器
loss = softmax_cross_entropy(logits, y)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = optimizer.minimize(loss)

# 3. 执行（Session）
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
```

#### 2.1.2 静态图的优势

1. **执行效率高**：图结构已知，可以进行全局优化
2. **内存效率好**：可以预分配内存，复用中间结果
3. **易于部署**：图可以序列化，部署到各种环境
4. **编译器友好**：便于做图优化、算子融合等

#### 2.1.3 静态图的劣势

1. **调试困难**：不直观，难以查看中间结果
2. **灵活性低**：控制流表达不如宿主语言自然
3. **学习曲线陡**：需要理解placeholder、Session等概念

### 2.2 动态计算图

**动态计算图（Dynamic Computational Graph）** 的特点是：计算图在执行过程中动态构建，每次执行都可能不同。

#### 2.2.1 工作流程

```python
# PyTorch 风格
import torch

# 定义网络（动态构建计算图）
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# 执行（前向传播时动态构建计算图）
x = torch.randn(32, 784)
output = model(x)  # 每次调用都构建新的计算图

# 反向传播
loss = output.sum()
loss.backward()
```

#### 2.2.2 动态图的优势

1. **调试直观**：可以直接print、debug每个变量
2. **灵活性高**：支持Python原生控制流（if、for、while）
3. **易学易用**：跟写普通Python代码一样
4. **模型结构可变**：网络结构可以在运行时改变

#### 2.2.3 动态图的劣势

1. **执行效率略低**：图结构每次都需要重新构建
2. **内存开销大**：无法预分配内存，中间结果需要保存
3. **优化受限**：难以进行全局优化

### 2.3 两者的对比

| 特性 | 静态计算图 | 动态计算图 |
|------|------------|------------|
| 图构建时机 | 执行前 | 执行中 |
| 灵活性 | 低 | 高 |
| 调试 | 困难 | 直观 |
| 执行效率 | 高 | 中等 |
| 内存效率 | 高 | 中等 |
| 部署 | 方便 | 较难 |
| 代表框架 | TensorFlow 1.x, JAX | PyTorch, Chainer |

### 2.4 现代框架的动静融合

现代AI框架正在走向动静融合的道路：

#### 2.4.1 TensorFlow 2.x 的改变

TensorFlow 2.x默认启用eager execution（动态图），同时保留静态图（通过tf.function）：

```python
import tensorflow as tf

# 默认是动态图
x = tf.constant([[1.0, 2.0]])
y = tf.constant([[3.0], [4.0]])
result = tf.matmul(x, y)  # 立即执行

# 通过@tf.function转换为静态图
@tf.function
def model(x):
    return tf.matmul(x, W) + b
```

#### 2.4.2 PyTorch 2.x 的torch.compile

PyTorch 2.0引入了torch.compile，将动态图JIT编译为静态图执行：

```python
import torch

model = MyModel().cuda()
# 编译模型，生成优化后的静态图
model = torch.compile(model)

# 执行（更快）
output = model(x)
```

#### 2.4.3 JAX的函数式设计

JAX采用纯函数式设计，默认静态，但通过lax.scan等实现动态控制流：

```python
import jax
import jax.numpy as jnp

# JIT编译（静态化）
@jax.jit
def jitted_func(x):
    return x @ x.T

# 纯函数，无副作用
def forward(params, x):
    return x @ params['W'] + params['b']
```

---

## 3 计算图的调度与执行

计算图构建完成后，需要被调度到硬件上执行。本节介绍计算图的调度机制。

### 3.1 算子调度基础

#### 3.1.1 调度顺序

计算图的调度需要遵守**拓扑排序**：一个算子只有在所有输入都准备好后才能执行。

以计算图为例：

```
    A ──► B ──► D
          │
    C ────┘
```

合法的调度顺序：
- A, B, D（当C先于B完成时）
- A, C, B, D
- C, A, B, D

#### 3.1.2 简单调度实现

```python
from collections import deque

class SimpleScheduler:
    def __init__(self, graph):
        """
        graph: dict, {node_name: Node}
              Node: {name, inputs: [node_names], op: callable}
        """
        self.graph = graph
        self.in_degree = {}
        self.build_in_degree()

    def build_in_degree(self):
        """计算每个节点的入度"""
        for node_name in self.graph:
            if node_name not in self.in_degree:
                self.in_degree[node_name] = 0
            for inp in self.graph[node_name].get('inputs', []):
                if inp not in self.in_degree:
                    self.in_degree[inp] = 0
                self.in_degree[node_name] += 1

    def topological_sort(self):
        """Kahn算法拓扑排序"""
        queue = deque([n for n, d in self.in_degree.items() if d == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in self.graph[node].get('outputs', []):
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def schedule(self):
        """返回调度顺序"""
        return self.topological_sort()
```

### 3.2 单设备调度

在单个设备（CPU或GPU）上，调度相对简单：

```python
class SingleDeviceScheduler:
    def __init__(self, graph):
        self.graph = graph
        self.executor = ThreadPoolExecutor(max_workers=1)

    def execute(self, inputs):
        """按拓扑顺序执行"""
        values = {}

        # 拓扑排序
        order = self.topological_sort()

        for node_name in order:
            node = self.graph[node_name]
            # 获取输入值
            input_values = [values[inp] for inp in node['inputs']]
            # 执行算子
            output = node['op'](*input_values)
            values[node_name] = output

        return values
```

### 3.3 并发调度

当算子之间没有数据依赖时，可以并发执行：

```python
class ConcurrentScheduler:
    def __init__(self, graph):
        self.graph = graph
        self.executor = ThreadPoolExecutor(max_workers=4)

    def execute(self, inputs):
        values = inputs.copy()
        pending = set(self.graph.keys())
        running = []

        while pending or running:
            # 启动所有可以启动的算子
            for node_name in list(pending):
                node = self.graph[node_name]
                if all(inp in values for inp in node['inputs']):
                    input_values = [values[inp] for inp in node['inputs']]
                    future = self.executor.submit(node['op'], *input_values)
                    running.append((node_name, future))
                    pending.remove(node_name)

            # 等待至少一个完成
            if running:
                done, running = self.executor.wait(
                    running, return_when=FIRST_COMPLETED
                )
                for node_name, future in done:
                    values[node_name] = future.result()

        return values
```

### 3.4 异构调度

在手机/边缘设备等异构环境中，计算图包含在不同硬件上执行的算子：

```python
class HeterogeneousScheduler:
    def __init__(self, graph, device_assignment):
        """
        device_assignment: dict, {node_name: 'cpu'|'gpu'|'npu'}
        """
        self.graph = graph
        self.device_assignment = device_assignment
        self.devices = {
            'cpu': CPUDevice(),
            'gpu': GPUDevice(),
            'npu': NPUDevice()
        }

    def execute(self, inputs):
        values = {}

        # 按设备分组，同设备内按拓扑序
        device_tasks = {}
        for node_name, device in self.device_assignment.items():
            if device not in device_tasks:
                device_tasks[device] = []
            device_tasks[device].append(node_name)

        # 在各设备上并行执行
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = {}
            for device, tasks in device_tasks.items():
                device_obj = self.devices[device]
                future = executor.submit(self.execute_on_device,
                                       device_obj, tasks, values)
                futures[future] = device

            # 收集结果
            for future in futures:
                future.result()

        return values

    def execute_on_device(self, device, tasks, values):
        for node_name in self.topological_sort(tasks):
            node = self.graph[node_name]
            input_values = [values[inp] for inp in node['inputs']]
            # 将输入传输到设备
            device_inputs = device.to_device(input_values)
            # 在设备上执行
            output = device.execute(node['op'], device_inputs)
            # 传回主机
            values[node_name] = device.to_host(output)
```

### 3.5 图执行模式

#### 3.5.1 逐算子下发

最简单的方式，每个算子依次下发执行：

```python
# PyTorch默认方式（eager mode）
for node in topological_order:
    op = node.op
    inputs = get_inputs(node)
    output = op(*inputs)  # 立即执行
```

优点：灵活、易调试
缺点：每次都需要调度开销

#### 3.5.2 整图下沉

将整个图一次性下发到设备执行：

```python
# TensorFlow静态图方式
@tf.function
def compiled_model(x):
    return model(x)

output = compiled_model(x)  # 整图下发
```

优点：减少调度开销，可做全局优化
缺点：灵活性降低

#### 3.5.3 子图下沉

将计算图分成若干子图，较大较重的子图下沉到加速器执行：

```python
# 子图下沉示例
# CPU执行控制流，子图下沉到NPU

for node in graph:
    if is_heavy_compute(node):
        # 下沉到NPU
        npu.execute_subgraph(node.subgraph)
    else:
        # CPU继续执行
        cpu.execute(node)
```

---

## 4 图优化技术

AI框架在执行计算图之前或过程中，会进行多种优化。本节介绍主要的图优化技术。

### 4.1 算子融合

**算子融合（Operator Fusion）** 将多个连续的算子合并为一个，减少内存访问和内核启动开销。

#### 4.1.1 为什么不融合

考虑以下计算：

```python
# 两步操作
h = x @ W1 + b1  # Matrix multiply + bias
h = torch.relu(h)  # ReLU activation
```

如果没有融合，需要：
1. 分配临时内存存储 $h$
2. 执行矩阵乘法内核
3. 执行ReLU内核（需要读取 $h$，写入新内存）

#### 4.1.2 融合后的优势

融合后：

```python
# 融合操作
h = relu_with_bias(x, W1, b1)  # 单个内核
```

只需要：
1. 分配输出内存
2. 执行一个融合内核（直接在输出内存计算）

**融合操作的优点**：
- 减少内存访问：无需读取/写入临时张量
- 减少内核启动开销：一个内核替代两个
- 提高数据局部性：更好地利用寄存器

#### 4.1.3 常见融合模式

| 融合模式 | 说明 | 示例 |
|----------|------|------|
| BiasAdd + Act | 融合偏置和激活 | (x @ W + b) + ReLU |
| Conv + BN | 融合卷积和批归一化 | Conv + BatchNorm |
| MatMul + Add | 融合矩阵乘和加法 | x @ W + b |
| Elementwise融合 | 融合多个逐元素操作 | (a + b) * (c - d) |

```python
# PyTorch的算子融合示例
# 安装torch.compile后自动融合
model = torch.compile(model, backend="inductor")

# 或者手动使用torch.jit.script
@torch.jit.script
def fused_relu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    return torch.relu(torch.addmm(b, x, w))
```

### 4.2 代数简化

利用代数规则简化计算图：

#### 4.2.1 常量折叠

```python
# 输入: (x + 2) * 3 + x * 0
# 第一步：常量运算
x + 6 + 0
# 结果: x + 6
```

#### 4.2.2 恒等变换

```python
# x * 1 = x
# x / 1 = x
# x + 0 = x
# x - 0 = x
```

#### 4.2.3 公共子表达式消除

```python
# 原始: y = (x + 1) * (x + 1)
# 优化: t = x + 1; y = t * t
```

### 4.3 内存优化

#### 4.3.1 内存共享

```python
# 原始：需要额外的输出张量
y = x + 1
z = y * 2

# 优化：如果y不再使用，可以复用x的内存
tmp = x + 1
tmp = tmp * 2  # 直接在tmp上操作
```

#### 4.3.2 内存池

预先分配一大块内存，按需分配给各个张量：

```python
class MemoryPool:
    def __init__(self, size):
        self.pool = bytearray(size)
        self.free_list = [(0, size)]

    def allocate(self, size):
        """分配size字节，返回起始地址"""
        for i, (start, end) in enumerate(self.free_list):
            if end - start >= size:
                self.free_list.pop(i)
                return start
        raise OutOfMemoryError()

    def free(self, start, size):
        """释放内存"""
        self.free_list.append((start, start + size))
        self.free_list.sort()
```

#### 4.3.3 激活重计算

通过重计算而不是存储来节省激活内存：

```python
# 原始：存储所有激活用于反向
def forward_with_storage(model, x):
    activations = []
    for layer in model.layers:
        x = layer(x)
        activations.append(x)  # 存储所有激活
    return x, activations

# 优化：只存储检查点
def forward_with_checkpoint(model, x, checkpoint_every=3):
    checkpoints = [x]
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if (i + 1) % checkpoint_every == 0:
            checkpoints.append(x)
    return x, checkpoints

def backward_with_recompute(model, checkpoints, grad_output):
    # 从后往前，只保留部分激活
    pass
```

### 4.4 布局优化

张量数据在内存中的排列方式（布局）会影响计算效率。

#### 4.4.1 内存布局

常见的内存布局：
- **NCHW**：批量、高、宽、通道（NVIDIA GPU常用）
- **NHWC**：批量、高、宽、通道（TensorFlow默认）
- **CHW**：通道、高、宽
- **HWC**：高、宽、通道

#### 4.4.2 布局转换

```python
# 假设输入是NHWC，但GPU需要NCHW
x = torch.randn(N, H, W, C)  # NHWC
x = x.permute(0, 3, 1, 2)    # 转换为NCHW
# 执行卷积
y = torch.nn.functional.conv2d(x, weight)
# 转换回NHWC
y = y.permute(0, 2, 3, 1)
```

布局转换有开销，图优化可以：
- 消除不必要的布局转换
- 融合布局转换和计算

---

## 5 算子融合详解

算子融合是现代AI编译器（如TVM、TensorRT）的核心技术。本节深入介绍算子融合的原理和实现。

### 5.1 融合的数学基础

考虑两个逐元素操作：

$$y = f(g(x))$$

融合后：

$$h(x) = (f \circ g)(x)$$

融合的优势：
- 内存：只需存储一个中间结果（而非两个）
- 带宽：只读取一次 $x$，只写一次 $y$
- 指令效率：更好的数据局部性

### 5.2 常见融合模式

#### 5.2.1 ReLU融合

```python
# 融合前
y1 = x @ W + b
y2 = relu(y1)
# 融合后
y = relu_with_bias(x, W, b)  # 单个kernel

# 对于Conv + BN + ReLU常见于ResNet
# 融合为单个Conv+BN+ReLU kernel
```

#### 5.2.2 Pointwise融合

```python
# 多个pointwise操作可以融合
y = (x + 1) * (x - 2) + 3
# 融合为单个elementwise kernel
# 避免创建中间张量
```

#### 5.2.3 Reduction融合

```python
# Sum + Square + Mean 融合
loss = ((y_pred - y_true) ** 2).mean()
# 单个kernel完成所有计算
```

### 5.3 融合的实现

#### 5.3.1 图层面的融合

```python
def find_fusible_patterns(graph):
    """在计算图中搜索可融合的模式"""
    patterns = []

    for node in graph.nodes:
        # 搜索 (MatMul + BiasAdd + Activation) 模式
        if is_matmul(node) and has_bias(node) and has_activation(node):
            patterns.append(FusionPattern(node))

    return patterns

def fuse_patterns(graph, patterns):
    """执行融合"""
    for pattern in patterns:
        # 创建融合节点
        fused_node = create_fused_node(pattern)
        # 替换原节点
        replace_nodes(graph, pattern.nodes, fused_node)
```

#### 5.3.2 代码生成

融合后的kernel需要代码生成：

```python
# TVM风格的代码生成
def gen_fused_kernel(nodes):
    code = """
    void fused_kernel(const float* x, const float* w, float* y) {
        for (int i = 0; i < N; i++) {
            // Matmul + ReLU fused
            float sum = 0;
            for (int j = 0; j < K; j++) {
                sum += x[i*K+j] * w[j];
            }
            y[i] = sum > 0 ? sum : 0;  // ReLU inline
        }
    }
    """
    return code
```

### 5.4 融合的边界

并非所有算子都可以或应该融合：

1. **内存限制**：融合后的kernel可能占用过多寄存器
2. **代码大小**：过大的融合kernel增加编译时间
3. **并行度**：某些融合可能降低并行度
4. **数值稳定性**：融合可能影响数值精度

---

## 6 控制流在计算图中的表示

神经网络中经常需要控制流（如条件分支、循环）。本节介绍如何在计算图中表示控制流。

### 6.1 为什么需要控制流

很多神经网络结构包含控制流：

```python
# RNN中的循环
def rnn_cell(x, h):
    new_h = tanh(W_xh @ x + W_hh @ h + b)
    return new_h

# 展开RNN（需要循环）
for t in range(seq_len):
    h = rnn_cell(x[:, t], h)

# 注意力中的条件分支
if attention_score > threshold:
    attended = attend(query, keys, values)
else:
    attended = query
```

### 6.2 动态图对控制流的处理

动态图直接使用宿主语言（Python）的控制流，最自然：

```python
# PyTorch中RNN实现
class RNN(nn.Module):
    def forward(self, x):
        h = torch.zeros(...)
        for t in range(x.size(0)):
            h = self.rnn_cell(x[t], h)
        return h
```

每次执行时，Python解释器自然地处理循环，图是动态构建的。

### 6.3 静态图对控制流的处理

静态图需要将控制流表达为图中的节点。

#### 6.3.1 条件分支

TensorFlow使用Switch和Merge算子：

```python
# TensorFlow条件分支伪代码
# graph:
#   switch = tf.switch(cond, data)
#   branch_true = tf.identity(switch[0])  # 条件为真时的数据
#   branch_false = tf.identity(switch[1])  # 条件为假时的数据
#   result = tf.merge([branch_true, branch_false])
```

#### 6.3.2 循环

TensorFlow使用WhileLoop算子：

```python
# TensorFlow循环
# graph:
#   loop_vars = [iteration, accumulated_result]
#   cond = lambda i, acc: i < max_iter
#   body = lambda i, acc: [i+1, acc + f(i)]
#   result = tf.while_loop(cond, body, loop_vars)

i = tf.constant(0)
result = tf.constant(0)

def condition(i, result):
    return i < 10

def body(i, result):
    return i + 1, result + i

i, result = tf.while_loop(condition, body, [i, result])
```

### 6.4 控制流图与计算图的区别

**控制流图（Control Flow Graph, CFG）** 和**计算图（Computational Graph）** 是不同的概念：

| 特性 | 控制流图 | 计算图 |
|------|----------|--------|
| 节点 | 程序基本块 | 数据/运算 |
| 边 | 执行顺序 | 数据依赖 |
| 表示 | 路径选择 | 数据变换 |
| 用途 | 编译器优化 | AI框架计算 |

### 6.5 现代框架的处理方式

#### 6.5.1 PyTorch的方案

PyTorch主要依赖动态图，控制流直接用Python表达：

```python
# 动态图天然支持Python控制流
class DynamicRNN(nn.Module):
    def forward(self, x, mask=None):
        outputs = []
        for t in range(x.size(0)):
            if mask is not None and mask[t] == 0:
                continue
            output = self.cell(x[t])
            outputs.append(output)
        return torch.stack(outputs)
```

PyTorch 2.0的torch.compile会处理动态图中的控制流，但可能退化为逐算子执行。

#### 6.5.2 TensorFlow/JAX的方案

这些框架将控制流表示为图中的特殊节点：

```python
# JAX的循环（静态化）
import jax.lax as lax

def body_fn(carry, x):
    return carry + x, carry * x

# scan会在编译时展开循环
final, ys = lax.scan(body_fn, 0.0, xs)
```

### 6.6 控制流与自动微分

当控制流依赖于输入数据时，自动微分会变得复杂：

```python
# 依赖于输入的条件分支
if x.sum() > 0:
    y = relu(x)
else:
    y = sigmoid(x)

# 微分时，需要知道运行走了哪个分支
# dy/dx 取决于运行时条件
```

解决方案：
1. **分支都计算**：两个分支都计算，根据条件加权
2. **重新计算**：不需要的分支结果丢弃（重计算策略）
3. **Steensgaard方法**：假设分支不相关，忽略依赖

---

## 7 计算图与PyTorch实现

本节深入介绍PyTorch中计算图的实际工作方式。

### 7.1 PyTorch计算图的构建

#### 7.1.1 叶子节点与中间节点

```python
import torch

# 叶子节点：用户创建的，requires_grad=True
a = torch.tensor([1.0], requires_grad=True)
print(f"a.is_leaf: {a.is_leaf}")  # True

# 中间节点：操作产生的
b = a * 2
c = b + a
print(f"b.is_leaf: {b.is_leaf}")  # False
print(f"c.is_leaf: {c.is_leaf}")  # False
```

#### 7.1.2 grad_fn

每个非叶子节点都有一个`grad_fn`，指示如何计算梯度：

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y + 3

print(f"x.grad_fn: {x.grad_fn}")  # None (叶子节点)
print(f"y.grad_fn: {y.grad_fn}")  # <MulBackward0>
print(f"z.grad_fn: {z.grad_fn}")  # <AddBackward0>
```

### 7.2 反向传播与计算图

当调用`backward()`时，PyTorch沿计算图反向传播梯度：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3

# 反向传播
# dz/dz = 1 (初始化)
# dz/dy = 3 (乘法的反向)
# dz/dx = dz/dy * dy/dx = 3 * 2x = 3 * 4 = 12
z.backward()
print(x.grad)  # tensor([12.])
```

### 7.3 计算图释放

默认情况下，backward()后计算图会被释放：

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y ** 2

z.backward()  # 图被释放

# 再次backward会报错
try:
    y.backward()  # RuntimeError: grad can be implicitly created only for scalar outputs
except RuntimeError as e:
    print(f"Error: {e}")

# 如果需要保留图
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y ** 2

z.backward(retain_graph=True)  # 保留图
print(x.grad)  # tensor([32.])

# 梯度会累加
z.backward()
print(x.grad)  # tensor([64.])  # 累加了
```

### 7.4 torch.jit.script与静态图

`torch.jit.script`将Python代码转换为静态图表示：

```python
import torch

@torch.jit.script
def scripted_function(x: torch.Tensor) -> torch.Tensor:
    # 编译为静态图
    if x.sum() > 0:
        return torch.relu(x)
    else:
        return torch.sigmoid(x)

# 第一次调用会触发编译
result = scripted_function(torch.randn(3))

# 之后调用会使用优化后的静态图
result = scripted_function(torch.randn(3))
```

### 7.5 torch.compile (PyTorch 2.0)

PyTorch 2.0的`torch.compile`是最强大的图优化工具：

```python
import torch

model = MyModel().cuda()

# 编译模型
# modes: "default", "reduce-overhead", "max-autotune"
compiled_model = torch.compile(model, mode="default")

# 执行（会快很多）
output = compiled_model(x)
```

编译过程：
1. **跟踪（Tracing）**：执行前向传播，记录算子序列
2. **优化（Optimization）**：应用图优化（融合、常量折叠等）
3. **编译（Compilation）**：生成优化的内核

---

## 8 计算图在实际应用中的示例

### 8.1 ResNet中的计算图

ResNet是经典的卷积神经网络，其计算图包含：

```python
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual  # 残差连接
        out = torch.relu(out)

        return out
```

其计算图大致如下：

```
输入x ──┬─► Conv1 ──► BN1 ──► ReLU ──► Conv2 ──► BN2 ──┬─► Add ──► ReLU ──► 输出
        │                                                  ▲
        └────────────── Shortcut (Identity/Conv) ─────────┘
```

### 8.2 Transformer中的计算图

Transformer使用注意力机制，计算图更复杂：

```python
class TransformerLayer(nn.Module):
    def forward(self, x):
        # Self-Attention
        attn_output = self.self_attn(q=x, k=x, v=x)
        x = x + attn_output  # 残差
        x = self.norm1(x)

        # Feed-Forward
        ff_output = self.ffn(x)
        x = x + ff_output  # 残差
        x = self.norm2(x)

        return x
```

注意力计算本身是一个复杂的计算图：

```
Q, K, V ──► MatMul ──► Softmax ──► MatMul ──► 输出
     │                         ▲
     └───── Transpose ─────────┘
```

### 8.3 分布式训练中的计算图

在数据并行中，每个设备有相同的计算图结构，但处理不同数据：

```
设备0: x0 ──► Forward ──► Loss0 ──► Backward ──► grad_W0
设备1: x1 ──► Forward ──► Loss1 ──► Backward ──► grad_W1
设备2: x2 ──► Forward ──► Loss2 ──► Backward ──► grad_W2
设备3: x3 ──► Forward ──► Loss3 ──► Backward ──► grad_W3
                           │
                           ▼
                    AllReduce(grad_Wi) ──► 聚合梯度 ──► 更新W
```

---

## 本章小结

本章系统介绍了计算图的核心概念及其在AI框架中的实现。

**核心要点回顾**：

1. **计算图是AI框架的核心数据结构**：将神经网络表示为有向无环图，节点是张量和算子，边表示数据流动。

2. **静态图vs动态图**：
   - 静态图在执行前构建，效率高但不灵活
   - 动态图在执行中构建，灵活但有开销
   - 现代框架趋向动静融合

3. **计算图调度**：按拓扑序调度算子执行，支持并发和异构设备调度。

4. **图优化技术**：
   - 算子融合减少内存访问和内核启动开销
   - 代数简化消除冗余计算
   - 内存优化提高利用率

5. **控制流表示**：动态图直接用Python控制流，静态图需要Switch/WhileLoop等特殊算子。

6. **PyTorch实现**：基于动态计算图，通过autograd实现自动微分，支持torch.jit.script和torch.compile进行静态优化。

**思考与练习**：

1. 绘制函数 $z = \max(x^2, y^2) + \min(x, y)$ 的计算图。
2. 静态计算图和动态计算图各适合什么场景？
3. 为什么算子融合能提高效率？哪些算子适合融合？
4. 在PyTorch中，如果需要多次反向传播但保留计算图，应该如何做？
5. 如果要在计算图中表示一个循环神经网络的前向传播，应该怎么处理？
6. 比较TensorFlow的静态图和PyTorch的动态图，它们各自的优缺点是什么？
7. 什么是计算图的拓扑排序？为什么调度需要遵守拓扑序？

---

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=346347233&bvid=BV1rR4y197HM&cid=911588326&page=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

<!--Copyright © ZOMI 适用于[License](https://github.com/chenzomi12/AISystem)版权许可-->
