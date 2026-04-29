# AppendixC 实践项目

本附录提供了一系列实践项目，涵盖神经网络实现、模型量化、分布式训练、AI编译器、推理系统和大模型训练等多个核心领域。每个项目都包含详细的目标、环境要求、实现步骤和参考代码，旨在帮助读者将理论知识转化为实践能力。

---

## 项目一：神经网络基础实践

**难度**：⭐⭐（初级）

**学习目标**
- 理解神经网络的基本原理和结构
- 掌握前向传播和反向传播算法
- 学会使用NumPy从零实现深度学习
- 理解梯度下降和参数更新的机制

**环境要求**
- Python 3.8+
- NumPy
- Matplotlib
- MNIST数据集（可在线下载）

**实现步骤**

**第一步：数据加载与预处理**

下载MNIST数据集，将28x28的灰度图片展开为784维向量，并进行归一化处理。将数字标签转换为one-hot编码格式。将数据划分为训练集（60000样本）和测试集（10000样本）。

**第二步：网络参数初始化**

实现Xavier初始化方法初始化权重矩阵，初始化包括各层的权重矩阵和偏置向量。权重维度应与相邻两层神经元数量相匹配。

**第三步：实现激活函数与损失函数**

实现ReLU激活函数：f(x) = max(0, x)，及其导数。实现Softmax函数，将输出层的logits转换为概率分布，注意数值稳定性处理。实现交叉熵损失函数。

**第四步：实现前向传播**

按照网络结构依次计算每层的线性变换和激活函数输出。线性变换：z = Wx + b，激活函数对线性变换结果应用ReLU，输出层应用Softmax函数。保存各层中间值用于反向传播计算。

**第五步：实现反向传播**

计算输出层误差，根据链式法则逐层反向传播误差，计算每层参数的梯度。权重梯度：∂L/∂W = a^T * δ，偏置梯度：∂L/∂b = δ。

**第六步：实现参数更新**

使用计算得到的梯度更新网络参数。标准SGD更新：W = W - η * ∂L/∂W，其中η是学习率。

**第七步：模型评估与测试**

在测试集上评估模型性能，计算分类准确率。实现预测函数对单个样本或批量样本进行预测。

**参考代码**

```python
import numpy as np

class Linear:
    """线性层：y = Wx + b"""
    def __init__(self, input_dim, output_dim):
        # Xavier初始化
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.grad_W = self.x.T @ grad
        self.grad_b = np.sum(grad, axis=0)
        return grad @ self.W.T

    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

class ReLU:
    """ReLU激活函数"""
    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * self.mask

class SoftmaxCrossEntropy:
    """Softmax激活函数与交叉熵损失的组合"""
    def forward(self, x, y):
        # 数值稳定的softmax
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        # 计算损失
        n = y.shape[0]
        self.y = y
        self.probs = np.clip(self.probs, 1e-10, 1 - 1e-10)
        loss = -np.sum(y * np.log(self.probs)) / n
        return loss

    def backward(self):
        n = self.y.shape[0]
        return (self.probs - self.y) / n

class MLP:
    """多层感知器"""
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, y):
        for layer in self.layers:
            if isinstance(layer, SoftmaxCrossEntropy):
                loss = layer.forward(x, y)
            else:
                x = layer.forward(x)
        return loss

    def backward(self):
        grad = 1
        for layer in reversed(self.layers):
            if isinstance(layer, SoftmaxCrossEntropy):
                grad = layer.backward()
            else:
                grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(lr)

    def predict(self, x):
        for layer in self.layers:
            if isinstance(layer, SoftmaxCrossEntropy):
                continue
            x = layer.forward(x)
        return np.argmax(x, axis=1)
```

**训练流程示例**

```python
# 构建网络：784 -> 256 -> 128 -> 10
model = MLP([
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
    SoftmaxCrossEntropy()
])

# 训练配置
lr = 0.01
batch_size = 64
epochs = 10

# 训练循环
for epoch in range(epochs):
    # 打乱数据
    indices = np.random.permutation(len(X_train))
    X_shuffle = X_train[indices]
    y_shuffle = y_train[indices]

    total_loss = 0
    num_batches = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffle[i:i+batch_size]
        y_batch = y_shuffle[i:i+batch_size]

        # 前向传播
        loss = model.forward(X_batch, y_batch)
        # 反向传播
        model.backward()
        # 更新参数
        model.update(lr)

        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = np.mean(model.predict(X_test) == np.argmax(y_test, axis=1))
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
```

**预期结果**
- 训练准确率达到90%以上
- 测试准确率达到85%以上
- 可视化loss曲线显示收敛趋势

---

## 项目二：模型量化与部署

**难度**：⭐⭐⭐（中级）

**学习目标**
- 理解模型量化的基本原理
- 掌握PyTorch动态量化和静态量化的使用方法
- 学会将模型导出为ONNX格式
- 使用ONNX Runtime进行推理部署

**环境要求**
- Python 3.8+
- PyTorch 1.10+
- ONNX
- ONNX Runtime
- torchvision
- NVIDIA GPU（可选，用于加速推理）

**实现步骤**

**第一步：环境准备与模型加载**

安装PyTorch、ONNX、ONNX Runtime等依赖。加载预训练的ResNet18模型，设置为评估模式。记录原始模型的参数量和大小。

```python
import torch
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 获取模型信息
num_params = sum(p.numel() for p in model.parameters())
model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
print(f"原始模型参数量: {num_params:,}")
print(f"原始模型大小: {model_size:.2f} MB")
```

**第二步：实现动态量化**

动态量化主要对权重进行量化，激活值在推理时动态量化。PyTorch的动态量化非常简单，只需一行代码即可完成。

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
print(f"量化后模型大小: {get_model_size(quantized_model):.2f} MB")
```

**第三步：实现静态量化**

静态量化需要额外进行校准步骤。首先为模型插入量化标记，然后准备校准数据集，运行推理并收集激活值统计信息。

```python
# 静态量化配置
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 校准（使用少量数据）
with torch.no_grad():
    for images, _ in calibrate_loader:
        model(images)

# 转换
quantized_model = torch.quantization.convert(model, inplace=True)
```

**第四步：转换为ONNX格式**

使用torch.onnx.export函数导出模型，指定输入张量的形状和类型，处理动态轴以便支持变长输入。

```python
# 导出ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    quantized_model, dummy_input, "resnet18_quantized.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

**第五步：使用ONNX Runtime推理**

创建推理会话，实现图像预处理和后处理流程，运行推理并处理输出结果。

```python
import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession("resnet18_quantized.onnx")

# 准备输入数据
input_data = preprocess_image(image_path)  # 图像预处理

# 运行推理
outputs = session.run(None, {'input': input_data})
predictions = postprocess_output(outputs)
```

**第六步：性能对比分析**

测量不同模型的推理延迟和吞吐量，评估量化对模型精度的影响。

```python
def measure_latency(model, input_data, num_iterations=100):
    """测量推理延迟（毫秒）"""
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_data)
            latencies.append(time.time() - start)
    return np.mean(latencies) * 1000

def measure_throughput(model, input_data, num_iterations=100):
    """测量吞吐量（samples/sec）"""
    model.eval()
    batch_size = input_data.shape[0]
    with torch.no_grad():
        start = time.time()
        for _ in range(num_iterations):
            _ = model(input_data)
        elapsed = time.time() - start
    return num_iterations * batch_size / elapsed
```

**参考代码**

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import onnxruntime as ort

def get_model_size(model):
    """计算模型大小（MB）"""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

def preprocess_image(image_path):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).numpy()

def evaluate_model(model, test_loader, device='cpu'):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 完整量化流程
def quantize_and_deploy():
    # 1. 加载模型
    model = models.resnet18(pretrained=True)
    model.eval()
    print(f"原始模型大小: {get_model_size(model):.2f} MB")

    # 2. 动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print(f"动态量化后大小: {get_model_size(quantized_model):.2f} MB")

    # 3. 导出ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(quantized_model, dummy_input, "resnet18_dynamic.onnx")

    # 4. ONNX Runtime推理
    session = ort.InferenceSession("resnet18_dynamic.onnx")
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = session.run(None, {'input': input_data})
    print(f"推理输出形状: {output[0].shape}")

if __name__ == "__main__":
    quantize_and_deploy()
```

**预期结果**
- 量化后模型大小减少70%以上
- 推理速度提升50%以上
- Top-1准确率下降不超过2%

---

## 项目三：分布式训练实践

**难度**：⭐⭐⭐（中级）

**学习目标**
- 理解数据并行和模型并行的原理
- 掌握PyTorch DataParallel的使用方法
- 掌握DistributedDataParallel的实现
- 比较两种并行模式的性能差异

**环境要求**
- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU（需要多GPU环境）
- NCCL（NVIDIA集合通信库）

**实现步骤**

**第一步：环境验证**

检查系统中的GPU数量，验证CUDA环境配置正确。安装NCCL确保高效的GPU通信。

```python
import torch

# 检查GPU数量
num_gpus = torch.cuda.device_count()
print(f"可用GPU数量: {num_gpus}")

# 验证CUDA
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

**第二步：实现DataParallel训练**

DataParallel是最简单的多GPU并行方式，只需几行代码即可完成配置。自动实现数据分发和结果聚合。

```python
import torch
import torch.nn as nn
import torchvision.models as models

# DataParallel实现
model = models.resnet50(pretrained=True)
model = model.cuda()
model = nn.DataParallel(model)

# 训练循环
for inputs, labels in train_loader:
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**第三步：实现DistributedDataParallel训练**

DDP需要更多配置但性能更好。首先初始化分布式环境，然后使用DistributedDataParallel包装模型。

```python
import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """DDP训练函数"""
    setup(rank, world_size)

    # 创建本地模型
    model = models.resnet50().cuda()
    model = DDP(model, device_ids=[rank])

    # 创建分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=4
    )

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

**第四步：启动多进程训练**

使用torchrun或torch.distributed.launch启动多进程训练。

```bash
# 使用torchrun启动
torchrun --nproc_per_node=2 train_ddp.py

# 或使用torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py
```

**第五步：实现梯度累积**

当单个GPU显存不足以容纳大批量时，使用梯度累积模拟大批量训练。

```python
# 梯度累积实现
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss = loss / accumulation_steps  # 缩放损失

    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**第六步：实现混合精度训练**

使用torch.cuda.amp实现自动混合精度，减少显存占用并加速训练。

```python
# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

for data, target in train_loader:
    data, target = data.cuda(), target.cuda()

    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**第七步：性能对比测试**

测量DP和DDP的训练速度、GPU利用率和通信开销。

```python
# 性能对比测试
def benchmark_training(model, train_loader, num_iterations=100):
    model.train()
    start = time.time()
    total_samples = 0

    for i, (data, target) in enumerate(train_loader):
        if i >= num_iterations:
            break
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_samples += data.size(0)

    elapsed = time.time() - start
    throughput = total_samples / elapsed
    print(f"吞吐量: {throughput:.2f} samples/sec")
    return throughput
```

**参考代码**

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import time

def benchmark_throughput(model, train_loader, num_iterations=100):
    """测量训练吞吐量"""
    model.train()
    start = time.time()
    total_samples = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i, (data, target) in enumerate(train_loader):
        if i >= num_iterations:
            break
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_samples += data.size(0)

    elapsed = time.time() - start
    return total_samples / elapsed

# 完整的DDP训练框架
class DDPTrainer:
    def __init__(self, model, train_dataset, world_size, batch_size=32):
        self.model = model
        self.train_dataset = train_dataset
        self.world_size = world_size
        self.batch_size = batch_size

    def setup(self, rank):
        """初始化分布式环境"""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)

    def create_data_loader(self, rank):
        """创建分布式数据加载器"""
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=rank
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler
        )

    def train(self, rank, num_epochs=10):
        """训练函数"""
        self.setup(rank)
        model = self.model.cuda()
        model = DDP(model, device_ids=[rank])
        loader = self.create_data_loader(rank)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            loader.sampler.set_epoch(epoch)
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} completed")

        dist.destroy_process_group()

    @staticmethod
    def launch(world_size, train_dataset, model_fn, batch_size=32, num_epochs=10):
        """启动多进程训练"""
        mp.spawn(
            lambda rank: DDPTrainer(
                model_fn(), train_dataset, world_size, batch_size
            ).train(rank, num_epochs),
            args=(),
            nprocs=world_size,
            join=True
        )
```

**预期结果**
- DDP相比单GPU有接近线性的加速比
- GPU利用率达到80%以上
- DDP比DP有更好的训练效率

---

## 项目四：AI编译器探索

**难度**：⭐⭐⭐⭐（中高级）

**学习目标**
- 理解TVM编译器的整体架构
- 掌握使用Relay IR导入模型的方法
- 实现算子融合优化
- 使用AutoTVM进行自动调优
- 编译生成目标硬件代码

**环境要求**
- Python 3.8+
- Apache TVM（需编译安装）
- CUDA（用于GPU优化）
- ONNX（用于模型转换）

**实现步骤**

**第一步：环境安装**

克隆TVM仓库并编译安装，配置TVM_HOME环境变量。

```bash
# 克隆TVM仓库
git clone --recursive https://github.com/apache/tvm
cd tvm

# 编译TVM
mkdir build
cp cmake/config.cmake build/
cmake -H. -Bbuild -G Ninja
ninja -C build

# 配置Python路径
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python:$PYTHONPATH
```

**第二步：导入模型到TVM**

使用Relay前端从ONNX导入模型，或使用torch2tvm从PyTorch直接导入。

```python
import tvm
from tvm import relay
import onnx

# 从ONNX导入模型
onnx_model = onnx.load("resnet18.onnx")
shape_dict = {"input": [1, 3, 224, 224]}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
print("模型导入成功！")
print(mod)
```

**第三步：配置目标硬件**

指定编译的目标硬件平台，配置优化级别和自动调度规则。

```python
# 配置目标
target = tvm.target.Target("cuda -keys=cuda,gpu")

# 或者CPU目标
target = tvm.target.Target("llvm")

# 优化级别
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
```

**第四步：实现算子融合**

分析模型的计算图，识别可以融合的算子模式（如Conv-BN-ReLU），执行融合优化并验证正确性。

```python
# 算子融合配置
with tvm.transform.PassContext(opt_level=3):
    # 融合优化 passes
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyInference(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ])
    mod = seq(mod)
```

**第五步：使用AutoTVM进行自动调优**

识别需要调优的算子，定义调优任务的搜索空间，运行调优器搜索最优配置。

```python
from tvm import autotvm

# 获取调优任务
task = autotvm.get_task("dense", target=target)

# 配置搜索策略
tuning_option = dict(
    n_trials=200,
    early_stopping=50,
    measure_option=autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=20, repeat=3),
    ),
)

# 运行调优
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(tuning_option)
```

**第六步：编译生成优化代码**

使用调优得到的最佳配置编译模型，生成可部署的运行时模块。

```python
# 使用最佳配置编译
with autotvm.apply_history_best(history_best):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

# 保存编译结果
lib.export_library("deploy.so")
```

**第七步：部署和性能验证**

加载编译后的模块，测量推理延迟和吞吐量，对比优化效果。

```python
from tvm.contrib import graph_executor

# 加载编译模块
dev = tvm.cuda()
module = graph_executor.GraphModule(lib["default"](dev))

# 运行推理
module.set_input("input", input_data)
module.run()
output = module.get_output(0).numpy()

# 性能测量
import time
num_iterations = 100
start = time.time()
for _ in range(num_iterations):
    module.run()
elapsed = time.time() - start
latency = elapsed / num_iterations * 1000  # 毫秒
throughput = batch_size * num_iterations / elapsed
```

**参考代码**

```python
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import time

class TVMCompiler:
    """TVM编译器封装"""

    def __init__(self, target="cuda"):
        self.target = tvm.target.Target(target)
        self.mod = None
        self.params = None
        self.lib = None
        self.module = None

    def from_onnx(self, onnx_model, shape_dict):
        """从ONNX模型导入"""
        self.mod, self.params = relay.frontend.from_onnx(
            onnx_model, shape_dict
        )
        return self

    def optimize(self, opt_level=3, fuse_ops=True):
        """优化模型"""
        with tvm.transform.PassContext(opt_level=opt_level):
            if fuse_ops:
                seq = tvm.transform.Sequential([
                    relay.transform.InferType(),
                    relay.transform.FoldConstant(),
                    relay.transform.SimplifyInference(),
                    relay.transform.FuseOps(fuse_opt_level=2),
                ])
                self.mod = seq(self.mod)
        return self

    def build(self):
        """编译模型"""
        with tvm.transform.PassContext(opt_level=3):
            self.lib = relay.build(
                self.mod,
                target=self.target,
                params=self.params
            )
        return self

    def deploy(self):
        """部署模型"""
        if self.target.kind.name == "cuda":
            dev = tvm.cuda()
        else:
            dev = tvm.cpu()
        self.module = graph_executor.GraphModule(
            self.lib["default"](dev)
        )
        return self

    def run(self, input_data):
        """运行推理"""
        self.module.set_input("input", input_data)
        self.module.run()
        return self.module.get_output(0).numpy()

    def benchmark(self, input_shape, n_warmup=10, n_run=100):
        """性能基准测试"""
        input_data = tvm.nd.array(
            np.random.randn(*input_shape).astype("float32")
        )
        self.module.set_input("input", input_data)

        # 预热
        for _ in range(n_warmup):
            self.module.run()

        # 测量
        start = time.time()
        for _ in range(n_run):
            self.module.run()
        elapsed = time.time() - start

        return {
            "latency_ms": elapsed / n_run * 1000,
            "throughput_fps": n_run / elapsed * input_shape[0],
        }

# 使用示例
def compile_resnet18():
    import onnx
    onnx_model = onnx.load("resnet18.onnx")

    compiler = TVMCompiler(target="cuda")
    compiler.from_onnx(onnx_model, {"input": [1, 3, 224, 224]})
    compiler.optimize(opt_level=3)
    compiler.build()
    compiler.deploy()

    # 性能测试
    results = compiler.benchmark([1, 3, 224, 224])
    print(f"延迟: {results['latency_ms']:.2f} ms")
    print(f"吞吐量: {results['throughput_fps']:.2f} FPS")

    return compiler
```

**预期结果**
- 成功将模型导入TVM并完成编译
- 通过AutoTVM调优实现显著的性能提升
- 优化后推理速度显著快于原生框架

---

## 项目五：推理系统构建

**难度**：⭐⭐⭐⭐（中高级）

**学习目标**
- 理解Triton Inference Server的架构
- 掌握配置模型仓库的方法
- 实现客户端推理请求
- 理解模型编排和批量推理

**环境要求**
- Python 3.8+
- Docker（用于运行Triton Server）
- tritonclient库
- NVIDIA GPU（可选）

**实现步骤**

**第一步：环境准备**

安装Docker和NVIDIA Docker支持（如果使用GPU）。下载并启动Triton Inference Server镜像。

```bash
# 拉取Triton镜像
docker pull nvcr.io/nvidia/tritonserver:23.10-py3

# 验证安装
docker run --rm nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --version
```

**第二步：准备模型仓库**

创建模型仓库目录结构，每个模型需要包含模型定义和权重文件。

```bash
# 模型仓库目录结构
model_repository/
└── resnet18/
    ├── 1/
    │   └── model.onnx    # 模型权重
    └── config.pbtxt     # 模型配置文件
```

**第三步：创建模型配置文件**

编写config.pbtxt配置文件，定义模型的输入输出、批量大小、设备类型等。

```protobuf
name: "resnet18"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

**第四步：启动Triton Server**

启动推理服务器，加载模型并提供服务。

```bash
# 启动服务器（CPU）
docker run --rm -p8000:8000 -p8001:8001 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models

# 启动服务器（GPU）
docker run --rm --gpus=1 -p8000:8000 -p8001:8001 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --gpu_id=0
```

**第五步：实现客户端推理请求**

使用tritonclient库连接服务器，发送推理请求并处理响应。

```python
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 创建客户端
client = httpclient.InferenceServerClient(url="localhost:8000")

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).numpy()

# 发送推理请求
def infer(client, image_path):
    # 预处理图像
    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # 添加batch维度

    # 创建请求
    inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [httpclient.InferRequestedOutput("output")]

    # 发送请求
    results = client.infer("resnet18", inputs, outputs=outputs)

    # 获取结果
    output_data = results.as_numpy("output")
    predicted_class = np.argmax(output_data)
    return predicted_class, output_data
```

**第六步：实现批量推理和动态 batching**

配置Triton的动态 batching 功能，提高推理吞吐量。

```python
# 动态batching配置
config.pbtxt中添加：
dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 100000
}
```

**第七步：性能监控**

使用Triton的metrics API监控推理性能。

```python
import tritonclient.http as httpclient

# 获取metrics
client = httpclient.InferenceServerClient(url="localhost:8000")
metrics = client.get_metrics()

# 解析metrics
# - inference_request_count: 推理请求数
# - inference_success_count: 成功推理数
# - avg_request_latency_ms: 平均延迟
print(metrics)
```

**参考代码**

```python
import tritonclient.http as httpclient
import numpy as np
import time
from typing import List, Tuple

class TritonClient:
    """Triton Inference Client封装"""

    def __init__(self, url="localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """图像预处理"""
        from PIL import Image
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image)
        return img_tensor.numpy()

    def infer(
        self,
        model_name: str,
        image_paths: List[str]
    ) -> Tuple[np.ndarray, float]:
        """发送批量推理请求"""
        # 预处理
        input_data = np.stack([self.preprocess_image(p) for p in image_paths])

        # 创建输入
        inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data.astype(np.float32))

        # 创建输出
        outputs = [httpclient.InferRequestedOutput("output")]

        # 发送请求
        start_time = time.time()
        results = self.client.infer(model_name, inputs, outputs=outputs)
        latency = time.time() - start_time

        # 获取结果
        output_data = results.as_numpy("output")
        return output_data, latency

    def infer_async(
        self,
        model_name: str,
        image_paths: List[str],
        callback=None
    ):
        """异步推理"""
        input_data = np.stack([self.preprocess_image(p) for p in image_paths])

        inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data.astype(np.float32))

        outputs = [httpclient.InferRequestedOutput("output")]

        self.client.async_infer(
            model_name,
            inputs,
            outputs,
            callback=callback
        )

    def get_server_status(self):
        """获取服务器状态"""
        return self.client.get_server_metadata()

    def get_model_status(self, model_name: str):
        """获取模型状态"""
        return self.client.get_model_metadata(model_name)

    def benchmark(
        self,
        model_name: str,
        image_paths: List[str],
        num_iterations: int = 100
    ):
        """性能基准测试"""
        latencies = []
        throughputs = []

        for _ in range(num_iterations):
            batch_size = len(image_paths)
            output, latency = self.infer(model_name, image_paths)
            latencies.append(latency * 1000)  # 转换为毫秒
            throughputs.append(batch_size / latency)

        return {
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_throughput": np.mean(throughputs),
        }

# 使用示例
if __name__ == "__main__":
    client = TritonClient("localhost:8000")

    # 检查服务器状态
    status = client.get_server_status()
    print(f"服务器状态: {status}")

    # 推理测试
    image_paths = ["test_image.jpg"]  # 替换为实际图像路径
    predicted_class, latency = client.infer("resnet18", image_paths)
    print(f"预测类别: {predicted_class}, 延迟: {latency*1000:.2f}ms")

    # 性能测试
    results = client.benchmark("resnet18", image_paths, num_iterations=100)
    print(f"平均延迟: {results['avg_latency_ms']:.2f}ms")
    print(f"P95延迟: {results['p95_latency_ms']:.2f}ms")
    print(f"吞吐量: {results['avg_throughput']:.2f} samples/sec")
```

**预期结果**
- 成功启动Triton Server并加载模型
- 能够发送推理请求并获取正确结果
- 动态 batching 提升推理吞吐量

---

## 项目六：大模型分布式训练

**难度**：⭐⭐⭐⭐⭐（高级）

**学习目标**
- 理解大模型分布式训练的原理
- 掌握DeepSpeed的配置和使用方法
- 实现混合并行训练（数据并行+模型并行+流水线并行）
- 测试训练效率和内存优化效果

**环境要求**
- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU（需要多GPU环境，至少2张）
- DeepSpeed
- transformers库

**实现步骤**

**第一步：环境安装**

安装DeepSpeed及其依赖。

```bash
# 安装DeepSpeed
pip install deepspeed

# 验证安装
ds_report
```

**第二步：配置DeepSpeed环境**

创建DeepSpeed配置文件，定义优化策略、ZeRO优化等级、混合精度等参数。

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": false
}
```

**第三步：初始化DeepSpeed训练**

在训练代码中初始化DeepSpeed引擎。

```python
import deepspeed
import torch

# DeepSpeed配置
ds_config = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3
    }
}

# 模型定义
model = YourModel()  # 替换为实际模型

# 初始化DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

print(f"GPU数量: {deepspeed.comm.get_world_size()}")
```

**第四步：实现混合并行训练**

配置数据并行、模型并行和流水线并行。

```python
import deepspeed

# 混合并行配置
ds_config = {
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_clipping": 1.0,
    "pipeline": {
        "enabled": True,
        "num_stages": 4  # 流水线阶段数
    },
    "tensor_parallel": {
        "enabled": True,
        "tp_size": 2  # 张量并行大小
    }
}

# 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

**第五步：实现训练循环**

使用DeepSpeed引擎进行训练。

```python
def train_step(model_engine, batch):
    """单个训练步骤"""
    # 将数据移动到设备
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(model_engine.device)

    # 前向传播
    outputs = model_engine(**batch)
    loss = outputs.loss

    # 反向传播
    model_engine.backward(loss)
    model_engine.step()

    return loss.item()

# 训练循环
for epoch in range(num_epochs):
    model_engine.train()
    for batch in train_loader:
        loss = train_step(model_engine, batch)

    # 验证
    model_engine.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model_engine(**batch)
            # 计算验证指标

    if epoch % save_interval == 0:
        model_engine.save_checkpoint(save_dir, tag=epoch)
```

**第六步：测试训练效率**

测量训练吞吐量、GPU利用率和内存使用情况。

```python
import torch
import time
import deepspeed.comm as dist

def benchmark_training(model_engine, train_loader, num_iterations=100):
    """基准测试训练性能"""
    model_engine.train()
    total_samples = 0
    start_time = time.time()

    for i, batch in enumerate(train_loader):
        if i >= num_iterations:
            break

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(model_engine.device)

        outputs = model_engine(**batch)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        total_samples += batch['input_ids'].shape[0]

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed

    # 获取GPU内存使用
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存已分配: {memory_allocated:.2f} GB")
        print(f"GPU内存已预留: {memory_reserved:.2f} GB")

    return throughput

def measure_gpu_utilization():
    """测量GPU利用率"""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())
```

**参考代码**

```python
import deepspeed
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import time

class DeepSpeedTrainer:
    """DeepSpeed训练器封装"""

    def __init__(self, model_name, ds_config_path):
        self.model_name = model_name
        self.ds_config = ds_config_path
        self.model_engine = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None

    def setup_model(self):
        """初始化模型"""
        config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=config
        )

    def setup_deepspeed(self, **kwargs):
        """初始化DeepSpeed"""
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=self.ds_config,
            **kwargs
        )

    def create_dataloaders(self, train_dataset, val_dataset, batch_size):
        """创建数据加载器"""
        from transformers import DataLoader

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def train_step(self, batch):
        """单步训练"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.model_engine.device)

        outputs = self.model_engine(**batch)
        loss = outputs.loss

        self.model_engine.backward(loss)
        self.model_engine.step()

        return loss.item()

    def evaluate(self):
        """验证"""
        self.model_engine.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.model_engine.device)

                outputs = self.model_engine(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs, log_interval=10):
        """训练循环"""
        for epoch in range(num_epochs):
            self.model_engine.train()
            epoch_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                if batch_idx % log_interval == 0:
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss:.4f}")

            avg_loss = epoch_loss / num_batches
            val_loss = self.evaluate()
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

    def save_checkpoint(self, save_dir, tag):
        """保存检查点"""
        self.model_engine.save_checkpoint(save_dir, tag=tag)

    def load_checkpoint(self, save_dir, tag):
        """加载检查点"""
        self.model_engine.load_checkpoint(save_dir, tag=tag)

# 使用示例
def main():
    # 配置
    ds_config = "deepspeed_config.json"

    # 初始化训练器
    trainer = DeepSpeedTrainer("bert-base-uncased", ds_config)

    # 设置模型
    trainer.setup_model()

    # 初始化DeepSpeed
    trainer.setup_deepspeed()

    # 创建数据加载器（需要准备实际数据集）
    # trainer.create_dataloaders(train_dataset, val_dataset, batch_size=4)

    # 训练
    # trainer.train(num_epochs=3)

if __name__ == "__main__":
    main()
```

**预期结果**
- 成功使用DeepSpeed进行大模型分布式训练
- ZeRO优化显著降低GPU内存占用
- 混合并行训练效率显著提升

---

## 项目提交要求

每个项目完成后，学员需要提交以下材料：

**第一，项目代码**

需要提交完整的、可运行的代码，包含必要的注释和说明文档。代码应该有良好的结构，便于阅读和理解。

**第二，实验报告**

详细记录实验过程和结果，包括环境配置、参数设置、性能指标等。报告应该包含可视化图表展示性能对比。

**第三，结果分析**

对实验结果进行深入分析，解释性能差异的原因，提出改进方向。

**第四，心得总结**

总结项目实施过程中的收获和遇到的困难，以及对AI系统相关知识的理解提升。

---

## 附录：常见问题与解决方案

**问题一：GPU内存不足**

解决方案：启用混合精度训练、使用梯度累积、启用ZeRO优化、减小批量大小。

**问题二：分布式训练通信失败**

解决方案：检查NCCL安装、验证网络连接、确保MASTER_ADDR和MASTER_PORT配置正确。

**问题三：模型转换失败**

解决方案：检查模型格式兼容性、验证算子支持情况、使用更简单的模型进行测试。

**问题四：推理速度不理想**

解决方案：启用更多的编译器优化、使用AutoTVM调优、考虑使用TensorRT等硬件特定优化。
