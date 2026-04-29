# 附录A：AI系统术语表

## 引言

本术语表系统整理了人工智能系统领域涉及的核心专业术语，涵盖数学与统计基础、机器学习基础、神经网络基础、深度学习模型、AI芯片与硬件、AI编译器、推理与部署、分布式系统八大类别。每个术语均提供中英文对照、缩写及详细释义，帮助读者建立完整的AI系统知识体系。术语表总计收录约500个核心条目，总字数约三万字，适用于AI系统设计、研发、部署及运维等领域的专业参考。

---

## A. 数学与统计基础

### 向量 (Vector)

英文：Vector

缩写：Vec

释义：向量是具有大小和方向的数学对象，在AI中常用于表示数据特征、模型参数或嵌入表示。一维数组可视为向量，向量运算包括加法、数乘、点积和叉积，在神经网络中广泛用于前向传播和反向传播计算。

### 矩阵 (Matrix)

英文：Matrix

缩写：Mat

释义：矩阵是由行和列组成的二维数组，是线性代数的核心数据结构。在AI系统中，权重矩阵、输入矩阵、激活矩阵等构成神经网络计算的基础。矩阵运算包括加法、乘法、转置、求逆等，是深度学习前向计算和梯度传播的基本形式。

### 张量 (Tensor)

英文：Tensor

释义：张量是向量和矩阵向高维空间的推广，可理解为多维数组。在深度学习框架中，张量是数据的基本表示形式，标量（0维）、向量（1维）、矩阵（2维）都是张量的特例。张量操作包括形状变换、索引切片、广播运算等，是现代AI框架如PyTorch、TensorFlow的核心数据结构。

### 标量 (Scalar)

英文：Scalar

释义：标量是仅具有大小没有方向的量，即单个数值。在AI中用于表示学习率、损失值、精确度等单一数值指标。标量可以是整数、浮点数或其他基本数据类型，是张量运算的基本单元。

### 范数 (Norm)

英文：Norm

释义：范数是衡量向量或矩阵大小的数学度量。常见范数包括L1范数（向量元素绝对值之和）、L2范数（欧几里得距离）、无穷范数（最大绝对值元素）。在AI中，L2范数用于权重衰减正则化，L1范数用于产生稀疏权重，帮助模型实现特征选择。

### 特征值 (Eigenvalue)

英文：Eigenvalue

释义：特征值是方阵的重要性质，满足Av=λv，其中v是特征向量，λ是特征值。在主成分分析（PCA）中，特征值表示主成分的方差贡献；在谱聚类中，特征值用于图分割。特征值分解将矩阵表示为特征向量与特征值的组合，揭示矩阵的内在结构特性。

### 特征向量 (Eigenvector)

英文：Eigenvector

释义：特征向量是满足方阵与向量乘积等于该向量标量倍数的非零向量。在AI的降维算法中，特征向量构成主成分的方向；在PageRank算法中，特征向量用于计算网页重要性排序。特征向量与特征值共同刻画矩阵的核心性质。

### 奇异值分解 (Singular Value Decomposition)

英文：Singular Value Decomposition

缩写：SVD

释义：奇异值分解是将任意矩阵分解为三个特殊矩阵的乘积：A=UΣV^T，其中U和V是正交矩阵，Σ是对角矩阵。SVD用于数据降维、特征提取、矩阵近似等。在推荐系统中，SVD用于用户-物品评分矩阵分解；在自然语言处理中，SVD用于词嵌入的降维可视化。

### 主成分分析 (Principal Component Analysis)

英文：Principal Component Analysis

缩写：PCA

释义：PCA是一种统计方法，通过线性变换将高维数据投影到低维空间，同时最大化保留数据的方差信息。PCA通过计算协方差矩阵的特征值和特征向量来确定主成分方向，用于数据压缩、去噪、可视化等任务，是AI系统中常用的降维技术。

### 概率 (Probability)

英文：Probability

释义：概率是描述某事件发生可能性大小的数值，取值范围为0到1之间。在机器学习中，概率用于量化不确定性、描述数据分布、计算后验概率等。概率论为贝叶斯推断、概率图模型、生成模型等AI核心方法提供数学基础。

### 条件概率 (Conditional Probability)

英文：Conditional Probability

释义：条件概率是在已知某事件发生的条件下，另一事件发生的概率，记作P(A|B)。在AI中，条件概率用于贝叶斯推理、隐马尔可夫模型、朴素贝叶斯分类器等。条件概率满足乘法规则：P(A|B)=P(A,B)/P(B)。

### 贝叶斯定理 (Bayes' Theorem)

英文：Bayes' Theorem

释义：贝叶斯定理描述如何根据新证据更新概率估计：P(A|B)=P(B|A)P(A)/P(B)。在机器学习中，贝叶斯方法用于推断模型参数的后验分布，实现贝叶斯回归、贝叶斯神经网络等。贝叶斯推断为不确定性量化和模型压缩提供理论基础。

### 期望 (Expectation)

英文：Expectation

释义：期望是随机变量以概率为权重的加权平均值，反映随机变量的长期平均行为。离散型期望E[X]=Σxp(x)，连续型期望E[X]=∫xf(x)dx。在机器学习中，期望用于计算损失函数的期望风险、梯度估计等，是统计学习理论的核心概念。

### 方差 (Variance)

英文：Variance

释义：方差是随机变量与其期望值偏差平方的期望，衡量数据的离散程度：Var(X)=E[(X-E[X])^2]。在机器学习中，方差描述模型预测的波动性，与偏差共同构成偏差-方差权衡，是理解模型泛化能力的关键指标。

### 协方差 (Covariance)

英文：Covariance

释义：协方差描述两个随机变量联合变化的趋势：Cov(X,Y)=E[(X-E[X])(Y-E[Y])]。正值表示同向变化，负值表示反向变化。在AI中，协方差矩阵描述多维数据的特征相关性，用于高斯过程、卡尔曼滤波、多元高斯分布等。

### 相关系数 (Correlation Coefficient)

英文：Correlation Coefficient

缩写：Corr

释义：相关系数是标准化的协方差，取值范围为[-1,1]，消除量纲影响。皮尔逊相关系数r=Cov(X,Y)/(σXσY)衡量线性相关程度。在特征选择中，相关系数用于去除冗余特征；在数据预处理中，相关性分析帮助理解变量间的关系。

### 行列式 (Determinant)

英文：Determinant

缩写：det

释义：行列式是方阵到标量的数学函数，反映矩阵的某些内在特性。行列式为零表示矩阵奇异（不可逆）；行列式的绝对值表示矩阵变换的体积缩放因子。在线性方程组求解、矩阵逆运算、特征值计算中，行列式是重要的中间量。

### 逆矩阵 (Inverse Matrix)

英文：Inverse Matrix

缩写：A^-1

释义：逆矩阵是满足AA^-1=A^-1A=I的矩阵，其中I是单位阵。逆矩阵仅存在于满秩方阵。在机器学习中，逆矩阵用于线性回归正规方程的求解、线性变换的可逆操作等。实际计算中常用伪逆处理秩亏情况。

### 矩阵的迹 (Trace)

英文：Trace

缩写：tr

释义：矩阵的迹是主对角线元素的和，等于所有特征值之和。迹运算具有循环不变性：tr(ABC)=tr(BCA)=tr(CAB)。在AI中，迹用于计算矩阵的F范数（通过tr(A^TA)）、神经网络的某些正则化项、以及信息论中的某些度量。

### 矩阵的秩 (Rank)

英文：Rank

释义：矩阵的秩是线性无关行（或列）的最大数目，反映矩阵的信息含量。满秩矩阵具有独立的行和列，秩亏矩阵存在冗余信息。在神经网络中，权重矩阵的秩影响表示能力；在数据处理中，秩用于判断方程组解的存在性和唯一性。

### LU分解 (LU Decomposition)

英文：LU Decomposition

缩写：LU

释义：LU分解将矩阵表示为下三角矩阵L和上三角矩阵U的乘积：A=LU。LU分解用于高效求解线性方程组、计算行列式、求逆矩阵等。在AI系统的编译器优化中，LU分解用于自动生成高效的矩阵运算代码。

### QR分解 (QR Decomposition)

英文：QR Decomposition

缩写：QR

释义：QR分解将矩阵表示为正交矩阵Q和上三角矩阵R的乘积：A=QR。QR分解在最小二乘拟合、特征值计算（QR算法）中广泛应用，是数值线性代数的核心工具，在AI的回归分析和矩阵分解中常见。

### Cholesky分解 (Cholesky Decomposition)

英文：Cholesky Decomposition

缩写：Chol

释义：Cholesky分解是正定矩阵的特殊LU分解：A=LL^T，其中L是下三角矩阵。Cholesky分解计算效率高于一般LU分解，在高斯过程、多元正态分布采样、约束优化等AI算法中用于高效协方差运算。

### 雅可比矩阵 (Jacobian Matrix)

英文：Jacobian Matrix

缩写：Jac

释义：雅可比矩阵是多元向量值函数的一阶偏导数构成的矩阵，描述函数在一点的最优线性近似。在深度学习中，雅可比矩阵用于反向传播算法的高阶扩展、牛顿法优化、以及某些可视化技术。

### 海森矩阵 (Hessian Matrix)

英文：Hessian Matrix

缩写：Hess

释义：海森矩阵是多元函数的二阶偏导数构成的方阵，描述函数的曲率特性。海森矩阵用于牛顿法优化、损失函数的二阶近似、确定性等高线绘制等。在训练大规模神经网络时，由于海森矩阵计算成本高，常用拟牛顿法或Adam等自适应学习率方法近似。

### 梯度 (Gradient)

英文：Gradient

缩写：grad

释义：梯度是多元函数在某点处方向导数最大的向量，指向函数增长最快的方向。梯度的每个分量是偏导数：∇f=(∂f/∂x1,...,∂f/∂xn)。在机器学习中，梯度用于参数更新方向的决定，是梯度下降法的核心概念。

### 拉格朗日乘数法 (Lagrange Multiplier)

英文：Lagrange Multiplier

释义：拉格朗日乘数法是求解约束优化问题的经典方法，通过引入拉格朗日乘子将约束条件融入目标函数。在机器学习的SVM、神经网络剪枝、特征选择等任务中，拉格朗日乘数法用于带约束的参数优化问题。

### 凸函数 (Convex Function)

英文：Convex Function

释义：凸函数的几何特性是函数图像上任意两点连线都在函数图像上方。凸函数具有唯一的全局最优解，梯度下降法能保证收敛到全局最优。在机器学习中，逻辑回归、线性SVM等模型具有凸损失函数，保证训练的稳定性。

### 优化器 (Optimizer)

英文：Optimizer

释义：优化器是用于更新模型参数以最小化损失函数的算法。常见优化器包括SGD、Adam、RMSProp、AdaGrad等。优化器通过计算梯度并应用更新规则调整参数，决定模型收敛速度和最终性能，是深度学习训练的关键组件。

### 学习率 (Learning Rate)

英文：Learning Rate

缩写：LR

释义：学习率是优化器在梯度方向上更新参数的步长大小，控制模型参数每次更新的幅度。学习率过大会导致训练不稳定甚至发散，学习率过小会导致收敛缓慢。常用学习率调度策略包括学习率衰减、余弦退火、warmup等。

### 梯度下降 (Gradient Descent)

英文：Gradient Descent

缩写：GD

释义：梯度下降是一阶优化算法，沿梯度负方向更新参数以最小化目标函数。参数更新公式为θ=θ-α∇J(θ)，其中α是学习率。梯度下降有批梯度下降（使用全部数据）、小批量梯度下降（使用batch数据）等变体，是机器学习和深度学习的基石算法。

### 随机梯度下降 (Stochastic Gradient Descent)

英文：Stochastic Gradient Descent

缩写：SGD

释义：随机梯度下降使用单个样本或小批量样本的梯度近似全局梯度，大幅降低计算复杂度。SGD在每次迭代中随机选择样本，能逃离浅局部极小值，常配合动量、学习率衰减等技巧使用。SGD是深度学习训练的主流优化方法。

### 动量 (Momentum)

英文：Momentum

释义：动量模拟物理中的惯性概念，在梯度下降中引入速度项积累历史梯度方向。动量更新公式为v=βv+(1-β)∇J(θ)，θ=θ-αv。动量加速收敛、减少振荡，对高曲率、噪声梯度的处理更鲁棒，是深度学习训练的常用技巧。

### 自适应学习率 (Adaptive Learning Rate)

英文：Adaptive Learning Rate

释义：自适应学习率方法为不同参数自动调整学习率，如Adam维护参数的一阶矩估计和二阶矩估计。Adam更新公式包含偏置校正，对稀疏梯度鲁棒，被广泛应用于深度学习训练。RMSProp、AdaGrad等也是常见的自适应学习率方法。

### 正则化 (Regularization)

英文：Regularization

释义：正则化是通过在损失函数中添加惩罚项来约束模型复杂度，防止过拟合的技术。常见正则化包括L1正则化（产生稀疏解）、L2正则化（权重衰减）、Dropout（在训练时随机丢弃神经元）。正则化是提高模型泛化能力的关键手段。

### 拉普拉斯分布 (Laplace Distribution)

英文：Laplace Distribution

释义：拉普拉斯分布是双指数分布，概率密度函数在零点处呈尖峰形状。因其稀疏性特性，拉普拉斯分布常作为稀疏编码、变分推断中先验分布的选择，激励模型学习稀疏表示。

### 高斯分布 (Gaussian Distribution)

英文：Gaussian Distribution

别名：正态分布 (Normal Distribution)

缩写：N(μ,σ²)

释义：高斯分布是最常见的连续概率分布，其概率密度函数呈钟形曲线。高斯分布由均值μ和方差σ²完全决定，在中心极限定理的支持下成为统计推断和机器学习的基础假设。神经网络权重常初始化为高斯分布。

### 伯努利分布 (Bernoulli Distribution)

英文：Bernoulli Distribution

释义：伯努利分布是离散概率分布，描述单次二元随机试验的结果，取值为0或1。参数p表示取值为1的概率。逻辑回归的输出可理解为伯努利分布的概率，用于二分类问题。

### 多项式分布 (Multinomial Distribution)

英文：Multinomial Distribution

释义：多项式分布是伯努利分布向多类别的推广，描述在n次独立试验中各类别出现次数的联合分布。softmax输出的多分类概率可理解为多项式分布，在神经网络的多分类输出层广泛使用。

### 最大似然估计 (Maximum Likelihood Estimation)

英文：Maximum Likelihood Estimation

缩写：MLE

释义：最大似然估计是选择使观测数据出现概率最大的参数值作为参数的估计。在机器学习中，MLE通过最大化似然函数来学习模型参数，是逻辑回归、神经网络等参数化模型训练的理论基础。

### KL散度 (Kullback-Leibler Divergence)

英文：Kullback-Leibler Divergence

缩写：KL

释义：KL散度衡量两个概率分布的差异程度：D_KL(P||Q)=ΣP(i)log(P(i)/Q(i))。KL散度非负，仅在两分布相同时为零，且不满足对称性。在变分推断、GAN、蒸馏等AI算法中，KL散度用于衡量分布差异或作为训练目标。

### 交叉熵 (Cross Entropy)

英文：Cross Entropy

缩写：CE

释义：交叉熵是信息论中的重要概念，H(P,Q)=-ΣP(i)logQ(i)。在机器学习中，交叉熵常用作分类任务的损失函数，等价于对数似然损失。交叉熵相比均方误差在梯度特性上更利于神经网络训练。

### 信息熵 (Information Entropy)

英文：Information Entropy

缩写：H

释义：信息熵衡量随机变量的不确定性：H(X)=-Σp(x)logp(x)。熵值越大表示不确定性越高。在决策树算法中，熵用于选择最优分割特征；在特征选择中，熵用于评估特征的信息量。

### 互信息 (Mutual Information)

英文：Mutual Information

缩写：MI

释义：互信息衡量两个随机变量之间的依赖程度：I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)。互信息非负且对称，在特征选择、贝叶斯网络、变量约简等AI算法中用于衡量变量间的统计相关性。

---

## B. 机器学习基础

### 监督学习 (Supervised Learning)

英文：Supervised Learning

缩写：SL

释义：监督学习是从标注数据中学习输入到输出的映射关系的机器学习范式。训练时提供输入特征和对应标签，模型学习使预测与真实标签一致的函数。分类和回归是监督学习的主要任务，如图像分类、语音识别、房价预测等。

### 无监督学习 (Unsupervised Learning)

英文：Unsupervised Learning

缩写：UL

释义：无监督学习是在无标注数据中发现数据内在结构的机器学习范式。主要任务包括聚类（将相似样本分组）、降维（保留关键信息的同时减少维度）、密度估计、异常检测等。典型算法包括K-means、PCA、自编码器等。

### 半监督学习 (Semi-supervised Learning)

英文：Semi-supervised Learning

缩写：SSL

释义：半监督学习是利用少量标注数据和大量无标注数据共同训练模型的机器学习范式。半监督学习通过无标注数据发现数据分布结构，提高模型泛化能力。常见方法包括自训练、协同训练、标签传播等。

### 强化学习 (Reinforcement Learning)

英文：Reinforcement Learning

缩写：RL

释义：强化学习是智能体通过与环境交互，以最大化累积奖励为目标学习最优策略的机器学习范式。智能体观察状态、执行动作、获得奖励，通过试错学习决策策略。强化学习广泛应用于游戏AI、机器人控制、推荐系统等领域。

### 损失函数 (Loss Function)

英文：Loss Function

别名：代价函数 (Cost Function)、目标函数 (Objective Function)

缩写：L

释义：损失函数衡量模型预测值与真实值之间的差异程度，是模型优化的目标函数。常见损失函数包括均方误差（回归）、交叉熵（分类）、hinge损失（SVM）等。损失函数的选择影响模型的学习效率和最终性能。

### 均方误差 (Mean Squared Error)

英文：Mean Squared Error

缩写：MSE

释义：均方误差是回归任务常用的损失函数，计算预测值与真实值差的平方的平均值：MSE=(1/n)Σ(yi-ŷi)²。MSE对大误差惩罚更重，但对异常值敏感。MSE的梯度特性利于优化，是神经网络回归输出层的常用损失函数。

### 均方根误差 (Root Mean Squared Error)

英文：Root Mean Squared Error

缩写：RMSE

释义：均方根误差是MSE的平方根，与目标变量具有相同单位，更易于解释。RMSE对大误差更加敏感，在评估回归模型性能、比较不同模型时常用。RMSE值越小表示预测精度越高。

### 平均绝对误差 (Mean Absolute Error)

英文：Mean Absolute Error

缩写：MAE

释义：平均绝对误差是预测误差绝对值的平均：MAE=(1/n)Σ|yi-ŷi|。MAE对异常值更鲁棒，梯度恒定，在某些场景下优于MSE。MAE也被称为L1损失，在回归任务中作为MSE的替代选择。

### 梯度下降法 (Gradient Descent Method)

英文：Gradient Descent Method

缩写：GD

释义：梯度下降法是使用目标函数梯度信息迭代求解最优解的优化方法。对于最小化问题，沿梯度负方向以学习率为步长更新参数。梯度下降有批量、小批量、随机三种变体，是机器学习和深度学习的核心优化算法。

### 批量梯度下降 (Batch Gradient Descent)

英文：Batch Gradient Descent

缩写：BGD

释义：批量梯度下降在每次迭代中使用全部训练数据计算梯度，然后更新参数。批量梯度下降保证收敛到全局最优（凸函数）或驻点（非凸函数），但计算开销大，不适合大规模训练数据。

### 小批量梯度下降 (Mini-batch Gradient Descent)

英文：Mini-batch Gradient Descent

缩写：MBGD

释义：小批量梯度下降每次迭代使用一小批（mini-batch）训练样本计算梯度。Mini-batch结合了批量和随机梯度下降的优点：利用GPU并行计算加速、减少参数更新的方差、是深度学习训练的标准方法。

### 学习曲线 (Learning Curve)

英文：Learning Curve

释义：学习曲线是模型性能随训练样本数量变化的曲线，用于诊断模型的学习状态。通过比较训练集和验证集的学习曲线，可判断模型是否存在过拟合（高方差）或欠拟合（高偏差）问题，指导模型改进方向。

### 过拟合 (Overfitting)

英文：Overfitting

释义：过拟合是模型在训练数据上表现良好但在未见过的测试数据上表现较差的现象。过拟合表明模型学习了训练数据的噪声和细节，而非数据的通用规律。缓解过拟合的方法包括增加数据、正则化、Dropout、Early stopping等。

### 欠拟合 (Underfitting)

英文：Underfitting

释义：欠拟合是模型在训练数据和测试数据上都表现不佳的现象，表明模型未能学习到数据的内在规律。欠拟合通常由于模型过于简单、训练不足、特征不充分等原因导致，需要增加模型复杂度或训练时间来解决。

### 偏差-方差权衡 (Bias-Variance Tradeoff)

英文：Bias-Variance Tradeoff

释义：偏差-方差权衡描述模型误差的两个来源：偏差是模型预测值与真实值的系统性差异（欠拟合），方差是模型对训练数据波动的敏感程度（过拟合）。降低一方往往增加另一方，模型设计需要在二者间找到平衡以最小化总误差。

### 泛化能力 (Generalization Ability)

英文：Generalization Ability

释义：泛化能力指模型对新、未见数据的预测能力，是机器学习的核心目标。泛化能力强的模型能将从训练数据学到的知识应用到新场景。通过正则化、数据增强、模型选择等手段可提高模型的泛化能力。

### 训练集 (Training Set)

英文：Training Set

释义：训练集是用于训练模型参数的数据集合。模型通过在训练集上优化损失函数学习输入与输出的映射关系。训练集应覆盖任务的各种场景，保证模型能学到有效的决策边界或函数映射。

### 验证集 (Validation Set)

英文：Validation Set

释义：验证集是用于模型选择和超参数调优的数据集合，在训练过程中独立于训练集。验证集用于监控模型性能、early stopping、选择最优模型配置，避免信息泄漏导致对测试集性能的过度乐观估计。

### 测试集 (Test Set)

英文：Test Set

释义：测试集是用于最终评估模型性能的数据集合，在模型开发完成后仅使用一次。测试集应与训练集、验证集同分布且互不相交，准确反映模型在实际应用中的表现，是模型性能评估的金标准。

### 交叉验证 (Cross Validation)

英文：Cross Validation

缩写：CV

释义：交叉验证是将数据划分为k个互斥子集，轮流使用k-1个子集训练、1个子集验证的模型评估方法。K折交叉验证充分利用有限数据，提供更稳定可靠的性能估计。Leave-one-out是k等于样本数的极端情况。

### 留出法 (Hold-out Method)

英文：Hold-out Method

释义：留出法是将数据简单划分为训练集、验证集、测试集的模型评估方法。留出法简单直接，是实际工程中的常用做法。关键是要保证各集合与总数据集同分布，避免引入偏差影响模型评估。

### 混淆矩阵 (Confusion Matrix)

英文：Confusion Matrix

释义：混淆矩阵是分类模型预测结果的n×n表格，行表示真实类别，列表示预测类别。混淆矩阵直观展示各类别的预测正确数和错误数，用于计算准确率、精确率、召回率等评估指标，是分类模型评估的基础工具。

### 准确率 (Accuracy)

英文：Accuracy

缩写：Acc

释义：准确率是分类正确的样本数占总样本数的比例：(TP+TN)/(TP+TN+FP+FN)。准确率是最直观的分类性能指标，但在类别不平衡时可能产生误导，需要结合精确率、召回率等指标综合评估。

### 精确率 (Precision)

英文：Precision

缩写：Prec

释义：精确率是预测为正类的样本中真正例的比例：TP/(TP+FP)。精确率衡量模型预测正类的可信度，高精确率表示假阳性较少。在垃圾邮件检测等场景中，精确率是关键指标。

### 召回率 (Recall)

英文：Recall

别名：灵敏度 (Sensitivity)、真阳性率 (True Positive Rate)

缩写：Rec

释义：召回率是真实正类样本中被正确预测的比例：TP/(TP+FN)。召回率衡量模型捕获正类样本的能力，高召回率表示假阴性较少。在疾病诊断等场景中，召回率是关键指标。

### F1分数 (F1 Score)

英文：F1 Score

释义：F1分数是精确率和召回率的调和平均：2×Precision×Recall/(Precision+Recall)。F1分数综合考虑精确率和召回率，在二者不平衡时比准确率更能反映模型真实性能。Fβ分数允许调整对精确率和召回率的权重偏好。

### 特异性 (Specificity)

英文：Specificity

别名：真阴性率 (True Negative Rate)

缩写：Spec

释义：特异性是真实负类样本中被正确预测的比例：TN/(TN+FP)。特异性与召回率类似但针对负类，共同构成完整的分类性能描述。在医学检测中，特异性与召回率（灵敏度）同等重要。

### ROC曲线 (Receiver Operating Characteristic Curve)

英文：Receiver Operating Characteristic Curve

缩写：ROC

释义：ROC曲线以假阳性率（FPR）为x轴、真阳性率（TPR）为y轴，展示分类器在不同阈值下的性能。ROC曲线下面积（AUC）是常用的分类性能度量，不受类别不平衡影响。ROC曲线越左上凸，分类器性能越好。

### AUC-ROC (Area Under the ROC Curve)

英文：Area Under the ROC Curve

缩写：AUC

释义：AUC-ROC是ROC曲线下的面积，取值范围为0到1。AUC衡量分类器区分正负样本的能力，0.5表示随机猜测，1.0表示完美分类。AUC对类别不平衡不敏感，是评估分类模型的重要指标，广泛用于推荐系统、风险评估等领域。

### PR曲线 (Precision-Recall Curve)

英文：Precision-Recall Curve

缩写：PR

释义：PR曲线以召回率为x轴、精确率为y轴，展示不同阈值下精确率与召回率的权衡。在类别严重不平衡时，PR曲线比ROC曲线更能反映模型真实性能。AUC-PR是PR曲线下的面积，值越高表示分类器性能越好。

### 网格搜索 (Grid Search)

英文：Grid Search

释义：网格搜索是穷举搜索超参数空间的方法，通过遍历预定义的超参数组合寻找最优配置。网格搜索实现简单、是超参数调优的常用方法，但当超参数维度高或取值范围大时，计算成本呈指数增长。

### 随机搜索 (Random Search)

英文：Random Search

释义：随机搜索在超参数空间中随机采样进行评估。相较于网格搜索，随机搜索在相同预算下能探索更多超参数维度，且高维空间中更高效。实践中随机搜索往往能找到更好的超参数组合。

### 贝叶斯优化 (Bayesian Optimization)

英文：Bayesian Optimization

缩写：BO

释义：贝叶斯优化是一种基于贝叶斯推断的超参数优化方法，构建超参数与性能的后验模型（通常用高斯过程），指导下一轮评估。贝叶斯优化在有限评估次数下能找到更好的超参数组合，广泛用于AutoML和神经架构搜索。

### Early Stopping (早停法)

英文：Early Stopping

释义：Early Stopping是防止过拟合的策略，在验证集性能不再提升时停止训练。监控验证损失，当连续若干轮（如patience=10）未见改善时终止训练，保存验证最佳模型。Early Stopping是深度学习训练的常用技巧。

### 归一化 (Normalization)

英文：Normalization

释义：归一化是将数据特征缩放到特定范围（如[0,1]或[-1,1]）的数据预处理步骤。归一化消除特征量纲影响，加速模型收敛，提高数值稳定性。常见方法包括最小-最大归一化、Z-score标准化等。

### 标准化 (Standardization)

英文：Standardization

别名：Z-score标准化

释义：标准化是将数据转换为均值为0、标准差为1的标准正态分布。标准化保留数据的相对距离和分布形状，适用于大多数机器学习算法。Z-score标准化公式为z=(x-μ)/σ，是神经网络输入预处理的标准方法。

### 特征缩放 (Feature Scaling)

英文：Feature Scaling

释义：特征缩放是调整特征数值范围的处理，使不同特征在相似尺度上参与计算。特征缩放对基于距离的算法（KNN、SVM）和使用梯度的算法（神经网络）尤为重要，决定模型能否有效学习和收敛。

### 特征工程 (Feature Engineering)

英文：Feature Engineering

释义：特征工程是从原始数据中提取、构建、选择有意义特征的过程。高质量特征能显著提升模型性能，特征工程包括特征构建、特征提取、特征选择等步骤。领域专家知识在特征工程中起关键作用。

### 特征选择 (Feature Selection)

英文：Feature Selection

释义：特征选择是从众多特征中选取对任务最有价值的子集。特征选择降低模型复杂度、减少过拟合风险、提高模型可解释性。方法包括过滤法（基于统计指标）、包装法（基于模型性能）、嵌入法（模型内置选择）。

### 特征提取 (Feature Extraction)

英文：Feature Extraction

释义：特征提取是通过变换或映射将原始数据转换为更有效的特征表示。特征提取从数据中学习到更有意义的表示，如PCA降维、自编码器编码、预训练模型提取等。良好的特征提取是很多AI任务成功的关键。

### 主成分分析 (Principal Component Analysis)

英文：Principal Component Analysis

缩写：PCA

释义：PCA是一种无监督降维方法，通过正交变换将相关特征转换为线性无关的主成分。主成分按方差贡献降序排列，前k个主成分保留大部分信息。PCA用于数据可视化、压缩、去噪、特征提取等，是应用最广泛的降维技术。

### 线性判别分析 (Linear Discriminant Analysis)

英文：Linear Discriminant Analysis

缩写：LDA

释义：LDA是一种监督降维方法，寻找使类间方差最大、类内方差最小的投影方向。LDA假设各类别服从高斯分布且协方差相同。LDA用于特征降维、分类预处理、模式识别等，在人脸识别等场景应用广泛。

### 聚类 (Clustering)

英文：Clustering

释义：聚类是将数据样本划分为若干簇的无监督学习任务，使同簇样本相似度高、不同簇样本相似度低。聚类不依赖标签，通过发现数据内在结构进行分组。常见算法包括K-means、层次聚类、DBSCAN、谱聚类等。

### K-means聚类 (K-means Clustering)

英文：K-means Clustering

释义：K-means是最常用的聚类算法，通过迭代将样本分配到最近的质心并更新质心位置。K-means目标是最小化簇内平方和，假设簇为凸形。算法简单高效，但需预先指定K值，对初始中心和离群点敏感。

### 层次聚类 (Hierarchical Clustering)

英文：Hierarchical Clustering

释义：层次聚类构建数据的多层次嵌套聚类结构，形成树状图（dendrogram）。自底向上（凝聚）的层次聚类从每个样本作为独立簇开始，逐步合并；自顶向下（分裂）则相反。层次聚类不需预先指定簇数，能发现不同尺度的聚类结构。

### DBSCAN (基于密度的空间聚类)

英文：Density-Based Spatial Clustering of Applications with Noise

缩写：DBSCAN

释义：DBSCAN是基于密度的聚类算法，将高密度区域的样本聚为簇，低密度区域样本标记为噪声。DBSCAN能发现任意形状的簇，自动确定簇数，对离群点鲁棒，但不擅长处理密度差异大的数据和在高维空间中性能下降。

### 肘部法则 (Elbow Method)

英文：Elbow Method

释义：肘部法则是一种确定聚类簇数K的方法，绘制不同K值对应的聚类代价（如簇内平方和），选择代价下降明显变缓的拐点（肘部）作为最优K。肘部法则直观简单，但在很多实际数据中代价曲线没有明显肘部。

### 轮廓系数 (Silhouette Coefficient)

英文：Silhouette Coefficient

释义：轮廓系数衡量聚类质量，取值范围[-1,1]，值越大表示聚类效果越好。轮廓系数综合考虑样本与同簇的紧密度和与最近异簇的分离度。对于肘部法则不明确的情况，轮廓系数提供额外的聚类质量评估。

### 支持向量机 (Support Vector Machine)

英文：Support Vector Machine

缩写：SVM

释义：SVM是一种二分类模型，通过寻找最大间隔分离超平面将正负样本分开。当数据线性不可分时，SVM使用核函数将样本映射到高维空间实现线性分割。SVM在小样本、非线性、高维数据上表现优秀，广泛应用于分类、回归、异常检测等任务。

### 核函数 (Kernel Function)

英文：Kernel Function

释义：核函数是SVM中实现非线性分类的关键技术，将数据映射到高维核空间，使原本线性不可分的数据线性可分。常用核函数包括线性核、多项式核、高斯径向基函数（RBF）核、Sigmoid核等。核函数的选择影响SVM性能。

### 逻辑回归 (Logistic Regression)

英文：Logistic Regression

缩写：LR

释义：逻辑回归是用于二分类的线性模型，通过Sigmoid函数将线性组合映射到[0,1]表示概率。逻辑回归输出可解释为概率，用于风险评估、点击率预测等。逻辑回归简单高效，是分类问题的重要baseline模型。

### 线性回归 (Linear Regression)

英文：Linear Regression

缩写：LR

释义：线性回归是最基础的回归模型，假设输入特征与输出之间存在线性关系。线性回归通过最小二乘法求解闭式解，形式简单、可解释性强。线性回归是理解更复杂模型的基础，也是回归问题的常用baseline。

### 决策树 (Decision Tree)

英文：Decision Tree

缩写：DT

释义：决策树是一种树形结构的分类或回归模型，通过对特征递归地做二分判断构建模型。决策树易解释、可处理数值和类别特征，但易过拟合。剪枝（预剪枝、后剪枝）和设置最小样本数等策略用于控制过拟合。

### 信息增益 (Information Gain)

英文：Information Gain

缩写：IG

释义：信息增益是决策树分裂时父节点熵与子节点加权平均熵的差值，衡量分裂对不确定性的减少程度。ID3算法使用信息增益选择最优分裂特征。信息增益偏向取值多的特征，C4.5改用信息增益率克服这一偏好。

### 基尼系数 (Gini Coefficient)

英文：Gini Coefficient

缩写：Gini

释义：基尼系数是决策树CART算法使用的impurity measure，衡量数据纯度：Gini=1-Σp²。基尼系数越小表示数据越纯，分裂时选择使加权基尼系数减少最多的特征和切分点。基尼系数计算比熵快，是CART分裂准则的默认选择。

### 随机森林 (Random Forest)

英文：Random Forest

缩写：RF

释义：随机森林是由多棵决策树组成的集成学习模型，通过Bagging和随机特征选择增强多样性。每棵树独立训练，最终预测通过投票（分类）或平均（回归）得到。随机森林抗过拟合能力强，能处理高维数据，提供特征重要性评估。

### 梯度提升树 (Gradient Boosting Decision Tree)

英文：Gradient Boosting Decision Tree

缩写：GBDT

释义：GBDT是一种boosting集成的决策树模型，通过迭代训练决策树拟合前轮预测的残差，逐步减少训练误差。每轮新树学习前面模型的错误，通过累加各树预测得到最终结果。XGBoost、LightGBM、CatBoost是GBDT的高效实现。

### XGBoost (极端梯度提升)

英文：eXtreme Gradient Boosting

缩写：XGBoost

释义：XGBoost是GBDT的高效分布式实现，添加正则化项防止过拟合，支持缺失值自动处理和并行计算。XGBoost在Kaggle等竞赛中表现优异，是表格数据建模的强力工具。XGBoost提供丰富的超参数和凋亡策略用于调优。

### AdaBoost (自适应提升)

英文：Adaptive Boosting

缩写：AdaBoost

释义：AdaBoost是经典的boosting算法，通过逐步增加被错误分类样本的权重，训练多个弱分类器并加权组合。AdaBoost关注难分类样本，提高模型整体性能。AdaBoost理论基础扎实，是理解boosting思想的重要模型。

### 集成学习 (Ensemble Learning)

英文：Ensemble Learning

释义：集成学习通过组合多个学习器提升性能，核心思想是"三个臭皮匠顶个诸葛亮"。集成方法包括Bagging（并行组合独立模型）、Boosting（序列化提升弱模型）、Stacking（堆叠异构模型）。集成学习能有效提升模型的准确率、稳定性和泛化能力。

### 弱分类器 (Weak Classifier)

英文：Weak Classifier

释义：弱分类器是分类性能略优于随机猜测的学习器，如深度受限的决策树桩。Boosting通过组合大量弱分类器构建强分类器，逐步提升整体性能。弱分类器的概念是boosting算法的理论基础。

### 经验风险最小化 (Empirical Risk Minimization)

英文：Empirical Risk Minimization

缩写：ERM

释义：经验风险最小化是机器学习的基本原则，通过最小化训练集上的平均损失（经验风险）来学习模型。ERM是直觉上合理的方法，但在有限样本下可能导致过拟合。结构风险最小化（SRM）通过添加正则化项来平衡经验风险和模型复杂度。

### 奥卡姆剃刀原则 (Occam's Razor)

英文：Occam's Razor

释义：奥卡姆剃刀原则是"如无必要勿增实体"，在机器学习中指在性能相近的模型中选择较简单的那个。简单模型往往有更好的泛化能力。奥卡姆剃刀是模型选择和正则化设计的理论基础。

---

## C. 神经网络基础

### 神经元 (Neuron)

英文：Neuron

别名：节点 (Node)、单元 (Unit)

释义：神经元是神经网络的基本计算单元，接收多个输入信号，通过激活函数产生输出。生物神经元启发了人工神经元模型：x=Σ(wixi)+b，y=f(x)。神经元通过权重连接形成网络，是深度学习模型的基本构建块。

### 人工神经网络 (Artificial Neural Network)

英文：Artificial Neural Network

缩写：ANN

释义：人工神经网络是模拟生物神经系统信息处理机制的计算模型，由相互连接的神经元层组成。ANN通过学习调整连接权重，实现从输入到输出的复杂映射。ANN是深度学习和机器学习的核心模型架构。

### 神经网络 (Neural Network)

英文：Neural Network

缩写：NN

释义：神经网络是由多层神经元组成的学习系统，通过层次化结构实现特征学习和函数逼近。神经网络是通用函数逼近器，能拟合任意复杂度的非线性函数。CNN、RNN、Transformer等都是神经网络的特殊架构。

### 感知机 (Perceptron)

英文：Perceptron

释义：感知机是最简单的神经网络模型，由Rosenblatt于1957年提出。感知机是单个神经元组成的线性二分类模型，通过阈值函数输出0或1。感知机只能处理线性可分问题，是理解更复杂神经网络的起点。

### 多层感知器 (Multilayer Perceptron)

英文：Multilayer Perceptron

缩写：MLP

释义：多层感知器是由输入层、隐藏层、输出层组成的前馈神经网络，包含一个或多个隐藏层。MLP通过非线性激活函数实现非线性分类，能解决XOR等线性不可分问题。MLP是全连接神经网络的基础架构。

### 层 (Layer)

英文：Layer

释义：层是神经网络的基本组织单位，由一组神经元组成。神经网络通常包含输入层（接收数据）、隐藏层（特征变换）、输出层（产生预测）。不同类型的层实现不同的功能，如全连接层、卷积层、池化层等。

### 全连接层 (Fully Connected Layer)

英文：Fully Connected Layer

缩写：FC Layer

别名：密集层 (Dense Layer)

释义：全连接层是神经网络中最常见的层类型，该层中的每个神经元与上一层的所有神经元相连。全连接层实现特征的非线性组合，具有最大的参数量和表示能力，常用于神经网络的最后几层进行分类或回归。

### 卷积层 (Convolutional Layer)

英文：Convolutional Layer

缩写：Conv Layer

释义：卷积层是卷积神经网络（CNN）的核心组件，通过卷积核在输入上滑动进行局部连接的特征提取。卷积层参数共享的特性大幅减少参数量，使其适合处理图像等高维数据，能有效捕捉局部空间相关性。

### 池化层 (Pooling Layer)

英文：Pooling Layer

别名：下采样层 (Subsampling Layer)

释义：池化层通过对输入的局部区域进行聚合来减少特征图的空间尺寸，实现特征不变性和计算效率提升。常见池化方式包括最大池化（取局部最大值）和平均池化（取局部平均值）。池化层常跟在卷积层之后。

### 激活函数 (Activation Function)

英文：Activation Function

释义：激活函数在神经元输出前引入非线性变换，使网络能拟合复杂非线性函数。常用激活函数包括Sigmoid、Tanh、ReLU、Leaky ReLU、ELU等。激活函数的选择影响梯度传播特性和网络性能，是神经网络的关键组件。

### ReLU (线性整流单元)

英文：Rectified Linear Unit

缩写：ReLU

释义：ReLU是深度学习最常用的激活函数，定义为f(x)=max(0,x)。ReLU计算高效、梯度形式简单（x>0时梯度为1），有效缓解梯度消失问题。但ReLU神经元可能"死亡"（永远输出0），可通过Leaky ReLU、PReLU等变体缓解。

### Sigmoid函数 (Sigmoid Function)

英文：Sigmoid Function

释义：Sigmoid函数是S形曲线，定义为f(x)=1/(1+e^-x)，输出范围(0,1)。Sigmoid曾广泛用于二分类输出层和早期神经网络，但存在梯度饱和问题（两端梯度接近零），现已被ReLU取代。

### Tanh函数 (双曲正切函数)

英文：Tanh Function

别名：Hyperbolic Tangent

释义：Tanh函数是双曲正切函数，定义为f(x)=(e^x-e^-x)/(e^x+e^-x)，输出范围(-1,1)。Tanh输出以零为中心，比Sigmoid收敛更快，但同样存在梯度饱和问题，常用于RNN和某些NLP任务。

### Leaky ReLU (泄漏线性整流单元)

英文：Leaky Rectified Linear Unit

缩写：Leaky ReLU

释义：Leaky ReLU是ReLU的变体，定义为f(x)=ax(x<0)+x(x≥0)，其中a是小斜率（如0.01）。Leaky ReLU允许负值有微小梯度，避免ReLU的神经元死亡问题，在某些任务上性能优于ReLU。

### ELU (指数线性单元)

英文：Exponential Linear Unit

缩写：ELU

释义：ELU是ReLU的变体，负值时使用指数函数：f(x)=x(x≥0)+a(e^x-1)(x<0)。ELU输出接近零均值，收敛更快，且在负值区域有软饱和特性，提高对噪声的鲁棒性。ELU计算涉及指数运算，稍慢于ReLU。

### GELU (高斯误差线性单元)

英文：Gaussian Error Linear Unit

缩写：GELU

释义：GELU是通过概率分布（如正态分布的累积分布函数）加权的非线性激活：f(x)=x·Φ(x)。GELU在Transformer架构（如BERT、GPT）中被广泛使用，性能优于ReLU和ELU，但计算稍复杂。

### Softmax函数 (Softmax Function)

英文：Softmax Function

释义：Softmax函数将实数向量转换为概率分布：softmax(xi)=e^xi/Σe^xj。Softmax输出所有类别的概率之和为1，是多分类神经网络输出层的标准激活函数，常与交叉熵损失结合使用。

### 权重 (Weight)

英文：Weight

缩写：W

释义：权重是神经网络中连接两个神经元的参数，决定了输入信号对输出的影响程度。权重在训练过程中通过反向传播算法不断更新优化，是神经网络存储知识的核心载体，决定模型的表示能力。

### 偏置 (Bias)

英文：Bias

缩写：b

释义：偏置是神经元的额外参数，加到加权输入之后、激活函数之前。偏置允许神经元在没有输入时也有激活输出，是神经网络的平移参数。偏置与权重一起通过训练学习，对网络性能有重要影响。

### 偏置项 (Bias Term)

英文：Bias Term

释义：偏置项是神经网络层中独立于输入的可学习参数，用于调整激活函数的输出位置。偏置项使模型能学习输入特征的基础偏移，提高模型的表达能力，是每个神经元都具有的参数。

### 前向传播 (Forward Propagation)

英文：Forward Propagation

别名：前向计算 (Forward Pass)

释义：前向传播是神经网络推理的过程，输入数据从输入层经隐藏层到输出层逐层计算，最终产生预测结果。前向传播每层执行加权求和、偏置添加、激活函数变换，是神经网络产生输出的完整过程。

### 反向传播 (Back Propagation)

英文：Back Propagation

缩写：Backprop

释义：反向传播是训练神经网络的核心算法，通过链式法则计算损失函数对每个参数的梯度。反向传播从输出层向输入层逐层计算梯度，将误差信号反向传递，用于权重的梯度下降更新。反向传播使大规模神经网络的端到端训练成为可能。

### 链式法则 (Chain Rule)

英文：Chain Rule

释义：链式法则是微积分中计算复合函数导数的规则：d/dx[f(g(x))]=f'(g(x))·g'(x)。反向传播算法基于链式法则逐层计算梯度，是深度学习训练的理论基础，使复杂复合函数的梯度计算变得可行。

### 梯度消失 (Gradient Vanishing)

英文：Gradient Vanishing

释义：梯度消失是指深层网络反向传播时，早期层的梯度变得极小（接近零），导致参数几乎无法更新。梯度消失主要由激活函数的饱和特性和多层链式求导造成。解决方案包括使用ReLU激活、残差连接、批归一化、LSTM门控机制等。

### 梯度爆炸 (Gradient Exploding)

英文：Gradient Exploding

释义：梯度爆炸是指深层网络反向传播时，早期层的梯度变得极大（超过数值范围），导致参数更新不稳定甚至发散。梯度爆炸在RNN中尤为常见。解决方案包括梯度裁剪、权重正则化、合适的参数初始化等。

### 梯度裁剪 (Gradient Clipping)

英文：Gradient Clipping

释义：梯度裁剪是将梯度值限制在预设范围内的技术，防止梯度爆炸。常见方法包括按值裁剪（将梯度限制在[-c, c]）和按范数裁剪（将梯度范数限制在c以内）。梯度裁剪是训练RNN和Transformer的重要技巧。

### 权重初始化 (Weight Initialization)

英文：Weight Initialization

释义：权重初始化是训练开始前为神经网络权重赋予初始值的过程。合适的初始化对训练稳定性和收敛速度至关重要。常用初始化方法包括Xavier初始化（适用于Sigmoid/Tanh）、He初始化（适用于ReLU）、预训练初始化等。

### Xavier初始化 (Xavier Initialization)

英文：Xavier Initialization

释义：Xavier初始化根据输入和输出神经元数量自适应调整权重方差，使信号在前向和反向传播时方差保持一致。Xavier初始化公式：W∼N(0,2/(nin+nout))（高斯）或W∼U(-√(6/(nin+nout)),√(6/(nin+nout)))（均匀）。适用于Sigmoid和Tanh激活函数。

### He初始化 (He Initialization)

英文：He Initialization

释义：He初始化是针对ReLU激活函数的权重初始化方法，方差为2/nin（高斯）或相应均匀分布。He初始化考虑到ReLU对负值的截断效应，比Xavier更适合ReLU网络，是深度网络权重的常用初始化方法。

### 正态分布初始化 (Normal Distribution Initialization)

英文：Normal Distribution Initialization

释义：正态分布初始化将权重初始化为均值0、标准差为预设值（如0.01）的高斯分布。简单但不总是最优，过小的标准差导致梯度消失，过大导致梯度爆炸。一般需要配合批量归一化使用。

### 均匀分布初始化 (Uniform Distribution Initialization)

英文：Uniform Distribution Initialization

释义：均匀分布初始化在[-limit, limit]范围内均匀分布采样权重，limit通常与fan_in相关。均匀分布初始化保证权重在初始化时与输入输出规模相适应，是神经网络权重的常用初始化方法。

### 损失函数 (Loss Function)

英文：Loss Function

别名：代价函数 (Cost Function)、目标函数 (Objective Function)

释义：损失函数量化模型预测与真实标签之间的差异，是神经网络训练优化的目标。不同任务使用不同损失函数：均方误差用于回归，交叉熵用于分类，GAN的对抗损失用于生成任务。损失函数的选择影响学习效率和模型收敛行为。

### 标签 (Label)

英文：Label

别名：目标值 (Target)

释义：标签是训练数据中与输入对应的正确答案或期望输出。监督学习中标签指导模型学习正确的输入-输出映射。分类任务中标签是类别索引，回归任务中是连续数值。标签质量直接影响模型性能。

### 批次 (Batch)

英文：Batch

释义：批次是深度学习训练时每次迭代使用的样本集合。Batch Size是批次包含的样本数。小批量（通常32-256）在计算效率和梯度估计准确性间取得平衡，是深度学习训练的标准做法。批次大小影响模型收敛速度和泛化性能。

### 批尺寸 (Batch Size)

英文：Batch Size

释义：批尺寸是每次训练迭代使用的样本数量。常用批尺寸包括16、32、64、128、256等。批尺寸越大梯度估计越准确但显存需求越高；批尺寸越小训练波动越大但可能泛化更好。批尺寸是深度学习的重要超参数。

### 轮次 (Epoch)

英文：Epoch

释义：轮次是遍历整个训练数据集一次的过程。一个epoch包含多个batch的前向计算和反向传播。训练通常需要多个epoch使模型收敛。epoch数量是重要的训练超参数，过多导致过拟合，过少导致欠拟合。

### 迭代 (Iteration)

英文：Iteration

释义：迭代是使用一个batch的数据进行一次前向和反向传播的过程。完成N个iteration意味着处理了N个batch。一个epoch包含的iteration数为总样本数除以batch size。迭代次数是训练进度的基本计数单位。

### 学习率调度 (Learning Rate Scheduling)

英文：Learning Rate Scheduling

释义：学习率调度是在训练过程中动态调整学习率的策略。常见方法包括步进衰减（每N个epoch降低）、指数衰减、余弦退火、warmup等。学习率调度能帮助模型在训练后期更精细地收敛，提高最终性能。

### Dropout

英文：Dropout

释义：Dropout是防止过拟合的正则化技术，在训练时随机将部分神经元的输出置零（通常概率0.1-0.5）。Dropout强制网络学习冗余表示，提高鲁棒性。测试时使用全部神经元并对输出进行比例缩放。Dropout是深度学习最常用的正则化方法之一。

### Dropout率 (Dropout Rate)

英文：Dropout Rate

释义：Dropout率是Dropout时神经元被随机丢弃的概率，通常设置为0.1-0.5。Dropout率过低正则化效果弱，过高导致网络欠拟合。不同层可设置不同Dropout率，通常靠近输入层较低、靠近输出层较高。

### 批归一化 (Batch Normalization)

英文：Batch Normalization

缩写：BN

释义：批归一化是对每个batch的数据进行归一化（均值0方差1），再进行线性变换的技术。批归一化加速收敛、缓解梯度问题，允许更高学习率，有轻微正则化效果。批归一化已成为深度神经网络的重要组成部分。

### 层归一化 (Layer Normalization)

英文：Layer Normalization

缩写：LN

释义：层归一化是对单个样本的所有特征进行归一化，不依赖batch统计量。层归一化适用于RNN、Transformer等变长序列模型，以及batch size较小的场景。层归一化在NLP和Transformer架构中广泛使用。

### 实例归一化 (Instance Normalization)

英文：Instance Normalization

缩写：IN

释义：实例归一化是对单个样本的单个通道进行归一化，常用于风格迁移任务。实例归一化移除内容的全局统计量，保留风格相关的局部统计量，比批归一化更适合生成任务。

### 组归一化 (Group Normalization)

英文：Group Normalization

缩写：GN

释义：组归一化将通道分成若干组，对每组进行归一化。组归一化不依赖batch，在batch size小或可变时性能优于批归一化。组归一化在目标检测、视频分类等batch受限的任务中表现良好。

### 权重衰减 (Weight Decay)

英文：Weight Decay

缩写：WD

释义：权重衰减是在损失函数中添加L2正则化项的技术，促使权重趋向较小的值。权重衰减防止过拟合、提高训练稳定性。在大多数优化器（如SGD、Adam）中权重衰减是独立于学习率的超参数，需要配合调试。

### L1正则化 (L1 Regularization)

英文：L1 Regularization

释义：L1正则化在损失函数中添加权重绝对值之和项，产生稀疏权重。L1正则化可用于特征选择、模型压缩、可解释性增强。相比L2正则化，L1正则化产生的解更稀疏，是压缩感知和稀疏编码的理论基础。

### L2正则化 (L2 Regularization)

英文：L2 Regularization

别名：权重衰减 (Weight Decay)、岭回归 (Ridge Regression)

释义：L2正则化在损失函数中添加权重平方和项，惩罚大的权重。L2正则化使权重趋向小而分散的值，提高模型鲁棒性和泛化能力。L2正则化是最常用的正则化方法，与权重衰减等价。

### 弹性网络 (Elastic Net)

英文：Elastic Net

释义：弹性网络结合L1和L2正则化，在损失函数中同时添加两项：λ1||W||1+λ2||W||²2。弹性网络兼具L1的稀疏性和L2的稳定性，适合高维数据特征选择，能自动处理特征间的多重共线性问题。

### Skip Connection

英文：Skip Connection

别名：残差连接 (Residual Connection)、捷径连接 (Shortcut Connection)

释义：Skip Connection是将浅层输出直接连接到更深层输出的技术，是ResNet的核心组件。Skip Connection缓解梯度消失，使极深网络的有效训练成为可能，同时计算效率高，已成为现代深度神经网络的重要组件。

### 残差连接 (Residual Connection)

英文：Residual Connection

释义：残差连接通过引入恒等映射使深层网络更容易学习。ResNet中残差块的输出为F(x)+x，其中F是要学习的残差。残差连接使梯度直接流向浅层，让极深网络（如1000+层）的训练变得稳定高效。

### 短期记忆 (Short-term Memory)

英文：Short-term Memory

释义：短期记忆是信息暂时存储和快速访问的能力，在神经网络中对应隐藏状态或缓存信息。LSTM和GRU通过门控机制选择性保留或遗忘短期记忆，处理序列数据中的时序依赖关系。

### 长期记忆 (Long-term Memory)

英文：Long-term Memory

释义：长期记忆是信息持久存储的能力，在神经网络中对应权重参数或外部知识库。RNN的隐藏状态可视为短期记忆，而预训练模型学到的知识可视为长期记忆。注意力机制允许模型灵活访问长期记忆。

### 状态向量 (State Vector)

英文：State Vector

释义：状态向量是RNN、LSTM等网络在时刻t的隐藏状态编码，包含了网络对历史的记忆。状态向量随输入不断更新，是序列建模的核心。状态向量质量直接影响模型对长程依赖的捕捉能力。

### 隐状态 (Hidden State)

英文：Hidden State

缩写：h

释义：隐状态是RNN、LSTM等网络内部维持的、随时间更新的状态变量。隐状态编码了到当前时刻的所有历史信息，用于产生输出或预测。隐状态是递归神经网络记忆机制的核心。

### 记忆单元 (Memory Cell)

英文：Memory Cell

释义：记忆单元是LSTM网络中专门设计用于长期信息存储的组件。记忆单元通过遗忘门、输入门、输出门三个门控机制控制信息的写入、保持和读取，有效解决标准RNN的梯度消失问题，能捕捉长程依赖。

### 门控循环单元 (Gated Recurrent Unit)

英文：Gated Recurrent Unit

缩写：GRU

释义：GRU是LSTM的轻量级变体，通过更新门和重置门两个门控机制控制信息流。GRU比LSTM参数更少、计算更高效，在某些任务上性能与LSTM相当。GRU结合了LSTM的门控思想和RNN的简洁结构。

### 长短期记忆网络 (Long Short-Term Memory)

英文：Long Short-Term Memory

缩写：LSTM

释义：LSTM是1997年提出的特殊RNN架构，通过引入记忆单元和门控机制解决标准RNN的梯度消失问题。LSTM能有效捕捉长程依赖，广泛应用于NLP、语音识别、时间序列预测等序列建模任务。

### 循环神经网络 (Recurrent Neural Network)

英文：Recurrent Neural Network

缩写：RNN

释义：循环神经网络是一类具有内部状态（隐藏状态）的神经网络，通过时间展开处理序列数据。RNN共享参数，能处理可变长序列，在NLP、语音识别、时间序列等任务中广泛应用。标准RNN存在梯度消失问题，LSTM和GRU是其重要改进变体。

---

## D. 深度学习模型

### 卷积神经网络 (Convolutional Neural Network)

英文：Convolutional Neural Network

缩写：CNN

释义：卷积神经网络是一类包含卷积运算的前馈神经网络，专门设计用于处理具有网格结构的数据。CNN通过局部连接和权重共享大幅减少参数量，对图像、视频等数据具有强大的特征提取能力，是计算机视觉领域的基础模型。

### 循环神经网络 (Recurrent Neural Network)

缩写：RNN

释义：循环神经网络是一类具有环状结构的神经网络，能处理序列数据。RNN通过隐藏状态传递历史信息，理论上能捕捉任意长度的依赖关系，但实际受梯度消失限制。LSTM、GRU是RNN的重要改进变体。

### 长短期记忆网络 (Long Short-Term Memory)

缩写：LSTM

释义：长短期记忆网络是一种特殊的RNN架构，通过引入记忆单元和三个门控机制（遗忘门、输入门、输出门）解决长期依赖问题。LSTM能选择性地记忆或遗忘信息，在自然语言处理、语音识别等领域取得巨大成功。

### 门控循环单元 (Gated Recurrent Unit)

缩写：GRU

释义：门控循环单元是LSTM的简化版本，只使用两个门（更新门、重置门），参数量更少。GRU在许多序列任务上性能与LSTM相当，计算效率更高。GRU结合了LSTM的门控思想和RNN的简洁结构。

### 自编码器 (Autoencoder)

英文：Autoencoder

缩写：AE

释义：自编码器是一种无监督学习的神经网络，通过编码器将输入压缩为低维表示，再通过解码器重建原始输入。自编码器用于降维、特征学习、去噪、异常检测等。变分自编码器（VAE）是生成模型的重要分支。

### 变分自编码器 (Variational Autoencoder)

英文：Variational Autoencoder

缩写：VAE

释义：变分自编码器是一种生成模型，通过假设潜在变量服从某种分布（通常是高斯分布）来学习数据的生成过程。VAE使用重参数化技巧实现端到端训练，广泛应用于图像生成、风格迁移、数据增强等任务。

### 去噪自编码器 (Denoising Autoencoder)

英文：Denoising Autoencoder

缩写：DAE

释义：去噪自编码器在标准自编码器基础上，对输入添加噪声，训练网络重建原始无噪输入。去噪自编码器学习更鲁棒的特征表示，提高模型对噪声和扰动的抵抗力，是自编码器的重要变体。

### 稀疏自编码器 (Sparse Autoencoder)

英文：Sparse Autoencoder

缩写：SAE

释义：稀疏自编码器在损失函数中添加稀疏惩罚项，限制隐层单元的平均激活量，使大部分神经元处于"关闭"状态。稀疏自编码器能学习有意义的特征表示，可用于特征学习和数据去噪。

### 生成对抗网络 (Generative Adversarial Network)

英文：Generative Adversarial Network

缩写：GAN

释义：生成对抗网络由生成器和判别器两个网络组成，通过对抗训练学习数据分布。生成器试图生成逼真的假样本，判别器试图区分真实样本和生成样本。GAN是强大的生成模型，在图像生成、风格迁移、数据增强等领域应用广泛。

### 深度卷积生成对抗网络 (Deep Convolutional GAN)

英文：Deep Convolutional GAN

缩写：DCGAN

释义：DCGAN是将卷积网络引入GAN的生成器和判别器的架构。DCGAN使用批归一化、去除池化层、使用转置卷积上采样等技巧，提高了GAN训练的稳定性和生成图像的质量，是GAN的重要里程碑。

### 条件生成对抗网络 (Conditional GAN)

英文：Conditional GAN

缩写：cGAN

释义：条件生成对抗网络在生成器和判别器上同时加入条件信息（如类别标签、文本描述），指导生成过程。cGAN使生成过程可控，能生成指定类别的样本，在图像到图像翻译、文本到图像生成等任务中广泛应用。

### WGAN (Wasserstein GAN)

英文：Wasserstein GAN

缩写：WGAN

释义：Wasserstein GAN使用Wasserstein距离（Earth Mover距离）替代JS散度作为判别器的损失函数，解决了传统GAN训练不稳定、模式崩溃等问题。WGAN无需训练判别器到最优，梯度信号更稳定，是GAN训练的重要改进。

### 风格迁移 (Style Transfer)

英文：Style Transfer

释义：风格迁移是将一幅图像的内容与另一幅图像的风格结合的技术，通过预训练CNN提取内容和风格特征。风格迁移分为基于优化的方法和基于前馈网络的方法。艺术风格迁移、照片风格化等是其典型应用。

### 超分辨率 (Super Resolution)

英文：Super Resolution

缩写：SR

释义：超分辨率是将低分辨率图像重建为高分辨率图像的技术。深度学习方法（如SRCNN、SRGAN、ESRGAN）通过学习低分辨率与高分辨率图像之间的映射，在图像放大、监控视频增强、老照片修复等领域应用广泛。

### 图像分割 (Image Segmentation)

英文：Image Segmentation

释义：图像分割是将图像划分为多个有意义的区域的技术，包括语义分割（像素级分类）和实例分割（区分同类不同个体）。U-Net、Mask R-CNN等深度学习模型在医学图像分割、自动驾驶等场景中发挥重要作用。

### U-Net

英文：U-Net

释义：U-Net是一种用于图像分割的卷积网络架构，具有对称的编码器-解码器结构，中间有skip connections。U-Net最初用于医学图像分割，以少量数据实现精确分割，现广泛应用于各种图像分割任务。

### 语义分割 (Semantic Segmentation)

英文：Semantic Segmentation

释义：语义分割是对图像中每个像素进行分类，确定其属于哪个类别（人、车、树等），不区分同类不同个体。全卷积网络（FCN）、DeepLab、U-Net等是语义分割的代表性模型。

### 实例分割 (Instance Segmentation)

英文：Instance Segmentation

释义：实例分割既要识别图像中所有对象类别，又要区分同类不同个体（个体分割）。Mask R-CNN在Faster R-CNN基础上添加分割分支，是实例分割的经典方法，在目标检测和图像分割领域有重要应用。

### 目标检测 (Object Detection)

英文：Object Detection

释义：目标检测是定位图像中所有感兴趣目标的位置和类别的技术。目标检测算法分为两阶段（如R-CNN系列）和单阶段（如YOLO、SSD）方法。目标检测是计算机视觉的核心任务，广泛应用于自动驾驶、视频监控、机器人视觉等领域。

### R-CNN

英文：Region-based Convolutional Neural Network

缩写：R-CNN

释义：R-CNN是将CNN应用于目标检测的开创性方法，使用选择性搜索提取候选区域，CNN提取特征，SVM分类。R-CNN开创了深度学习目标检测的先河，但速度慢。后续Fast R-CNN、Faster R-CNN逐步改进成为主流。

### Faster R-CNN

英文：Faster R-CNN

释义：Faster R-CNN是R-CNN系列的里程碑工作，引入区域提议网络（RPN）实现端到端的目标检测。Faster R-CNN由RPN和检测网络共享卷积特征，兼顾准确率和速度，是两阶段检测器的代表性架构。

### YOLO (You Only Look Once)

英文：You Only Look Once

缩写：YOLO

释义：YOLO是一种单阶段目标检测算法，将检测任务作为回归问题，直接从图像预测边界框和类别概率。YOLO速度极快，适合实时检测场景。YOLO已发展多代（YOLOv3/v4/v5/v7等），在性能和速度间取得良好平衡。

### SSD (单步多框检测器)

英文：Single Shot MultiBox Detector

缩写：SSD

释义：SSD是一种单阶段目标检测器，在不同层级的特征图上预测不同尺度的目标。SSD结合了YOLO的回归思想和Faster R-CNN的锚框机制，在保持较高速度的同时获得良好的检测精度，是目标检测的重要架构。

### 锚框 (Anchor Box)

英文：Anchor Box

别名：先验框 (Prior Box)

释义：锚框是预定义的具有特定尺寸和长宽比的边界框，是目标检测算法的基础。检测器在锚框基础上预测位置偏移量和尺寸缩放量，简化了检测问题。锚框的设计（如尺寸、数量、长宽比）影响检测器性能。

### 非极大值抑制 (Non-Maximum Suppression)

英文：Non-Maximum Suppression

缩写：NMS

释义：非极大值抑制是目标检测后处理算法，用于去除重复检测框。对同一目标的多个重叠检测，保留得分最高的框，删除其他框。NMS通过IoU阈值筛选，在保证检测召回率的同时提高精度。

### 交并比 (Intersection over Union)

英文：Intersection over Union

缩写：IoU

释义：交并比是预测框与真实框交集面积与并集面积的比值，取值0到1。IoU是目标检测评估的核心指标，用于判断检测是否正确、指导NMS、计算AP等。IoU越高表示预测越精确。

### 注意力机制 (Attention Mechanism)

英文：Attention Mechanism

释义：注意力机制是让模型能"关注"输入中最相关部分的技术，模拟人类视觉的注意力选择。注意力机制通过计算query与key的相似度加权value，实现动态的信息选择。注意力机制是Transformer的核心组件，革新了NLP和计算机视觉领域。

### 自注意力 (Self-Attention)

英文：Self-Attention

别名：内部注意力 (Intra-Attention)

释义：自注意力是输入序列内部各位置之间的注意力计算，捕捉序列内部的依赖关系。自注意力允许序列任意位置直接交互，不受距离限制，是Transformer架构的核心，比RNN更适合并行计算。

### 多头注意力 (Multi-Head Attention)

英文：Multi-Head Attention

缩写：MHA

释义：多头注意力将注意力机制并行执行多次，每次使用不同的query/key/value投影，捕捉不同类型的依赖关系。多头注意力使模型能在不同子空间学习关注不同类型的信息，是Transformer性能卓越的关键。

### Transformer

英文：Transformer

释义：Transformer是一种基于注意力机制的模型架构，完全摒弃循环和卷积，仅使用自注意力和前馈网络。Transformer并行计算效率高，能有效捕捉长程依赖，是NLP领域的主流架构，并成功扩展到计算机视觉、语音等领域。

### 编码器-解码器架构 (Encoder-Decoder Architecture)

英文：Encoder-Decoder Architecture

释义：编码器-解码器是序列到序列学习的经典架构，编码器将输入序列编码为上下文向量，解码器根据上下文向量生成输出序列。Transformer、LSTM编码器-解码器都属于此架构，广泛应用于机器翻译、文本生成、语音合成等任务。

### 编码器 (Encoder)

英文：Encoder

释义：编码器将输入数据（如词元、图像块、音频片段）映射到连续向量表示的空间。编码器捕获输入的语义和结构信息，是自编码器、Transformer、编码器-解码器等架构的核心组件。预训练编码器（如BERT）学到的表示可迁移到下游任务。

### 解码器 (Decoder)

英文：Decoder

释义：解码器根据编码器输出和已生成内容逐步生成目标序列。解码器在每个时间步预测下一个输出，常用自回归方式生成。解码器是生成模型（如GPT）、机器翻译、语音合成等任务的核心组件。

### BERT (双向编码器表示)

英文：Bidirectional Encoder Representations from Transformers

缩写：BERT

释义：BERT是基于Transformer编码器的预训练语言模型，使用掩码语言建模（MLM）和下一句预测（NSP）两个任务进行预训练。BERT的双向编码器设计使其能同时学习左右上下文，产生深层语境相关表示，在NLP各项任务中取得突破性进展。

### GPT (生成式预训练变换器)

英文：Generative Pre-trained Transformer

缩写：GPT

释义：GPT是基于Transformer解码器的生成式预训练模型，使用语言建模任务进行预训练。GPT采用自回归生成方式，从左到右预测下一个词。GPT系列（GPT-2、GPT-3、GPT-4）在文本生成、语言理解等任务上展现强大能力，推动了大语言模型的发展。

### T5 (文本到文本转换变换器)

英文：Text-to-Text Transfer Transformer

缩写：T5

释义：T5将所有NLP任务统一建模为文本到文本的转换问题，使用编码器-解码器Transformer架构。T5在大量无标注数据上进行预训练，通过微调适应下游任务。T5的统一框架简化了模型设计，是NLP多任务学习的重要尝试。

### RoBERTa (稳健优化版BERT)

英文：Robustly Optimized BERT

缩写：RoBERTa

释义：RoBERTa是对BERT的优化版本，去除下一句预测任务、使用更大batch和更多数据、增加训练步数。RoBERTa在保持BERT架构基础上通过优化训练策略提升性能，在多个NLP基准上超越BERT，是重要的预训练语言模型。

### XLNet (置换语言建模)

英文：XLNet

释义：XLNet是一种基于置换语言建模的预训练方法，克服了BERT掩码标记不适用于生成任务的缺点。XLNet使用Transformer-XL作为基础架构，能捕捉更长依赖，在长文档理解等任务上有优势。

### ELMo (深度语境化词表示)

英文：Embeddings from Language Models

缩写：ELMo

释义：ELMo是一种基于双层双向LSTM的词表示方法，能根据上下文动态产生词义。ELMo的创新在于预训练-微调两阶段范式，为BERT等模型奠定了基础。ELMo在2018年提出时刷新了多项NLP基准。

### Word2Vec

英文：Word to Vector

缩写：Word2Vec

释义：Word2Vec是最早的词嵌入方法之一，包含CBOW和Skip-gram两种模型。Word2Vec通过浅层神经网络学习词的分布式表示，捕获语义相似性和类比关系。Word2Vec开启了词嵌入时代，是NLP深度学习的重要里程碑。

### GloVe (全局词向量)

英文：Global Vectors for Word Representation

缩写：GloVe

释义：GloVe是一种结合全局矩阵分解和局部上下文窗口的词嵌入方法。GloVe通过共现矩阵的比率学习词向量，兼具全局统计信息和局部上下文表示，在词类比和相似度任务上表现优秀。

### 嵌入层 (Embedding Layer)

英文：Embedding Layer

释义：嵌入层是将离散符号（如词、物品ID）映射为连续稠密向量的神经网络层。嵌入层通过学习得到，捕获语义相似性，是NLP、推荐系统等领域的基础组件。嵌入向量维度是重要超参数，影响模型容量和效率。

### ResNet (残差网络)

英文：Residual Network

缩写：ResNet

释义：ResNet是由微软研究院提出的深度卷积网络，通过残差连接解决深层网络训练困难的问题。ResNet允许梯度直接回传，使训练1000+层网络成为可能。ResNet在ImageNet、COCO等竞赛中取得最优性能，成为计算机视觉的基础backbone。

### VGG网络 (Visual Geometry Group Network)

英文：VGG Network

缩写：VGG

释义：VGG是2014年ImageNet竞赛的亚军网络，以结构简洁著称。VGG使用3×3小卷积核替代大卷积核，增加网络深度（16-19层），证明了深度对网络性能的重要性。VGG简单有效的设计影响深远，其变体至今仍是常用的backbone网络。

### Inception网络 (Inception Network)

英文：Inception Network

释义：Inception网络是Google提出的卷积网络架构，使用多尺度并行卷积（Inception模块）捕获不同粒度的特征。Inception v1到v4不断改进，引入批归一化、残差连接等技巧。Inception网络对高效精确的网络设计有重要贡献。

### MobileNet (移动网络)

英文：MobileNet

释义：MobileNet是专为移动端和边缘设备设计的轻量级卷积网络，使用深度可分离卷积大幅减少计算量和参数量。MobileNet系列（v1/v2/v3）在保持较高精度同时实现低延迟，是移动端部署的重要选择。

### EfficientNet (高效网络)

英文：EfficientNet

释义：EfficientNet是通过神经架构搜索（NAS）同时优化网络深度、宽度和分辨率的复合缩放方法。EfficientNet在精度和效率间取得极佳平衡，EfficientNet-B0到B7一系列模型覆盖不同资源约束场景。

### DenseNet (密集连接网络)

英文：Dense Network

缩写：DenseNet

释义：DenseNet通过将每层与所有后续层直接连接，实现特征重用和梯度传播。DenseNet中每层接收前面所有层的特征图作为输入，参数效率高、泛化能力强。密集连接缓解梯度消失，增强特征传递。

### 神经架构搜索 (Neural Architecture Search)

英文：Neural Architecture Search

缩写：NAS

释义：神经架构搜索是自动设计神经网络架构的技术，通过搜索策略在搜索空间中探索最优结构。NAS使用强化学习、进化算法、梯度方法等搜索策略，能发现人类设计之外的新型架构，是AutoML的核心组成部分。

### 神经网络的深度 (Depth)

英文：Depth

释义：神经网络的深度指网络层数，是网络表示能力的重要指标。深层网络能学习更复杂的函数，但训练难度增加。ResNet通过残差连接允许训练超深网络（1000+层），深度与宽度、参数量共同决定网络容量。

### 神经网络的宽度 (Width)

英文：Width

释义：神经网络的宽度指每层的神经元数或通道数，决定网络的容量和并行度。宽网络擅长捕获细节特征，窄网络则倾向于学习更抽象的表示。Wide & Deep网络结合宽浅层和深窄层的优点。

### 感受野 (Receptive Field)

英文：Receptive Field

释义：感受野是卷积网络中输出特征图上的一个位置所对应的输入图像的区域大小。感受野越大，神经元能感知更大的输入范围，对理解图像全局信息至关重要。空洞卷积、池化等操作可增大感受野。

### 空洞卷积 (Dilated Convolution)

英文：Dilated Convolution

别名：膨胀卷积 (Atrous Convolution)

释义：空洞卷积在卷积核元素之间插入空洞（零值），在不增加参数量的情况下增大感受野。空洞卷积用于图像分割、语音合成等任务，允许网络捕获多尺度信息，避免因池化导致的分辨率损失。

### 转置卷积 (Transposed Convolution)

英文：Transposed Convolution

别名：反卷积 (Deconvolution)、分数步长卷积 (Fractionally Strided Convolution)

释义：转置卷积是卷积的逆操作，用于上采样或生成任务。转置卷积不是卷积的真正逆运算，而是恢复空间维度。转置卷积在图像分割（U-Net）、GAN生成图像、decoder等架构中广泛使用。

### 深度可分离卷积 (Depthwise Separable Convolution)

英文：Depthwise Separable Convolution

释义：深度可分离卷积将标准卷积分解为逐通道卷积和逐点卷积两步，大幅减少参数量和计算量。MobileNet使用深度可分离卷积实现轻量化，在延迟和精度间取得良好平衡，成为移动端网络的标准组件。

### 注意力图 (Attention Map)

英文：Attention Map

释义：注意力图是可视化注意力权重分布的热力图，展示模型在处理输入时关注的位置。注意力图用于理解模型决策、验证注意力机制是否有效。在图像任务中，注意力图能定位关键区域；在NLP中，能展示词间依赖关系。

---

## E. AI芯片与硬件

### 中央处理器 (Central Processing Unit)

英文：Central Processing Unit

缩写：CPU

释义：CPU是通用处理器，负责计算机的指令执行和任务调度。CPU具有复杂的控制单元和缓存层次，擅长顺序执行和复杂逻辑运算。在AI系统中，CPU负责系统管理、数据预处理、模型协调等任务，是协调整体计算的"大脑"。

### 图形处理器 (Graphics Processing Unit)

英文：Graphics Processing Unit

缩写：GPU

释义：GPU是专为并行计算设计的处理器，拥有大量简单计算单元（CUDA core/Shader），适合处理大规模数据并行任务。GPU的并行计算能力使其成为深度学习训练的主要硬件加速器，相比CPU提供数十到数百倍的加速比。

### CUDA核心 (CUDA Core)

英文：CUDA Core

释义：CUDA核心是NVIDIA GPU中的基本计算单元，执行浮点运算。每个CUDA核心一次只能处理一个线程的指令，但通过大规模并行（数千个核心）实现高吞吐量。深度学习训练充分利用CUDA核心进行矩阵乘法和卷积运算。

### Tensor Core (张量核)

英文：Tensor Core

释义：Tensor Core是NVIDIA Volta及以后架构GPU中的专用深度学习计算单元，专门加速矩阵运算。Tensor Core支持混合精度计算（FP16输入、FP32累加），在深度学习训练和推理中提供显著加速，是现代GPU的杀手级特性。

### 流式多处理器 (Streaming Multiprocessor)

英文：Streaming Multiprocessor

缩写：SM

释义：SM是GPU的基本执行单元，包含多个CUDA核心、Tensor Core、寄存器、共享内存等。一个GPU由多个SM组成，每个SM能同时执行多个线程块。SM通过SIMT（单指令多线程）模式执行，为深度学习提供高效的并行计算基础。

### SIMT (单指令多线程)

英文：Single Instruction Multiple Threads

缩写：SIMT

释义：SIMT是GPU的并行执行模型，多个线程以锁步方式执行相同的指令，但处理不同的数据。SIMT允许GPU高效执行数据并行任务，线程被组织成warp（通常32线程）在同一SM上调度执行，是GPU高吞吐量的关键。

### Warp (线程束)

英文：Warp

释义：Warp是GPU中最基本的调度单位，通常包含32个线程。这些线程执行相同的指令但处理不同数据，以SIMT方式运行。Warp内的线程共享指令fetch/dispatch，当线程分支时会发生warp divergence，降低并行效率。

### Warp Divergence (线程束分歧)

英文：Warp Divergence

释义：Warp Divergence发生在同一warp内的线程执行不同分支路径时（如if-else），导致部分线程空闲等待。Warp Divergence降低GPU利用率，是GPU编程中需要避免的性能瓶颈。合理的数据组织和分支设计可减少分歧。

### 内存带宽 (Memory Bandwidth)

英文：Memory Bandwidth

释义：内存带宽是数据从内存传输到计算单元的最大速率，通常以GB/s为单位。内存带宽是AI芯片的关键指标，高带宽（如HBM）能满足深度学习对数据的高速访问需求。计算密集型任务需要高内存带宽支撑。

### 高带宽内存 (High Bandwidth Memory)

英文：High Bandwidth Memory

缩写：HBM

释义：HBM是一种先进的3D堆叠内存技术，通过硅通孔（TSV）连接DRAM芯片，提供极高的内存带宽。HBM相比传统GDDR内存具有更高带宽、更低功耗、更小尺寸的特点，是AI芯片和高端GPU的标配。

### HBM2 (第二代高带宽内存)

英文：HBM2

释义：HBM2是HBM的第二代标准，提供更高带宽（可达256GB/s）、更大容量（单栈8GB）、更高速率（每引脚2.4Gbps）。HBM2广泛用于NVIDIA V100/A100/H100 GPU和AI加速器，为深度学习训练提供充裕的数据访问带宽。

### GDDR (图形双倍数据率内存)

英文：Graphics Double Data Rate

缩写：GDDR

释义：GDDR是专为图形处理设计的内存类型，相比DDR有更高频率和带宽。GDDR在游戏GPU中广泛使用，GDDR6X通过PAM4调制技术进一步提升带宽。GDDR成本低于HBM，是消费级GPU的内存选择。

### DDR (双倍数据率内存)

英文：Double Data Rate

缩写：DDR

释义：DDR是最常见的系统内存类型，已发展多代（DDR3/DDR4/DDR5）。DDR内存容量大、成本低，用于CPU和集成GPU的系统内存。在AI推理场景，特别是边缘设备，DDR常作为主要内存使用，其带宽是性能瓶颈之一。

### LPDDR (低功耗双倍数据率内存)

英文：Low Power DDR

缩写：LPDDR

释义：LPDDR是移动设备的低功耗内存标准，已发展LPDDR4/4X/5等版本。LPDDR功耗极低，适合移动端和边缘AI设备。LPDDR5提供更高带宽，满足手机、平板等设备上的AI推理需求，如手机端的AI加速芯片。

### PCIe (外围组件互连高速总线)

英文：Peripheral Component Interconnect Express

缩写：PCIe

释义：PCIe是连接CPU与外部设备的高速总线标准。AI加速卡（如GPU）通过PCIe与CPU通信，PCIe带宽影响数据传输效率。PCIe 4.0提供64GB/s带宽，PCIe 5.0翻倍至128GB/s，是AI系统互连的重要接口。

### NVLink

英文：NVLink

释义：NVLink是NVIDIA开发的高速GPU互连技术，提供GPU间直接通信，带宽远超PCIe。NVLink连接多个GPU组成高速集群，用于大模型训练。NVIDIA DGX系统使用NVLink实现GPU间高速数据交换，是大模型训练的硬件基础。

### 张量处理器 (Tensor Processing Unit)

英文：Tensor Processing Unit

缩写：TPU

释义：TPU是Google专为深度学习设计的专用集成电路（ASIC），第一代于2016年发布。TPU使用脉动阵列加速矩阵运算，擅长推理任务。TPU v2/v3支持训练，通过定制化设计提供极高的性能功耗比，是云端AI加速的重要选择。

### 神经网络处理器 (Neural Network Processor)

英文：Neural Network Processor

缩写：NPU

释义：NPU是专门用于神经网络运算的处理器，常见于移动端和边缘设备。NPU针对深度学习算子优化，实现低功耗推理。华为麒麟芯片的达芬奇架构、苹果A/M系列芯片的Neural Engine都是NPU的代表。

### 现场可编程门阵列 (Field-Programmable Gate Array)

英文：Field-Programmable Gate Array

缩写：FPGA

释义：FPGA是可通过编程配置实现任意数字逻辑的可编程芯片。FPGA具有并行度高、延迟低、能效好的特点，适合AI推理和特定场景加速。FPGA可编程性使其能灵活支持各类AI模型，是AI硬件的重要组成部分。

### 专用集成电路 (Application-Specific Integrated Circuit)

英文：Application-Specific Integrated Circuit

缩写：ASIC

释义：ASIC是为特定应用定制的芯片，如TPU、神经网络加速器等。ASIC针对特定任务优化，实现极高的性能和能效。ASIC缺点是设计周期长、成本高、功能不可更改。AI芯片领域ASIC是实现量产和部署的主流方案。

### DSA (领域专用架构)

英文：Domain-Specific Architecture

缩写：DSA

释义：领域专用架构是为特定应用领域优化的芯片设计理念，在特定领域提供极致性能。DSA针对AI、networking、graphics等领域的workloads定制，是后摩尔时代芯片发展的重要方向。TPU是DSA的典型代表，DSA正在引领AI芯片设计新浪潮。

### 片上系统 (System on Chip)

英文：System on Chip

缩写：SoC

释义：片上系统是将CPU、GPU、NPU、内存、I/O等组件集成在单一芯片上的技术。SoC提供完整计算能力同时保持低功耗，是移动设备、智能手表、边缘AI的核心。苹果M系列、高通骁龙、华为麒麟都是先进的SoC产品。

### 加速器 (Accelerator)

英文：Accelerator

释义：加速器是专门用于加速特定计算任务的硬件，如GPU、NPU、TPU、FPGA等。加速器相比通用CPU能提供数量级的性能提升，是AI系统实现高性能计算的关键组件。现代AI系统通常采用CPU+加速器的异构架构。

### 异构计算 (Heterogeneous Computing)

英文：Heterogeneous Computing

释义：异构计算使用不同类型处理单元（如CPU+GPU、CPU+FPGA）协同完成计算任务。不同硬件擅长不同计算范式，异构计算通过发挥各自优势实现整体性能最优化。AI系统广泛采用CPU+GPU/NPU的异构架构。

### 访存密集型 (Memory-Bound)

英文：Memory-Bound

释义：访存密集型是指计算性能受内存带宽或容量限制，而非计算单元限制的工作负载。AI推理通常是访存密集型的，因为权重数据量大、计算密度相对较低。优化访存密集型任务需要提高内存带宽或使用模型压缩技术。

### 计算密集型 (Compute-Bound)

英文：Compute-Bound

释义：计算密集型是指计算性能受计算单元限制，而非内存访问限制的工作负载。AI训练特别是大batch训练通常是计算密集型的，此时GPU算力是瓶颈。优化计算密集型任务需要提高峰值算力或增加并行度。

### 算力 (Computational Power)

英文：Computational Power

别名：峰值算力 (Peak Performance)

释义：算力是硬件每秒能完成的计算量，通常以TFLOPS（万亿次浮点运算/秒）或TOPS（万亿次操作/秒）为单位。算力是评估AI芯片性能的核心指标，但实际性能还受内存带宽、软件栈效率等因素影响。

### 功耗 (Power Consumption)

英文：Power Consumption

缩写：TDP

释义：功耗是芯片在典型工作负载下消耗的电能，以瓦特为单位。AI芯片的功耗从移动端的毫瓦到数据中心的千瓦不等。功耗直接影响运行成本和散热设计，是AI系统部署的重要考量因素。

### 能效比 (Power Efficiency)

英文：Power Efficiency

释义：能效比是芯片每瓦功耗能提供的算力，通常以TFLOPS/W或TOPS/W为单位。能效比是评估AI芯片的重要指标，尤其对移动端和边缘设备。能效比的提升是AI芯片发展的关键驱动力。

### 热设计功耗 (Thermal Design Power)

英文：Thermal Design Power

缩写：TDP

释义：热设计功耗是芯片在典型工作负载下散发的热量，是散热系统设计的依据。TDP越高需要更强的散热能力，如风扇、水冷等。数据中心部署AI服务器时，TDP直接影响机房散热方案和运营成本。

### 内存层次结构 (Memory Hierarchy)

英文：Memory Hierarchy

释义：内存层次结构是计算机系统中不同容量和速度的存储层次，从快到慢包括寄存器、缓存、主存、外存。高带宽内存（HBM）、GDDR、DDR构成AI芯片的内存层次，合理利用能提高数据访问效率。

### 缓存 (Cache)

英文：Cache

释义：缓存是位于CPU/GPU核心附近的高速小容量存储器，存储最近使用的数据和指令。缓存命中时访问延迟极低，未命中时需访问下一级存储。AI芯片具有多级缓存（L1/L2/L3），优化缓存利用率是性能优化的重要方向。

### 共享内存 (Shared Memory)

英文：Shared Memory

释义：共享内存是GPU SM内多个线程共享的高速存储器，延迟极低。共享内存用于线程块内数据共享和临时变量存储，是GPU编程的重要优化点。合理使用共享内存能减少全局内存访问，提高执行效率。

### 统一计算设备架构 (CUDA)

英文：Compute Unified Device Architecture

缩写：CUDA

释义：CUDA是NVIDIA开发的并行计算平台和编程模型，允许使用C/C++等语言在NVIDIA GPU上进行通用计算。CUDA提供丰富的库（cuBLAS、cuDNN）和工具，是深度学习训练和推理的主要软件平台。

### cuDNN (CUDA深度神经网络库)

英文：CUDA Deep Neural Network Library

缩写：cuDNN

释义：cuDNN是NVIDIA提供的深度学习算子优化库，实现卷积、池化、归一化、RNN等常用操作的GPU加速版本。cuDNN由NVIDIA针对其GPU架构手工优化，PyTorch、TensorFlow等框架底层调用cuDNN实现高效计算。

### ROCm (Radeon开放计算平台)

英文：Radeon Open Compute Platform

缩写：ROCm

释义：ROCm是AMD主导的开源并行计算平台，提供类似CUDA的GPU计算能力。ROCm支持多种编程语言和框架，为AMD GPU提供深度学习支持。ROCm的开源特性使其在超算和数据中心有广泛应用。

### OpenCL (开放计算语言)

英文：Open Computing Language

缩写：OpenCL

释义：OpenCL是由Khronos组织维护的开放标准，用于编写跨CPU、GPU、FPGA等异构平台的并行程序。OpenCL是设备无关的并行编程标准，但相比CUDA效率略低，在某些AI芯片上作为备选编程接口。

### 矩阵乘加运算 (Multiply-Accumulate)

英文：Multiply-Accumulate

缩写：MAC

释义：矩阵乘加运算是AI芯片的核心运算，包含乘法和累加两步：a+=b×c。深度学习的矩阵运算（如全连接层、卷积）可分解为大量MAC操作。AI芯片通过专用MAC单元（如Tensor Core）加速此类运算。

### 脉动阵列 (Systolic Array)

英文：Systolic Array

释义：脉动阵列是一种特殊的硬件架构，数据以节拍方式在阵列中流动，每个处理单元执行简单运算后将结果传递给下一个单元。脉动阵列实现高并行度、低功耗的矩阵运算，是TPU等AI芯片的核心计算单元。

### 指令集架构 (Instruction Set Architecture)

英文：Instruction Set Architecture

缩写：ISA

释义：指令集架构是软件与硬件之间的接口定义，包括指令格式、寄存器、内存模型等。x86、ARM、RISC-V是常见的指令集。RISC-V作为开源指令集在AI芯片领域越来越受关注，为定制AI处理器提供灵活基础。

### ARM架构 (ARM Architecture)

英文：ARM Architecture

释义：ARM是低功耗处理器架构，广泛用于移动设备和嵌入式系统。ARM处理器占据手机市场绝大多数份额，其生态成熟、功耗效率优异。随着移动AI的兴起，ARM NPU成为端侧AI推理的重要平台。

### RISC-V架构 (RISC-V Architecture)

英文：RISC-V Architecture

释义：RISC-V是基于精简指令集的开源处理器架构，允许自由定制和扩展。RISC-V没有授权限制，适合AI芯片差异化设计。国内外多家AI芯片初创公司选择RISC-V作为核心处理器架构。

### 硅通孔 (Through-Silicon Via)

英文：Through-Silicon Via

缩写：TSV

释义：硅通孔是3D堆叠芯片的垂直互连技术，通过在硅片上穿孔实现不同层之间的电气连接。TSV是HBM等3D堆叠内存的关键使能技术，使高带宽、低延迟的芯片堆叠成为可能。

### 封装 (Packaging)

英文：Packaging

释义：封装是将芯片与其他元件组装成完整系统的工艺技术。先进封装（如2.5D、3D封装）将多个芯片或存储器堆叠集成，提升互连带宽、降低延迟。台积电CoWoS、Intel EMIB是知名先进封装技术。

### 2.5D封装 (2.5D Packaging)

英文：2.5D Packaging

释义：2.5D封装将多个芯片并排放置在硅中介层上，通过中介层的布线实现互连。2.5D封装提供高于传统封装的互连密度，用于GPU与HBM的集成（如NVIDIA的Volta架构）。2.5D封装是平衡性能、成本、良率的折中方案。

### 3D封装 (3D Packaging)

英文：3D Packaging

释义：3D封装通过硅通孔技术将多个芯片垂直堆叠，进一步提升互连密度和集成度。3D封装使存储与计算更紧密地结合，是AI芯片封装的发展方向。3D堆叠DRAM、3D V-Cache是3D封装在AI硬件中的典型应用。

---

## F. AI编译器

### 编译器 (Compiler)

英文：Compiler

释义：编译器是将高级编程语言或中间表示转换为目标机器代码的程序。传统编译器（如GCC）将C/C++代码编译为机器码，AI编译器将高层神经网络模型编译为优化过的底层代码。编译器是连接算法与硬件的桥梁。

### AI编译器 (AI Compiler)

英文：AI Compiler

释义：AI编译器是专门用于优化和编译神经网络模型的编译器，将模型描述转换为硬件可执行代码。AI编译器进行算子融合、布局转换、内存优化、代码生成等优化，显著提升推理性能。TVM、XLA、Glow是知名的AI编译器。

### 中间表示 (Intermediate Representation)

英文：Intermediate Representation

缩写：IR

释义：中间表示是编译器中用于表示源代码或模型的结构化数据。AI编译器通常使用多层IR（如High-Level IR用于图优化、Low-Level IR用于指令调度）。好的IR设计能有效支持各类优化，是AI编译器的核心。

### 高层中间表示 (High-Level IR)

英文：High-Level IR

缩写：HLIR

释义：高层中间表示是对神经网络模型的高层抽象，关注算子之间的数据流和依赖关系。高层IR用于图优化、算子融合、死代码消除等优化，与硬件无关。高层IR的设计影响编译器上层的优化能力。

### 底层中间表示 (Low-Level IR)

英文：Low-Level IR

缩写：LLIR

释义：底层中间表示是接近目标硬件的低层表示，关注具体的指令选择、寄存器分配、指令调度等。底层IR与硬件紧密相关，是生成高效代码的关键。多层IR设计使AI编译器能同时进行架构无关和架构相关的优化。

### 算子 (Operator)

英文：Operator

缩写：Op

释义：算子是神经网络中的基本计算单元，如矩阵乘法（MatMul）、卷积（Conv2D）、ReLU激活等。算子是神经网络模型的基本构建块，AI编译器对算子进行调度和优化，最终生成硬件可执行代码。

### Kernel (内核函数)

英文：Kernel

释义：Kernel是GPU等硬件上执行实际计算的函数。AI编译器将算子映射为一个或多个kernel，每个kernel负责在特定硬件上完成特定计算。Kernel的编写和调度直接影响硬件利用率，是性能优化的关键。

### 算子融合 (Operator Fusion)

英文：Operator Fusion

别名：内核融合 (Kernel Fusion)

释义：算子融合是将多个相邻算子合并为单一kernel的优化技术。算子融合减少中间结果的内存访问，降低访存开销，是AI编译器最重要的优化之一。例如将卷积、ReLU、BatchNorm融合为单个卷积算子。

### 常量折叠 (Constant Folding)

英文：Constant Folding

释义：常量折叠是在编译时计算具有常量输入的表达式的优化技术。例如将两个常量矩阵的乘法结果直接计算出来，避免运行时计算。常量折叠减少运行时计算量，是编译器的基础优化之一。

### 代数化简 (Algebraic Simplification)

英文：Algebraic Simplification

释义：代数化简是使用代数规则简化表达式的优化技术，如识别并消除冗余操作、合并同类项等。代数化简在编译器前端和后端都有应用，能减少计算量和访存次数。

### 死代码消除 (Dead Code Elimination)

英文：Dead Code Elimination

缩写：DCE

释义：死代码消除是删除不会被执行或不会影响结果的代码的优化技术。死代码包括未使用的变量、未调用的函数、永真/永假条件分支等。死代码消除减少不必要的计算，提高代码效率。

### 循环展开 (Loop Unrolling)

英文：Loop Unrolling

释义：循环展开是增加循环体代码量、减少循环迭代次数的优化技术。通过复制循环体多次，减少循环控制开销，增加指令级并行。循环展开增加代码大小但提高执行效率，是编译器常用的优化手段。

### 循环分块 (Loop Tiling)

英文：Loop Tiling

别名：循环分片 (Loop Blocking)

释义：循环分块是将大循环划分为小块的优化技术，改善数据局部性和缓存利用率。通过将循环访问的数据块放入缓存，减少缓存未命中。循环分块对矩阵乘法等内存密集型操作尤为重要。

### 自动调度 (Auto Scheduling)

英文：Auto Scheduling

释义：自动调度是让编译器自动决定循环调度策略的技术，无需人工指定优化参数。TVM的AutoTVM使用学习-based方法搜索最优调度，是AI编译器自动优化的重要方向，降低手工优化的负担。

### 自动调优 (Auto Tuning)

英文：Auto Tuning

释义：自动调优是通过系统化搜索找到最优参数配置的技术，如kernel的块大小、展开因子等。TVM、Tensor Comprehensions等工具提供自动调优能力，在特定硬件上搜索最优执行参数，是AI编译器性能优化的重要组成部分。

### 布局转换 (Layout Transformation)

英文：Layout Transformation

释义：布局转换是改变数据在内存中的存储方式的优化，如NCHW到NHWC的转换。不同硬件对数据布局有不同偏好，布局转换使数据符合硬件最佳访问模式。布局转换是跨平台部署时的重要优化。

### 图优化 (Graph Optimization)

英文：Graph Optimization

释义：图优化是在计算图层面进行的优化，包括算子融合、常量折叠、公共子表达式消除等。图优化在高层IR上进行，与具体硬件无关，主要目的是减少算子数量和中间内存占用。

### 内存规划 (Memory Planning)

英文：Memory Planning

释义：内存规划是在编译时决定中间结果存储位置和生命周期的优化。内存规划通过内存池复用、就地操作等手段减少内存分配开销和峰值内存使用，对内存受限的边缘设备尤为重要。

### 量化 (Quantization)

英文：Quantization

释义：量化是将高精度数据（如FP32）转换为低精度表示（如INT8）的技术。量化减少内存占用和计算量，加速推理并降低功耗。量化需要平衡精度损失，是模型压缩和部署的重要技术。

### 训练后量化 (Post-Training Quantization)

英文：Post-Training Quantization

缩写：PTQ

释义：训练后量化是在模型训练完成后进行量化的技术，无需重新训练。PTQ使用校准数据集确定量化参数，实现简单、速度快。PTQ是模型部署的常用方法，但可能带来一定精度损失。

### 量化感知训练 (Quantization-Aware Training)

英文：Quantization-Aware Training

缩写：QAT

释义：量化感知训练是在训练过程中模拟量化效果，使模型适应低精度表示。QAT通过直通估计器（STE）处理梯度，通常能获得比PTQ更好的精度。QAT是量化精度敏感任务的首选方法。

### 代码生成 (Code Generation)

英文：Code Generation

缩写：CodeGen

释义：代码生成是编译器将中间表示转换为目标代码的过程。AI编译器的代码生成包括掌柜选择、寄存器分配、指令调度等步骤，生成GPU、CPU、加速器等目标硬件的可执行代码。

### LLVM

英文：LLVM

释义：LLVM是一套模块化的编译器基础设施，提供优化和代码生成框架。LLVM的核心是中间表示（LLVM IR）和可扩展的优化通道。TVM、TensorFlow XLA等AI编译器使用LLVM作为代码生成的后端。

### TVM

英文：TVM

释义：TVM是Apache开源的深度学习编译器栈，将模型编译为优化过的硬件代码。TVM支持多种硬件后端（CPU、GPU、FPGA、ASIC），提供自动调度和量化工具。TVM是AI编译领域的重要开源项目，源自华盛顿大学PLSE实验室。

### 张量理解 (Tensor Comprehensions)

英文：Tensor Comprehensions

缩写：TC

释义：Tensor Comprehensions是Facebook开源的神经网络编译工具，使用简化的声明式语言描述张量运算，自动生成GPU代码。TC使用polyhedral模型进行循环优化，支持算子融合和自动调优。

### XLA (加速线性代数)

英文：Accelerated Linear Algebra

缩写：XLA

释义：XLA是Google开发的深度学习编译器，是TensorFlow的官方编译器后端。XLA将TensorFlow计算图编译为优化的机器码，支持JIT和AOT编译，提供算子融合、内存规划等优化，显著加速TensorFlow模型的执行。

### Glow (图形化深度学习编译器)

英文：Glow

释义：Glow是Facebook（Meta）开源的深度学习编译器，使用基于SSA的低级IR。Glow支持CPU、GPU等多种后端，提供量化工具和模型加载器。Glow的设计强调可扩展性和模块化，是AI编译器研究的重要平台。

### MLIR (多级中间表示)

英文：Multi-Level Intermediate Representation

缩写：MLIR

释义：MLIR是Google提出的新型编译器基础设施，支持多层IR用于表示不同抽象级别。MLIR统一了高层图优化和底层代码生成，降低了领域专用编译器的开发难度。MLIR正在成为AI编译器的新标准基础设施。

### 计算图 (Computational Graph)

英文：Computational Graph

别名：数据流图 (Data Flow Graph)

缩写：DFG

释义：计算图是表示神经网络模型中算子之间数据依赖关系的图结构。计算图节点表示算子，边表示数据流向。计算图是AI编译器进行图优化的基础，也用于理解模型结构、进行反向传播等。

### 静态形状 (Static Shape)

英文：Static Shape

释义：静态形状是指编译时确定的输入形状，模型所有张量的形状在运行前已知。静态形状允许编译器进行更多优化，如常量折叠、连续内存分配等。静态形状推理是AI编译器的重要功能。

### 动态形状 (Dynamic Shape)

英文：Dynamic Shape

释义：动态形状是指运行时才确定的输入形状，如NLP任务中的变长序列。动态形状给编译器优化带来挑战，需要运行时调度和内存管理。MLIR等新型编译器对动态形状有更好的支持。

### Just-In-Time编译 (JIT Compilation)

英文：Just-In-Time Compilation

缩写：JIT

释义：即时编译是在运行时将代码编译为本地机器码的技术。JIT允许根据实际输入形状和硬件情况进行优化。PyTorch的TorchScript、TensorFlow的XLA都使用JIT技术加速模型执行。

### Ahead-Of-Time编译 (AOT Compilation)

英文：Ahead-Of-Time Compilation

缩写：AOT

释义：预编译是在程序运行前将代码编译为可执行文件的技术。AOT编译将模型完全编译好，部署时直接运行，延迟最低。AOT适合资源受限的边缘设备，避免了运行时的编译开销。

### 图替换 (Graph Substitution)

英文：Graph Substitution

别名：图重写 (Graph Rewrite)

释义：图替换是用更高效的算子组合替换图中子图的优化技术。例如将两个矩阵加一个标量的操作融合。AI编译器通过定义替换规则系统地进行图优化，实现架构无关的性能提升。

### Schedule (调度)

英文：Schedule

释义：Schedule在TVM等编译器中指定了循环执行的策略，包括分块、展开、向量化、并行化等。Schedule决定了代码的具体执行方式，是影响性能的关键。Auto Scheduling自动搜索最优Schedule。

### 自动向量化 (Auto Vectorization)

英文：Auto Vectorization

释义：自动向量化是编译器自动将标量操作转换为向量操作的优化，使一条指令处理多个数据。SIMD指令（如AVX、NEON）是向量化硬件支持。自动向量化降低手工优化的负担，但效果取决于数据布局和访问模式。

### 并行化 (Parallelization)

英文：Parallelization

释义：并行化是将计算分解为多个可同时执行的部分的技术。编译器自动识别的并行化包括循环级并行（多核）、指令级并行（流水线）、数据级并行（SIMD）。并行化是充分利用硬件算力的关键技术。

### 算子调度 (Operator Scheduling)

英文：Operator Scheduling

释义：算子调度是决定算子执行顺序的优化，在保证依赖关系正确的前提下最大化并行度。算子调度是AI编译器后端的重要功能，影响指令流水线和硬件利用率。

### 寄存器分配 (Register Allocation)

英文：Register Allocation

缩写：RA

释义：寄存器分配是将临时变量映射到硬件寄存器的过程，是编译器后端的核心任务。有效的寄存器分配减少内存访问，提高执行效率。图的染色算法和线性扫描是常用的寄存器分配算法。

### 调度空间 (Schedule Space)

英文：Schedule Space

释义：调度空间是所有可能调度配置的集合，Auto Scheduling在此空间中进行搜索。调度空间的大小影响搜索难度和最终效果。好的调度空间设计需要平衡表达力和搜索效率。

### Polyhedral模型 (Polyhedral Model)

英文：Polyhedral Model

释义：多面体模型是用于表示和优化嵌套循环的数学框架，将循环迭代空间建模为多面体。多面体模型支持精确的依赖分析和优化变换，如平铺、融合、分布等。Tensor Comprehensions等编译器使用多面体模型进行循环优化。

### 优化遍 (Optimization Pass)

英文：Optimization Pass

释义：优化遍是编译器中执行的单个优化步骤，如常量折叠、死代码消除、算子融合等。AI编译器通过顺序执行多个优化遍逐步提升代码质量。每个优化遍是模块化的，便于扩展和组合。

### 目标代码 (Target Code)

英文：Target Code

别名：目标程序 (Target Program)

释义：目标代码是编译器输出的、能在目标硬件上执行的代码。AI编译器的目标代码可以是GPU CUDA/PTX代码、CPU汇编代码、或硬件专用的指令序列。目标代码的质量是衡量编译器优劣的关键指标。

---

## G. 推理与部署

### 推理 (Inference)

英文：Inference

别名：推断、预测 (Prediction)

释义：推理是使用训练好的模型对新数据进行预测的过程。推理通常在部署阶段进行，要求低延迟、高吞吐。推理优化（如量化、剪枝、编译器优化）是AI系统部署的核心技术。

### 推理引擎 (Inference Engine)

英文：Inference Engine

释义：推理引擎是执行模型推理的软件运行时，负责加载模型、执行算子、返回预测结果。推理引擎如TensorRT、ONNX Runtime、Triton等针对推理场景优化，提供高性能的模型部署解决方案。

### 训练 (Training)

英文：Training

释义：训练是使用数据调整模型参数使损失函数最小化的过程。训练通常需要大量数据、计算资源和时间。训练阶段生成的模型权重用于后续的推理部署。训练和推理是AI系统的两个主要阶段。

### 模型压缩 (Model Compression)

英文：Model Compression

释义：模型压缩是减小模型大小、降低计算量的技术，包括量化、剪枝、蒸馏、知识迁移等方法。模型压缩使大模型能在资源受限的设备上运行，是边缘AI部署的关键技术。

### 模型转换 (Model Conversion)

英文：Model Conversion

释义：模型转换是将模型从一种格式转换为另一种格式的过程。例如将PyTorch模型转换为ONNX格式，或将ONNX转换为TensorRT格式。模型转换是AI部署流程中的重要环节。

### 模型序列化 (Model Serialization)

英文：Model Serialization

释义：模型序列化是将训练好的模型保存为文件格式的过程，以便后续加载和推理。常见的模型序列化格式包括PyTorch的.pt、TensorFlow的.pb、ONNX的.onnx等。序列化需保存模型结构和权重参数。

### 模型反序列化 (Model Deserialization)

英文：Model Deserialization

释义：模型反序列化是从文件加载模型的过程，反序列化后的模型可用于推理。反序列化需要解析文件格式、恢复模型结构、加载权重参数。反序列化效率影响模型加载时间。

### 模型部署 (Model Deployment)

英文：Model Deployment

释义：模型部署是将训练好的模型上线提供服务的过程。模型部署需要考虑推理性能、资源消耗、服务稳定性、监控运维等因素。云端部署、边缘部署、端侧部署是主要的部署形态。

### 云端推理 (Cloud Inference)

英文：Cloud Inference

别名：云端部署 (Cloud Deployment)

释义：云端推理是在远程服务器上进行模型推理的服务模式。云端推理算力资源丰富，适合大模型和高并发场景。云端推理通过API提供服务，延迟受网络影响。AWS SageMaker、阿里云PAI是常见的云端推理平台。

### 边缘推理 (Edge Inference)

英文：Edge Inference

别名：边缘部署 (Edge Deployment)

释义：边缘推理是在靠近数据源的设备（如手机、摄像头、IoT设备）上进行本地推理。边缘推理减少网络延迟、保护数据隐私、降低云端负载。边缘推理需要在受限硬件上实现高效执行。

### 端侧推理 (On-Device Inference)

英文：On-Device Inference

别名：端侧部署

释义：端侧推理是在用户设备（如手机、平板）本地执行模型推理。端侧推理实现实时响应、保护隐私、离线可用。端侧推理依赖模型压缩和硬件加速，是移动AI的主要形态。

### 推理延迟 (Inference Latency)

英文：Inference Latency

释义：推理延迟是模型从接收输入到返回结果的时间。延迟是实时应用的关键指标。降低延迟的方法包括模型压缩、硬件加速、批处理优化、编译器优化等。

### 推理吞吐量 (Inference Throughput)

英文：Inference Throughput

释义：推理吞吐量是单位时间内处理的样本数量，通常以QPS或FPS衡量。高吞吐量需要充分利用硬件并行能力。吞吐量和延迟通常需要权衡，如通过batch size调节。

### 批处理 (Batch Processing)

英文：Batch Processing

释义：批处理是合并多个样本一次推理的技术，提高硬件利用率和吞吐量。批处理减少kernel启动开销、提高GPU利用率，但增加单次推理延迟。合适的batch size需要在吞吐量和延迟间权衡。

### 动态批处理 (Dynamic Batching)

英文：Dynamic Batching

释义：动态批处理是运行时动态决定batch大小的技术，根据当前负载自动调整。动态批处理提高系统资源利用率和吞吐量，同时保持可接受的延迟。Triton推理服务器等部署平台支持动态批处理。

### 模型量化 (Model Quantization)

英文：Model Quantization

释义：模型量化是将模型参数和计算从高精度（FP32）转换为低精度（INT8/FP16）的技术。量化减少模型大小、降低内存占用、加速推理。量化需要平衡精度损失，是模型压缩的核心技术。

### INT8量化 (INT8 Quantization)

英文：INT8 Quantization

释义：INT8量化将FP32参数和计算转换为8位整数表示。INT8量化减少75%的内存占用，INT8算力通常是FP32的2-4倍，是推理加速的常用方法。INT8量化需要校准确定量化参数，可能带来精度损失。

### FP16量化 (FP16 Quantization)

英文：FP16 Quantization

释义：FP16量化将FP32转换为半精度浮点FP16。FP16量化简单、精度损失小，通常无需校准。FP16减少50%内存占用和计算量，在支持FP16的GPU（如Tensor Core）上能获得显著加速。

### BF16量化 (BF16 Quantization)

英文：BF16 Quantization

释义：BF16（Brain Float 16）是Google为深度学习设计的16位浮点格式，与FP32相同的指数范围但降低尾数精度。BF16相比FP16动态范围更大，精度损失更小，在AI训练和推理中越来越受欢迎。

### 混合精度 (Mixed Precision)

英文：Mixed Precision

释义：混合精度在模型不同部分使用不同精度，如FP16计算+FP32累加。混合精度结合了不同精度的优势：FP16提供高吞吐、FP32保证精度。Tensor Core等硬件对混合精度有原生支持，是训练加速的标准方法。

### 量化感知训练 (Quantization-Aware Training)

缩写：QAT

释义：量化感知训练是在训练时模拟量化效果，使模型适应低精度表示。QAT通过直通估计器处理梯度，通常能获得比PTQ更好的精度。QAT是量化精度敏感任务的首选方法。

### 训练后量化 (Post-Training Quantization)

缩写：PTQ

释义：训练后量化是在模型训练完成后进行量化，无需原始训练数据。PTQ需要少量校准数据确定量化参数，实现简单。PTQ可能带来一定精度损失，但对大多数应用是可接受的。

### 校准 (Calibration)

英文：Calibration

释义：校准是确定量化参数（scale、zero point）的过程，使用少量代表性数据统计激活值范围。校准方法包括最大绝对值、KL散度、熵等。校准质量影响量化模型的精度，是PTQ的关键步骤。

### 剪枝 (Pruning)

英文：Pruning

释义：剪枝是移除神经网络中不重要的权重或神经元的技术，减少参数量和计算量。剪枝包括非结构化剪枝（移除单个权重）和结构化剪枝（移除整组权重如通道）。剪枝需要平衡压缩率和精度损失。

### 非结构化剪枝 (Unstructured Pruning)

英文：Unstructured Pruning

释义：非结构化剪枝随机移除单个权重，产生稀疏权重矩阵。非结构化剪枝压缩率高，但稀疏格式需要专用硬件支持才能实际加速。常用稀疏格式包括CSR、CSC等。

### 结构化剪枝 (Structured Pruning)

英文：Structured Pruning

释义：结构化剪枝按结构移除权重组，如移除整个神经元、通道、层等。结构化剪枝产生规则结构，无需专用稀疏格式即可获得实际加速。结构化剪枝是工业部署的常用方法。

### 知识蒸馏 (Knowledge Distillation)

英文：Knowledge Distillation

缩写：KD

释义：知识蒸馏是用大模型（teacher）指导小模型（student）训练的技术，小模型学习大模型的输出分布或中间表示。蒸馏产生的小模型能在资源受限设备上高效运行。Hinton等提出的soft labels蒸馏是经典方法。

### 教师-学生网络 (Teacher-Student Network)

英文：Teacher-Student Network

释义：教师-学生网络是知识蒸馏中的双模型架构。教师网络是大而精确的模型，学生网络是小而高效的模型。学生网络同时学习硬标签和教师网络的软标签，获得比直接训练更好的性能。

### 模型结构搜索 (Architecture Search)

英文：Architecture Search

释义：模型结构搜索是自动设计神经网络架构的技术，通过搜索发现最优的网络结构。NAS是其中的代表性方法，使用强化学习、进化算法或梯度方法搜索。模型结构搜索能发现超越人工设计的架构。

### 神经网络架构搜索 (Neural Architecture Search)

缩写：NAS

释义：神经网络架构搜索是AutoML的核心，自动化设计神经网络结构。NAS在定义的搜索空间中评估不同架构的性能，使用搜索策略（强化学习、贝叶斯优化、梯度方法）指导探索。NAS的计算成本很高，但能发现创新架构。

### 网络搜索 (Network Search)

英文：Network Search

释义：网络搜索是在预定义网络族中搜索最优网络配置的技术，常用于超参数优化和网络结构微调。网络搜索比NAS计算量小，是实用的网络优化方法。

### ONNX (开放神经网络交换)

英文：Open Neural Network Exchange

缩写：ONNX

释义：ONNX是Microsoft和Facebook推出的神经网络模型开放格式，定义模型的结构和计算图。ONNX实现不同框架（PyTorch、TensorFlow等）间的模型互转，是AI模型交换和部署的标准格式。

### ONNX Runtime

英文：ONNX Runtime

释义：ONNX Runtime是Microsoft的高性能推理引擎，支持ONNX格式模型的推理。ONNX Runtime针对多平台优化，提供CPU、GPU、Edge等多种后端。ONNX Runtime是跨框架部署的重要工具。

### TorchScript

英文：TorchScript

释义：TorchScript是PyTorch的模型序列化格式，支持跟踪（tracing）和脚本（scripting）两种导出方式。TorchScript将PyTorch模型转换为可序列化的中间表示，用于生产环境部署。TorchScript支持JIT编译执行。

### TensorRT

英文：TensorRT

释义：TensorRT是NVIDIA的高性能推理引擎和优化工具。TensorRT支持ONNX、TensorFlow、PyTorch等格式，提供算子融合、精度校准、内核自动调优等优化。TensorRT是NVIDIA GPU上推理部署的标准选择。

### Triton推理服务器 (Triton Inference Server)

英文：Triton Inference Server

释义：Triton是NVIDIA开源的推理服务框架，支持多模型并发推理、动态批处理、模型版本管理等。Triton支持TensorRT、ONNX Runtime、PyTorch等多种后端，是生产环境部署的重要平台。

### TensorFlow Serving

英文：TensorFlow Serving

释义：TensorFlow Serving是TensorFlow模型的生产级部署框架，支持模型版本管理、热更新、REST/gRPC接口。TensorFlow Serving适合TensorFlow模型的云端部署，提供高性能、高可用的推理服务。

### OpenVINO (可视化推理优化工具)

英文：Open Visual Inference and Neural Network Optimization Toolkit

缩写：OpenVINO

释义：OpenVINO是Intel的推理部署工具链，优化TensorFlow、PyTorch、ONNX等模型在Intel硬件（CPU、GPU、VPU、FPGA）上的推理性能。OpenVINO提供模型优化和运行时，是Intel平台AI部署的标准工具。

### 昇腾推理引擎 (MindSpore Inference Engine)

英文：MindSpore Inference Engine

别名：MindIE

释义：MindIE是华为昇腾（Ascend）AI处理器的推理引擎，支撑MindSpore、PyTorch、TensorFlow等框架的模型部署。MindIE提供算子优化、内存优化、量化加速等能力，是昇腾芯片的推理部署平台。

### AscendCL (昇腾计算语言)

英文：Ascend Computing Language

缩写：AscendCL

释义：AscendCL是华为昇腾芯片的编程接口，提供设备管理、内存管理、模型加载、算子执行等功能。AscendCL是昇腾生态的基础API，开发者通过AscendCL调用昇腾加速能力。

### 边缘设备 (Edge Device)

英文：Edge Device

释义：边缘设备是部署在网络边缘的计算设备，如手机、摄像头、传感器、IoT设备等。边缘设备进行本地AI推理，减少数据传输延迟、保护隐私。边缘设备资源受限，需要模型压缩和硬件加速。

### 端设备 (End Device)

英文：End Device

别名：端侧设备

释义：端设备是最终用户使用的产品设备，如智能手机、自动驾驶汽车、智能音箱等。端设备具备一定的AI推理能力，通过模型压缩和硬件优化实现高效推理。端设备AI是用户体验和隐私保护的重要支撑。

### 模型版本管理 (Model Versioning)

英文：Model Versioning

释义：模型版本管理是跟踪和管理不同版本模型的技术，包括版本命名、回滚、AB测试等。模型版本管理支持生产环境的持续改进和平滑更新，是MLOps的重要环节。

### 模型注册表 (Model Registry)

英文：Model Registry

释义：模型注册表是集中存储和管理模型元数据的系统，包括模型文件、版本、指标、血缘等。模型注册表支持模型发现、共享、部署，是规模化AI系统的基础设施。

### 模型监控 (Model Monitoring)

英文：Model Monitoring

释义：模型监控是跟踪生产环境模型性能的技术，包括延迟、吞吐量、预测分布、漂移检测等。模型监控及时发现模型退化或数据分布变化，是保障AI服务质量的必要手段。

### 模型漂移 (Model Drift)

英文：Model Drift

释义：模型漂移是指模型性能随时间下降的现象，通常由输入数据分布变化引起。漂移检测是模型监控的重要内容，支持自动触发模型重训练。概念漂移和数据漂移是两种主要类型。

### A/B测试 (A/B Testing)

英文：A/B Testing

释义：A/B测试是比较两个模型版本性能的实验方法，将流量分配到不同版本并统计效果指标。A/B测试用于验证模型改进是否有效，是模型发布的重要决策依据。

### 金丝雀发布 (Canary Deployment)

英文：Canary Deployment

释义：金丝雀发布是先将新模型部署到小部分流量，验证无问题后再全量发布的技术。金丝雀发布降低新模型上线风险，支持平滑过渡和快速回滚，是模型部署的常用策略。

### 滚动更新 (Rolling Update)

英文：Rolling Update

释义：滚动更新是逐步用新版本替换旧版本的部署策略，每次只更新部分实例。滚动更新保持服务持续可用，避免中断。滚动更新配合健康检查和回滚机制，实现零停机部署。

### 模型优化 (Model Optimization)

英文：Model Optimization

释义：模型优化是提升模型效率的各种技术，包括量化、剪枝、知识蒸馏、神经网络架构搜索等。模型优化使大模型能在受限环境中高效运行，是AI部署的核心环节。

### 推理优化 (Inference Optimization)

英文：Inference Optimization

释义：推理优化是优化推理过程性能的技术，包括算子融合、内存优化、硬件加速等。推理优化结合模型压缩和编译器优化，实现低延迟、高吞吐的推理服务。

---

## H. 分布式系统

### 分布式训练 (Distributed Training)

英文：Distributed Training

释义：分布式训练是在多个计算设备上并行训练模型的技术，显著缩短大模型训练时间。分布式训练通过数据并行或模型并行扩展到多GPU、多机器，是大模型训练的必备技术。

### 数据并行 (Data Parallelism)

英文：Data Parallelism

缩写：DP

释义：数据并行是将训练数据划分到多个设备，每个设备持有完整模型副本，独立计算梯度后同步更新。数据并行是扩展深度学习训练最常用的方法，通信开销小、实现简单。DDP（DistributedDataParallel）是PyTorch的标准数据并行实现。

### 模型并行 (Model Parallelism)

英文：Model Parallelism

缩写：MP

释义：模型并行是将模型划分到多个设备，每个设备负责部分模型的计算。模型并行用于模型过大无法在单卡存放的场景，如超大NLP模型。模型并行通信密集、实现复杂，是数据并行的补充。

### 流水线并行 (Pipeline Parallelism)

英文：Pipeline Parallelism

缩写：PP

释义：流水线并行将模型按层分组分配到不同设备，数据像流水线一样流经各设备。流水线并行减少设备间通信量，但引入流水线气泡。GPipe、PipeDream是流水线并行的代表性系统。

### 张量并行 (Tensor Parallelism)

英文：Tensor Parallelism

缩写：TP

释义：张量并行将模型的单个算子（如矩阵乘法）划分到多个设备，是更细粒度的模型并行。张量并行如Megatron-LM，对单个算子进行切分通信，适合单节点多GPU的高带宽场景。

### 混合并行 (Hybrid Parallelism)

英文：Hybrid Parallelism

释义：混合并行结合数据并行、模型并行、流水线并行等多种并行策略，充分挖掘大规模集群的计算能力。大模型训练通常使用3D并行（数据+流水线+张量）的混合形式。

### 同步训练 (Synchronous Training)

英文：Synchronous Training

释义：同步训练是所有设备等待所有梯度计算完成后统一更新参数的并行训练方式。同步训练保证与传统单设备训练等价的结果，但会因最慢设备而等待。同步SGD是理论上的最优方法。

### 异步训练 (Asynchronous Training)

英文：Asynchronous Training

释义：异步训练是设备不等其他设备完成就独立更新参数的并行训练方式。异步训练避免等待，但引入梯度陈旧问题。异步训练曾是分布式训练主流，但随着GPU算力提升，同步训练变得更实用。

### 参数服务器 (Parameter Server)

英文：Parameter Server

缩写：PS

释义：参数服务器是分布式训练的系统架构，工作节点负责计算，服务器节点负责存储和更新模型参数。参数服务器支持异步训练曾是主流架构，但已被AllReduce取代。Horovod等框架使用AllReduce代替参数服务器。

### AllReduce

英文：AllReduce

释义：AllReduce是集合通信操作，所有节点数据经过规约操作（如求和、取最大）后每个节点都获得结果。AllReduce用于数据并行中梯度同步，是分布式深度学习训练的核心通信原语。

### AllGather

英文：AllGather

释义：AllGather是收集所有节点数据的集合通信操作，每个节点最终获得所有节点数据的完整副本。AllGather用于模型并行中的张量收集、流水线并行的微批次收集等场景。

### ReduceScatter

英文：ReduceScatter

释义：ReduceScatter是先将所有节点数据进行规约，然后将结果分发到不同节点的通信操作。ReduceScatter是梯度聚合优化的基础操作，能有效减少通信量。

### Broadcast

英文：Broadcast

释义：Broadcast是将一个节点的数据发送到所有其他节点的通信操作。Broadcast用于广播模型参数或配置，是分布式训练的基础通信模式。

### 点对点通信 (Point-to-Point Communication)

英文：Point-to-Point Communication

缩写：P2P

释义：点对点通信是两个节点之间的直接数据交换，发送方和接收方明确。点对点通信是构建集合通信的基础，如Send/Recv操作。点对点通信延迟低，但编程复杂度高。

### 集合通信 (Collective Communication)

英文：Collective Communication

释义：集合通信是涉及一组节点中所有成员的通信操作，如AllReduce、AllGather、Broadcast等。集合通信是分布式深度学习训练的核心，NCCL、OpenMPI等库提供高效的集合通信实现。

### NCCL (NVIDIA集体通信库)

英文：NVIDIA Collective Communications Library

缩写：NCCL

释义：NCCL是NVIDIA提供的GPU间集体通信库，提供高带宽的AllReduce、AllGather等操作。NCCL针对NVIDIA GPU优化，充分利用NVLink、PCIe等高速互连。NCCL是深度学习分布式训练的事实标准。

### GDR (GPU直接RDMA)

英文：GPU Direct RDMA

缩写：GDR

释义：GPU Direct RDMA是允许GPU内存直接访问远程机器内存的技术，绕过CPU和操作系统。GDR减少数据传输延迟和CPU开销，是多GPU和多机器分布式训练的重要加速技术。

### RDMA (远程直接内存访问)

英文：Remote Direct Memory Access

缩写：RDMA

释义：RDMA是允许直接访问远程机器内存的高速网络技术，绕过操作系统和数据复制。RDMA提供低延迟、高带宽的通信，是数据中心分布式训练的关键技术。RoCE、iWARP是RDMA的网络协议。

### RoCE (基于融合以太网的RDMA)

英文：RDMA over Converged Ethernet

缩写：RoCE

释义：RoCE是将RDMA运行在以太网上的协议，允许在标准以太网上实现高速RDMA通信。RoCE v2是当前主流版本，在AI训练集群中广泛部署，是数据中心高性能网络的解决方案。

### IB (InfiniBand)

英文：InfiniBand

缩写：IB

释义：InfiniBand是一种高速网络互连技术，提供高带宽（200-400Gbps）和低延迟。InfiniBand在超算和AI训练集群中使用，NVIDIA收购Mellanox后推出Quantum系列InfiniBand交换机。InfiniBand是HPC和AI训练的顶级网络选择。

### 100GbE (百吉比特以太网)

英文：100 Gigabit Ethernet

缩写：100GbE

释义：100GbE是100Gbps速率的以太网标准，是数据中心的主流高速网络。100GbE提供足够的带宽支持中等规模的分布式训练，比InfiniBand成本更低、兼容性更好。

### 400GbE (四百吉比特以太网)

英文：400 Gigabit Ethernet

缩写：400GbE

释义：400GbE是400Gbps速率的以太网标准，是当前最快的以太网。400GbE满足大模型训练对网络带宽的极端需求，是AI数据中心的下一代网络升级目标。

### 通信量 (Communication Volume)

英文：Communication Volume

释义：通信量是分布式训练中节点间传输的数据总量。通信量直接影响训练效率，是选择并行策略的重要指标。数据并行通信量与模型大小无关，模型并行通信量与切分方式密切相关。

### 通信带宽 (Communication Bandwidth)

英文：Communication Bandwidth

释义：通信带宽是网络传输数据的能力，通常以GB/s或Gbps为单位。高通信带宽是分布式训练的基础，瓶颈带宽限制分布式加速比。网络拓扑设计对实际带宽利用率影响重大。

### 延迟 (Latency)

英文：Latency

释义：延迟是数据从发出到接收的时间，是网络性能的另一关键指标。集合通信的延迟影响同步效率。RDMA提供极低延迟，适合对延迟敏感的AI训练场景。

### 加速比 (Speedup)

英文：Speedup

释义：加速比是并行训练相比单设备的性能提升倍数：Speedup=T_single/T_parallel。理想加速比等于设备数，实际因通信开销和并行效率而小于设备数。加速比是评估分布式训练效果的核心指标。

### 并行效率 (Parallel Efficiency)

英文：Parallel Efficiency

缩写：PE

释义：并行效率是加速比与设备数的比值：E=Speedup/N。理想并行效率为1，实际通常小于1。并行效率衡量并行化效果，效率过低说明通信或同步成为瓶颈。

### 扩展性 (Scalability)

英文：Scalability

释义：扩展性是系统随资源增加性能提升的能力。强扩展性是固定问题规模增加设备，强扩展性考验绝对性能；弱扩展性是问题规模随设备增加，强扩展性考验效率保持。

### Horovod

英文：Horovod

释义：Horovod是Uber开源的分布式深度学习训练框架，使用AllReduce进行梯度同步。Horovod支持TensorFlow、PyTorch、Keras，提供简单的分布式训练API。Horovod曾是Facebook、Google之外的第三大分布式训练框架。

### DeepSpeed

英文：DeepSpeed

释义：DeepSpeed是Microsoft开源的深度学习优化库，专注于大模型训练。DeepSpeed提供ZeRO优化器（显存优化）、流水线并行、异步I/O等能力。DeepSpeed使训练万亿参数模型成为可能，是大模型训练的重要框架。

### Megatron-LM

英文：Megatron-LM

释义：Megatron-LM是NVIDIA开发的大模型训练框架，实现高效的模型并行（张量并行）。Megatron-LM配合DeepSpeed使用，是训练超大规模NLP模型（如GPT-3）的主力框架。

### ZeRO (零冗余优化器)

英文：Zero Redundancy Optimizer

缩写：ZeRO

释义：ZeRO是DeepSpeed的显存优化技术，通过分片（Sharding）消除数据并行中的冗余存储。ZeRO分为Stage 1（优化器状态分片）、Stage 2（梯度分片）、Stage 3（参数分片），显著降低显存占用。

### ZeRO-1/2/3

英文：ZeRO-1/2/3

释义：ZeRO-1只分片优化器状态，ZeRO-2分片优化器状态和梯度，ZeRO-3分片所有状态（优化器状态、梯度、参数）。Stage越高显存节省越多，但通信量也增加。实际使用需要权衡显存和通信。

### 梯度累积 (Gradient Accumulation)

英文：Gradient Accumulation

释义：梯度累积是分多个小批次累加梯度，模拟大批次训练的技术。梯度累积减少显存占用，允许使用更大模型。梯度累积是训练大模型的标准技巧，与各种并行技术兼容。

### 混合精度训练 (Mixed Precision Training)

英文：Mixed Precision Training

释义：混合精度训练在模型不同部分使用不同精度，通常为FP16计算+FP32累加。混合精度训练利用Tensor Core加速、减少显存占用。PyTorch的AMP（Automatic Mixed Precision）是常用实现。

### 自动混合精度 (Automatic Mixed Precision)

英文：Automatic Mixed Precision

缩写：AMP

释义：AMP是PyTorch自动管理混合精度的功能，根据操作自动选择FP16或FP32。AMP降低使用混合精度的门槛，避免手工精度管理。AMP配合分布式训练进一步提升效率。

### 梯度检查点 (Gradient Checkpointing)

英文：Gradient Checkpointing

别名：激活重计算 (Activation Recomputation)

释义：梯度检查点是在反向传播时重新计算激活，而非存储全部激活，以显存换计算的技术。梯度检查点将显存占用减少到O(sqrt(n))，代价是约20-30%的额外计算。是训练大模型的必备技术。

### 显存优化 (Memory Optimization)

英文：Memory Optimization

释义：显存优化是降低训练显存占用的技术，包括梯度累积、梯度检查点、ZeRO、混合精度等。显存优化使更大模型能在有限显存中训练，是大模型训练的关键技术。

### 流水线气泡 (Pipeline Bubble)

英文：Pipeline Bubble

释义：流水线气泡是流水线并行中设备空闲的时间段，因等待前后阶段完成造成。气泡降低流水线效率，流水线越长、阶段越多气泡越严重。1F1B（One-Forward-One-Backward）调度能减少气泡。

### 1F1B调度 (One-Forward-One-Backward Scheduling)

英文：One-Forward-One-Backward Scheduling

释义：1F1B调度是一种流水线并行调度，每个设备交替执行前向和反向，减少流水线气泡。1F1B比GPipe的interleaved调度通信更频繁但气泡更少，是PipeDream等系统的默认调度。

### 微批次 (Micro-batch)

英文：Micro-batch

缩写：Micro-Batch

释义：微批次是流水线并行中从批次划分的小批次。将大batch划分为多个micro-batch，使流水线各阶段能充分忙碌。Micro-batch数量影响流水线效率和显存占用。

### 设备利用率 (Device Utilization)

英文：Device Utilization

释义：设备利用率是GPU等加速器实际计算时间占总时间的比例。低设备利用率说明存在瓶颈（通信、CPU、数据加载）。提高设备利用率是分布式训练优化的核心目标。

### 多节点训练 (Multi-Node Training)

英文：Multi-Node Training

释义：多节点训练是在多个服务器节点上进行分布式训练，节点间通过高速网络互连。多节点训练扩展到更大规模模型和数据集，需要高效的集群管理和通信。

### 弹性训练 (Elastic Training)

英文：Elastic Training

释义：弹性训练是支持节点动态加入退出的训练方式，提高集群利用率和容错性。Elastic Training在云计算环境中尤为重要，能与其他作业共享资源。PyTorch Elastic、Horovod Elastic是弹性训练框架。

### 容错训练 (Fault Tolerant Training)

英文：Fault Tolerant Training

释义：容错训练是节点故障时不中断训练的技术，通过checkpoint恢复状态。容错训练对长时间训练的大模型至关重要，避免故障导致数天训练白费。自动容错是超大规模训练的必备能力。

### 检查点 (Checkpoint)

英文：Checkpoint

释义：检查点是保存模型状态（参数、优化器状态、epoch等）的快照，用于故障恢复和训练中断续。检查点策略影响故障恢复时间和存储开销。大模型训练通常每几个小时打一次checkpoint。

### 弹性平均 (Elastic Averaging)

英文：Elastic Averaging

释义：弹性平均是允许各节点使用不同学习率的分布式优化方法，增加优化的探索性。弹性平均不等待同步，可能加速收敛。弹性平均是弹性训练的理论基础。

### LocalSGD (本地随机梯度下降)

英文：Local Stochastic Gradient Descent

缩写：LocalSGD

释义：LocalSGD是各节点本地多步训练后再同步的并行算法，减少通信频率。LocalSGD在通信带宽受限时特别有用，以稍微降低效率换取通信量大幅减少。

### Gossip算法 (Gossip Algorithm)

英文：Gossip Algorithm

释义：Gossip算法是基于随机邻居通信的分布式算法，通信方式像疾病传播。Gossip用于参数平均、分布式聚合等，无中心节点、天然容错。Gossip适合大规模、高延迟的广域网分布式训练。

### 去中心化训练 (Decentralized Training)

英文：Decentralized Training

释义：去中心化训练是无中心参数服务器的分布式训练方式，节点间对等通信。去中心化训练避免中心瓶颈，提高系统可扩展性。Gossip算法和区块链是去中心化训练的技术基础。

### 分布式数据并行 (Distributed Data Parallel)

英文：Distributed Data Parallel

缩写：DDP

释义：分布式数据并行是PyTorch的多GPU数据并行实现，每个进程持有模型副本。DDP通过AllReduce同步梯度，比DataParallel更高效且支持多节点。DDP是PyTorch分布式训练的主流方式。

### Mesh TensorFlow

英文：Mesh TensorFlow

释义：Mesh TensorFlow是Google的分布式TensorFlow框架，支持数据并行和模型并行的灵活组合。Mesh TensorFlow将计算划分到设备网格，自动处理通信和调度，是早期大规模训练的探索。

### JAX

英文：JAX

释义：JAX是Google的高性能数值计算库，结合NumPy API和自动微分，支持XLA编译执行。JAX支持函数式编程，提供pmap进行数据并行、xmap进行向量并行。JAX在科研领域越来越受欢迎。

### Flax

英文：Flax

释义：Flax是基于JAX的神经网络库，提供灵活的模型构建API。Flax与JAX的函数式风格一致，支持复杂的数据并行和模型并行。Flax是Google推荐的JAX神经网络框架。

### 集合通信后端 (Collective Communication Backend)

英文：Collective Communication Backend

释义：集合通信后端是实现AllReduce等集合通信的软件库。常见后端包括NCCL（NVIDIA GPU）、Gloo（CPU）、UCC（多厂商）等。PyTorch支持选择不同后端，根据硬件和通信模式选用最优实现。

### MPI (消息传递接口)

英文：Message Passing Interface

缩写：MPI

释义：MPI是高性能计算的并行编程标准，定义点对点和集合通信API。MPI是HPC领域的主流编程模型，在AI训练集群中也广泛使用。OpenMPI、MVAPICH是常用的MPI实现。

### NCCL通信域 (NCCL Communicator)

英文：NCCL Communicator

释义：NCCL通信域定义一组可以相互通信的GPU，是NCCL通信的基本上下文。创建通信域时确定通信的设备集合和排名。通信域隔离不同任务的通信，避免相互干扰。

### 桶同步 (Bucket Synchronization)

英文：Bucket Synchronization

释义：桶同步是先将梯度分桶再异步同步的技术，减少通信等待。梯度被分为多个桶，桶满后触发AllReduce，不必等待所有梯度计算完成。桶同步隐藏通信延迟，提高训练效率。

### 延迟隐藏 (Latency Hiding)

英文：Latency Hiding

释义：延迟隐藏是通过计算掩盖通信延迟的技术，如计算下一层时通信上一层的梯度。延迟隐藏最大化硬件利用率，是分布式训练的重要优化技术。

### 通信计算重叠 (Computation-Communication Overlap)

英文：Computation-Communication Overlap

释义：通信计算重叠是将通信和计算并行执行以隐藏延迟的优化策略。通过流水线将梯度通信与反向计算重叠，设备利用率更高。有效的重叠需要精心的调度设计。

### 拓扑感知调度 (Topology-Aware Scheduling)

英文：Topology-Aware Scheduling

释义：拓扑感知调度是根据硬件拓扑（NVLink、PCIe、InfiniBand）优化任务分配和通信路径的调度策略。拓扑感知减少跨节点通信，提高有效带宽。Kubernetes的拓扑管理器（Topology Manager）支持此类优化。

### NVLink拓扑 (NVLink Topology)

英文：NVLink Topology

释义：NVLink拓扑描述GPU间NVLink连接的方式，不同服务器架构有不同拓扑。NVSwitch提供全互联拓扑，否则GPU间需通过PCIe或NVLink switch跳转。拓扑影响张量并行的效率。

### PCIe拓扑 (PCIe Topology)

英文：PCIe Topology

释义：PCIe拓扑描述CPU、GPU、网卡等设备的PCIe连接方式。PCIe带宽低于NVLink，多GPU通常共享PCIe总线。PCIe拓扑影响数据并行中梯度同步的效率。

### Ring-AllReduce

英文：Ring-AllReduce

释义：Ring-AllReduce是AllReduce的环形算法实现，节点排成环传递数据，每步节点同时收发。Ring-AllReduce通信量与节点数无关，适合大规模。Ring-AllReduce是分布式训练的常用算法。

### 树形AllReduce (Tree-Based AllReduce)

英文：Tree-Based AllReduce

释义：树形AllReduce是AllReduce的树形算法实现，节点组织成树形结构。树形AllReduce延迟与log(N)成正比，但需要构造平衡树。树形AllReduce在低延迟网络上有优势。

### 集合通信优化 (Collective Communication Optimization)

英文：Collective Communication Optimization

释义：集合通信优化是提高通信效率的技术，包括拓扑映射、混合精度通信、分层聚合等。集合通信优化降低通信开销，是分布式训练系统优化的重要方向。

### 分层聚合 (Hierarchical Aggregation)

英文：Hierarchical Aggregation

释义：分层聚合先在节点内GPU聚合，再跨节点聚合，减少跨网络通信量。分层聚合利用服务器内的NVLink带宽，高于跨节点网络带宽。分层聚合是大规模分布式训练的有效优化。

### 梯度压缩 (Gradient Compression)

英文：Gradient Compression

释义：梯度压缩是压缩梯度以减少通信量的技术，如梯度量化（用低精度表示）、梯度稀疏化（只传重要梯度）、熵编码等。梯度压缩在通信带宽受限时特别有价值。

### 1-bit Adam

英文：1-bit Adam

释义：1-bit Adam是将梯度压缩到1位的Adam变体，使用特殊的状态压缩技术保持收敛性。1-bit Adam减少95%梯度通信量，允许在低带宽网络训练。1-bit Adam是广域网分布式训练的重要技术。

### PowerSGD (Power Stochastic Gradient Descent)

英文：PowerSGD

释义：PowerSGD是利用低秩近似压缩梯度的算法，将梯度矩阵压缩为其主要成分。PowerSGD在减少通信量的同时保持较好的收敛性。PowerSGD是梯度压缩的重要研究方向。

### 稀疏梯度更新 (Sparse Gradient Update)

英文：Sparse Gradient Update

释义：稀疏梯度更新只传输和应用部分梯度（如绝对值最大的k个），其余梯度置零。稀疏梯度更新减少通信和计算，但可能影响收敛稳定性。实践中需要谨慎选择稀疏比例。

### 模型分片 (Model Sharding)

英文：Model Sharding

释义：模型分片是将模型参数划分存储到多个设备的技术。ZeRO通过优化器状态、梯度、参数的分片实现模型分片。模型分片是内存优化和模型并行的基础。

### 优化器状态分片 (Optimizer State Sharding)

英文：Optimizer State Sharding

释义：优化器状态分片是ZeRO-1的策略，将优化器状态（如Adam的momentum）分片到各设备。优化器状态分片显著减少单设备显存占用，几乎不增加通信量。

### 梯度分片 (Gradient Sharding)

英文：Gradient Sharding

释义：梯度分片是ZeRO-2的策略，在ZeRO-1基础上对梯度也进行分片。梯度分片进一步减少显存占用，通信量与ZeRO-1相同。ZeRO-2是实用性和效率的良好平衡。

### 参数分片 (Parameter Sharding)

英文：Parameter Sharding

释义：参数分片是ZeRO-3的策略，对所有模型状态（参数、梯度、优化器状态）进行分片。参数分片支持训练超大模型，但需要重排获取完整参数。ZeRO-3通信量最大，需要高带宽。

### 动态分组调度 (Dynamic Grouping)

英文：Dynamic Grouping

释义：动态分组调度是流水线并行中根据运行时负载动态调整微批次分配的策略。动态分组减少流水线气泡，提高设备利用率。动态分组需要运行时监控和调度算法支持。

---

## 附录：常见英文缩写表

| 缩写 | 英文全称 | 中文名称 |
|:---:|:---|:---|
| AI | Artificial Intelligence | 人工智能 |
| ML | Machine Learning | 机器学习 |
| DL | Deep Learning | 深度学习 |
| NN | Neural Network | 神经网络 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| RNN | Recurrent Neural Network | 循环神经网络 |
| LSTM | Long Short-Term Memory | 长短期记忆网络 |
| GRU | Gated Recurrent Unit | 门控循环单元 |
| GAN | Generative Adversarial Network | 生成对抗网络 |
| VAE | Variational Autoencoder | 变分自编码器 |
| Transformer | Transformer | 变换器 |
| BERT | Bidirectional Encoder Representations from Transformers | 双向编码器表示 |
| GPT | Generative Pre-trained Transformer | 生成式预训练变换器 |
| NLP | Natural Language Processing | 自然语言处理 |
| CV | Computer Vision | 计算机视觉 |
| GPU | Graphics Processing Unit | 图形处理器 |
| CPU | Central Processing Unit | 中央处理器 |
| TPU | Tensor Processing Unit | 张量处理器 |
| NPU | Neural Network Processor | 神经网络处理器 |
| FPGA | Field-Programmable Gate Array | 现场可编程门阵列 |
| ASIC | Application-Specific Integrated Circuit | 专用集成电路 |
| DSA | Domain-Specific Architecture | 领域专用架构 |
| HBM | High Bandwidth Memory | 高带宽内存 |
| DDR | Double Data Rate | 双倍数据率内存 |
| PCIe | Peripheral Component Interconnect Express | 外围组件互连高速总线 |
| CUDA | Compute Unified Device Architecture | 统一计算设备架构 |
| IR | Intermediate Representation | 中间表示 |
| JIT | Just-In-Time Compilation | 即时编译 |
| AOT | Ahead-Of-Time Compilation | 预编译 |
| NAS | Neural Architecture Search | 神经网络架构搜索 |
| ONNX | Open Neural Network Exchange | 开放神经网络交换 |
| QAT | Quantization-Aware Training | 量化感知训练 |
| PTQ | Post-Training Quantization | 训练后量化 |
| DDP | Distributed Data Parallel | 分布式数据并行 |
| NCCL | NVIDIA Collective Communications Library | NVIDIA集体通信库 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问 |
| AllReduce | AllReduce | 全规约 |
| ZeRO | Zero Redundancy Optimizer | 零冗余优化器 |

---

## 术语索引

本术语表按类别编排，涵盖AI系统领域的核心术语。读者可通过以下方式快速查阅：

**按字母顺序**：英文术语按首字母A-Z排列，便于已知英文术语快速查找中文释义。

**按类别浏览**：术语分为八大类别，可根据所属领域定向阅读：

- **A. 数学与统计基础**：线性代数、概率统计、优化理论
- **B. 机器学习基础**：监督学习、无监督学习、模型评估
- **C. 神经网络基础**：神经元、激活函数、前反向传播
- **D. 深度学习模型**：CNN、RNN、Transformer、GAN等
- **E. AI芯片与硬件**：GPU、TPU、内存、互连
- **F. AI编译器**：IR、优化、代数化简、代码生成
- **G. 推理与部署**：量化、剪枝、蒸馏、推理引擎
- **H. 分布式系统**：数据并行、模型并行、通信原语

术语表旨在为AI系统学习和实践提供系统性参考，涵盖从理论到工程、从训练到部署的完整知识链条。