# Stanford CS231n — 卷积神经网络与视觉识别

Stanford 2023/2024 版 [CS231n](http://vision.stanford.edu/teaching/cs231n/) 全三周作业的完整实现，覆盖从 k-NN 到扩散模型的视觉深度学习全链路。

---

## 项目结构

```
assignments/
├── assignment1/               # 图像分类基础 (纯 NumPy)
│   ├── knn.ipynb                         # k-近邻分类器
│   ├── softmax.ipynb                     # Softmax 线性分类器
│   ├── two_layer_net.ipynb               # 两层全连接网络
│   ├── FullyConnectedNets.ipynb          # 模块化深度网络 + 优化器
│   ├── features.ipynb                    # HOG / 颜色直方图特征提取
│   └── cs231n/
│       ├── classifiers/    kNN · 线性SVM · Softmax · FC网络
│       ├── layers.py       Affine · ReLU · Softmax（前向+反向）
│       ├── optim.py        SGD · Momentum · Adam · RMSProp
│       └── solver.py       训练循环 + 学习率衰减
│
├── assignment2/               # 卷积网络与 PyTorch
│   └── assignment2/
│       ├── ConvolutionalNetworks.ipynb     # 手写 CNN（im2col）
│       ├── BatchNormalization.ipynb        # BN 前向/反向推导
│       ├── Dropout.ipynb                   # Dropout 正则化
│       ├── PyTorch.ipynb                   # PyTorch 入门 → CNN
│       ├── RNN_Captioning_pytorch.ipynb    # LSTM 图像字幕
│       └── cs231n/
│           ├── layers.py     Conv · MaxPool · BN · Dropout · RNN · LSTM
│           ├── fast_layers.py   Cython 加速卷积/池化
│           ├── im2col.py        im2col 卷积展开算法
│           └── classifiers/     CNN · FC Net · RNN 字幕模型
│
├── assignment3/               # Transformer · 自监督 · 扩散模型
│   └── assignment3/
│       ├── Transformer_Captioning.ipynb      # Transformer 字幕 + ViT
│       ├── Self_Supervised_Learning.ipynb    # SimCLR 对比学习
│       ├── DDPM.ipynb                        # 扩散概率模型
│       ├── CLIP_DINO.ipynb                   # 视觉大模型特征提取
│       └── cs231n/
│           ├── transformer_layers.py    MultiHeadAttention · TransformerBlock
│           ├── gaussian_diffusion.py    DDPM 前向扩散 + 逆向去噪
│           ├── unet.py                  U-Net（时间步条件注入）
│           ├── clip_dino.py             CLIP/DINO 零样本推理
│           └── simclr/                  NT-Xent 对比损失 · 数据增强
```

---

## Assignment 1 — 图像分类流水线

### k-近邻分类器
- 实现了 **3 个版本**的距离计算：双重循环 / 单循环 / 完全向量化
- 速度对比：37.9s → 0.61s（**62 倍加速**）
- **5 折交叉验证**搜索最优 k [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
- 最优 k=10，测试集准确率 **28.2%**

### Softmax 线性分类器
- 推导并实现向量化的 softmax 损失与解析梯度
- SGD + L2 正则化 + 学习率网格搜索
- 最佳验证集 **37.5%**，测试集 **36.1%**

```python
# 超参数搜索示例
learning_rates = [5e-8, 1e-7, 5e-7, 1e-6]
regularization_strengths = [1e4, 2.5e4, 5e4, 7.5e4]
```

### 两层全连接网络
- 手动实现 Affine → ReLU → Affine → Softmax 的前向传播与反向传播
- 调参：隐藏层大小、学习率、正则化强度、训练轮数
- 验证集 **49.4%**，测试集 **51.6%**

### 模块化深度网络
- 构建任意深度/宽度的 FC 网络框架（`FullyConnectedNet`）
- 从零实现 **SGD+Momentum**、**RMSProp**、**Adam** 三种优化器
- 梯度检验、小数据集过拟合等完整性检查全部通过

### 图像特征提取
- 提取 **HOG**（方向梯度直方图）和 **颜色直方图** 特征
- 手工特征 + 神经网络组合提升分类性能

---

## Assignment 2 — 卷积网络与序列模型

### 手写卷积神经网络
- 基于 **im2col** 算法实现 `conv_forward/backward` 和 `max_pool_forward/backward`
- 构建 `{Conv → BN → ReLU → MaxPool} × N → FC → Softmax` 多层 CNN
- 健全性检验：初始 loss ≈ log(C)，50 样本过拟合至 100% 训练准确率

### Batch Normalization
- 推导并实现训练模式与测试模式的 BN 前向/反向传播
- 手写 γ、β 的梯度链式求导
- 实验验证：BN 显著加速收敛，提升最终准确率

### Dropout 正则化
- 实现 **inverted dropout** 的前向/反向
- 理解其正则化机理：迫使每个神经元独立学习有用特征

### PyTorch CIFAR-10 CNN
- 用 PyTorch 搭建 `Conv→ReLU→MaxPool → Conv→ReLU→MaxPool → FC` 网络
- 1 epoch 即达到 **44.9%**（随机猜测 10%），完整训练可得更高
- 首次从 NumPy 手工实现过渡到 PyTorch 框架

### RNN/LSTM 图像字幕生成
- 构建 **编码器-解码器**架构：CNN 图像特征 → LSTM 解码器 → 单词序列
- COCO 2014 数据集 + 预训练 VGG-16 特征
- 实现 teacher forcing 训练与束搜索推理

---

## Assignment 3 — 前沿深度学习

### Transformer 图像字幕 + ViT 视觉分类器
- 从零实现 **多头缩放点积注意力**（Q/K/V 投影、softmax 缩放、多头拼接）
- 构建 Transformer Encoder（位置编码 + 层归一化 + 残差连接）
- 图像字幕训练最终 loss **0.022**（目标 < 0.05）
- 实现 **Vision Transformer**：图像分块 → Patch Embedding → Transformer → 分类头
- ViT 单 batch 过拟合 Top-1 **34.4%**，验证架构正确性

### SimCLR 自监督对比学习
- 实现 **NT-Xent 损失**（归一化温度缩放交叉熵）
- 构建数据增强流水线：随机裁剪 + 颜色抖动 + 水平翻转 + 灰度化
- 在无标签 CIFAR-10 上训练 ResNet 骨干网络
- 仅用 **10% 有标签数据微调**即可达到 **≥70% Top-1 准确率**
- 证明了表示学习的力量：少样本也能获得强性能

### DDPM — 扩散概率模型
- 实现完整 DDPM 管线：前向扩散 + 逆向去噪
- 构建带**时间步条件**的 U-Net 噪声预测网络 εθ(xt, t)
- 线性和余弦 β 调度 + 重参数化技巧
- 在 emoji 数据集上从纯噪声迭代生成新表情符号

### CLIP & DINO 视觉大模型
- 使用预训练 **CLIP** 进行零样本图像分类
- 使用 **DINO** 提取密集视觉特征
- 对比零样本 vs 微调性能差异

---

## 关键概念问答

### Q1: 为什么 Softmax 初始化时 loss 约等于 -log(0.1)？
> 随机初始化时，W ≈ 0，所有类别的得分 s_j 接近 0，因此 softmax 概率 $p_i = e^0 / (10 \cdot e^0) = 0.1$，交叉熵 loss $= -\log(0.1) \approx 2.3$。

### Q2: 哪些激活函数会导致梯度消失？
> **Sigmoid** 在输入极大或极小时梯度趋近于零。当输入极大且网络极深时，会发生梯度爆炸或梯度消失；ReLU 在正区间梯度恒为 1，缓解了这一问题；Leaky ReLU 进一步在负区间保留微小梯度。

### Q3: 训练集准确率远高于验证集怎么办？
> 这明显是**过拟合**。解决方案：在更大的数据集上训练（降低方差）、增加正则化强度（约束模型复杂度）、使用 dropout 等技巧提升泛化能力。

### Q4: 多头注意力的三个关键设计为什么是必要的？
> 1. **多头 vs 单头**：词向量是多义载体，单头只能捕获一种语义关系；多头允许从不同子空间学习特征，类比 SVM 的高维映射，能区分多义词的不同语义。
> 2. **除以 √(d/h)**：点积 QKᵀ 的结果随维度增大而增大，方差被放大，softmax 后权重趋向极端（大值→1，小值→0），梯度几近于零；缩放后方差得到控制。
> 3. **输出线性变换**：多头拼接后各子空间信息相对独立，可学习的线性变换能动态融合不同子空间的特征，提升最终表示的表达能力。

### Q5: Softmax 权重可视化为什么模糊？
> Softmax 是**线性模型**，表达能力有限，无法捕捉 CIFAR-10 图像中复杂的非线性特征。线性模型无法学习到物体类别的精细层次化模式，因此权重无法形成可辨识的类别视觉表征（如飞机的轮廓、汽车的车身等）。

---

## 实验结果汇总

| 模型 | 测试准确率 | 备注 |
|:-----|:----------|:-----|
| k-NN (k=10) | 28.2% | 5折交叉验证 |
| Softmax 线性分类器 | 36.1% | 原始像素 + bias trick |
| 两层 FC 网络 | 51.6% | SGD + L2 正则化 |
| PyTorch CNN (1 epoch) | 44.9% | 还有上升空间 |
| SimCLR + 10% 标签 | ≥ 70% | 自监督 + 微调 |
| Transformer 字幕 | loss 0.022 | 目标 < 0.05 |

---

## 关键收获

- **向量化至关重要** — 无循环 kNN 比双重循环快 62 倍
- **CNN 碾压全连接** — 卷积归纳偏置是视觉任务的核心
- **Batch Normalization** — 稳定训练、加速收敛、允许更深网络
- **自监督学习** — 用少量标签达到强性能，预训练 + 微调范式无敌
- **Attention is All You Need** — Transformer 统一了视觉与语言建模

---

## 环境配置

```bash
# 基础依赖
pip install numpy matplotlib scipy jupyter torch torchvision

# Cython 加速（可选）
pip install cython
cd assignment2/assignment2/cs231n && python setup.py build_ext --inplace

# 启动
jupyter notebook
```

---

*基于 Stanford CS231n: Convolutional Neural Networks for Visual Recognition 课程作业。*
