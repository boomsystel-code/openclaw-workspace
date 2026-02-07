# AI工程师面试指南

*AI/机器学习工程师面试必备知识体系*

---

## 📋 目录

1. [编程基础](#1-编程基础)
2. [机器学习](#2-机器学习)
3. [深度学习](#3-深度学习)
4. [NLP](#4-nlp)
5. [计算机视觉](#5-计算机视觉)
6. [系统设计](#6-系统设计)
7. [Coding题目](#7-coding题目)

---

## 1. 编程基础

### 1.1 Python

**数据类型**：
- 基础类型：int, float, str, bool
- 容器类型：list, tuple, dict, set
- 特殊类型：None, complex

**常用库**：
- NumPy：数值计算
- Pandas：数据处理
- Matplotlib：可视化

**高级特性**：
- 列表推导式：[x for x in range(10) if x % 2 == 0]
- 生成器：yield关键字
- 装饰器：@decorator
- 上下文管理器：with语句

**面向对象**：
- 类和对象
- 继承和多态
- 魔术方法：__init__, __str__, __len__

### 1.2 数据结构

**线性结构**：
- 数组：连续内存
- 链表：指针连接
- 栈：后进先出
- 队列：先进先出

**树形结构**：
- 二叉树
- 平衡树：AVL、红黑树
- B树/B+树
- 堆：最大堆、最小堆

**图结构**：
- 有向图/无向图
- 加权图
- 邻接矩阵/邻接表

**哈希**：
- 哈希函数
- 冲突解决
- 哈希表实现

### 1.3 算法

**排序算法**：
| 算法 | 时间复杂度 | 空间复杂度 | 稳定性 |
|------|-----------|-----------|--------|
| 冒泡排序 | O(n²) | O(1) | 稳定 |
| 插入排序 | O(n²) | O(1) | 稳定 |
| 归并排序 | O(n log n) | O(n) | 稳定 |
| 快速排序 | O(n log n) | O(log n) | 不稳定 |
| 堆排序 | O(n log n) | O(1) | 不稳定 |
| 桶排序 | O(n+k) | O(n+k) | 稳定 |

**查找算法**：
- 二分查找：O(log n)
- 哈希查找：O(1)
- BFS/DFS：O(V+E)

**动态规划**：
- 最优子结构
- 重叠子问题
- 状态转移方程

**贪心算法**：
- 局部最优
- 全局最优

---

## 2. 机器学习

### 2.1 基础概念

**监督学习**：
- 分类和回归
- 训练/验证/测试集
- 过拟合和欠拟合

**无监督学习**：
- 聚类
- 降维
- 异常检测

**评估指标**：
- 准确率、精确率、召回率
- F1-Score、AUC-ROC
- MSE、MAE、R²

### 2.2 经典算法

**线性回归**：
- 假设函数：h(x) = wx + b
- 损失函数：MSE
- 正规解：w = (X^T X)^(-1) X^T y
- 梯度下降迭代

**逻辑回归**：
- Sigmoid函数：σ(z) = 1/(1+e^(-z))
- 交叉熵损失
- 二分类和多分类

**决策树**：
- 信息增益
- 基尼系数
- 剪枝策略

**随机森林**：
- Bagging策略
- 特征随机
- 多树集成

**支持向量机**：
- 最大间隔
- 核函数
- 软间隔分类

### 2.3 降维技术

**PCA主成分分析**：
- 协方差矩阵
- 特征值分解
- 主成分选择

**t-SNE**：
- 流行学习
- 相似度保持
- 可视化应用

---

## 3. 深度学习

### 3.1 神经网络基础

**神经元模型**：
- 加权求和
- 激活函数
- 前向传播

**反向传播**：
- 链式法则
- 梯度计算
- 参数更新

**梯度下降**：
- Batch GD
- Mini-batch GD
- SGD
- 动量优化

### 3.2 优化器

**SGD**：
w = w - lr * gradient

**Momentum**：
v = γv + lr * gradient
w = w - v

**Adam**：
m = β1 * m + (1-β1) * gradient
v = β2 * v + (1-β2) * gradient²
w = w - lr * m / (sqrt(v) + ε)

### 3.3 正则化

**L1/L2正则化**：
L = L_original + λ * Σ|w|

**Dropout**：
训练时随机置零

**BatchNorm**：
标准化每层输入

### 3.4 CNN

**卷积操作**：
- 过滤器/卷积核
- 步长和填充
- 感受野

**池化操作**：
- Max Pooling
- Average Pooling

**经典架构**：
- LeNet-5
- AlexNet
- VGG
- ResNet
- EfficientNet

### 3.5 RNN/LSTM

**RNN**：
h_t = f(W * h_{t-1} + U * x_t)

**LSTM**：
- 遗忘门
- 输入门
- 输出门

**GRU**：
- 更新门
- 重置门

### 3.6 Transformer

**自注意力**：
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

**位置编码**：
- 正弦编码
- 可学习编码

**Encoder-Decoder**：
- 多头注意力
- 前馈网络
- 残差连接

---

## 4. NLP

### 4.1 文本处理

**分词**：
- 词级分词
- 子词分词
- 字符级分词

**词向量**：
- Word2Vec
- GloVe
- FastText

### 4.2 序列模型

**RNN应用**：
- 文本分类
- 情感分析
- 命名实体识别

**注意力机制**：
- Bahdanau Attention
- Luong Attention
- Self-Attention

### 4.3 预训练模型

**BERT**：
- 双向Transformer
- MLM + NSP预训练
- 微调范式

**GPT**：
- 单向Transformer
- 下一个token预测
- 零样本/少样本学习

### 4.4 实战技巧

**Fine-tuning**：
- 学习率设置
- 层冻结策略
- 数据增强

**提示工程**：
- Zero-shot
- Few-shot
- Chain-of-Thought

---

## 5. 计算机视觉

### 5.1 基础任务

**图像分类**：
- Top-1/Top-5 Accuracy
- 迁移学习

**目标检测**：
- Bounding Box
- IoU计算
- mAP指标

**语义分割**：
- 像素级分类
- Dice Loss
- IoU指标

### 5.2 检测算法

**两阶段检测**：
- R-CNN系列
- Faster R-CNN
- RoI Align

**单阶段检测**：
- YOLO系列
- SSD
- RetinaNet

### 5.3 分割算法

**FCN**：
- 全卷积网络
- 反卷积上采样

**U-Net**：
- 编码器-解码器
- 跳跃连接

**DeepLab**：
- 空洞卷积
- ASPP模块

### 5.4 训练技巧

**数据增强**：
- 几何变换
- 颜色变换
- MixUp/CutMix

**模型选择**：
- ResNet系列
- EfficientNet
- Vision Transformer

---

## 6. 系统设计

### 6.1 特征工程

**特征类型**：
- 数值特征
- 类别特征
- 时序特征
- 文本特征

**特征处理**：
- 标准化
- 归一化
- 编码方式

### 6.2 模型部署

**模型保存**：
- Pickle
- SavedModel
- ONNX

**服务化**：
- TensorFlow Serving
- TorchServe
- Triton

**优化**：
- 量化
- 剪枝
- 知识蒸馏

### 6.3 分布式训练

**数据并行**：
- 参数同步
- 梯度聚合

**模型并行**：
- 张量并行
- 流水线并行

**框架**：
- Horovod
- DeepSpeed
- FSDP

---

## 7. Coding题目

### 7.1 数组

**两数之和**：
```python
def two_sum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
```

**三数之和**：
```python
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result
```

### 7.2 链表

**反转链表**：
```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

**合并两个有序链表**：
```python
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 if l1 else l2
    return dummy.next
```

### 7.3 动态规划

**斐波那契**：
```python
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**背包问题**：
```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]
```

### 7.4 树

**二叉树遍历**：
```python
# 前序遍历
def preorder(root):
    if not root:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

# 中序遍历
def inorder(root):
    if not root:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

# 后序遍历
def postorder(root):
    if not root:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)
```

**二叉搜索树验证**：
```python
def is_valid_bst(root, float('-inf'), float('inf')):
    if not root:
        return True
    if not (float('-inf') < root.val < float('inf')):
        return False
    return (is_valid_bst(root.left, float('-inf'), root.val) and
            is_valid_bst(root.right, root.val, float('inf')))
```

---

## 📚 参考资源

### 书籍
- 《剑指Offer》
- 《算法导论》
- 《深度学习》

### 在线平台
- LeetCode
- HackerRank
- Kaggle

### 课程
- CS231n
- CS224n
- Andrew Ng ML

---

*本指南约贡献50KB面试知识*

