

---

# 🚀 Machine Learning 完整系统教程

*从YouTube深度学习课程提取的系统化知识体系*

---

## 📊 概述

**来源**: Machine Learning for Everybody - Full Course
**内容类型**: 机器学习完整入门教程
**涵盖主题**: 机器学习基础、神经网络、深度学习、实战技巧

---

## 🎯 核心主题详解


🤖 1. 机器学习导论
--------------------------------------------------



### 1. 什么是机器学习？

机器学习是人工智能的一个核心分支，它使计算机能够从数据中自动学习和改进，而不需要针对每个任务进行明确编程。

**核心思想**：
- 给算法提供大量数据
- 让算法自己发现数据中的模式和规律
- 基于历史数据进行预测和决策

**与传统编程的区别**：
| 传统编程 | 机器学习 |
|---------|---------|
| 人工编写规则 | 算法从数据学习规则 |
| 固定逻辑 | 可适应新数据 |
| 人工维护规则 | 自动优化 |

**机器学习的工作流程**：
```
收集数据 → 清洗数据 → 特征工程 → 选择模型 → 训练模型 → 评估模型 → 部署应用
```

---

### 2. 机器学习的应用领域

**图像处理**：
- 人脸识别（Face Recognition）
- 物体检测（Object Detection）
- 图像分类（Image Classification）
- 医学影像分析

**自然语言处理**：
- 机器翻译（Machine Translation）
- 情感分析（Sentiment Analysis）
- 聊天机器人（Chatbot）
- 文本摘要（Text Summarization）

**语音识别**：
- 语音转文字（Speech to Text）
- 语音合成（Text to Speech）
- 声纹识别（Voice Recognition）

**推荐系统**：
- 商品推荐（Amazon, Taobao）
- 内容推荐（YouTube, TikTok）
- 个性化推送

**金融领域**：
- 信用评分（Credit Scoring）
- 欺诈检测（Fraud Detection）
- 股票预测（Stock Prediction）
- 风险评估

**医疗健康**：
- 疾病诊断（Disease Diagnosis）
- 药物发现（Drug Discovery）
- 基因分析（Genomic Analysis）
- 健康监测

---

### 3. 机器学习的挑战

**数据挑战**：
- 数据质量（噪声、缺失值）
- 数据量不足
- 数据不平衡
- 隐私和合规问题

**算法挑战**：
- 选择合适的算法
- 超参数调优
- 过拟合问题
- 模型可解释性

**工程挑战**：
- 模型部署
- 实时性能
- 可扩展性
- 成本控制


**课程要点** (190条):

- And now she's going to teach you about machine learning in a way that is accessible to absolute beginners
- If you are someone who is interested in machine learning and you think you are considered as everyone, then this video is for you
- If there are certain things that I have done, and you know, you're somebody with more experience than me, please feel free to correct me in the comments and we can all as a community learn from this together
- Without wasting any time, let's just dive straight into the code and I will be teaching you guys concepts as we go
- So this here is the UCI machine learning repository
- Now there's a camera, there's a detector that actually records certain patterns of you know, how this light hits the camera
- So before I move on, let me just give you a quick little crash course on what I just said
- Well, the first question is, what is machine learning

🧠 2. 神经网络基础
--------------------------------------------------



### 4. 神经网络基础

**神经网络定义**：
神经网络是一种受人脑结构启发的计算系统，由大量互相连接的神经元组成，能够学习和处理复杂模式。

**生物神经元到人工神经元**：

**生物神经元**：
- 细胞体（Soma）：包含细胞核
- 树突（Dendrites）：接收输入信号
- 轴突（Axon）：发送输出信号
- 突触（Synapse）：连接其他神经元

**人工神经元（Perceptron）**：
```
输入 (x₁, x₂, ..., xₙ)
    ↓
权重分配 (w₁, w₂, ..., wₙ)
    ↓
加权求和: Σ(wᵢ × xᵢ) + b
    ↓
激活函数: f(Σ)
    ↓
输出 (y)
```

**关键参数**：
- **权重（Weights）**：决定每个输入的重要程度
- **偏置（Bias）**：调整激活阈值
- **激活函数**：引入非线性

**神经网络结构**：
```
输入层 (Input Layer)
    ↓
隐藏层1 (Hidden Layer 1)
    ↓
隐藏层2 (Hidden Layer 2)
    ↓
...
    ↓
隐藏层n (Hidden Layer n)
    ↓
输出层 (Output Layer)
```

**层的作用**：
- **输入层**：接收原始数据（像素值、特征向量）
- **隐藏层**：提取和转换特征
- **输出层**：产生最终预测

---

### 5. 激活函数详解

**为什么需要激活函数？**
- 引入非线性，使网络能够拟合复杂模式
- 控制信息流动
- 使神经网络能够表示任意函数

**常用激活函数**：

**Sigmoid函数**：
- 公式：σ(x) = 1 / (1 + e^(-x))
- 输出范围：(0, 1)
- 用途：二分类问题的输出层
- 问题：梯度消失

**ReLU函数（Rectified Linear Unit）**：
- 公式：f(x) = max(0, x)
- 输出范围：[0, +∞)
- 优点：简单高效、训练收敛快
- 问题：神经元死亡（Dying ReLU）
- 变体：Leaky ReLU、ELU

**Tanh函数**：
- 公式：tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- 输出范围：(-1, 1)
- 特点：零中心化
- 用途：循环神经网络（RNN）

**Softmax函数**：
- 用途：多分类问题的输出层
- 特点：输出概率分布，总和为1

---

### 6. 神经网络如何学习

**前向传播（Forward Propagation）**：
1. 输入数据进入网络
2. 每层进行加权求和 + 偏置 + 激活
3. 最终输出预测结果

**反向传播（Backpropagation）**：
1. 计算输出误差（损失函数）
2. 从输出层向输入层传播误差
3. 利用链式法则计算每个参数的梯度
4. 使用梯度下降更新参数

**参数更新公式**：
```
w_new = w_old - learning_rate × gradient
b_new = b_old - learning_rate × gradient
```


**课程要点** (42条):

- Now the final type of model that I wanted to talk about is known as a neural net or neural network
- So you have an input layer, this is where all your features would go
- And they have all these arrows pointing to some sort of hidden layer
- And then all these arrows point to some sort of output layer
- Each of these layers in here, this is something known as a neuron
- Now I'm also adding this bias term, which just means okay, I might want to shift this by a little bit
- So the sum of this, this, this and this, go into something known as an activation function, okay
- And then after applying this activation function, we get an output

🔥 3. 深度学习核心
--------------------------------------------------



### 7. 深度学习核心概念

**什么是深度学习？**
深度学习是机器学习的一个子领域，使用多层神经网络（深层网络）来自动学习数据的层次化特征表示。

**深度 vs 浅层网络**：
- 浅层网络：1-2个隐藏层
- 深层网络：多个隐藏层（数十甚至数百层）

**为什么需要深层网络？**
- 学习数据的层次化特征
- 底层：学习基本特征（边缘、纹理）
- 中层：组合基本特征（形状、部件）
- 高层：学习高级语义（物体、概念）

**深度学习的优势**：
- 自动特征学习（无需手工特征工程）
- 处理非结构化数据（图像、文本、音频）
- 大规模数据下性能卓越
- 持续学习改进

**深度学习的挑战**：
- 需要大量数据
- 计算资源要求高
- 训练时间长
- 可解释性差
- 超参数调优复杂

---

### 8. 过拟合与欠拟合

**欠拟合（Underfitting）**：
- **原因**：模型太简单，无法捕捉数据模式
- **表现**：训练误差和测试误差都很高
- **解决**：
  - 增加模型复杂度
  - 增加特征
  - 减少正则化
  - 训练更长时间

**过拟合（Overfitting）**：
- **原因**：模型太复杂，记住训练数据的噪声
- **表现**：训练误差很低，测试误差很高
- **解决**：
  - 增加训练数据
  - L1/L2正则化
  - Dropout
  - 数据增强
  - 早停（Early Stopping）

**如何识别过拟合**：
- 训练损失持续下降，验证损失上升
- 训练准确率接近100%，但验证准确率较低
- 模型在训练集表现远好于测试集

---

### 9. 正则化技术

**L1正则化**：
- 在损失函数中添加权重绝对值之和
- 产生稀疏权重（特征选择）
- 公式：L = L_original + λ × Σ|w|

**L2正则化（权重衰减）**：
- 在损失函数中添加权重平方和
- 防止权重过大
- 公式：L = L_original + λ × Σw²

**Dropout**：
- 训练时随机"关闭"部分神经元
- 防止神经元过度依赖
- 类似模型集成
- 代码示例：
```python
tf.keras.layers.Dropout(0.5)  # 50%神经元失效
```

**Batch Normalization**：
- 标准化每层的输入
- 优点：
  - 加速训练
  - 稳定梯度
  - 正则化效果
  - 允许更高学习率


⚙️ 4. 训练过程与优化
--------------------------------------------------



### 10. 训练过程详解

**损失函数（Loss Function）**：
衡量模型预测与实际值之间的差距。

**均方误差（MSE）**：
- 用途：回归任务
- 公式：MSE = (1/n) Σ(y - ŷ)²
- 特点：对大误差更敏感

**交叉熵损失（Cross-Entropy）**：
- 用途：分类任务
- 二分类：Binary Cross-Entropy
- 多分类：Categorical Cross-Entropy
- 公式：-Σ y·log(ŷ)

**对数损失（Log Loss）**：
- 用于二分类问题
- 公式：- (y·log(p) + (1-y)·log(1-p))

---

### 11. 优化算法

**随机梯度下降（SGD）**：
- 公式：w = w - η × ∂L/∂w
- 特点：简单，但收敛慢且震荡

**SGD with Momentum**：
- 公式：v = γv + η × ∂L/∂w; w = w - v
- 效果：加速收敛，减少震荡

**Adam优化器**：
- 结合Momentum和RMSprop
- 自适应学习率
- 默认参数：β₁=0.9, β₂=0.999, ε=1e-8
- 优点：快速稳定，内存效率高
- 公式：
```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t
v_t = β₂ × v_{t-1} + (1-β₂) × g²_t
w = w - η × m̂_t / (√v̂_t + ε)
```

---

### 12. 超参数调优

**学习率（Learning Rate）**：
- 太小：收敛太慢
- 太大：可能不收敛
- 技巧：学习率衰减、warm-up

**批量大小（Batch Size）**：
- Full Batch：使用全部数据，内存消耗大
- Mini-Batch：平衡性能和内存
- Stochastic：每次一个样本，噪声大

**迭代次数（Epoch）**：
- 一个epoch：所有数据训练一次
- 早停（Early Stopping）：防止过拟合

**超参数搜索方法**：
- 网格搜索（Grid Search）
- 随机搜索（Random Search）
- 贝叶斯优化（Bayesian Optimization）

---

### 13. 模型评估

**分类指标**：
- **准确率（Accuracy）**：正确预测的比例
- **精确率（Precision）**：预测为正类中实际为正类的比例
- **召回率（Recall）**：实际正类中被正确预测的比例
- **F1-Score**：精确率和召回率的调和平均
- **AUC-ROC**：分类器区分能力的度量

**回归指标**：
- **MSE**：均方误差，对大误差更敏感
- **MAE**：平均绝对误差，稳健性好
- **R²**：决定系数，衡量模型解释力

**验证方法**：
- 留出法（Hold-out）
- K折交叉验证（K-Fold Cross Validation）
- 留一法（LOOCV）


**课程要点** (45条):

- So this loss is also going to be high, let's give it 1
- Well, model C has a smallest loss, so it's probably model C
- And that loss, that's the final reported performance of my test set, or this would be the final reported performance of my model
- So let's talk about this thing called loss, because I think I kind of just glossed over it, right
- So this would give a slightly higher loss than this
- And this would even give a higher loss, because it's even more off
- So here are some examples of loss functions and how we can actually come up with numbers
- And basically, L one loss just takes the absolute value of whatever your you know, real value is, whatever the real output label is, subtracts the predicted value, and takes the absolute value of that

📱 5. 实际应用场景
--------------------------------------------------



### 14. 实际应用场景

**计算机视觉（Computer Vision）**：
- 图像分类（Image Classification）
- 目标检测（Object Detection）
- 语义分割（Semantic Segmentation）
- 实例分割（Instance Segmentation）
- 人脸识别（Face Recognition）
- 姿态估计（Pose Estimation）

**自然语言处理（NLP）**：
- 文本分类（Text Classification）
- 情感分析（Sentiment Analysis）
- 命名实体识别（NER）
- 机器翻译（Machine Translation）
- 问答系统（Question Answering）
- 文本生成（Text Generation）

**语音技术**：
- 语音识别（Speech Recognition）
- 语音合成（Speech Synthesis）
- 声纹识别（Speaker Recognition）
- 语音增强（Speech Enhancement）

**推荐系统**：
- 协同过滤（Collaborative Filtering）
- 深度学习推荐（Deep Learning for RecSys）
- 个性化推荐（Personalized Recommendation）

**强化学习**：
- 游戏AI（Game AI）
- 机器人控制（Robot Control）
- 自动驾驶（Autonomous Driving）
- 资源优化（Resource Optimization）

---

### 15. 未来发展趋势

**技术趋势**：
- 更高效的模型架构
- 自监督学习
- 多模态学习
- 小样本学习
- 持续学习

**应用趋势**：
- 边缘AI（Edge AI）
- AI芯片
- 可解释AI
- AI安全与隐私

**行业趋势**：
- AI民主化
- 行业垂直化
- 自动化机器学习（AutoML）
- AI即服务（AIaaS）


**课程要点** (1条):

- Okay, so that's the probability and doing a quick division, we get that this is equal to around 96

🔧 6. 核心技术与算法
--------------------------------------------------



### 16. 核心算法与技术

**监督学习算法**：
- 线性回归（Linear Regression）
- 逻辑回归（Logistic Regression）
- 决策树（Decision Tree）
- 随机森林（Random Forest）
- 支持向量机（SVM）
- K近邻（KNN）
- 梯度提升（Gradient Boosting）

**无监督学习算法**：
- K均值聚类（K-Means）
- 层次聚类（Hierarchical Clustering）
- DBSCAN
- 主成分分析（PCA）
- t-SNE
- UMAP

**深度学习算法**：
- 多层感知机（MLP）
- 卷积神经网络（CNN）
- 循环神经网络（RNN）
- 长短期记忆网络（LSTM）
- 门控循环单元（GRU）
- Transformer
- 自编码器（Autoencoder）
- 生成对抗网络（GAN）

**强化学习算法**：
- Q-Learning
- Deep Q-Network (DQN)
- Policy Gradient
- Actor-Critic (A2C, A3C)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

---

### 17. 迁移学习与微调

**迁移学习定义**：
将一个任务学到的知识应用到另一个相关任务。

**为什么需要迁移学习**：
- 解决数据不足问题
- 加速模型训练
- 提高模型性能

**迁移学习策略**：
1. **特征提取**：冻结预训练模型权重，只训练新添加的分类器
2. **微调**：解冻部分层，进行端到端训练
3. **完全微调**：解冻所有层，重新训练

**预训练模型**：
- 图像：ResNet, VGG, EfficientNet, ViT
- 文本：BERT, GPT, RoBERTa, T5
- 多模态：CLIP, DALL-E, Stable Diffusion

**微调技巧**：
- 使用较小的学习率
- 先冻结再解冻
- 使用早停
- 数据增强


**课程要点** (69条):

- In this video, we'll talk about supervised and unsupervised learning models, we'll go through maybe a little bit of the logic or math behind them, and then we'll also see how we can program it on Google CoLab
- And in supervised learning, we're using labeled inputs
- Now in supervised learning, all of these inputs have a label associated with them, this is the output that we might want the computer to be able to predict
- And in unsupervised learning, we use unlabeled data to learn about patterns in the data
- But in this class today, we'll be focusing on supervised learning and unsupervised learning and learning different models for each of those
- Alright, so let's talk about supervised learning first
- So in supervised learning, there are some different tasks, there's one classification, and basically classification, just saying, okay, predict discrete classes
- This is something known as multi class classification

💻 7. Python与TensorFlow实战
--------------------------------------------------



### 18. Python与TensorFlow实战

**环境配置**：
```bash
# 创建虚拟环境
conda create -n ml python=3.9
conda activate ml

# 安装TensorFlow
pip install tensorflow
# 或 PyTorch
pip install torch torchvision
```

**基本数据类型**：
- **Tensor（张量）**：多维数组
- **0维**：标量（Scalar）
- **1维**：向量（Vector）
- **2维**：矩阵（Matrix）
- **3维及以上**：张量（Tensor）

**Keras Sequential模型**：
```python
import tensorflow as tf
from tensorflow import keras

# 创建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {{test_acc}}')
```

**Keras Functional API**：
```python
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

**回调函数**：
```python
# 早停
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 学习率调度
lr_scheduler = keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.01 * (0.1 ** (epoch // 30))
)

# 模型检查点
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True
)
```

**模型保存与加载**：
```python
# 保存整个模型
model.save('my_model.h5')

# 加载模型
model = keras.models.load_model('my_model.h5')

# 只保存权重
model.save_weights('my_weights.h5')

# 加载权重
model.load_weights('my_weights.h5')
```

**TensorFlow Lite（移动端部署）**：
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```


**课程要点** (15条):

- Kind: captions Language: en Kylie Ying has worked at many interesting places such as MIT, CERN, and Free Code Camp
- So actually, I'm going to call this for code camp magic example
- And then I'm going to show you how we can do that in our code
- Let's see how we would be able to do that within our code
- So the reason why we, you know, use these packages and so that we don't have to manually code all these things ourselves, because it would be really difficult
- And chances are the way that we would code it, either would have bugs, or it'd be really slow, or I don't know a whole bunch of issues
- And these are just simple ways on how to implement them
- Wouldn't it be great if there are just some, you know, full time professionals that are dedicated to solving this problem, and they could literally just give us their code that's already running really fast

⭐ 8. 最佳实践与陷阱
--------------------------------------------------



### 19. 最佳实践指南

**数据准备最佳实践**：

1. **数据清洗**：
   - 处理缺失值
   - 处理异常值
   - 数据标准化/归一化

2. **特征工程**：
   - 特征选择
   - 特征提取
   - 特征编码（One-Hot, Label Encoding）

3. **数据增强**：
   - 图像：旋转、翻转、缩放、裁剪
   - 文本：同义词替换、随机删除
   - 音频：时间偏移、音高变化

**模型构建最佳实践**：

1. **从简单开始**：
   - 先用简单模型验证流程
   - 逐步增加复杂度
   - 记录实验结果

2. **使用验证集**：
   - 划分训练集/验证集/测试集
   - 用验证集调参
   - 用测试集最终评估

3. **监控训练过程**：
   - 观察损失曲线
   - 使用TensorBoard可视化
   - 记录学习率变化

**训练优化最佳实践**：

1. **学习率策略**：
   - 从0.01开始尝试
   - 使用学习率衰减
   - 考虑warm-up

2. **批量大小**：
   - GPU内存允许下使用较大batch
   - 通常32或64效果不错
   - 考虑梯度累积

3. **正则化**：
   - Dropout（0.2-0.5）
   - L2正则化
   - 数据增强

**部署最佳实践**：

1. **模型优化**：
   - 量化（FP16、INT8）
   - 剪枝
   - 知识蒸馏

2. **推理优化**：
   - 批处理推理
   - 模型缓存
   - 异步处理

3. **监控与维护**：
   - 监控模型性能
   - 数据漂移检测
   - 模型更新策略

---

### 20. 常见错误与解决方案

**错误1：训练不收敛**
- **可能原因**：学习率太高/太低、数据未标准化
- **解决**：调整学习率、标准化数据

**错误2：过拟合**
- **可能原因**：模型太复杂、训练数据不足
- **解决**：正则化、数据增强、早停

**错误3：梯度消失**
- **可能原因**：深层网络、激活函数选择不当
- **解决**：使用ReLU、残差连接、BatchNorm

**错误4：内存不足**
- **可能原因**：批量太大、模型太复杂
- **解决**：减小batch、使用混合精度

**错误5：模型效果差**
- **可能原因**：数据质量差、特征不足
- **解决**：改进数据、特征工程

---

### 21. 学习资源推荐

**在线课程**：
- Coursera Machine Learning (Andrew Ng)
- fast.ai Practical Deep Learning
- Stanford CS231n (CV)
- Stanford CS224n (NLP)

**书籍**：
- 《动手学深度学习》
- 《深度学习》（花书）
- 《机器学习》（西瓜书）

**实践平台**：
- Kaggle（竞赛）
- Google Colab（免费GPU）
- Papers With Code（论文代码）

**社区**：
- GitHub
- Reddit (r/MachineLearning)
- Stack Overflow


**课程要点** (56条):

- So then, once you know, we've made a bunch of adjustments, we can put our validation set through this model
- We take model C, and we run our test set through this model
- And this test set is used as a final check to see how generalizable that chosen model is
- 8, this basically means everything between 60% and 80% of the length of the data set will go towards validation
- And then, like everything from 80 to 100, I'm going to pass my test data
- So here, I'm just going to make this the validation data set
- And then the next one, I'm going to make this the test data set
- Now, the reason why I'm switching that to false is because my validation and my test sets are for the purpose of you know, if I have data that I haven't seen yet, how does my sample perform on those


---

## 📚 学习路径建议

### 第一阶段：基础（2-4周）
1. Python编程基础
2. NumPy和Pandas
3. 机器学习基础概念
4. 完成简单项目（房价预测）

### 第二阶段：深度学习（4-6周）
1. 神经网络原理
2. TensorFlow/PyTorch基础
3. 完成MNIST分类
4. 尝试CNN项目

### 第三阶段：专项深入（8-12周）
- **计算机视觉**：图像分类、目标检测
- **自然语言处理**：文本分类、情感分析
- **强化学习**：游戏AI、机器人控制

### 第四阶段：项目实战（持续）
1. 完成3-5个完整项目
2. 参与Kaggle竞赛
3. 阅读经典论文
4. 建立个人作品集

---

## 🎓 核心概念速查

| 概念 | 说明 | 重要性 |
|------|------|--------|
| 监督学习 | 从标注数据学习 | ⭐⭐⭐⭐⭐ |
| 神经网络 | 受脑启发的计算模型 | ⭐⭐⭐⭐⭐ |
| 反向传播 | 训练神经网络的核心算法 | ⭐⭐⭐⭐⭐ |
| 梯度下降 | 优化参数的迭代方法 | ⭐⭐⭐⭐⭐ |
| 过拟合 | 模型记住噪声 | ⭐⭐⭐⭐⭐ |
| 正则化 | 防止过拟合的技术 | ⭐⭐⭐⭐ |
| 卷积神经网络 | 处理图像的神经网络 | ⭐⭐⭐⭐ |
| 循环神经网络 | 处理序列的神经网络 | ⭐⭐⭐⭐ |
| 迁移学习 | 利用预训练模型 | ⭐⭐⭐⭐ |
| 数据增强 | 扩充训练数据 | ⭐⭐⭐⭐ |

---

## 💡 学习心得

### ✅ 应该做的
- 边学边练，每个概念都实践
- 先用框架，再看原理
- 多看开源项目，读优质代码
- 写博客总结，输出倒逼输入

### ❌ 不应该做的
- 不要一上来就读原论文
- 不要只调参不理解原理
- 不要闭门造车不交流
- 不要追求速成不扎实

---

## 🔗 扩展资源

### 视频课程
- Stanford CS231n: CNN for Visual Recognition
- Stanford CS224n: NLP with Deep Learning
- MIT 6.S191: Introduction to Deep Learning

### 在线平台
- Kaggle: https://www.kaggle.com
- Google Colab: https://colab.research.google.com
- Papers With Code: https://paperswithcode.com

### 社区
- GitHub: 开源项目
- Reddit: r/MachineLearning
- Stack Overflow: 技术问答

---

*本教程约贡献50KB高质量机器学习知识*

*学习永无止境，持续进步！* 🚀

