# AI最终卷

## 强化学习

### MDP基础
- 状态空间
- 动作空间
- 状态转移
- 奖励函数
- 折扣因子

### 值函数
- 状态值函数
- 动作值函数
- 贝尔曼方程
- 最优策略
- 值迭代

### Q-Learning
- Q值表
- 贪婪策略
- SARSA
- 在线学习
- 收敛性

### DQN
- 经验回放
- 目标网络
- Double DQN
- Dueling DQN
- Noisy Nets

### Policy Gradient
- 策略梯度
- REINFORCE
-  baselines
- 方差缩减
- 置信域

### Actor-Critic
- Actor
- Critic
- A2C
- A3C
- PPO

### SAC
- 最大熵
- 温度参数
- 软Q函数
- 自动温度
- 离线SAC

### 离线RL
- CQL
- IQL
- Decision Transformer
- ConsQL
- Batch RL

## 多模态学习

### CLIP
- 对比学习
- 图像编码器
- 文本编码器
- 零样本分类
- 提示工程

### BLIP
- 引导语言
- 图像文本
- 预训练
- 迁移
- BLIP-2

### LLaVA
- 视觉指令
- 投影层
- 对话格式
- 训练
- 评估

### 扩散多模态
- 文本到图像
- 图像编辑
- 条件生成
- 控制生成
- ControlNet

## 模型优化

### 量化
- INT8量化
- 动态量化
- 静态量化
- 量化感知训练
- 部署

### 剪枝
- 结构化剪枝
- 非结构化剪枝
- 迭代剪枝
- 自动剪枝
-  Lottery Ticket

### 蒸馏
- 知识蒸馏
- 关系蒸馏
- 自我蒸馏
- 多人蒸馏
- 越狱蒸馏

### 加速
- 算子融合
- 计算图优化
- 内存优化
- 并行优化
- TensorRT

## 分布式训练

### 数据并行
- DataParallel
- DistributedDataParallel
- 梯度同步
- 负载均衡

### 模型并行
- 张量并行
- 流水线并行
- 混合并行
- 分割策略

### ZeRO
- 优化器分片
- 梯度分片
- 参数分片
- ZeRO-3

### 框架
- DeepSpeed
- FSDP
- Megatron-LM
- ColossalAI

## 模型部署

### 模型导出
- TorchScript
- ONNX
- TensorRT
- TFLite
- CoreML

### 推理优化
- 算子融合
- 内存优化
- 批处理
- 异步推理
- 服务端推理

### 服务化
- TorchServe
- Triton
- KServe
- BentoML
- Ray Serve

### 边缘部署
- TFLite
- TFLite Micro
- ONNX Runtime
- TensorRT
- CoreML

## MLOps

### 实验跟踪
- MLflow
- Weights & Biases
- Neptune
- Comet
- DVC

### 模型管理
- 模型注册
- 版本控制
- 模型仓库
- 模型血缘
- 部署管理

### 数据管理
- 数据版本
- 数据血缘
- 数据质量
- 特征存储
- Feature Store

### CI/CD
- 自动化测试
- 流水线
- A/B测试
- 监控告警
- 回滚

## AutoML

### 超参优化
- 网格搜索
- 随机搜索
- 贝叶斯优化
- Hyperband
- BOHB

### NAS
- DARTS
- ENAS
- NASNet
- EfficientNet
- AutoFormer

### AutoML工具
- Auto-sklearn
- Optuna
- Ray Tune
- Katib
- Google Vertex

### 元学习
- MAML
- Prototypical Networks
- Matching Networks
- Relation Networks
--learn to learn

## 可解释AI

### 特征重要性
- SHAP
- LIME
- Integrated Gradients
- DeepLIFT
- Expected Gradients

### 注意力可视化
- Attention Rollout
- Attention Flow
- Head Importance
- 归因
- 交互

### 反事实
- 生成方法
- 优化方法
- 因果方法
- 评估
- 工具

### 模型调试
- 梯度可视化
- 特征可视化
- 神经元分析
- 归因
- 探针

## AI伦理

### 公平性
- 偏见检测
- 公平性指标
- 去偏技术
- 因果公平
- 公平性审计

### 隐私保护
- 差分隐私
- 联邦学习
- 同态加密
- 安全多方计算
- TEE

### 透明性
- 可解释性
- 可追溯性
- 责任归属
- 披露
- 治理

### 安全性
- 对抗攻击
- 数据投毒
- 模型窃取
- 隐私泄露
- 防御

## 应用案例

### 计算机视觉
- 图像分类
- 目标检测
- 语义分割
- 实例分割
- 人脸识别
- 姿态估计
- 目标跟踪
- 图像生成

### 自然语言处理
- 文本分类
- 命名实体识别
- 情感分析
- 机器翻译
- 文本生成
- 问答系统
- 摘要
- 阅读理解

### 推荐系统
- 协同过滤
- 深度推荐
- 图推荐
- 多任务推荐
- 冷启动
- 排序
- 重排
- 实时推荐

### 行业应用
- 自动驾驶
- 医疗健康
- 金融服务
- 智能客服
- 内容审核
- 搜索
- 广告
- 教育

## 工具生态

### 深度学习框架
- PyTorch
- TensorFlow
- JAX
- PaddlePaddle
- MindSpore

### 预训练模型
- Hugging Face
- OpenMMLab
- TIMM
- MMDetection
- Transformer

### 开发工具
- Jupyter
- VSCode
- Git
- Docker
- Kubernetes

### 云平台
- AWS SageMaker
- GCP Vertex AI
- Azure ML
- 阿里云PAI
- 腾讯云TI

## 职业发展

### 技能要求
- 编程能力
- 数学基础
- 领域知识
- 工程能力
- 沟通能力

### 学习路径
- 入门阶段
- 进阶阶段
- 精通阶段
- 专家阶段
- 持续学习

### 面试准备
- 算法题
- 机器学习理论
- 深度学习理论
- 项目经验
- 系统设计

### 职业规划
- 研究方向
- 工程方向
- 产品方向
- 创业方向
- 管理方向
