# AI最终知识补充

## 1. 注意力机制详解

### 1.1 标准注意力
- Query-Key-Value
- 缩放点积
- Softmax归一化
- 加权求和

### 1.2 多头注意力
- 多头并行
- 投影矩阵
- 拼接输出
- 应用场景

### 1.3 位置编码
- 正弦位置编码
- 可学习位置编码
- RoPE
- ALiBi

### 1.4 稀疏注意力
- 局部窗口
- 随机注意力
- 全局token
- Longformer/BigBird

## 2. Transformer架构

### 2.1 编码器
- Self-attention
- Feed-forward
- 残差连接
- 层归一化

### 2.2 解码器
- Masked self-attention
- Cross-attention
- 预测头

### 2.3 变体
- Encoder-only (BERT)
- Decoder-only (GPT)
- Encoder-Decoder (T5)

### 2.4 训练技巧
- 学习率预热
- 标签平滑
- Dropout
- 梯度裁剪

## 3. BERT与GPT

### 3.1 BERT
- MLM预训练
- NSP预训练
- 下游任务
- RoBERTa/DeBERTa

### 3.2 GPT系列
- 语言建模
- 零样本能力
- 指令微调
- RLHF

### 3.3 开源模型
- LLaMA
- Mistral
- Falcon
- Qwen

### 3.4 模型微调
- 全参数微调
- LoRA
- Prefix Tuning
- Prompt Tuning

## 4. 扩散模型

### 4.1 基础理论
- 前向扩散
- 反向去噪
- 噪声调度
- 损失函数

### 4.2 采样算法
- DDPM
- DDIM
- PLMS
- Euler

### 4.3 条件生成
- 文本条件
- 图像条件
- 边缘条件
- ControlNet

### 4.4 高效扩散
- 潜在扩散
- 知识蒸馏
- 推理加速
- LoRA

## 5. 强化学习

### 5.1 值函数方法
- Q-Learning
- SARSA
- DQN
- Double DQN

### 5.2 策略梯度
- REINFORCE
- Actor-Critic
- A2C
- PPO

### 5.3 SAC
- 最大熵RL
- 自动温度
- 软Q函数
- 离线SAC

### 5.4 离线RL
- CQL
- IQL
- Decision Transformer
- Batch RL

## 6. 多模态学习

### 6.1 CLIP
- 对比学习
- 图像文本对
- 零样本分类
- 提示工程

### 6.2 视觉语言模型
- LLaVA
- MiniGPT-4
- Kosmos
- Fuyu

### 6.3 图像生成
- DALL-E
- Stable Diffusion
- ControlNet
- IP-Adapter

### 6.4 视频理解
- VideoBERT
- TimeSformer
- VideoMAE
- VideoLlama

## 7. 模型优化

### 7.1 量化
- INT8量化
- 动态/静态量化
- 量化感知训练
- GPTQ/AWQ

### 7.2 剪枝
- 结构化剪枝
- 非结构化剪枝
- 迭代剪枝
- Lottery Ticket

### 7.3 蒸馏
- 知识蒸馏
- 关系蒸馏
- 自蒸馏
- 越狱蒸馏

### 7.4 高效架构
- MobileNet
- EfficientNet
- ConvNeXt
- Swin Transformer

## 8. 分布式训练

### 8.1 数据并行
- DataParallel
- DistributedDataParallel
- 梯度同步

### 8.2 模型并行
- 张量并行
- 流水线并行
- 混合并行

### 8.3 ZeRO
- 优化器分片
- 梯度分片
- 参数分片

### 8.4 框架
- DeepSpeed
- FSDP
- Megatron-LM
- ColossalAI

## 9. 模型部署

### 9.1 模型导出
- TorchScript
- ONNX
- TensorRT
- TFLite

### 9.2 推理优化
- 算子融合
- 内存优化
- 批处理
- 异步

### 9.3 服务化
- TorchServe
- Triton
- KServe
- BentoML

### 9.4 边缘部署
- TFLite
- ONNX Runtime
- CoreML
- TensorRT

## 10. MLOps

### 10.1 实验跟踪
- MLflow
- Weights & Biases
- Neptune
- Comet

### 10.2 模型管理
- 模型注册
- 版本控制
- 血缘追踪
- A/B测试

### 10.3 CI/CD
- 自动化测试
- 流水线
- 监控告警
- 回滚

### 10.4 数据管理
- 数据版本
- 数据质量
- 特征存储
- Feature Store

## 11. AutoML

### 11.1 超参优化
- 网格搜索
- 随机搜索
- 贝叶斯优化
- Hyperband

### 11.2 NAS
- DARTS
- ENAS
- EfficientNet
- ProxylessNAS

### 11.3 AutoML工具
- Auto-sklearn
- Optuna
- Ray Tune
- Katib

### 11.4 元学习
- MAML
- Prototypical
- Matching
- Relation

## 12. 可解释AI

### 12.1 特征重要性
- SHAP
- LIME
- Integrated Gradients
- DeepLIFT

### 12.2 注意力可视化
- Attention Rollout
- Head Importance
- 归因

### 12.3 反事实
- 生成方法
- 优化方法
- 评估

### 12.4 模型调试
- 梯度可视化
- 特征可视化
- 神经元分析

## 13. AI伦理

### 13.1 公平性
- 偏见检测
- 公平性指标
- 去偏技术
- 因果公平

### 13.2 隐私保护
- 差分隐私
- 联邦学习
- 同态加密
- 安全计算

### 13.3 透明性
- 可解释性
- 责任归属
- 披露
- 治理

### 13.4 安全性
- 对抗攻击
- 数据投毒
- 模型窃取
- 防御

## 14. 应用案例

### 14.1 CV应用
- 图像分类
- 目标检测
- 语义分割
- 实例分割
- 人脸识别
- 姿态估计

### 14.2 NLP应用
- 文本分类
- 命名实体识别
- 情感分析
- 机器翻译
- 文本生成
- 问答系统

### 14.3 推荐系统
- 协同过滤
- 深度推荐
- 图推荐
- 多任务推荐
- 冷启动

### 14.4 行业应用
- 自动驾驶
- 医疗健康
- 金融服务
- 智能客服
- 内容审核

## 15. 工具生态

### 15.1 框架
- PyTorch
- TensorFlow
- JAX
- PaddlePaddle

### 15.2 库
- Hugging Face
- OpenMMLab
- TIMM
- LangChain

### 15.3 开发工具
- Jupyter
- VSCode
- Git
- Docker

### 15.4 云平台
- AWS SageMaker
- GCP Vertex AI
- Azure ML
- 阿里云PAI

## 16. 职业发展

### 16.1 技能要求
- 编程能力
- 数学基础
- 领域知识
- 工程能力

### 16.2 学习路径
- 入门阶段
- 进阶阶段
- 精通阶段
- 专家阶段

### 16.3 面试准备
- 算法题
- 理论题
- 项目题
- 系统设计

### 16.4 职业规划
- 研究方向
- 工程方向
- 产品方向
- 创业方向

---

**AI知识补充完成！** 📚

**接近10MB目标！** 🚀💪
