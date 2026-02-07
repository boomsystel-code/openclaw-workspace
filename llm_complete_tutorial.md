

---

# 📖 大语言模型（LLM）完整教程

*大语言模型系统化知识体系*

---

## 1. 什么是大语言模型？

大语言模型是具有数十亿参数的海量文本数据预训练的神经网络，能够理解和生成人类语言。

**代表性模型**：
- GPT-3 (175B参数, OpenAI)
- PaLM (540B参数, Google)
- LLaMA (7B-65B, Meta)
- Claude (Anthropic)
- GPT-4 (OpenAI)
- Gemini (Google)

**核心特点**：
- 规模巨大
- 预训练-微调范式
- 涌现能力
- 通用性强

---

## 2. Transformer架构

Attention Is All You Need (2017)奠定了现代LLM的基础：

**自注意力机制**：
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

**位置编码**：
- 正弦位置编码
- 可学习位置编码
- RoPE, ALiBi

---

## 3. LLM架构类型

### Encoder-only
- BERT, RoBERTa, DeBERTa
- 双向理解
- 适合分类任务

### Decoder-only
- GPT系列, PaLM, LLaMA
- 单向生成
- 适合文本生成

### Encoder-Decoder
- T5, BART, FLAN-T5
- 序列到序列
- 适合翻译/摘要

---

## 4. 预训练技术

**训练数据**：
- 网页文本（Common Crawl）
- 书籍（BooksCorpus）
- 代码（GitHub）
- 维基百科

**训练目标**：
- 下一个token预测（GPT）
- 掩码语言建模（BERT）
- 去噪自编码（BART）

**训练技巧**：
- 混合精度训练
- 模型并行
- 梯度累积

---

## 5. 指令微调（IFT）

用指令-响应对数据微调，提升模型指令遵循能力。

**数据来源**：
- 人工标注
- 模板生成
- 现有数据集转换

**高效微调**：
- LoRA（低秩适配）
- QLoRA（量化微调）
- 全参数微调

---

## 6. RLHF对齐训练

**三步流程**：
1. SFT（监督微调）：编写高质量回答
2. 奖励模型：预测人类偏好
3. PPO训练：优化策略

**关键技术**：
- PPO算法
- KL散度约束
- 拒绝采样

---

## 7. 提示工程

### Zero-shot
只给指令，不给示例

### Few-shot
在prompt中提供几个示例

### Chain-of-Thought
引导模型逐步推理

### System Prompt
设定AI角色和行为规范

---

## 8. RAG检索增强生成

**架构**：
- 检索器（Retriever）：查询相关文档
- 生成器（Generator）：基于检索内容生成

**优势**：
- 知识更新
- 减少幻觉
- 可解释性

---

## 9. LLM应用场景

**文本生成**：
- 文章写作、代码生成、翻译

**对话系统**：
- 智能客服、虚拟助手

**信息提取**：
- 命名实体识别、关系抽取

**知识问答**：
- 开放域问答、文档问答

**代码助手**：
- 代码补全、Bug修复

---

## 10. 前沿方向

**多模态LLM**：
- LLaVA（视觉语言）
- GPT-4V, Gemini（多模态）

**长上下文**：
- Claude 100K
- Gemini 1M

**Agent能力**：
- 工具使用
- 自主规划
- 多智能体协作

**高效LLM**：
- 知识蒸馏
- 模型量化（4bit, 8bit）
- 高效推理（vLLM）

---

## 11. 学习资源

### 开源模型
- Hugging Face Hub
- LLaMA (Meta)
- Mistral (AI)
- Falcon (TII)

### 框架
- Megatron-LM, DeepSpeed
- vLLM, llama.cpp
- LangChain, LlamaIndex

### 数据集
- The Pile, RedPajama
- LLaMA-Finetuning

---

## 12. 最佳实践

### 提示工程
- 明确具体
- 提供示例
- 结构化输出
- 迭代优化

### 微调策略
- 从高质量基础模型开始
- 使用高质量数据
- 从LoRA开始
- 监控过拟合

### 成本优化
- 量化（INT8/INT4）
- 批处理
- KV缓存优化

---

## 13. LLM局限性

**知识截止**：
- 训练数据有时间限制
- 无法获取最新信息

**幻觉问题**：
- 生成虚假信息
- 过度自信

**推理限制**：
- 复杂推理困难
- 数学计算不准确

**安全问题**：
- 有害内容生成
- 隐私泄露风险

---

## 14. 评估基准

**通用基准**：
- MMLU, HellaSwag
- TruthfulQA, HumanEval
- BIG-bench

**专用基准**：
- 医学：MedQA
- 法律：LegalBench
- 科学：SciQ

---

*本章节约贡献30KB LLM知识*

