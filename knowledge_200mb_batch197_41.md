# AI Knowledge 200MB - 197_41

## Deep Learning Foundations

### Neural Networks
- Perceptron and MLP
- Activation functions
- Backpropagation
- Initialization strategies
- Regularization techniques

### Optimization
- Gradient descent variants
- Adaptive optimizers (Adam, AdaGrad, RMSprop)
- Learning rate scheduling
- Gradient clipping
- Second-order methods

### Generalization
- Bias-variance tradeoff
- Overfitting and underfitting
- Cross-validation
- Early stopping
- Data augmentation

### Normalization
- Batch Normalization
- Layer Normalization
- Instance Normalization
- Group Normalization
- Weight Normalization

## Transformer Architecture

### Attention Mechanisms
- Self-attention
- Multi-head attention
- Scaled dot-product attention
- Local and sparse attention
- Linear attention variants

### Position Encoding
- Absolute positional encoding
- Relative positional encoding
- Rotary Position Embedding (RoPE)
- ALiBi (Attention with Linear Biases)
- Learned position embeddings

### Architecture Variants
- BERT (Encoder-only)
- GPT (Decoder-only)
- T5 (Encoder-decoder)
- Encoder-decoder transformers
- Mixture of Experts (MoE)

### Training Techniques
- Warmup strategies
- Mixed precision training
- Gradient checkpointing
- Learning rate scaling
- Optimizer scheduling

## Large Language Models

### GPT Series
- GPT-1: 117M parameters
- GPT-2: 1.5B parameters
- GPT-3: 175B parameters
- GPT-4: Multimodal, 128K context
- Scaling laws and emergent abilities

### LLaMA Family
- LLaMA 1: 7B-65B parameters
- LLaMA 2: 7B-70B parameters
- LLaMA 3: 8B-405B parameters
- Open-source and commercial use

### Training Methods
- Next token prediction
- In-context learning
- Chain-of-thought reasoning
- Prompt engineering
- Few-shot and zero-shot learning

### Fine-tuning Techniques
- Full fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Prefix tuning
- Adapter layers

## Computer Vision

### Image Classification
- LeNet, AlexNet, VGG
- ResNet and variants
- EfficientNet (EfficientNet-B0 to B7)
- Vision Transformers (ViT)
- ConvNeXt

### Object Detection
- Two-stage: Faster R-CNN, Mask R-CNN
- One-stage: YOLO (YOLOv3-v8)
- Anchor-free: FCOS, CenterNet
- DETR (End-to-end detection)

### Semantic Segmentation
- FCN (Fully Convolutional Networks)
- U-Net (Medical imaging)
- DeepLab (Atrous convolution)
- PSPNet (Pyramid scene parsing)
- SegFormer (Transformer-based)

### Instance Segmentation
- Mask R-CNN
- YOLACT (Real-time instance segmentation)
- SOLOv2 (Segmenting Objects by Locations)

### Video Understanding
- 3D CNNs (I3D, C3D)
- Video Transformers (TimeSformer)
- Video Swin Transformer
- Action recognition (SlowFast networks)

### Image Generation
- GAN (DCGAN, StyleGAN)
- Diffusion Models (DDPM, Stable Diffusion)
- VAE (Variational Autoencoders)
- Text-to-Image (DALL-E, Midjourney)

## Natural Language Processing

### Language Models
- BERT (Bidirectional Encoder)
- RoBERTa (Robust BERT)
- DeBERTa (Decoupled attention)
- ELECTRA (Efficient pre-training)

### Text Classification
- Sentiment analysis
- Topic classification
- Spam detection
- Intent classification

### Named Entity Recognition
- BERT-CRF
- Span-based NER
- Few-shot NER
- Cross-lingual NER

### Question Answering
- Extractive QA (SQuAD)
- Generative QA
- Multi-hop QA
- Open-domain QA

### Machine Translation
- Sequence-to-sequence models
- Transformer-based translation
- Back-translation
- Multilingual translation

### Text Summarization
- Extractive summarization
- Abstractive summarization
- BART, PEGASUS
- Long-form summarization

## Generative Models

### GAN (Generative Adversarial Networks)
- Generator and discriminator
- Training dynamics and instability
- WGAN-GP (Wasserstein GAN)
- StyleGAN (Style-based generation)
- BigGAN (Large-scale GAN)

### VAE (Variational Autoencoders)
- Encoder-decoder architecture
- Reparameterization trick
- Beta-VAE (Disentangled representations)
- VQ-VAE (Vector Quantized VAE)
- VQ-VAE-2 (Hierarchical VQ-VAE)

### Diffusion Models
- DDPM (Denoising Diffusion Probabilistic Models)
- Score-based generative modeling
- Latent Diffusion Models (Stable Diffusion)
- Classifier-free guidance
- DDIM (Denoising Diffusion Implicit Models)

### Autoregressive Models
- Transformer language models
- Efficient autoregressive generation
- Beam search and nucleus sampling
- Speculative decoding

## Reinforcement Learning

### Fundamentals
- MDP (Markov Decision Process)
- Bellman equations
- Value functions and Q-values
- Policy gradients

### Value-Based Methods
- Q-Learning
- DQN (Deep Q-Networks)
- Double DQN
- Dueling DQN
- Prioritized experience replay

### Policy Gradient Methods
- REINFORCE algorithm
- Actor-Critic architecture
- A3C (Asynchronous Advantage AC)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

### Model-Based RL
- World models
- Model predictive control
- Dreamer (Learning from imagined futures)
- MBPO (Model-Based Policy Optimization)

### Offline RL
- Batch RL challenges
- CQL (Conservative Q-Learning)
- IQL (Implicit Q-Learning)
- Decision Transformers

### Multi-Agent RL
- QMIX (Value decomposition)
- MAPPO (Multi-Agent PPO)
- Emergent communication
- Adversarial training

## Multimodal Learning

### Vision-Language Models
- CLIP (Contrastive Language-Image Pre-training)
- BLIP (Bootstrapping Language-Image Pre-training)
- BLIP-2 (Q-Former architecture)
- LLaVA (Large Language and Vision Assistant)
- Flamingo (Few-shot learning)

### Text-to-Image Generation
- DALL-E (Autoregressive image generation)
- Stable Diffusion (Latent diffusion)
- ControlNet (Controllable generation)
- SDXL (High-resolution generation)
- Image-to-Image translation

### Video Understanding
- VideoBERT (BERT for video)
- VideoMAE (Masked autoencoder for video)
- Action recognition models
- Video generation models

### 3D Vision
- PointNet (Point cloud processing)
- NeRF (Neural Radiance Fields)
- Gaussian Splatting
- 3D reconstruction from images

### Audio-Visual Learning
- Audio-visual correspondence
- Sound source separation
- Voice activity detection
- Lip reading

## Model Optimization

### Quantization
- INT8 quantization
- INT4 quantization
- GPTQ (Post-training quantization)
- AWQ (Activation-aware quantization)
- SmoothQuant (Math-aware quantization)

### Pruning
- Magnitude pruning
- Gradient-based pruning
- Structured pruning
- Lottery Ticket Hypothesis
- Iterative pruning

### Knowledge Distillation
- Teacher-student framework
- Relation knowledge distillation
- Self-distillation
- Multi-teacher distillation
- Born-again networks

### Neural Architecture Search
- Search space design
- DARTS (Differentiable architecture search)
- ENAS (Efficient NAS)
- EfficientNet-NAS
- ProxylessNAS

### Efficient Architectures
- MobileNet (Depthwise separable conv)
- EfficientNet (Compound scaling)
- Swin Transformer (Hierarchical ViT)
- ConvNeXt (Modernized CNN)
- RegNet (Regularized residual networks)

## Distributed Systems

### Data Parallelism
- Synchronous training
- Asynchronous training
- Gradient accumulation
- Communication optimization

### Model Parallelism
- Tensor parallelism
- Pipeline parallelism
- Sequential parallelism
- Expert parallelism

### ZeRO Optimization
- ZeRO Stage 1: Optimizer state partitioning
- ZeRO Stage 2: Gradient partitioning
- ZeRO Stage 3: Parameter partitioning
- Offload strategies (CPU/NVMe)

### Training Frameworks
- DeepSpeed (Microsoft)
- FSDP (Fully Sharded Data Parallel)
- Megatron-LM (NVIDIA)
- ColossalAI (HPC-AI)

### Inference Optimization
- KV cache management
- Dynamic batching
- Continuous batching
- Speculative decoding

## MLOps

### Experiment Tracking
- MLflow (Open-source platform)
- Weights & Biases (W&B)
- Neptune.ai
- Comet ML

### Model Management
- Model versioning
- Model registry
- Model lineage tracking
- A/B testing infrastructure

### Feature Engineering
- Feature stores (Feast, Tecton)
- Feature engineering automation
- Feature selection
- Feature importance

### Deployment
- TorchServe (PyTorch)
- Triton Inference Server
- KServe (Kubernetes-native)
- BentoML (Unified ML serving)

### Monitoring
- Data drift detection
- Model drift monitoring
- Performance metrics
- Alerting systems

### CI/CD for ML
- Automated testing
- Data validation
- Model validation
- Shadow deployment

## AI Safety and Ethics

### Alignment
- RLHF (Reinforcement Learning from Human Feedback)
- Constitutional AI
- Value learning approaches
- Specification gaming

### Interpretability
- SHAP values (Shapley additive explanations)
- LIME (Local interpretable model-agnostic)
- Grad-CAM (Gradient-weighted class activation)
- Attention visualization
- Mechanistic interpretability

### Robustness
- Adversarial attacks (FGSM, PGD)
- Adversarial training
- Certified robustness
- Distribution shift

### Privacy
- Differential Privacy (DP-SGD)
- Federated learning
- Membership inference attacks
- Secure multi-party computation

### Fairness
- Bias detection
- Fairness metrics
- Debiasing techniques
- Algorithmic fairness

### Governance
- AI risk management
- Regulatory compliance
- Ethical guidelines
- Transparency requirements

## Emerging Technologies

### Foundation Models
- Scaling laws research
- Emergent abilities at scale
- Generalist agents
- Universal representations

### Agent Systems
- Tool use and function calling
- Multi-step reasoning
- Planning and exploration
- Memory and retrieval

### Embodied AI
- Robotics learning
- Autonomous vehicles
- Human-robot interaction
- Simulation environments

### Scientific AI
- AlphaFold (Protein structure prediction)
- Drug discovery (Molecular generation)
- Material science (Crystal structure)
- Climate modeling

### Quantum ML
- Quantum neural networks
- Variational quantum algorithms
- Quantum embeddings
- Quantum advantage research

## Best Practices

### Training
- Learning rate warmup
- Gradient clipping
- Weight decay
- Mixed precision

### Debugging
- Gradient checking
- Loss monitoring
- Learning curves
- Error analysis

### Evaluation
- Proper metric selection
- Cross-validation
- Statistical testing
- Ablation studies

### Production
- Model optimization
- Latency reduction
- Cost optimization
- Scalability

## Career Development

### Required Skills
- Programming (Python, PyTorch)
- Mathematics (Linear algebra, Probability)
- Deep learning theory
- System design

### Learning Path
- Foundation (3-6 months)
- Specialization (6-12 months)
- Mastery (1-2 years)
- Expertise (2+ years)

### Interview Preparation
- Algorithm questions
- ML theory questions
- System design
- Project discussions

### Career Tracks
- Research Scientist
- ML Engineer
- Data Scientist
- AI Product Manager

## Future Directions

### AGI Research
- Cognitive architectures
- Knowledge representation
- Reasoning abilities
- Learning efficiency

### AI Safety
- Alignment at scale
- Robustness research
- Interpretability advances
- Governance frameworks

### Industry Trends
- Automation of ML
- No-code AI
- Edge AI
- Sustainable AI

---

**AI Knowledge 200MB - 197_41**
