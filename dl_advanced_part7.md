# 🚀 深度学习高级技术 Part 7

*更多前沿技术与实践*

---

## 105. 模型蒸馏与压缩

### 105.1 知识蒸馏

```python
class KnowledgeDistillation:
    """知识蒸馏"""
    
    def __init__(self, teacher_model, student_model, 
                 temperature=2.0, alpha=0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def train_epoch(self, train_loader):
        self.student.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs, targets = batch
            
            # 教师预测（不更新）
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
                teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            
            # 学生预测
            student_logits = self.student(inputs)
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
            
            # 蒸馏损失（KL散度）
            distill_loss = F.kl_div(
                student_log_probs, 
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # 标准损失
            ce_loss = F.cross_entropy(student_logits, targets)
            
            # 总损失
            loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

class SelfDistillation:
    """自蒸馏"""
    
    def __init__(self, model):
        self.model = model
        self.ema_model = copy.deepcopy(model)
        
        # EMA参数
        self.ema_decay = 0.999
    
    def train_epoch(self, train_loader):
        self.model.train()
        
        for batch in train_loader:
            inputs, targets = batch
            
            # 当前模型预测
            logits = self.model(inputs)
            
            # EMA模型预测
            with torch.no_grad():
                ema_logits = self.ema_model(inputs)
            
            # 自蒸馏损失
            loss = self._self_distillation_loss(logits, targets, ema_logits)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新EMA
            self._update_ema()
        
        return loss.item()
    
    def _self_distillation_loss(self, logits, targets, ema_logits):
        # 标准CE损失
        ce_loss = F.cross_entropy(logits, targets)
        
        # 蒸馏损失
        student_log_probs = F.log_softmax(logits, dim=1)
        ema_probs = F.softmax(ema_logits, dim=1)
        distill_loss = F.kl_div(
            student_log_probs, 
            ema_probs,
            reduction='batchmean'
        )
        
        return ce_loss + 0.1 * distill_loss
    
    def _update_ema(self):
        for param, ema_param in zip(
            self.model.parameters(), 
            self.ema_model.parameters()
        ):
            ema_param.data = self.ema_decay * ema_param.data + \
                            (1 - self.ema_decay) * param.data
```

### 105.2 量化

```python
class DynamicQuantization:
    """动态量化"""
    
    def __init__(self, model):
        self.model = model
    
    def quantize(self):
        """量化模型"""
        import torch.quantization
        
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.LSTMCell, nn.GRUCell, nn.GRUCell},
            dtype=torch.qint8
        )
        
        return quantized_model

class StaticQuantization:
    """静态量化"""
    
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
    
    def quantize(self):
        """静态量化"""
        import torch.quantization
        
        # 准备量化
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准
        self.model.eval()
        for batch in self.dataloader:
            with torch.no_grad():
                self.model(batch)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(
            self.model, inplace=False
        )
        
        return quantized_model

class GPTQQuantization:
    """GPTQ量化"""
    
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
    
    def quantize(self, bits=4, perchannel=True):
        """GPTQ量化"""
        import cudaq
        
        # 初始化GPTQ
        gptq = cudaq.GPTQ(self.model)
        gptq.quantize(self.dataloader, bits=bits, perchannel=perchannel)
        
        # 保存量化模型
        gptq.save('quantized_model')
        
        return gptq.model
```

### 105.3 剪枝

```python
class MagnitudePruning:
    """幅度剪枝"""
    
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune(self):
        """剪枝"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 计算阈值
                weights = module.weight.data.abs()
                threshold = np.percentile(
                    weights.cpu().numpy(), 
                    self.pruning_ratio * 100
                )
                
                # 创建掩码
                mask = weights > threshold
                
                # 应用剪枝
                module.weight.data = module.weight.data * mask.float()
                module.weight.grad = None
    
    def iterative_pruning(self, epochs, prune_interval):
        """迭代剪枝"""
        for epoch in range(epochs):
            # 训练
            self.train_epoch()
            
            # 剪枝
            if epoch % prune_interval == 0:
                self.prune()
                print(f"Epoch {epoch}: Pruned {self.pruning_ratio * 100}%")

class StructuredPruning:
    """结构化剪枝"""
    
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune_conv_channels(self):
        """剪枝卷积通道"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 计算每个通道的L1范数
                channel_norms = module.weight.data.abs().sum(dim=(1, 2, 3))
                
                # 选择要剪枝的通道
                num_prune = int(len(channel_norms) * self.pruning_ratio)
                prune_indices = torch.argsort(channel_norms)[:num_prune]
                
                # 更新输入通道
                new_in_channels = module.in_channels - num_prune
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    new_in_channels,
                    module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                )
                
                # 复制保留的通道
                keep_indices = [i for i in range(module.in_channels) 
                               if i not in prune_indices]
                new_conv.weight.data = module.weight.data[keep_indices]
                
                # 替换
                setattr(self.model, name, new_conv)
```

---

## 106. 分布式训练深入

### 106.1 FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

class FSDPTrainer:
    """FSDP训练器"""
    
    def __init__(self, model, optimizer, lr=0.01):
        # 包装模型
        self.model = FSDP(model)
        
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
    
    def train_epoch(self, dataloader):
        self.model.train()
        
        for batch in dataloader:
            # 移动到GPU
            batch = batch.cuda()
            
            # 前向
            loss = self.model(batch)
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            
            # 步骤
            self.optimizer.step()
    
    def save_checkpoint(self, path):
        """保存检查点"""
        FSDP.save_state_dict(
            self.model.state_dict(),
            path,
            rank0_only=True
        )
    
    def load_checkpoint(self, path):
        """加载检查点"""
        FSDP.load_state_dict(
            self.model.state_dict(),
            path
        )

class FSDPConfig:
    """FSDP配置"""
    
    def __init__(self):
        self.sharding_strategy = 'FULL_SHARD'  # 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'
        self.backward_prefetch = 'PRE_FORWARD'  # 'PRE_FORWARD', 'POST_FORWARD'
        self.forward_prefetch = True
        self.activation_checkpointing = True
        self.cpu_offload = False
```

### 106.2 DeepSpeed

```python
import deepspeed

class DeepSpeedTrainer:
    """DeepSpeed训练器"""
    
    def __init__(self, model, args):
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            args=args
        )
    
    def train_epoch(self, dataloader):
        self.model.train()
        
        for batch in dataloader:
            # 前向
            loss = self.model(batch)
            
            # 反向
            self.model.backward(loss)
            self.model.step()
    
    def save_checkpoint(self, tag):
        """保存检查点"""
        self.model.save_checkpoint(tag)

# DeepSpeed配置文件
DEEPSPEED_CONFIG = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True
    }
}
```

### 106.3 Megatron-LM

```python
class MegatronTrainer:
    """Megatron-LM训练器"""
    
    def __init__(self, model, args):
        self.model = model
        self.args = args
    
    def train_epoch(self, train_dataloader):
        # 设置
        self.model.set_train_batch_size(args.global_batch_size)
        
        # 迭代
        for iteration, batch in enumerate(train_dataloader):
            # 等待数据
            batch = self._get_batch(batch)
            
            # 前向
            loss = self.model(batch)
            
            # 反向
            self.model.backward(loss)
            
            # 优化器步骤
            self.model.step()
    
    def _get_batch(self, batch):
        """准备批次数据"""
        # 实现数据并行和模型并行的数据分割
        return batch

class PipelineParallelism:
    """流水线并行"""
    
    def __init__(self, model, devices):
        self.devices = devices
        self.model = model
        self.split_sizes = self._calculate_split_sizes()
    
    def _calculate_split_sizes(self):
        """计算分割大小"""
        total_params = sum(p.numel() for p in self.model.parameters())
        per_device = total_params // len(self.devices)
        
        # 基于层的分割
        layer_sizes = []
        for name, param in self.model.named_parameters():
            layer_sizes.append((name, param.numel()))
        
        return layer_sizes
```

---

## 107. 视觉Transformer深入

### 107.1 ViT变体

```python
class SwinTransformer(nn.Module):
    """Swin Transformer"""
    
    def __init__(self, img_size=224, patch_size=4, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        
        # 多个阶段
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(4):
            stage = SwinTransformerBlock(
                dim=embed_dim * (2 ** i),
                num_heads=num_heads[i],
                window_size=7,
                depth=depths[i]
            )
            self.stages.append(stage)
            
            if i < 3:
                downsample = PatchMerging(
                    dim=embed_dim * (2 ** i),
                    out_dim=embed_dim * (2 ** (i + 1))
                )
                self.downsample_layers.append(downsample)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim * 16)
        self.head = nn.Linear(embed_dim * 16, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            if i < 3:
                x = self.downsample_layers[i](x)
        
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)

class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    
    def __init__(self, dim, num_heads, window_size=7, depth=2):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = SwinAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size
            )
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ShiftedWindowAttention(nn.Module):
    """移位窗口注意力"""
    
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, x):
        B, H, W, C = x.shape
        
        # 移位窗口
        x = self._shifted_window(x)
        
        # 窗口分区
        x = self._window_partition(x)
        
        # 注意力
        x = self.attention(x, x, x)
        
        # 窗口合并
        x = self._window_reverse(x)
        
        return x
    
    def _shifted_window(self, x):
        """移位窗口"""
        shift_size = self.window_size // 2
        return torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
```

### 107.2 DeiT

```python
class DeiT(nn.Module):
    """Data-efficient Image Transformers"""
    
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        
        # Transformer
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # 蒸馏token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # 添加token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        # 分类
        return self.head(x[:, 0]), self.head_dist(x[:, 1])
```

### 107.3 视觉Prompt Tuning

```python
class VisualPromptTuning(nn.Module):
    """视觉Prompt Tuning"""
    
    def __init__(self, embed_dim=768, num_prompts=5, num_classes=1000):
        super().__init__()
        
        # 可学习的提示
        self.prompts = nn.Parameter(
            torch.randn(num_prompts, embed_dim) * 0.02
        )
        
        # 冻结的预训练模型
        self.backbone = load_pretrained_vit()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, images):
        # 获取图像特征
        features = self.backbone(images)
        
        # 添加提示
        batch_size = features.size(0)
        prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接
        x = torch.cat([features[:, :1], prompts, features[:, 1:]], dim=1)
        
        # 通过Transformer
        x = self.backbone.encoder(x)
        
        # CLS token用于分类
        return self.head(x[:, 0])
```

---

## 108. 扩散模型高级

### 108.1 Stable Diffusion XL

```python
class StableDiffusionXL(nn.Module):
    """Stable Diffusion XL"""
    
    def __init__(self, unet, vae, text_encoder, text_encoder_2):
        super().__init__()
        
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        # 比例因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    
    def encode_prompt(self, prompt, prompt_2=None):
        """编码提示"""
        # CLIP文本编码器1
        prompt_embeds = self.text_encoder(
            prompt, output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        
        # CLIP文本编码器2
        prompt_embeds_2 = self.text_encoder_2(
            prompt_2 or prompt, output_hidden_states=True
        )
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        # 拼接
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds
    
    def decode_latents(self, latents):
        """解码潜在变量"""
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        return images
    
    def train_step(self, prompt, image):
        """训练步骤"""
        # 编码图像
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],))
        noisy_latents = self._add_noise(latents, noise, timesteps)
        
        # 编码提示
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt)
        
        # 预测噪声
        noise_pred = self.unet(
            noisy_latents, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={'text_embeds': pooled_prompt_embeds}
        ).sample
        
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    def _add_noise(self, latents, noise, timesteps):
        """添加噪声"""
        return self.scheduler.add_noise(latents, noise, timesteps)
```

### 108.2 ControlNet

```python
class ControlNetConditioningEmbedding(nn.Module):
    """ControlNet条件编码"""
    
    def __init__(self, conditioning_channels=3, image_size=512, 
                 block_out_channels=(16, 32, 96, 256)):
        super().__init__()
        
        self.conv_in = nn.Conv2d(conditioning_channels, 16, 3, padding=1)
        
        self.blocks = nn.ModuleList()
        for i in range(len(block_out_channels)):
            block = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, block_out_channels[i], 3, padding=1, stride=2)
            )
            self.blocks.append(block)
        
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(block_out_channels[i], block_out_channels[i], 1)
            for i in range(len(block_out_channels))
        ])
    
    def forward(self, conditioning):
        # 编码
        feature_maps = []
        x = self.conv_in(conditioning)
        
        for block, zero_conv in zip(self.blocks, self.zero_convs):
            x = block(x)
            feature_maps.append(zero_conv(x))
        
        return feature_maps

class ControlledUNet(nn.Module):
    """带ControlNet的U-Net"""
    
    def __init__(self, base_model, controlnet):
        super().__init__()
        
        self.base_model = base_model
        self.controlnet = controlnet
        
        # 时间步条件
        self.time_embed = base_model.time_embed
        self.add_time_proj = base_model.add_time_proj
        self.add_position_norm = base_model.add_position_norm
    
    def forward(self, sample, timestep, conditioning):
        # ControlNet条件
        controlnet_residuals = self.controlnet(conditioning)
        
        # 时间步
        timesteps_proj = self.add_time_proj(timestep)
        timesteps_proj = self.time_embed(timesteps_proj)
        
        # 主UNet
        return self.base_model(
            sample, 
            timesteps_proj,
            controlnet_residuals=controlnet_residuals
        )
```

### 108.3 图像编辑

```python
class ImageEditingPipeline:
    """图像编辑流水线"""
    
    def __init__(self, sd_model, edit_model):
        self.sd_model = sd_model
        self.edit_model = edit_model
    
    def edit(self, image, source_prompt, target_prompt, strength=0.5):
        """编辑图像"""
        # 编码源图像
        latents = self.sd_model.vae.encode(image).latent_dist.sample()
        latents = latents * self.sd_model.vae.config.scaling_factor
        
        # 添加噪声（根据strength）
        noise = torch.randn_like(latents)
        timesteps = int((1 - strength) * 1000)
        noisy_latents = self.sd_model.scheduler.add_noise(
            latents, noise, 
            torch.tensor([timesteps] * len(latents))
        )
        
        # 编码提示
        target_embeds = self.sd_model.encode_prompt(target_prompt)
        
        # 扩散
        edited_latents = self.sd_model.unet(
            noisy_latents,
            torch.tensor([timesteps]),
            encoder_hidden_states=target_embeds
        ).sample
        
        # 解码
        edited_image = self.sd_model.decode_latents(edited_latents)
        
        return edited_image
```

---

## 109. 自动驾驶深度学习

### 109.1 BEV感知

```python
class BEVFormer(nn.Module):
    """BEVFormer"""
    
    def __init__(self, encoder, decoder, embed_dim=200):
        super().__init__()
        
        # 图像编码器
        self.encoder = encoder
        
        # BEV查询
        self.bev_embed = nn.Embedding(200 * 200, embed_dim)
        
        # 临时注意力
        self.temporal_attention = TemporalSelfAttention(embed_dim)
        
        # BEV解码器
        self.decoder = decoder
        
        # 检测头
        self.bbox_head = DetectionHead(embed_dim)
    
    def forward(self, images, timestamps):
        batch_size = images.size(0)
        
        # 多视角特征提取
        img_features = self.encoder(images)
        
        # 生成BEV查询
        bev_queries = self.bev_embed.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # 时间融合
        bev_features = self.temporal_attention(
            bev_queries, img_features, timestamps
        )
        
        # 解码
        outputs = self.decoder(bev_features)
        
        # 检测
        predictions = self.bbox_head(outputs)
        
        return predictions

class TemporalSelfAttention(nn.Module):
    """时间自注意力"""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, bev_queries, img_features, timestamps):
        # 收集历史特征
        history_features = self._collect_history(img_features, timestamps)
        
        # 时间注意力
        fused = torch.cat([bev_queries] + history_features, dim=1)
        
        attended, _ = self.attention(bev_queries, fused, fused)
        
        return self.norm(bev_queries + attended)
```

### 109.2 端到端自动驾驶

```python
class EndToEndAutonomousDriving(nn.Module):
    """端到端自动驾驶"""
    
    def __init__(self, perception, planning, control):
        super().__init__()
        
        self.perception = perception
        self.planning = planning
        self.control = control
    
    def forward(self, sensors_data):
        """端到端推理"""
        # 感知
        scene_features = self.perception(sensors_data)
        
        # 规划
        trajectory = self.planning(scene_features)
        
        # 控制
        control_signals = self.control(trajectory)
        
        return control_signals
    
    def train_step(self, sensors_data, expert_actions):
        """训练步骤"""
        # 感知损失
        perception_loss = self._perception_loss(sensors_data)
        
        # 规划损失
        planning_loss = self._planning_loss(sensors_data, expert_actions)
        
        # 控制损失
        control_loss = self._control_loss(sensors_data, expert_actions)
        
        # 总损失
        total_loss = (
            perception_loss + 
            0.5 * planning_loss + 
            0.1 * control_loss
        )
        
        return total_loss
```

---

## 110. 蛋白质与生物AI

### 110.1 AlphaFold2实现

```python
class AlphaFold2(nn.Module):
    """AlphaFold2"""
    
    def __init__(self, config):
        super().__init__()
        
        # MSA堆栈
        self.msa_stack = MSAStack(config)
        
        # 配对表示
        self.pair_stack = PairRepresentationStack(config)
        
        # 三角注意
        self.triangle_attention = TriangleAttention(config)
        
        # 头部模块
        self.head = StructureModule(config)
    
    def forward(self, msa, pair, single):
        # MSA更新
        msa = self.msa_stack(msa, pair)
        
        # 配对表示更新
        pair = self.pair_stack(pair)
        pair = self.triangle_attention(pair)
        
        # 3D结构模块
        outputs = self.head(single, pair)
        
        return outputs

class MSAStack(nn.Module):
    """MSA堆栈"""
    
    def __init__(self, config):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MSAColumnAttention(config)
            for _ in range(config.msa_depth)
        ])
    
    def forward(self, msa, pair):
        for layer in self.layers:
            msa = layer(msa, pair)
        return msa

class StructureModule(nn.Module):
    """结构模块"""
    
    def __init__(self, config):
        super().__init__()
        
        self.single_layer_norm = nn.LayerNorm(config.hidden_size)
        self.pair_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 注意力
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size),
            num_layers=3
        )
        
        # IPA头
        self.ipa = InvariantPointAttention(config)
        
        # 骨架头
        self.backbone_head = BackboneHead(config)
    
    def forward(self, single, pair):
        # 特征融合
        x = self.single_layer_norm(single)
        x = self.attention(x)
        
        # IPA
        x, angles = self.ipa(x, pair)
        
        # 骨架
        backbone = self.backbone_head(x)
        
        return {
            'frames': backbone['frames'],
            'angles': angles,
            'sidechains': backbone['sidechains']
        }
```

### 110.2 ESM-2

```python
class ESM2(nn.Module):
    """ESM-2蛋白质语言模型"""
    
    def __init__(self, num_layers=33, embed_dim=1280, 
                 attention_heads=20):
        super().__init__()
        
        # 嵌入
        self.embed = nn.Embedding(33, embed_dim)
        
        # 位置编码
        self.pos_embed = RotaryEmbedding(embed_dim)
        
        # Transformer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=attention_heads,
                dim_feedforward=embed_dim * 4
            )
            for _ in range(num_layers)
        ])
        
        # 预训练头
        self.lm_head = nn.Linear(embed_dim, 33)
        self.contact_head = nn.Linear(embed_dim, 1)
    
    def forward(self, tokens, return_repr=False):
        # 嵌入
        x = self.embed(tokens)
        
        # 位置编码
        seq_len = tokens.size(1)
        cos, sin = self.pos_embed(seq_len, x.device)
        
        # Transformer
        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)
        
        # 头
        logits = self.lm_head(x)
        
        if return_repr:
            return x
        return logits
```

---

*本部分贡献约35KB高级知识*

**持续学习！目标10MB！** 🚀💪

