# 🚀 深度学习高级技术 Part 9

*前沿技术与实践大全*

---

## 119. 大规模模型训练

### 119.1 训练基础设施

```python
class TrainingInfrastructure:
    """训练基础设施"""
    
    def __init__(self, num_nodes=8, gpus_per_node=8):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = num_nodes * gpus_per_node
    
    def setup_nccl(self, backend='nccl'):
        """设置NCCL通信"""
        import torch.distributed as dist
        
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=self.total_gpus,
            rank=self._get_rank()
        )
        
        torch.cuda.set_device(self._get_local_rank())
    
    def setup_fsdp(self, model, optimizer_class=torch.optim.Adam):
        """设置FSDP"""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        model = FSDP(
            model,
            sharding_strategy='full_shard',
            cpu_offload=True,
            backward_prefetch='pre_forward'
        )
        
        optimizer = optimizer_class(model.parameters())
        
        return model, optimizer
    
    def save_checkpoint(self, model, optimizer, epoch, path):
        """保存检查点"""
        import torch.distributed.checkpoint as dist_cp
        
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        
        dist_cp.save_state_dict(
            state_dict=state_dict,
            checkpoint_id=path
        )
    
    def load_checkpoint(self, model, optimizer, path):
        """加载检查点"""
        import torch.distributed.checkpoint as dist_cp
        
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        dist_cp.load_state_dict(
            state_dict=state_dict,
            checkpoint_id=path
        )

class MixedPrecisionTraining:
    """混合精度训练"""
    
    def __init__(self, use_bf16=False):
        self.use_bf16 = use_bf16
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, model, optimizer, data, loss_fn):
        """训练步骤"""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
            output = model(data)
            loss = loss_fn(output)
        
        # 梯度缩放
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()
```

### 119.2 优化器

```python
class AdvancedOptimizers:
    """高级优化器"""
    
    @staticmethod
    def lion_optimizer(model, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01):
        """Lion优化器"""
        return Lion(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    
    @staticmethod
    def schedule_free_adam(model, lr=0.01, betas=(0.9, 0.999), weight_decay=0.01):
        """Schedule-Free Adam"""
        return ScheduleFreeAdam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

class Lion(nn.Module):
    """Lion优化器实现"""
    
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01):
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        
        self.exp_avg = [torch.zeros_like(p) for p in self.params]
    
    def step(self, closure=None):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad
            
            # 权重衰减
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            
            # 更新动量
            self.exp_avg[i] = self.beta1 * self.exp_avg[i] + (1 - self.beta1) * grad
            
            # 更新参数
            p.data = p.data - self.lr * torch.sign(self.exp_avg[i])
        
        return loss

class ScheduleFreeAdam:
    """Schedule-Free Adam"""
    
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        
        self.exp_avg = [torch.zeros_like(p) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]
        
        # y用于Schedule-Free
        self.y = [p.data.clone() for p in self.params]
    
    def step(self, closure=None):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad
            
            # y更新
            self.y[i] = (1 - self.beta1) * self.y[i] + self.beta1 * grad
            
            # 动量更新
            self.exp_avg[i] = self.beta2 * self.exp_avg[i] + (1 - self.beta2) * (grad ** 2)
            
            # 参数更新
            bias_correction = 1 - self.beta2 ** (i + 1)
            step_size = self.lr / bias_correction
            
            p.data = p.data - step_size * self.y[i] / (torch.sqrt(self.exp_avg[i]) + 1e-8)
```

### 119.3 学习率调度

```python
class AdvancedSchedulers:
    """高级调度器"""
    
    @staticmethod
    def cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                          num_cycles=0.5, lr_end=1e-5):
        """带预热的余弦调度"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(lr_end / 1.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def one_cycle(optimizer, max_lr, total_steps, pct_start=0.3):
        """One Cycle策略"""
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start
        )

class WarmupScheduler:
    """预热调度器"""
    
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # 预热阶段：线性增加
            lr = self.min_lr + (self.max_lr - self.min_lr) * self.current_step / self.warmup_steps
        else:
            # 余弦退火
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

---

## 120. 自监督学习深入

### 120.1 对比学习

```python
class SimMIM:
    """SimMIM: 简单的掩码图像建模"""
    
    def __init__(self, encoder, decoder, patch_size=4, mask_ratio=0.5):
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def forward(self, images):
        batch_size, _, height, width = images.shape
        
        # 生成掩码
        mask = self._generate_mask(height, width)
        
        # 编码
        features = self.encoder(images, mask)
        
        # 解码
        predictions = self.decoder(features)
        
        return predictions, mask
    
    def _generate_mask(self, height, width):
        """生成随机掩码"""
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        num_mask = int(num_patches * self.mask_ratio)
        
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
        mask = torch.zeros(num_patches)
        mask[mask_indices] = 1
        
        return mask
    
    def loss(self, predictions, targets, mask):
        """掩码重建损失"""
        # 只计算被掩码区域的损失
        loss = F.mse_loss(predictions[mask.bool()], targets[mask.bool()])
        return loss

class DINOv2:
    """DINOv2自监督学习"""
    
    def __init__(self, student, teacher, feature_dim=65536, 
                 momentum=0.996, warmup_teacher_temp=0.04,
                 teacher_temp=0.04, warmup_teacher_temp_iters=30):
        self.student = student
        self.teacher = teacher
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_iters = warmup_teacher_temp_iters
        self.student_temp = 0.1
        
        # 投影头
        self.student_proj = self._build_projection_head()
        self.teacher_proj = self._build_projection_head()
        
        # 初始化教师
        self._init_teacher()
    
    def _build_projection_head(self):
        """构建投影头"""
        return nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, self.feature_dim)
        )
    
    def _init_teacher(self):
        """初始化教师网络"""
        for param_q, param_k in zip(
            self.student.parameters(), 
            self.teacher.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    def forward(self, global_views, local_views):
        """前向传播"""
        student_output = self.student(global_views)
        teacher_output = self.teacher(global_views)
        
        # 计算损失
        loss = self._dino_loss(student_output, teacher_output)
        
        return loss
    
    def _dino_loss(self, student_out, teacher_out):
        """DINO损失"""
        # 教师温度调度
        temp = self.teacher_temp
        if self.global_step < self.warmup_iters:
            temp = self.warmup_teacher_temp
        
        # 计算中心
        teacher_center = self._center_momentum(teacher_out)
        
        # 对比损失
        student_exp = F.softmax(student_out / self.student_temp, dim=1)
        teacher_exp = F.softmax((teacher_out - teacher_center) / temp, dim=1)
        
        loss = -torch.sum(teacher_exp * torch.log(student_exp), dim=1).mean()
        
        return loss
    
    def _center_momentum(self, output):
        """中心动量更新"""
        with torch.no_grad():
            self.center = self.center * 0.9 + output.mean(dim=0) * 0.1
        return self.center
    
    def update_teacher(self):
        """更新教师网络"""
        for param_q, param_k in zip(
            self.student.parameters(),
            self.teacher.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
```

### 120.2 掩码自动编码

```python
class MAE:
    """掩码自动编码器"""
    
    def __init__(self, encoder, decoder, patch_size=16, mask_ratio=0.75):
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # 编码器投影
        self.encoder_to_decoder = nn.Linear(encoder.embed_dim, decoder.embed_dim)
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        # 提取patch
        patches = self._to_patches(images)
        
        # 生成掩码
        mask, ids_restore = self._generate_mask(patches)
        
        # 编码可见patch
        visible_patches = patches[~mask.bool()].reshape(batch_size, -1, patches.shape[-1])
        visible_features = self.encoder(visible_patches)
        
        # 投影到decoder空间
        decoder_features = self.encoder_to_decoder(visible_features)
        
        # 用mask token填充
        full_features = self._fill_mask_tokens(decoder_features, ids_restore, mask)
        
        # 解码
        predictions = self.decoder(full_features)
        
        return predictions, patches, mask
    
    def _to_patches(self, images):
        """转换为patch"""
        B, C, H, W = images.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        patches = images.reshape(B, C, num_patches_h, self.patch_size, 
                                num_patches_w, self.patch_size)
        patches = patches.permute(0, 2, 4, 0, 1, 3, 5)
        patches = patches.reshape(B * num_patches_h * num_patches_w, C, 
                                  self.patch_size, self.patch_size)
        return patches
    
    def _generate_mask(self, patches):
        """生成随机掩码"""
        N, D = patches.shape[0], patches.shape[1]
        num_visible = int(N * (1 - self.mask_ratio))
        
        ids_shuffle = torch.randperm(N)
        ids_restore = torch.argsort(ids_shuffle)
        
        ids_keep = ids_shuffle[:num_visible]
        mask = torch.ones(N)
        mask[ids_keep] = 0
        
        return mask, ids_restore
    
    def _fill_mask_tokens(self, features, ids_restore, mask):
        """填充mask token"""
        mask_tokens = self.mask_token.expand(features.shape[0], ids_restore.shape[0] - features.shape[1], -1)
        
        full_features = torch.cat([features, mask_tokens], dim=1)
        full_features = torch.gather(
            full_features, dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        )
        
        return full_features
    
    def loss(self, predictions, targets, mask):
        """MSE损失"""
        loss = F.mse_loss(predictions[mask.bool()], targets[mask.bool()])
        return loss
```

---

## 121. 多模态融合

### 121.1 音频-视觉

```python
class AudioVisualModel(nn.Module):
    """音视频多模态模型"""
    
    def __init__(self, audio_encoder, video_encoder, fusion_dim=512):
        super().__init__()
        
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        
        # 融合模块
        self.fusion = nn.MultiheadAttention(fusion_dim, num_heads=8)
        
        # 分类头
        self.classifier = nn.Linear(fusion_dim, 1000)
    
    def forward(self, audio, video):
        # 编码
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)
        
        # 交叉注意力融合
        fused, _ = self.fusion(
            audio_features, video_features, video_features
        )
        
        # 分类
        output = self.classifier(fused.mean(dim=1))
        
        return output

class AudioSetModel(nn.Module):
    """AudioSet模型"""
    
    def __init__(self, num_classes=527):
        super().__init__()
        
        # 波形编码器
        self.wave2vec = Wav2Vec2Model()
        
        # 频谱编码器
        self.spectrogram = SpectrogramEncoder()
        
        # 融合
        self.fusion = nn.Linear(1024 + 128, 512)
        
        # 多标签分类
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, waveform, spectrogram):
        # 编码
        wave_features = self.wave2vec(waveform)
        spec_features = self.spectrogram(spectrogram)
        
        # 融合
        fused = torch.cat([wave_features, spec_features], dim=-1)
        fused = self.fusion(fused)
        
        # 分类
        output = self.classifier(fused)
        
        return torch.sigmoid(output)
```

### 121.2 文本-图像

```python
class CLIPTraining:
    """CLIP训练"""
    
    def __init__(self, vision_model, text_model, temperature=0.07):
        self.vision_model = vision_model
        self.text_model = text_model
        self.temperature = temperature
        
        # 投影层
        self.vision_proj = nn.Linear(vision_model.hidden_size, 512)
        self.text_proj = nn.Linear(text_model.hidden_size, 512)
        
        # 图像/文本dropout
        self.visual_dropout = nn.Dropout(0.0)
        self.text_dropout = nn.Dropout(0.0)
    
    def forward(self, images, texts):
        # 图像特征
        image_features = self.vision_model(images)
        image_features = self.vision_proj(image_features)
        image_features = F.normalize(image_features, dim=-1)
        
        # 文本特征
        text_features = self.text_model(**texts)
        text_features = self.text_proj(text_features)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features
    
    def contrastive_loss(self, image_features, text_features):
        """对比损失"""
        # logits
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 图像到文本
        labels = torch.arange(len(logits)).to(logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        
        # 文本到图像
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2

class BLIP2:
    """BLIP-2"""
    
    def __init__(self, vision_model, language_model, qformer):
        super().__init__()
        
        self.vision_model = vision_model
        self.language_model = language_model
        self.qformer = qformer
        
        # 查询token
        self.query_tokens = nn.Parameter(torch.zeros(1, 32, 768))
        
        # 投影层
        self.vision_proj = nn.Linear(1408, 768)
        self.lm_proj = nn.Linear(768, 4096)
    
    def forward(self, images, texts):
        # 视觉编码
        image_embeds = self.vision_model(images)
        
        # Q-Former处理
        query_output = self.qformer(
            self.query_tokens.expand(image_embeds.size(0), -1, -1),
            image_embeds
        )
        
        # 投影
        query_output = self.vision_proj(query_output)
        
        # 语言模型
        if texts is not None:
            outputs = self.language_model(
                input_ids=texts['input_ids'],
                encoder_hidden_states=query_output
            )
            return outputs
        
        return query_output
```

---

## 122. 强化学习应用

### 122.1 机器人控制

```python
class RobotControlPolicy:
    """机器人控制策略"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        mean = self.actor(state)
        
        if deterministic:
            return mean
        
        # 添加探索噪声
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        
        return action.clamp(-1, 1)
    
    def update(self, replay_buffer, batch_size=256):
        """更新"""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # TD目标
        with torch.no_grad():
            next_actions = self.select_action(next_states)
            q_next = self.critic(next_states)
            target_q = rewards + (1 - dones) * 0.99 * q_next
        
        # Critic损失
        q = self.critic(states)
        critic_loss = F.mse_loss(q, target_q)
        
        # Actor损失
        actor_loss = -self.critic(states).mean()
        
        return critic_loss, actor_loss

class Sim2Real:
    """仿真到真实"""
    
    def __init__(self, sim_env, real_env):
        self.sim_env = sim_env
        self.real_env = real_env
    
    def domain_randomization(self, policy, num_episodes=1000):
        """域随机化"""
        for episode in range(num_episodes):
            # 随机化仿真环境参数
            randomized_env = self._randomize_env()
            
            # 在随机化环境中训练
            self._train_policy(policy, randomized_env)
    
    def adapt_to_real(self, policy, real_data):
        """适应真实环境"""
        # 在线适应
        for state, action in real_data:
            # 计算域差距
            sim_state = self.sim_env.get_equivalent_state(state)
            gap = self._compute_gap(state, sim_state)
            
            # 调整策略
            policy.adjust(gap)
    
    def _compute_gap(self, real_state, sim_state):
        """计算域差距"""
        return torch.abs(real_state - sim_state).mean()
```

### 122.2 推荐系统

```python
class RLRecommendation:
    """强化学习推荐"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.replay_buffer = ReplayBuffer(100000)
    
    def get_recommendation(self, user_state):
        """获取推荐"""
        action_probs = self.actor(user_state)
        action = torch.distributions.Categorical(action_probs).sample()
        return action.item()
    
    def update(self, batch_size=64):
        """更新"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 计算优势
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_v = (next_probs * self.critic(next_states)).sum(dim=-1)
            advantage = rewards + (1 - dones) * 0.99 * next_v - self.critic(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Actor损失
        probs = self.actor(states).gather(1, actions.unsqueeze(1)).squeeze()
        actor_loss = -(torch.log(probs) * advantage).mean()
        
        # Critic损失
        critic_loss = F.mse_loss(
            self.critic(states).gather(1, actions.unsqueeze(1)).squeeze(),
            advantage + self.critic(states).gather(1, actions.unsqueeze(1)).squeeze()
        )
        
        return actor_loss, critic_loss

class BanditRecommendation:
    """多臂老虎机推荐"""
    
    def __init__(self, num_items, context_dim):
        self.num_items = num_items
        self.context_dim = context_dim
        
        # LinUCB
        self.A = [np.eye(context_dim) for _ in range(num_items)]
        self.b = [np.zeros(context_dim) for _ in range(num_items)]
    
    def select_arm(self, context):
        """选择物品"""
        ucb_scores = []
        
        for i in range(self.num_items):
            # 估计
            theta = np.linalg.inv(self.A[i]) @ self.b[i]
            pred = theta @ context
            
            # UCB
            uncertainty = np.sqrt(context @ np.linalg.inv(self.A[i]) @ context)
            ucb_scores.append(pred + 1.5 * uncertainty)
        
        return np.argmax(ucb_scores)
    
    def update(self, context, arm, reward):
        """更新"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

---

## 123. 机器学习系统设计

### 123.1 特征工程

```python
class FeatureEngineeringPipeline:
    """特征工程流水线"""
    
    def __init__(self):
        self.numerical_features = {}
        self.categorical_features = {}
        self.text_features = {}
        self.temporal_features = {}
    
    def add_numerical(self, name, transformation='identity'):
        """添加数值特征"""
        self.numerical_features[name] = transformation
    
    def add_categorical(self, name, encoding='label'):
        """添加类别特征"""
        self.categorical_features[name] = encoding
    
    def add_text(self, name, method='tfidf'):
        """添加文本特征"""
        self.text_features[name] = method
    
    def transform(self, data):
        """转换"""
        features = {}
        
        # 数值特征
        for name, trans in self.numerical_features.items():
            if trans == 'identity':
                features[name] = data[name].values
            elif trans == 'log':
                features[name] = np.log1p(data[name].values)
            elif trans == 'standardize':
                features[name] = (data[name] - data[name].mean()) / data[name].std()
        
        # 类别特征
        for name, enc in self.categorical_features.items():
            if enc == 'onehot':
                features[name] = pd.get_dummies(data[name]).values
            elif enc == 'label':
                encoder = LabelEncoder()
                features[name] = encoder.fit_transform(data[name])
        
        # 文本特征
        for name, method in self.text_features.items():
            if method == 'tfidf':
                vectorizer = TfidfVectorizer()
                features[name] = vectorizer.fit_transform(data[name]).toarray()
        
        return np.hstack([v for v in features.values()])
```

### 123.2 模型服务

```python
class ModelService:
    """模型服务"""
    
    def __init__(self, model_path, batch_size=32):
        self.model = self._load_model(model_path)
        self.batch_size = batch_size
        self.request_queue = queue.Queue()
        self.response_cache = {}
    
    def predict(self, features):
        """预测"""
        # 缓存命中
        cache_key = self._get_cache_key(features)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # 批处理
        self.request_queue.put((features, cache_key))
        
        if self.request_queue.qsize() >= self.batch_size:
            self._process_batch()
        
        return self._wait_for_result(cache_key)
    
    def _process_batch(self):
        """处理批次"""
        batch = []
        keys = []
        
        while len(batch) < self.batch_size:
            try:
                features, key = self.request_queue.get(timeout=0.1)
                batch.append(features)
                keys.append(key)
            except queue.Empty:
                break
        
        if batch:
            predictions = self.model.predict(np.array(batch))
            
            for key, pred in zip(keys, predictions):
                self.response_cache[key] = pred
    
    def _load_model(self, path):
        """加载模型"""
        import joblib
        return joblib.load(path)
    
    def _get_cache_key(self, features):
        """缓存键"""
        return hash(features.tobytes())

class AOTServing:
    """AOT提前编译服务"""
    
    def __init__(self, model, input_spec):
        self.compiled_model = self._aot_compile(model, input_spec)
    
    def _aot_compile(self, model, input_spec):
        """AOT编译"""
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch._inductor import compile_fx
        
        # 追踪模型
        traced = torch.fx.symbolic_trace(model)
        
        # 编译
        compiled = compile_fx(traced, input_spec)
        
        return compiled
```

### 123.3 监控与告警

```python
class MLMonitoring:
    """ML监控"""
    
    def __init__(self):
        self.metrics = {
            'prediction_latency': [],
            'prediction_distribution': [],
            'feature_drift': [],
            'prediction_drift': []
        }
    
    def log_prediction(self, features, prediction, latency):
        """记录预测"""
        self.metrics['prediction_latency'].append(latency)
        self.metrics['prediction_distribution'].append(prediction)
    
    def detect_drift(self, reference_data, current_data):
        """检测漂移"""
        drift_scores = {}
        
        for feature in range(reference_data.shape[1]):
            ks_stat, p_value = self._ks_test(
                reference_data[:, feature],
                current_data[:, feature]
            )
            drift_scores[f'feature_{feature}'] = {
                'statistic': ks_stat,
                'p_value': p_value,
                'drifted': p_value < 0.05
            }
        
        return drift_scores
    
    def _ks_test(self, ref, curr):
        """KS检验"""
        from scipy.stats import ks_2samp
        return ks_2samp(ref, curr)
    
    def trigger_alert(self, metric_name, threshold):
        """触发告警"""
        if metric_name == 'drift_ratio':
            drift_count = sum(1 for v in self.metrics['feature_drift'].values() 
                            if v.get('drifted'))
            if drift_count / len(self.metrics['feature_drift']) > threshold:
                return self._send_alert(f"Drift detected in {drift_count} features")
        
        return None
    
    def _send_alert(self, message):
        """发送告警"""
        # 集成到告警系统
        print(f"ALERT: {message}")
```

---

## 124. 行业应用案例

### 124.1 医疗AI

```python
class MedicalImageDiagnosis:
    """医学影像诊断"""
    
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.thresholds = {
            'sensitivity': 0.95,  # 召回率优先
            'specificity': 0.80
        }
    
    def diagnose(self, image):
        """诊断"""
        # 前向传播
        with torch.no_grad():
            probabilities = self.model(image)
        
        # 应用阈值
        predictions = (probabilities > self.thresholds['sensitivity']).float()
        
        # 解释
        explanation = self._explain_prediction(image, probabilities)
        
        return {
            'diagnosis': predictions,
            'probability': probabilities,
            'explanation': explanation,
            'confidence': self._calculate_confidence(probabilities)
        }
    
    def _explain_prediction(self, image, probabilities):
        """解释预测"""
        # 使用GradCAM
        return GradCAM(self.model).generate(image)
    
    def _calculate_confidence(self, probabilities):
        """计算置信度"""
        return probabilities.max().item()

class DrugDiscoveryPipeline:
    """药物发现流水线"""
    
    def __init__(self):
        self.molecule_encoder = MoleculeEncoder()
        self.property_predictor = PropertyPredictor()
        self.synthesis_planner = SynthesisPlanner()
        self.trial_designer = TrialDesigner()
    
    def discover_drug(self, target_protein):
        """发现药物"""
        # 1. 生成候选分子
        candidates = self._generate_candidates(target_protein)
        
        # 2. 预测性质
        for candidate in candidates:
            properties = self.property_predictor.predict(candidate)
            candidate.properties = properties
        
        # 3. 筛选
        filtered = self._filter_by_properties(candidates)
        
        # 4. 合成规划
        for drug in filtered:
            synthesis = self.synthesis_planner.plan(drug.molecule)
            drug.synthesis = synthesis
        
        # 5. 试验设计
        trials = self.trial_designer.design(filtered[:3])
        
        return trials
```

### 124.2 金融AI

```python
class AlgorithmicTrading:
    """算法交易"""
    
    def __init__(self, model, risk_manager):
        self.model = model
        self.risk_manager = risk_manager
    
    def predict_market(self, market_data):
        """预测市场"""
        features = self._extract_features(market_data)
        return self.model.predict(features)
    
    def execute_trade(self, prediction, portfolio, market_data):
        """执行交易"""
        # 风险检查
        if not self.risk_manager.check_risk(prediction, portfolio):
            return None
        
        # 仓位确定
        position_size = self._calculate_position(prediction, portfolio)
        
        # 执行
        order = self._place_order(prediction, position_size, market_data)
        
        return order
    
    def _extract_features(self, data):
        """提取特征"""
        return np.hstack([
            self._price_features(data),
            self._volume_features(data),
            self._technical_indicators(data)
        ])

class FraudDetection:
    """欺诈检测"""
    
    def __init__(self, model, threshold=0.9):
        self.model = model
        self.threshold = threshold
    
    def detect(self, transaction):
        """检测"""
        features = self._extract_features(transaction)
        probability = self.model.predict_proba(features)
        
        is_fraud = probability > self.threshold
        
        return {
            'is_fraud': is_fraud,
            'probability': probability,
            'risk_score': probability * 100,
            'reasons': self._explain_decision(transaction, probability)
        }
    
    def _explain_decision(self, transaction, probability):
        """解释决策"""
        return SHAPExplainer(self.model).explain(transaction)
```

### 124.3 自动驾驶

```python
class AutonomousDrivingStack:
    """自动驾驶堆栈"""
    
    def __init__(self):
        self.perception = PerceptionModule()
        self.prediction = TrajectoryPredictor()
        self.planning = BehaviorPlanner()
        self.control = VehicleController()
    
    def drive(self, sensor_data):
        """驾驶"""
        # 感知
        scene = self.perception.process(sensor_data)
        
        # 预测
        predictions = self.prediction.predict(scene)
        
        # 规划
        trajectory = self.planning.plan(scene, predictions)
        
        # 控制
        control_signals = self.control.execute(trajectory)
        
        return control_signals

class V2XCommunication:
    """车联网通信"""
    
    def __init__(self):
        self.vehicle_network = VehicleNetwork()
        self.infrastructure_network = InfrastructureNetwork()
    
    def share_information(self, vehicle_data):
        """共享信息"""
        # V2V
        self.vehicle_network.broadcast(vehicle_data)
        
        # V2I
        self.infrastructure_network.send(vehicle_data)
    
    def receive_information(self):
        """接收信息"""
        v2v_data = self.vehicle_network.receive()
        v2i_data = self.infrastructure_network.receive()
        
        return self._fuse_information(v2v_data, v2i_data)
```

---

## 125. 持续学习与适应

### 125.1 增量学习

```python
class IncrementalLearning:
    """增量学习"""
    
    def __init__(self, model, memory_size=1000):
        self.model = model
        self.memory = ExemplarMemory(memory_size)
        self.current_task = 0
    
    def adapt_to_new_task(self, new_data, task_id):
        """适应新任务"""
        self.current_task = task_id
        
        # 存储样本
        self.memory.add(new_data)
        
