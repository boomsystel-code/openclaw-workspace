# 🚀 深度学习高级技术 Part 11

*扩展知识与实践*

---

## 131. 高级NLP技术

### 131.1 大模型训练技巧

```python
class LLM TrainingPipeline:
    """大语言模型训练流水线"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        self.scheduler = self._create_scheduler()
        
        # 梯度缩放
        self.scaler = torch.cuda.amp.GradScaler()
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            progress = float(step - self.config.warmup_steps) / float(
                max(1, self.config.train_steps - self.config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            # 混合精度前向
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # 反向
            self.scaler.scale(loss).backward()
            
            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # 优化器步骤
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # 学习率调度
                self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

class FlashAttention:
    """Flash Attention实现"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.shape
        
        # 投影并分头
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention
        attn_output = self._flash_attention(Q, K, V, attn_mask)
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        return self.out_proj(attn_output)
    
    def _flash_attention(self, Q, K, V, mask):
        """Flash Attention核心实现"""
        # 计算softmax
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 与V相乘
        output = torch.matmul(attn, V)
        
        return output
```

### 131.2 分词与词表

```python
class BytePairEncoding:
    """字节对编码"""
    
    def __init__(self, vocab_size=10000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.word_freq = {}
        self.merges = {}
        self.vocab = {}
    
    def train(self, text):
        """训练BPE"""
        # 统计词频
        for word in text.split():
            self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # 初始化词表
        self.vocab = {chr(i + 256): i for i in range(self.vocab_size)}
        
        # 迭代合并
        for i in range(self.vocab_size - 256):
            # 找到最频繁的bigram
            best_pair = self._find_most_frequent_bigram()
            
            if not best_pair or self.word_freq[best_pair] < self.min_frequency:
                break
            
            # 合并
            self._merge_pair(best_pair)
            self.merges[best_pair] = len(self.vocab)
    
    def _find_most_frequent_bigram(self):
        """找到最频繁的bigram"""
        bigram_freq = {}
        
        for word, freq in self.word_freq.items():
            for i in range(len(word) - 1):
                bigram = (word[i], word[i + 1])
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + freq
        
        if not bigram_freq:
            return None
        
        return max(bigram_freq, key=bigram_freq.get)
    
    def _merge_pair(self, pair):
        """合并pair"""
        new_symbol = ''.join(pair)
        
        # 更新词频
        new_word_freq = {}
        for word, freq in self.word_freq.items():
            new_word = word.replace(''.join(pair), new_symbol)
            new_word_freq[new_word] = new_word_freq.get(new_word, 0) + freq
        
        self.word_freq = new_word_freq
    
    def encode(self, text):
        """编码"""
        tokens = list(text)
        
        while len(tokens) > 1:
            # 找到可合并的pair
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            best_idx = None
            best_pair = None
            for i, pair in enumerate(pairs):
                if pair in self.merges:
                    if best_idx is None or self.merges[pair] < self.merges[best_pair]:
                        best_idx = i
                        best_pair = pair
            
            if best_idx is None:
                break
            
            # 合并
            new_symbol = chr(len(self.vocab) + 256)
            self.vocab[new_symbol] = len(self.vocab)
            
            tokens[best_idx] = new_symbol
            del tokens[best_idx + 1]
        
        return tokens
```

### 131.3 长上下文处理

```python
class LongContextTransformer:
    """长上下文Transformer"""
    
    def __init__(self, base_model, max_context=32768):
        self.base_model = base_model
        self.max_context = max_context
        
        # 稀疏注意力模式
        self.local_window = 1024
        self.global_attention = [0]  # CLS token总是全局
    
    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        if seq_len <= self.local_window:
            # 短序列使用完整注意力
            return self.base_model(input_ids, attention_mask)
        
        # 构建稀疏注意力掩码
        sparse_mask = self._build_sparse_mask(seq_len)
        
        # 分块处理
        chunks = self._split_into_chunks(input_ids, self.local_window)
        outputs = []
        
        for i, chunk in enumerate(chunks):
            # 全局token
            global_token = input_ids[:, i * self.local_window:i * self.local_window + 1]
            
            # 局部注意力
            local_output = self.base_model(chunk, None)
            
            # 与全局交互
            for j, token_idx in enumerate(self.global_attention):
                if token_idx < len(chunk):
                    global_repr = local_output[:, token_idx:token_idx + 1]
            
            outputs.append(local_output)
        
        # 拼接
        return torch.cat(outputs, dim=1)
    
    def _build_sparse_mask(self, seq_len):
        """构建稀疏注意力掩码"""
        mask = torch.zeros(seq_len, seq_len)
        
        # 局部窗口
        for i in range(seq_len):
            start = max(0, i - self.local_window)
            end = min(seq_len, i + self.local_window + 1)
            mask[i, start:end] = 1
        
        # 全局token
        for global_idx in self.global_attention:
            mask[:, global_idx] = 1
            mask[global_idx, :] = 1
        
        return mask
    
    def _split_into_chunks(self, tensor, chunk_size):
        """分块"""
        return tensor.split(chunk_size, dim=1)
```

---

## 132. 视觉大模型

### 132.1 SAM分割

```python
class SegmentAnythingModel:
    """Segment Anything Model"""
    
    def __init__(self, encoder_dim=1280, num_masks=4):
        # 图像编码器
        self.image_encoder = ResNet50Backbone() if num_masks < 10 else ViT_B()
        
        # 提示编码器
        self.prompt_encoder = PromptEncoder()
        
        # 掩码解码器
        self.mask_decoder = MaskDecoder(
            transformer_dim=encoder_dim,
            num_masks=num_masks
        )
    
    def forward(self, image, prompts):
        """前向传播"""
        # 图像特征
        image_embeddings = self.image_encoder(image)
        
        # 提示编码
        sparse_embeddings, dense_embeddings = self.prompt_encoder(prompts)
        
        # 掩码预测
        masks, iou_pred = self.mask_decoder(
            image_embeddings,
            sparse_embeddings,
            dense_embeddings
        )
        
        return masks, iou_pred

class MaskDecoder:
    """掩码解码器"""
    
    def __init__(self, transformer_dim=256, num_masks=4):
        self.transformer_dim = transformer_dim
        self.num_masks = num_masks
        
        # Transformer
        self.transformer = MaskTransformer(transformer_dim)
        
        # 输出层
        self.output_tokens = nn.Embedding(4, transformer_dim)
        self.output_hypernet = nn.Linear(transformer_dim, 256)
        self.mask_features = nn.Conv2d(transformer_dim, 256, kernel_size=1)
    
    def forward(self, image_embeddings, sparse_embeddings, dense_embeddings):
        """前向传播"""
        # Transformer处理
        tokens = self.output_tokens.weight.unsqueeze(0).expand(
            image_embeddings.size(0), -1, -1
        )
        
        # 融合
        sos_tokens = torch.cat([tokens, sparse_embeddings], dim=1)
        
        # Transformer
        hs = self.transformer(
            sos_tokens,
            image_embeddings,
            dense_embeddings
        )
        
        # 输出掩码token
        mask_tokens = hs[:, :4]
        
        # 掩码特征
        mask_features = self.mask_features(image_embeddings)
        
        # 每个token预测一个掩码
        masks = []
        for i in range(self.num_masks):
            mask_features_i = mask_features * self.output_hypernet(mask_tokens[:, i]).unsqueeze(-1).unsqueeze(-1)
            mask = F.conv2d(mask_features_i, self.mask_features.weight, bias=None)
            masks.append(mask)
        
        masks = torch.stack(masks, dim=1)
        
        return masks
```

### 132.2 视觉生成

```python
class ImageGenerationModel:
    """图像生成模型"""
    
    def __init__(self, config):
        self.config = config
        
        # U-Net去噪网络
        self.unet = UNetModel(
            in_channels=4,
            model_channels=128,
            out_channels=4,
            num_res_blocks=2,
            attention_resolutions=[8, 16],
            channel_mult=[1, 2, 4, 8],
            num_heads=8
        )
        
        # 文本编码器
        self.text_encoder = CLIPTextEncoder()
        
        # VAE
        self.vae = AutoencoderKL(
            in_channels=3,
            latent_channels=4,
            out_channels=3
        )
    
    def train_step(self, images, prompts):
        """训练步骤"""
        # 编码图像
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        
        # 采样噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.size(0),))
        
        # 添加噪声
        noisy_latents = self._add_noise(latents, noise, timesteps)
        
        # 编码文本
        text_embeddings = self.text_encoder(prompts)
        
        # 预测噪声
        noise_pred = self.unet(
            noisy_latents, timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    def generate(self, prompts, num_images=1, guidance_scale=7.5):
        """生成图像"""
        # 编码文本
        text_embeddings = self.text_encoder(prompts)
        
        # 无分类器引导
        if guidance_scale > 1.0:
            uncond_embeddings = self.text_encoder([''] * len(prompts))
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 生成
        latents = self._generate_latents(text_embeddings)
        
        # 解码
        images = self.vae.decode(latents / 0.18215)
        
        return images
    
    def _generate_latents(self, text_embeddings):
        """生成潜在变量"""
        latents = torch.randn(
            text_embeddings.size(0) // 2,
            4, 64, 64
        ).cuda()
        
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(50)
        
        for t in scheduler.timesteps:
            # 预测噪声
            noise_pred = self.unet(
                latents, t,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # 引导
            if text_embeddings.size(0) > latents.size(0):
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 采样
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
```

### 132.3 视频理解

```python
class VideoUnderstandingModel:
    """视频理解模型"""
    
    def __init__(self, config):
        # 空间编码器
        self.spatial_encoder = VideoViT(
            num_frames=config.num_frames,
            patch_size=2,
            embed_dim=768
        )
        
        # 时间编码器
        self.temporal_encoder = TimeTransformer(
            embed_dim=768,
            num_layers=6
        )
        
        # 分类头
        self.classifier = nn.Linear(768, config.num_classes)
    
    def forward(self, video):
        """视频前向"""
        batch_size, num_frames, channels, height, width = video.shape
        
        # 空间特征提取
        spatial_features = []
        for t in range(num_frames):
            frame = video[:, t]
            features = self.spatial_encoder(frame)
            spatial_features.append(features)
        
        # 堆叠时间特征
        spatial_features = torch.stack(spatial_features, dim=1)  # [B, T, C]
        
        # 时间建模
        temporal_features = self.temporal_encoder(spatial_features)
        
        # 分类
        output = self.classifier(temporal_features[:, -1])  # 最后一帧
        
        return output

class TimeTransformer:
    """时间Transformer"""
    
    def __init__(self, embed_dim, num_layers=6, num_heads=8):
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x):
        # 添加时间位置编码
        x = x + self.temporal_embedding
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        return x
```

---

## 133. 语音AI

### 133.1 Whisper

```python
class WhisperModel:
    """Whisper语音识别模型"""
    
    def __init__(self, config):
        # 编码器
        self.encoder = AudioEncoder(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer
        )
        
        # 解码器
        self.decoder = WhisperDecoder(
            n_vocab=config.n_vocab,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer
        )
    
    def transcribe(self, audio):
        """转录"""
        # 提取梅尔频谱
        mel = self._extract_mel(audio)
        
        # 编码
        audio_features = self.encoder(mel)
        
        # 解码
        tokens = self.decoder.generate(audio_features)
        
        return self._decode_tokens(tokens)
    
    def _extract_mel(self, audio, n_mels=80, n_fft=400, hop_length=160):
        """提取梅尔频谱"""
        # 短时傅里叶变换
        stft = torch.stft(audio, n_fft, hop_length, window=torch.hann_window(n_fft), return_complex=True)
        magnitude = stft.abs()
        
        # 梅尔滤波器
        mel_filter = self._create_mel_filter(n_mels, n_fft)
        mel = torch.matmul(magnitude, mel_filter)
        
        # 对数压缩
        mel = torch.log(mel + 1e-9)
        
        return mel
```

### 133.2 语音合成

```python
class TextToSpeech:
    """文本转语音"""
    
    def __init__(self, config):
        # 文本编码器
        self.text_encoder = TextEncoder(
            n_vocab=config.n_vocab,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer
        )
        
        # 音频解码器
        self.audio_decoder = AudioDecoder(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer
        )
    
    def synthesize(self, text):
        """合成语音"""
        # 编码文本
        text_tokens = self._tokenize(text)
        text_features = self.text_encoder(text_tokens)
        
        # 生成音频
        mel = self.audio_decoder.generate(text_features)
        
        # 转换为波形
        waveform = self._mel_to_waveform(mel)
        
        return waveform
    
    def _mel_to_waveform(self, mel, n_fft=1024, hop_length=256):
        """梅尔频谱转波形"""
        # Griffin-Lim
        waveform = self._griffin_lim(mel, n_fft, hop_length)
        
        return waveform
    
    def _griffin_lim(self, mel_spectrogram, n_fft, hop_length):
        """Griffin-Lim算法"""
        # 初始化相位
        signal = torch.randn(mel_spectrogram.size(0), n_fft // 2 + 1, mel_spectrogram.size(1))
        
        for _ in range(50):
            # STFT
            stft = torch.stft(signal, n_fft, hop_length, window=torch.hann_window(n_fft), return_complex=True)
            
            # 更新相位
            phase = stft.angle()
            reconstruction = mel_spectrogram * torch.exp(1j * phase)
            
            # ISTFT
            signal = torch.istft(reconstruction, n_fft, hop_length, window=torch.hann_window(n_fft))
        
        return signal
```

### 133.3 语音转换

```python
class VoiceConversion:
    """语音转换"""
    
    def __init__(self, config):
        # 内容编码器
        self.content_encoder = ContentEncoder()
        
        # 说话者编码器
        self.speaker_encoder = SpeakerEncoder()
        
        # 解码器
        self.decoder = Decoder()
    
    def convert(self, source_audio, target_speaker):
        """转换语音"""
        # 提取内容
        content = self.content_encoder(source_audio)
        
        # 提取说话者特征
        speaker = self.speaker_encoder(target_speaker)
        
        # 融合并解码
        converted = self.decoder(content, speaker)
        
        return converted
```

---

## 134. 推荐系统深度

### 134.1 深度协同过滤

```python
class DeepCollaborativeFiltering:
    """深度协同过滤"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64, 32]):
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP层
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, user_ids, item_ids):
        # 嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # MLP
        x = self.mlp(x)
        
        # 输出
        output = self.output_layer(x)
        
        return output.squeeze()
```

### 134.2 图推荐

```python
class GraphRecommender:
    """图推荐系统"""
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        # 用户嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # 物品嵌入
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 图神经网络
        self.gnn = LightGCN(
            embedding_dim=embedding_dim,
            num_layers=3
        )
    
    def forward(self, user_item_graph, user_ids, item_ids):
        """前向传播"""
        # 图嵌入
        user_emb, item_emb = self.gnn(user_item_graph)
        
        # 采样正负样本
        pos_scores = (user_emb[user_ids] * item_emb[item_ids]).sum(dim=-1)
        neg_scores = (user_emb[user_ids] * item_emb[self.neg_items]).sum(dim=-1)
        
        return pos_scores, neg_scores

class LightGCN:
    """LightGCN"""
    
    def __init__(self, embedding_dim, num_layers=3):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.alpha = 1.0 / (num_layers + 1)
    
    def forward(self, graph, user_emb, item_emb):
        """前向传播"""
        # 初始嵌入
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        emb_list = [all_emb]
        
        # 多层图卷积
        for _ in range(self.num_layers):
            all_emb = self._propagate(graph, all_emb)
            emb_list.append(all_emb)
        
        # 加权求和
        final_emb = torch.zeros_like(all_emb)
        for emb in emb_list:
            final_emb += emb * self.alpha
        
        # 分离
        user_final = final_emb[:len(user_emb)]
        item_final = final_emb[len(user_emb):]
        
        return user_final, item_final
    
    def _propagate(self, graph, embeddings):
        """图传播"""
        # 归一化邻接矩阵
        norm_adj = self._normalize_adjacency(graph)
        
        # 传播
        return torch.sparse.mm(norm_adj, embeddings)
```

### 134.3 多任务推荐

```python
class MultiTaskRecommender:
    """多任务推荐"""
    
    def __init__(self, shared_bottom, task_towers):
        self.shared_bottom = shared_bottom
        self.task_towers = nn.ModuleDict(task_towers)
    
    def forward(self, user_features, item_features):
        """前向传播"""
        # 共享底层
        shared_repr = self.shared_bottom(user_features, item_features)
        
        # 多任务输出
        outputs = {}
        for task_name, tower in self.task_towers.items():
            outputs[task_name] = tower(shared_repr)
        
        return outputs
    
    def loss(self, predictions, labels):
        """多任务损失"""
        total_loss = 0
        
        for task_name, pred in predictions.items():
            label = labels[task_name]
            
            if task_name == 'click':
                loss = F.binary_cross_entropy_with_logits(pred, label)
            elif task_name == 'conversion':
                loss = F.binary_cross_entropy_with_logits(pred, label)
            elif task_name == 'dwell_time':
                loss = F.mse_loss(pred, label)
            
            total_loss += loss
        
        return total_loss
```

---

## 135. 异常检测

### 135.1 单类分类

```python
class OneClassSVM:
    """单类SVM"""
    
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = None
    
    def fit(self, X):
        """训练"""
        from sklearn.svm import OneClassSVM
        
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        ).fit(X)
        
        return self
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def score_samples(self, X):
        """异常分数"""
        return self.model.decision_function(X)

class DeepSVDD:
    """深度SVDD"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], radius=0.0):
        self.encoder = self._build_encoder(input_dim, hidden_dims)
        self.radius = radius
        self.center = None
    
    def _build_encoder(self, input_dim, hidden_dims):
        """构建编码器"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def fit(self, X, epochs=100):
        """训练"""
        self.encoder.train()
        
        optimizer = torch.optim.Adam(self.encoder.parameters())
        
        for epoch in range(epochs):
            for x in X:
                # 编码
                z = self.encoder(x)
                
                # 计算到中心的距离
                if self.center is None:
                    self.center = z.mean(dim=0)
                
                loss = ((z - self.center) ** 2).sum(dim=1).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        """预测"""
        self.encoder.eval()
        
        with torch.no_grad():
            z = self.encoder(X)
            distances = ((z - self.center) ** 2).sum(dim=1)
        
        # 返回异常分数
        return -distances  # 距离越大，异常分数越高
```

### 135.2 自编码器异常检测

```python
class AutoEncoderAnomalyDetection:
    """自编码器异常检测"""
    
    def __init__(self, input_dim, latent_dim=32):
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def fit(self, X, epochs=100):
        """训练"""
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(epochs):
            # 编码-解码
            z = self.encoder(X)
            reconstructed = self.decoder(z)
            
            # 重建损失
            loss = F.mse_loss(reconstructed, X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        """预测异常"""
        with torch.no_grad():
            z = self.encoder(X)
            reconstructed = self.decoder(z)
            
            # 重建误差
            reconstruction_error = F.mse_loss(reconstructed, X, reduction='none').sum(dim=1)
        
        return reconstruction_error
    
    def detect(self, X, threshold=None):
        """检测异常"""
        errors = self.predict(X)
        
        if threshold is None:
            threshold = errors.mean() + 3 * errors.std()
        
        return errors > threshold, errors
```

### 135.3 时序异常检测

```python
class TimeSeriesAnomalyDetection:
    """时序异常检测"""
    
    def __init__(self, input_dim, hidden_dim=64):
        # 编码器
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.2
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim, input_dim, num_layers=2,
            batch_first=True, dropout=0.2
        )
    
    def fit(self, X, epochs=100):
        """训练"""
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(epochs):
            # 编码
            _, (hidden, _) = self.encoder(X)
            
            # 解码
            reconstructed, _ = self.decoder(hidden.repeat(1, X.size(1), 1))
            
            # 损失
            loss = F.mse_loss(reconstructed, X)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return self
    
    def detect(self, X, threshold=None):
        """检测"""
        with torch.no_grad():
            _, (hidden, _) = self.encoder(X)
            reconstructed, _ = self.decoder(hidden.repeat(1, X.size(1), 1))
            
            # 重建误差
            errors = F.mse_loss(reconstructed, X, reduction='none').mean(dim=2)
        
        if threshold is None:
            threshold = errors.mean() + 3 * errors.std()
        
        return errors > threshold, errors
```

---

## 136. AutoML高级

### 136.1 神经架构搜索

```python
class NASController:
    """NAS控制器"""
    
    def __init__(self, search_space, hidden_size=100):
        self.controller = nn.LSTM(
            hidden_size, hidden_size, num_layers=2
        )
        
        self.embeddings = nn.Embedding(len(search_space), hidden_size)
        
        # 每个决策的解码器
        self.decoders = nn.ModuleDict()
        for key, options in search_space.items():
            self.decoders[key] = nn.Linear(hidden_size, len(options))
    
    def sample(self):
        """采样架构"""
        architecture = {}
        
        for key in search_space.keys():
            # 嵌入
            embed = self.embeddings.weight.mean(dim=0, keepdim=True)
            
            # 解码
            logits = self.decoders[key](embed)
            probs = F.softmax(logits, dim=-1)
            
            # 采样
            choice = torch.multinomial(probs, 1).item()
            architecture[key] = search_space[key][choice]
        
        return architecture

class DARTSOptimizer:
    """DARTS优化器"""
    
    def __init__(self, model, unrolled=False):
        self.model = model
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0003
        )
        self.unrolled = unrolled
    
    def step(self, X_valid, y_valid, X_train, y_train):
        """一步优化"""
        if self.unrolled:
            # 展开优化
            self._unrolled_step(X_valid, y_valid, X_train, y_train)
        else:
            # 近似优化
            self._approx_step(X_valid, y_valid)
    
    def _approx_step(self, X_valid, y_valid):
        """近似优化"""
        # 计算验证集上的梯度
        self.optimizer.zero_grad()
        loss = self.model(X_valid, y_valid)
        loss.backward()
        
        # 使用验证梯度更新架构参数
        self.model.update_alphas()
        
        # 使用训练集更新权重
        self.optimizer.step()
```

### 136.2 超参数优化

```python
class BayesianOptimization:
    """贝叶斯优化"""
    
    def __init__(self, objective, search_space, acquisition='EI'):
        self.objective = objective
        self.search_space = search_space
        self.acquisition = acquisition
        
        # 高斯过程
        self.gp = GaussianProcessRegressor()
        
        # 采集函数
        if acquisition == 'EI':
            self.acquisition_func = expected_improvement
        elif acquisition == 'UCB':
            self.acquisition_func = upper_confidence_bound
    
    def optimize(self, n_iterations=100):
        """优化"""
        # 初始样本
        X_samples = self._initial_samples(10)
        y_samples = [self.objective(x) for x in X_samples]
        
        for _ in range(n_iterations):
            # 拟合GP
            self.gp.fit(X_samples, y_samples)
            
            # 获取下一个采样点
            next_x = self._optimize_acquisition()
            
            # 评估
            next_y = self.objective(next_x)
            
            # 更新
            X_samples.append(next_x)
            y_samples.append(next_y)
        
        return min(y_samples), X_samples[np.argmin(y_samples)]
    
    def _initial_samples(self, n):
        """初始样本"""
        samples = []
        for _ in range(n):
            sample = {}
            for key, space in self.search_space.items():
                if isinstance(space, tuple):  # 连续
                    sample[key] = np.random.uniform(space[0], space[1])
                elif isinstance(space, list):  # 离散
                    sample[key] = np.random.choice(space)
            samples.append(sample)
        return samples
    
    def _optimize_acquisition(self):
        """优化采集函数"""
        # 在采集函数上优化
        return np.random.choice(self._initial_samples(1))[0]
```

### 136.3 元学习AutoML

```python
class MetaLearningAutoML:
    """元学习AutoML"""
    
    def __init__(self, meta_features_dim=100, task_embedding_dim=32):
        # 元特征编码器
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, task_embedding_dim)
        )
        
        # 预测器
        self.predictor = nn.Linear(task_embedding_dim,