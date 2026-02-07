# 深度学习高级技术 Part 13

## 131. 高级注意力机制

### 131.1 稀疏注意力

```python
class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_window=512):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, q, k, v):
        scale = 1.0 / (q.size(-1) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 稀疏性：只计算局部窗口
        L = q.size(2)
        for i in range(L):
            start = max(0, i - self.attention_window)
            end = min(L, i + self.attention_window + 1)
            scores[:, :, i, :start] = -1e9
            scores[:, :, i, end:] = -1e9
        
        return torch.matmul(F.softmax(scores, dim=-1), v)
```

### 131.2 RoPE旋转位置编码

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        return torch.polar(torch.ones_like(angles), angles)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

## 132. 混合专家模型

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, top_k, hidden_size):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
    
    def forward(self, x):
        router_probs = F.softmax(self.router(x), dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, 2, dim=-1)
        
        output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            mask = (top_k_indices == expert_idx).any(dim=-1)
            if mask.any():
                weight = top_k_weights[mask].sum(dim=-1, keepdim=True)
                output[mask] += expert(x[mask]) * weight
        
        return output
```

## 133. RAG检索增强

```python
class VectorDatabase:
    def __init__(self, embedding_dim=768):
        self.embeddings = []
        self.documents = []
    
    def search(self, query, top_k=10):
        scores = [cosine_similarity(query, e) for e in self.embeddings]
        return sorted(zip(scores, self.documents), reverse=True)[:top_k]

class HybridRetrieval:
    def __init__(self, dense, sparse):
        self.dense = dense
        self.sparse = sparse
    
    def search(self, query, top_k=10, alpha=0.5):
        dense_results = self.dense.search(query, top_k * 2)
        sparse_results = self.sparse.search(query, top_k * 2)
        return self._fusion(dense_results, sparse_results, top_k, alpha)
```

## 134. 离线强化学习

```python
class ConservativeQLearning:
    def __init__(self, state_dim, action_dim):
        self.Q = nn.Sequential(nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.Q_target = copy.deepcopy(self.Q)
    
    def update(self, batch):
        q_values = self.Q(torch.cat([batch.states, batch.actions], dim=1))
        with torch.no_grad():
            target = batch.rewards + 0.99 * self.Q_target(batch.next_states).max(dim=1)[0]
        loss = F.mse_loss(q_values.squeeze(), target)
        loss.backward()
        self.optimizer.step()
        self._soft_update()
        return loss.item()
```

## 135. CLIP多模态

```python
class CLIPModel(nn.Module):
    def __init__(self, vision, text, projection_dim=512):
        super().__init__()
        self.vision = vision
        self.text = text
        self.visual_proj = nn.Linear(768, projection_dim)
        self.text_proj = nn.Linear(768, projection_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, input_ids):
        img_feat = F.normalize(self.visual_proj(self.vision(images)), dim=-1)
        txt_feat = F.normalize(self.text_proj(self.text(input_ids=input_ids)), dim=-1)
        return torch.matmul(img_feat, txt_feat.T) * self.temperature.exp()
```

## 136. 知识蒸馏

```python
class KnowledgeDistillation:
    def __init__(self, teacher, student, temperature=2.0):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
    
    def train_step(self, x, target):
        with torch.no_grad():
            teacher_out = self.teacher(x)
        student_out = self.student(x)
        
        distill_loss = F.kl_div(F.log_softmax(student_out / self.temperature, dim=1),
                               F.softmax(teacher_out / self.temperature, dim=1),
                               reduction='batchmean') * self.temperature ** 2
        ce_loss = F.cross_entropy(student_out, target)
        return 0.5 * distill_loss + 0.5 * ce_loss
```

## 137. 分布式训练

```python
class FSDPTrainer:
    def __init__(self, model, lr=0.01):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        self.model = FSDP(model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    
    def train_step(self, batch):
        loss = self.model(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
```

## 138. Swin Transformer

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = ShiftedWindowAttention(dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    
    def forward(self, x):
        x = x + self.attn(self._window_partition(x))
        x = x + self.ffn(x)
        return x
```

## 139. 扩散模型

```python
class DiffusionModel(nn.Module):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
    
    def train_step(self, images):
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.size(0),))
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.unet(noisy, timesteps).sample
        return F.mse_loss(noise_pred, noise)
    
    def sample(self, prompt):
        text_emb = self.text_encoder(prompt)
        latents = torch.randn(1, 4, 64, 64)
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_emb).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return self.vae.decode(latents / 0.18215)
```

## 140. 自动驾驶BEV

```python
class BEVFormer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.bev_embed = nn.Embedding(200 * 200, 256)
        self.decoder = TransformerDecoder(256)
    
    def forward(self, images):
        img_features = self.encoder(images)
        bev_queries = self.bev_embed.weight.unsqueeze(0)
        output = self.decoder(bev_queries, img_features)
        return self.detection_head(output)
```

## 141. AlphaFold2

```python
class AlphaFold2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.msa_stack = MSAStack(config)
        self.pair_stack = PairRepresentationStack(config)
        self.structure_module = StructureModule(config)
    
    def forward(self, msa, pair, single):
        msa = self.msa_stack(msa, pair)
        pair = self.pair_stack(pair)
        return self.structure_module(single, pair)
```

## 142. 量子机器学习

```python
import pennylane as qml

class QuantumNN(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(10))
    
    def forward(self, x):
        @qml.qnode(self.dev)
        def circuit(x, w):
            qml.templates.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.templates.StronglyEntanglingLayers(w, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return torch.tensor(circuit(x.detach().numpy(), self.weights.detach().numpy()))
```

## 143. NAS搜索

```python
class NASController(nn.Module):
    def __init__(self, search_space):
        super().__init__()
        self.embeddings = nn.Embedding(len(search_space), 100)
        self.decoder = nn.Linear(100, len(search_space))
    
    def sample(self):
        return {k: self._sample_action(k) for k in search_space.keys()}
    
    def _sample_action(self, key):
        logits = self.decoder(self.embeddings.weight.mean(dim=0))
        return search_space[key][torch.multinomial(F.softmax(logits, dim=0), 1).item()]
```

## 144. 可解释AI

```python
class SHAPExplainer:
    def __init__(self, model, background):
        self.model = model
        self.background = background
    
    def explain(self, instance):
        return np.array([self._compute_shap(instance, i) for i in range(len(instance))])
    
    def _compute_shap(self, instance, feature_idx):
        with_feature = instance.clone()
        without_feature = instance.clone()
        with_feature[feature_idx] = 0
        return self.model(with_feature) - self.model(without_feature)
```

## 145. AutoML

```python
class AutoMLPipeline:
    def __init__(self):
        self.study = None
    
    def optimize(self, objective, n_trials=100):
        import optuna
        self.study = optuna.create_study()
        self.study.optimize(objective, n_trials=n_trials)
        return self.study.best_params
```

---

## 146. AI产品工程

```python
class MLOpsPipeline:
    def __init__(self):
        self.data = DataPipeline()
        self.train = TrainingPipeline()
        self.serve = ServingPipeline()
        self.monitor = MonitoringPipeline()
    
    def run(self, config):
        train_data, val_data = self.data.run(config.data)
        model = self.train.run(train_data, val_data, config.train)
        metrics = self.evaluate(model, val_data)
        if metrics['accuracy'] > config.threshold:
            self.serve.deploy(model)
        self.monitor.start()
        return metrics
```

## 147. AI伦理

```python
class FairnessAssessment:
    def __init__(self, model, X, sensitive_attrs, y):
        self.model = model
        self.X = X
        self.sensitive = sensitive_attrs
        self.y = y
    
    def assess(self):
        predictions = self.model.predict(self.X)
        return {
            'demographic_parity': self._demographic_parity(predictions),
            'equalized_odds': self._equalized_odds(predictions),
            'calibration': self._calibration(predictions)
        }
```

## 148. 持续学习

```python
class ContinualLearning:
    def __init__(self, model, memory_size=1000):
        self.model = model
        self.memory = ExemplarMemory(memory_size)
    
    def adapt(self, new_data):
        self.memory.add(new_data)
        self.model.partial_fit(new_data)
```

## 149. 行业应用

### 149.1 计算机视觉应用
- 安防监控
- 自动驾驶
- 医疗影像
- 工业检测

### 149.2 NLP应用
- 智能客服
- 机器翻译
- 内容生成
- 情感分析

### 149.3 推荐系统
- 电商推荐
- 内容推荐
- 广告投放
- 用户增长

## 150. 总结与展望

### 150.1 技术趋势
- 万亿参数大模型
- 多模态融合
- AI Agent
- 具身智能

### 150.2 未来方向
- 科学AI
- 隐私计算
- 可解释AI
- 绿色AI

### 150.3 学习建议
- 打好基础
- 持续实践
- 关注前沿
- 分享交流

---

**深度学习高级技术Part 13 完成！**

**当前知识库: 1.7MB / 10MB**

**持续学习中...** 🚀💪
