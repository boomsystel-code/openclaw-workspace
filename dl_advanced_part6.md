# 🚀 深度学习高级技术 Part 6

*最前沿的深度学习技术与应用*

---

## 100. 大语言模型架构深入

### 100.1 Transformer架构详解

**注意力机制的核心公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**多头注意力的优势**：
- 并行计算不同子空间的特征
- 捕捉不同类型的关系
- 增加模型的表达能力

### 100.2 Transformer变体

**Longformer（长上下文）**：
- 滑动窗口注意力：$O(n \times w)$
- 全局注意力：选择性位置
- 稀疏注意力模式

```python
class LongformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_window = config.attention_window
        self.attention_probs_dropout = nn.Dropout(config.attention_probs_dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        # 滑动窗口注意力
        seq_len = hidden_states.size(1)
        
        # 计算窗口注意力
        output = []
        for i in range(seq_len):
            start = max(0, i - self.attention_window)
            end = min(seq_len, i + self.attention_window + 1)
            
            window = hidden_states[:, start:end]
            attn_weights = self._compute_attention(hidden_states[:, i:i+1], window)
            output.append(attn_weights @ window)
        
        return torch.cat(output, dim=1)
```

**BigBird（稀疏注意力）**：
- 随机注意力：$O(n)$
- 窗口注意力：$O(n \times w)$
- 全局token：$O(n)$

```python
class BigBirdAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.random_ratio = config.random_ratio
        self.blocksize = config.blocksize
    
    def forward(self, query, key, value):
        # 1. 随机注意力
        random_attn = self._random_attention(query, key)
        
        # 2. 窗口注意力（滑动窗口）
        window_attn = self._sliding_attention(query, key)
        
        # 3. 全局token注意力
        global_attn = self._global_attention(query, key, value)
        
        # 融合
        attn = random_attn + window_attn + global_attn
        attn = attn / 3.0
        
        return attn @ value
```

### 100.3 位置编码

**RoPE（Rotary Position Embedding）**：

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # 计算频率
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    def forward(self, seq_len, device):
        # 生成位置
        positions = torch.arange(seq_len, device=device).float()
        
        # 计算角度
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        # 复数表示
        return torch.polar(torch.ones_like(angles), angles)

def rotate_half(x):
    """旋转一半"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用RoPE"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**ALiBi（Attention with Linear Biases）**：

```python
class ALiBiAttention(nn.Module):
    def __init__(self, num_heads, slope_init=0.5):
        super().__init__()
        self.num_heads = num_heads
        
        # 可学习的斜率
        self.slopes = nn.Parameter(torch.tensor([slope_init * (2 ** (-i)) 
                                                for i in range(num_heads)]))
    
    def forward(self, query, key, value):
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # ALiBi偏置
        seq_len = query.size(2)
        positions = torch.arange(seq_len, device=query.device).float()
        
        # 创建偏置矩阵
        bias = positions.unsqueeze(0) - positions.unsqueeze(1)
        bias = bias.abs() * -self.slopes.view(1, self.num_heads, 1, 1)
        
        # 应用偏置
        attn_scores = attn_scores + bias
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        return torch.matmul(attn_probs, value)
```

### 100.4 KV Cache优化

```python
class KVCache:
    def __init__(self, batch_size, num_heads, head_dim, max_seq_len):
        self.key_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim
        )
        self.value_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim
        )
        self.seen_tokens = 0
    
    def update(self, key_states, value_states):
        """更新cache"""
        self.key_cache[:, :, self.seen_tokens:self.seen_tokens+key_states.size(2)] = key_states
        self.value_cache[:, :, self.seen_tokens:self.seen_tokens+value_states.size(2)] = value_states
        self.seen_tokens += key_states.size(2)
    
    def get(self, seq_len):
        """获取cache"""
        return self.key_cache[:, :, :seq_len], self.value_cache[:, :, :seq_len]

class CacheAwareAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cache = None
    
    def forward(self, query, key, value, use_cache=False):
        if use_cache and self.cache is not None:
            # 更新cache
            self.cache.update(key, value)
            key, value = self.cache.get(self.cache.seen_tokens)
        
        # 标准注意力
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = attn_scores / (query.size(-1) ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        return torch.matmul(attn_probs, value)
```

---

## 101. 混合专家模型（MoE）

### 101.1 MoE架构

```python
class MixtralExpert(nn.Module):
    """Mixtral专家模块"""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
    
    def forward(self, x):
        # SwiGLU激活
        hidden = self.w1(x)
        gate = F.silu(hidden)
        hidden = self.w3(x) * gate
        return self.w2(hidden)

class MoELayer(nn.Module):
    """MoE层"""
    
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家路由器
        self.router = nn.Linear(hidden_size, num_experts)
        
        # 专家池
        self.experts = nn.ModuleList([
            MixtralExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 计算路由logits
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # 归一化权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 专家计算
        final_hidden = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_weight = top_k_weights[expert_mask].sum(dim=-1, keepdim=True)
                expert_output = self.experts[expert_idx](x[expert_mask])
                final_hidden[expert_mask] += expert_output * expert_weight.unsqueeze(-1)
        
        return final_hidden

class SwitchTransformer(nn.Module):
    """Switch Transformer"""
    
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.ReLU(),
                nn.Linear(intermediate_size, hidden_size)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 路由
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # 软切换：加权所有专家
        final_output = torch.zeros_like(x)
        
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(x)
            final_output += expert_output * routing_weights[:, expert_idx:expert_idx+1]
        
        return final_output
```

### 101.2 负载均衡损失

```python
class LoadBalancedLoss:
    def __init__(self, num_experts, expert_capacity_factor=1.0):
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor
    
    def compute_loss(self, routing_probs, expert_indices):
        # 1. 专家选择频率
        expert_selection_freq = torch.bincount(
            expert_indices.view(-1), 
            minlength=self.num_expects
        ).float()
        
        # 2. 理想均匀分布
        ideal_freq = torch.ones_like(expert_selection_freq) / self.num_experts
        
        # 3. 负载均衡损失
        load_balance_loss = F.kl_div(
            routing_probs.mean(dim=0).log(),
            ideal_freq,
            reduction='batchmean'
        )
        
        # 4. 帮助辅助损失：路由更均匀
        aux_loss = self.num_experts * torch.sum(
            routing_probs * (routing_probs.mean(dim=0) - ideal_freq)
        )
        
        return load_balance_loss + aux_loss
```

### 101.3 GLaM架构

```python
class GLaM(nn.Module):
    """GLaM: Generalist Language Model"""
    
    def __init__(self, num_layers, hidden_size, num_heads, 
                 num_experts, top_k):
        super().__init__()
        self.layers = nn.ModuleList([
            GLaMTransformerLayer(hidden_size, num_heads, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, attention_mask=None):
        # 残差连接
        residual = x
        
        # 自注意力
        x = self.attention_norm(x)
        x = x + self._causal_attention(x, attention_mask)
        
        # MoE FFN
        x = self.ffn_norm(x)
        x = x + self._moe_feed_forward(x)
        
        return x
```

---

## 102. 检索增强生成（RAG）深入

### 102.1 向量检索

```python
class VectorDatabase:
    """向量数据库"""
    
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.documents = []
        self.metadata = []
    
    def add(self, documents, embeddings, metadata=None):
        """添加文档"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata or [{}] * len(documents))
    
    def search(self, query_embedding, top_k=10):
        """向量检索"""
        # 计算相似度
        similarities = [
            cosine_similarity(query_embedding, emb) 
            for emb in self.embeddings
        ]
        
        # Top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'document': self.documents[idx],
                'embedding': self.embeddings[idx],
                'metadata': self.metadata[idx],
                'score': similarities[idx]
            }
            for idx in top_indices
        ]

class FAISSVectorStore:
    """FAISS向量存储"""
    
    def __init__(self, embedding_dim=768, metric='cosine'):
        self.embedding_dim = embedding_dim
        self.metric = metric
        
        if metric == 'cosine':
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.documents = []
    
    def add(self, documents, embeddings):
        """添加文档"""
        self.documents.extend(documents)
        
        # 归一化（余弦相似度需要）
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query_embedding, top_k=10):
        """检索"""
        # 归一化查询
        if self.metric == 'cosine':
            faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(
            query_embedding.astype('float32'), top_k
        )
        
        return [
            {
                'document': self.documents[idx],
                'score': score
            }
            for idx, score in zip(indices[0], scores[0])
        ]
```

### 102.2 检索策略

```python
class HybridRetrieval:
    """混合检索"""
    
    def __init__(self, dense_index, sparse_index):
        self.dense_retriever = dense_index
        self.sparse_retriever = sparse_index
    
    def search(self, query, top_k=10, alpha=0.5):
        """混合检索：稠密 + 稀疏"""
        # 稠密检索
        dense_results = self.dense_retriever.search(query, top_k * 2)
        
        # 稀疏检索
        sparse_results = self.sparse_retriever.search(query, top_k * 2)
        
        # 融合分数
        fused_results = self._fusion(
            dense_results, 
            sparse_results, 
            top_k, 
            alpha
        )
        
        return fused_results
    
    def _fusion(self, dense_results, sparse_results, top_k, alpha):
        """分数融合"""
        # 归一化
        max_dense = max(r['score'] for r in dense_results) if dense_results else 1
        max_sparse = max(r['score'] for r in sparse_results) if sparse_results else 1
        
        # RRF融合
        fused = {}
        for rank, result in enumerate(dense_results):
            doc_id = result['document']['id']
            rrf_score = 1.0 / (rank + 60)
            fused[doc_id] = {
                'document': result['document'],
                'score': alpha * (result['score'] / max_dense) + 
                        (1 - alpha) * rrf_score
            }
        
        for rank, result in enumerate(sparse_results):
            doc_id = result['document']['id']
            if doc_id in fused:
                fused[doc_id]['score'] += (1 - alpha) * rrf_score
            else:
                rrf_score = 1.0 / (rank + 60)
                fused[doc_id] = {
                    'document': result['document'],
                    'score': (1 - alpha) * rrf_score
                }
        
        # 排序
        sorted_results = sorted(fused.values(), 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        return sorted_results[:top_k]

class Reranker:
    """重排序模型"""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM"):
        from sentence_transformers import CrossEncoder
        
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, candidates, top_k=5):
        """重排序"""
        # 构建输入对
        pairs = [(query, cand['document']['text']) for cand in candidates]
        
        # 计算分数
        scores = self.model.predict(pairs)
        
        # 排序
        for cand, score in zip(candidates, scores):
            cand['rerank_score'] = score
        
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return candidates[:top_k]
```

### 102.3 Agentic RAG

```python
class AgenticRAG:
    """智能体驱动的RAG"""
    
    def __init__(self, llm, retriever, tools):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools
        self.memory = []
    
    def query(self, question):
        """智能查询"""
        # 1. 分析问题
        analysis = self._analyze_question(question)
        
        # 2. 决定检索策略
        if analysis['needs_knowledge']:
            # 执行检索
            retrieved_docs = self._retrieve(question)
            
            # 重排序
            reranked_docs = self._rerank(question, retrieved_docs)
            
            # 生成上下文
            context = self._build_context(reranked_docs)
        else:
            context = ""
        
        # 3. 调用工具（如需要计算）
        if analysis['needs_calculation']:
            result = self._call_tool(analysis['tool'], analysis['params'])
            context += f"\n\n[工具结果]: {result}"
        
        # 4. 生成回答
        answer = self._generate_answer(question, context)
        
        # 5. 记忆
        self.memory.append({
            'question': question,
            'answer': answer,
            'context': context
        })
        
        return answer
    
    def _analyze_question(self, question):
        """分析问题类型"""
        system_prompt = """
        分析用户问题的类型：
        1. 是否需要外部知识？
        2. 是否需要计算或工具调用？
        3. 需要什么类型的工具？
        
        返回JSON格式。
        """
        
        response = self.llm.generate(
            system_prompt + question,
            format='json'
        )
        
        return json.loads(response)
    
    def _retrieve(self, query):
        """检索"""
        return self.retriever.search(query, top_k=10)
    
    def _rerank(self, query, documents):
        """重排序"""
        reranker = Reranker()
        return reranker.rerank(query, documents, top_k=5)
    
    def _build_context(self, documents):
        """构建上下文"""
        context = "\n\n".join([
            f"文档{i+1}: {doc['document']['text']}"
            for i, doc in enumerate(documents)
        ])
        return context
```

### 102.4 RAG评估

```python
class RAGEvaluator:
    """RAG评估"""
    
    def __init__(self):
        self.metrics = {
            'faithfulness': FaithfulnessMetric(),
            'answer_relevance': AnswerRelevanceMetric(),
            'context_precision': ContextPrecisionMetric(),
            'context_recall': ContextRecallMetric()
        }
    
    def evaluate(self, question, answer, contexts, ground_truth=None):
        """评估RAG系统"""
        results = {}
        
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.compute(
                question, answer, contexts, ground_truth
            )
        
        # 综合分数
        results['overall'] = np.mean(list(results.values()))
        
        return results

class FaithfulnessMetric:
    """忠实度：回答是否忠实于检索的上下文"""
    
    def compute(self, question, answer, contexts, ground_truth=None):
        # 提取答案中的声明
        claims = self._extract_claims(answer)
        
        # 检查每个声明是否在上下文中
        supported_claims = []
        for claim in claims:
            if self._claim_supported(claim, contexts):
                supported_claims.append(claim)
        
        return len(supported_claims) / len(claims) if claims else 0
```

---

## 103. 强化学习高级技巧

### 103.1 分布式强化学习

```python
class ApeXDistributedRL:
    """Ape-X分布式RL"""
    
    def __init__(self, num_actors, num_learners, env_fn):
        self.num_actors = num_actors
        self.num_learners = num_learners
        
        # 优先级经验回放
        self.replay_buffer = PrioritizedReplayBuffer(capacity=1000000)
        
        # 共享参数服务器
        self.param_server = ParameterServer()
        
        # 参与者
        self.actors = [
            Actor(i, env_fn, self.param_server, self.replay_buffer)
            for i in range(num_actors)
        ]
        
        # 学习者
        self.learners = [
            Learner(j, self.param_server, self.replay_buffer)
            for j in range(num_learners)
        ]
    
    def train(self, total_timesteps):
        # 启动所有组件
        for actor in self.actors:
            actor.start()
        
        for learner in self.learners:
            learner.start()
        
        # 等待完成
        for actor in self.actors:
            actor.join()
        
        for learner in self.learners:
            learner.join()

class PrioritizedReplayBuffer:
    """优先级经验回放"""
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
        
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        max_priority = self.priorities[:self.position].max() if self.position > 0 else self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """采样"""
        # 计算beta
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * 
                   self.frame / self.beta_frames)
        self.frame += 1
        
        # 概率采样
        probabilities = self.priorities[:len(self.buffer)] ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # 重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, torch.tensor(weights)
    
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
```

### 103.2 离线强化学习

```python
class ConservativeQLearning:
    """保守Q学习（CQL）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Q网络
        self.Q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 目标Q网络
        self.Q_target = copy.deepcopy(self.Q)
        
        self.optimizer = optim.Adam(self.Q.parameters(), lr=1e-4)
        
        # CQL参数
        self.alpha = 1.0  # 保守系数
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # 计算当前Q值
        q_values = self.Q(torch.cat([states, actions], dim=1))
        
        # TD目标
        with torch.no_grad():
            next_q = self.Q_target(next_states).max(dim=1)[0]
            target = rewards + (1 - dones) * 0.99 * next_q
        
        # 标准MSE损失
        td_loss = F.mse_loss(q_values.squeeze(), target)
        
        # CQL损失：在采样动作上的Q值期望
        cql_q_values = self.Q(torch.cat([states, actions], dim=1))
        cql_loss = self.alpha * (cql_q_values.mean() - q_values.mean())
        
        # 总损失
        loss = td_loss + cql_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update()
        
        return loss.item()
    
    def _soft_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(
                0.005 * param.data + 0.995 * target_param.data
            )

class ImplicitQLearning:
    """隐式Q学习（IQL）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, tau=0.005):
        super().__init__()
        
        # 价值网络
        self.V = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q网络
        self.Q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 策略网络
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.V_optimizer = optim.Adam(self.V.parameters(), lr=1e-4)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=1e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        
        self.tau = tau
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # 1. 更新V网络
        with torch.no_grad():
            q_values = self.Q(torch.cat([states, actions], dim=1))
        v_values = self.V(states)
        
        v_loss = F.mse_loss(v_values, q_values.detach())
        
        self.V_optimizer.zero_grad()
        v_loss.backward()
        self.V_optimizer.step()
        
        # 2. 更新Q网络
        q_values = self.Q(torch.cat([states, actions], dim=1))
        with torch.no_grad():
            next_v = self.V(next_states)
        q_targets = rewards + (1 - dones) * 0.99 * next_v
        q_loss = F.mse_loss(q_values, q_targets)
        
        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()
        
        # 3. 更新策略
        policy_actions = self.policy(states)
        q_values_policy = self.Q(torch.cat([states, policy_actions], dim=1))
        
        # 期望：取Q值高的动作
        policy_loss = -q_values_policy.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
```

### 103.3 强化学习训练技巧

```python
class RLTrainer:
    """RL训练器"""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        
        # 归一化
        self.state_normalizer = RunningMeanStd(shape=env.observation_space.shape)
        self.reward_scaler = RunningMeanStd()
    
    def train(self, num_episodes=1000, eval_interval=10):
        """训练"""
        for episode in range(num_episodes):
            state = self.env.reset()
            
            episode_reward = 0
            done = False
            
            while not done:
                # 归一化状态
                state_norm = self.state_normalizer.normalize(state)
                
                # 选择动作
                action = self.agent.select_action(state_norm)
                
                # 执行
                next_state, reward, done, _ = self.env.step(action)
                
                # 归一化奖励
                reward = self.reward_scaler.normalize(reward)
                
                # 存储
                self.agent.replay_buffer.push(
                    state_norm, action, reward, 
                    self.state_normalizer.normalize(next_state), done
                )
                
                # 更新
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    batch = self.agent.replay_buffer.sample(self.agent.batch_size)
                    self.agent.update(batch)
                
                state = next_state
                episode_reward += reward
            
            # 评估
            if episode % eval_interval == 0:
                eval_reward = self.evaluate()
                print(f"Episode {episode}: Train {episode_reward:.2f}, Eval {eval_reward:.2f}")
    
    def evaluate(self, num_episodes=10):
        """评估"""
        total_reward = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                state_norm = self.state_normalizer.normalize(state)
                action = self.agent.select_action(state_norm, eval=True)
                
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
        
        return total_reward / num_episodes

class RunningMeanStd:
    """运行均值标准差"""
    
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 0
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        
        self.mean, self.var, self.count = self._update(
            self.mean, self.var, self.count,
            batch_mean, batch_var, batch_count
        )
    
    def _update(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        new_count = count + batch_count
        new_mean = mean + delta * batch_count / new_count
        new_var = var + batch_var * batch_count + \
                  delta ** 2 * count * batch_count / new_count
        new_var /= new_count
        return new_mean, new_var, new_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
```

---

## 104. 多模态学习

### 104.1 CLIP深入解析

```python
class CLIPModel(nn.Module):
    """CLIP模型"""
    
    def __init__(self, vision_model, text_model, projection_dim=512):
        super().__init__()
        
        self.vision_model = vision_model
        self.text_model = text_model
        
        # 投影层
        self.visual_projection = nn.Linear(vision_model.hidden_size, projection_dim)
        self.text_projection = nn.Linear(text_model.hidden_size, projection_dim)
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, input_ids, attention_mask):
        # 图像特征
        image_features = self.vision_model(images)
        image_embeddings = self.visual_projection(image_features)
        
        # 文本特征
        text_features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = self.text_projection(text_features)
        
        # 归一化
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # 计算相似度
        logits = torch.matmul(image_embeddings, text_embeddings.T) * self.temperature.exp()
        
        return logits
    
    def contrastive_loss(self, image_embeddings, text_embeddings):
        """对比损失"""
        # 图像到文本
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        labels = torch.arange(len(logits)).to(logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2

class CLIPVisionEncoder(nn.Module):
    """CLIP视觉编码器"""
    
    def __init__(self, embed_dim=768, image_size=224, patch_size=16):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # 图像分块嵌入
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码
        self.position_embedding = nn.Embedding(self.num_patches + 1, embed_dim)
        self.register_buffer(
            "position_ids", 
            torch.arange(self.num_patches + 1).expand(1, -1)
        )
        
        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12),
            num_layers=12
        )
        
        # 层归一化
        self.layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, images):
        # 分块
        patches = self.patch_embedding(images)
        patches = patches.flatten(2).transpose(1, 2)
        
        # 添加class token
        class_token = self.class_token.expand(patches.size(0), -1, -1)
        patches = torch.cat([class_token, patches], dim=1)
        
        # 添加位置编码
        patches = patches + self.position_embedding(self.position_ids)
        
        # Transformer
        features = self.transformer(patches)
        
        # CLS token
        return self.layernorm(features[:, 0, :])
```

### 104.2 多模态大模型

```python
class LLaVAModel(nn.Module):
    """LLaVA多模态模型"""
    
    def __init__(self, vision_tower, language_model, mm_projection_dim=512):
        super().__init__()
        
        self.vision_tower = vision_tower
        self.language_model = language_model
        
        # 视觉投影
        self.mm_projector = nn.Sequential(
            nn.Linear(vision_tower.hidden_size, 4096),
            nn.GELU(),
            nn.Linear(4096, language_model.hidden_size)
        )
        
        # 图像标记
        self.image_newline = nn.Parameter(
            torch.zeros(1, 1, language_model.hidden_size)
        )
    
    def forward(self, images, input_ids, attention_mask):
        # 图像特征
        image_features = self.vision_tower(images)
        
        # 投影到语言模型空间
        image_features = self.mm_projector(image_features)
        
        # 找到输入中的图像占位符位置
        image_token_mask = (input_ids == IMAGE_TOKEN_ID)
        
        # 在占位符位置插入图像特征
        input_ids_new = []
        attention_mask_new = []
        image_features_new = []
        
        for b in range(input_ids.size(0)):
            ids = []
            masks = []
            feats = []
            
            for i, (id_, is_image) in enumerate(zip(input_ids[b], image_token_mask[b])):
                if is_image:
                    # 插入图像特征
                    feats.append(image_features[b])
                else:
                    ids.append(id_)
                    masks.append(attention_mask[b