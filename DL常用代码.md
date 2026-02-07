# 深度学习常用代码

*精选的深度学习核心代码片段*

---

## 1. PyTorch基础

### 1.1 张量操作

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建张量
x = torch.randn(3, 4)  # 正态分布
x = torch.zeros(3, 4)  # 零张量
x = torch.ones(3, 4)   # 全一张量
x = torch.arange(0, 10, 2)  # 序列

# 张量操作
x = x.view(-1, 1)  # 重塑
x = x.unsqueeze(0)  # 增加维度
x = x.squeeze()    # 移除维度
x = x.clone()       # 复制
```

### 1.2 模型定义

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
```

### 1.3 卷积网络

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
```

---

## 2. 训练循环

```python
import torch.optim as optim

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# 或
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 或
criterion = nn.MSELoss()

# 训练
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## 3. 学习率调度

```python
# 余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

# 阶梯衰减
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# 预热+余弦
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# 1Cycle策略
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, epochs=num_epochs, 
    steps_per_epoch=len(train_loader)
)
```

---

## 4. 正则化

```python
# Dropout
nn.Dropout(p=0.5)

# 批归一化
nn.BatchNorm2d(num_features)

# 权重衰减（L2正则）
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

# 标签平滑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 5. 注意力机制

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        
        return self.w_o(output)
```

---

## 6. Transformer模块

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))
```

---

## 7. 混合精度训练

```python
scaler = torch.cuda.amp.GradScaler()

for inputs, labels in train_loader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 8. 分布式训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 模型
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# 数据
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
```

---

## 9. 模型保存和加载

```python
# 保存整个模型
torch.save(model, 'model.pth')

# 保存状态字典
torch.save(model.state_dict(), 'model_state.pth')

# 加载
model.load_state_dict(torch.load('model_state.pth'))
model.eval()

# 推理
with torch.no_grad():
    outputs = model(input)
```

---

## 10. 梯度检查点

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

---

*深度学习常用代码整理完成！* 💻🚀
