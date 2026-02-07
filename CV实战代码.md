# CV实战代码

*精选的计算机视觉实战代码*

---

## 1. 图像分类

### 1.1 CNN图像分类

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 全局池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
```

### 1.2 使用预训练模型

```python
from torchvision import models

# 使用预训练ResNet
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(2048, num_classes)  # 修改输出层

# 冻结特征提取层
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.requires_grad = True

# 推理
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

image = Image.open('test.jpg')
input_tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    outputs = resnet(input_tensor)
    probabilities = F.softmax(outputs, dim=1)
```

---

## 2. 目标检测

### 2.1 Faster R-CNN

```python
from torchvision import models
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_Weights

# 加载预训练模型
model = models.detection.fasterrcnn_mobilenet_v3_large(weights="DEFAULT")

# 修改类别数
num_classes = 10  # 自己的类别数 + 背景
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 训练
model.train()
for images, targets in train_loader:
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

# 推理
model.eval()
with torch.no_grad():
    predictions = model(input_images)
```

### 2.2 YOLOv8

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # nano
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium

# 训练
model.train(data='coco.yaml', epochs=100, imgsz=640)

# 推理
results = model('image.jpg')
results[0].show()

# 检测视频
results = model('video.mp4', save=True)
```

---

## 3. 语义分割

### 3.1 U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._conv_block(512, 1024)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e4))
        
        # 解码
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)
```

### 3.2 使用DeepLabV3

```python
from torchvision import models

# 加载预训练DeepLabV3
model = models.segmentation.deeplabv3_resnet50(weights=models.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)

# 推理
model.eval()
with torch.no_grad():
    output = model(input_image)['out']
    prediction = output.argmax(dim=1)
```

---

## 4. 图像增强

```python
from torchvision import transforms

# 训练增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

# 验证增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])
```

---

## 5. 图像生成

### 5.1 GAN训练

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_channels * 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * 28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# 训练
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        
        # 训练判别器
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        d_real = discriminator(real_images)
        d_fake = discriminator(fake_images.detach())
        
        loss_D = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        
        # 训练生成器
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        
        d_fake = discriminator(fake_images)
        loss_G = criterion(d_fake, real_labels)
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
```

### 5.2 Diffusion Model

```python
import torch
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, model, num_timesteps=1000, beta_schedule='linear'):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        
        # 定义噪声调度
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def _cosine_beta_schedule(self, s=0.008, num_timesteps=1000):
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 添加噪声
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod_prod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # 预测噪声
        return self.model(x_t, t)
```

---

## 6. 视觉Transformer

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio))
            for _ in range(depth)
        ])
        
        # Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, num_patches, D]
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification
        return self.head(x[:, 0, :])
```

---

## 7. 图像处理

```python
from PIL import Image
import torchvision.transforms.functional as TF

# 读取图像
image = Image.open('image.jpg')

# 转换为张量
tensor = TF.to_tensor(image)  # [C, H, W]

# 调整大小
resized = TF.resize(tensor, [224, 224])

# 中心裁剪
cropped = TF.center_crop(tensor, [224, 224])

# 水平翻转
flipped = TF.hflip(tensor)

# 转换为PIL图像
pil_image = TF.to_pil_image(tensor)

# 归一化
normalized = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
```

---

## 8. 模型评估

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

# 分类报告
print(classification_report(y_true, y_pred))

# IoU（用于分割）
def calculate_iou(pred, target, num_classes):
    iou = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union > 0:
            iou.append(intersection / union)
    return sum(iou) / len(iou) if iou else 0
```

---

*CV实战代码整理完成！* 📸🖼️
