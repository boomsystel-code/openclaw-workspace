# OpenClaw AI生图完全指南

*最后更新: 2026-02-07*

---

## 🎨 概述

OpenClaw支持多种AI生图方式：

1. **ComfyUI** - 本地Stable Diffusion工作流
2. **DALL-E** (GPT技能) - OpenAI云端生成
3. **AI绘本生成** - 百度AI绘本
4. **Google Imagen** - Google图像生成

---

## 🖼️ 1. ComfyUI（推荐 - 本地运行）

### 前提条件

ComfyUI技能已安装，需要手动安装ComfyUI：

```bash
# 克隆ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI

# 创建虚拟环境
cd ~/ComfyUI
python3 -m venv venv

# 安装依赖
~/ComfyUI/venv/bin/pip install -r requirements.txt

# 启动服务器
~/ComfyUI/venv/bin/python ~/ComfyUI/main.py --listen 127.0.0.1
```

### 使用方法

#### 生成图像

告诉OpenClaw：
> "用ComfyUI生成一张赛博朋克风格的东京街景"

OpenClaw会自动：
1. 读取默认工作流
2. 修改提示词节点
3. 设置随机种子
4. 运行工作流
5. 返回生成的图像

#### 指定工作流

> "运行 `~/workflows/my-custom-workflow.json`"

#### 下载模型

> "下载这些模型权重：https://example.com/model.safetensors"

### 工作流文件位置

```
~/ComfyUI/models/checkpoints/     # 基础模型
~/ComfyUI/models/loras/           # LoRA模型
~/ComfyUI/models/vae/             # VAE模型
~/ComfyUI/output/                 # 输出目录
```

---

## 🎨 2. DALL-E（OpenAI云端）

### 配置

编辑 `~/.openclaw/openclaw.json`：

```json
{
  "skills": {
    "entries": {
      "gpt": {
        "enabled": true,
        "env": {
          "OPENAI_API_KEY": "sk-your-api-key"
        },
        "config": {
          "model": "gpt-image-1"
        }
      }
    }
  }
}
```

### 使用方法

#### 生成单张图像

> "用DALL-E生成一只在月球上弹吉他的猫"

#### 生成多张变体

> "生成4张不同风格的小镇风景画"

#### 编辑现有图像

上传图片后说：
> "把这张图的背景改成秋天森林"

---

## 📔 3. AI绘本生成（百度）

### 配置

```json
{
  "skills": {
    "entries": {
      "ai-picture-book": {
        "enabled": true,
        "apiKey": "your-baidu-api-key",
        "env": {
          "BAIDU_API_KEY": "your-api-key"
        }
      }
    }
  }
}
```

### 获取API Key

1. 访问：https://console.bce.baidu.com/qianfan/ais/console/apiKey
2. 注册账号并创建API Key

### 使用方法

#### 生成静态绘本

> "创建一个关于小女孩子喜欢读书的绘本"

#### 生成动态绘本

> "创建一个10秒的动态绘本：一只小狗在海滩上奔跑"

#### 查询进度

> "查询绘本生成任务 26943ed4-f5a9-4306-a05b-b087665433a0"

---

## 🖼️ 4. Google Imagen

### 配置

```json
{
  "skills": {
    "entries": {
      "google-imagen": {
        "enabled": true,
        "apiKey": "your-google-api-key",
        "env": {
          "GOOGLE_API_KEY": "your-key"
        }
      }
    }
  }
}
```

### 使用方法

#### 人像摄影风格

> "用Google Imagen生成一张专业人像摄影：中年男人在书房"

#### 超写实风景

> "生成超写实风景：托斯卡纳的日出，橄榄树庄园"

---

## 📊 功能对比

| 特性 | ComfyUI | DALL-E | 百度绘本 | Google Imagen |
|------|---------|--------|----------|---------------|
| **位置** | 本地 | 云端 | 云端 | 云端 |
| **成本** | 免费（需GPU） | 按次付费 | 按次付费 | 按次付费 |
| **定制性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **速度** | 取决于硬件 | 快 | 中等 | 快 |
| **隐私** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **易用性** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 💡 最佳使用场景

### ComfyUI 最佳用于：
- 高质量、定制化图像
- LoRA风格微调
- ControlNet精确控制
- 批量生成
- 私密图像生成

### DALL-E 最佳用于：
- 快速原型设计
- 概念草图
- 不需要复杂控制的场景
- 偶尔使用

### 百度绘本 最佳用于：
- 儿童绘本创作
- 教育内容
- 故事配图

### Google Imagen 最佳用于：
- 人像摄影
- 风景写实
- 高质量商业图片

---

## 🔧 提示词技巧

### ComfyUI提示词格式

```
[主体], [细节描述], [风格], [光照], [质量修饰词]

# 示例
cyberpunk tokyo street, neon lights, rain slick streets, 
reflective puddles, futuristic buildings, 
cinematic lighting, 8k resolution, highly detailed
```

### DALL-E提示词格式

```
[场景描述] + [风格] + [细节]

# 示例
A small cat playing guitar on the moon, 
soft moonlight, photorealistic, 8k
```

### Google Imagen提示词

```
[类型] of [subject]: [description], 
[photography/illustration/painting], 
[lighting], [composition]

# 示例
Portrait photography of a middle-aged man in a study, 
warm ambient lighting, shallow depth of field, 
professional headshot style
```

---

## 📁 模型存放位置

```
# ComfyUI
~/ComfyUI/models/
├── checkpoints/          # 基础模型 (SDXL, SD1.5等)
├── loras/              # LoRA风格模型
├── vae/                # VAE模型
├── controlnet/          # ControlNet模型
├── upscale_models/      # 超分模型
└── embeddings/         # 文本嵌入

# 输出
~/ComfyUI/output/       # 生成图像保存位置
```

---

## 🚀 快速开始

### 选项1：ComfyUI（本地方便）

```bash
# 安装
git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
cd ~/ComfyUI
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 启动
./venv/bin/python main.py --listen 127.0.0.1

# 使用
# 告诉OpenClaw生成图像
```

### 选项2：DALL-E（快速上手）

```bash
# 配置
export OPENAI_API_KEY="sk-your-key"

# 使用
# 直接告诉OpenClaw生成图像
```

---

## ⚠️ 常见问题

### Q1: ComfyUI连接失败

```
错误：Connection refused to 127.0.0.1:8188

解决：
1. 确保ComfyUI已安装
2. 启动服务器：~/ComfyUI/venv/bin/python ~/ComfyUI/main.py --listen 127.0.0.1
3. 检查端口是否被占用
```

### Q2: 模型加载失败

```
解决：
1. 检查模型文件是否完整
2. 确认模型放在正确目录
3. 查看ComfyUI日志获取详细信息
```

### Q3: 图像生成太慢

```
解决：
1. 使用更小的模型（如SD1.5而不是SDXL）
2. 减少采样步数
3. 使用Streamlined工作流
4. 升级GPU（如有）
```

### Q4: DALL-E API错误

```
解决：
1. 检查API Key是否有效
2. 确认账户有足够配额
3. 检查网络连接
```

---

## 🔗 相关技能

| 技能 | 功能 | 位置 |
|------|------|------|
| `comfyui` | 本地SD工作流 | `~/.openclaw/workspace/skills/comfyui/` |
| `gpt` | DALL-E图像生成 | `~/.openclaw/workspace/skills/gpt/` |
| `ai-picture-book` | 百度AI绘本 | `~/.openclaw/workspace/skills/ai-picture-book/` |
| `imagemagick` | 图像处理 | `~/.openclaw/workspace/skills/imagemagick/` |
| `table-image-generator` | 表格转图像 | `~/.openclaw/workspace/skills/table-image-generator/` |

---

## 📚 资源链接

### ComfyUI资源
- **官网**: https://github.com/comfyanonymous/ComfyUI
- **模型库**: https://civitai.com
- **工作流分享**: https://openart.ai

### DALL-E资源
- **文档**: https://platform.openai.com/docs/guides/images
- **API**: https://platform.openai.com/api-keys

### 学习资源
- **提示词工程**: https://promptengineering.org
- **Stable Diffusion指南**: https://stable-diffusion-art.com

---

## 🎯 下一步

1. **选择方案**：
   - 需要高质量/私密 → ComfyUI
   - 快速使用 → DALL-E
   - 绘本创作 → 百度AI

2. **安装配置**：
   - 按上述指南安装
   - 配置API密钥

3. **开始生成**：
   - 尝试简单提示词
   - 逐步增加复杂度

---

*文档生成时间: 2026-02-07*
