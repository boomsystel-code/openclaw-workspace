# 云端AI视频工具快速开始

## 当前状态

| 工具 | 状态 | 下一步 |
|------|------|--------|
| **Veo 3.1 (Google)** | ⚠️ 需配置gcloud | 安装或使用网页版 |
| **Sora (OpenAI)** | ❌ 需API Key | 申请访问 |
| **Runway Gen-3** | ❌ 需账号 | 注册 |

---

## 🚀 快速开始方案

### 方案A: 网页版（最快）

#### Veo 3.1 (Google)
1. 打开: https://aistudio.google.com
2. 登录Google账号
3. 找到Video或Veo功能
4. 输入提示词 → 生成

#### Runway
1. 打开: https://app.runwayml.com
2. 注册账号（可用Google登录）
3. 进入Gen-3 Alpha
4. 输入提示词 → 生成

#### Pika Labs（免费额度多）
1. 打开: https://pika.art
2. Discord或网页版登录
3. 直接输入提示词

---

### 方案B: API配置（长期使用）

#### 安装gcloud
```bash
# macOS
brew install google-cloud-sdk

# 配置
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable videointelligence.googleapis.com
```

#### 配置OpenAI
```bash
# 获取Key
访问: https://platform.openai.com/api-keys

# 添加到OpenClaw
openclaw configure --section openai
```

---

## 📝 今日练习：第一次生成

### Step 1: 选择平台
- 新手推荐 → **Pika**（免费，门槛低）
- 质量优先 → **Runway**（效果好）
- 速度快 → **Veo**（Google）

### Step 2: 写提示词
```
简单版:
"一只小猫在草地上跑"

进阶版:
"Cinematic shot of a fluffy kitten running through 
a sunlit meadow, shallow depth of field, 
golden hour lighting, 4k quality"
```

### Step 3: 生成参数
- 长度: 3-5秒（开始）
- 分辨率: 最高可用
- 风格: 默认即可

### Step 4: 保存结果
- 下载视频到: `~/Desktop/ai_videos/`
- 记录参数: 提示词、平台、效果评分

---

## 📂 练习记录模板

```markdown
## 2026-02-08 练习

### 练习1
- **平台**: Pika
- **提示词**: "A cat running in grass"
- **结果**: [链接/描述]
- **评分**: ⭐⭐⭐
- **改进**: 增加相机运动描述

### 练习2
- **平台**: Runway
- **提示词**: "Cinematic wide shot of ocean waves at sunset"
- **结果**: ...
- **评分**: ⭐⭐⭐⭐
```

---

## 🎯 今日目标

- [ ] 选择一个平台注册
- [ ] 完成3个不同类型提示词练习
- [ ] 保存最佳结果到 `~/Desktop/ai_videos/`
- [ ] 记录学习心得

---

## 💡 提示词技巧

### 必包含元素
```
1. 主体 (Subject) - 什么
2. 动作 (Action) - 做什么
3. 场景 (Setting) - 在哪
4. 氛围 (Mood) - 什么感觉
5. 相机 (Camera) - 怎么拍
```

### 格式
```
[相机] [主体] [动作] in [场景], [氛围] lighting, [风格]
```

### 示例
```
"Close-up shot of a hummingbird hovering near 
a red flower, natural sunlight, slow motion, 
cinematic, 8k"
```

---

## 下一步

完成网页版练习后，我们将：
1. 配置API实现自动化
2. 搭建本地工作流
3. 批量生成商业内容

---

*创建时间: 2026-02-08*
