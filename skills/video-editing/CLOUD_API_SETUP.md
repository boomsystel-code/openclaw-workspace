# 云端AI视频工具配置 (2026-02-08)

## 当前API状态

| 服务商 | 状态 | 配置 |
|--------|------|------|
| **Google** | ✅ 已配置 | `google:default` |
| **OpenAI** | ❌ 未配置 | 需要API Key |

---

## 1️⃣ Veo 3.1（Google）- 可立即使用

### 配置检查
```bash
# 检查Google API
cat ~/.openclaw/openclaw.json | grep "google"
```

### 使用方式
```bash
# 通过gcloud CLI
gcloud auth application-default login
gcloud config set project YOUR_PROJECT

# 或者直接用Python SDK
pip install google-generativeai
```

### 视频生成示例
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# Veo 3.1 API调用
response = model.generate_video(
    prompt="A cinematic shot of a astronaut walking on Mars at sunset",
    duration="8s",
    aspect_ratio="16:9"
)
```

---

## 2️⃣ Sora（OpenAI）- 需要配置

### 获取API Key
1. 访问: https://platform.openai.com/api-keys
2. 创建新密钥
3. 添加到config:
```bash
openclaw configure --section openai
```

### 使用方式
```bash
# 环境变量
export OPENAI_API_KEY="sk-..."
```

---

## 3️⃣ Runway Gen-3 - 独立订阅

### 注册地址
https://runwayml.com

### API获取
1. 注册账号
2. Account → API Keys
3. 复制密钥

---

## 📋 今日任务清单

### 任务1: 确认Google API可用
- [ ] 检查项目ID
- [ ] 启用Veo API
- [ ] 测试生成

### 任务2: 申请Sora（可选）
- [ ] 检查OpenAI账户
- [ ] 申请Sora访问
- [ ] 添加API Key

### 任务3: 首次生成
- [ ] 写第一个提示词
- [ ] 生成5秒视频
- [ ] 保存结果

---

## 💡 提示词练习

### 入门练习
```text
1. "A cat sitting on a windowsill, rain outside, cozy lighting"
2. "A drone shot of a mountain peak at sunrise, clouds below"
3. "A close-up of a watch mechanism, steam punk style"
```

### 进阶提示词
```text
"Cinematic wide shot of an ancient temple hidden in mist, 
dramatic lighting, slow camera push-in, 8k quality, film grain"
```

---

## ⚠️ 成本提醒

| 工具 | 成本/分钟 | 免费额度 |
|------|-----------|----------|
| Veo 3.1 | ~$0.10 | 部分免费 |
| Sora | ~$0.50-2.00 | 需申请 |
| Runway | ~$0.50 | 有限 |

---

*创建时间: 2026-02-08*
