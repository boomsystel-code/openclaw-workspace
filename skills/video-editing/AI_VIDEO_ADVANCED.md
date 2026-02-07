# AI视频生成高级学习计划 (2026-02-08)

## 📚 基础知识回顾（已掌握）
- ✅ FFmpeg高级操作
- ✅ Real-ESRGAN超分辨率
- ✅ RIFE/DAIN AI补帧
- ✅ GFPGAN人脸修复
- ✅ AnimeGAN风格迁移

---

## 🎯 新学习路线

### 第一阶段：AI视频生成（2周）

#### 1. 文本到视频 (Text-to-Video)
| 工具 | 类型 | 特点 |
|------|------|------|
| **Sora** | 云端API | OpenAI出品，高质量 |
| **Veo 3.1** | 云端API | Google出品，支持音频 |
| **Runway Gen-3** | 云端API | 创意控制强 |
| **Pika Labs** | 云端API | 快速迭代 |
| **Luma Dream Machine** | 云端API | 免费额度多 |
| **Zeroscope** | 本地开源 | 轻量级，可本地运行 |

#### 2. 图像到视频 (Image-to-Video)
| 工具 | 用途 |
|------|------|
| **AnimateDiff** | 静态图变动画 |
| **Stable Video Diffusion** | SD生态 |
| **DomoAI** | 风格迁移 |
| **Pika Labs** | 图片起首 |

#### 3. 视频到视频 (Video-to-Video)
| 工具 | 功能 |
|------|------|
| **StyleGAN** | 风格转换 |
| **ComfyUI-VideoHelperSuite** | 工作流 |
| **Deforum** | 动画生成 |

### 第二阶段：高级工作流（2周）

#### 1. AI视频工作流平台
```
Timeline Studio - AI视频编辑
TwitCanva-Video-Workflow - 节点式工作流
VideoGraphAI - YouTube Shorts自动化
AutoShorts - 全自动短视频生成
```

#### 2. 本地部署
```bash
# Zeroscope（本地文本到视频）
git clone https://github.com/deathlessinfiniti2r3q/zeroscope.git
cd zeroscope
pip install -r requirements.txt
python inference.py --prompt "一个宇航员在火星上"

# AnimateDiff（本地图像到视频）
git clone https://github.com/guoyww/AnimateDiff.git
# 需要RTX 3090+显卡
```

#### 3. 自动化脚本
```bash
# AI视频生成工作流
./ai_video_workflow.sh "提示词" --engine zeroscope --duration 5s

# 批量处理
./batch_video_enhance.sh --input ./raw --output ./enhanced
```

### 第三阶段：商业应用（持续）

#### 1. 短视频自动化
```
AI脚本 → 文本到视频 → AI配音 → 自动字幕 → 多平台发布
  ↓            ↓           ↓           ↓
LLM API     Veo 3.1     ElevenLabs   FFmpeg
```

#### 2. 广告创意生成
```
产品图 → AI风格迁移 → 动态广告 → A/B测试
   ↓         ↓           ↓         ↓
MJ/SD    Runway/Pika   自动化    数据分析
```

#### 3. 定制化服务
```
客户需求 → 提示词工程 → AI生成 → 人工精修 → 成片
    ↓           ↓          ↓         ↓
需求分析   Sora/Veo    多次迭代   FFmpeg   交付
```

---

## 🛠️ 推荐工具清单

### 云端工具（快速出活）
| 工具 | 用途 | 成本 |
|------|------|------|
| **Sora** | 文本→视频 | 按量计费 |
| **Veo 3.1** | 文本→视频+音频 | Google积分 |
| **Runway Gen-3** | 高级视频 | 订阅制 |
| **ElevenLabs** | AI配音 | 免费+付费 |
| **HeyGen** | 数字人 | 订阅制 |

### 本地工具（量大管饱）
| 工具 | 用途 | 硬件要求 |
|------|------|----------|
| **Zeroscope** | 文本→视频 | RTX 2060+ |
| **ComfyUI** | 节点工作流 | RTX 3060+ |
| **AnimateDiff** | 图→视频 | RTX 3080+ |
| **FFmpeg** | 后处理 | 无要求 |

---

## 📖 学习资源

### GitHub项目
- [Timeline-Studio](https://github.com/chatman-media/timeline-studio) - AI视频编辑
- [VideoGraphAI](https://github.com/SankaiAI/TwitCanva-Video-Workflow) - 节点式工作流
- [AutoShorts](https://github.com/Anil-matcha/AutoShorts) - 短视频自动化
- [Zeroscope](https://github.com/deathlessinfiniti2r3q/zeroscope) - 本地视频生成
- [MaxVideoAI](https://github.com/camgraphe/MaxVideoAi) - 多引擎聚合

### 学习路径
```
1. 先用云端工具熟悉流程 (Sora/Veo)
2. 再学提示词工程 (cinematic, camera moves)
3. 最后本地部署降低成本 (ComfyUI)
4. 搭建自动化工作流
```

---

## 🎯 本周任务

### Day 1-2: 探索云端工具
- [ ] 注册Sora/Veo账号
- [ ] 尝试5个不同类型提示词
- [ ] 记录最佳参数

### Day 3-4: 本地环境搭建
- [ ] 安装ComfyUI
- [ ] 配置VideoHelperSuite
- [ ] 测试AnimateDiff

### Day 5-6: 工作流设计
- [ ] 设计自动化脚本
- [ ] 搭建Prompt模板库
- [ ] 写一篇学习笔记

### Day 7: 实战项目
- [ ] 生成一个完整的AI短视频
- [ ] 包含：AI生成片段 + 配音 + 字幕 + 背景音乐

---

## 💡 提示词技巧

### 电影感提示词结构
```
[镜头类型] [主体] [动作] [场景] [氛围] [光照] [相机运动]
Example: 
Close-up shot of an astronaut walking on Mars desert, 
dust particles floating in golden hour light, 
slow push-in camera, cinematic, 8k
```

### 风格化提示词
```
Style: [艺术家/风格] + [情绪] + [色彩方案]
Example:
Studio Ghibli style, whimsical atmosphere, 
warm orange and teal color palette, soft lighting
```

---

## 📊 成本对比

| 方案 | 1分钟视频成本 | 质量 | 速度 |
|------|---------------|------|------|
| Sora API | $0.50-2.00 | ⭐⭐⭐⭐⭐ | 快 |
| Veo 3.1 | $0.30-1.00 | ⭐⭐⭐⭐⭐ | 快 |
| Runway | $0.50-1.50 | ⭐⭐⭐⭐ | 中 |
| 本地Zeroscope | 电费~$0.10 | ⭐⭐⭐ | 慢 |
| 本地ComfyUI | 电费~$0.05 | ⭐⭐⭐⭐ | 慢 |

---

*创建时间: 2026-02-08*
*目标: 掌握AI视频生成全链路*
