# 🎨 ComfyUI 安装完成！

## ✅ 安装验证

```bash
# 检查Python环境
~/ComfyUI/venv/bin/python --version
# Python 3.14.2

# 检查PyTorch
~/ComfyUI/venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# PyTorch: 2.10.0
```

## 🚀 快速开始

### 1️⃣ 启动 ComfyUI

```bash
# 后台启动（推荐）
~/.openclaw/workspace/scripts/start_comfyui.sh bg

# 前台启动（调试用）
~/.openclaw/workspace/scripts/start_comfyui.sh
```

### 2️⃣ 检查状态

```bash
~/.openclaw/workspace/scripts/status_comfyui.sh
```

### 3️⃣ 停止 ComfyUI

```bash
~/.openclaw/workspace/scripts/stop_comfyui.sh
```

## 📖 OpenClaw中使用

安装完成后，直接告诉OpenClaw：

> "用ComfyUI生成一张赛博朋克风格的东京街景"

OpenClaw会自动：
1. ✅ 读取默认工作流
2. ✅ 修改提示词
3. ✅ 运行生成
4. ✅ 返回图像给你

## 📁 文件位置

```
~/ComfyUI/
├── main.py              # 主程序
├── venv/                # Python虚拟环境
├── models/              # 模型文件（需要下载）
│   ├── checkpoints/     # 基础模型
│   ├── loras/          # LoRA模型
│   └── vae/            # VAE模型
└── output/              # 生成图像输出
```

## 📥 下载模型（可选）

```bash
# 下载SDXL基础模型
~/ComfyUI/venv/bin/python ~/.openclaw/workspace/skills/comfyui/scripts/download_weights.py \
  --base ~/ComfyUI \
  --subfolder checkpoints \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# 下载SD1.5基础模型
~/ComfyUI/venv/bin/python ~/.openclaw/workspace/skills/comfyui/scripts/download_weights.py \
  --base ~/ComfyUI \
  --subfolder checkpoints \
  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
```

## 🌐 访问 Web UI

启动后，访问：
- **Web UI**: http://127.0.0.1:8188
- **API**: http://127.0.0.1:8188/api

## ⚠️ 常见问题

### Q: 启动后无法访问？

```bash
# 检查是否启动
~/.openclaw/workspace/scripts/status_comfyui.sh

# 查看日志
tail -20 ~/comfyui.log
```

### Q: 内存不足？

- 使用较小的模型（SD1.5而不是SDXL）
- 减少批量大小
- 使用512x512而不是1024x1024

### Q: 速度慢？

- 确保使用GPU（MPS/CUDA）
- 减少采样步数（20-30步足够）

## 💡 提示

1. **首次使用**：先下载一个基础模型
2. **Web UI**：可以直接在浏览器中可视化编辑工作流
3. **OpenClaw集成**：OpenClaw会自动处理工作流编辑和生成

## 🔗 相关文档

- 📄 **AI生图指南**: ~/.openclaw/workspace/AI生图指南.md
- 📄 **ComfyUI技能**: ~/.openclaw/workspace/skills/comfyui/SKILL.md
- 🌐 **ComfyUI官网**: https://github.com/comfyanonymous/ComfyUI
- 🎨 **模型下载**: https://civitai.com

---

*创建时间: 2026-02-07*
