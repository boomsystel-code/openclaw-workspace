#!/bin/bash
# ComfyUI Installation Script for OpenClaw
# 安装位置: ~/ComfyUI

echo "🚀 开始安装 ComfyUI..."
echo "================================"

# 1. 检查前置条件
echo "📋 检查前置条件..."
command -v git >/dev/null 2>&1 || { echo "❌ 需要安装 Git" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ 需要安装 Python 3" >&2; exit 1; }

# 2. 设置安装路径
COMFYUI_DIR="$HOME/ComfyUI"
VENV_DIR="$COMFYUI_DIR/venv"

echo "📁 安装路径: $COMFYUI_DIR"

# 3. 克隆仓库（如果不存在）
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "📦 克隆 ComfyUI 仓库..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    if [ $? -ne 0 ]; then
        echo "❌ 克隆失败"
        exit 1
    fi
else
    echo "✅ ComfyUI 已存在，跳过克隆"
fi

# 4. 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 创建 Python 虚拟环境..."
    cd "$COMFYUI_DIR"
    python3 -m venv venv
else
    echo "✅ 虚拟环境已存在，跳过创建"
fi

# 5. 激活虚拟环境并安装依赖
echo "📦 安装 Python 依赖..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null 2>&1

# 安装依赖（静默模式，只显示进度）
if [ -f "$COMFYUI_DIR/requirements.txt" ]; then
    pip install -r "$COMFYUI_DIR/requirements.txt" 2>&1 | grep -E "^(Collecting|Installing|Successfully installed|ERROR)" || true
else
    echo "⚠️  requirements.txt 不存在"
fi

deactivate

echo ""
echo "================================"
echo "✅ ComfyUI 安装完成！"
echo ""
echo "📝 下一步操作："
echo ""
echo "1️⃣  启动 ComfyUI 服务器："
echo "   $COMFYUI_DIR/venv/bin/python $COMFYUI_DIR/main.py --listen 127.0.0.1"
echo ""
echo "2️⃣  或在后台运行："
echo "   cd $COMFYUI_DIR && nohup $VENV_DIR/bin/python main.py --listen 127.0.0.1 > ~/comfyui.log 2>&1 &"
echo ""
echo "3️⃣  测试是否运行成功："
echo "   curl http://127.0.0.1:8188"
echo ""
echo "4️⃣  下载模型（可选）："
echo "   ~/ComfyUI/venv/bin/python ~/.openclaw/workspace/skills/comfyui/scripts/download_weights.py --base ~/ComfyUI"
echo ""
echo "📖 详细文档：查看 ~/.openclaw/workspace/AI生图指南.md"
echo ""
