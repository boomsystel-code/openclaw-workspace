#!/bin/bash
# ComfyUI 启动脚本
# 使用方法: ./start_comfyui.sh [background]

COMFYUI_DIR="$HOME/ComfyUI"
VENV_DIR="$COMFYUI_DIR/venv"
LOG_FILE="$HOME/comfyui.log"

echo "🚀 启动 ComfyUI..."
echo "📁 安装目录: $COMFYUI_DIR"

# 检查是否已在运行
if curl -s http://127.0.0.1:8188 >/dev/null 2>&1; then
    echo "✅ ComfyUI 已在运行 (http://127.0.0.1:8188)"
    exit 0
fi

# 激活虚拟环境并启动
source "$VENV_DIR/bin/activate"

if [ "$1" == "background" ] || [ "$1" == "bg" ]; then
    echo "📦 后台启动..."
    cd "$COMFYUI_DIR"
    nohup "$VENV_DIR/bin/python" main.py --listen 127.0.0.1 > "$LOG_FILE" 2>&1 &
    echo "✅ ComfyUI 已后台启动"
    echo "📝 日志: $LOG_FILE"
    echo "🌐 访问: http://127.0.0.1:8188"
    sleep 3
    curl -s http://127.0.0.1:8188 >/dev/null 2>&1 && echo "✅ 服务可用" || echo "⚠️  正在启动..."
else
    echo "📦 前台启动 (按 Ctrl+C 停止)..."
    cd "$COMFYUI_DIR"
    "$VENV_DIR/bin/python" main.py --listen 127.0.0.1
fi
