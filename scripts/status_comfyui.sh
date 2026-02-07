#!/bin/bash
# ComfyUI 状态检查脚本

echo "🔍 ComfyUI 状态检查"
echo "========================"

# 检查进程
PIDS=$(ps aux | grep "ComfyUI/main.py" | grep -v grep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "✅ 进程运行中: $PIDS"
else
    echo "❌ 进程未运行"
fi

# 检查端口
if curl -s http://127.0.0.1:8188 >/dev/null 2>&1; then
    echo "✅ API 可访问: http://127.0.0.1:8188"
else
    echo "❌ API 不可访问"
fi

# 检查安装
if [ -d "$HOME/ComfyUI" ]; then
    echo "✅ 安装目录: $HOME/ComfyUI"
else
    echo "❌ 安装目录不存在"
fi

# 检查虚拟环境
if [ -d "$HOME/ComfyUI/venv" ]; then
    echo "✅ 虚拟环境: $HOME/ComfyUI/venv"
    $HOME/ComfyUI/venv/bin/python --version
else
    echo "❌ 虚拟环境不存在"
fi

# 检查输出目录
if [ -d "$HOME/ComfyUI/output" ]; then
    echo "✅ 输出目录: $HOME/ComfyUI/output"
    COUNT=$(ls -1 "$HOME/ComfyUI/output" 2>/dev/null | wc -l)
    echo "📁 生成图像数: $COUNT"
else
    echo "⚠️  输出目录不存在"
fi

# 检查日志
if [ -f "$HOME/comfyui.log" ]; then
    echo "📝 日志文件: $HOME/comfyui.log"
    LAST_LINE=$(tail -3 "$HOME/comfyui.log")
    echo "📊 最后日志:"
    echo "$LAST_LINE"
else
    echo "⚠️  日志文件不存在"
fi

echo ""
echo "========================"
echo "💡 使用命令:"
echo "   启动: ~/.openclaw/workspace/scripts/start_comfyui.sh bg"
echo "   停止: ~/.openclaw/workspace/scripts/stop_comfyui.sh"
