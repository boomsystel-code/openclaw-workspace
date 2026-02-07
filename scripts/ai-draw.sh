#!/bin/bash
# OpenClaw AI生图快捷命令
# 使用方法: ./ai-draw.sh "你的图像描述"

PROMPT="$1"

if [ -z "$PROMPT" ]; then
    echo "🎨 AI生图命令"
    echo "=============="
    echo ""
    echo "使用方法:"
    echo "  ./ai-draw.sh \"赛博朋克城市夜景\""
    echo "  ./ai-draw.sh \"一只在月球上弹吉他的猫\""
    echo ""
    echo "📋 前置条件:"
    echo "  - ComfyUI已启动 (http://127.0.0.1:8188)"
    echo "  - OpenClaw技能已配置"
    echo ""
    echo "💡 提示: 描述越详细越好！"
    echo ""
    exit 0
fi

echo "🎨 正在生成图像..."
echo "📝 提示词: $PROMPT"
echo ""

# 检查ComfyUI是否运行
if ! curl -s http://127.0.0.1:8188 >/dev/null 2>&1; then
    echo "⚠️  ComfyUI未运行，启动中..."
    cd ~/ComfyUI
    nohup ~/ComfyUI/venv/bin/python main.py --listen 127.0.0.1 > ~/comfyui.log 2>&1 &
    echo "⏳ 等待启动..."
    sleep 10
fi

echo "✅ ComfyUI已就绪"
echo ""
echo "📝 现在告诉OpenClaw:"
echo ""
echo "  \"用ComfyUI生成: $PROMPT\""
echo ""
echo "🌐 或直接访问: http://127.0.0.1:8188"
