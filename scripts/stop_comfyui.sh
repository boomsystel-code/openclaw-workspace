#!/bin/bash
# ComfyUI 停止脚本

echo "🛑 停止 ComfyUI..."

# 查找并终止进程
PIDS=$(ps aux | grep "ComfyUI/main.py" | grep -v grep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "📝 找到进程: $PIDS"
    kill $PIDS 2>/dev/null
    sleep 2
    kill -9 $PIDS 2>/dev/null
    echo "✅ ComfyUI 已停止"
else
    echo "ℹ️  ComfyUI 未运行"
fi

# 清理临时文件（可选）
# rm -f ~/ComfyUI/output/* 2>/dev/null

echo "✨ 完成"
