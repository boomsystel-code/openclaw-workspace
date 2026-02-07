#!/bin/bash
# Session Summary Sync - 会话总结同步脚本
# 所有sub-agent完成任务后调用此脚本，同步学习内容到全局记忆

SUMMARY_FILE="$HOME/.openclaw/workspace/memory/session_summary.md"
MEMORY_FILE="$HOME/.openclaw/workspace/MEMORY.md"
TEMP_FILE="/tmp/memory_update_$$.md"

# 输入参数: $1=任务类型 $2=学习内容 $3=经验教训
TYPE=${1:-"general"}
LEARNING=${2:-""}
LESSON=${3:-""}

DATE=$(date "+%Y-%m-%d %H:%M")

cat >> "$SUMMARY_FILE" << EOF

## Session @ $DATE
**Type**: $TYPE
**Learning**: $LEARNING
**Lesson**: $LESSON

---

EOF

# 每周日更新MEMORY.md
if [ $(date +%u) -eq 7 ]; then
    echo "🌱 每周总结同步到MEMORY.md"
    # 提取本周重点添加到MEMORY.md
fi

echo "✅ 已同步到: $SUMMARY_FILE"
