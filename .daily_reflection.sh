#!/bin/bash
# Daily reflection and self-improvement script for OpenClaw

REFLECTION_FILE="$HOME/.openclaw/workspace/memory/$(date +%Y-%m-%d).md"
MEMORY_FILE="$HOME/.openclaw/workspace/MEMORY.md"

echo "=========================================="
echo " 每日复盘思考 - $(date '+%Y-%m-%d %H:%M')"
echo "=========================================="
echo ""

# 1. 检查今日完成的任务
echo "📋 今日完成:"
read -p "- " DONE

# 2. 遇到的问题和错误
echo ""
echo "⚠️ 今日问题/错误:"
read -p "- " ISSUES

# 3. 学到的新东西
echo ""
echo "💡 今日学习:"
read -p "- " LEARNED

# 4. 思考和感悟
echo ""
echo "🧠 思考感悟:"
read -p "- " THOUGHTS

# 5. 次日计划
echo ""
echo "📅 次日计划:"
read -p "- " TOMORROW

# 写入今日复盘
cat >> "$REFLECTION_FILE" << EOF

## $(date '+%Y-%m-%d %H:%M')

### ✅ 完成事项
$DONE

### ⚠️ 问题/错误
$ISSUES

### 💡 今日学习
$LEARNED

### 🧠 思考感悟
$THOUGHTS

### 📅 次日计划
$TOMORROW

---
EOF

echo ""
echo "✅ 复盘已保存: $REFLECTION_FILE"

# 更新MEMORY.md（每周总结时用）
if [ $(date +%H) -eq 23 ]; then
    echo "🌙 晚间思考已记录"
fi

echo ""
echo "完成复盘后，建议："
echo "1. 查看MEMORY.md中的长期记忆"
echo "2. 更新AGENTS.md中的经验教训"
echo "3. 规划明天的重点任务"
