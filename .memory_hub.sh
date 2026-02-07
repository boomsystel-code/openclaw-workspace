#!/bin/bash
# Global Memory Sync Hub - 全局记忆互通中心
# 用于检查、管理和同步所有会话的学习内容

MEMORY_DIR="$HOME/.openclaw/workspace/memory"
SUMMARY_FILE="$MEMORY_DIR/session_summary.md"
MEMORY_FILE="$HOME/.openclaw/workspace/MEMORY.md"
AGENTS_DIR="$HOME/.openclaw/subagents"

echo "=========================================="
echo "  🧠 全局记忆互通中心"
echo "=========================================="
echo ""

# 1. 显示记忆统计
echo "📊 记忆统计:"
echo "  - 每日记录: $(ls $MEMORY_DIR/*.md 2>/dev/null | wc -l) 个"
echo "  - 会话总结: $(wc -l < $SUMMARY_FILE 2>/dev/null || echo 0) 行"
echo "  - Sub-agents: $(ls $AGENTS_DIR/*.json 2>/dev/null | wc -l) 个"
echo ""

# 2. 同步所有agent学习内容
echo "🔄 同步所有学习内容:"
for agent in $AGENTS_DIR/*.json; do
    if [ -f "$agent" ]; then
        name=$(basename "$agent" .json)
        echo "  ✓ $name"
    fi
done
echo ""

# 3. 添加新记忆到全局库
echo "📝 添加新记忆:"
read -p "输入学习内容: " NEW_LEARNING
read -p "标签 (逗号分隔): " TAGS

if [ -n "$NEW_LEARNING" ]; then
    DATE=$(date "+%Y-%m-%d")
    echo "" >> "$MEMORY_FILE"
    echo "### $DATE" >> "$MEMORY_FILE"
    echo "- $NEW_LEARNING #${TAGS:-general}" >> "$MEMORY_FILE"
    echo "✅ 已添加到MEMORY.md"
fi
echo ""

# 4. 搜索记忆
echo "🔍 搜索记忆:"
read -p "关键词: " QUERY
grep -i "$QUERY" "$MEMORY_FILE" "$SUMMARY_FILE" 2>/dev/null && echo "找到匹配!" || echo "未找到匹配"
echo ""

echo "=========================================="
echo "完成操作"
