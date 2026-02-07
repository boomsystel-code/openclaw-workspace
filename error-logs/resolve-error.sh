#!/bin/bash
# 快速解决错误脚本
# 使用方法: ./resolve-error.sh ERROR-001 "解决方案"

ERROR_ID="${1:-}"
SOLUTION="${2:-}"

if [ -z "$ERROR_ID" ] || [ -z "$SOLUTION" ]; then
    echo "❌ 用法错误"
    echo "用法: ./resolve-error.sh ERROR-001 '解决方案'"
    exit 1
fi

# 找到错误文件
TODAY=$(date +%Y-%m-%d)
ERROR_FILE=~/.openclaw/workspace/error-logs/errors/$TODAY/${ERROR_ID,,}.md

if [ ! -f "$ERROR_FILE" ]; then
    # 尝试在其他日期查找
    ERROR_FILE=$(find ~/.openclaw/workspace/error-logs/errors -name "${ERROR_ID,,}.md" 2>/dev/null | head -1)
fi

if [ -z "$ERROR_FILE" ] || [ ! -f "$ERROR_FILE" ]; then
    echo "❌ 找不到错误文件: $ERROR_ID"
    exit 1
fi

# 更新状态和解决方案
sed -i '' "s/status: open/status: resolved/" "$ERROR_FILE"
sed -i '' "s/## 解决方案\n\n待补充.../## 解决方案\n\n$SOLUTION/" "$ERROR_FILE"

# 添加解决时间
echo "" >> "$ERROR_FILE"
echo "---" >> "$ERROR_FILE"
echo "**解决时间:** $(date "+%Y-%m-%d %H:%M:%S")" >> "$ERROR_FILE"

echo "✅ 错误已解决: $ERROR_ID"
echo "📝 解决方案: $SOLUTION"
echo ""
echo "🎯 下一步:"
echo "  1. 提炼经验教训到action-items.md"
echo "  2. 如果是常见模式，更新patterns.md"
echo "  3. 查看统计: python ~/.openclaw/workspace/error-logs/error_logger.py --stats"
