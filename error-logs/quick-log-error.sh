#!/bin/bash
# 快速错误记录脚本
# 使用方法: ./quick-log-error.sh "错误描述" [category] [severity]

ERROR_MSG="${1:-}"
CATEGORY="${2:-other}"
SEVERITY="${3:-medium}"

if [ -z "$ERROR_MSG" ]; then
    echo "❌ 请提供错误描述"
    echo "用法: ./quick-log-error.sh '错误描述' [category] [severity]"
    echo "示例: ./quick-log-error.sh 'API调用超时' technical high"
    exit 1
fi

# 获取下一个错误ID
COUNTER_FILE=~/.openclaw/workspace/error-logs/.counter
if [ -f "$COUNTER_FILE" ]; then
    COUNTER=$(($(cat "$COUNTER_FILE") + 1))
else
    COUNTER=1
fi
echo "$COUNTER" > "$COUNTER_FILE"
ERROR_ID=$(printf "ERROR-%03d" $COUNTER)

# 创建错误记录
TODAY=$(date +%Y-%m-%d)
ERROR_DIR=~/.openclaw/workspace/error-logs/errors/$TODAY
mkdir -p "$ERROR_DIR"

ERROR_FILE="$ERROR_DIR/${ERROR_ID}.md"

cat > "$ERROR_FILE" << EOF
---
title: "$ERROR_ID"
date: $(date "+%Y-%m-%d %H:%M:%S")
status: open
severity: $SEVERITY
category: $CATEGORY
---

## 错误摘要

**错误代码:** $ERROR_ID
**发生时间:** $(date "+%Y-%m-%d %H:%M:%S")
**严重程度:** $SEVERITY
**状态:** open

## 错误信息

\`\`\`
$ERROR_MSG
\`\`\`

## 发生场景

待补充...

## 解决方案

待补充...

---

## 检查清单

- [x] 错误信息已记录
- [ ] 解决方案已找到
- [ ] 经验教训已提炼
- [ ] 已更新patterns.md
EOF

echo "✅ 错误已记录: $ERROR_ID"
echo "📄 文件: $ERROR_FILE"
echo ""
echo "🎯 下一步:"
echo "  1. 编辑文件完善信息: nano $ERROR_FILE"
echo "  2. 找到解决方案后运行: ./resolve-error.sh $ERROR_ID '解决方案'"
echo "  3. 查看所有错误: ./list-errors.sh"
