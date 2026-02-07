#!/bin/bash
# 列出最近错误
# 使用方法: ./list-errors.sh [limit]

LIMIT="${1:-10}"

echo ""
echo "📋 最近错误列表 (最多$LIMIT个)"
echo "================================================"

count=0

# 查找所有错误文件（按日期倒序）
for errors_dir in $(ls -td ~/.openclaw/workspace/error-logs/errors/*/ 2>/dev/null); do
    if [ "$count" -ge "$LIMIT" ]; then
        break
    fi
    
    for error_file in $(ls -1 "$errors_dir"ERROR-*.md 2>/dev/null); do
        if [ "$count" -ge "$LIMIT" ]; then
            break
        fi
        
        filename=$(basename "$error_file")
        
        # 读取状态
        if grep -q "status: resolved" "$error_file"; then
            status="✅"
        else
            status="🔴"
        fi
        
        # 读取严重程度
        if grep -q "severity: critical" "$error_file"; then
            severity="🔴"
        elif grep -q "severity: high" "$error_file"; then
            severity="🟠"
        elif grep -q "severity: medium" "$error_file"; then
            severity="🟡"
        else
            severity="🟢"
        fi
        
        # 读取错误信息前几行
        error_msg=$(sed -n '/## 错误信息/,/```/p' "$error_file" | sed '1d;$d' | head -1 | sed 's/^[[:space:]]*//' | cut -c1-60)
        if [ -z "$error_msg" ]; then
            error_msg="(无描述)"
        fi
        
        date_str=$(basename "$errors_dir")
        
        echo ""
        echo "$status $severity $filename"
        echo "   📅 $date_str | $error_msg..."
        echo "   📄 $error_file"
        
        count=$((count + 1))
    done
done

if [ $count -eq 0 ]; then
    echo ""
    echo "📭 没有找到错误记录"
    echo ""
    echo "🎯 开始使用:"
    echo "  ./quick-log-error.sh '描述错误' [category] [severity]"
fi

echo ""
echo "================================================"
echo "💡 命令:"
echo "  记录错误: ./quick-log-error.sh '错误描述'"
echo "  解决错误: ./resolve-error.sh ERROR-001 '解决方案'"
echo "  查看统计: python error_logger.py --stats"
echo ""
