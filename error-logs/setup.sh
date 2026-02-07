#!/bin/bash
# 错误日志系统初始化脚本
# 创建所有必要的目录和文件

set -e

ERROR_LOGS_DIR=~/.openclaw/workspace/error-logs

echo "🚀 初始化错误日志系统..."
echo ""

# 创建目录结构
echo "📁 创建目录结构..."
mkdir -p "$ERROR_LOGS_DIR/errors/$(date +%Y-%m-%d)"
mkdir -p "$ERROR_LOGS_DIR/analysis"
mkdir -p "$ERROR_LOGS_DIR/learnings/by-category"
mkdir -p "$ERROR_LOGS_DIR/statistics"
mkdir -p "$ERROR_LOGS_DIR/scripts"

# 设置可执行权限
chmod +x "$ERROR_LOGS_DIR"/*.sh 2>/dev/null || true

# 初始化计数器
echo "1" > "$ERROR_LOGS_DIR/.counter"

# 复制Python脚本
cp error_logger.py "$ERROR_LOGS_DIR/scripts/"

echo "✅ 错误日志系统初始化完成！"
echo ""
echo "📁 结构:"
echo "  $ERROR_LOGS_DIR/"
echo "  ├── README.md                    # 总体说明"
echo "  ├── error_logger.py            # Python工具"
echo "  ├── quick-log-error.sh         # 快速记录错误"
echo "  ├── resolve-error.sh           # 快速解决错误"
echo "  ├── list-errors.sh            # 列出最近错误"
echo "  ├── errors/                    # 错误记录"
echo "  │   └── YYYY-MM-DD/"
echo "  │       ├── ERROR-001.md"
echo "  │       └── summary.md"
echo "  ├── analysis/                  # 分析报告"
echo "  │   ├── patterns.md            # 错误模式"
echo "  │   └── weekly-summary.md"
echo "  ├── learnings/                 # 经验教训"
echo "  │   ├── action-items.md        # 行动项"
echo "  │   └── by-category/"
echo "  └── statistics/                # 统计数据"
echo "      └── metrics.json"
echo ""
echo "🎯 快速开始:"
echo "  记录错误: ./quick-log-error.sh 'API调用失败' technical medium"
echo "  解决错误: ./resolve-error.sh ERROR-001 '已添加重试机制'"
echo "  列出错误: ./list-errors.sh"
echo ""
echo "📚 文档: 查看 README.md 获取详细说明"
