#!/bin/bash

#=========================================
# OpenClaw 一键备份与恢复工具
# 使用说明：
#   ./openclaw_backup.sh backup    # 备份
#   ./openclaw_backup.sh restore   # 恢复
#   ./openclaw_backup.sh status    # 查看状态
#=========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 路径配置
OPENCLAW_DIR="$HOME/.openclaw"
WORKSPACE_DIR="$OPENCLAW_DIR/workspace"
BACKUP_DIR="$HOME/openclaw_backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/openclaw_backup_$TIMESTAMP"

# 打印函数
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助
show_help() {
    echo "
╔════════════════════════════════════════════════════════════╗
║            OpenClaw 一键备份/恢复工具 v1.0                 ║
╚════════════════════════════════════════════════════════════╝

使用方法:
  ./openclaw_backup.sh backup    - 备份所有配置和文件
  ./openclaw_backup.sh restore  - 恢复最新备份
  ./openclaw_backup.sh status   - 查看当前状态
  ./openclaw_backup.sh list     - 列出所有备份
  ./openclaw_backup.sh help     - 显示此帮助

备份内容:
  ✓ 核心配置 (openclaw.json, credentials, agents)
  ✓ 工作区文件 (MEMORY.md, AI知识库, 脚本)
  ✓ 扩展配置 (extensions, devices)
  ✓ 定时任务 (cron)
  ✓ 其他重要文件

注意事项:
  - 恢复前会自动停止OpenClaw服务
  - 恢复后需要重启服务
  - 建议定期备份以防数据丢失

"
}

# 查看状态
check_status() {
    echo "
╔════════════════════════════════════════════════════════════╗
║                   当前状态                                 ║
╚════════════════════════════════════════════════════════════╝
"

    # 检查OpenClaw是否存在
    if [ -d "$OPENCLAW_DIR" ]; then
        print_success "OpenClaw目录: $OPENCLAW_DIR ✓"

        # 关键文件检查
        if [ -f "$OPENCLAW_DIR/openclaw.json" ]; then
            print_success "核心配置文件存在 ✓"
        else
            print_warning "核心配置文件缺失"
        fi

        if [ -f "$WORKSPACE_DIR/MEMORY.md" ]; then
            MEMORY_SIZE=$(du -h "$WORKSPACE_DIR/MEMORY.md" | cut -f1)
            print_success "长期记忆文件: MEMORY.md ($MEMORY_SIZE) ✓"
        else
            print_warning "长期记忆文件缺失"
        fi

        # 空间使用
        echo ""
        echo "空间使用:"
        du -sh "$OPENCLAW_DIR" 2>/dev/null | while read size dir; do
            echo "  - OpenClaw目录: $size"
        done

        # 工作区文件统计
        if [ -d "$WORKSPACE_DIR" ]; then
            FILE_COUNT=$(find "$WORKSPACE_DIR" -type f | wc -l)
            echo "  - 工作区文件: $FILE_COUNT 个"
        fi

    else
        print_warning "OpenClaw目录不存在"
    fi

    echo ""
    echo "最近的备份:"
    if [ -d "$BACKUP_DIR" ]; then
        ls -lt "$BACKUP_DIR" 2>/dev/null | head -5 | while read line; do
            echo "  $line"
        done
    else
        echo "  无备份记录"
    fi
}

# 列出所有备份
list_backups() {
    echo "
╔════════════════════════════════════════════════════════════╗
║                   可用备份列表                            ║
╚════════════════════════════════════════════════════════════╝
"

    if [ -d "$BACKUP_DIR" ]; then
        BACKUP_COUNT=$(ls -1d "$BACKUP_DIR"/*/ 2>/dev/null | wc -l)
        echo "备份总数: $BACKUP_COUNT"
        echo ""

        ls -lt "$BACKUP_DIR" 2>/dev/null | head -10 | while read line; do
            echo "  $line"
        done
    else
        print_warning "没有找到备份目录"
    fi
}

# 创建备份
create_backup() {
    print_status "开始创建备份..."

    # 创建备份目录
    mkdir -p "$BACKUP_PATH"
    mkdir -p "$BACKUP_PATH/openclaw"
    mkdir -p "$BACKUP_PATH/workspace"

    # 停止OpenClaw服务（可选，不强制停止）
    print_status "正在备份核心配置..."

    # 备份OpenClaw目录（排除日志和临时文件）
    cd "$OPENCLAW_DIR"
    tar -czf "$BACKUP_PATH/openclaw/config.tar.gz" \
        --exclude='logs/*' \
        --exclude='*.sock' \
        --exclude='.DS_Store' \
        openclaw.json \
        credentials/ \
        agents/ \
        extensions/ \
        devices/ \
        cron/ \
        identity/ \
        completions/ 2>/dev/null || true

    # 备份工作区
    print_status "正在备份工作区..."

    # 创建工作区备份目录
    mkdir -p "$BACKUP_PATH/workspace"

    # 需要备份的关键文件
    BACKUP_FILES=(
        "MEMORY.md"
        "ai_knowledge.md"
        "ai_assistant.py"
        "bilibili_ai_knowledge.md"
        "deep_learning_detailed.md"
        "extended_learning.md"
        "learning_notes.md"
        "AGENTS.md"
        "SOUL.md"
        "TOOLS.md"
        "USER.md"
        "IDENTITY.md"
        "HEARTBEAT.md"
        "README_AI_ASSISTANT.md"
        "ai_core.py"
        "ai_enhanced.py"
        "BTC*.py"
        "btc*.py"
    )

    # 复制所有关键文件
    for pattern in "${BACKUP_FILES[@]}"; do
        find "$WORKSPACE_DIR" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | while read file; do
            relpath="${file#$WORKSPACE_DIR/}"
            # 复制文件
            cp "$file" "$BACKUP_PATH/workspace/"
            print_status "  备份: $relpath"
        done
    done

    # 备份子目录
    print_status "正在备份项目目录..."
    cp -r "$WORKSPACE_DIR/btc_trading_system" "$BACKUP_PATH/workspace/" 2>/dev/null || true
    cp -r "$WORKSPACE_DIR/.clawhub" "$BACKUP_PATH/workspace/" 2>/dev/null || true

    # 创建恢复脚本
    print_status "正在生成恢复脚本..."

    cat > "$BACKUP_PATH/restore.sh" << 'RESTORE_EOF'
#!/bin/bash
# OpenClaw 一键恢复脚本
# 使用: ./restore.sh

set -e
BACKUP_DIR="$(dirname \"$0\")\"
OPENCLAW_DIR="$HOME/.openclaw"
WORKSPACE_DIR="$OPENCLAW_DIR/workspace"

echo "OpenClaw 恢复脚本"
echo "================="

# 停止服务
echo "停止OpenClaw服务..."
openclaw gateway stop 2>/dev/null || true

# 恢复配置
echo "恢复核心配置..."
tar -xzf "$BACKUP_DIR/openclaw/config.tar.gz" -C "$BACKUP_DIR/openclaw/" 2>/dev/null || true
cp -rf "$BACKUP_DIR/openclaw/"* "$OPENCLAW_DIR/" 2>/dev/null || true

# 恢复工作区
echo "恢复工作区..."
cp -rf "$BACKUP_DIR/workspace/"* "$WORKSPACE_DIR/" 2>/dev/null || true

echo "恢复完成！"
echo ""
echo "下一步:"
echo "1. 重启服务: openclaw gateway restart"
echo "2. 检查状态: openclaw status"
RESTORE_EOF

    chmod +x "$BACKUP_PATH/restore.sh"

    # 创建备份信息
    cat > "$BACKUP_PATH/backup_info.txt" << EOF
OpenClaw Backup Info
===================
Created: $(date)
Timestamp: $TIMESTAMP
OpenClaw Version: $(cat "$OPENCLAW_DIR/openclaw.json" 2>/dev/null | grep 'lastTouchedVersion' | cut -d'"' -f4 || echo 'unknown')

Backup Contents:
- Core config (openclaw.json, credentials, agents)
- Extensions (extensions/, devices/)
- Cron tasks (cron/)
- Workspace files ($(find "$BACKUP_PATH/workspace" -type f | wc -l) files)

To restore:
1. cd $BACKUP_PATH
2. ./restore.sh
3. openclaw gateway restart
EOF

    # 清理旧备份（保留最近5个）
    print_status "清理旧备份..."
    cd "$BACKUP_DIR"
    ls -td openclaw_backup_* 2>/dev/null | tail -n +6 | while read old; do
        rm -rf "$old"
    done

    # 最终统计
    BACKUP_SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)
    FILE_COUNT=$(find "$BACKUP_PATH" -type f | wc -l)

    echo ""
    print_success "备份完成！"
    echo "================="
    echo "  备份位置: $BACKUP_PATH"
    echo "  备份大小: $BACKUP_SIZE"
    echo "  文件数量: $FILE_COUNT"
    echo ""
    echo "恢复命令:"
    echo "  cd $BACKUP_PATH"
    echo "  ./restore.sh"
}

# 恢复备份
restore_backup() {
    echo "OpenClaw 恢复向导"
    echo "================="

    # 查找最新备份
    if [ ! -d "$BACKUP_DIR" ]; then
        print_error "没有找到备份目录: $BACKUP_DIR"
        exit 1
    fi

    LATEST_BACKUP=$(ls -td "$BACKUP_DIR"/openclaw_backup_* 2>/dev/null | head -1)

    if [ -z "$LATEST_BACKUP" ]; then
        print_error "没有找到可用备份"
        exit 1
    fi

    echo "找到最新备份: $LATEST_BACKUP"
    echo ""

    read -p "确认恢复? (y/n): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消恢复"
        exit 0
    fi

    # 停止服务
    print_status "停止OpenClaw服务..."
    openclaw gateway stop 2>/dev/null || true

    # 恢复配置
    print_status "恢复核心配置..."
    tar -xzf "$LATEST_BACKUP/openclaw/config.tar.gz" -C "$LATEST_BACKUP/openclaw/" 2>/dev/null || true
    cp -rf "$LATEST_BACKUP/openclaw/"* "$OPENCLAW_DIR/" 2>/dev/null || true

    # 恢复工作区
    print_status "恢复工作区..."
    cp -rf "$LATEST_BACKUP/workspace/"* "$WORKSPACE_DIR/" 2>/dev/null || true

    # 重启服务
    print_status "重启服务..."
    openclaw gateway restart 2>/dev/null || true

    echo ""
    print_success "恢复完成！"
    echo "==============="
    echo "建议执行: openclaw status"
}

# 主程序
case "$1" in
    backup)
        create_backup
        ;;
    restore)
        restore_backup
        ;;
    status)
        check_status
        ;;
    list)
        list_backups
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
