#!/bin/bash
# 自动配置定时任务
# 使用方法: ./setup-cron-tasks.sh

echo "🚀 开始配置定时任务..."
echo ""

# 1. 每日健康检查 (08:00)
echo "📅 创建每日健康检查任务..."
openclaw cron add daily-health \
  --schedule "0 8 * * *" \
  --payload "每日健康检查：
1. 获取BTC实时价格和波动率
2. 检查交易系统运行状态
3. 运行错误日志统计
4. 总结今日重要事项" \
  --announce \
  --name "每日健康检查"

# 2. 每日记忆整理 (22:00)
echo "📅 创建每日记忆整理任务..."
openclaw cron add daily-memory \
  --schedule "0 22 * * *" \
  --payload "每日记忆整理：
1. 回顾今日对话要点
2. 提取重要决策到MEMORY.md
3. 清理临时笔记文件
4. 更新memory索引" \
  --announce \
  --name "每日记忆整理"

# 3. 每周错误统计 (周一 09:00)
echo "📆 创建每周错误统计任务..."
openclaw cron add weekly-error \
  --schedule "0 9 * * 1" \
  --payload "每周错误统计分析：
1. 运行 python3 ~/.openclaw/workspace/error-logs/error_logger.py --stats
2. 分析本周新错误模式
3. 更新 ~/.openclaw/workspace/error-logs/analysis/patterns.md
4. 生成改进建议" \
  --announce \
  --name "每周错误统计"

# 4. 每周技能回顾 (周日 20:00)
echo "📆 创建每周技能回顾任务..."
openclaw cron add weekly-skill \
  --schedule "0 20 * * 0" \
  --payload "每周技能回顾：
1. 回顾本周使用的Skills
2. 评估每个Skill的效果
3. 识别需要改进的技能
4. 更新Skill配置或学习新技能" \
  --announce \
  --name "每周技能回顾"

# 5. 每月进化回顾 (每月1日 10:00)
echo "📊 创建每月进化回顾任务..."
openclaw cron add monthly-evolution \
  --schedule "0 10 1 * *" \
  --payload "每月进化回顾：
1. 量化本月成长指标（准确率、错误率、解决率）
2. 识别重大改进和突破
3. 设定下月进化目标
4. 更新MEMORY.md核心记录" \
  --announce \
  --name "每月进化回顾"

# 6. 每月系统优化 (每月1日 11:00)
echo "📊 创建每月系统优化任务..."
openclaw cron add monthly-optimize \
  --schedule "0 11 1 * *" \
  --payload "每月系统优化：
1. 分析资源使用情况
2. 优化OpenClaw配置
3. 清理无用文件
4. 更新文档和说明" \
  --announce \
  --name "每月系统优化"

echo ""
echo "✅ 所有定时任务创建完成！"
echo ""
echo "📋 当前定时任务列表:"
openclaw cron list
echo ""
echo "💡 查看任务详情: openclaw cron list --json"
