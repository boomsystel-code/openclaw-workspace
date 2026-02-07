# 定时任务清单

## 📅 每日任务 (Daily)

### 1. 每日健康检查 (Daily Health Check)
- **任务ID:** daily-health-check
- **时间:** 每天 08:00
- **命令:** `openclaw cron add daily-health-check --schedule "0 8 * * *" --payload "执行系统健康检查：1)检查BTC价格波动 2)检查交易系统状态 3)检查错误日志 4)总结今日要点"`
- **状态:** 待创建

### 2. 每日记忆整理 (Daily Memory Flush)
- **任务ID:** daily-memory-flush
- **时间:** 每天 22:00
- **命令:** `openclaw cron add daily-memory-flush --schedule "0 22 * * *" --payload "整理今日memory文件：1)提取重要决策 2)更新MEMORY.md 3)清理临时笔记"`
- **状态:** 待创建

## 📆 每周任务 (Weekly)

### 1. 每周错误统计 (Weekly Error Stats)
- **任务ID:** weekly-error-stats
- **时间:** 每周一 09:00
- **命令:** `openclaw cron add weekly-error-stats --schedule "0 9 * * 1" --payload "运行错误统计：1)执行 error_logger.py --stats 2)分析新错误模式 3)更新patterns.md 4)生成学习建议"`
- **状态:** 待创建

### 2. 每周技能回顾 (Weekly Skill Review)
- **任务ID:** weekly-skill-review
- **时间:** 每周日 20:00
- **命令:** `openclaw cron add weekly-skill-review --schedule "0 20 * * 0" --payload "回顾本周技能使用：1)列出已用Skills 2)评估效果 3)识别改进空间 4)更新Skill配置"`
- **状态:** 待创建

## 📆 每月任务 (Monthly)

### 1. 每月进化回顾 (Monthly Evolution Review)
- **任务ID:** monthly-evolution-review
- **时间:** 每月1日 10:00
- **命令:** `openclaw cron add monthly-evolution-review --schedule "0 10 1 * *" --payload "执行月度进化回顾：1)量化本月成长指标 2)识别重大改进 3)设定下月目标 4)更新MEMORY.md核心记录"`
- **状态:** 待创建

### 2. 每月系统优化 (Monthly System Optimization)
- **任务ID:** monthly-system-optimization
- **时间:** 每月1日 11:00
- **命令:** `openclaw cron add monthly-system-optimization --schedule "0 11 1 * *" --payload "系统优化检查：1)分析资源使用 2)优化配置 3)清理无用文件 4)更新文档"`
- **状态:** 待创建

## 📊 任务创建脚本

### 一次性创建所有任务
```bash
# 每日任务
openclaw cron add daily-health-check --schedule "0 8 * * *" --payload "执行系统健康检查" --announce

openclaw cron add daily-memory-flush --schedule "0 22 * * *" --payload "整理今日memory文件" --announce

# 每周任务
openclaw cron add weekly-error-stats --schedule "0 9 * * 1" --payload "运行错误统计分析" --announce

openclaw cron add weekly-skill-review --schedule "0 20 * * 0" --payload "回顾本周技能使用" --announce

# 每月任务
openclaw cron add monthly-evolution-review --schedule "0 10 1 * *" --payload "执行月度进化回顾" --announce

openclaw cron add monthly-system-optimization --schedule "0 11 1 * *" --payload "系统优化检查" --announce
```

## 🔧 相关脚本位置

- 错误日志: `~/.openclaw/workspace/error-logs/`
- 记忆文件: `~/.openclaw/workspace/memory/*.md`
- 主记忆: `~/.openclaw/workspace/MEMORY.md`
- BTC系统: `~/.openclaw/workspace/btc_*.py`
