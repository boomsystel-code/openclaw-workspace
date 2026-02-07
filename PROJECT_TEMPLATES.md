# 📋 项目模板索引

> 最后更新: 2026-02-07
> 提供标准化的项目结构模板

---

## 🏗️ 模板类型

### 1. Python项目模板

```
project_name/
├── 📄 README.md           # 项目说明
├── 📄 requirements.txt   # 依赖
├── 📄 setup.py           # 安装脚本
├── 📁 src/               # 源代码
│   └── __init__.py
├── 📁 tests/            # 测试
│   └── __init__.py
└── 📁 docs/             # 文档
    └── CHANGELOG.md
```

**使用场景**: 创建新的Python项目

### 2. 知识文档模板

```markdown
# 标题

> 简短描述

## 📊 概述
- 主题:
- 难度:
- 更新日期:

## 📖 内容
### 1. 章节一
### 2. 章节二

## 💡 关键点
-

## 🔗 相关链接
-

## 📝 备注
-
```

**使用场景**: 创建新的知识文档

### 3. 每日日志模板

```markdown
# YYYY-MM-DD

## 🎯 今日目标
- [ ] 任务1
- [ ] 任务2

## ✅ 完成事项
- 事项1
- 事项2

## 🔄 进行中
- 事项

## 💭 思考
-

## 📅 明日计划
-
```

**使用场景**: 每日记录 (`memory/YYYY-MM-DD.md`)

### 4. BTC交易策略模板

```markdown
# 策略名称

## 📊 策略概述
- 类型: 趋势/均值回归/套利
- 风险等级: 低/中/高
- 预期收益:

## 🎯 入场条件
- 条件1:
- 条件2:

## 🛡️ 止损规则
-

## 🎯 止盈规则
-

## 📈 回测结果
- 收益率:
- 最大回撤:
- 胜率:
```

**使用场景**: 创建新的交易策略

### 5. 错误日志模板

```markdown
## ERROR-XXX

**时间**: YYYY-MM-DD HH:MM
**类型**: technical/user-interface/automation/integration
**严重程度**: critical/high/medium/low

### 错误信息
```

**错误描述**

### 场景
-

### 解决方案
-

### 经验教训
-
```

**使用场景**: 记录新错误

### 6. 技能开发模板

```
skill_name/
├── 📄 SKILL.md           # 技能描述
├── 📄 README.md          # 使用说明
├── 📁 src/              # 源代码
└── 📁 assets/           # 资源文件
```

**使用场景**: 开发新技能

---

## 📁 现有项目结构参考

### BTC交易系统
```
btc_trading_system/
├── run_btc_trader.py
├── btc_ai_system.py
├── test_data_connection.py
├── OPTIMIZATION_PLAN.md
├── README.md
└── .venv/
```

### 记忆系统
```
memory/
├── MEMORY.md            # 长期记忆
├── YYYY-MM-DD.md       # 每日日志
└── heartbeat-state.json
```

---

## 🚀 快速创建

```bash
# 创建项目目录
mkdir -p project_name/{src,tests,docs}

# 复制模板
cp ~/.openclaw/workspace/templates/README.md ./README.md
```

---

*创建时间: 2026-02-07*
