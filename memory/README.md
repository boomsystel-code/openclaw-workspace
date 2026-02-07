# 每日记忆记录

*每日append-only记录，不做修改*

## 文件命名规范

```
memory/
├── README.md              ← 本文件
├── YYYY-MM-DD.md         ← 每日记录（自动创建）
└── 索引.md              ← 月度索引（每月自动生成）
```

## 使用规则

### 每日记录格式
```markdown
# YYYY-MM-DD

## 📝 今日要点
- 重要对话/决策

## 💡 学到的新知
- 新技能
- 新工具

## 🔧 完成的任务
- 任务1
- 任务2

## ⚠️ 遇到的问题
- 问题1（可能需要记录到error-logs）

## 📅 明日计划
- 计划1
```

## 快速命令

### 创建今日记录
```bash
# 自动创建今日记录
touch ~/.openclaw/workspace/memory/$(date +%Y-%m-%d).md
```

### 查看今日
```bash
cat ~/.openclaw/workspace/memory/$(date +%Y-%m-%d).md
```

### 查看本周
```bash
ls ~/.openclaw/workspace/memory/*.md | xargs grep "^#" | head -20
```

---

## 📅 历史记录

### 2026年2月
- [2026-02-07](./2026-02-07.md) - 错误日志系统+定时任务
- [2026-02-06](./2026-02-06.md) - 投资大师模块+Ridge模型
- [2026-02-05](./2026-02-05.md) - BTC程序开发

---

*最后更新: 2026-02-07*
