# OpenClaw 一键备份与恢复工具

## 📖 使用说明

这是一个帮你备份和恢复OpenClaw配置的工具，保存了所有重要数据和设置。

---

## 🚀 快速开始

### 备份（保存当前状态）
```bash
cd ~/.openclaw/workspace
./openclaw_backup.sh backup
```

### 恢复（还原到备份状态）
```bash
cd ~/openclaw_backup/openclaw_backup_YYYYMMDD_HHMMSS
./restore.sh
```

### 查看状态
```bash
cd ~/.openclaw/workspace
./openclaw_backup.sh status
```

---

## 📁 备份内容

### ✅ 已备份
- **核心配置**: openclaw.json, credentials, agents, extensions
- **工作区文件**: MEMORY.md（长期记忆，最重要！）, AI知识库, 脚本
- **扩展配置**: devices/, extensions/, cron/
- **身份信息**: identity/

### ❌ 未备份
- 日志文件（logs/*）
- 临时文件（*.sock）
- YouTube字幕文件（*.vtt）

---

## 💾 备份位置

所有备份保存在：`~/openclaw_backup/`

```
~/openclaw_backup/
├── openclaw_backup_20260206_061930/
│   ├── backup_info.txt      # 备份信息
│   ├── restore.sh           # 恢复脚本（一键恢复）
│   ├── openclaw/           # 核心配置备份
│   │   └── config.tar.gz
│   └── workspace/          # 工作区备份
│       ├── MEMORY.md
│       ├── AI知识库文件
│       └── ...
├── openclaw_backup_20260206_062024/
└── ...
```

---

## 🔧 使用场景

### 场景1：定期备份
建议每周或每次大更新后运行一次：
```bash
cd ~/.openclaw/workspace
./openclaw_backup.sh backup
```

### 场景2：恢复备份
当OpenClaw出现问题时：
```bash
cd ~/openclaw_backup/openclaw_backup_最新日期
./restore.sh
openclaw gateway restart
```

### 场景3：检查备份状态
```bash
cd ~/.openclaw/workspace
./openclaw_backup.sh status
./openclaw_backup.sh list
```

---

## ⚠️ 注意事项

1. **恢复前建议**：
   - 确认当前状态不需要再保留
   - 最好先停止OpenClaw服务

2. **恢复后**：
   - 需要重启OpenClaw服务
   - 建议运行 `openclaw status` 检查

3. **自动清理**：
   - 系统会自动保留最近5个备份
   - 旧备份会被自动删除

---

## 📊 命令参考

| 命令 | 说明 |
|------|------|
| `./openclaw_backup.sh backup` | 创建新备份 |
| `./openclaw_backup.sh restore` | 恢复最新备份 |
| `./openclaw_backup.sh status` | 查看当前状态 |
| `./openclaw_backup.sh list` | 列出所有备份 |
| `./openclaw_backup.sh help` | 显示帮助 |

---

## 🔍 故障排除

### 问题1：恢复后OpenClaw无法启动
```bash
# 检查配置
openclaw doctor

# 查看日志
openclaw logs

# 手动重启
openclaw gateway restart
```

### 问题2：备份失败
```bash
# 检查权限
ls -la ~/.openclaw/

# 检查磁盘空间
df -h ~

# 手动检查
cd ~/.openclaw/workspace
./openclaw_backup.sh status
```

### 问题3：找不到备份
```bash
# 检查备份目录
ls -la ~/openclaw_backup/

# 查看所有备份
./openclaw_backup.sh list
```

---

## 💡 建议

1. **重要更新后立即备份**：修改MEMORY.md、添加新技能后
2. **定期备份**：每周至少一次
3. **多地备份**：可以将 `~/openclaw_backup/` 同步到云端
4. **版本命名**：自动使用时间戳，无需手动命名

---

## 📞 获取帮助

- OpenClaw文档：https://docs.openclaw.ai
- Discord社区：https://discord.com/invite/clawd
- GitHub：https://github.com/openclaw/openclaw

---

*工具版本：v1.0*
*最后更新：2026-02-06*
