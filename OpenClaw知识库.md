# OpenClaw 知识库

*最后更新: 2026-02-07*

---

## 🦞 OpenClaw 简介

OpenClaw 是一个自托管的AI助手网关，连接你的聊天应用到AI编码Agent。支持多平台、多渠道。

### 核心特点

- **自托管**: 运行在自有硬件上，数据完全私有
- **多渠道**: WhatsApp、Telegram、Discord、iMessage、Google Chat、Slack、Signal等
- **Agent原生**: 为编码Agent设计，支持工具使用、会话管理、记忆、多Agent路由
- **开源**: MIT许可，社区驱动

### 系统架构

```
消息渠道 (WhatsApp/Telegram/Discord/...)
    │
    ▼
┌─────────────────────────────┐
│ Gateway (控制平面)           │
│ ws://127.0.0.1:18789        │
└──────────────┬──────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
  Pi Agent   CLI命令    Web界面
```

---

## 📦 安装与配置

### 系统要求

- **运行时**: Node.js ≥ 22
- **推荐模型**: Anthropic Claude Pro/Max (100/200) + Opus 4.6
- **平台**: macOS、Linux、Windows (WSL2)

### 快速安装

```bash
# npm安装
npm install -g openclaw@latest
# 或 pnpm
pnpm add -g openclaw@latest

# 运行引导向导（推荐）
openclaw onboard --install-daemon

# 启动网关
openclaw gateway --port 18789 --verbose
```

### 配置示例

`~/.openclaw/openclaw.json`:

```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-6"
  },
  "channels": {
    "telegram": {
      "botToken": "你的BOT_TOKEN"
    },
    "whatsapp": {
      "allowFrom": ["+1234567890"]
    }
  }
}
```

---

## 💬 支持的渠道

| 渠道 | 类型 | 说明 |
|------|------|------|
| **WhatsApp** | 即时通讯 | Baileys |
| **Telegram** | 即时通讯 | grammY |
| **Discord** | 社区平台 | discord.js |
| **Slack** | 团队协作 | Bolt |
| **Google Chat** | 团队协作 | Chat API |
| **Signal** | 即时通讯 | signal-cli |
| **iMessage** | 即时通讯 | BlueBubbles (推荐) / imsg |
| **Microsoft Teams** | 团队协作 | Bot Framework |
| **Matrix** | 去中心化 | 扩展支持 |
| **Zalo** | 即时通讯 | 扩展支持 |
| **WebChat** | Web界面 | 内置Web UI |

### Telegram 配置

```json
{
  "channels": {
    "telegram": {
      "botToken": "123456:ABCDEF",
      "groups": {
        "*": {
          "requireMention": true
        }
      },
      "allowFrom": ["*"]  // 或指定用户ID列表
    }
  }
}
```

### WhatsApp 配置

```bash
# 登录设备
pnpm openclaw channels login

# 配置文件
{
  "channels": {
    "whatsapp": {
      "allowFrom": ["+1234567890"],
      "groups": {
        "*": {
          "requireMention": true
        }
      }
    }
  }
}
```

---

## 🔧 核心工具

### CLI 命令

```bash
# 发送消息
openclaw message send --to +1234567890 --message "Hello"

# 与Agent对话
openclaw agent --message "帮我写代码" --thinking high

# 查看状态
openclaw status

# 健康检查
openclaw doctor

# 配对管理
openclaw pairing approve
```

### Gateway 命令

```bash
# 启动网关
openclaw gateway --port 18789 --verbose

# 重启网关
openclaw gateway restart

# 查看日志
openclaw logs
```

---

## 🧠 Agent 系统

### 会话类型

1. **Main Session** (主会话) - 直接聊天，全权限
2. **Group Session** (群组会话) - 群聊中激活
3. **Isolated Session** (隔离会话) - Docker沙箱运行

### 模型配置

```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-6",
    "thinking": "high",  // 思考级别
    "verbose": true
  }
}
```

### 模型降级策略

支持配置多个模型作为备用：
- 主要模型不可用时自动切换
- 支持 Anthropic、OpenAI 等多种提供商

---

## 🛡️ 安全模型

### 默认安全策略

- **DM配对**: 新发送者需要配对码验证
- **群组**: 需要@提及才激活
- **工具**: 默认仅允许安全工具

### 权限控制

```json
{
  "channels": {
    "telegram": {
      "dmPolicy": "pairing",  // pairing | open
      "allowFrom": ["*"]  // 白名单
    }
  }
}
```

### 沙箱模式

群组/频道会话可启用Docker沙箱：
```json
{
  "agents": {
    "defaults": {
      "sandbox": {
        "mode": "non-main"
      }
    }
  }
}
```

---

## 🎨 高级功能

### Voice Wake (语音唤醒)

支持 macOS/iOS/Android 的语音唤醒：
- 持续监听唤醒词
- 按键说话模式
- ElevenLabs 语音合成

### Talk Mode (对话模式)

- 实时语音对话
- 语音转文字 + 文字转语音
- 支持 iOS/Android

### Live Canvas (画布)

- Agent驱动的可视化工作区
- A2UI 界面协议
- 支持实时渲染和交互

### Browser Control (浏览器控制)

```json
{
  "browser": {
    "enabled": true,
    "color": "#FF4500"
  }
}
```

---

## 📱 平台支持

### macOS

- 菜单栏应用
- Voice Wake + PTT
- WebChat + 调试工具
- 远程网关控制

### iOS

- Canvas 界面
- 语音唤醒
- 摄像头/屏幕录制
- Bonjour 配对

### Android

- Canvas 界面
- 语音对话
- 摄像头/屏幕录制
- 可选 SMS 支持

### Linux

- 推荐作为远程Gateway
- 支持所有CLI工具
- 远程访问 via Tailscale/SSH

---

## 🔄 远程访问

### Tailscale

```json
{
  "gateway": {
    "tailscale": {
      "mode": "serve",  // off | serve | funnel
      "resetOnExit": true
    }
  }
}
```

- **serve**: 仅Tailnet内网访问
- **funnel**: 公网HTTPS访问（需密码认证）

### SSH 隧道

支持SSH隧道远程连接Gateway

---

## 🔌 工具与自动化

### 可用工具

- **browser**: 浏览器自动化
- **canvas**: 画布渲染
- **nodes**: 设备节点控制
- **cron**: 定时任务
- **sessions**: 多Agent会话管理
- **exec**: 命令执行
- **read/write/edit**: 文件操作

### Cron 任务

```json
{
  "cron": {
    "schedule": "0 8 * * *",
    "payload": {
      "kind": "systemEvent",
      "text": "每日问候"
    }
  }
}
```

### Webhooks

支持外部HTTP触发

---

## 🎯 Skills 系统

### 技能类型

1. **Bundled Skills** - 内置技能
2. **Managed Skills** - 托管技能（ClawHub）
3. **Workspace Skills** - 工作区自定义技能

### ClawHub

```bash
# 搜索技能
clawhub search bitcoin

# 安装技能
clawhub install bitcoin-trading

# 更新技能
clawhub update
```

### 自定义技能结构

```
~/.openclaw/workspace/skills/
├── my-skill/
│   ├── SKILL.md      # 技能定义
│   └── scripts/       # 脚本文件
```

---

## 💻 开发指南

### 从源码运行

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 安装依赖
pnpm install

# 构建
pnpm build

# 开发模式（自动重载）
pnpm gateway:watch
```

### 发布渠道

- **stable**: 稳定版 (npm latest)
- **beta**: 测试版
- **dev**: 开发版

```bash
# 切换渠道
openclaw update --channel stable|beta|dev
```

---

## 📊 监控与运维

### 健康检查

```bash
openclaw doctor
```

### 日志

```bash
# 查看日志
openclaw logs

# 实时日志
openclaw logs --follow
```

### 使用统计

- Token 使用量追踪
- 成本统计
- 会话分析

---

## 🚨 故障排除

### 常见问题

1. **无法连接Gateway**
   ```bash
   # 检查服务状态
   openclaw gateway status
   
   # 重启服务
   openclaw gateway restart
   ```

2. **Telegram 配对失败**
   - 检查 Bot Token 格式
   - 确认 webhook 配置
   - 验证频道权限

3. **浏览器工具不可用**
   - 检查 Chrome/Chromium 安装
   - 验证浏览器路径配置

### 调试模式

```bash
# 启用详细日志
openclaw gateway --verbose

# 查看所有日志
openclaw logs --level debug
```

---

## 📚 资源链接

### 官方资源

- **官网**: https://openclaw.ai
- **文档**: https://docs.openclaw.ai
- **GitHub**: https://github.com/openclaw/openclaw
- **Discord**: https://discord.gg/clawd
- **ClawHub**: https://clawhub.com

### 社区

- Discord 社区: "Friends of the Crustacean"
- Twitter: @openclaw
- X账号: @openclaw

### 深度资源

- DeepWiki: https://deepwiki.com/openclaw/openclaw
- 架构文档: https://docs.openclaw.ai/concepts/architecture

---

## 💡 使用技巧

### 高效使用

1. **配置多个模型**: 设置降级策略保证可用性
2. **启用语音**: 语音唤醒提高交互效率
3. **使用Skills**: 安装相关技能增强能力
4. **配置自动化**: Cron任务实现定时提醒

### 安全建议

1. **限制DM访问**: 默认使用配对模式
2. **沙箱隔离**: 群组会话启用Docker
3. **定期更新**: 保持最新版本
4. **备份配置**: 定期备份配置文件

---

## 🔧 配置文件参考

### 完整配置结构

```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-6",
    "thinking": "high"
  },
  "gateway": {
    "bind": "127.0.0.1",
    "port": 18789
  },
  "channels": {
    "telegram": { ... },
    "whatsapp": { ... },
    "discord": { ... }
  },
  "browser": {
    "enabled": true
  },
  "nodes": {
    "voicewake": {
      "enabled": true
    }
  }
}
```

---

*文档生成时间: 2026-02-07*
