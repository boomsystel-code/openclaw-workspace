# OpenClaw能力提升指南

*最后更新: 2026-02-07*

---

## 🎯 提升路径总览

OpenClaw的能力可以通过以下方式提升：

```
┌─────────────────────────────────────────┐
│           OpenClaw 能力提升体系          │
├─────────────────────────────────────────┤
│  Level 1: 内置技能优化                   │
│  Level 2: ClawHub安装技能               │
│  Level 3: 自定义技能开发                 │
│  Level 4: 技能组合与自动化               │
│  Level 5: 多Agent协作                    │
└─────────────────────────────────────────┘
```

---

## 📦 Level 1: 内置技能优化

### 当前已安装的Skills位置

```bash
# 查看工作区技能
ls -la ~/.openclaw/workspace/skills/

# 查看托管技能
ls -la ~/.openclaw/skills/

# 查看捆绑技能（源码）
ls -la openclaw/skills/
```

### 推荐启用的内置技能

| 技能 | 功能 | 启用方式 |
|------|------|----------|
| `summarize` | 文档/URL总结 | 默认启用 |
| `browser` | 浏览器控制 | 配置启用 |
| `canvas` | 画布渲染 | macOS默认 |
| `nodes` | 设备节点控制 | 默认启用 |

### 配置示例

```json
{
  "skills": {
    "entries": {
      "browser": {
        "enabled": true,
        "config": {
          "color": "#FF4500"
        }
      },
      "sag": {
        "enabled": true
      }
    }
  }
}
```

---

## 📦 Level 2: ClawHub技能安装

### 常用推荐技能

#### 💰 金融与投资
- **BTC交易技能** - 加密货币交易分析
- **Finance** - 股票/ETF追踪
- **Crypto** - 加密货币监控

#### 📝 生产力工具
- **Things Mac** - 任务管理
- **Apple Notes** - 笔记同步
- **Calendar** - 日历管理

#### 🎨 创意工具
- **Image Generation** - AI生图
- **Spotify** - 音乐控制
- **Apple Music** - 音乐管理

#### 🔧 开发工具
- **GitHub** - GitHub操作
- **Cursor Agent** - AI编程
- **Claude Code** - 代码生成

### 安装命令

```bash
# 浏览可用技能
clawhub search

# 安装特定技能
clawhub install bitcoin-trading
clawhub install things-mac
clawhub install github

# 更新所有技能
clawhub update --all

# 同步技能
clawhub sync --all
```

### 安装后配置

```json
{
  "skills": {
    "entries": {
      "bitcoin-trading": {
        "enabled": true,
        "env": {
          "API_KEY": "your-api-key"
        }
      },
      "github": {
        "enabled": true,
        "config": {
          "token": "your-github-token"
        }
      }
    }
  }
}
```

---

## 🛠️ Level 3: 自定义技能开发

### 开发流程

```
1. 规划技能功能
       ↓
2. 创建技能目录结构
       ↓
3. 编写SKILL.md
       ↓
4. 开发配套脚本/工具
       ↓
5. 测试与调试
       ↓
6. 发布到ClawHub（可选）
```

### 技能目录结构

```
my-skill/
├── SKILL.md          # 技能定义文件（必需）
├── scripts/          # 脚本目录
│   ├── main.py       # 主脚本
│   └── utils.py      # 工具函数
├── README.md         # 详细文档（可选）
└── requirements.txt   # 依赖列表（可选）
```

### 最小化技能示例

#### SKILL.md

```yaml
---
name: hello-world
description: 向世界问好
metadata:
  {
    "openclaw": {
      "emoji": "👋",
      "user-invocable": true
    }
  }
---

## 功能

打印Hello World消息。

## 使用方法

```
/hello-world [名字]
```

### 示例

```
/hello-world           # 输出: Hello, World!
/hello-world Alice     # 输出: Hello, Alice!
```

## 实现

使用Python内置print函数。
```

#### scripts/hello.py

```python
#!/usr/bin/env python3
"""Hello World Skill for OpenClaw"""

import sys

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "World"
    print(f"Hello, {name}!")

if __name__ == "__main__":
    main()
```

### 进阶技能示例：API调用技能

```yaml
---
name: crypto-price
description: 获取加密货币实时价格
metadata:
  {
    "openclaw": {
      "emoji": "₿",
      "requires": {
        "bins": ["curl"]
      },
      "user-invocable": true
    }
  }
---

## 功能

通过CoinGecko API获取加密货币实时价格。

## 使用方法

```
/crypto-price bitcoin    # BTC价格
/crypto-price ethereum   # ETH价格
```

## 注意事项

- 使用CoinGecko免费API
- 有速率限制
```

#### scripts/crypto_price.py

```python
#!/usr/bin/env python3
"""Crypto Price Skill"""

import sys
import urllib.request
import json

def get_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            return data.get(coin_id, {}).get('usd', 'N/A')
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    coin = sys.argv[1].lower() if len(sys.argv) > 1 else "bitcoin"
    price = get_price(coin)
    print(f"{coin.capitalize()}: ${price}")
```

### 技能发布

#### 发布到ClawHub

```bash
# 1. 准备技能
cd my-skill

# 2. 确保SKILL.md完整
cat SKILL.md

# 3. 提交到GitHub
git init
git add .
git commit -m "Add my custom skill"
gh repo create my-openclaw-skill --public --source=. --push

# 4. 发布到ClawHub（需要CLI）
clawhub publish
```

---

## 🔄 Level 4: 技能组合自动化

### Cron定时任务

```json
{
  "cron": {
    "daily-briefing": {
      "schedule": "0 8 * * *",
      "payload": {
        "kind": "systemEvent",
        "text": "获取今日天气和日程安排"
      }
    },
    "crypto-alert": {
      "schedule": "*/30 * * * *",
      "payload": {
        "kind": "systemEvent",
        "text": "检查BTC价格并发送提醒"
      }
    }
  }
}
```

### 技能组合示例

#### 早晨例程

```bash
#!/bin/bash
# morning-routine.sh

# 获取天气
clawhub run weather --location="Beijing"

# 检查日程
clawhub run calendar --today

# 获取新闻摘要
clawhub run summarize --source="https://news.ycombinator.com"

# 播放音乐
clawhub run spotify --playlist="Morning Vibes"
```

#### BTC监控系统

```yaml
---
name: btc-monitor
description: BTC价格监控与提醒
metadata:
  {
    "openclaw": {
      "requires": {
        "bins": ["curl"]
      },
      "user-invocable": true,
      "disable-model-invocation": false
    }
  }
---

## 功能

监控BTC价格，在达到阈值时发送提醒。

## 使用方法

```
/btc-monitor set-alert 70000    # 设置70000美元提醒
/btc-monitor status            # 查看当前状态
/btc-monitor check             # 立即检查价格
```

## 触发条件

- 价格上涨/下跌5%
- 突破关键阻力位
- 每日定时检查
```

---

## 🤝 Level 5: 多Agent协作

### Agent配置

```json
{
  "agents": {
    "list": [
      {
        "id": "researcher",
        "name": "Research Agent",
        "model": "anthropic/claude-opus-4-6",
        "skills": ["browser", "summarize", "web-search"],
        "system": "你是一个研究助手，专门负责信息收集和分析。"
      },
      {
        "id": "coder",
        "name": "Coding Agent",
        "model": "anthropic/claude-sonnet-4-6",
        "skills": ["github", "cursor-agent", "filesystem"],
        "system": "你是一个编程助手，专门负责代码编写和调试。"
      },
      {
        "id": "trader",
        "name": "Trading Agent",
        "model": "anthropic/claude-haiku-4-6",
        "skills": ["crypto-price", "finance", "news-feed"],
        "system": "你是一个交易助手，专门负责加密货币分析和交易信号。"
      }
    ],
    "defaults": {
      "model": "anthropic/claude-opus-4-6",
      "sandbox": {
        "mode": "non-main"
      }
    }
  }
}
```

### Agent间通信

```bash
# 发送消息给其他Agent
sessions_send --sessionKey=researcher --message="分析最新的BTC技术分析报告"

# 获取Agent历史
sessions_history --sessionKey=coder

# 列出所有Agent
sessions_list
```

### 协作工作流

```
用户需求
    ↓
┌────────────────────────────────────┐
│  协调Agent (Coordinator)            │
│  - 分析需求                          │
│  - 分配任务                          │
└────────────────────────────────────┘
    ↓
┌────────────┬────────────┬────────────┐
│ Researcher │   Coder    │  Trader   │
│ 收集信息    │  编写代码   │  分析交易  │
└────────────┴────────────┴────────────┘
    ↓
┌────────────────────────────────────┐
│  整合结果                          │
│  - 汇总分析                        │
│  - 生成报告                        │
└────────────────────────────────────┘
    ↓
输出结果
```

---

## 📊 技能评估矩阵

### 评估维度

| 维度 | 说明 | 权重 |
|------|------|------|
| **实用性** | 解决实际问题的能力 | 30% |
| **易用性** | 学习成本和使用门槛 | 20% |
| **稳定性** | 运行时可靠性 | 20% |
| **安全性** | 安全风险评估 | 15% |
| **维护性** | 更新频率和社区支持 | 15% |

### 推荐技能列表

#### ⭐⭐⭐⭐⭐ 必装技能

| 技能 | 评分 | 用途 |
|------|------|------|
| `summarize` | 5/5 | 文档摘要 |
| `browser` | 5/5 | 网页自动化 |
| `github` | 5/5 | 代码管理 |
| `things-mac` | 5/5 | 任务管理 |

#### ⭐⭐⭐⭐ 推荐技能

| 技能 | 评分 | 用途 |
|------|------|------|
| `crypto-price` | 4/5 | 加密货币 |
| `weather` | 4/5 | 天气查询 |
| `spotify` | 4/5 | 音乐控制 |
| `calendar` | 4/5 | 日程管理 |

#### ⭐⭐⭐ 探索技能

| 技能 | 评分 | 用途 |
|------|------|------|
| `notion` | 3/5 | 笔记同步 |
| `obsidian` | 3/5 | Markdown笔记 |
| `apple-notes` | 3/5 | Apple笔记 |
| `lastfm` | 3/5 | 音乐追踪 |

---

## 🛡️ 安全最佳实践

### 1. 技能来源验证

```bash
# 检查技能GitHub仓库
gh repo view clawhub/bitcoin-trading

# 查看Stars和Forks
gh repo view clawhub/bitcoin-trading --json=name,stargazerCount,forkCount

# 检查最近更新
gh api repos/clawhub/bitcoin-trading/commits
```

### 2. 代码审计

```bash
# 克隆技能仓库进行审计
git clone https://github.com/clawhub/my-skill.git
cd my-skill

# 检查脚本内容
cat scripts/main.py

# 检查权限
ls -la scripts/
```

### 3. 沙箱测试

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

## 📈 性能优化

### 1. 技能加载优化

```json
{
  "skills": {
    "load": {
      "watch": true,
      "watchDebounceMs": 100
    }
  }
}
```

### 2. 禁用不需要的技能

```json
{
  "skills": {
    "entries": {
      "unused-skill": {
        "enabled": false
      },
      "another-unused": {
        "enabled": false
      }
    }
  }
}
```

### 3. 技能分组

```bash
# 为不同场景创建不同工作区
~/projects/openclaw-workflows/
├── productivity/skills/
├── development/skills/
└── trading/skills/
```

---

## 🔧 故障排除

### 技能无法加载

```bash
# 1. 检查日志
openclaw logs --level=debug

# 2. 验证技能配置
cat ~/.openclaw/skills/my-skill/SKILL.md

# 3. 检查依赖
which required-binary

# 4. 测试环境变量
echo $MY_API_KEY
```

### 权限问题

```bash
# 重新安装技能
clawhub uninstall my-skill
clawhub install my-skill

# 重启Gateway
openclaw gateway restart
```

### 性能问题

```bash
# 检查技能数量
ls ~/.openclaw/skills/*/SKILL.md | wc -l

# 禁用不需要的技能
clawhub disable unused-skill

# 清理缓存
openclaw doctor --fix
```

---

## 📚 学习资源

### 官方文档

- **Skills系统**: https://docs.openclaw.ai/tools/skills
- **技能配置**: https://docs.openclaw.ai/tools/skills-config
- **ClawHub**: https://clawhub.com
- **GitHub仓库**: https://github.com/openclaw/openclaw

### 社区资源

- **Discord**: https://discord.gg/clawd
- **示例技能**: https://github.com/openclaw/openclaw/tree/main/skills

### 推荐学习路径

```
1. 熟悉内置技能
       ↓
2. 安装ClawHub热门技能
       ↓
3. 修改现有技能
       ↓
4. 创建简单自定义技能
       ↓
5. 开发复杂多技能系统
       ↓
6. 贡献到ClawHub
```

---

## 🎯 下一步行动

### 立即行动 (今天)

- [ ] 浏览ClawHub热门技能
- [ ] 安装3-5个感兴趣的技能
- [ ] 配置API密钥和环境变量

### 本周计划

- [ ] 创建一个简单自定义技能
- [ ] 配置定时任务自动化
- [ ] 设置Agent协作工作流

### 本月目标

- [ ] 建立个人技能库
- [ ] 优化现有工作流
- [ ] 贡献1个技能到ClawHub

---

*文档生成时间: 2026-02-07*
