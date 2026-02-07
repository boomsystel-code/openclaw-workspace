# OpenClaw Skills系统详解

*最后更新: 2026-02-07*

---

## 🎯 Skills系统概述

OpenClaw使用**AgentSkills**兼容的技能文件夹来教Agent如何使用工具。每个技能是一个目录，包含一个`SKILL.md`文件（包含YAML前言和说明）。

### 核心特点

- **技能驱动**: Agent通过Skills学习使用工具
- **模块化**: 每个技能专注于特定功能
- **可扩展**: 支持自定义创建和安装
- **安全可控**: 加载时过滤和权限控制

---

## 📁 Skills位置与优先级

### 三个存放位置

1. **Bundled Skills** (捆绑技能)
   - 随OpenClaw安装附带
   - 路径: npm包或OpenClaw.app内

2. **Managed/Local Skills** (托管/本地技能)
   - 路径: `~/.openclaw/skills`
   - 对所有Agent可见

3. **Workspace Skills** (工作区技能)
   - 路径: `<workspace>/skills`
   - 用户拥有，仅对当前Agent可见

### 优先级（高到低）

```
<workspace>/skills (最高)
    ↓
~/.openclaw/skills
    ↓
捆绑技能 (最低)
```

### 额外配置

通过`skills.load.extraDirs`添加额外技能目录（最低优先级）：
```json
{
  "skills": {
    "load": {
      "extraDirs": [
        "~/Projects/agent-scripts/skills"
      ]
    }
  }
}
```

---

## 📄 SKILL.md格式规范

### 必须包含的字段

```yaml
---
name: skill-name
description: 技能简短描述
---
```

### 可选字段

```yaml
---
name: nano-banana-pro
description: 通过Gemini 3 Pro生成或编辑图像
metadata:
  {
    "openclaw": {
      "requires": { "bins": ["uv"], "env": ["GEMINI_API_KEY"] },
      "primaryEnv": "GEMINI_API_KEY",
      "emoji": "🖼️",
      "homepage": "https://example.com",
      "os": ["darwin", "linux"]
    },
    "user-invocable": true,
    "disable-model-invocation": false,
    "command-dispatch": "tool",
    "command-tool": "tool-name"
  }
---
```

### 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | 必填 | 技能名称（唯一标识符） |
| `description` | 必填 | 功能描述 |
| `metadata` | 可选 | 元数据配置 |
| `user-invocable` | 可选 | 是否可通过斜杠命令调用（默认true） |
| `disable-model-invocation` | 可选 | 是否从模型提示中排除（默认false） |
| `command-dispatch` | 可选 | 设为"tool"可绕过模型直接调用工具 |
| `command-tool` | 可选 | 指定调用的工具名称 |

### Metadata字段详解

```yaml
metadata:
  {
    "openclaw": {
      "always": true,                    # 总是加载，跳过其他过滤
      "emoji": "🎯",                     # macOS UI显示的emoji
      "homepage": "https://...",         # 技能官网URL
      "os": ["darwin", "linux"],         # 支持的操作系统
      "requires": {
        "bins": ["uv", "python"],         # 必须存在的二进制命令
        "anyBins": ["python3", "python"], # 至少一个必须存在
        "env": ["API_KEY"],              # 必须存在的环境变量
        "config": ["browser.enabled"]    # 必须为真的配置项
      },
      "primaryEnv": "API_KEY",           # 主要API密钥环境变量
      "install": [
        {
          "id": "brew",
          "kind": "brew",
          "formula": "some-formula",
          "bins": ["some-bin"],
          "label": "安装描述"
        }
      ]
    }
  }
```

---

## 🔧 安装器规格 (Installers)

### Brew安装

```yaml
metadata:
  {
    "openclaw": {
      "install": [
        {
          "id": "brew",
          "kind": "brew",
          "formula": "gemini-cli",
          "bins": ["gemini"],
          "label": "Install Gemini CLI (brew)",
          "os": ["darwin"]
        }
      ]
    }
  }
```

### Node安装

```yaml
{
  "id": "node",
  "kind": "node",
  "package": "some-cli-tool",
  "bins": ["some-tool"],
  "label": "Install via npm"
}
```

### Go安装

```yaml
{
  "id": "go",
  "kind": "go",
  "install": "github.com/user/repo@latest",
  "bins": ["repo-tool"],
  "label": "Install Go tool"
}
```

### 下载安装

```yaml
{
  "id": "download",
  "kind": "download",
  "url": "https://example.com/tool.tar.gz",
  "archive": "tar.gz",
  "extract": "auto",
  "stripComponents": 1,
  "targetDir": "~/.openclaw/tools/",
  "bins": ["tool-name"],
  "label": "Download and install"
}
```

---

## ⚙️ Skills配置

### 完整配置示例 (~/.openclaw/openclaw.json)

```json
{
  "skills": {
    "allowBundled": ["gemini", "peekaboo"],
    "load": {
      "extraDirs": [
        "~/Projects/agent-scripts/skills"
      ],
      "watch": true,
      "watchDebounceMs": 250
    },
    "install": {
      "preferBrew": true,
      "nodeManager": "npm"
    },
    "entries": {
      "nano-banana-pro": {
        "enabled": true,
        "apiKey": "GEMINI_KEY_HERE",
        "env": {
          "GEMINI_API_KEY": "your-api-key-here"
        },
        "config": {
          "endpoint": "https://api.example.com",
          "model": "nano-pro"
        }
      },
      "peekaboo": {
        "enabled": true
      },
      "sag": {
        "enabled": false
      }
    }
  }
}
```

### 字段说明

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `allowBundled` | 数组 | 仅允许的捆绑技能列表 |
| `load.extraDirs` | 数组 | 额外扫描的技能目录 |
| `load.watch` | 布尔 | 是否监控技能文件变化（默认true） |
| `load.watchDebounceMs` | 数值 | 监控防抖延迟（默认250ms） |
| `install.preferBrew` | 布尔 | 优先使用brew安装（默认true） |
| `install.nodeManager` | 字符串 | Node包管理器（npm/pnpm/yarn/bun） |
| `entries.<skill>` | 对象 | 单个技能配置 |

### 单技能配置

```json
{
  "entries": {
    "skill-name": {
      "enabled": false,           # 禁用技能
      "apiKey": "KEY",           # API密钥（便捷方式）
      "env": {
        "VAR_NAME": "value"      # 环境变量
      },
      "config": {
        "customKey": "value"     # 自定义配置
      }
    }
  }
}
```

---

## 🔒 环境变量注入

### 注入时机

当Agent运行时，OpenClaw会：
1. 读取技能元数据
2. 应用`skills.entries.<skill>.env`和`apiKey`到`process.env`
3. 构建系统提示（含可用技能）
4. 运行结束后恢复原始环境

### 重要说明

- 变量仅在Agent运行期间注入
- 不会影响全局Shell环境
- 如果变量已存在，则不会覆盖

### 沙箱环境变量

当会话在沙箱中运行时，技能进程在Docker内执行，沙箱不继承主机`process.env`。

解决方案：
```json
{
  "agents": {
    "defaults": {
      "sandbox": {
        "docker": {
          "env": {
            "API_KEY": "your-key"
          }
        }
      }
    }
  }
}
```

---

## 🔍 技能过滤（Gate）

### 基于条件的加载

OpenClaw在加载时根据元数据过滤技能：

```yaml
---
name: advanced-skill
description: 需要特定环境的技能
metadata:
  {
    "openclaw": {
      "requires": {
        "bins": ["uv"],
        "env": ["ANTHROPIC_API_KEY"],
        "config": ["browser.enabled"]
      }
    }
  }
---
```

### 过滤条件

| 条件 | 说明 |
|------|------|
| `bins` | PATH中必须存在的命令 |
| `anyBins` | PATH中至少一个必须存在 |
| `env` | 必须存在的环境变量 |
| `config` | openclaw.json中必须为真的配置项 |
| `os` | 仅在特定操作系统上加载 |

### 始终加载

```yaml
metadata:
  {
    "openclaw": {
      "always": true
    }
  }
```

---

## 📦 ClawHub - 技能市场

### ClawHub简介

ClawHub是OpenClaw的公共技能注册表。

- **官网**: https://clawhub.com
- **功能**: 浏览、安装、更新、备份技能

### 常用命令

```bash
# 列出可用技能
clawhub search

# 搜索特定技能
clawhub search bitcoin

# 安装技能到工作区
clawhub install

# 更新所有已安装技能
clawhub update --all

# 同步（扫描并发布更新）
clawhub sync --all
```

### 安装位置

默认安装到当前工作目录下的`./skills`（或回退到配置的OpenClaw工作区）。

---

## 🛡️ 安全注意事项

### 核心原则

1. **信任第三方技能要谨慎**
   - 阅读技能代码后再启用
   - 避免运行不信任来源的技能

2. **沙箱隔离**
   - 对不信任输入和危险工具使用沙箱运行
   - 参考: [Sandboxing](/gateway/sandboxing)

3. **密钥保护**
   - 使用`skills.entries.<skill>.env`和`apiKey`注入密钥
   - 避免在提示和日志中暴露密钥

4. **完整威胁模型**
   - 参考: [Security](/gateway/security)

---

## 🎯 用户可调用技能

### 斜杠命令

当`user-invocable: true`时，技能可通过斜杠命令调用：

```
/skill-name [参数]
```

### 示例

```yaml
---
name: summarize
description: 总结文档或URL内容
user-invocable: true
---
```

调用：`/summarize https://example.com/article`

### 工具直接调度

当设置`command-dispatch: tool`时，斜杠命令绕过模型直接调用工具：

```yaml
---
name: timer
description: 设置定时器和提醒
command-dispatch: tool
command-tool: timer
---
```

---

## 🔄 技能热重载

### 监控配置

```json
{
  "skills": {
    "load": {
      "watch": true,
      "watchDebounceMs": 250
    }
  }
}
```

### 工作机制

1. OpenClaw默认监控技能文件夹
2. 当SKILL.md变化时，自动刷新技能快照
3. 变化在下一次Agent对话时生效
4. 无需重启Gateway

### 性能影响

技能列表注入系统提示的成本：
- 基础开销（至少1个技能时）: 195字符
- 每个技能: 97字符 + 名称+描述+位置长度

公式：
```
总字符数 = 195 + Σ(97 + len(name) + len(description) + len(location))
```

---

## 🖥️ 远程macOS节点

### Linux Gateway + macOS节点

当Gateway运行在Linux上但连接了macOS节点时：
- 可使用macOS专属技能（需对应二进制在macOS节点上）
- 通过`nodes`工具执行这些技能

### 条件

- macOS节点需启用`system.run`权限
- 节点需报告其命令支持
- 二进制需通过`system.run`探测

---

## 📚 创建自定义技能

### 技能模板

```yaml
---
name: my-custom-skill
description: 我的自定义技能描述
metadata:
  {
    "openclaw": {
      "emoji": "✨",
      "requires": {
        "bins": ["some-cli-tool"],
        "env": ["MY_API_KEY"]
      },
      "primaryEnv": "MY_API_KEY",
      "install": [
        {
          "id": "brew",
          "kind": "brew",
          "formula": "some-cli-tool",
          "bins": ["some-cli-tool"],
          "label": "Install Some CLI Tool"
        }
      ]
    }
  }
---
```

### 技能说明内容

```
## 使用方法

### 基本用法

使用此技能执行XXX功能：

1. [参数说明]
2. [示例]
3. [注意事项]

### 示例

```
[命令示例]
```

## 工具要求

- 必须安装 `some-cli-tool`
- 需要设置 `MY_API_KEY` 环境变量

## 限制

- [功能限制]
- [平台要求]
```

---

## 🔧 内置技能示例

### 常用内置技能

| 技能名 | 功能 | 要求 |
|--------|------|------|
| `gemini` | Gemini CLI编程辅助 | `gemini`二进制 |
| `peekaboo` | macOS UI自动化 | macOS |
| `sag` | 语音生成（TTS） | 音频设备 |
| `summarize` | 文档总结 | summarize CLI |

### 技能位置

```
~/.openclaw/skills/           # 本地/托管技能
<workspace>/skills/           # 工作区技能
openclaw/skills/             # 捆绑技能（源码）
```

---

## 💡 最佳实践

### 1. 技能命名

- 使用kebab-case（小写连字符）
- 保持简短描述性
- 避免与内置技能冲突

### 2. 环境变量

- 使用`primaryEnv`指定主密钥
- 避免硬编码密钥
- 提供默认值说明

### 3. 平台兼容性

- 明确指定支持的操作系统
- 提供跨平台解决方案
- 使用`os`元数据过滤

### 4. 文档质量

- 提供清晰的使用示例
- 说明输入输出格式
- 列出已知限制

### 5. 安全考虑

- 最小权限原则
- 沙箱中运行危险操作
- 审计第三方依赖

---

## 🔗 相关资源

### 官方资源

- **Skills文档**: https://docs.openclaw.ai/tools/skills
- **配置参考**: https://docs.openclaw.ai/tools/skills-config
- **ClawHub**: https://clawhub.com
- **GitHub**: https://github.com/openclaw/openclaw

### 相关概念

- [Agent系统](/concepts/agent)
- [会话管理](/concepts/session)
- [沙箱隔离](/gateway/sandboxing)
- [安全模型](/gateway/security)

---

*文档生成时间: 2026-02-07*
