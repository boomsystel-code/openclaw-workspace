## 🆕 新增技能 (2026-02-07)

| 技能名称 | 功能 | 状态 |
|---------|------|------|
| **antigravity** | AI网站/网页应用生成器 | ✅ 已整合 |

#### Antigravity 技能详情

**功能**: AI驱动的网站和Web应用生成  
**支持框架**: React, Vue, Next.js, Static HTML  
**集成工具**: Cline, Aider, Windsurf, Vercel, Netlify

**核心文件**:
```
skills/antigravity/
├── SKILL.md                # 主技能文件 (6.1KB)
├── README.md               # 使用说明 (3.6KB)
├── scripts/
│   └── antigravity_cli.py # CLI工具 (12.6KB)
└── references/
    ├── frameworks.md       # 框架对比指南 (5.3KB)
    └── workflows.md        # 自动化工作流 (5.8KB)
```

**使用示例**:
```bash
# 生成 React 项目
python scripts/antigravity_cli.py generate \
    --prompt "创建待办事项应用" \
    --framework react \
    --style tailwind

# 列出可用模板
python scripts/antigravity_cli.py list
```

**集成优势**:
- OpenClaw Agent 负责任务编排
- Antigravity 负责代码生成
- Cursor/Cline 负责精细调整
- Vercel/Netlify 负责部署
