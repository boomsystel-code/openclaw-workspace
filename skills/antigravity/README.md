# Antigravity AI Web Builder Skill

**Version**: 1.0.0  
**Created**: 2026-02-07  
**Author**: OpenClaw Integration

## Overview

This skill provides AI-powered website and web application generation capabilities, integrating Antigravity-style AI coding tools with OpenClaw's orchestration system.

## What's Included

### SKILL.md
Main skill file with:
- Quick start examples
- Supported frameworks (React, Vue, Next.js, Static)
- Project type templates
- Integration guides
- Troubleshooting

### scripts/antigravity_cli.py
Command-line interface for:
- Project generation from prompts
- Framework scaffolding
- Template listing
- Deployment commands

### references/
Documentation:
- `frameworks.md` - Framework comparison and selection guide
- `workflows.md` - Complete automation workflows

## Quick Usage

### Generate a Project
```bash
python scripts/antigravity_cli.py generate \
    --prompt "Create a React todo app" \
    --framework react \
    --style tailwind \
    --output ./my-todo-app
```

### List Templates
```bash
python scripts/antigravity_cli.py list
```

### Deploy
```bash
python scripts/antigravity_cli.py deploy --output ./my-project
```

## Integration with OpenClaw

### Skills That Work Well Together
- `cursor-agent` - Fine-tune generated code
- `github` - Version control
- `vercel` - Deployment
- `netlify` - Alternative deployment
- `supabase` - Database integration
- `postgres` - Database integration

### Typical Workflow
```
1. User Request → OpenClaw
2. OpenClaw parses requirements
3. Antigravity generates code
4. Cursor refines (optional)
5. GitHub creates repo
6. Vercel deploys
7. OpenClaw confirms completion
```

## Requirements

- Python 3.8+
- Node.js (for generated projects)
- Git (for version control)
- Vercel CLI (optional, for deployments)

## Installation

This skill is auto-installed via ClawHub:

```bash
clawhub install antigravity
```

Or manually place in `skills/` directory:

```bash
git clone https://github.com/your-org/antigravity-skill.git \
    ~/.openclaw/workspace/skills/antigravity
```

## Configuration

### Environment Variables (Optional)
```bash
ANITGRAVITY_API_KEY=your_api_key
MISTRAL_API_KEY=your_mistral_key
GEMINI_API_KEY=your_gemini_key
```

### OpenClaw Config
No special configuration required. Skill triggers on keywords:
- "build", "create", "generate"
- "website", "webapp", "landing page"
- "React", "Vue", "Next.js"

## Examples

### Simple Landing Page
```
"Build a landing page for my consulting business"
→ Generates static HTML + Tailwind
```

### Todo Application
```
"Create a React todo app with dark mode"
→ Generates React + Vite + Tailwind
```

### Dashboard
```
"Build an analytics dashboard with charts"
→ Generates React + Recharts + Tailwind
```

### Full-Stack App
```
"Create a Next.js blog with Markdown support"
→ Generates Next.js 14 + MDX + Tailwind
```

## Troubleshooting

### Generated Code Doesn't Work
1. Check Node.js version (14+ required)
2. Run `npm install` first
3. Check browser console for errors

### Style Issues
1. Verify Tailwind is installed
2. Check class names are valid
3. Ensure responsive prefixes are correct

### Build Failures
1. Check TypeScript errors
2. Verify all imports exist
3. Ensure environment variables are set

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## License

MIT License - See LICENSE file for details

## Changelog

### v1.0.0 (2026-02-07)
- Initial release
- Basic project generation
- Framework scaffolding
- Workflow automation
- Reference documentation

---

**Questions?** Check the `references/` directory or create an issue.
