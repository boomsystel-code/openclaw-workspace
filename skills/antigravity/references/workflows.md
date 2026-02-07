# OpenClaw + Antigravity Workflows

## Workflow 1: Rapid Prototype → Production

### Step 1: Generate Prototype
```
User: "Build a SaaS landing page"
Antigravity: Generates React + Tailwind landing page
```

### Step 2: Review and Refine
```
OpenClaw: Open in Cursor (cursor-agent skill)
Cursor: Fine-tune design, add animations
```

### Step 3: Git Integration
```
github skill: Create repository
          → Commit changes
          → Push to remote
```

### Step 4: Deploy
```
vercel skill: Deploy with auto-detect
         → Configure custom domain
         → Set up environment variables
```

### Step 5: Monitor
```
github skill: Check deployment status
        → View analytics
        → Set up webhooks
```

---

## Workflow 2: Multi-Page Application

### Requirements
```
User: "Create an e-commerce site with product catalog, cart, and checkout"
```

### Breakdown
1. **Generation**: Generate Next.js app with product pages
2. **Backend**: Create API routes for cart/checkout
3. **Database**: Use supabase skill for product data
4. **Payments**: Integrate Stripe (manual or skill)
5. **Auth**: Add NextAuth.js (manual or skill)

### Commands Sequence
```bash
# 1. Generate base
antigravity generate --prompt "E-commerce site..." --framework nextjs

# 2. Open in Cursor for refinement
cursor-agent edit --path ./generated --prompt "Add cart functionality"

# 3. Create GitHub repo
github create-repo e-commerce-site

# 4. Deploy
vercel deploy
```

---

## Workflow 3: Component Library

### Create Design System
```
Request: "Create a component library with buttons, forms, cards, modals"
Framework: React + TypeScript + Tailwind
```

### Individual Components
1. Generate base library structure
2. Create components one by one:
   - `Button.tsx`
   - `Input.tsx`
   - `Card.tsx`
   - `Modal.tsx`
3. Add Storybook for documentation
4. Publish to npm (optional)

### Usage Example
```tsx
import { Button, Card, Modal } from '@my-org/ui-library';

function App() {
  return (
    <Card>
      <h1>My App</h1>
      <Button variant="primary">Click Me</Button>
    </Card>
  );
}
```

---

## Workflow 4: Blog + CMS

### Option A: Static Site Generator
```
Framework: Next.js + Markdown files
Use case: Developer blogs, documentation
```

### Option B: Headless CMS
```
Framework: Next.js + Contentful/Sanity
Use case: Marketing blogs, news sites
```

### Implementation
```bash
# 1. Generate Next.js blog template
antigravity generate --prompt "Next.js blog with markdown support" --framework nextjs

# 2. Add CMS integration (supabase skill)
supabase create-table posts

# 3. Connect to frontend
# Create API routes for CRUD operations

# 4. Deploy
vercel deploy
```

---

## Workflow 5: Dashboard + Charts

### Generate Dashboard
```
Request: "Admin dashboard with user table, analytics charts, and settings"
Framework: React + Recharts/Victory + Tailwind
```

### Add Interactivity
1. State management: Zustand or Redux Toolkit
2. Data fetching: React Query (TanStack Query)
3. Charts: Recharts, Victory, or Chart.js
4. Tables: TanStack Table (React Table)

### Sample Structure
```
src/
├── components/
│   ├── charts/
│   │   ├── RevenueChart.tsx
│   │   └── UserGrowthChart.tsx
│   ├── tables/
│   │   └── UsersTable.tsx
│   └── Layout.tsx
├── hooks/
│   └── useDashboardData.ts
├── pages/
│   ├── dashboard.tsx
│   └── settings.tsx
└── types/
    └── index.ts
```

---

## Workflow 6: Mobile-Responsive App

### Generate Mobile-First
```
Request: "Mobile-first React app with responsive navbar and hamburger menu"
Framework: React + Tailwind (mobile-first classes)
```

### Key Considerations
1. Use Tailwind's responsive prefixes (`md:`, `lg:`)
2. Test on multiple screen sizes
3. Consider PWA capabilities
4. Add touch-friendly interactions

### Example
```tsx
<nav className="flex justify-between items-center p-4">
  <div className="hidden md:block">
    {/* Desktop menu */}
  </div>
  <button className="md:hidden">
    {/* Hamburger icon */}
  </button>
</nav>
```

---

## Best Practices Checklist

### Before Generation
- [ ] Define clear requirements
- [ ] Choose framework (React/Next.js/Vue)
- [ ] Decide on styling (Tailwind/CSS modules)
- [ ] Plan data fetching strategy
- [ ] Consider SEO requirements

### After Generation
- [ ] Review generated code
- [ ] Add TypeScript types
- [ ] Set up linting and formatting
- [ ] Configure CI/CD pipeline
- [ ] Add tests (unit + integration)
- [ ] Set up error boundaries
- [ ] Implement loading states

### Before Deployment
- [ ] Optimize images
- [ ] Minify bundle
- [ ] Set up environment variables
- [ ] Configure CORS policies
- [ ] Set up monitoring (Sentry)
- [ ] Test on staging environment

---

## Error Handling Patterns

### Generation Failed
```bash
# Try simpler prompt
antigravity generate --prompt "Simple React app" --framework react

# Or use template
antigravity scaffold --template react-ts
```

### Build Failed
```bash
# Check dependencies
npm install

# Check for TypeScript errors
npm run build

# Fix manually or regenerate
```

### Deployment Failed
```bash
# Check platform-specific issues
vercel logs

# Verify environment variables
vercel env list

# Deploy with different settings
vercel --prod --force
```

---

## Automation Scripts

### generate-and-deploy.sh
```bash
#!/bin/bash

# Generate project
echo "🎨 Generating project..."
python scripts/antigravity_cli.py generate \
    --prompt "$1" \
    --framework "$2" \
    --style tailwind \
    --output "./$3"

# Install dependencies
echo "📦 Installing dependencies..."
cd "$3"
npm install

# Build project
echo "🔨 Building project..."
npm run build

# Deploy
echo "🚀 Deploying to Vercel..."
vercel --yes --prod

echo "✅ Done!"
```

### Usage
```bash
chmod +x generate-and-deploy.sh
./generate-and-deploy.sh "SaaS landing page" react my-saas
```
