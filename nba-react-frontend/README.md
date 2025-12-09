# NBA Prediction System - React Frontend

A modern, animated React frontend for the NBA Game Prediction application, built with Next.js, TypeScript, Tailwind CSS, and shadcn/ui components.

## 🎨 Features

- **Modern Tech Stack**: Next.js 16, TypeScript, Tailwind CSS v4
- **Component Library**: shadcn/ui with Radix UI primitives
- **Animations**: Framer Motion for smooth, engaging animations
- **NBA Themed**: Custom gradient backgrounds with NBA colors (blue #1d428a, red #c8102e)
- **Responsive Design**: Mobile-first approach, works on all devices
- **Dark Mode Ready**: Built-in dark mode support

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. **Install dependencies**:
```bash
npm install
```

2. **Run development server**:
```bash
npm run dev
```

3. **Open your browser**:
Navigate to [http://localhost:3000](http://localhost:3000)

## 📁 Project Structure

```
nba-react-frontend/
├── app/
│   ├── page.tsx              # Home page with hero
│   ├── dashboard/
│   │   └── page.tsx          # Dashboard page
│   ├── about/
│   │   └── page.tsx          # About page
│   ├── layout.tsx            # Root layout
│   └── globals.css           # Global styles
├── components/
│   ├── ui/
│   │   ├── background-paths.tsx  # Original animated component
│   │   └── button.tsx            # shadcn Button component
│   ├── nba-hero.tsx              # NBA-themed hero component
│   └── demo-background-paths.tsx # Demo component
├── lib/
│   └── utils.ts              # Utility functions (cn helper)
├── public/                   # Static assets
└── components.json           # shadcn/ui configuration
```

## 🎯 Component Integration

### BackgroundPaths Component

The `BackgroundPaths` component creates an animated SVG background with floating paths:

**Location**: `components/ui/background-paths.tsx`

**Usage**:
```tsx
import { BackgroundPaths } from "@/components/ui/background-paths";

export default function Page() {
  return <BackgroundPaths title="Your Title Here" />;
}
```

**Props**:
- `title` (optional): String - The main title text (default: "Background Paths")

**Features**:
- Animated SVG paths that loop infinitely
- Framer Motion spring animations for text
- Responsive typography (scales from mobile to desktop)
- Dark mode compatible
- Customizable button with hover effects

### NBA Hero Component

Custom NBA-themed landing page component:

**Location**: `components/nba-hero.tsx`

**Features**:
- NBA gradient background (blue to red)
- Animated title with letter-by-letter reveal
- Feature cards with icons
- Call-to-action buttons
- Basketball emoji decoration
- Fully responsive

## 🎨 Styling

### Tailwind CSS v4

The project uses Tailwind CSS v4 with custom configuration:

```css
/* globals.css */
@import "tailwindcss";

/* Custom theme colors matching NBA branding */
```

### Color Palette

- **Primary Gradient**: Slate-900 → Blue-900 → Red-900
- **NBA Blue**: #1d428a
- **NBA Red**: #c8102e
- **Text**: White with opacity variations
- **Accents**: Blue-300, Green-300, Red-300

## 📦 Dependencies

### Core
- `next@16.0.7` - React framework
- `react@19` - UI library
- `typescript@5` - Type safety

### UI & Styling
- `tailwindcss@4` - Utility-first CSS
- `@tailwindcss/postcss` - PostCSS plugin
- `shadcn/ui` - Component library
- `@radix-ui/react-slot` - Composition primitive
- `class-variance-authority` - Variant handling
- `clsx` - Class merging
- `tailwind-merge` - Tailwind class merging

### Animation
- `framer-motion@12` - Animation library

## 🛠️ Available Scripts

```bash
# Development
npm run dev          # Start dev server (http://localhost:3000)

# Production
npm run build        # Build for production
npm start            # Start production server

# Linting
npm run lint         # Run ESLint

# Type checking
npm run type-check   # Run TypeScript compiler
```

## 🎭 Component Customization

### Adding New Components

Use shadcn CLI to add more components:

```bash
npx shadcn@latest add [component-name]
```

Available components: button, card, input, dialog, dropdown-menu, and many more.

### Modifying Colors

Edit the gradient backgrounds in component files:

```tsx
// From this:
className="bg-gradient-to-br from-slate-900 via-blue-900 to-red-900"

// To custom colors:
className="bg-gradient-to-br from-purple-900 via-pink-900 to-orange-900"
```

### Animation Speed

Adjust Framer Motion transition durations:

```tsx
// Slower animation
transition={{ duration: 3 }}

// Faster animation
transition={{ duration: 0.5 }}
```

## 🔗 Integration with Python Backend

To connect with the existing NBA prediction Python backend:

1. **Create API routes** in `app/api/` directory:

```typescript
// app/api/import/route.ts
export async function POST(request: Request) {
  const formData = await request.formData();
  // Forward to Python backend
  const response = await fetch('http://localhost:8050/api/import', {
    method: 'POST',
    body: formData,
  });
  return response;
}
```

2. **Update environment variables**:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8050
```

3. **Call from components**:

```tsx
const response = await fetch('/api/import', {
  method: 'POST',
  body: formData,
});
```

## 🌐 Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project on [Vercel](https://vercel.com)
3. Deploy automatically

### Other Platforms

```bash
# Build production bundle
npm run build

# Start production server
npm start
```

The app will run on port 3000 by default.

## 📚 Pages

### Home (`/`)
- Animated hero section with NBA theme
- Feature cards
- Call-to-action buttons
- Basketball decorations

### Dashboard (`/dashboard`)
- Feature navigation cards
- Quick stats display
- Links to main functionality

### About (`/about`)
- Project overview
- Features list
- ML models description
- Technology stack

## 🎨 Design Decisions

### Why Next.js?
- Server-side rendering for better SEO
- File-based routing
- API routes for backend integration
- Image optimization
- Production-ready out of the box

### Why shadcn/ui?
- Copy-paste components (owns the code)
- Built on Radix UI (accessible)
- Customizable with Tailwind
- TypeScript first
- No npm package bloat

### Why Framer Motion?
- Declarative animations
- Spring physics
- SVG path animations
- Gesture support
- Production-tested

## 🔧 Troubleshooting

### Port already in use
```bash
# Kill process on port 3000
npx kill-port 3000

# Or use different port
npm run dev -- -p 3001
```

### Module not found errors
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
```

### Type errors
```bash
# Check TypeScript errors
npm run type-check
```

## 📝 License

Part of the NBA Prediction System project. Open source.

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📞 Support

For issues or questions:
- Check existing documentation
- Review Next.js docs: https://nextjs.org/docs
- Review shadcn/ui docs: https://ui.shadcn.com

---

**Built with ❤️ using Next.js, TypeScript, and Tailwind CSS**
