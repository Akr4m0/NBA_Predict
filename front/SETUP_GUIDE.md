# Frontend Setup Guide

## Current Status

✅ All files properly configured
✅ Package.json updated with project branding
✅ Vite config cleaned (removed unnecessary dependencies)
✅ HTML metadata updated with NBA Prediction branding
✅ README.md rewritten for the project
✅ All page components present
✅ UI components properly structured

## Next Steps to Run the Application

### 1. Install Dependencies

Before running the application, you need to install all dependencies:

```bash
cd front
npm install
```

Or if you prefer using Bun (faster):

```bash
cd front
bun install
```

### 2. Start Development Server

```bash
npm run dev
```

The application will start at: `http://localhost:8080`

### 3. Verify Installation

After installation completes, you should see:

- A successful npm install with no errors
- All packages installed in `node_modules/`
- Development server running without errors

### 4. Expected Behavior

When you open `http://localhost:8080`, you should see:

- **Home Page**: Dark themed hero section with NBA prediction branding
- **Navigation**: Working links to all pages
- **Smooth Animations**: Framer Motion animations throughout
- **Responsive Design**: Works on mobile, tablet, and desktop

## Available Pages

After starting the dev server, you can navigate to:

- `/` - Home page with hero section
- `/dashboard` - Main dashboard hub
- `/import` - Data import interface
- `/train` - Model training page
- `/predictions` - View predictions
- `/analysis` - Performance analysis
- `/verify` - Prediction verification
- `/about` - About page

## Troubleshooting

### Issue: Dependencies won't install

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete package-lock.json and try again
rm package-lock.json
npm install
```

### Issue: Port 8080 already in use

**Solution**:
Change the port in `vite.config.ts`:
```typescript
server: {
  port: 3000, // Change to any available port
}
```

### Issue: Module not found errors

**Solution**:
Ensure all dependencies are installed:
```bash
npm install --legacy-peer-deps
```

### Issue: Build errors

**Solution**:
Check TypeScript errors:
```bash
npm run lint
```

## Build for Production

When you're ready to deploy:

```bash
# Create optimized production build
npm run build

# Preview production build locally
npm run preview
```

The production files will be in the `dist/` folder.

## Project Structure Verification

All required files are in place:

```
front/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Footer.tsx ✓
│   │   │   └── Navbar.tsx ✓
│   │   ├── shared/
│   │   │   ├── DropZone.tsx ✓
│   │   │   ├── FeatureCard.tsx ✓
│   │   │   ├── MetricCard.tsx ✓
│   │   │   ├── ModelCard.tsx ✓
│   │   │   ├── NavigationCard.tsx ✓
│   │   │   ├── PredictionCard.tsx ✓
│   │   │   └── StatsTicker.tsx ✓
│   │   └── ui/
│   │       └── [50+ shadcn components] ✓
│   ├── pages/
│   │   ├── Index.tsx ✓
│   │   ├── Dashboard.tsx ✓
│   │   ├── Import.tsx ✓
│   │   ├── Train.tsx ✓
│   │   ├── Predictions.tsx ✓
│   │   ├── Analysis.tsx ✓
│   │   ├── Verify.tsx ✓
│   │   ├── About.tsx ✓
│   │   └── NotFound.tsx ✓
│   ├── hooks/ ✓
│   ├── lib/ ✓
│   ├── App.tsx ✓
│   └── main.tsx ✓
├── public/ ✓
├── index.html ✓
├── package.json ✓
├── vite.config.ts ✓
├── tailwind.config.ts ✓
├── tsconfig.json ✓
└── README.md ✓
```

## Changes Made

### 1. Removed External Platform Dependencies

- ❌ Removed `lovable-tagger` package
- ❌ Removed `componentTagger` from Vite config
- ❌ Removed all platform-specific references

### 2. Updated Branding

- ✅ Changed project name to `nba-prediction-frontend`
- ✅ Updated HTML title and meta tags
- ✅ Added proper SEO descriptions
- ✅ Removed placeholder content

### 3. Configuration Files

- ✅ Simplified `vite.config.ts`
- ✅ Updated `package.json` with proper project info
- ✅ Rewrote `README.md` with complete documentation

## Design System

The application uses a professional, dark-themed design:

### Colors
- Background: Deep navy (#0a0f1c, #0d1117)
- Primary Accent: Orange (#ff6b00, #f97316)
- Cards: Glassmorphism with backdrop-blur
- Text: White with gray variants

### Typography
- Bold headings with wide tracking
- Clean sans-serif body text
- Monospace for data/stats

### Effects
- Smooth Framer Motion animations
- Hover glows and transitions
- Gradient text accents
- Responsive grid layouts

## Integration with Backend

To connect to your Python backend:

1. Create `.env` file in `/front`:
```env
VITE_API_URL=http://localhost:5000
VITE_DASHBOARD_URL=http://localhost:8050
```

2. Update API calls in your components to use these environment variables

3. Ensure CORS is properly configured on your backend

## Deployment Options

### Vercel (Recommended for React/Vite)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Netlify
```bash
# Build
npm run build

# Drag and drop the 'dist' folder to Netlify
```

### Custom Server
```bash
# Build
npm run build

# Copy 'dist' folder to your web server
# Configure server to serve index.html for all routes (SPA mode)
```

## Notes

- The frontend is completely standalone and can run independently
- Backend integration requires API endpoints (not included in frontend)
- All UI components are self-contained
- No external platform dependencies remain
- Ready for production deployment

## Support

If you encounter any issues:

1. Check this guide for common solutions
2. Verify all dependencies are installed
3. Check browser console for errors
4. Review the main project documentation

---

**Ready to run!** Just install dependencies and start the dev server.
