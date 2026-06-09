# NBA Prediction Frontend

Modern, elegant web interface for the NBA Game Prediction System - leveraging machine learning to predict NBA game outcomes with professional-grade accuracy.

## Overview

This is the frontend application for the NBA Prediction System, built with React, TypeScript, and Tailwind CSS. It provides a sleek, dark-themed interface inspired by professional sports analytics platforms like Flashscore.

## Features

- **Home Page**: Engaging hero section with project overview and quick stats
- **Dashboard**: Central hub for navigation and quick statistics
- **Data Import**: Drag-and-drop interface for CSV/Excel file uploads
- **Model Training**: Interactive model selection and training configuration
- **Predictions**: View upcoming game predictions with confidence scores
- **Analysis**: Comprehensive model performance comparison and metrics
- **Verification**: Compare predictions with actual results
- **About**: Project information and methodology explanation

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - High-quality UI components
- **Framer Motion** - Smooth animations
- **React Router** - Client-side routing
- **Recharts** - Data visualization
- **Lucide React** - Icon library
- **React Hook Form** - Form handling
- **Zod** - Schema validation
- **TanStack Query** - Data fetching and caching

## Getting Started

### Prerequisites

- Node.js 18+ and npm (or use [nvm](https://github.com/nvm-sh/nvm))
- Alternatively, use Bun for faster installation

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Navigate to the frontend directory
cd front

# Install dependencies
npm install
# or with bun
bun install

# Start development server
npm run dev
# or with bun
bun run dev
```

The application will be available at `http://localhost:8080`

### Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
front/
├── src/
│   ├── components/
│   │   ├── layout/        # Navigation, Footer
│   │   ├── shared/        # Reusable components
│   │   └── ui/            # shadcn/ui components
│   ├── pages/             # Route pages
│   │   ├── Index.tsx      # Home page
│   │   ├── Dashboard.tsx  # Main dashboard
│   │   ├── Import.tsx     # Data import
│   │   ├── Train.tsx      # Model training
│   │   ├── Predictions.tsx # View predictions
│   │   ├── Analysis.tsx   # Performance analysis
│   │   ├── Verify.tsx     # Verification
│   │   └── About.tsx      # About page
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # Utilities
│   ├── App.tsx            # Root component
│   └── main.tsx           # Entry point
├── public/                # Static assets
└── index.html             # HTML template
```

## Design System

### Colors

- **Background**: Deep navy/black (#0a0f1c, #0d1117)
- **Cards**: Translucent glass effect with subtle borders
- **Primary Accent**: Orange (#ff6b00, #f97316)
- **Secondary Accent**: Blue (#3b82f6)
- **Success**: Green (#10b981)
- **Text**: White (#ffffff) and gray variants

### Typography

- Clean, modern sans-serif fonts
- Bold headings with wide letter spacing
- Monospace for data and statistics

### Components

- Glassmorphism effects (backdrop-blur, translucent backgrounds)
- Smooth animations with Framer Motion
- Hover effects with subtle glows
- Responsive grid layouts
- Gradient text for emphasis

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run test` - Run tests
- `npm run test:watch` - Run tests in watch mode

## Connecting to Backend

The frontend is designed to work with the Python backend API. Update API endpoints in the configuration to connect to your backend server.

Default backend URL: `http://localhost:8050` (Python Dash dashboard) or your custom API server.

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project in Vercel
3. Deploy automatically

### Netlify

1. Build the project: `npm run build`
2. Deploy the `dist` folder to Netlify

### Docker

```bash
# Build Docker image
docker build -t nba-prediction-frontend .

# Run container
docker run -p 8080:8080 nba-prediction-frontend
```

## Environment Variables

Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:5000
VITE_DASHBOARD_URL=http://localhost:8050
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)

## Performance

- Lazy loading for routes and heavy components
- Optimized images and assets
- Code splitting by route
- Virtualized lists for large datasets

## Accessibility

- ARIA labels on interactive elements
- Keyboard navigation support
- Sufficient color contrast (WCAG AA)
- Screen reader compatible

## License

This project is part of the NBA Prediction System.

## Support

For issues or questions, please refer to the main project documentation or create an issue in the repository.

---

**Built with modern web technologies for professional NBA game predictions.**
