# NBA Game Prediction System

> Machine Learning-powered NBA game prediction platform with professional analytics

A comprehensive full-stack application for predicting NBA game outcomes using advanced machine learning algorithms. Features a Python backend with multiple ML models and a modern React frontend with a dark, elegant design.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![React](https://img.shields.io/badge/react-18.3-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Features

### Backend (Python)
- **Multiple ML Models**: Decision Tree, Random Forest, XGBoost, Baseline
- **Data Import**: CSV/Excel file support with automatic cleaning
- **Performance Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Interactive Dashboard**: Dash-based web visualization
- **Prediction Verification**: Compare predictions with actual results
- **CLI Interface**: Complete command-line tool

### Frontend (React)
- **Modern UI**: Dark theme with glassmorphism effects
- **9 Pages**: Home, Dashboard, Import, Train, Predictions, Analysis, Verify, About
- **Responsive Design**: Mobile, tablet, and desktop support
- **Smooth Animations**: Framer Motion throughout
- **Data Visualization**: Recharts for performance charts
- **Professional Design**: Subtle, elegant, Flashscore-inspired

---

## 📁 Project Structure

```
NBA_Prediction_Decision_tree/
├── backend/                  # Python ML backend
│   ├── nba_predictor.py     # Main CLI application
│   ├── database.py          # SQLite database operations
│   ├── data_importer.py     # Data import and cleaning
│   ├── predictive_models.py # ML models
│   ├── performance_evaluator.py # Model evaluation
│   ├── dashboard.py         # Dash dashboard
│   ├── real_data_loader.py  # Verification
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Backend documentation
│
├── front/                    # React frontend
│   ├── src/
│   │   ├── pages/           # 9 page components
│   │   ├── components/      # Reusable components
│   │   ├── hooks/           # Custom hooks
│   │   └── lib/             # Utilities
│   ├── package.json         # Node dependencies
│   ├── README.md            # Frontend documentation
│   └── SETUP_GUIDE.md       # Setup instructions
│
├── data/                     # Data files and database
│   ├── games.csv            # Sample NBA game data
│   └── nba_predictions.db   # SQLite database
│
├── docs/                     # Documentation
│   ├── PROJECT_DOCUMENTATION.txt  # Complete project docs
│   ├── FRONTEND_REVIEW.md         # Frontend verification
│   ├── DATABASE_SCHEMA.txt        # Database structure
│   ├── INTEGRATION_SUMMARY.md     # Integration guide
│   ├── NEW_FEATURES.md            # Feature documentation
│   ├── PRESENTATION_CONTENT.txt   # Presentation material
│   └── warnings_guide.md          # Development notes
│
├── .gitignore
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Import data
python nba_predictor.py import ../data/games.csv --description "NBA 2023 Season"

# 4. Train models
python nba_predictor.py train 1

# 5. Launch dashboard
python nba_predictor.py dashboard
```

Visit: `http://localhost:8050`

### Frontend Setup

```bash
# 1. Navigate to frontend
cd front

# 2. Install dependencies
npm install
# or
bun install

# 3. Start development server
npm run dev
# or
bun run dev
```

Visit: `http://localhost:8080`

---

## 📊 Usage Examples

### Import NBA Data

```bash
cd backend
python nba_predictor.py import ../data/games.csv --description "2023 Season"
```

### Train All Models

```bash
python nba_predictor.py train 1 --models decision_tree random_forest xgboost
```

### Compare Model Performance

```bash
python nba_predictor.py compare
```

### One-Command Workflow

```bash
python nba_predictor.py auto ../data/games.csv --description "Quick Test"
```

---

## 🧠 Machine Learning Models

### 1. Decision Tree Classifier
- Interpretable decision paths
- Feature importance analysis
- Good baseline performance

### 2. Random Forest Classifier
- Ensemble of multiple trees
- Higher accuracy and robustness
- Reduced overfitting

### 3. XGBoost (Gradient Boosting)
- State-of-the-art performance
- Advanced gradient boosting
- Handles complex patterns

### 4. Baseline Model
- Simple majority predictor
- Performance benchmark

---

## 📈 Feature Engineering

The system automatically creates these features:

- **Team Encoding**: Numerical representation of teams
- **Temporal Features**: Month, day of week, day of year
- **Season Features**: Season identifiers
- **Statistical Features**: FG%, rebounds, assists, etc.
- **Historical Features**: Win/loss records, streaks
- **Score Features**: Point differentials
- **Location Features**: Home/away advantages

---

## 🎨 Design System (Frontend)

### Colors
- **Background**: `#0a0f1c` (Deep navy)
- **Primary Accent**: `#ff6b00` (Orange)
- **Secondary Accent**: `#3b82f6` (Blue)
- **Success**: `#10b981` (Green)
- **Text**: `#ffffff` (White) with gray variants

### Typography
- Bold headings with wide tracking
- Clean sans-serif body text
- Monospace for data/statistics

### Effects
- Glassmorphism: `backdrop-blur-md bg-white/5`
- Smooth animations: Framer Motion (0.3-0.6s)
- Hover glows and scale effects
- Gradient text accents

---

## 📦 Dependencies

### Backend (Python)
```
pandas
numpy
scikit-learn
xgboost
dash
plotly
openpyxl
```

### Frontend (React)
```
react, react-dom, react-router-dom
typescript
vite
tailwindcss
framer-motion
recharts
lucide-react
shadcn/ui components
react-hook-form
zod
```

---

## 🗄️ Database Schema

SQLite database with 4 main tables:

1. **import_records**: Track data imports
2. **models**: Store trained model metadata
3. **prediction_results**: Performance metrics
4. **game_data**: Historical NBA games

See `docs/DATABASE_SCHEMA.txt` for details.

---

## 📝 Available Commands (Backend)

| Command | Description | Example |
|---------|-------------|---------|
| `import` | Import historical data | `python nba_predictor.py import data.csv` |
| `train` | Train models | `python nba_predictor.py train 1` |
| `compare` | Compare models | `python nba_predictor.py compare` |
| `list` | List datasets | `python nba_predictor.py list` |
| `report` | Generate report | `python nba_predictor.py report 1` |
| `dashboard` | Launch dashboard | `python nba_predictor.py dashboard` |
| `auto` | Import + train + compare | `python nba_predictor.py auto data.csv` |

---

## 🌐 Frontend Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Home | Hero section with features |
| `/dashboard` | Dashboard | Main navigation hub |
| `/import` | Import Data | Upload CSV/Excel files |
| `/train` | Train Models | Configure and train ML models |
| `/predictions` | Predictions | View game predictions |
| `/analysis` | Analysis | Model performance charts |
| `/verify` | Verification | Compare with actual results |
| `/about` | About | Project information |

---

## 🔧 Configuration

### Backend Configuration

Default database location: `../data/nba_predictions.db`

Override with:
```bash
python nba_predictor.py --db /custom/path/database.db
```

### Frontend Configuration

Create `.env` file in `/front`:
```env
VITE_API_URL=http://localhost:5000
VITE_DASHBOARD_URL=http://localhost:8050
```

---

## 🚢 Deployment

### Backend
```bash
cd backend
pip install -r requirements.txt
python nba_predictor.py dashboard --port 8050
```

### Frontend

**Vercel:**
```bash
cd front
npm run build
vercel --prod
```

**Netlify:**
```bash
cd front
npm run build
# Deploy dist/ folder
```

**Docker:**
```bash
cd front
docker build -t nba-prediction-frontend .
docker run -p 8080:8080 nba-prediction-frontend
```

---

## 📚 Documentation

- **[Backend README](backend/README.md)** - Backend usage guide
- **[Frontend README](front/README.md)** - Frontend documentation
- **[Frontend Setup Guide](front/SETUP_GUIDE.md)** - Detailed setup
- **[Project Documentation](docs/PROJECT_DOCUMENTATION.txt)** - Complete specs
- **[Frontend Review](docs/FRONTEND_REVIEW.md)** - Verification report
- **[Database Schema](docs/DATABASE_SCHEMA.txt)** - Database structure

---

## 🧪 Testing

### Backend
```bash
cd backend
python -m pytest  # If tests are added
```

### Frontend
```bash
cd front
npm run test
```

---

## 🐛 Troubleshooting

### Backend Issues

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"Database locked"**
- Close other connections to the database
- Ensure only one process is writing

### Frontend Issues

**Dependencies won't install**
```bash
npm cache clean --force
rm package-lock.json
npm install
```

**Port 8080 already in use**
- Change port in `vite.config.ts` or:
```bash
npm run dev -- --port 3000
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🎯 Future Enhancements

See `docs/PROJECT_DOCUMENTATION.txt` for comprehensive list of planned features:

- Neural Networks & Deep Learning
- Live game predictions
- Player-level analysis
- User accounts & profiles
- Betting integration
- Mobile app
- Multi-sport expansion

---

## 📞 Support

For issues or questions:
- Check documentation in `/docs`
- Review troubleshooting sections
- Check backend/frontend README files

---

## ⚡ Performance

- **Backend**: SQLite for fast queries
- **Frontend**: Code splitting, lazy loading, optimized images
- **Models**: Efficient feature engineering and caching

---

## 🔒 Security

- No hardcoded credentials
- Environment variables for sensitive data
- Input validation on data import
- CORS configuration for API

---

**Built with Python, React, and Machine Learning for professional NBA game predictions.**

*Version 1.0.0 - January 2026*
