# Quick Start Guide

Get the NBA Prediction System running in 5 minutes!

---

## 📋 Prerequisites

- **Python 3.8+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **Git** (optional) - For cloning

---

## 🚀 Backend Setup (2 minutes)

```bash
# 1. Navigate to backend folder
cd backend

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Import sample data
python nba_predictor.py import ../data/games.csv --description "Sample NBA Data"

# 4. Train models (this may take a few minutes)
python nba_predictor.py train 1

# 5. Launch dashboard
python nba_predictor.py dashboard
```

**Open:** http://localhost:8050

---

## 🎨 Frontend Setup (2 minutes)

**Note:** On Windows PowerShell, if you get execution policy errors, use Command Prompt instead (cmd).

```bash
# 1. Navigate to frontend folder
cd front

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```

**Open:** http://localhost:8080

---

## ⚡ One-Command Quick Test

### Backend Quick Test
```bash
cd backend
python nba_predictor.py auto ../data/games.csv --description "Quick Test"
python nba_predictor.py dashboard
```

### Frontend Quick Test
```bash
cd front
npm install && npm run dev
```

---

## 🎯 What You'll See

### Backend Dashboard (http://localhost:8050)
- Model performance comparison
- Accuracy metrics
- Data statistics
- Interactive charts

### Frontend (http://localhost:8080)
- Modern dark-themed home page
- Navigation to all features
- Data import interface
- Model training controls
- Prediction views
- Performance analysis

---

## 🐛 Common Issues

### Backend

**"ModuleNotFoundError: No module named 'pandas'"**
```bash
pip install -r requirements.txt
```

**"Database is locked"**
- Close any other programs using the database
- Only run one instance of the dashboard

### Frontend

**PowerShell execution policy error**
- Use Command Prompt (cmd) instead
- Or see: front/SETUP_GUIDE.md for solutions

**"Port 8080 already in use"**
```bash
# Change port
npm run dev -- --port 3000
```

---

## 📂 Project Structure

```
NBA_Prediction_Decision_tree/
├── backend/       # Python ML backend
├── front/         # React frontend
├── data/          # CSV files and database
└── docs/          # Documentation
```

---

## 📚 Next Steps

1. **Read the full README.md** for complete documentation
2. **Explore backend/README.md** for backend commands
3. **Check front/README.md** for frontend features
4. **Review docs/** for detailed documentation

---

## 🎓 Learning Path

1. ✅ Install and run (you're here!)
2. 📊 Import your own NBA data
3. 🧠 Train different ML models
4. 📈 Compare model performance
5. 🎯 Make predictions
6. ✔️ Verify against actual results

---

## 💡 Tips

- **Start with the backend** to see the ML models in action
- **Then explore the frontend** for the modern UI
- **Use the auto command** for quick testing
- **Check logs** if something goes wrong

---

## 🆘 Need Help?

- Check **README.md** for detailed documentation
- Review **PROJECT_ORGANIZATION.md** for structure
- See **front/SETUP_GUIDE.md** for frontend troubleshooting
- Check **docs/** folder for detailed guides

---

**Ready to predict NBA games? Let's go! 🏀**
