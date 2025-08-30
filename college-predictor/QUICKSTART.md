# 🚀 Quick Start Guide

## One-Command Setup

```bash
./start.sh
```

This script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Train the ML model
- ✅ Start both backend and frontend
- ✅ Open on `localhost:8000` (API) and `localhost:3000` (UI)

## Manual Setup (3 Steps)

### Step 1: Backend Setup
```bash
cd college-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cd backend && python train_model.py
```

### Step 2: Start Backend API
```bash
cd backend
python main.py
# API runs on http://localhost:8000
```

### Step 3: Start Frontend (New Terminal)
```bash
cd frontend
python -m http.server 3000
# UI runs on http://localhost:3000
```

## Test the System

1. **Open** `http://localhost:3000`
2. **Enter** your percentile (e.g., 95.5)
3. **Select** category, branch, exam type
4. **Click** "Find My Colleges"
5. **View** ranked college predictions!

## Example Test Data

- **Percentile**: 95.5
- **Category**: General
- **Branch**: Computer Science
- **Exam**: CET

Expected: 10+ college matches with admission probabilities

## API Test

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "percentile": 95.5,
    "category": "General", 
    "preferred_branch": "Computer Science",
    "exam_type": "CET"
  }'
```

## Need Help?

- Check `README.md` for detailed instructions
- View API docs at `http://localhost:8000/docs`
- Ensure Python 3.8+ is installed
