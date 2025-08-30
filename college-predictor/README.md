# 🎓 College Predictor (CET/JEE) - Complete ML Project

A complete machine learning project that predicts the top colleges a student can get based on CET/JEE percentile, category, branch preference, and exam type.

## 🚀 Features

- **ML Training Pipeline**: Python/scikit-learn based model training
- **REST API**: FastAPI backend serving predictions
- **React Frontend**: Single-file React component with responsive UI
- **Sample Dataset**: Realistic college data with cutoffs and rankings
- **Easy Setup**: Run locally with or without virtual environment

## 📁 Project Structure

```
college-predictor/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── train_model.py       # ML training pipeline
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html          # Single-file React app
│   └── package.json        # Frontend metadata
├── data/
│   └── colleges_data.csv   # Sample college dataset
├── models/                 # Trained model files (generated)
├── docs/                   # Documentation
├── .env.example           # Environment configuration
└── README.md              # This file
```

## 🛠️ Quick Setup & Run

### Option 1: Using Python Virtual Environment (Recommended)

```bash
# 1. Clone/navigate to project directory
cd college-predictor

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r backend/requirements.txt

# 4. Train the ML model
cd backend
python train_model.py
cd ..

# 5. Start the backend API
cd backend
python main.py
# API will be available at http://localhost:8000
```

In a new terminal:
```bash
# 6. Start the frontend
cd college-predictor/frontend
python -m http.server 3000
# Frontend will be available at http://localhost:3000
```

### Option 2: Direct Installation (System Python)

```bash
# 1. Navigate to project directory
cd college-predictor

# 2. Install dependencies directly
pip install -r backend/requirements.txt

# 3. Train model and start backend
cd backend
python train_model.py
python main.py &  # Run in background
cd ..

# 4. Start frontend
cd frontend
python -m http.server 3000
```

### Option 3: Using Node.js for Frontend (Optional)

```bash
# After setting up backend:
cd frontend
npm install
npm run serve-node
```

## 🔧 Detailed Setup Instructions

### Prerequisites

- **Python 3.8+**: Required for backend and ML pipeline
- **Modern Browser**: Chrome, Firefox, Safari, or Edge for frontend
- **Optional**: Node.js 14+ for alternative frontend serving

### Backend Setup

1. **Install Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```
   This will:
   - Load the sample dataset
   - Train a Random Forest classifier
   - Save the model to `../models/`
   - Display training metrics

3. **Start the API Server**:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`

4. **Verify API**:
   ```bash
   curl http://localhost:8000/health
   ```

### Frontend Setup

The frontend is a single HTML file with embedded React. No build process required!

1. **Serve the Frontend**:
   ```bash
   cd frontend
   python -m http.server 3000
   ```

2. **Access the App**:
   Open `http://localhost:3000` in your browser

## 📊 API Endpoints

### Core Endpoints

- **GET** `/` - Health check and API info
- **GET** `/health` - Detailed health status
- **GET** `/options` - Available form options (branches, categories, etc.)
- **POST** `/predict` - Get college predictions for a student
- **GET** `/colleges` - Get all available colleges
- **GET** `/colleges/{exam_type}` - Get colleges by exam type
- **POST** `/retrain` - Retrain the ML model

### Example API Usage

```bash
# Get college predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "percentile": 95.5,
    "category": "General",
    "preferred_branch": "Computer Science",
    "exam_type": "CET",
    "max_results": 10
  }'
```

## 🎯 How It Works

### ML Model
- **Algorithm**: Random Forest Classifier
- **Features**: Student percentile, category, branch, exam type, college tier, fees, placements
- **Target**: Admission probability (0-1)
- **Training**: Synthetic data generation based on cutoff percentiles

### Prediction Logic
1. Takes student inputs (percentile, category, branch, exam type)
2. Filters colleges matching exam type and branch
3. Calculates admission probability for each college
4. Returns ranked list of colleges by admission probability

### Frontend Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Validation**: Form validation and error handling
- **Visual Indicators**: Color-coded probability bars and badges
- **College Details**: Fees, placements, tier information

## 📈 Sample Predictions

For a **95th percentile General category** student interested in **Computer Science** via **CET**:

1. **VJTI Mumbai** - 98.2% probability
2. **COEP Pune** - 97.9% probability  
3. **Government College of Engineering Pune** - 97.5% probability
4. **Pune Institute of Computer Technology** - 95.5% probability
5. **Maharashtra Institute of Technology** - 94.2% probability

## 🔄 Development Workflow

### Adding New Colleges
1. Update `data/colleges_data.csv` with new college data
2. Retrain the model: `python backend/train_model.py`
3. Restart the API server

### Updating the Model
- Modify training parameters in `train_model.py`
- Add new features to the dataset
- Retrain and test the model

### Frontend Customization
- Edit `frontend/index.html` directly
- No build process required - changes are immediate
- Modify styling in the `<style>` section

## 🧪 Testing

### Test the ML Model
```bash
cd backend
python train_model.py
```

### Test the API
```bash
# Start the server
python backend/main.py

# In another terminal, test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/options
```

### Test the Frontend
1. Open `http://localhost:3000`
2. Fill in the form with test data
3. Verify predictions are displayed correctly

## 📋 Data Schema

### College Dataset (`colleges_data.csv`)
| Column | Description | Example |
|--------|-------------|---------|
| `college_name` | Name of the college | "IIT Bombay" |
| `branch` | Engineering branch | "Computer Science" |
| `exam_type` | CET or JEE | "JEE" |
| `general_cutoff` | General category cutoff percentile | 99.8 |
| `obc_cutoff` | OBC category cutoff percentile | 99.6 |
| `sc_cutoff` | SC category cutoff percentile | 98.2 |
| `st_cutoff` | ST category cutoff percentile | 97.8 |
| `ews_cutoff` | EWS category cutoff percentile | 99.7 |
| `state` | College location state | "Maharashtra" |
| `tier` | College tier (1-3) | 1 |
| `fees_per_year` | Annual fees in INR | 200000 |
| `placement_percentage` | Placement success rate | 95 |

### API Request Schema
```json
{
  "percentile": 95.5,
  "category": "General",
  "preferred_branch": "Computer Science", 
  "exam_type": "CET",
  "max_results": 20
}
```

### API Response Schema
```json
{
  "predictions": [
    {
      "college_name": "VJTI Mumbai",
      "branch": "Computer Science",
      "state": "Maharashtra",
      "tier": 2,
      "fees_per_year": 150000,
      "placement_percentage": 88,
      "admission_probability": 0.982,
      "cutoff_percentile": 98.5
    }
  ],
  "student_info": {...},
  "total_colleges": 15
}
```

## 🚨 Important Notes

### Data Accuracy
⚠️ **This project uses synthetic/sample data for demonstration**. Real performance depends on actual cutoff and allotment data from:
- Maharashtra CET Cell (for CET)
- JoSAA (for JEE)
- Individual college websites

### Production Considerations
- Replace sample data with real cutoff data
- Add database storage for scalability
- Implement user authentication if needed
- Add caching for better performance
- Use environment-specific configurations

## 🛠️ Troubleshooting

### Common Issues

1. **"Model not loaded" error**:
   ```bash
   cd backend
   python train_model.py  # Retrain the model
   ```

2. **CORS errors in frontend**:
   - Ensure API is running on port 8000
   - Check browser console for specific errors

3. **Import errors**:
   ```bash
   pip install -r backend/requirements.txt  # Reinstall dependencies
   ```

4. **Port already in use**:
   ```bash
   lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
   lsof -ti:3000 | xargs kill -9  # Kill process on port 3000
   ```

### Performance Issues
- Model training takes 10-30 seconds on first run
- Predictions are typically under 100ms
- Frontend loads instantly (single HTML file)

## 🔮 Future Enhancements

- **Real Data Integration**: Connect to official cutoff APIs
- **Advanced ML**: Deep learning models, ensemble methods
- **User Accounts**: Save predictions, track applications
- **Mobile App**: React Native or Flutter version
- **Analytics**: Application success tracking
- **Notifications**: Cutoff alerts and updates

## 📄 License

MIT License - Feel free to use and modify for educational or commercial purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review API documentation at `http://localhost:8000/docs`
- Verify all dependencies are installed correctly

---

**Happy College Hunting! 🎓✨**
