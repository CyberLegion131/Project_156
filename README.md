# 🏥 Diabetic Nephropathy Detection System

A machine learning-powered web application for early detection and risk assessment of diabetic nephropathy using clinical biomarkers.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start Guide](#quick-start-guide)
- [Manual Setup](#manual-setup)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

## 🎯 Overview

This application provides healthcare professionals with an AI-powered tool to assess diabetic nephropathy risk using standard clinical parameters. The system combines a FastAPI backend with a React frontend to deliver real-time predictions through an intuitive web interface.

## ✨ Features

- **🤖 AI-Powered Predictions**: Advanced machine learning model for DN risk assessment
- **📊 Model Accuracy Display**: Real-time model performance metrics
- **🎨 Intuitive Dashboard**: Clean, user-friendly interface for healthcare professionals
- **⚡ Real-time Results**: Instant predictions with clinical recommendations
- **🔒 Secure Authentication**: JWT-based user authentication system
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🚀 Easy Deployment**: One-click startup scripts for local development

## 🛠 Prerequisites

Before running the application, ensure you have the following installed:

### Required Software:
- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **npm** (comes with Node.js)
- **Git** - [Download Git](https://git-scm.com/)

### System Requirements:
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **Network**: Internet connection for initial setup

### 📦 Python Dependencies
All Python dependencies are consolidated in a **single `requirements.txt` file** at the project root for simplified setup. This includes:
- **Web Framework**: FastAPI, Uvicorn
- **Machine Learning**: scikit-learn, ONNX, pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Development Tools**: Jupyter, pytest
- **Database**: SQLAlchemy
- **Authentication**: JWT, bcrypt

## 🚀 Quick Start Guide

### Option 1: Automated Setup (Recommended for Windows)

1. **Navigate to the project directory**:
   ```bash
   cd C:\Users\User\Desktop\FinalProject_KL
   ```

2. **Run the automated startup script**:
   ```bash
   # For Windows Command Prompt
   start-app.bat
   
   # OR for PowerShell
   .\start-app.ps1
   ```

3. **Access the application**:
   - **Frontend**: Open [http://localhost:3000](http://localhost:3000) in your browser
   - **Backend API**: [http://localhost:8000](http://localhost:8000)
   - **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

That's it! The application should now be running with both frontend and backend services.

### Option 2: Manual Setup

If you prefer to set up the services manually or the automated script doesn't work:

#### Backend Setup:

1. **Navigate to backend directory and start the server**:
   ```bash
   # Windows (Single command)
   cd C:\Users\User\Desktop\FinalProject_KL\dn_detection_app\backend; C:\Users\User\Desktop\FinalProject_KL\dn_detection_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   
   # OR step by step:
   cd C:\Users\User\Desktop\FinalProject_KL\dn_detection_app\backend
   C:\Users\User\Desktop\FinalProject_KL\dn_detection_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   
   # macOS/Linux
   cd dn_detection_app/backend
   source ../../dn_detection_env/bin/activate
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Frontend Setup:

1. **Open a new terminal and navigate to frontend directory**:
   ```bash
   # Windows (Full path)
   cd C:\Users\User\Desktop\FinalProject_KL\dn_detection_app\frontend
   
   # OR relative path from project root
   cd dn_detection_app/frontend
   ```

2. **Install dependencies** (first time only):
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```

## 📚 API Documentation

### Authentication
```http
POST http://localhost:8000/api/v1/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

### Make Prediction
```http
POST http://localhost:8000/api/v1/predict
Authorization: Bearer <your-jwt-token>
Content-Type: application/json

{
  "age": 65,
  "gender": 1,
  "glucose": 180,
  "hba1c": 8.5,
  "creatinine": 1.8,
  "urea": 45,
  "systolic_bp": 145,
  "diastolic_bp": 90
}
```

### Input Parameters:
- **age**: Patient age (1-120 years)
- **gender**: 0 = Female, 1 = Male
- **glucose**: Blood glucose level (mg/dL)
- **hba1c**: Glycated hemoglobin percentage
- **creatinine**: Serum creatinine level (mg/dL)
- **urea**: Blood urea nitrogen (optional)
- **systolic_bp**: Systolic blood pressure (optional)
- **diastolic_bp**: Diastolic blood pressure (optional)

### Response Example:
```json
{
  "binary_prediction": 0,
  "prediction_text": "No DN",
  "model_accuracy": 94.2,
  "timestamp": "2024-01-01T10:00:00"
}
```

## 📁 Project Structure

```
FinalProject_KL/
├── 📄 README.md                    # This file
├── 🔧 start-app.bat               # Windows startup script
├── 🔧 start-app.ps1               # PowerShell startup script
├── 📊 dn_detection.db             # SQLite database
├── 🐍 dn_detection_env/           # Python virtual environment
└── 📱 dn_detection_app/           # Main application
    ├── 🔙 backend/                # FastAPI backend
    │   ├── 📄 requirements.txt    # Python dependencies
    │   ├── 🐳 Dockerfile         # Backend container config
    │   └── 📂 app/               # Application code
    │       ├── main.py           # FastAPI main application
    │       ├── models/           # ML models and data models
    │       ├── routers/          # API route handlers
    │       └── database/         # Database configuration
    ├── 🎨 frontend/               # React frontend
    │   ├── 📄 package.json       # Node.js dependencies
    │   ├── 🐳 Dockerfile         # Frontend container config
    │   ├── 📂 public/            # Static assets
    │   └── 📂 src/               # React source code
    │       ├── components/       # React components
    │       ├── pages/           # Page components
    │       └── services/        # API services
    ├── 🤖 ml_model/              # Machine learning pipeline
    │   ├── training_pipeline.py  # Model training script
    │   ├── inference_pipeline.py # Model inference
    │   └── models/              # Trained model files
    └── 📚 docs/                  # Documentation
```

## 🧪 Testing

### Run Backend Tests:
```bash
cd dn_detection_app/backend
python -m pytest tests/ -v
```

### Run Frontend Tests:
```bash
cd dn_detection_app/frontend
npm test
```

### Test API Endpoints:
```bash
cd dn_detection_app/backend
python test_api.py
```

## 🐛 Troubleshooting

### Common Issues and Solutions:

#### 1. "Port already in use" Error
**Problem**: Port 3000 or 8000 is already occupied
**Solution**:
```bash
# Kill processes on port 3000 (Frontend)
npx kill-port 3000

# Kill processes on port 8000 (Backend)
npx kill-port 8000
```

#### 2. Python Virtual Environment Issues
**Problem**: Virtual environment not activating or missing dependencies
**Solution**:
```bash
# Create new virtual environment
python -m venv dn_detection_env

# Activate it
dn_detection_env\Scripts\activate  # Windows
source dn_detection_env/bin/activate  # macOS/Linux

# Install ALL dependencies from single requirements file
pip install -r requirements.txt
```

#### 3. Node.js Dependency Issues
**Problem**: npm install fails or missing packages
**Solution**:
```bash
cd dn_detection_app/frontend
rm -rf node_modules package-lock.json  # Clean install
npm install
```

#### 4. Database Connection Error
**Problem**: SQLite database not found
**Solution**: The database file should be created automatically. If not:
```bash
cd dn_detection_app/backend
python -c "from app.database.database import create_tables; create_tables()"
```

#### 5. CORS Errors
**Problem**: Frontend can't connect to backend
**Solution**: Ensure both services are running on correct ports:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

#### 6. Model Loading Issues
**Problem**: ML model not loading
**Solution**: Ensure model files exist in `ml_model/models/` directory

### Getting More Help:

1. **Check the logs**: Look for error messages in the terminal output
2. **Verify ports**: Ensure ports 3000 and 8000 are available
3. **Check dependencies**: Make sure all required software is installed
4. **Restart services**: Close terminals and run startup script again

## 📞 Support

### Default Login Credentials:
- **Username**: `admin`
- **Password**: `admin123`

### Application URLs:
- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Database Admin**: SQLite browser can be used to view `dn_detection.db`

### System Status Endpoints:
- **Health Check**: http://localhost:8000/health
- **Model Info**: http://localhost:8000/api/v1/model/info

### For Additional Support:
- Check the terminal output for error messages
- Review the API documentation at http://localhost:8000/docs
- Ensure all prerequisites are properly installed
- Try restarting the application using the startup scripts

---

## 🎉 You're Ready to Go!

Once the application is running:

1. **Open your browser** and navigate to http://localhost:3000
2. **Login** using the default credentials (admin/admin123)
3. **Enter patient data** in the form
4. **View predictions** and model accuracy
5. **Explore the API** at http://localhost:8000/docs

The system is now ready to provide diabetic nephropathy risk assessments! 🚀

---

**⚠️ Important Note**: This application is designed for educational and research purposes. For clinical use, please ensure proper validation and compliance with healthcare regulations.