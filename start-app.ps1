#!/usr/bin/env powershell
# DN Detection Application Startup Script
# This script starts both backend and frontend services

Write-Host "🚀 Starting Diabetic Nephropathy Detection Application..." -ForegroundColor Green
Write-Host "=" -Repeat 60 -ForegroundColor Blue

# Set the base directory
$BASE_DIR = "c:\Users\chand\Desktop\FinalProject_KL"
$BACKEND_DIR = "$BASE_DIR\dn_detection_app\backend"
$FRONTEND_DIR = "$BASE_DIR\dn_detection_app\frontend"
$VENV_PYTHON = "$BASE_DIR\dn_detection_env\Scripts\python.exe"

Write-Host "📁 Base Directory: $BASE_DIR" -ForegroundColor Yellow
Write-Host "🐍 Backend Directory: $BACKEND_DIR" -ForegroundColor Yellow
Write-Host "⚛️  Frontend Directory: $FRONTEND_DIR" -ForegroundColor Yellow

# Function to start backend
function Start-Backend {
    Write-Host "🐍 Starting FastAPI Backend..." -ForegroundColor Cyan
    Set-Location $BACKEND_DIR
    Start-Process -FilePath $VENV_PYTHON -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" -WindowStyle Normal
    Write-Host "✅ Backend started at http://localhost:8000" -ForegroundColor Green
}

# Function to start frontend
function Start-Frontend {
    Write-Host "⚛️  Starting React Frontend..." -ForegroundColor Cyan
    Set-Location $FRONTEND_DIR
    Start-Process -FilePath "npm" -ArgumentList "start" -WindowStyle Normal
    Write-Host "✅ Frontend will start at http://localhost:3000" -ForegroundColor Green
}

# Start both services
try {
    Write-Host "🔄 Starting Backend Service..." -ForegroundColor Blue
    Start-Backend
    
    Write-Host "⏳ Waiting 3 seconds for backend to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    Write-Host "🔄 Starting Frontend Service..." -ForegroundColor Blue
    Start-Frontend
    
    Write-Host "`n🎉 Both services are starting!" -ForegroundColor Green
    Write-Host "📊 Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host "📊 API Docs: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "🌐 Frontend App: http://localhost:3000" -ForegroundColor White
    Write-Host "`n⚠️  Two new terminal windows will open for the services." -ForegroundColor Yellow
    Write-Host "💡 To stop the services, close those terminal windows or press Ctrl+C in each." -ForegroundColor Yellow
    
    Write-Host "`n✨ Application started successfully! ✨" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Error starting application: $_" -ForegroundColor Red
    Write-Host "💡 Make sure you're running this from the correct directory." -ForegroundColor Yellow
}

Write-Host "`n🔍 Press any key to exit this script..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")