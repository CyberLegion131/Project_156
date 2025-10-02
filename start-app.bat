@echo off
echo 🚀 Starting Diabetic Nephropathy Detection Application...
echo ============================================================

set BASE_DIR=c:\Users\chand\Desktop\FinalProject_KL
set BACKEND_DIR=%BASE_DIR%\dn_detection_app\backend
set FRONTEND_DIR=%BASE_DIR%\dn_detection_app\frontend
set VENV_PYTHON=%BASE_DIR%\dn_detection_env\Scripts\python.exe

echo 📁 Base Directory: %BASE_DIR%
echo 🐍 Backend Directory: %BACKEND_DIR%
echo ⚛️  Frontend Directory: %FRONTEND_DIR%

echo.
echo 🐍 Starting FastAPI Backend...
cd /d "%BACKEND_DIR%"
start "DN Backend" "%VENV_PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo ⏳ Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak >nul

echo.
echo ⚛️  Starting React Frontend...
cd /d "%FRONTEND_DIR%"
start "DN Frontend" npm start

echo.
echo 🎉 Both services are starting!
echo 📊 Backend API: http://localhost:8000
echo 📊 API Docs: http://localhost:8000/docs
echo 🌐 Frontend App: http://localhost:3000
echo.
echo ⚠️  Two new terminal windows opened for the services.
echo 💡 To stop the services, close those terminal windows.
echo.
echo ✨ Application started successfully! ✨
echo.
pause