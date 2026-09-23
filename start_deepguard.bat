.@echo off

title DeepGuard AI

echo ==========================================
echo          DeepGuard AI
echo      Deepfake Detection System
echo ==========================================
echo.

cd /d "%~dp0"

echo Starting Django backend...
start "DeepGuard Backend" cmd /k "call .venv\Scripts\activate && cd backend && python manage.py runserver"

timeout /t 3 /nobreak >nul

echo Starting React frontend...
start "DeepGuard Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo DeepGuard AI services are starting...
echo ==========================================
echo.
echo Backend : http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo.
echo Close the two service windows to stop them.
echo ==========================================

pause