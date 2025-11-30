@echo off
echo ========================================
echo   NeuroWheel - Starting Application
echo ========================================
echo.

streamlit run app.py --server.maxUploadSize=500 --server.headless=true

pause

