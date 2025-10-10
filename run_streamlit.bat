@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo.
echo =========================================
echo    THREADX - INTERFACE STREAMLIT
echo =========================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  echo ✅ Activation environnement virtuel ThreadX
  call ".venv\Scripts\activate.bat"
) else (
  echo ❌ ERREUR: Environnement virtuel introuvable
  exit /b 1
)

if not exist "apps\streamlit\app_minimal.py" (
  echo ❌ ERREUR: apps\streamlit\app_minimal.py introuvable
  exit /b 1
)

echo 🚀 Lancement ThreadX Streamlit
streamlit run "apps\streamlit\app_minimal.py" --server.port=8504 --server.address=localhost --server.headless=true --global.showWarningOnDirectExecution=false
