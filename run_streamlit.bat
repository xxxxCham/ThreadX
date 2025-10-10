@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

:: ThreadX Streamlit Launcher
:: ===========================
:: Lance l'interface web Streamlit pour ThreadX
:: Optimisé pour Windows avec environnement virtuel

echo.
echo =========================================
echo    THREADX - INTERFACE STREAMLIT
echo =========================================
echo.

:: Se placer dans le dossier du script
cd /d "%~dp0"

:: Vérifier et activer l'environnement virtuel
if exist ".venv\Scripts\activate.bat" (
    echo ✅ Activation environnement virtuel ThreadX
    call ".venv\Scripts\activate.bat"
) else (
    echo ❌ ERREUR: Environnement virtuel introuvable
    echo.
    echo Pour créer l'environnement virtuel:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Configuration optimale pour ThreadX
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set THREADX_ENV=production

:: Créer dossier cache si nécessaire
if not exist "cache\streamlit" (
    echo 📁 Création dossier cache Streamlit
    mkdir "cache\streamlit" 2>nul
)

:: Vérifier fichier principal Streamlit
if exist "apps\streamlit\app.py" (
    echo 🚀 Lancement ThreadX Streamlit (Interface principale)
    echo.
    echo 📍 URL: http://localhost:8504
    echo ⏹️  Pour arrêter: Ctrl+C
    echo.

    :: Lancer avec configuration optimisée
    streamlit run apps\streamlit\app_minimal.py ^^
        --server.port 8504 ^^
        --server.address localhost ^^
        --server.headless false ^^
        --server.runOnSave true ^^
        --server.allowRunOnSave true ^^
        --global.showWarningOnDirectExecution false

) else (
    echo ❌ ERREUR: Fichier principal ThreadX Streamlit introuvable
    echo.
    echo Fichiers attendus: apps\streamlit\app_minimal.py
    echo.
    echo Structure attendue:
    echo    ThreadX/
    echo    ├── apps/
    echo    │   └── streamlit/
    echo    │       └── app.py
    echo    └── .venv/
    echo.
    pause
    exit /b 1
)

endlocal