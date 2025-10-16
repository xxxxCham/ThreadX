@echo off
REM =====================================================
REM ThreadX - Lanceur Windows (Double-clic)
REM =====================================================
REM Script de lancement ultra-simple pour Windows
REM Double-cliquez simplement sur ce fichier !
REM =====================================================

title ThreadX - Crypto Trading Framework

echo.
echo ========================================================
echo            THREADX LAUNCHER - WINDOWS
echo ========================================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    echo.
    echo Telechargez Python depuis: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte
echo.

REM Lancer le script Python
echo Lancement de ThreadX...
echo.
python start_threadx.py

REM Pause si erreur
if errorlevel 1 (
    echo.
    echo [ERREUR] Le lancement a echoue
    echo.
    pause
)

exit /b 0
