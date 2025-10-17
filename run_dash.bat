@echo off
REM run_dash.bat — ouvre une nouvelle fenêtre PowerShell et lance l'app Dash
REM Usage: double-cliquer pour lancer; ou exécuter depuis un terminal pour voir les retours

REM Se placer dans le dossier du script
pushd "%~dp0"

REM Construire la commande PowerShell à exécuter dans la nouvelle fenêtre
set "PS_CMD_SetLocation=Set-Location -LiteralPath '%CD%';"
set "PS_CMD_SetEnv=$env:THREADX_CONFIG_DIR='%CD%\configs';"
set "PS_CMD_Info=Write-Host 'THREADX_CONFIG_DIR=' $env:THREADX_CONFIG_DIR; Write-Host 'Starting Dash app...';"

IF EXIST ".venv\Scripts\Activate.ps1" (
    echo Virtualenv found: .venv — launching PowerShell and activating it
    REM Lancer dans une nouvelle fenêtre PowerShell et garder la fenêtre ouverte
    start "ThreadX Dash" powershell -NoExit -ExecutionPolicy Bypass -Command "%PS_CMD_SetLocation% & '.\.venv\Scripts\Activate.ps1'; %PS_CMD_SetEnv% %PS_CMD_Info% python -m apps.dash_app"
) ELSE (
    echo Virtualenv not found — launching system Python in PowerShell
    start "ThreadX Dash" powershell -NoExit -ExecutionPolicy Bypass -Command "%PS_CMD_SetLocation% %PS_CMD_SetEnv% %PS_CMD_Info% python -m apps.dash_app"
)

REM Revenir au dossier précédent
popd

