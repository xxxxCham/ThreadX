<#
.SYNOPSIS
 Script de configuration pour exécution dans Codex / CI Windows / PowerShell.

.DESCRIPTION
 - Crée un environnement virtuel
 - Adapte requirements en supprimant les suffixes CUDA (+cu121)
 - Installe dépendances (fallback CPU si besoin)
 - Installe le package en mode editable
 - Affiche résumé des libs critiques
#>
[CmdletBinding()]
param(
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'

Write-Host '[1/7] Version Python:' -ForegroundColor Cyan
& $Python --version

$venv = '.venv'
if (-not (Test-Path $venv)) {
    Write-Host '[2/7] Création venv' -ForegroundColor Cyan
    & $Python -m venv $venv
}
else {
    Write-Host '[2/7] venv déjà présent' -ForegroundColor Yellow
}

# Activer venv
$activate = Join-Path $venv 'Scripts' 'Activate.ps1'
. $activate

Write-Host '[3/7] Pré-traitement requirements (suppression +cuXXXX)' -ForegroundColor Cyan
$tmpReq = New-TemporaryFile
(Get-Content requirements.txt) -replace '\+cu\d+', '' | Set-Content $tmpReq

Write-Host '[4/7] Installation des dépendances' -ForegroundColor Cyan
pip install --upgrade pip
try {
    pip install -r $tmpReq
}
catch {
    Write-Warning 'Echec installation directe, fallback CPU pour torch*'
    $content = Get-Content $tmpReq | Where-Object { $_ -notmatch '^torch==|^torchvision==|^torchaudio==' }
    $content | Set-Content $tmpReq
    pip install -r $tmpReq
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

Write-Host '[5/7] Installation du package local (editable)' -ForegroundColor Cyan
pip install -e .

Write-Host '[6/7] Résumé imports' -ForegroundColor Cyan
$modules = @('torch', 'pandas', 'numpy', 'sklearn', 'streamlit')
foreach ($m in $modules) {
    try {
        $code = @"
import importlib, sys
name = '$m'
try:
    mod = importlib.import_module(name)
    print(f'OK: {name} version=' + getattr(mod,'__version__','?'))
except Exception as e:
    print(f'FAIL: {name} -> {e}')
"@
        python -c $code
    }
    catch {
        Write-Warning "Module $m non importable (erreur PowerShell)"
    }
}

Write-Host '[7/7] Terminé.' -ForegroundColor Green
