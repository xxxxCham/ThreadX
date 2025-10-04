# ThreadX Tkinter Application Launcher
# ===================================
# 
# Script PowerShell pour lancer l'application Tkinter ThreadX sur Windows
# Gère automatiquement l'environnement virtuel et les dépendances
#
# Usage: 
#   .\run_tkinter.ps1
#   .\run_tkinter.ps1 -Debug
#
# Author: ThreadX Framework
# Version: 1.0

param(
    [switch]$Debug,
    [switch]$Help
)

# Configuration des couleurs
$ColorConfig = @{
    Success = "Green"
    Error   = "Red"
    Warning = "Yellow"
    Info    = "Cyan"
    Header  = "Magenta"
}

function Write-ColorMessage {
    param(
        [string]$Message,
        [string]$Color = "White",
        [string]$Icon = ""
    )
    if ($Icon) {
        Write-Host "$Icon " -ForegroundColor $Color -NoNewline
    }
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 50 -ForegroundColor $ColorConfig.Header
    Write-Host "  $Title" -ForegroundColor $ColorConfig.Header
    Write-Host "=" * 50 -ForegroundColor $ColorConfig.Header
    Write-Host ""
}

function Test-PathExists {
    param(
        [string]$Path,
        [string]$Description
    )
    if (Test-Path $Path) {
        Write-ColorMessage "✅ $Description trouvé: $Path" $ColorConfig.Success
        return $true
    }
    else {
        Write-ColorMessage "❌ $Description non trouvé: $Path" $ColorConfig.Error
        return $false
    }
}

function Show-Help {
    Write-Header "ThreadX Tkinter Launcher - Aide"
    Write-Host "Usage:"
    Write-Host "  .\run_tkinter.ps1              # Lancement normal" -ForegroundColor White
    Write-Host "  .\run_tkinter.ps1 -Debug       # Lancement avec mode debug" -ForegroundColor White
    Write-Host "  .\run_tkinter.ps1 -Help        # Afficher cette aide" -ForegroundColor White
    Write-Host ""
    Write-Host "Description:"
    Write-Host "  Lance l'application Tkinter ThreadX avec vérifications automatiques" -ForegroundColor Gray
    Write-Host "  - Vérifie l'environnement virtuel Python" -ForegroundColor Gray
    Write-Host "  - Installe ThreadX si nécessaire" -ForegroundColor Gray
    Write-Host "  - Vérifie les dépendances critiques" -ForegroundColor Gray
    Write-Host "  - Test l'authentification API" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

# Afficher l'aide si demandée
if ($Help) {
    Show-Help
}

# Configuration des chemins
$ScriptDir = $PSScriptRoot
$ProjectRoot = $ScriptDir
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$TkinterApp = Join-Path $ProjectRoot "apps\tkinter\run_tkinter.py"

Write-Header "ThreadX Tkinter Application Launcher"

# Vérifier si on est dans le bon répertoire
$PyProjectPath = Join-Path $ProjectRoot "pyproject.toml"
if (-not (Test-Path $PyProjectPath)) {
    Write-ColorMessage "❌ Erreur: pyproject.toml non trouvé" $ColorConfig.Error
    Write-ColorMessage "   Assurez-vous d'être dans le répertoire racine ThreadX" $ColorConfig.Error
    Write-ColorMessage "   Répertoire actuel: $PWD" $ColorConfig.Warning
    Write-ColorMessage "   Répertoire attendu: $ProjectRoot" $ColorConfig.Warning
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

# Vérifications des prérequis
Write-ColorMessage "🔍 Vérification des prérequis..." $ColorConfig.Info

if (-not (Test-PathExists $VenvPath "Environnement virtuel")) {
    Write-ColorMessage "💡 Créez l'environnement virtuel:" $ColorConfig.Warning
    Write-ColorMessage "   python -m venv .venv" $ColorConfig.Warning
    Write-ColorMessage "   .venv\Scripts\activate" $ColorConfig.Warning
    Write-ColorMessage "   pip install -e ." $ColorConfig.Warning
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

if (-not (Test-PathExists $PythonExe "Python dans l'environnement virtuel")) {
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

if (-not (Test-PathExists $TkinterApp "Application Tkinter")) {
    Write-ColorMessage "💡 Vérifiez que le fichier apps\tkinter\run_tkinter.py existe" $ColorConfig.Warning
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

Write-Host ""

# Vérifier la version de Python
Write-ColorMessage "🐍 Vérification de Python..." $ColorConfig.Info
try {
    $PythonVersion = & $PythonExe --version
    Write-ColorMessage "✅ $PythonVersion" $ColorConfig.Success
}
catch {
    Write-ColorMessage "❌ Erreur lors de la vérification de Python" $ColorConfig.Error
    Write-ColorMessage $_.Exception.Message $ColorConfig.Error
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

Write-Host ""

# Vérifier l'installation de ThreadX
Write-ColorMessage "🔍 Vérification de l'installation ThreadX..." $ColorConfig.Info
try {
    $ThreadXCheck = & $PythonExe -c "import threadx; print('✅ ThreadX version:', threadx.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorMessage $ThreadXCheck $ColorConfig.Success
    }
    else {
        throw "ThreadX non installé"
    }
}
catch {
    Write-ColorMessage "⚠️ ThreadX non installé ou non trouvé" $ColorConfig.Warning
    Write-ColorMessage "💡 Installation de ThreadX..." $ColorConfig.Warning
    
    try {
        & $PythonExe -m pip install -e .
        if ($LASTEXITCODE -ne 0) {
            throw "Échec de l'installation"
        }
        Write-ColorMessage "✅ ThreadX installé avec succès" $ColorConfig.Success
    }
    catch {
        Write-ColorMessage "❌ Échec de l'installation de ThreadX" $ColorConfig.Error
        Read-Host "Appuyez sur Entrée pour quitter"
        exit 1
    }
}

Write-Host ""

# Vérifier les dépendances critiques
Write-ColorMessage "📦 Vérification des dépendances..." $ColorConfig.Info

$Dependencies = @(
    @{Name = "Tkinter"; ImportTest = "import tkinter; print('✅ Tkinter disponible')"; Required = $true },
    @{Name = "Pandas"; ImportTest = "import pandas; print('✅ Pandas disponible')"; Required = $false },
    @{Name = "NumPy"; ImportTest = "import numpy; print('✅ NumPy disponible')"; Required = $false }
)

foreach ($Dep in $Dependencies) {
    try {
        $Result = & $PythonExe -c $Dep.ImportTest 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorMessage $Result $ColorConfig.Success
        }
        else {
            throw "Import failed"
        }
    }
    catch {
        if ($Dep.Required) {
            Write-ColorMessage "❌ $($Dep.Name) non disponible" $ColorConfig.Error
            if ($Dep.Name -eq "Tkinter") {
                Write-ColorMessage "   Tkinter est normalement inclus avec Python" $ColorConfig.Error
                Write-ColorMessage "💡 Réinstallez Python avec Tkinter ou installez python-tk" $ColorConfig.Warning
            }
            Read-Host "Appuyez sur Entrée pour quitter"
            exit 1
        }
        else {
            Write-ColorMessage "⚠️ $($Dep.Name) non trouvé, installation..." $ColorConfig.Warning
            try {
                & $PythonExe -m pip install $Dep.Name.ToLower()
                Write-ColorMessage "✅ $($Dep.Name) installé" $ColorConfig.Success
            }
            catch {
                Write-ColorMessage "❌ Échec installation $($Dep.Name)" $ColorConfig.Error
            }
        }
    }
}

Write-Host ""

# Vérifier les variables d'environnement (optionnel)
Write-ColorMessage "🔐 Vérification de l'authentification API..." $ColorConfig.Info

if ($env:BINANCE_API_KEY) {
    Write-ColorMessage "✅ BINANCE_API_KEY définie" $ColorConfig.Success
}
else {
    Write-ColorMessage "⚠️ BINANCE_API_KEY non définie (endpoints publics utilisés)" $ColorConfig.Warning
}

if ($env:COINGECKO_API_KEY) {
    Write-ColorMessage "✅ COINGECKO_API_KEY définie" $ColorConfig.Success
}
else {
    Write-ColorMessage "⚠️ COINGECKO_API_KEY non définie (limites publiques)" $ColorConfig.Warning
}

# Test d'authentification si disponible
$TestAuthPath = Join-Path $ProjectRoot "test_auth.py"
if (Test-Path $TestAuthPath) {
    Write-ColorMessage "🧪 Test rapide d'authentification..." $ColorConfig.Info
    try {
        & $PythonExe $TestAuthPath --test-only 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorMessage "⚠️ Problèmes d'authentification détectés" $ColorConfig.Warning
            Write-ColorMessage "💡 Exécutez test_auth.py pour plus de détails" $ColorConfig.Warning
        }
    }
    catch {
        Write-ColorMessage "⚠️ Impossible de tester l'authentification" $ColorConfig.Warning
    }
}

Write-Host ""

# Arguments de ligne de commande
$AppArgs = @()
if ($Debug) {
    $AppArgs += "--debug"
    Write-ColorMessage "🐛 Mode debug activé" $ColorConfig.Warning
}

# Changer vers le répertoire du projet
Set-Location $ProjectRoot

# Lancer l'application Tkinter
Write-ColorMessage "🚀 Lancement de l'application ThreadX Tkinter..." $ColorConfig.Success
Write-ColorMessage "   Python: $PythonExe" $ColorConfig.Info
Write-ColorMessage "   Script: $TkinterApp" $ColorConfig.Info
if ($AppArgs.Count -gt 0) {
    Write-ColorMessage "   Arguments: $($AppArgs -join ' ')" $ColorConfig.Info
}
Write-Host ""

# Afficher les instructions d'utilisation
Write-ColorMessage "💡 Instructions:" $ColorConfig.Warning
Write-ColorMessage "   - L'application va se lancer dans une nouvelle fenêtre" $ColorConfig.Warning
Write-ColorMessage "   - Fermez cette console pour arrêter l'application" $ColorConfig.Warning
Write-ColorMessage "   - Logs visibles dans la console" $ColorConfig.Warning
Write-Host ""

# Attendre un peu pour que l'utilisateur lise les messages
Start-Sleep -Seconds 2

# Lancer l'application
try {
    if ($AppArgs.Count -gt 0) {
        & $PythonExe $TkinterApp $AppArgs
    }
    else {
        & $PythonExe $TkinterApp
    }
    
    $ExitCode = $LASTEXITCODE
    
    Write-Host ""
    if ($ExitCode -ne 0) {
        Write-ColorMessage "❌ L'application s'est fermée avec une erreur (code: $ExitCode)" $ColorConfig.Error
        Write-ColorMessage "💡 Vérifiez les messages d'erreur ci-dessus" $ColorConfig.Warning
        Write-ColorMessage "💡 Pour plus de détails, exécutez:" $ColorConfig.Warning
        Write-ColorMessage "   .\run_tkinter.ps1 -Debug" $ColorConfig.Warning
    }
    else {
        Write-ColorMessage "✅ Application fermée normalement" $ColorConfig.Success
    }
}
catch {
    Write-ColorMessage "❌ Erreur lors du lancement de l'application" $ColorConfig.Error
    Write-ColorMessage $_.Exception.Message $ColorConfig.Error
}

Write-Host ""
Write-ColorMessage "🔗 Liens utiles:" $ColorConfig.Info
Write-ColorMessage "   - Documentation: README.md" $ColorConfig.Info
Write-ColorMessage "   - Configuration API: .env.example" $ColorConfig.Info
Write-ColorMessage "   - Test authentification: test_auth.py" $ColorConfig.Info
Write-ColorMessage "   - Configuration env: setup_env.ps1" $ColorConfig.Info

Write-Host ""
Read-Host "Appuyez sur Entrée pour quitter"