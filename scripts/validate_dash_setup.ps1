# ThreadX Dash - Script de Validation Installation
# =================================================
# Vérifie que toutes les dépendances sont installées
# et que le layout Dash fonctionne correctement.
#
# Usage: .\validate_dash_setup.ps1

Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  ThreadX Dash - Validation Installation (PROMPT 4)   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Vérifier Python
Write-Host "1. Vérification Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ Python détecté: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "   ❌ Python non trouvé. Installer Python 3.10+." -ForegroundColor Red
    exit 1
}

# Vérifier dépendances Dash
Write-Host "`n2. Vérification dépendances Dash..." -ForegroundColor Yellow
$requiredPackages = @("dash", "dash-bootstrap-components", "plotly")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    $installed = python -c "import $($package.Replace('-','_')); print('OK')" 2>&1
    if ($installed -eq "OK") {
        Write-Host "   ✅ $package installé" -ForegroundColor Green
    }
    else {
        Write-Host "   ❌ $package manquant" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "`n   Installation des packages manquants..." -ForegroundColor Yellow
    pip install $missingPackages
}

# Vérifier fichiers créés (PROMPT 4)
Write-Host "`n3. Vérification fichiers PROMPT 4..." -ForegroundColor Yellow
$requiredFiles = @(
    "apps\dash_app.py",
    "src\threadx\ui\layout.py",
    "src\threadx\ui\__init__.py",
    "src\threadx\ui\components\__init__.py"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file existe" -ForegroundColor Green
    }
    else {
        Write-Host "   ❌ $file manquant" -ForegroundColor Red
    }
}

# Vérifier imports (layout sans erreur)
Write-Host "`n4. Vérification imports layout..." -ForegroundColor Yellow
$importTest = python -c "from threadx.ui.layout import create_layout; print('OK')" 2>&1
if ($importTest -eq "OK") {
    Write-Host "   ✅ Import create_layout OK" -ForegroundColor Green
}
else {
    Write-Host "   ❌ Erreur import: $importTest" -ForegroundColor Red
}

# Vérifier port disponible
Write-Host "`n5. Vérification port Dash..." -ForegroundColor Yellow
$port = if ($env:THREADX_DASH_PORT) { $env:THREADX_DASH_PORT } else { 8050 }
$portInUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue

if ($portInUse) {
    Write-Host "   ⚠️  Port $port déjà utilisé. Changer THREADX_DASH_PORT." -ForegroundColor Yellow
}
else {
    Write-Host "   ✅ Port $port disponible" -ForegroundColor Green
}

# Récapitulatif
Write-Host "`n╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              VALIDATION TERMINÉE                       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "Pour lancer l'application:" -ForegroundColor White
Write-Host "   python apps\dash_app.py`n" -ForegroundColor Yellow

Write-Host "Accès UI:" -ForegroundColor White
Write-Host "   http://127.0.0.1:$port`n" -ForegroundColor Yellow
