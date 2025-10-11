# Script PowerShell - Synchronisation INTELLIGENTE ThreadX
# Test avec BTCUSDC puis lancement complet si souhaité

$ErrorActionPreference = "Stop"

Write-Host "`n🧠 ThreadX - Synchronisation INTELLIGENTE des Données" -ForegroundColor Cyan
Write-Host "Utilise les données existantes au lieu de tout re-télécharger" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

# Vérifier Python
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Environnement virtuel non trouvé" -ForegroundColor Red
    exit 1
}

# Définir PYTHONPATH
$env:PYTHONPATH = "$PWD\src"

Write-Host "`n🧪 PHASE 1: Test avec BTCUSDC" -ForegroundColor Yellow
Write-Host "="*80 -ForegroundColor Yellow
Write-Host ""

# Lancer le test avec BTCUSDC
& .venv\Scripts\python.exe scripts\sync_data_smart.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Test échoué avec BTCUSDC" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Test BTCUSDC réussi!" -ForegroundColor Green
Write-Host ""

# Demander si on continue avec tous les symboles
$response = Read-Host "Voulez-vous synchroniser TOUS les symboles maintenant? (o/n)"

if ($response -eq "o" -or $response -eq "O" -or $response -eq "oui") {
    Write-Host "`n🚀 PHASE 2: Synchronisation complète" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor Green
    Write-Host ""

    & .venv\Scripts\python.exe scripts\sync_data_smart.py --full

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Synchronisation complète terminée!" -ForegroundColor Green
    }
    else {
        Write-Host "`n⚠️  Synchronisation terminée avec des erreurs" -ForegroundColor Yellow
    }
}
else {
    Write-Host "`n⏭️  Synchronisation complète annulée" -ForegroundColor Yellow
    Write-Host "💡 Pour lancer manuellement: python scripts\sync_data_smart.py --full" -ForegroundColor Cyan
}

Write-Host "`n🎉 Terminé!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
