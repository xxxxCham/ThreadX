# Script PowerShell pour exécuter la synchronisation automatique
# Usage: .\scripts\run_data_sync.ps1

$ErrorActionPreference = "Stop"

Write-Host "🚀 ThreadX - Synchronisation Automatique des Données" -ForegroundColor Cyan
Write-Host "=" * 80

# Vérifier l'environnement virtuel
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Environnement virtuel non trouvé" -ForegroundColor Red
    Write-Host "Exécutez d'abord: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Définir PYTHONPATH
$env:PYTHONPATH = "$PWD\src"

# Exécuter la synchronisation
Write-Host "`n📊 Démarrage de la synchronisation..." -ForegroundColor Green
Write-Host "=" * 80

try {
    & .venv\Scripts\python.exe scripts\auto_data_sync.py

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Synchronisation terminée avec succès!" -ForegroundColor Green
    }
    else {
        Write-Host "`n⚠️  Synchronisation terminée avec des avertissements" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "`n❌ Erreur lors de la synchronisation: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n📁 Vérification des fichiers créés..." -ForegroundColor Yellow
$data_path = "data\crypto_data_parquet"

if (Test-Path $data_path) {
    $file_count = (Get-ChildItem -Path $data_path -Recurse -File -Filter "*.parquet").Count
    $total_size = (Get-ChildItem -Path $data_path -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB

    Write-Host "📊 Statistiques:" -ForegroundColor Cyan
    Write-Host "  - Fichiers Parquet: $file_count" -ForegroundColor White
    Write-Host "  - Taille totale: $([math]::Round($total_size, 2)) MB" -ForegroundColor White
}
else {
    Write-Host "⚠️  Aucun fichier de données trouvé" -ForegroundColor Yellow
}

Write-Host "`n🎉 Processus terminé!" -ForegroundColor Green
