# Script PowerShell - Synchronisation automatique ThreadX
# Télécharge les données depuis le 1er janvier 2025 jusqu'à hier

$ErrorActionPreference = "Stop"

Write-Host "`n🚀 ThreadX - Synchronisation Automatique des Données" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

# Vérifier Python
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "❌ Environnement virtuel non trouvé" -ForegroundColor Red
    exit 1
}

# Définir PYTHONPATH
$env:PYTHONPATH = "$PWD\src"

Write-Host "📊 Démarrage de la synchronisation..." -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

# Exécuter le script
& .venv\Scripts\python.exe scripts\sync_data_2025.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Synchronisation terminée avec succès!" -ForegroundColor Green
}
else {
    Write-Host "`n⚠️  Synchronisation terminée avec des erreurs" -ForegroundColor Yellow
}

# Afficher les statistiques des fichiers
Write-Host "`n📁 Statistiques des données locales:" -ForegroundColor Cyan

$data_paths = @(
    "data\raw\1m",
    "data\processed"
)

foreach ($path in $data_paths) {
    if (Test-Path $path) {
        $count = (Get-ChildItem -Path $path -Recurse -File -Filter "*.parquet" -ErrorAction SilentlyContinue).Count
        $size = (Get-ChildItem -Path $path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB

        Write-Host "  $path" -ForegroundColor White
        Write-Host "    - Fichiers: $count" -ForegroundColor Gray
        Write-Host "    - Taille: $([math]::Round($size, 2)) MB" -ForegroundColor Gray
    }
}

Write-Host "`n🎉 Terminé!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
