# Script de renommage des dossiers d'indicateurs
# Ajoute USDC après chaque nom de token (sauf si déjà présent)

$ErrorActionPreference = "Stop"

Write-Host "`n🔄 Renommage des dossiers d'indicateurs vers format USDC" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

$base_path = "D:\ThreadX\data\indicators"

# Vérifier que le dossier existe
if (-not (Test-Path $base_path)) {
    Write-Host "❌ Dossier non trouvé: $base_path" -ForegroundColor Red
    exit 1
}

# Liste des dossiers qui ont déjà USDC (à ignorer)
$already_usdc = @(
    "XRPUSDC", "SOLUSDC", "BTCUSDC", "BNBUSDC",
    "ETHUSDC", "DOGEUSDC", "ADAUSDC"
)

# Compteurs
$total = 0
$renamed = 0
$skipped = 0
$errors = 0

# Scanner tous les dossiers
$folders = Get-ChildItem -Path $base_path -Directory | Sort-Object Name

Write-Host "`n📊 Analyse de $($folders.Count) dossiers..." -ForegroundColor Yellow
Write-Host ""

foreach ($folder in $folders) {
    $total++
    $old_name = $folder.Name

    # Ignorer si déjà USDC
    if ($already_usdc -contains $old_name) {
        Write-Host "⏭️  Ignoré (déjà USDC): $old_name" -ForegroundColor Gray
        $skipped++
        continue
    }

    # Créer le nouveau nom avec USDC
    $new_name = "${old_name}USDC"
    $old_path = $folder.FullName
    $new_path = Join-Path $base_path $new_name

    # Vérifier si le nouveau nom existe déjà
    if (Test-Path $new_path) {
        Write-Host "⚠️  Existe déjà: $old_name → $new_name" -ForegroundColor Yellow
        $skipped++
        continue
    }

    try {
        # Renommer le dossier
        Rename-Item -Path $old_path -NewName $new_name -Force
        Write-Host "✅ Renommé: $old_name → $new_name" -ForegroundColor Green
        $renamed++
    }
    catch {
        Write-Host "❌ Erreur: $old_name - $_" -ForegroundColor Red
        $errors++
    }
}

# Résumé
Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "📊 RÉSUMÉ DU RENOMMAGE" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "📁 Total analysé: $total" -ForegroundColor White
Write-Host "✅ Renommés: $renamed" -ForegroundColor Green
Write-Host "⏭️  Ignorés: $skipped" -ForegroundColor Yellow
Write-Host "❌ Erreurs: $errors" -ForegroundColor Red
Write-Host "="*80 -ForegroundColor Cyan

if ($renamed -gt 0) {
    Write-Host "`n🎉 Renommage terminé avec succès!" -ForegroundColor Green

    # Afficher quelques exemples
    Write-Host "`n📝 Exemples de dossiers renommés:" -ForegroundColor Cyan
    $examples = Get-ChildItem -Path $base_path -Directory -Filter "*USDC" |
    Select-Object -First 10 |
    ForEach-Object { $_.Name }

    foreach ($example in $examples) {
        Write-Host "  • $example" -ForegroundColor Gray
    }
}
else {
    Write-Host "`n⚠️  Aucun dossier n'a été renommé" -ForegroundColor Yellow
}

Write-Host ""
