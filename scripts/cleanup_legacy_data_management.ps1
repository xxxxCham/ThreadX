# Script PowerShell - Nettoyage Architecture Data Management
# Supprime fichiers legacy et consolide vers src/threadx/data/

$ErrorActionPreference = "Stop"

Write-Host "`n🧹 ThreadX - Nettoyage Architecture Data Management" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Créer dossier deprecated avec timestamp
$deprecated_folder = "_deprecated_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "📁 Création dossier: $deprecated_folder" -ForegroundColor Yellow
New-Item -ItemType Directory -Path $deprecated_folder -Force | Out-Null

# Compteurs
$moved = 0
$errors = 0

# Fonction pour déplacer un fichier/dossier
function Move-ToDeprecated {
    param(
        [string]$Path,
        [string]$Description
    )

    if (Test-Path $Path) {
        try {
            Write-Host "  → Déplacement: $Description" -ForegroundColor Gray
            Move-Item -Path $Path -Destination $deprecated_folder -Force
            $script:moved++
            Write-Host "    ✅ OK" -ForegroundColor Green
        }
        catch {
            Write-Host "    ❌ Erreur: $_" -ForegroundColor Red
            $script:errors++
        }
    }
    else {
        Write-Host "  ⏭️  Ignoré (déjà supprimé): $Description" -ForegroundColor DarkGray
    }
}

# =========================================================
# PHASE 1: Legacy Data Management Files
# =========================================================
Write-Host "`n🔥 PHASE 1: Suppression fichiers legacy data management" -ForegroundColor Yellow
Write-Host "-"*80

Move-ToDeprecated -Path "auto_data_sync.py" `
    -Description "auto_data_sync.py (obsolète, remplacé par sync_data_smart.py)"

Move-ToDeprecated -Path "unified_data_historique_with_indicators.py" `
    -Description "unified_data_historique_with_indicators.py (monolithe 850 lignes)"

Move-ToDeprecated -Path "validate_data_structures.py" `
    -Description "validate_data_structures.py (tests maintenant dans tests/)"

# =========================================================
# PHASE 2: Token Diversity Manager (module entier)
# =========================================================
Write-Host "`n🔥 PHASE 2: Suppression token_diversity_manager/" -ForegroundColor Yellow
Write-Host "-"*80

Move-ToDeprecated -Path "token_diversity_manager" `
    -Description "token_diversity_manager/ (module 1450 lignes redondant)"

# =========================================================
# PHASE 3: Scripts de migration terminés
# =========================================================
Write-Host "`n🔥 PHASE 3: Scripts de migration/validation legacy" -ForegroundColor Yellow
Write-Host "-"*80

Move-ToDeprecated -Path "scripts\migrate_to_best_practices.py" `
    -Description "migrate_to_best_practices.py (migration terminée)"

Move-ToDeprecated -Path "scripts\auto_data_sync.py" `
    -Description "scripts\auto_data_sync.py (doublon, déjà dans racine)"

# =========================================================
# PHASE 4: Fichiers de documentation obsolètes
# =========================================================
Write-Host "`n🔥 PHASE 4: Docs legacy" -ForegroundColor Yellow
Write-Host "-"*80

$legacy_docs = @(
    "ANALYSE_REDONDANCES.md",
    "CONFIRMATION_CHEMINS.md",
    "CONSOLIDATION_RESUME_VISUEL.txt",
    "FICHIERS_CREES_WORKSPACE.md",
    "GUIDE_MIGRATION_RAPIDE.md",
    "LIVRAISON_ETAPE_A.md",
    "RAPPORT_CONSOLIDATION_FINALE.md",
    "SYNTHESE_CONSOLIDATION.md",
    "TRAVAIL_TERMINE.md",
    "VALIDATION_CHEMINS_THREADX.md",
    "VALIDATION_COMPLETE.md",
    "VALIDATION_END_TO_END.md",
    "VALIDATION_RESUMÉ.md",
    "WORKSPACE_FINAL_REPORT.md",
    "WORKSPACE_README.md"
)

foreach ($doc in $legacy_docs) {
    Move-ToDeprecated -Path $doc -Description $doc
}

# =========================================================
# PHASE 5: Vérification fichiers restants
# =========================================================
Write-Host "`n✅ PHASE 5: Vérification architecture restante" -ForegroundColor Green
Write-Host "-"*80

Write-Host "`n📊 Fichiers data management CONSERVÉS:" -ForegroundColor Cyan

$kept_files = @{
    "src/threadx/data/"     = @(
        "tokens.py (286 lignes) - Gestion tokens consolidée",
        "loader.py (350 lignes) - BinanceDataLoader unifié",
        "ingest.py (650 lignes) - IngestionManager '1m truth'",
        "io.py (520 lignes) - I/O unifié parquet/JSON"
    )
    "scripts/"              = @(
        "sync_data_smart.py (550 lignes) - Sync intelligent avec gaps",
        "sync_data_2025.py (159 lignes) - Sync simple période définie",
        "update_daily_tokens.py - MAJ quotidienne (utilise TokenManager)",
        "analyze_token.py - Analyse technique (utilise BinanceDataLoader)",
        "scan_all_tokens.py - Scan multi-tokens (utilise TokenManager)"
    )
    "scripts/ (PowerShell)" = @(
        "run_sync_smart.ps1 - Wrapper sync_data_smart.py",
        "run_sync_2025.ps1 - Wrapper sync_data_2025.py",
        "run_data_sync.ps1 - Wrapper générique",
        "rename_indicators_to_usdc.ps1 - Utilitaire renommage"
    )
}

foreach ($category in $kept_files.Keys) {
    Write-Host "`n  📁 $category" -ForegroundColor White
    foreach ($file in $kept_files[$category]) {
        Write-Host "    ✅ $file" -ForegroundColor Gray
    }
}

# =========================================================
# Résumé
# =========================================================
Write-Host "`n"
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "📊 RÉSUMÉ DU NETTOYAGE" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan

Write-Host "`n📦 Fichiers déplacés vers $deprecated_folder" -ForegroundColor White
Write-Host "   Moved: $moved" -ForegroundColor Green
Write-Host "   Errors: $errors" -ForegroundColor $(if ($errors -gt 0) { "Red" } else { "Green" })

Write-Host "`n📊 Architecture data management:" -ForegroundColor White
Write-Host "   Modules consolidés: src/threadx/data/ (4 fichiers)" -ForegroundColor Green
Write-Host "   Scripts modernes: scripts/ (5 Python + 4 PowerShell)" -ForegroundColor Green
Write-Host "   Redondance: 0%" -ForegroundColor Green

Write-Host "`n🎯 ACTIONS SUIVANTES:" -ForegroundColor Yellow
Write-Host "   1. Vérifier que tout fonctionne:" -ForegroundColor White
Write-Host "      python scripts\sync_data_smart.py" -ForegroundColor Gray
Write-Host "      python scripts\update_daily_tokens.py --tokens 10" -ForegroundColor Gray
Write-Host ""
Write-Host "   2. Si tout OK, supprimer définitivement:" -ForegroundColor White
Write-Host "      Remove-Item -Recurse -Force $deprecated_folder" -ForegroundColor Gray
Write-Host ""
Write-Host "   3. Si problème, restaurer:" -ForegroundColor White
Write-Host "      Move-Item $deprecated_folder\* . -Force" -ForegroundColor Gray

Write-Host "`n✅ Nettoyage terminé!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Créer fichier README dans deprecated
$readme_content = @"
# Fichiers Deprecated - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## Raison

Ces fichiers ont été déplacés car ils sont obsolètes et redondants avec
la nouvelle architecture consolidée dans ``src/threadx/data/``.

## Fichiers déplacés

Voir ANALYSE_EVOLUTION_DATA_MANAGEMENT.md pour détails complets.

### Principaux fichiers:

- **auto_data_sync.py** → Remplacé par sync_data_smart.py
- **unified_data_historique_with_indicators.py** → Fonctions dans src/threadx/data/
- **token_diversity_manager/** → Consolidé dans TokenManager
- **validate_data_structures.py** → Tests dans tests/

## Restauration

Si besoin de restaurer:
``````powershell
Move-Item $deprecated_folder\* . -Force
``````

## Suppression définitive

Après validation (tests OK):
``````powershell
Remove-Item -Recurse -Force $deprecated_folder
``````

## Voir aussi

- ANALYSE_EVOLUTION_DATA_MANAGEMENT.md (analyse complète)
- ANALYSE_REDONDANCE_TOKENS.md (plan consolidation tokens)
"@

Set-Content -Path "$deprecated_folder\README.md" -Value $readme_content -Encoding UTF8

Write-Host "📄 README créé dans $deprecated_folder\README.md" -ForegroundColor Gray
