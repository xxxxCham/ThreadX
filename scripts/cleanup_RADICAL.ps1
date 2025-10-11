# Script PowerShell - VRAI Nettoyage RADICAL ThreadX
# Supprime DÉFINITIVEMENT tout ce qui est redondant (pas de déplacement, SUPPRESSION)

$ErrorActionPreference = "Stop"

Write-Host "`n🔥 ThreadX - NETTOYAGE RADICAL (Suppression définitive)" -ForegroundColor Red
Write-Host "="*80 -ForegroundColor Red
Write-Host "⚠️  Ce script SUPPRIME des fichiers (pas de backup)" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Êtes-vous sûr de vouloir continuer? (oui/non)"
if ($response -ne "oui") {
    Write-Host "`n❌ Annulé" -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# Compteurs
$deleted_files = 0
$deleted_dirs = 0
$errors = 0

# Fonction pour supprimer
function Remove-Item-Safe {
    param(
        [string]$Path,
        [string]$Description
    )

    if (Test-Path $Path) {
        try {
            Write-Host "  🗑️  Suppression: $Description" -ForegroundColor Gray
            Remove-Item -Path $Path -Recurse -Force
            if ((Get-Item $Path -ErrorAction SilentlyContinue) -is [System.IO.DirectoryInfo]) {
                $script:deleted_dirs++
            }
            else {
                $script:deleted_files++
            }
            Write-Host "    ✅ OK" -ForegroundColor Green
        }
        catch {
            Write-Host "    ❌ Erreur: $_" -ForegroundColor Red
            $script:errors++
        }
    }
    else {
        Write-Host "  ⏭️  Déjà supprimé: $Description" -ForegroundColor DarkGray
    }
}

# =========================================================
# PHASE 1: Supprimer apps/data_manager/ (TOUT LE DOSSIER)
# =========================================================
Write-Host "`n🔥 PHASE 1: apps/data_manager/ (redondant avec src/threadx/data/)" -ForegroundColor Red
Write-Host "-"*80

Remove-Item-Safe -Path "apps\data_manager" `
    -Description "apps/data_manager/ (TOUT le dossier - redondant)"

# =========================================================
# PHASE 2: Supprimer scripts PowerShell redondants
# =========================================================
Write-Host "`n🔥 PHASE 2: Scripts PowerShell redondants" -ForegroundColor Red
Write-Host "-"*80

Remove-Item-Safe -Path "scripts\run_data_sync.ps1" `
    -Description "run_data_sync.ps1 (utilise auto_data_sync.py qui n'existe plus)"

Remove-Item-Safe -Path "scripts\rename_indicators_to_usdc.ps1" `
    -Description "rename_indicators_to_usdc.ps1 (migration terminée)"

# =========================================================
# PHASE 3: Supprimer rapports d'analyse redondants
# =========================================================
Write-Host "`n🔥 PHASE 3: Rapports d'analyse (garder 1 seul)" -ForegroundColor Red
Write-Host "-"*80

# Garder uniquement RAPPORT_NETTOYAGE_FINAL.md, supprimer les autres
Remove-Item-Safe -Path "ANALYSE_EVOLUTION_DATA_MANAGEMENT.md" `
    -Description "ANALYSE_EVOLUTION_DATA_MANAGEMENT.md (redondant)"

Remove-Item-Safe -Path "SYNTHESE_NETTOYAGE_DATA.md" `
    -Description "SYNTHESE_NETTOYAGE_DATA.md (redondant)"

Remove-Item-Safe -Path "ANALYSE_REDONDANCE_TOKENS.md" `
    -Description "ANALYSE_REDONDANCE_TOKENS.md (info dans RAPPORT_NETTOYAGE_FINAL)"

# =========================================================
# PHASE 4: Supprimer fichiers de test/validation obsolètes
# =========================================================
Write-Host "`n🔥 PHASE 4: Fichiers test/validation obsolètes" -ForegroundColor Red
Write-Host "-"*80

Remove-Item-Safe -Path "test_consolidated_modules.py" `
    -Description "test_consolidated_modules.py (tests dans tests/)"

Remove-Item-Safe -Path "test_imports_directs.py" `
    -Description "test_imports_directs.py (tests dans tests/)"

Remove-Item-Safe -Path "test_parquet_reading.py" `
    -Description "test_parquet_reading.py (tests dans tests/)"

Remove-Item-Safe -Path "test_paths_usage.py" `
    -Description "test_paths_usage.py (tests dans tests/)"

Remove-Item-Safe -Path "validate_paths.py" `
    -Description "validate_paths.py (validation dans tests/)"

Remove-Item-Safe -Path "demo_unified_functions.py" `
    -Description "demo_unified_functions.py (démo obsolète)"

# =========================================================
# PHASE 5: Supprimer fichiers legacy racine
# =========================================================
Write-Host "`n🔥 PHASE 5: Fichiers legacy à la racine" -ForegroundColor Red
Write-Host "-"*80

Remove-Item-Safe -Path "launch_data_manager.py" `
    -Description "launch_data_manager.py (apps/data_manager supprimé)"

Remove-Item-Safe -Path "run_threadx.py" `
    -Description "run_threadx.py (legacy, utiliser scripts/)"

Remove-Item-Safe -Path "generate_example_paths.py" `
    -Description "generate_example_paths.py (utilitaire temporaire)"

Remove-Item-Safe -Path "paths_generated.txt" `
    -Description "paths_generated.txt (fichier généré temporaire)"

Remove-Item-Safe -Path "disable_type_checking.py" `
    -Description "disable_type_checking.py (utilitaire temporaire)"

# =========================================================
# PHASE 6: Supprimer notebooks/exploratory temporaires
# =========================================================
Write-Host "`n🔥 PHASE 6: Notebooks exploratory" -ForegroundColor Red
Write-Host "-"*80

Remove-Item-Safe -Path "explore_parquet_data.ipynb" `
    -Description "explore_parquet_data.ipynb (exploration temporaire)"

# =========================================================
# PHASE 7: Nettoyer scripts/ (garder seulement les 5 essentiels)
# =========================================================
Write-Host "`n🔥 PHASE 7: Scripts obsolètes dans scripts/" -ForegroundColor Red
Write-Host "-"*80

# Supprimer setup scripts (déjà executés)
Remove-Item-Safe -Path "scripts\codex_setup.ps1" `
    -Description "codex_setup.ps1 (setup terminé)"

Remove-Item-Safe -Path "scripts\codex_setup.sh" `
    -Description "codex_setup.sh (setup terminé)"

# =========================================================
# PHASE 8: Supprimer _deprecated_* (si l'utilisateur confirme)
# =========================================================
Write-Host "`n🔥 PHASE 8: Dossiers _deprecated_*" -ForegroundColor Red
Write-Host "-"*80

$deprecated_folders = Get-ChildItem -Directory -Filter "_deprecated_*"
if ($deprecated_folders.Count -gt 0) {
    Write-Host "Trouvé $($deprecated_folders.Count) dossier(s) deprecated" -ForegroundColor Yellow
    $response_dep = Read-Host "Supprimer les dossiers deprecated? (oui/non)"

    if ($response_dep -eq "oui") {
        foreach ($folder in $deprecated_folders) {
            Remove-Item-Safe -Path $folder.FullName `
                -Description "$($folder.Name)/ (backup deprecated)"
        }
    }
    else {
        Write-Host "  ⏭️  Dossiers deprecated conservés" -ForegroundColor Yellow
    }
}

# =========================================================
# RÉSUMÉ
# =========================================================
Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "📊 RÉSUMÉ DU NETTOYAGE RADICAL" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green

Write-Host "`n🗑️  Fichiers supprimés: $deleted_files" -ForegroundColor White
Write-Host "📁 Dossiers supprimés: $deleted_dirs" -ForegroundColor White
Write-Host "❌ Erreurs: $errors" -ForegroundColor $(if ($errors -gt 0) { "Red" } else { "Green" })

Write-Host "`n✅ FICHIERS CONSERVÉS (essentiels uniquement):" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 src/threadx/data/" -ForegroundColor White
Write-Host "   ✅ tokens.py        (TokenManager)" -ForegroundColor Gray
Write-Host "   ✅ loader.py        (BinanceDataLoader)" -ForegroundColor Gray
Write-Host "   ✅ ingest.py        (IngestionManager)" -ForegroundColor Gray
Write-Host "   ✅ io.py            (I/O unifié)" -ForegroundColor Gray

Write-Host "`n📁 scripts/" -ForegroundColor White
Write-Host "   ✅ sync_data_smart.py      (sync intelligent)" -ForegroundColor Gray
Write-Host "   ✅ sync_data_2025.py       (sync période)" -ForegroundColor Gray
Write-Host "   ✅ update_daily_tokens.py  (MAJ quotidienne)" -ForegroundColor Gray
Write-Host "   ✅ analyze_token.py        (analyse technique)" -ForegroundColor Gray
Write-Host "   ✅ scan_all_tokens.py      (scan multi-tokens)" -ForegroundColor Gray
Write-Host "   ✅ run_sync_smart.ps1      (wrapper PowerShell)" -ForegroundColor Gray
Write-Host "   ✅ run_sync_2025.ps1       (wrapper PowerShell)" -ForegroundColor Gray

Write-Host "`n📁 Racine/" -ForegroundColor White
Write-Host "   ✅ RAPPORT_NETTOYAGE_FINAL.md (documentation unique)" -ForegroundColor Gray
Write-Host "   ✅ README.md" -ForegroundColor Gray
Write-Host "   ✅ requirements.txt" -ForegroundColor Gray
Write-Host "   ✅ pyproject.toml" -ForegroundColor Gray

Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "✅ Nettoyage radical terminé!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
