# =====================================================================
# Script de Nettoyage Workspace ThreadX
# =====================================================================
# 
# Ce script :
# 1. Sauvegarde les anciennes configurations
# 2. Supprime les fichiers redondants
# 3. Valide que le workspace unique fonctionne
#
# Usage: .\cleanup_workspace.ps1
# =====================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🧹 NETTOYAGE WORKSPACE THREADX" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$rootPath = "d:\ThreadX"
$archivePath = Join-Path $rootPath ".archive\workspace_backup_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"

# =====================================================================
# ÉTAPE 1: Créer archive de sauvegarde
# =====================================================================
Write-Host "📦 ÉTAPE 1/4: Sauvegarde configurations existantes..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------------"

if (-not (Test-Path $archivePath)) {
    New-Item -ItemType Directory -Path $archivePath -Force | Out-Null
    Write-Host "✅ Dossier archive créé: $archivePath" -ForegroundColor Green
}

# Sauvegarder .vscode/settings.json
$vscodePath = Join-Path $rootPath ".vscode"
$settingsPath = Join-Path $vscodePath "settings.json"

if (Test-Path $settingsPath) {
    Copy-Item $settingsPath -Destination (Join-Path $archivePath "settings.json.bak") -Force
    Write-Host "✅ Sauvegardé: .vscode/settings.json" -ForegroundColor Green
}
else {
    Write-Host "ℹ️  Aucun .vscode/settings.json trouvé" -ForegroundColor Gray
}

# Sauvegarder configs/pyrightconfig.json
$pyrightPath = Join-Path $rootPath "configs\pyrightconfig.json"

if (Test-Path $pyrightPath) {
    Copy-Item $pyrightPath -Destination (Join-Path $archivePath "pyrightconfig.json.bak") -Force
    Write-Host "✅ Sauvegardé: configs/pyrightconfig.json" -ForegroundColor Green
}
else {
    Write-Host "ℹ️  Aucun pyrightconfig.json trouvé" -ForegroundColor Gray
}

Write-Host ""

# =====================================================================
# ÉTAPE 2: Analyser configurations redondantes
# =====================================================================
Write-Host "🔍 ÉTAPE 2/4: Analyse configurations redondantes..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------------"

$redundantFiles = @()

# Vérifier .vscode/settings.json
if (Test-Path $settingsPath) {
    $redundantFiles += @{
        Path   = $settingsPath
        Type   = "Settings VS Code (consolidé dans workspace)"
        Action = "Supprimer"
    }
}

# Vérifier configs/pyrightconfig.json
if (Test-Path $pyrightPath) {
    $redundantFiles += @{
        Path   = $pyrightPath
        Type   = "Pyright config (remplacé par workspace)"
        Action = "Supprimer"
    }
}

if ($redundantFiles.Count -eq 0) {
    Write-Host "✅ Aucune configuration redondante trouvée!" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Configurations redondantes détectées:" -ForegroundColor Yellow
    foreach ($file in $redundantFiles) {
        Write-Host "   - $($file.Path)" -ForegroundColor White
        Write-Host "     Type: $($file.Type)" -ForegroundColor Gray
    }
}

Write-Host ""

# =====================================================================
# ÉTAPE 3: Supprimer fichiers redondants
# =====================================================================
Write-Host "🗑️  ÉTAPE 3/4: Suppression fichiers redondants..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------------"

$deleted = 0

foreach ($file in $redundantFiles) {
    try {
        Remove-Item $file.Path -Force -ErrorAction Stop
        Write-Host "✅ Supprimé: $($file.Path)" -ForegroundColor Green
        $deleted++
    }
    catch {
        Write-Host "❌ Erreur suppression: $($file.Path)" -ForegroundColor Red
        Write-Host "   Raison: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Supprimer dossier .vscode s'il est vide
if (Test-Path $vscodePath) {
    $vscodeItems = Get-ChildItem $vscodePath -ErrorAction SilentlyContinue
    
    if ($vscodeItems.Count -eq 0) {
        Remove-Item $vscodePath -Force -Recurse -ErrorAction SilentlyContinue
        Write-Host "✅ Supprimé dossier vide: .vscode/" -ForegroundColor Green
        $deleted++
    }
    else {
        Write-Host "ℹ️  Dossier .vscode conservé (contient $($vscodeItems.Count) fichier(s))" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "📊 Résultat: $deleted fichier(s)/dossier(s) supprimé(s)" -ForegroundColor Cyan

Write-Host ""

# =====================================================================
# ÉTAPE 4: Valider workspace unique
# =====================================================================
Write-Host "✅ ÉTAPE 4/4: Validation workspace unique..." -ForegroundColor Yellow
Write-Host "-------------------------------------------------------------"

$workspacePath = Join-Path $rootPath "ThreadX.code-workspace"

if (Test-Path $workspacePath) {
    Write-Host "✅ Workspace unique trouvé: ThreadX.code-workspace" -ForegroundColor Green
    
    # Vérifier contenu JSON valide
    try {
        $workspaceContent = Get-Content $workspacePath -Raw | ConvertFrom-Json
        
        # Vérifier sections essentielles
        $hasSettings = $workspaceContent.settings -ne $null
        $hasLaunch = $workspaceContent.launch -ne $null
        $hasTasks = $workspaceContent.tasks -ne $null
        $hasFolders = $workspaceContent.folders -ne $null
        
        Write-Host ""
        Write-Host "📋 Contenu workspace:" -ForegroundColor White
        Write-Host "   - Settings: $(if($hasSettings){'✅'}else{'❌'})" -ForegroundColor $(if ($hasSettings) { 'Green' }else { 'Red' })
        Write-Host "   - Launch configs: $(if($hasLaunch){'✅'}else{'❌'})" -ForegroundColor $(if ($hasLaunch) { 'Green' }else { 'Red' })
        Write-Host "   - Tasks: $(if($hasTasks){'✅'}else{'❌'})" -ForegroundColor $(if ($hasTasks) { 'Green' }else { 'Red' })
        Write-Host "   - Folders: $(if($hasFolders){'✅'}else{'❌'})" -ForegroundColor $(if ($hasFolders) { 'Green' }else { 'Red' })
        
        if ($hasSettings -and $hasLaunch -and $hasTasks -and $hasFolders) {
            Write-Host ""
            Write-Host "✅ Workspace complet et valide!" -ForegroundColor Green
        }
        else {
            Write-Host ""
            Write-Host "⚠️  Workspace incomplet (sections manquantes)" -ForegroundColor Yellow
        }
        
    }
    catch {
        Write-Host "❌ Erreur lecture workspace: JSON invalide" -ForegroundColor Red
    }
    
}
else {
    Write-Host "❌ Workspace ThreadX.code-workspace introuvable!" -ForegroundColor Red
}

Write-Host ""

# =====================================================================
# RÉSUMÉ FINAL
# =====================================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📊 RÉSUMÉ NETTOYAGE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "✅ Sauvegarde: $archivePath" -ForegroundColor Green
Write-Host "✅ Fichiers supprimés: $deleted" -ForegroundColor Green
Write-Host "✅ Workspace unique: ThreadX.code-workspace" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎯 PROCHAINES ÉTAPES" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Fermer VS Code" -ForegroundColor White
Write-Host "2. Rouvrir avec workspace:" -ForegroundColor White
Write-Host "   code ThreadX.code-workspace" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Tester les configurations debug (F5)" -ForegroundColor White
Write-Host "4. Tester les tâches (Ctrl+Shift+P > Tasks)" -ForegroundColor White
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ NETTOYAGE TERMINÉ!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
