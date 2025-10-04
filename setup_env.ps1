# ThreadX Environment Setup Script for Windows
# ==========================================
# 
# Script PowerShell pour configurer facilement les variables d'environnement ThreadX
# 
# Usage:
#   .\setup_env.ps1
#   .\setup_env.ps1 -Interactive
#   .\setup_env.ps1 -TestOnly

param(
    [switch]$Interactive,
    [switch]$TestOnly,
    [switch]$Help
)

function Show-Help {
    Write-Host "ThreadX Environment Setup Script" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Ce script aide à configurer les variables d'environnement pour ThreadX"
    Write-Host ""
    Write-Host "Paramètres:"
    Write-Host "  -Interactive    Mode interactif pour saisir les clés"
    Write-Host "  -TestOnly       Teste uniquement les variables existantes"
    Write-Host "  -Help           Affiche cette aide"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\setup_env.ps1               # Configuration avec .env"
    Write-Host "  .\setup_env.ps1 -Interactive  # Saisie interactive"
    Write-Host "  .\setup_env.ps1 -TestOnly     # Test uniquement"
    Write-Host ""
}

function Test-EnvironmentVariables {
    Write-Host "🔍 Vérification des variables d'environnement..." -ForegroundColor Yellow
    Write-Host ""
    
    $variables = @{
        'BINANCE_API_KEY' = 'Clé API Binance'
        'BINANCE_API_SECRET' = 'Secret API Binance'
        'COINGECKO_API_KEY' = 'Clé API CoinGecko (optionnelle)'
        'ALPHA_VANTAGE_API_KEY' = 'Clé API Alpha Vantage (optionnelle)'
        'POLYGON_API_KEY' = 'Clé API Polygon (optionnelle)'
    }
    
    $foundCount = 0
    $totalCount = $variables.Count
    
    foreach ($var in $variables.GetEnumerator()) {
        $value = [Environment]::GetEnvironmentVariable($var.Key, "User")
        if (-not $value) {
            $value = [Environment]::GetEnvironmentVariable($var.Key, "Process")
        }
        
        if ($value) {
            $maskedValue = if ($value.Length -gt 8) { 
                $value.Substring(0, 4) + "..." + $value.Substring($value.Length - 4) 
            } else { "***" }
            Write-Host "✅ $($var.Key.PadRight(25)): $maskedValue ($($var.Value))" -ForegroundColor Green
            $foundCount++
        } else {
            Write-Host "❌ $($var.Key.PadRight(25)): Non définie ($($var.Value))" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "📊 Résumé: $foundCount/$totalCount variables configurées" -ForegroundColor Cyan
    
    # Vérification spéciale Binance
    $binanceKey = [Environment]::GetEnvironmentVariable("BINANCE_API_KEY", "User")
    $binanceSecret = [Environment]::GetEnvironmentVariable("BINANCE_API_SECRET", "User")
    if (-not $binanceKey) { $binanceKey = [Environment]::GetEnvironmentVariable("BINANCE_API_KEY", "Process") }
    if (-not $binanceSecret) { $binanceSecret = [Environment]::GetEnvironmentVariable("BINANCE_API_SECRET", "Process") }
    
    if ($binanceKey -and $binanceSecret) {
        Write-Host "✅ Configuration Binance complète (clé + secret)" -ForegroundColor Green
    } elseif ($binanceKey -or $binanceSecret) {
        Write-Host "⚠️ Configuration Binance incomplète (clé OU secret manquant)" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Configuration Binance absente" -ForegroundColor Red
    }
    
    return $foundCount
}

function Set-EnvironmentVariable {
    param(
        [string]$Name,
        [string]$Value,
        [string]$Description
    )
    
    if ($Value -and $Value -ne "") {
        [Environment]::SetEnvironmentVariable($Name, $Value, "User")
        Write-Host "✅ $Name configuré" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠️ $Name ignoré (valeur vide)" -ForegroundColor Yellow
        return $false
    }
}

function Read-EnvFile {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ Fichier $FilePath non trouvé" -ForegroundColor Red
        return @{}
    }
    
    $envVars = @{}
    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line -split "=", 2
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            
            # Supprimer les guillemets si présents
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            
            # Ignorer les valeurs placeholder
            if (-not $value.Contains("your_") -and -not $value.Contains("here")) {
                $envVars[$key] = $value
            }
        }
    }
    
    return $envVars
}

function Set-InteractiveVariables {
    Write-Host "🔧 Configuration interactive des variables d'environnement" -ForegroundColor Cyan
    Write-Host ""
    
    $variables = @(
        @{ Name = "BINANCE_API_KEY"; Description = "Clé API Binance"; Required = $true }
        @{ Name = "BINANCE_API_SECRET"; Description = "Secret API Binance"; Required = $true }
        @{ Name = "COINGECKO_API_KEY"; Description = "Clé API CoinGecko (optionnelle)"; Required = $false }
        @{ Name = "ALPHA_VANTAGE_API_KEY"; Description = "Clé API Alpha Vantage (optionnelle)"; Required = $false }
        @{ Name = "POLYGON_API_KEY"; Description = "Clé API Polygon (optionnelle)"; Required = $false }
    )
    
    $setCount = 0
    
    foreach ($var in $variables) {
        Write-Host "🔑 $($var.Description)" -ForegroundColor White
        
        # Vérifier si la variable existe déjà
        $existing = [Environment]::GetEnvironmentVariable($var.Name, "User")
        if ($existing) {
            $masked = if ($existing.Length -gt 8) { 
                $existing.Substring(0, 4) + "..." + $existing.Substring($existing.Length - 4) 
            } else { "***" }
            Write-Host "   Valeur actuelle: $masked" -ForegroundColor Gray
        }
        
        $prompt = if ($var.Required) { 
            "   Entrez la nouvelle valeur (REQUIS)" 
        } else { 
            "   Entrez la nouvelle valeur (optionnel, Entrée pour ignorer)" 
        }
        
        do {
            $value = Read-Host $prompt
            
            if ($value -and $value -ne "") {
                if (Set-EnvironmentVariable -Name $var.Name -Value $value -Description $var.Description) {
                    $setCount++
                }
                break
            } elseif (-not $var.Required) {
                Write-Host "   ⚠️ Variable ignorée" -ForegroundColor Yellow
                break
            } else {
                Write-Host "   ❌ Cette variable est requise, veuillez entrer une valeur" -ForegroundColor Red
            }
        } while ($var.Required)
        
        Write-Host ""
    }
    
    return $setCount
}

function Test-ThreadXAuthentication {
    Write-Host "🧪 Test de l'authentification ThreadX..." -ForegroundColor Yellow
    
    if (Test-Path "test_auth.py") {
        try {
            & python test_auth.py
        } catch {
            Write-Host "❌ Erreur lors du test: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "💡 Assurez-vous que Python et ThreadX sont installés" -ForegroundColor Cyan
        }
    } else {
        Write-Host "⚠️ Script test_auth.py non trouvé - test ignoré" -ForegroundColor Yellow
    }
}

function Show-SecurityNotes {
    Write-Host ""
    Write-Host "🔒 NOTES DE SÉCURITÉ" -ForegroundColor Red
    Write-Host "===================" -ForegroundColor Red
    Write-Host ""
    Write-Host "✅ Bonnes pratiques:" -ForegroundColor Green
    Write-Host "  • Les variables sont stockées dans le profil utilisateur Windows"
    Write-Host "  • Ne partagez jamais vos clés API"
    Write-Host "  • Utilisez des clés avec permissions limitées"
    Write-Host "  • Surveillez l'utilisation de vos clés"
    Write-Host ""
    Write-Host "⚠️ Important:" -ForegroundColor Yellow
    Write-Host "  • Redémarrez votre terminal/IDE après configuration"
    Write-Host "  • Les variables sont persistantes pour cet utilisateur"
    Write-Host "  • Utilisez l'authentification IP quand possible"
    Write-Host ""
}

function Show-Links {
    Write-Host "🔗 LIENS UTILES" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Obtenir vos clés API:"
    Write-Host "  • Binance: https://www.binance.com/en/my/settings/api-management"
    Write-Host "  • CoinGecko: https://www.coingecko.com/en/api/pricing"
    Write-Host "  • Alpha Vantage: https://www.alphavantage.co/support/#api-key"
    Write-Host "  • Polygon: https://polygon.io/"
    Write-Host ""
    Write-Host "Documentation:"
    Write-Host "  • ThreadX: ./README.md"
    Write-Host "  • Configuration: ./.env.example"
    Write-Host ""
}

# === MAIN SCRIPT ===

if ($Help) {
    Show-Help
    exit
}

Write-Host "🔐 THREADX ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

if ($TestOnly) {
    $found = Test-EnvironmentVariables
    Test-ThreadXAuthentication
    exit
}

# Test initial
$initialFound = Test-EnvironmentVariables
Write-Host ""

if ($Interactive) {
    # Mode interactif
    $setCount = Set-InteractiveVariables
    Write-Host "✅ Configuration terminée - $setCount variables configurées" -ForegroundColor Green
} else {
    # Mode fichier .env
    Write-Host "📄 Recherche du fichier .env..." -ForegroundColor Yellow
    
    $envFile = ".env"
    if (-not (Test-Path $envFile)) {
        Write-Host "❌ Fichier .env non trouvé" -ForegroundColor Red
        Write-Host "💡 Créez un fichier .env à partir de .env.example" -ForegroundColor Cyan
        Write-Host "💡 Ou utilisez le mode interactif: .\setup_env.ps1 -Interactive" -ForegroundColor Cyan
        exit 1
    }
    
    $envVars = Read-EnvFile $envFile
    
    if ($envVars.Count -eq 0) {
        Write-Host "⚠️ Aucune variable valide trouvée dans .env" -ForegroundColor Yellow
        Write-Host "💡 Vérifiez que vos clés ne contiennent pas 'your_' ou 'here'" -ForegroundColor Cyan
        exit 1
    }
    
    Write-Host "✅ $($envVars.Count) variables trouvées dans .env" -ForegroundColor Green
    Write-Host ""
    
    $setCount = 0
    foreach ($var in $envVars.GetEnumerator()) {
        if (Set-EnvironmentVariable -Name $var.Key -Value $var.Value -Description "Variable depuis .env") {
            $setCount++
        }
    }
    
    Write-Host ""
    Write-Host "✅ Configuration terminée - $setCount variables configurées" -ForegroundColor Green
}

# Test final
Write-Host ""
Write-Host "🔍 Vérification finale..." -ForegroundColor Yellow
$finalFound = Test-EnvironmentVariables

# Test d'authentification
Write-Host ""
Test-ThreadXAuthentication

# Notes de sécurité et liens
Show-SecurityNotes
Show-Links

Write-Host "🎉 Configuration terminée!" -ForegroundColor Green
Write-Host "💡 Redémarrez votre terminal/IDE pour prendre en compte les changements" -ForegroundColor Cyan