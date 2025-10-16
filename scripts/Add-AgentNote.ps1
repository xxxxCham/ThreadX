param(
  [Parameter(Mandatory=$true)] [string]$Agent,
  [Parameter(Mandatory=$true)] [string]$Action,
  [string]$Scope = ""
)

$repoRoot = (Resolve-Path ".").Path
$journal = Join-Path $repoRoot "docs\sessions\Interventions_IA.txt"
if (-not (Test-Path $journal)) {
  Write-Host "Journal introuvable: $journal"; exit 1
}
$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$gitUser = (git config user.name 2>$null)
if ([string]::IsNullOrWhiteSpace($gitUser)) { $gitUser = $Agent }
$lastHash = (git rev-parse --short HEAD 2>$null)
if ([string]::IsNullOrWhiteSpace($lastHash)) { $lastHash = "no-git" }
$line = "{0} | Agent={1} | Modifs={2} | Scope={3} | Commit={4}" -f $ts, $gitUser, $Action, $Scope, $lastHash
Add-Content -Path $journal -Value $line -Encoding utf8
Write-Host "Ajouté: $line"
