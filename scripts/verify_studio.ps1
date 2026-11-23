# Cloth Swap Studio - local diagnostics on Windows PowerShell 5.1+
# May write Python caches, a diagnostic report and web UI build outputs.
# Does not install packages, download models, modify network/RDP or start rendering.
param([switch]$SkipUIBuild, [switch]$SkipPythonTests)

$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$report = Join-Path $root ('studio_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.txt')
Set-Location $root

function Write-Report([string]$msg) {
    Write-Host $msg
    Add-Content -LiteralPath $report -Value $msg -Encoding UTF8
}
function Run-Check([string]$label, [scriptblock]$command) {
    Write-Report ('==== ' + $label + ' ====')
    try {
        $all = & $command 2>&1 | Out-String
        Write-Report $all
        Write-Report ('ExitCode: ' + $LASTEXITCODE)
    } catch {
        Write-Report ('ERROR: ' + $_.Exception.Message)
    }
}

Write-Report ('Project root: ' + $root)
Write-Report ('Run date: ' + (Get-Date).ToString('o'))
Write-Report 'Safety: no model execution, installs or network setting changes.'

$python = Join-Path $root '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source } else { $python = $null }
}
if ($python) {
    Run-Check 'Python environment' { & $python --version }
    Run-Check 'Python syntax compile' { & $python -m compileall -q cloth_swap tests }
    if (!$SkipPythonTests) { Run-Check 'Python regression tests' { & $python -m pytest -q } }
} else { Write-Report 'Python was not found; skipping Python checks.' }

$gpuTool = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($gpuTool) {
    Run-Check 'GPU metadata (no GPU inference)' {
        & $gpuTool.Source --query-gpu=name,driver_version,memory.total,memory.used --format=csv
    }
} else { Write-Report 'nvidia-smi not found.' }

$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (!$npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if ($npm -and !$SkipUIBuild) {
    Push-Location (Join-Path $root 'studio-ui')
    try { Run-Check 'Local web UI build (no dependency installation)' { & $npm.Source run build } }
    finally { Pop-Location }
} else { Write-Report 'UI build skipped (npm missing or -SkipUIBuild).' }

Write-Report ('Report saved: ' + $report)
