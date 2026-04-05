param(
  [string]$Distro = "Ubuntu-24.04",
  [string]$LinuxVenvPath = "~/.venvs/quant-platform-rocm",
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$frontendRoot = Join-Path $repoRoot "frontend"
$backendScriptWindows = Join-Path $repoRoot "scripts\wsl_rocm\start_backend.sh"

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "wsl.exe is not available on this machine."
}

$availableDistros = @(
  wsl.exe -l -q |
    ForEach-Object { ($_ -replace "`0", "").Trim() } |
    Where-Object { $_ }
)
if (-not ($availableDistros -contains $Distro)) {
  throw "WSL distro '$Distro' was not found. Available distros: $($availableDistros -join ', ')"
}

$repoRootLinux = (wsl.exe -d $Distro -- wslpath -a $repoRoot).Trim()
$backendScriptLinux = (wsl.exe -d $Distro -- wslpath -a $backendScriptWindows).Trim()

$backendCommand = @"
cd '$repoRootLinux' && \
chmod +x '$backendScriptLinux' && \
QP_ROCM_VENV='$LinuxVenvPath' QUANT_PLATFORM_HOST='0.0.0.0' QUANT_PLATFORM_PORT='$BackendPort' '$backendScriptLinux'
"@

Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "wsl.exe -d $Distro -- bash -lc ""$backendCommand"""
) | Out-Null

if (-not $SkipFrontend) {
  $frontendCommand = @"
Set-Location '$frontendRoot'
`$env:VITE_API_BASE_URL='http://127.0.0.1:$BackendPort'
npm run dev -- --host 127.0.0.1 --port $FrontendPort
"@
  Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $frontendCommand
  ) | Out-Null
}

Write-Host "WSL ROCm launch requested."
Write-Host "Backend distro: $Distro"
Write-Host "Backend port: $BackendPort"
if (-not $SkipFrontend) {
  Write-Host "Frontend port: $FrontendPort"
  Write-Host "Frontend URL: http://127.0.0.1:$FrontendPort"
}
