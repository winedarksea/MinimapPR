# MinimapPR bootstrapper for Windows.
# Usage:
#   irm https://minimappr.com/install.ps1 | iex
#   # or, to skip the full extras:
#   $env:MINIMAPPR_INSTALL_BASE = "1"; irm https://minimappr.com/install.ps1 | iex

$ErrorActionPreference = "Stop"

function Write-Info($msg) {
    Write-Host "==> $msg"
}

$extra = "full"
if ($env:MINIMAPPR_INSTALL_BASE -eq "1") {
    $extra = ""
}
$packageSpec = "minimappr"
if ($extra -ne "") {
    $packageSpec = "minimappr[$extra]"
}

# ---------------------------------------------------------------------------
# 1. Install uv if missing
# ---------------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Info "uv not found, installing..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

$uvBinDir = Join-Path $env:USERPROFILE ".local\bin"
if ($env:Path -notlike "*$uvBinDir*") {
    $env:Path = "$uvBinDir;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv installation failed or uv is not on PATH"
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Install MinimapPR
# ---------------------------------------------------------------------------
Write-Info "Installing $packageSpec via uv tool install (this may take a while for the full extras)..."
uv tool install $packageSpec

$toolBinDir = $uvBinDir
try {
    $dir = (uv tool dir --bin 2>$null)
    if ($dir) { $toolBinDir = $dir.Trim() }
} catch {}
if ($env:Path -notlike "*$toolBinDir*") {
    $env:Path = "$toolBinDir;$env:Path"
}

$minimapprExe = Join-Path $toolBinDir "minimappr.exe"
if (-not (Test-Path $minimapprExe)) {
    $cmd = Get-Command minimappr -ErrorAction SilentlyContinue
    if ($cmd) { $minimapprExe = $cmd.Source }
}
if (-not (Test-Path $minimapprExe)) {
    Write-Warning "minimappr was installed but was not found at $minimapprExe. Add $toolBinDir to PATH or restart your shell."
}

# ---------------------------------------------------------------------------
# 3. Optional shortcut
# ---------------------------------------------------------------------------
$addShortcut = Read-Host "Add a Start Menu / Desktop shortcut? [y/N]"
if ($addShortcut -match '^[Yy]') {
    $shell = New-Object -ComObject WScript.Shell
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktopPath "MinimapPR.lnk"
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "cmd.exe"
    $shortcut.Arguments = "/c start `"`" `"$minimapprExe`" && timeout /t 2 >nul && start http://127.0.0.1:8080"
    $shortcut.WorkingDirectory = Split-Path $minimapprExe
    $shortcut.Description = "MinimapPR - realtime environmental awareness"
    $shortcut.Save()
    Write-Info "Wrote $shortcutPath"
}

# ---------------------------------------------------------------------------
# 4. Optional launch now
# ---------------------------------------------------------------------------
$launchNow = Read-Host "Launch MinimapPR now? [y/N]"
if ($launchNow -match '^[Yy]') {
    Write-Info "Starting MinimapPR..."
    Start-Process -FilePath $minimapprExe
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:8080"
    Write-Info "MinimapPR is running at http://127.0.0.1:8080"
}

# ---------------------------------------------------------------------------
# 5. Next steps
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "MinimapPR is installed."
Write-Host ""
Write-Host "  Run:        minimappr"
Write-Host "  UI:         http://127.0.0.1:8080"
Write-Host "  Uninstall:  uv tool uninstall minimappr"
Write-Host ""
