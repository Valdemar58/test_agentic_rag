# Build the tessa-export image and save it for the customer (Windows PowerShell).
# ASCII only: Windows PowerShell 5.1 reads scripts without BOM in the ANSI code page,
# so non-ASCII characters break parsing.
# Usage: powershell -File tools/tessa_export/build_image.ps1 [-Version 0.1.6]
# TESSA_SDK_PATH and CARD_SERVICE_PATH come from the environment or from .env in the repo root.
param([string]$Version = "0.1.6")

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.*)\s*$') {
            $name = $Matches[1]
            $value = $Matches[2].Trim('"')
            if (-not [Environment]::GetEnvironmentVariable($name)) {
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }
}
if (-not $env:TESSA_SDK_PATH -or -not $env:CARD_SERVICE_PATH) {
    throw "Set TESSA_SDK_PATH and CARD_SERVICE_PATH (environment variables or .env)"
}

$image = "tessa-export:$Version"
docker build -f tools/tessa_export/Dockerfile `
    --build-context "tessa_sdk=$env:TESSA_SDK_PATH" `
    --build-context "card_service=$env:CARD_SERVICE_PATH" `
    -t $image .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

$dist = Join-Path $root "dist\tessa-export-$Version"
New-Item -ItemType Directory -Force $dist | Out-Null
docker save $image -o (Join-Path $dist "tessa-export-$Version.tar")
if ($LASTEXITCODE -ne 0) { throw "docker save failed" }
Copy-Item tools/tessa_export/config.example.yaml (Join-Path $dist "config.yaml")
Copy-Item tools/tessa_export/seed_cards.yaml $dist
Copy-Item tools/tessa_export/README.md (Join-Path $dist "README.md")
Copy-Item tools/tessa_export/tessa.env.example (Join-Path $dist "tessa.env")
Write-Output "Done: $dist"
Get-ChildItem $dist | Select-Object Name, Length
