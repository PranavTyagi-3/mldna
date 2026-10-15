# Builds a self-contained playground: a local git remote, a local DVC remote, a
# shared model store, a producer project and a consumer app. Nothing leaves
# your machine.   Usage: .\setup.ps1 [target-folder]   (default ~\mldna-quickstart)
param([string]$Target = (Join-Path $HOME "mldna-quickstart"))
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($args -join ' ')" }
}

$Here = $PSScriptRoot
if (Test-Path $Target) {
    throw "$Target already exists. Remove it, or pass another folder: .\setup.ps1 <folder>"
}

New-Item -ItemType Directory -Force (Join-Path $Target "remotes\dvc-store") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Target "model-store") | Out-Null
$Target = (Resolve-Path $Target).Path
$Producer = Join-Path $Target "producer"
$Consumer = Join-Path $Target "consumer"
Invoke-Checked git init -q --bare (Join-Path $Target "remotes\producer.git")

Invoke-Checked mldna init churn --dir $Producer --namespace quickstart `
    --tracking-source filesystem --filesystem-path (Join-Path $Target "model-store") `
    --dvc-backend local --dvc-local-path (Join-Path $Target "remotes\dvc-store")
Copy-Item (Join-Path $Here "producer\train.py") $Producer
Copy-Item (Join-Path $Here "producer\dataset\customers.csv") (Join-Path $Producer "dataset")
Add-Content -Encoding utf8 (Join-Path $Producer "mldna.yaml") "tracking:`n  key_metric: accuracy"
Invoke-Checked git -C $Producer remote add origin (Join-Path $Target "remotes\producer.git")

Invoke-Checked mldna init --consumer --dir $Consumer `
    --tracking-source filesystem --filesystem-path (Join-Path $Target "model-store")
Copy-Item (Join-Path $Here "consumer\app.py") $Consumer

Write-Host ""
Write-Host "Ready in $Target - next: cd `"$Producer`" and follow step 3 of the README."
