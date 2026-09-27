# Whole pipeline on a small sample: download, preprocess, short CPU train, evaluate, infer.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Steps = if ($env:STEPS) { $env:STEPS } else { 500 }
$N = if ($env:N) { $env:N } else { 64 }
function Run { & python @args; if ($LASTEXITCODE) { throw "failed: python $args" } }
Run -m satfin.data.download --start 2025-06-01T18:00 --hours 3 --sector M1
Run -m satfin.data.download --start 2025-06-05T18:00 --hours 2 --sector M1
Run -m satfin.data.download --start 2025-06-10T18:00 --hours 2 --sector M2
Run -m satfin.data.preprocess
if (-not (Test-Path runs/demo/best.pt)) { Run -m satfin.train --run demo --steps $Steps }
Run -m satfin.evaluate --ckpt runs/demo/best.pt --n $N
$Seq = (Get-ChildItem data/processed -Directory -Filter "*20250610*" | Select-Object -First 1).FullName
Run -m satfin.infer --ckpt runs/demo/best.pt --seq $Seq --every 10 --k 9 --range 0:31
Write-Host "Done. Dashboard: streamlit run app/dashboard.py"
