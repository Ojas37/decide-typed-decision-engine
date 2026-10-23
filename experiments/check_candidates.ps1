# Hard-constraint checks for each artifact set. Uses TRAIN rows only (never dev).
# Run from the starter/ directory:  powershell -File experiments\check_candidates.ps1
$ErrorActionPreference = "Continue"
$wd = (Get-Location).Path
$tmp = "C:\Users\Ojas Neve\AppData\Local\Temp\opencode"
$sets = [ordered]@{
  baseline = "artifacts"
  A  = "experiments\candidates\A\artifacts"
  B  = "experiments\candidates\B\artifacts"
  AB = "experiments\candidates\AB\artifacts"
}
$port = 8780
$summary = @{}
foreach ($name in $sets.Keys) {
  $art = $sets[$name]; $port++
  "=================== $name  ($art)"
  $size = (Get-ChildItem $art -File | Measure-Object Length -Sum).Sum / 1MB
  "artifact size: {0:N1} MB (limit 500)" -f $size
  python experiments\route_cal_checks.py $art 2>&1 | Select-String -NotMatch "Warning"
  python smoke_test.py --artifacts $art --n 300 2>&1 | Select-String -Pattern "uncached|route |approve |grounded |passed|FAILED|  -"
  $env:ARTIFACTS_DIR = (Join-Path $wd $art)
  $srv = Start-Process -FilePath python -ArgumentList "-m uvicorn app:app --host 127.0.0.1 --port $port" -PassThru -NoNewWindow `
         -RedirectStandardOutput "$tmp\uv_$name.out" -RedirectStandardError "$tmp\uv_$name.err" -WorkingDirectory $wd
  Start-Sleep -Seconds 6
  python selfcheck.py --url "http://127.0.0.1:$port/v1/decide" --eval data/train.jsonl --limit 400 --json "$tmp\lat_$name.json" 2>&1 | Out-Null
  Stop-Process -Id $srv.Id -Force
  Remove-Item Env:ARTIFACTS_DIR
  $j = Get-Content "$tmp\lat_$name.json" | ConvertFrom-Json
  "HTTP (selfcheck on 400 TRAIN rows): mean p95 {0:N2} ms | per-decision p95 route {1:N2} approve {2:N2} grounded {3:N2} | valid {4}/{5} | gates {6}" -f `
    $j.latency_ms.mean_p95, $j.latency_ms.p95.route, $j.latency_ms.p95.approve, $j.latency_ms.p95.grounded, $j.schema.valid, $j.schema.total, `
    (($j.gates.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ",")
}
