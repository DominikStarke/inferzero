# The golden test set of the shared GEMM layer (docs/gemm-inventory.md table C; the user, 2026-11-27): per model, the
# logits of runz --dump-logits for each kind of path, hashed. A change meant to keep the numbers must keep every hash;
# a run compares itself with a baseline file and, inside the run, the paths that must equal the model's base run.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\gemm_golden.ps1 [-Models a.gguf,b.gguf] [-Kinds base,host,...]
#       [-Baseline <results.tsv of an earlier run>] [-Out <folder>] [-TimeoutS 1500] [-List]
# Kinds (each skipped where it does not apply to the model):
#   base     the fixed prompt (tests/gemm_golden_prompt.txt, ~1,100 tokens) in chunks of 512, then 16 generated tokens
#   short    a 4-token prompt, 16 generated (the 2-8 row GEMVs)
#   mtp      base with --mtp (the verification passes), models with an MTP block: found by "MTP" / "mtp" in the file's
#            path (the test set's files say so); an MTP file named otherwise skips this kind
#   slots    runz --slots-check (two conversations alone, then batched): must say OK
#   host     base with 1 GiB of the weights in host memory (ARC_HOST_WEIGHTS_GB=1): must equal base. Dense models
#   tp2, tp3 base split over 2 / 3 "GPUs" on the one GPU (ARC_TP_SIM=1 --gpus N): their own hashes
#   layer    base over 2 stages (--gpus 2 --split layer): must equal base
#   fusion   each fused form switched off or on: ARC_H8_NORM=0, ARC_H8_REUSE=0, ARC_H8_SWI=0, ARC_GEMV_GU=1, ARC_GDN_FUSE2=1
#            (dense), ARC_GDN_FUSE=0 (Qwen), ARC_INK_FUSE=0 (Inkling), the experts without moe_rg (MoE): must equal base.
#            The "form" column says whether the fused form ran (ARC_GEMM_PLAN=1 on the base and fusion rows): a row
#            whose form was not taken proves nothing. Also "switch ARC_BATCH=64" (command-list batching, not a fusion)
#   option   the options page's ARC_H8=0: its own hash (another path, other numbers)
# Writes <Out>\results.tsv (model, kind, expect, hash, status, form, seconds) and prints each row; exit 1 when any row
# is DIFFERS (a path that must equal base does not), CHANGED (a hash unlike the baseline's), SHORT (it generated fewer
# than its 16 tokens: the end-of-text token came first, so less was checked), FAIL, ERROR, or, with -Baseline, MISSING
# (a row the baseline lacks), GONE (a baseline row of the models and kinds run that this run did not produce), or a
# default model not found in the HF cache.
# Every GPU run goes through bench\gpu_guard.ps1. The driver version and the commit are recorded: a baseline holds
# only for the build and driver it was made with. The runs see the defaults plus their own switch: the shell's ARC_*,
# INFERZERO_* and ZE_* variables are cleared for them (but ARC_DATA_DIR and ZE_AFFINITY_MASK) and put back after,
# their names in results.tsv's header. Windows PowerShell 5.1 and 7.
param(
  [string[]]$Models = @(),
  [string[]]$Kinds = @(),
  [string]$Baseline = "",
  [string]$Out = "",
  [int]$TimeoutS = 1500,
  [switch]$List
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
$guard = Join-Path $root "bench\gpu_guard.ps1"
$prompt = Join-Path $PSScriptRoot "gemm_golden_prompt.txt"
$allKinds = @("base", "short", "mtp", "slots", "host", "tp2", "tp3", "layer", "fusion", "option")
# (a comma list in one argument too: -Kinds base,host from cmd or a script)
$Kinds = @($Kinds | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })
foreach ($k in $Kinds) { if ($allKinds -notcontains $k) { throw "unknown kind '$k' (kinds: $($allKinds -join ', '))" } }
if ($Kinds.Count -eq 0) { $Kinds = $allKinds }

# the default models: those of the laptop's test set found in the Hugging Face cache; one not found is said, and is a
# failure under -Baseline (a run with fewer models must not pass as the whole set)
$notFound = @()
if ($Models.Count -eq 0) {
  $hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
  foreach ($n in @("Qwen3.5-4B-UD-Q4_K_XL.gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                   "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "Inkling-2-Mini-UD-IQ4_XS.gguf")) {
    $f = Get-ChildItem -Path $hub -Recurse -Filter $n -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($f) { $Models += $f.FullName } else { $notFound += $n; Write-Host "gemm_golden: model not found in the HF cache: $n" }
  }
}
if ($Models.Count -eq 0) { throw "no models: give -Models a.gguf,b.gguf" }

# what a model is, from its file name (the test set's files): MoE, Inkling 2, an MTP block
function Get-ModelKind([string]$m) {
  $n = [IO.Path]::GetFileName($m)
  $ink = $n -match "Inkling"
  return @{
    name = $n
    inkling = $ink
    moe = $ink -or ($n -match "A3B|A4B|Flash-Next|DeepSeek")
    mtp = (-not $ink) -and ($m -match "MTP|mtp")  # (the path: unsloth's MTP files say it in their folder's name)
  }
}
# the runs of one kind on one model: name, runz arguments (after the model), environment, expectation
function Get-Runs([hashtable]$mk, [string]$kind) {
  $long = @("--text-file", $prompt, "--chunk", "512", "--n", "16")
  switch ($kind) {
    "base" { return , @(@{ id = "base"; args = $long; env = @{}; expect = "-" }) }
    "short" { return , @(@{ id = "short"; args = @("--tokens", "1,2,3,4", "--n", "16"); env = @{}; expect = "-" }) }
    "mtp" { if ($mk.mtp) { return , @(@{ id = "mtp"; args = $long + @("--mtp"); env = @{}; expect = "-" }) } }
    "slots" { if (-not $mk.inkling) { return , @(@{ id = "slots"; args = @("--text-file", $prompt, "--slots-check", "--n", "16"); env = @{}; expect = "ok" }) } }
    "host" { if (-not $mk.moe) { return , @(@{ id = "host"; args = $long; env = @{ ARC_HOST_WEIGHTS_GB = "1" }; expect = "base" }) } }
    "tp2" { if (-not $mk.inkling) { return , @(@{ id = "tp2"; args = $long + @("--gpus", "2"); env = @{ ARC_TP_SIM = "1" }; expect = "-" }) } }
    "tp3" { if (-not $mk.inkling) { return , @(@{ id = "tp3"; args = $long + @("--gpus", "3"); env = @{ ARC_TP_SIM = "1" }; expect = "-" }) } }
    "layer" { if (-not $mk.inkling) { return , @(@{ id = "layer"; args = $long + @("--gpus", "2", "--split", "layer"); env = @{ ARC_TP_SIM = "1" }; expect = "base" }) } }
    "fusion" {
      $r = @()
      # (one row each: env, the switches that turn the fused form off (or the opt-in one on); form, what ARC_GEMM_PLAN=1
      #  shows when the form runs (a kernel or a -D option); on, where it must show: "base" for a form the row turns
      #  off (the base run takes it, the row must not), "row" for an opt-in form (the row must take it). A form that
      #  did not run proves nothing: the row's "form" column says "not taken" (the audit of 2026-11-28). "" form: not
      #  visible in the plan log)
      $sw = @()
      if (-not $mk.moe) {
        $sw += @(@(@{ ARC_H8_NORM = "0" }, '-> rmsnorm_h8r ', "base"), @(@{ ARC_H8_REUSE = "0" }, "", "base"),
                 @(@{ ARC_H8_SWI = "0" }, 'fused: [^|]*swiglu\+rot', "base"), @(@{ ARC_GEMV_GU = "1" }, '-DGU2', "row"),
                 @(@{ ARC_GDN_FUSE2 = "1" }, '-DGDN_FUSE2', "row"))
      }
      # (the DeltaNet conv + state update and the gates in the recurrence: every Qwen model, the 35B's too)
      if (-not $mk.inkling) { $sw += , @(@{ ARC_GDN_FUSE = "0" }, '-DFUSE_GATES|-DCONV_UPD', "base") }
      # (Inkling's controller gain + forget + apply in one launch)
      if ($mk.inkling) { $sw += , @(@{ ARC_INK_FUSE = "0" }, '-> ink_ctrl_fused ', "base") }
      # (the experts without moe_rg: on the 35B every chunk then takes decode + GEMM (moe_gemm), which moe_rg says it
      #  equals bit for bit (engine_moe.h); Inkling's NVFP4 / IQ4_NL down layers take the per-tile kernels instead, so
      #  its row compares moe_rg with per tile, not with decode + GEMM; checked with ARC_GEMM_PLAN=1, 2026-11-28)
      if ($mk.moe) { $sw += , @(@{ ARC_MOE_RG_MIN = "1000000"; ARC_MOE_GEMM_MIN = "1" }, '-> moe_rg ', "base") }
      foreach ($s in $sw) {
        $e = $s[0]
        $id = "fusion " + (($e.Keys | Sort-Object | ForEach-Object { "$_=$($e[$_])" }) -join " ")
        $x = "base"
        # (Inkling without moe_rg: per tile, another kernel and other numbers than moe_rg (2026-11-28: DE5E8415DE35CD96
        #  against base 8C9B0CF29F635EFC), so its own hash, recorded, not a must-equal row; its KL is still to measure)
        if ($mk.inkling -and $e.ContainsKey("ARC_MOE_RG_MIN")) { $id += " (per tile)"; $x = "-" }
        $r += @{ id = $id; args = $long; env = $e; expect = $x; form = $s[1]; formOn = $s[2] }
      }
      # (not a fusion: command-list batching, a switch that must leave the numbers alone)
      if (-not $mk.moe) { $r += @{ id = "switch ARC_BATCH=64"; args = $long; env = @{ ARC_BATCH = "64" }; expect = "base" } }
      return , $r
    }
    "option" { if (-not $mk.moe) { return , @(@{ id = "option ARC_H8=0"; args = $long; env = @{ ARC_H8 = "0" }; expect = "-" }) } }
  }
  return , @()
}

if ($List) {
  foreach ($m in $Models) {
    $mk = Get-ModelKind $m
    foreach ($k in $Kinds) { foreach ($r in (Get-Runs $mk $k)) { "{0}`t{1}`t{2}" -f $mk.name, $r.id, $r.expect } }
  }
  return
}
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found: build it first (.\build.ps1 zero\runz.cpp zero\serverz.cpp)" }
if (-not $Out) { $Out = Join-Path $env:TEMP ("inferzero-golden\" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
New-Item -ItemType Directory -Force $Out | Out-Null

# the build and the driver the hashes belong to
$commit = ""
try { $commit = (& git -C $root rev-parse --short HEAD 2>$null) } catch {}
$driver = ""
try {
  $driver = (Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match "Intel" } |
             ForEach-Object { "$($_.Name) $($_.DriverVersion)" }) -join "; "
} catch {}
$base = @{}
if ($Baseline) {
  foreach ($l in (Get-Content -Encoding UTF8 $Baseline)) {
    if ($l.StartsWith("#") -or $l.StartsWith("model`t")) { continue }
    $v = $l -split "`t"
    if ($v.Count -ge 4) { $base[$v[0] + "|" + $v[1]] = $v[3] }
  }
}
$rows = New-Object System.Collections.ArrayList
$produced = New-Object System.Collections.Generic.HashSet[string]
$bad = 0
$notTaken = 0
$baseLog = ""
function Get-Sha([string]$f) { if (Test-Path $f) { return (Get-FileHash -Algorithm SHA256 $f).Hash.Substring(0, 16) } return "" }

# (every run sees the defaults plus its own switch only: the ARC_* / INFERZERO_* / ZE_* variables of the caller's
#  shell are saved, cleared for the runs and put back at the end, so a golden "base" can't be a non-default path.
#  Kept: ARC_DATA_DIR, where gpu_guard and the engine find the GPU hang marker, and ZE_AFFINITY_MASK, the GPU chosen)
$keep = @("ARC_DATA_DIR", "ZE_AFFINITY_MASK")
$saved = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -match "^(ARC_|INFERZERO_|ZE_)" })) {
  if ($keep -contains $v.Name) { continue }
  $saved[$v.Name] = $v.Value
  Remove-Item "Env:$($v.Name)"
}
$cleared = @($saved.Keys | Sort-Object) -join " "
$kept = @($keep | Where-Object { Test-Path "Env:$_" }) -join " "
try {
foreach ($m in $Models) {
  $mk = Get-ModelKind $m
  $baseHash = ""
  $baseLog = ""
  # (base first: the kinds that must equal it compare with this run's)
  $order = @("base") + @($Kinds | Where-Object { $_ -ne "base" })
  foreach ($k in $order) {
    if ($k -eq "base" -and $Kinds -notcontains "base" -and -not ($Kinds | Where-Object { @("host", "layer", "fusion") -contains $_ })) { continue }
    foreach ($r in (Get-Runs $mk $k)) {
      $tag = ($mk.name + "_" + $r.id) -replace '[^A-Za-z0-9_.=-]', '_'
      $dump = Join-Path $Out "$tag.bin"
      $log = Join-Path $Out "$tag.log"
      $a = @($m) + $r.args
      if ($r.id -ne "slots") { $a += @("--dump-logits", $dump) }
      if (Test-Path $dump) { Remove-Item $dump }  # (a dump left by an earlier run must not pass for this run's)
      # (the base and fusion rows log their kernels (ARC_GEMM_PLAN=1, logging only: the logits are the same), so a
      #  fusion row can show whether its form ran)
      $plan = $r.id -eq "base" -or $r.ContainsKey("form")
      if ($plan) { $env:ARC_GEMM_PLAN = "1" }
      foreach ($e in $r.env.Keys) { Set-Item "Env:$e" $r.env[$e] }
      $t0 = Get-Date
      # (Windows PowerShell 5.1 with "Stop" ends the script at a child's first stderr line: "Continue" for the run, so
      #  a failed run is an ERROR row, not the end of the test. -ExecutionPolicy Bypass: a pwsh 7 caller's 5.1 child
      #  would otherwise refuse gpu_guard.ps1 where 5.1's policy is Restricted)
      $eap = $ErrorActionPreference
      $ErrorActionPreference = "Continue"
      $o = & powershell -NoProfile -ExecutionPolicy Bypass -File $guard -TimeoutS $TimeoutS -- $runz @a 2>&1 | Out-String
      $code = $LASTEXITCODE
      $ErrorActionPreference = $eap
      foreach ($e in $r.env.Keys) { Remove-Item "Env:$e" -ErrorAction SilentlyContinue }
      if ($plan) { Remove-Item Env:ARC_GEMM_PLAN -ErrorAction SilentlyContinue }
      if ($r.id -eq "base") { $baseLog = $o }
      # (did the form run: "taken", "NOT TAKEN" (equal to base proves nothing then), "-" (no form, or none visible))
      $form = "-"
      if ($r.ContainsKey("form") -and $r.form) {
        # (the plan's lines only: a start-up line (a spill warning) names the build options of kernels compiled but not
        #  run, -DCONV_UPD with ARC_GDN_FUSE=0 for one)
        $planOnly = { param($s) (($s -split "`r?`n") | Where-Object { $_ -match 'gemm-plan:' }) -join "`n" }
        $inBase = (& $planOnly $baseLog) -match $r.form
        $inRow = (& $planOnly $o) -match $r.form
        $form = $(if ($r.formOn -eq "row") { if ($inRow) { "taken" } else { "NOT TAKEN" } }
                  else { if ($inBase -and -not $inRow) { "taken" } else { "NOT TAKEN" } })
      }
      $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds)
      [IO.File]::WriteAllText($log, $o, (New-Object Text.UTF8Encoding $false))
      $hash = ""
      $status = "rec"
      if ($r.id -eq "slots") {
        $ok = ($code -eq 0) -and -not ($o -match "MISMATCH")
        $hash = $(if ($ok) { "ok" } else { "mismatch" })
        $status = $(if ($ok) { "ok" } else { "FAIL" })
      } else {
        $hash = Get-Sha $dump
        # (a run that stopped at the end-of-text token before its 16 tokens checked less than it says: SHORT, not ok.
        #  The fixed prompt ends mid-sentence so that the models write on; the 27B and 35B stopped at once otherwise)
        $gen = -1
        if ($o -match "generation: (\d+) tokens") { $gen = [int]$Matches[1] }
        # (no such line: what was generated is unknown, so the run proves nothing: ERROR)
        if (-not $hash -or $code -ne 0 -or $gen -lt 0) { $status = "ERROR" }
        elseif ($gen -lt 16) { $status = "SHORT" }
        elseif ($r.expect -eq "base") { $status = $(if ($hash -eq $baseHash) { "ok" } else { "DIFFERS" }) }
      }
      if ($r.id -eq "base" -and $status -ne "ERROR") { $baseHash = $hash }
      $key = $mk.name + "|" + $r.id
      [void]$produced.Add($key)
      if ($Baseline -and -not $base.ContainsKey($key) -and $status -ne "ERROR") {
        $status = "MISSING"  # (a row the baseline does not have: nothing to compare it with)
      } elseif ($base.ContainsKey($key) -and $status -ne "ERROR" -and $status -ne "FAIL") {
        if ($base[$key] -ne $hash) { $status = "CHANGED" } elseif ($status -eq "rec") { $status = "ok" }
      }
      if (@("DIFFERS", "CHANGED", "FAIL", "ERROR", "SHORT", "MISSING") -contains $status) { $bad++ }
      if ($form -eq "NOT TAKEN") { $notTaken++ }
      $line = "{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}" -f $mk.name, $r.id, $r.expect, $hash, $status, $form, $sec
      [void]$rows.Add($line)
      Write-Host $line
      if (Test-Path $dump) { Remove-Item $dump }  # (the logits are large; the hash is what is kept)
    }
  }
}
} finally {
  foreach ($n in $saved.Keys) { Set-Item "Env:$n" $saved[$n] }  # (the caller's shell as it was)
}
# (the baseline's rows this run did not produce: GONE, for the models and kinds this run covers; a row's kind is the
#  first word of its id, "switch" being the fusion kind's)
if ($Baseline) {
  $names = @($Models | ForEach-Object { [IO.Path]::GetFileName($_) })
  foreach ($key in @($base.Keys | Sort-Object)) {
    if ($produced.Contains($key)) { continue }
    $mn = ($key -split '\|', 2)[0]; $id = ($key -split '\|', 2)[1]
    $kind = ($id -split ' ')[0]
    if ($kind -eq "switch") { $kind = "fusion" }
    if ($names -notcontains $mn -or $Kinds -notcontains $kind) { continue }
    $line = "{0}`t{1}`t-`t{2}`tGONE`t-`t0" -f $mn, $id, $base[$key]
    [void]$rows.Add($line); Write-Host $line; $bad++
  }
  foreach ($n in $notFound) { Write-Host "gemm_golden: $n not run (not in the HF cache): not ok under -Baseline"; $bad++ }
}
$head = @("# tests/gemm_golden.ps1, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), commit $commit", "# driver: $driver",
          "# cleared for the runs: $(if ($cleared) { $cleared } else { '-' }); kept: $(if ($kept) { $kept } else { '-' })",
          "# baseline: $(if ($Baseline) { $Baseline } else { '-' })$(if ($notFound.Count) { '; models not found: ' + ($notFound -join ' ') })",
          "model`tkind`texpect`thash`tstatus`tform`tseconds")
$res = Join-Path $Out "results.tsv"
[IO.File]::WriteAllText($res, (($head + $rows) -join "`n") + "`n", (New-Object Text.UTF8Encoding $false))
Write-Host "gemm_golden: $($rows.Count) runs, $bad not ok, $notTaken fusion rows whose form did not run (no proof); $res"
if ($bad -gt 0) { exit 1 }
