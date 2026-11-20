# gpu_sweep: every kernel and tile setting a user can change, measured on one GPU with a real model, for any Intel GPU
# (Arc iGPUs, A-series, B-series, Data Center GPU Max). bench/gpu_sweep.sh is the same for Linux; the two must stay the
# same (change both together). Paste the whole output into the GitHub issue.
#   .\bench\gpu_sweep.ps1 -Model <Qwen3.8 27B or another dense gguf> [-MoeModel <35B-A3B or Flash Next gguf>]
#        [-DFlash <DFlash2 drafter folder>] [-Device N] [-Quick] [-Phases GPDTKM]
# It covers the prompt GEMM on DP4a and XMX (W8, H8, f16), the XMX tiles, the GEMVs, attention, DeltaNet, MoE and
# drafting; bench\f33_sweep.ps1 does the chunked DeltaNet's and XMX attention's KL / emulation checks.
# Phases (all by default, in this order; -Phases picks some):
#   G  generation (--bench-pp 128 --n 128, tg128): each setting alone from the defaults, every value from the smallest
#      to the largest;
#   P  the prompt (--bench-pp 4096): the same for the prompt's settings;
#   then for G / P / M:
#      H  each setting's best value: its logits against the defaults (a short prompt, --dump-logits, SHA-256):
#         "same", or "differs: needs a KL check";
#      C  the best combination: only each setting's best value, added one at a time from the largest gain down, each
#         kept only when the run gets faster still; then the defaults against it at tg128 and 512 / 4K / 16K prompts;
#   D  drafting on a coding question (256 tokens): plain, the MTP block with 2-7 drafts, a DFlash2 drafter (-DFlash),
#      each one's text checked against plain;
#   T  the XMX tile per GEMM kind (the prompt's --prof per gemm_* kernel for each tile setting, two rounds in
#      alternating order): which tile each of the model's GEMMs wants;
#   K  where the time goes: --prof (the prompt) and --prof-gen (generation) by kernel, the defaults against the best
#      combination, the largest kernels first, with each kernel's change;
#   M  the MoE model's settings (with -MoeModel), generation and prompt, with its H / C / K.
# A value wins when it is 2 % faster than the defaults; the defaults are measured again after each sweep (drift).
# Prompt chunks stay at most 4096 tokens. -Device N: which GPU (ZE_AFFINITY_MASK). The progress (phase, step of steps,
# time so far and left) shows on the console while it runs. Expect ~1.5-2.5 hours for the 27B (-Quick: ~45 minutes).
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [string]$MoeModel = "",
  [string]$DFlash = "",
  [string]$Device = "",
  [string]$Phases = "GPDTKM",
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the inferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
if ($MoeModel -and -not (Test-Path $MoeModel)) { throw "not found: $MoeModel" }
if ($DFlash -and -not (Test-Path $DFlash -PathType Container)) { throw "not a folder: $DFlash" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("inferzero-gpu-sweep-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory $tmp -Force | Out-Null
$prompt = Join-Path $tmp "prompt.txt"
$readme = Get-Content (Join-Path $root "README.md") -Raw -ErrorAction SilentlyContinue
if (-not $readme) { $readme = "The quick brown fox. " * 200 }
[IO.File]::WriteAllText($prompt, $readme.Substring(0, [Math]::Min(9000, $readme.Length)), (New-Object Text.UTF8Encoding $false))
$question = Join-Path $tmp "question.txt"
[IO.File]::WriteAllText($question, "<|im_start|>user`nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using " +
  "a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>`n<|im_start|>assistant`n<think>`n`n</think>`n`n",
  (New-Object Text.UTF8Encoding $false))
function Has([string]$p) { $Phases.ToUpper().Contains($p) }

# --- the settings: name|values|quick values, each list from the smallest value to the largest
$GEN = @(
  "ARC_GVROWS|1 2 4|1 2 4", "ARC_GEMV_GRF|128 256|128 256", "ARC_GEMV_MT_GRF|128 256|128 256", "ARC_GEMV_WAVE|0 1|0 1",
  "ARC_GEMV_DB|0 1|0 1", "ARC_GEMV_PF|0 1|0 1", "ARC_GEMV_SMALL_WG|0 1|0 1", "ARC_GEMV_GU|0 1|0 1", "ARC_GV_IQ3S|0 1|0 1",
  "ARC_IQ3S_TILE|0 1|0 1", "ARC_WIDE_ROWS|0 1|0 1", "ARC_BATCH|0 8 16 32 64 128|0 32 128", "ARC_ATTN_DEC_GRF|128 256|128 256",
  "ARC_ATTN_DEC_SG|128 256 512 1024 2048|256 512 2048", "ARC_ATTN_DEC_SMAX|64 128 256 512 1024|128 256 1024")
$PP = @(
  "ARC_NO_XMX|0 1|0 1", "ARC_XMX_FORCE|0 1|0 1", "ARC_H8|0 1|0 1", "ARC_H8_MIN|256 1024 100000|256 100000", "ARC_140T|0 1|0 1",
  "ARC_140T_RES|0 1|0 1", "ARC_X16|0 1|0 1", "ARC_GEMM_TM|16 32|16 32", "ARC_GEMM_RG|1 2 4 8|2 4 8", "ARC_GEMM_WGY|2 4 8 16|4 8 16",
  "ARC_GEMM_CG|1 2 4|1 2 4", "ARC_GEMM_WPF|0 2 4 8|0 4", "ARC_GEMM_2D|0 1|0 1", "ARC_GEMM_2D_PF|0 1|0 1",
  "ARC_GEMM_XW_WGX|1 2 4 8|1 4 8", "ARC_GEMM_XW_LOCK|0 1|0 1", "ARC_GEMM_XW_2D|0 1 2|0 1", "ARC_GEMM_XW_PF|0 1 2 3|0 1 3",
  "ARC_GEMM_XW_PFD|1 2 4 8|1 2 4", "ARC_GEMM_XW_GU|0 1|0 1", "ARC_GEMM_XW_NC_E|1 2|1 2", "ARC_GEMM_SPLIT|0 9 17 34 68|0 17 34",
  "ARC_GEMM_GRF|128 256|128 256", "ARC_H8_GRF|128 256|128 256", "ARC_H8_NORM|0 1|0 1", "ARC_H8_SWI|0 1|0 1", "ARC_H8_REUSE|0 1|0 1",
  "ARC_H8_FUSE|0 1|0 1", "ARC_ATTN_XMX|0 1|0 1", "ARC_ATTN_FX|0 1|0 1", "ARC_ATTN_X8|0 1|0 1", "ARC_GDN_L1PF|0 1 2 4 8 16|0 2 8",
  "ARC_GDN_PF|0 2|0 2", "ARC_GDN_CHUNK|0 1 2 3|0 3", "ARC_W8_PARTIAL_GB|0 auto|0 auto", "--chunk|1024 2048 4096|2048 4096")
$MOE = @(
  "ARC_MOE_GRF|128 256|128 256", "ARC_MOE_CG|1 2 4 8|1 2 4", "ARC_MOE_WY|1 2 4 8|1 2 4", "ARC_MOE_GR_GU|1 2 4|1 2 4",
  "ARC_MOE_GR_DN|1 2 4 8|1 4 8", "ARC_MOE_TM|16 32|16 32", "ARC_MOE_RG_XMX|0 1|0 1", "ARC_MOE_GEMM_PAD|0 1|0 1", "ARC_XALL|0 1|0 1",
  "ARC_XCPU|0 1|0 1")
$TILES8 = @("32x16 (default)|", "32x32|ARC_GEMM_XW_NC=2", "16x32|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=2", "32x16 WGX 4|ARC_GEMM_XW_WGX=4",
  "16x64|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=4", "64x16|ARC_GEMM_XW_TM=64", "32x16 WGX 1|ARC_GEMM_XW_WGX=1", "32x32 WGX 4|ARC_GEMM_XW_NC=2 ARC_GEMM_XW_WGX=4")
$TILES16 = @("32x64 (default)|", "32x32|ARC_GEMM_XW_NC=2", "32x96|ARC_GEMM_XW_NC=6", "32x128|ARC_GEMM_XW_NC=8",
  "16x128|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=8", "64x64|ARC_GEMM_XW_TM=64", "32x64 WGX 1|ARC_GEMM_XW_WGX=1", "32x64 WGX 2|ARC_GEMM_XW_WGX=2")
$vars = @("ARC_GEMM_XW_TM", "ARC_GEMM_XW_NC") + @(($GEN + $PP + $MOE) | ForEach-Object { ($_ -split '\|')[0] } | Where-Object { $_ -like "ARC_*" })
if ($Device) { $env:ZE_AFFINITY_MASK = $Device }
function Vals([string]$item) { $p = $item -split '\|'; if ($Quick) { $p[2] -split ' ' } else { $p[1] -split ' ' } }

# --- progress on the console (not in the results): the phase, its step of the steps planned, the time so far and left
$script:ph = @{ name = ""; step = 0; steps = 0; t0 = [DateTime]::Now }; $allT0 = [DateTime]::Now
function Start-Phase([string]$n, [int]$steps) { $script:ph = @{ name = $n; step = 0; steps = $steps; t0 = [DateTime]::Now } }
function Fmt-T([double]$s) { "{0}h{1:00}m" -f [int][Math]::Floor($s / 3600), [int][Math]::Floor(($s % 3600) / 60) }
function Show-Progress([string]$what) {
  $script:ph.step++
  $steps = [Math]::Max($script:ph.steps, $script:ph.step)
  $el = ([DateTime]::Now - $script:ph.t0).TotalSeconds
  $left = if ($script:ph.step -gt 1) { ", ~" + (Fmt-T ($el * ($steps - $script:ph.step + 1) / ($script:ph.step - 1))) + " left in it" } else { "" }
  $msg = "[{0} {1}/{2}] {3} | {4} in this phase{5}, {6} in all" -f $script:ph.name, $script:ph.step, $steps,
    $what.Substring(0, [Math]::Min(40, $what.Length)), (Fmt-T $el), $left, (Fmt-T ([DateTime]::Now - $allT0).TotalSeconds)
  Write-Progress -Activity "gpu_sweep" -Status $msg
  Write-Host $msg -ForegroundColor DarkGray
}

# --- one runz call: settings "NAME=value ..." (and --chunk=N), then runz's arguments; returns the output lines.
# The first time a model meets a set of settings (and drafting flags), an untimed warm-up run comes first (a 512-token
# prompt and 16 tokens with the same switches): a new setting builds new kernels, and a build inside a timed run would
# count against it. The kernels then come from the cache, so every timed run measures the kernels, not their build.
$warmed = @{}
function Invoke-Runz([string]$m, [string]$sset, [string[]]$a) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $extra = @(); $draft = @()
  for ($i = 0; $i -lt $a.Count; $i++) {  # (the drafting flags: their kernels are built only with them)
    if ($a[$i] -eq "--mtp") { $draft += "--mtp" }
    elseif ($a[$i] -in "--mtp-drafts", "--dflash", "--draft" -and $i + 1 -lt $a.Count) { $draft += @($a[$i], $a[$i + 1]); $i++ }
  }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    foreach ($kv in ($sset -split ' ' | Where-Object { $_ })) {
      if ($kv -like "--chunk=*") { $extra += @("--chunk", $kv.Substring(8)) } else { $p = $kv -split '=', 2; Set-Item "Env:$($p[0])" $p[1] }
    }
    $key = "$m|$sset|$($draft -join ' ')"
    if (-not $script:warmed.ContainsKey($key)) {
      $script:warmed[$key] = $true
      Write-Host ("[{0}] warm-up: {1} {2}" -f $script:ph.name, $(if ($sset) { $sset } else { "defaults" }), ($draft -join ' ')) -ForegroundColor DarkGray
      $null = & $runz $m --bench-pp 512 --n 16 @draft @extra 2>&1
    }
    Show-Progress ($(if ($sset) { $sset } else { "defaults" }) + " " + ($a[0..([Math]::Min(3, $a.Count - 1))] -join " "))
    return (& $runz $m @a @extra 2>&1 | ForEach-Object { "$_" })
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}
function Best-PP($o) {
  $r = @($o | ForEach-Object { if ($_ -match 'pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s') { [double]::Parse($Matches[1], $inv) } })
  if ($r.Count) { ($r | Measure-Object -Maximum).Maximum } else { 0 }
}
function TG($o) { $l = @($o | Where-Object { $_ -match 'tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s' }); if ($l.Count -and $l[-1] -match '->\s+([\d.]+)') { [double]::Parse($Matches[1], $inv) } else { 0 } }
function Gen-TPS($o) { $l = @($o | Where-Object { $_ -match 'generation: \d+ tokens in [\d.]+ ms \(([\d.]+) t/s\)' }); if ($l.Count -and $l[-1] -match '\(([\d.]+) t/s\)') { [double]::Parse($Matches[1], $inv) } else { 0 } }
function Err-Of($o) { ($o | Where-Object { $_ -match 'error|cannot|failed|exception' } | Select-Object -First 1) }
function Pct([double]$a, [double]$b) { if ($b -gt 0) { [string]::Format($inv, "{0:+0.0;-0.0;+0.0} %", 100 * ($a / $b - 1)) } else { "" } }
function F([double]$x, [int]$d = 1) { $x.ToString("F$d", $inv) }
function Measure-It([string]$m, [string]$kind, [string]$sset) {
  if ($kind -eq "gen") { TG (Invoke-Runz $m $sset @("--bench-pp", "128", "--n", "128")) } else { Best-PP (Invoke-Runz $m $sset @("--bench-pp", "4096", "--n", "0")) }
}
function Hash-Of([string]$m, [string]$sset) {
  $f = Join-Path $tmp "l.bin"; Remove-Item $f -ErrorAction SilentlyContinue
  $null = Invoke-Runz $m $sset @("--text-file", $prompt, "--n", "8", "--dump-logits", $f)
  if (Test-Path $f) { (Get-FileHash -Algorithm SHA256 $f).Hash.Substring(0, 16) } else { "failed" }
}
function Prof-Rows($o) {
  $on = $false; $rows = [ordered]@{}
  foreach ($l in $o) {
    if ($l -match '^GPU time by kernel') { $on = $true; continue }
    if ($on -and $l -match '^\s+([A-Za-z0-9_]+)\s+([\d.]+) ms') { $rows[$Matches[1]] = [double]::Parse($Matches[2], $inv) }
  }
  $rows
}

# --- G / P / M: each setting alone; returns each setting's best winning value as @{ kv; gain }
function Sweep([string]$title, [string]$m, [string]$kind, [string[]]$list) {
  $unit = if ($kind -eq "gen") { "tok/s (tg128)" } else { "tok/s (pp4096)" }
  Write-Output ""; Write-Output "=== $title`: one setting at a time, $unit; the base is the defaults"
  $n = 0; foreach ($it in $list) { if ($kind -eq "gen" -and $it -like "--chunk*") { continue }; $n += @(Vals $it).Count }
  Start-Phase $title.Substring(0, 1) ($n + 2)
  $base = Measure-It $m $kind ""
  Write-Output ("{0,-28} {1,10}" -f "defaults", (F $base 2))
  $win = @()
  foreach ($it in $list) {
    $name = ($it -split '\|')[0]
    if ($kind -eq "gen" -and $name -eq "--chunk") { continue }
    $bestR = 0; $bestKv = ""
    foreach ($v in (Vals $it)) {
      $kv = if ($name -eq "--chunk") { "--chunk=$v" } else { "$name=$v" }
      $r = Measure-It $m $kind $kv
      $note = if ($r -le 0) { "failed" } elseif ($r -ge $base * 1.02) { "wins" } else { "" }
      if ($r -ge $base * 1.02 -and $r -gt $bestR) { $bestR = $r; $bestKv = $kv }
      Write-Output ("{0,-28} {1,10}  {2,-9} {3}" -f $kv, (F $r 2), (Pct $r $base), $note)
    }
    if ($bestKv) { $win += @{ kv = $bestKv; gain = $bestR / $base } }
  }
  $again = Measure-It $m $kind ""
  Write-Output ("{0,-28} {1,10}  {2,-9}" -f "defaults again (drift)", (F $again 2), (Pct $again $base))
  return , $win
}
function Check-And-Combine([string]$m, [string]$kind, $win) {
  if (-not $win -or $win.Count -eq 0) { Write-Output "(no setting 2 % faster than the defaults)"; return "" }
  Write-Output "--- H. each setting's best value: its logits against the defaults (a short prompt and 8 tokens)"
  Start-Phase "H" ($win.Count + 1)
  $ref = Hash-Of $m ""
  foreach ($w in $win) { $h = Hash-Of $m $w.kv; Write-Output ("{0,-28} {1}" -f $w.kv, $(if ($h -eq $ref) { "same" } else { "differs ($h): needs a KL check" })) }
  Write-Output "--- C. the best combination: each setting's best value, added from the largest gain down, kept when faster"
  Start-Phase "C" ($win.Count + 1 + 2 * $(if ($Quick) { 3 } else { 4 }))
  $cur = ""; $curR = Measure-It $m $kind ""
  Write-Output ("{0,-40} {1,10}" -f "defaults", (F $curR 2))
  foreach ($w in ($win | Sort-Object { $_.gain } -Descending)) {
    $try = (($cur, $w.kv) | Where-Object { $_ }) -join " "
    $r = Measure-It $m $kind $try
    if ($r -ge $curR * 1.01) { Write-Output ("{0,-40} {1,10}  {2,-9} kept" -f ("+ " + $w.kv), (F $r 2), (Pct $r $curR)); $cur = $try; $curR = $r }
    else { Write-Output ("{0,-40} {1,10}  {2,-9} left out" -f ("+ " + $w.kv), (F $r 2), (Pct $r $curR)) }
  }
  Write-Output ("the best combination: " + $(if ($cur) { $cur } else { "the defaults" }))
  $both = @(@{ l = "defaults"; s = "" }); if ($cur) { $both += @{ l = "best"; s = $cur } }
  foreach ($b in $both) {
    $g = TG (Invoke-Runz $m $b.s @("--bench-pp", "128", "--n", "128"))
    $p512 = Best-PP (Invoke-Runz $m $b.s @("--bench-pp", "512", "--n", "0"))
    $p4k = Best-PP (Invoke-Runz $m $b.s @("--bench-pp", "4096", "--n", "0"))
    $p16 = if ($Quick) { "-" } else { F (Best-PP (Invoke-Runz $m $b.s @("--bench-pp", "16384", "--n", "0"))) }
    Write-Output ("{0,-9} tg128 {1,8}   pp512 {2,9}   pp4096 {3,9}   pp16384 {4,9}" -f $b.l, (F $g 2), (F $p512), (F $p4k), $p16)
  }
  return $cur
}

# --- D: drafting
function Drafting([string]$m) {
  Write-Output ""; Write-Output "=== D. drafting: one coding question, 256 tokens, the best of 2 runs (tok/s); every text against plain"
  $runs = [ordered]@{ "plain" = @(); "mtp (default drafts)" = @("--mtp") }
  foreach ($d in 2..7) { $runs["mtp, $d drafts"] = @("--mtp", "--mtp-drafts", "$d") }
  if ($DFlash) { $runs["dflash (auto)"] = @("--dflash", $DFlash, "--draft", "auto"); $runs["dflash only"] = @("--dflash", $DFlash, "--draft", "dflash") }
  Start-Phase "D" (2 * $runs.Count)
  $ref = $null; $plain = 0
  foreach ($k in $runs.Keys) {
    $best = 0; $o = $null
    foreach ($rep in 1..2) { $o = Invoke-Runz $m "" (@("--text-file", $question, "--n", "256", "--show") + $runs[$k]); $t = Gen-TPS $o; if ($t -gt $best) { $best = $t } }
    $txt = ($o -join "`n"); $h = if ($txt -match '(?s)----- text -----\n(.*?)\n----------------') { $Matches[1].GetHashCode() } else { 0 }
    if ($null -eq $ref) { $ref = $h }
    $acc = ($o | Select-String -Pattern 'MTP: \d+ drafts, \d+ accepted \([\d.]+ %\)' | Select-Object -Last 1)
    Write-Output ("{0,-24} {1,8} tok/s  {2,-9} {3}{4}" -f $k, (F $best 2), $(if ($k -eq "plain") { "" } else { Pct $best $plain }),
      $(if ($h -eq $ref) { "same text" } else { "TEXT DIFFERS from plain" }), $(if ($acc) { "; " + $acc.Matches[0].Value } else { "" }))
    if ($k -eq "plain") { $plain = $best }
  }
}

# --- T: the XMX tile per GEMM kind
function Tiles([string]$m, [bool]$lanes8) {
  $sets = if ($lanes8) { $TILES8 } else { $TILES16 }
  if ($Quick) { $sets = $sets[0..3] }
  Write-Output ""; Write-Output "=== T. the XMX tile per GEMM kind (the prompt's --prof, ms; the lower of 2 rounds, alternating order)"
  Start-Phase "T" (1 + 2 * $sets.Count)
  $null = Invoke-Runz $m "" @("--bench-pp", "4096", "--n", "0")  # (untimed: the first measured setting ran ~8 % slow without it)
  $names = @($sets | ForEach-Object { ($_ -split '\|')[0] }); $ms = @{}; $pp = @{}; $note = @{}; $kinds = [System.Collections.Generic.List[string]]::new()
  $rev = $sets.Clone(); [array]::Reverse($rev)
  foreach ($round in 1, 2) {
    foreach ($s in $(if ($round -eq 1) { $sets } else { $rev })) {
      $n = ($s -split '\|')[0]; $e = ($s -split '\|', 2)[1]
      $o = Invoke-Runz $m $e @("--bench-pp", "4096", "--n", "0", "--prof")
      if (-not $ms.ContainsKey($n)) { $ms[$n] = @{} }
      $rows = Prof-Rows $o
      foreach ($k in $rows.Keys) { if ($k -like "gemm_*") { if (-not $kinds.Contains($k)) { $kinds.Add($k) }; if (-not $ms[$n].ContainsKey($k) -or $rows[$k] -lt $ms[$n][$k]) { $ms[$n][$k] = $rows[$k] } } }
      $p = Best-PP $o; if (-not $pp.ContainsKey($n) -or $p -gt $pp[$n]) { $pp[$n] = $p }
      $t = $o | Where-Object { $_ -match 'H8 GEMMs on XMX with|gemm_xw.* differs|not supported' } | Select-Object -First 1; if ($t) { $note[$n] = $t.Trim() }
    }
  }
  $hdr = "{0,-16}" -f "GEMM"; foreach ($n in $names) { $hdr += " {0,15}" -f $n }; Write-Output ($hdr + "   fastest")
  $sd = 0.0; $sb = 0.0
  foreach ($k in $kinds) {
    $line = "{0,-16}" -f $k; $best = ""; $bv = [double]::MaxValue
    foreach ($n in $names) { if ($ms[$n].ContainsKey($k)) { $v = $ms[$n][$k]; $line += " {0,15}" -f (F $v); if ($v -lt $bv) { $bv = $v; $best = $n } } else { $line += " {0,15}" -f "-" } }
    $d = $ms[$names[0]][$k]; if ($d) { $sd += $d; $sb += $bv; $line += "   $best (" + (Pct $bv $d) + ")" } else { $line += "   $best" }
    Write-Output $line
  }
  $line = "{0,-16}" -f "prompt tok/s"; foreach ($n in $names) { $line += " {0,15}" -f $(if ($pp[$n]) { F $pp[$n] } else { "failed" }) }; Write-Output $line
  Write-Output ("all GEMMs at the default: {0} ms; each GEMM at its fastest tile: {1} ms ({2})" -f (F $sd), (F $sb), (Pct $sb $sd))
  foreach ($n in $names) { Write-Output ("  {0,-16} {1}" -f $n, $(if ($note.ContainsKey($n)) { $note[$n] } else { "(no tile line)" })) }
}

# --- K: where the time goes, the defaults against a settings set
function Kernels([string]$m, [string]$what, [string]$sset, [string]$label) {
  $a = if ($what -eq "prompt") { @("--bench-pp", "4096", "--n", "0", "--prof") } else { @("--bench-pp", "128", "--n", "64", "--prof-gen") }
  $o0 = Invoke-Runz $m "" $a; $k0 = Prof-Rows $o0
  $o1 = if ($sset) { Invoke-Runz $m $sset $a } else { $o0 }; $k1 = Prof-Rows $o1
  $r0 = if ($what -eq "prompt") { Best-PP $o0 } else { TG $o0 }; $r1 = if ($what -eq "prompt") { Best-PP $o1 } else { TG $o1 }
  Write-Output ("--- K. $label, $what by kernel (ms); defaults {0} tok/s, best {1} tok/s while profiled (slower than unprofiled runs)" -f (F $r0 2), (F $r1 2))
  $t0 = ($k0.Values | Measure-Object -Sum).Sum; $t1 = ($k1.Values | Measure-Object -Sum).Sum
  Write-Output ("{0,-22} {1,10} {2,7} {3,10} {4,7} {5,9}" -f "kernel", "defaults", "share", "best", "share", "change")
  $keys = @($k0.Keys) + @($k1.Keys | Where-Object { -not $k0.Contains($_) }); $i = 0
  foreach ($k in $keys) {
    if (++$i -gt 25) { Write-Output ("({0} more kernels)" -f ($keys.Count - 25)); break }
    $x = [double]$k0[$k]; $y = [double]$k1[$k]
    Write-Output ("{0,-22} {1,10} {2,6}% {3,10} {4,6}% {5,9}" -f $k, (F $x), (F $(if ($t0) { 100 * $x / $t0 } else { 0 })), (F $y),
      (F $(if ($t1) { 100 * $y / $t1 } else { 0 })), $(if ($x) { Pct $y $x } else { "" }))
  }
  Write-Output ("{0,-22} {1,10} {2,7} {3,10} {4,7} {5,9}" -f "total", (F $t0), "", (F $t1), "", (Pct $t1 $t0))
}

Write-Output ("inferzero gpu_sweep: {0}{1}; device {2}; phases {3}{4}; {5}" -f (Split-Path $Model -Leaf), $(if ($MoeModel) { ", " + (Split-Path $MoeModel -Leaf) } else { "" }),
  $(if ($Device) { $Device } else { "default" }), $Phases, $(if ($Quick) { ", quick" } else { "" }), [DateTime]::UtcNow.ToString("yyyy-MM-dd HH:mm 'UTC'"))
try { git -C $root log -1 --format='commit %h %ad' --date=short 2>$null } catch {}
Start-Phase "warm-up" 1
$o = Invoke-Runz $Model "" @("--bench-pp", "1024", "--n", "1")
$o | Where-Object { $_ -match '^inferzero: (GPU|XMX|H8|the H8|prompt attention)|^loaded:' } | Select-Object -First 8
$lanes8 = -not ($o | Where-Object { $_ -match 'no 8-lane sub-groups' })
$xmx = [bool]($o | Where-Object { $_ -match 'XMX on for the prompt GEMMs|H8 GEMMs on XMX' })
$genBest = ""; $ppBest = ""
if (Has "G") { $w = Sweep "G. generation" $Model "gen" $GEN; $genBest = Check-And-Combine $Model "gen" $w }
if (Has "P") { $w = Sweep "P. the prompt" $Model "pp" $PP; $ppBest = Check-And-Combine $Model "pp" $w }
if (Has "D") { Drafting $Model }
if (Has "T") { if ($xmx) { Tiles $Model $lanes8 } else { Write-Output ""; Write-Output "=== T. skipped: this GPU's log shows no XMX prompt GEMMs (no tiles to choose)" } }
if (Has "K") {
  Write-Output ""; Write-Output "=== K. where the time goes"; Start-Phase "K" 4
  Kernels $Model "prompt" $ppBest "the prompt's best combination"
  Kernels $Model "generation" $genBest "generation's best combination"
}
if ((Has "M") -and $MoeModel) {
  Start-Phase "warm-up" 1; $null = Invoke-Runz $MoeModel "" @("--bench-pp", "1024", "--n", "1")
  $w = Sweep "M. MoE generation" $MoeModel "gen" $MOE; $mg = Check-And-Combine $MoeModel "gen" $w
  $w = Sweep "M. MoE prompt" $MoeModel "pp" $MOE; $mp = Check-And-Combine $MoeModel "pp" $w
  if (Has "K") { Start-Phase "K" 4; Kernels $MoeModel "prompt" $mp "the MoE prompt's best combination"; Kernels $MoeModel "generation" $mg "MoE generation's best combination" }
}
Write-Progress -Activity "gpu_sweep" -Completed
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Output ""; Write-Output "done: paste everything above into the issue"
