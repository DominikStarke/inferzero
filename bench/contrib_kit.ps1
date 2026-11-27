# The contributor kit (docs/gemm-inventory.md table J; the user, 2026-11-27): one command on a contributor's card that
# writes one file to attach to the GitHub issue, with what the shared GEMM layer's inventory needs from that card:
#   A  which kernel and options each matrix product takes (ARC_GEMM_PLAN=1), and the start-up lines (XMX, H8, spills);
#   E  the speed: prompt 512 and 4,096 tokens, generation 128 tokens, two runs each;
#   G  the launches a generated token (ARC_LAUNCH_STATS=1) and the kernels' launch counts (--prof / --prof-gen);
#   F  what --ctx auto picks and the GPU memory the server reports for it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File bench\contrib_kit.ps1 -Model <a.gguf>[,<b.gguf>] [-Device N]
#       [-Out <file>] [-Quick]
# -Device N: which GPU (ZE_AFFINITY_MASK). -Quick: one speed run each instead of two. About 10-20 minutes a model.
# bench/contrib_kit.sh is the same for Linux; the two must stay the same (change both together). Every GPU run goes
# through bench\gpu_guard.ps1. The runs use the defaults: the shell's ARC_* / INFERZERO_* switches are cleared for
# them (but ARC_DATA_DIR) and put back after, their names in the file. Nothing personal is written: no user or host
# name, no paths but the model's file name. Windows PowerShell 5.1 and 7.
param(
  [string[]]$Model = @(),
  [string]$Device = "",
  [string]$Out = "",
  [switch]$Quick,
  [switch]$SelfTest  # (the name hiding over sample lines, with no GPU run: prints each line before and after)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
$serverz = Join-Path $root "bin\serverz.exe"
$guard = Join-Path $root "bench\gpu_guard.ps1"
$Model = @($Model | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($SelfTest -and $Model.Count -eq 0) { $Model = @($PSCommandPath) }  # (a file that exists, for its folder)
if ($Model.Count -eq 0) { throw "give -Model <a.gguf>[,<b.gguf>]: the models you run (the 27B, the 35B-A3B, Inkling 2 ...)" }
foreach ($m in $Model) { if (-not (Test-Path $m)) { throw "model not found: $m" } }
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found: run setup, or build it (.\build.ps1 zero\runz.cpp zero\serverz.cpp)" }
if (-not $Out) { $Out = Join-Path (Get-Location) ("inferzero-kit-" + (Get-Date -Format "yyyyMMdd-HHmm") + ".txt") }
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("inferzero-kit-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$sb = New-Object System.Text.StringBuilder
# (no user name in the file: the user's folder becomes ~ and each model's folder <model dir>, wherever a line has them,
#  e.g. the kernel cache's path in a start line)
$hide = @(@($Model | ForEach-Object { Split-Path (Resolve-Path $_).Path -Parent } | Sort-Object Length -Descending) |
          ForEach-Object { , @($_, "<model dir>") })  # (the comma: each pair stays one item of the list)
$hide += , @($env:USERPROFILE, "~")
# (then the user and computer names wherever they stand as a word, a folder of a drive or OneDrive's included; names
#  under 3 characters are left, or "a" would eat the file)
#  (built with a foreach and ArrayList.Add, not a pipeline: a pipeline that keeps one pair hands back its two strings,
#  and the foreach below would then replace single letters; the Researcher's review, a 2-letter user name)
function Get-Names([string]$user, [string]$computer) {
  $l = New-Object System.Collections.ArrayList
  foreach ($p in @(@($user, "<user>"), @($computer, "<host>"))) { if ($p[0] -and $p[0].Length -ge 3) { [void]$l.Add($p) } }
  return , $l
}
$names = Get-Names $env:USERNAME $env:COMPUTERNAME
function Hide-Names([string]$s, $nameList = $null) {
  if ($null -eq $nameList) { $nameList = $names }
  foreach ($h in $hide) {  # (any casing and either slash: "c:\users\name", "C:/Users/Name")
    if (-not $h[0]) { continue }
    foreach ($p in @($h[0], $h[0].Replace("\", "/"))) {
      $s = [regex]::Replace($s, [regex]::Escape($p), $h[1].Replace('$', '$$'), "IgnoreCase")
    }
  }
  foreach ($n in $nameList) {
    $s = [regex]::Replace($s, "(?<![A-Za-z0-9])" + [regex]::Escape($n[0]) + "(?![A-Za-z0-9])", $n[1], "IgnoreCase")
  }
  return $s
}
function Add-Line([string]$s) {
  $s = Hide-Names $s
  [void]$sb.AppendLine($s); Write-Host $s
}
if ($SelfTest) {
  $md = Split-Path (Resolve-Path $Model[0]).Path -Parent
  foreach ($t in @("cache at $env:USERPROFILE\.inferzero\kernels", "cache at $($env:USERPROFILE.ToLower())\x",
                   "cache at $($env:USERPROFILE.ToUpper().Replace('\', '/'))/x", "model at $($md.ToUpper())\m.gguf",
                   "D:\$($env:USERNAME.ToUpper())\models and E:\data\$env:USERNAME-x\y", "on $env:COMPUTERNAME ($($env:COMPUTERNAME.ToLower()))")) {
    Write-Host "in:  $t"
    Write-Host "out: $(Hide-Names $t)"
  }
  # (a user name under 3 characters is left, and only it: the host is still hidden, and no single letter is replaced)
  $short = Get-Names "jo" $env:COMPUTERNAME
  $t = "D:\models on $env:COMPUTERNAME, E:\x"
  $o2 = Hide-Names $t $short
  Write-Host "in:  $t   (user 'jo')"
  Write-Host "out: $o2"
  $ok = ($short.Count -eq 1) -and ($o2 -eq "D:\models on <host>, E:\x")
  Write-Host "short user name: $(if ($ok) { 'ok' } else { 'WRONG' }) ($($short.Count) pair)"
  if (-not $ok) { exit 1 }
  return
}

# the defaults: the shell's switches cleared for the runs, put back at the end
$saved = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -match "^(ARC_|INFERZERO_)" -and $_.Name -ne "ARC_DATA_DIR" })) {
  $saved[$v.Name] = $v.Value
  Remove-Item "Env:$($v.Name)"
}
$hadMask = Test-Path Env:ZE_AFFINITY_MASK
$oldMask = $env:ZE_AFFINITY_MASK
if ($Device) { $env:ZE_AFFINITY_MASK = $Device }

# one runz run through gpu_guard: its lines matching $keep go into the file under a heading
function Invoke-Step([string]$title, [string]$m, [string[]]$a, [hashtable]$envs, [string]$keep, [int]$t = 1800) {
  Add-Line ""
  Add-Line "## $title"
  foreach ($e in $envs.Keys) { Set-Item "Env:$e" $envs[$e] }
  # (Windows PowerShell 5.1 with "Stop" ends the script at a child's first stderr line: "Continue" here, so a failed
  #  run is written as its exit code and last lines)
  $eap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $o = & powershell -NoProfile -File $guard -TimeoutS $t -- $runz $m @a 2>&1 | Out-String
  $code = $LASTEXITCODE
  $ErrorActionPreference = $eap
  foreach ($e in $envs.Keys) { Remove-Item "Env:$e" -ErrorAction SilentlyContinue }
  Add-Line ("runz <model> " + ($a -join " ") + $(if ($envs.Count) { "  (" + (($envs.Keys | ForEach-Object { "$_=$($envs[$_])" }) -join " ") + ")" } else { "" }))
  foreach ($l in ($o -split "`r?`n")) { if ($l -match $keep) { Add-Line $l.TrimEnd() } }
  if ($code -ne 0) { Add-Line "(exit code $code)"; foreach ($l in (($o -split "`r?`n") | Select-Object -Last 12)) { Add-Line $l.TrimEnd() } }
}

Write-Host "Close any serverz you run before starting: the kit starts one for its memory step and stops it."
try {
  $rel = "(not a package)"
  $pb = Join-Path $root "bin\PREBUILT.txt"
  if (Test-Path $pb) { $rel = (Get-Content -Encoding UTF8 $pb | Select-Object -First 1) }
  $commit = ""
  try { $commit = (& git -C $root rev-parse --short HEAD 2>$null) } catch {}
  Add-Line "# inferzero contributor kit, $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
  Add-Line "release: $rel$(if ($commit) { ", commit $commit" })"
  Add-Line "OS: Windows $([Environment]::OSVersion.Version)"
  try {
    $cpu = (Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1).Name
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory / 1GB)
    Add-Line "CPU: $cpu; RAM: $ram GB"
    foreach ($g in (Get-CimInstance Win32_VideoController -ErrorAction Stop)) { Add-Line "GPU: $($g.Name), driver $($g.DriverVersion), PCI $(if ("$($g.PNPDeviceID)" -match 'DEV_([0-9A-Fa-f]{4})') { $Matches[1] })" }
  } catch { Add-Line "(no system information: $($_.Exception.Message))" }
  Add-Line "device: $(if ($Device) { "ZE_AFFINITY_MASK=$Device" } else { 'the default' })"
  Add-Line "switches cleared for the runs: $(if ($saved.Count) { (@($saved.Keys | Sort-Object) -join ' ') } else { '-' })"
  $runs = $(if ($Quick) { 1 } else { 2 })
  foreach ($m in $Model) {
    Add-Line ""
    Add-Line "# model: $([IO.Path]::GetFileName($m)) ($([math]::Round((Get-Item $m).Length / 1GB, 2)) GB)"
    # A: the plan log and the start lines
    Invoke-Step "A. kernels (ARC_GEMM_PLAN=1)" $m @("--bench-pp", "512", "--n", "16") @{ ARC_GEMM_PLAN = "1" } "^(gemm-plan|inferzero:|loaded:)"
    # E: speed
    for ($i = 1; $i -le $runs; $i++) {
      Invoke-Step "E. speed, prompt 512 + generation 128 (run $i)" $m @("--bench-pp", "512", "--n", "128") @{} "^(pp\d|tg\d|prompt:|generation:)"
      Invoke-Step "E. speed, prompt 4096 (run $i)" $m @("--bench-pp", "4096", "--n", "1") @{} "^(pp\d|prompt:)"
    }
    # G: launches and the kernels' counts
    Invoke-Step "G. launches a generated token (ARC_LAUNCH_STATS=1)" $m @("--bench-pp", "512", "--n", "32") @{ ARC_LAUNCH_STATS = "1" } "^(launches:|tg\d)"
    Invoke-Step "G. generation profile (--prof-gen)" $m @("--bench-pp", "512", "--n", "32", "--prof-gen") @{} "(launches\)|tg steps)"
    Invoke-Step "G. prompt profile (--prof)" $m @("--bench-pp", "512", "--n", "1", "--prof") @{} "launches\)"
    # F: --ctx auto in the server: its memory line, then the server is stopped
    Add-Line ""
    Add-Line "## F. memory (serverz --ctx auto)"
    $log = Join-Path $tmp "serverz.log"
    Remove-Item $log, "$log.out" -ErrorAction SilentlyContinue
    $p = Start-Process powershell -ArgumentList @("-NoProfile", "-File", $guard, "-TimeoutS", "900", "--", $serverz, "-m", $m,
                                                  "--port", "18999", "--ctx", "auto") -PassThru -WindowStyle Hidden `
                                -RedirectStandardOutput "$log.out" -RedirectStandardError $log
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt 840) {
      Start-Sleep -Milliseconds 500
      if ((Test-Path $log) -and (Select-String -Path $log, "$log.out" -Pattern "listening on" -Quiet -ErrorAction SilentlyContinue)) { break }
      if ($p.HasExited) { break }
    }
    Get-Process serverz -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $serverz } | Stop-Process -Force
    $p.WaitForExit(30000) | Out-Null
    foreach ($f in @($log, "$log.out")) {
      if (Test-Path $f) {
        foreach ($l in (Get-Content -Encoding UTF8 $f)) {
          if ($l -match "^gpu_guard:") { continue }  # (its own command line)
          if ($l -match "memory \(--ctx auto\)|--ctx auto|loaded|context|error|failed") { Add-Line ($l -replace '^\S+ \S+ [IWE] ', '').TrimEnd() }
        }
      }
    }
  }
} finally {
  foreach ($n in $saved.Keys) { Set-Item "Env:$n" $saved[$n] }
  if ($hadMask) { $env:ZE_AFFINITY_MASK = $oldMask } elseif ($Device) { Remove-Item Env:ZE_AFFINITY_MASK -ErrorAction SilentlyContinue }
  [IO.File]::WriteAllText($Out, $sb.ToString().Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding $false))
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
Write-Host ""
Write-Host "Done. Please attach this file to the GitHub issue: $Out"
