# The contributor kit (docs/gemm-inventory.md table J; the user, 2026-11-27): one command on a contributor's card that
# writes one file to attach to the GitHub issue, with what the shared GEMM layer's inventory needs from that card:
#   A  which kernel and options each matrix product takes (ARC_GEMM_PLAN=1), and the start-up lines (XMX, H8, spills);
#   E  the speed: prompt 512 and 4,096 tokens, generation 128 tokens, two runs each;
#   G  the launches a generated token (ARC_LAUNCH_STATS=1) and the kernels' launch counts (--prof / --prof-gen);
#   F  what --ctx auto picks and the GPU memory the server reports for it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File bench\contrib_kit.ps1 -Model <a.gguf>[,<b.gguf>] [-Device N]
#       [-Out <file>] [-Quick] [-SelfTest]
# -Device N: which GPU (ZE_AFFINITY_MASK). -Quick: one speed run each instead of two. About 10-20 minutes a model.
# -SelfTest: the name hiding over fixed sample lines, no GPU run; exit 1 on any line that does not come out as expected.
# bench/contrib_kit.sh is the same for Linux; the two must stay the same (change both together). Every GPU run goes
# through bench\gpu_guard.ps1. It starts only when no other GPU job runs (a server of yours included: close it first),
# and stops only the server it started. The runs use the defaults: the shell's ARC_* / INFERZERO_* switches are cleared
# for them (but ARC_DATA_DIR) and put back after, their names in the file.
# Nothing personal is written (the file is posted publicly): the models' folders become <model dir>, this folder
# <inferzero>, the temporary folder <temp> and the profile ~ (each in any casing, either slash, its 8.3 short form too),
# the user and computer names <user> / <host> wherever they stand as a word (a name with letters outside ASCII also as
# the console may mangle it), any other absolute path <path>\<model file> for a given model's file or <file>, and
# gpu_guard's own lines are left out. Windows PowerShell 5.1 and 7.
param(
  [string[]]$Model = @(),
  [string]$Device = "",
  [string]$Out = "",
  [switch]$Quick,
  [switch]$SelfTest
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
$serverz = Join-Path $root "bin\serverz.exe"
$guard = Join-Path $root "bench\gpu_guard.ps1"
$Model = @($Model | ForEach-Object { $_ -split "," } | Where-Object { $_ })

# ---- the hiding (a list of folder -> placeholder pairs, the names, the last pass over absolute paths)
function Get-ShortPath([string]$p) {  # (the 8.3 form, as %TEMP% often has it: C:\Users\JOHNSM~1\...; "" when none)
  try { if ($p -and (Test-Path -LiteralPath $p)) { return (New-Object -ComObject Scripting.FileSystemObject).GetFolder($p).ShortPath } } catch {}
  return ""
}
# (a name or folder as a regex: ASCII escaped, a letter outside ASCII as 1-4 of anything but a separator, since a
#  narrow-string program's output decoded as UTF-8 turns "é" into "?" or U+FFFD, or into two or three characters)
#  (character by character: a separator matches either slash, so "C:\Users\x" is found as "c:/users/x" too)
function Get-NamePattern([string]$n) {
  $sb2 = New-Object System.Text.StringBuilder
  foreach ($ch in $n.ToCharArray()) {
    if ($ch -eq '\' -or $ch -eq '/') { [void]$sb2.Append('[\\/]') }
    elseif ([int]$ch -lt 128) { [void]$sb2.Append([regex]::Escape([string]$ch)) }
    else { [void]$sb2.Append('[^\\/\s]{1,4}') }
  }
  return $sb2.ToString()
}
function New-HideList([string[]]$modelDirs, [string]$rootDir, [string[]]$tempDirs, [string[]]$profileDirs) {
  $pairs = New-Object System.Collections.ArrayList
  foreach ($d in $modelDirs) { if ($d) { [void]$pairs.Add(@($d, "<model dir>")) } }
  if ($rootDir) { [void]$pairs.Add(@($rootDir, "<inferzero>")) }
  foreach ($d in $tempDirs) { if ($d) { [void]$pairs.Add(@($d.TrimEnd('\'), "<temp>")) } }
  foreach ($d in $profileDirs) { if ($d) { [void]$pairs.Add(@($d.TrimEnd('\'), "~")) } }
  # (the longest first, so a folder inside another (the temporary folder inside the profile) wins)
  $sorted = New-Object System.Collections.ArrayList
  foreach ($p in @($pairs | Sort-Object { $_[0].Length } -Descending)) { [void]$sorted.Add($p) }
  return , $sorted
}
# (built with a foreach and ArrayList.Add, not a pipeline: a pipeline that keeps one pair hands back its two strings;
#  names under 3 characters are left, or "a" would eat the file)
function Get-Names([string]$user, [string[]]$computers) {
  $l = New-Object System.Collections.ArrayList
  $seen = @{}
  foreach ($p in (@(, @($user, "<user>")) + @($computers | ForEach-Object { , @($_, "<host>") }))) {
    if ($p[0] -and $p[0].Length -ge 3 -and -not $seen.ContainsKey($p[0].ToLower())) { $seen[$p[0].ToLower()] = 1; [void]$l.Add($p) }
  }
  # (the longest first: the DNS name before the bare computer name it holds)
  $sorted = New-Object System.Collections.ArrayList
  foreach ($p in @($l | Sort-Object { $_[0].Length } -Descending)) { [void]$sorted.Add($p) }
  return , $sorted
}
# An absolute path's end cannot be found once folder names may hold spaces ("My Models", "OneDrive - Contoso"), so
# from its start (a drive letter or \\) to the end of the line, or to a <...> the folder pass wrote, becomes <path>;
# its trailing spaces before such a <...> are kept. One exception: a given model's file name in that span gives
# <path>\<model file>, and what follows it is hidden the same way (the Researcher's review, 2026-11-28).
function Hide-Paths([string]$s, [string[]]$modelFiles) {
  $m = [regex]::Match($s, '(?<![\w<~>])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n<]*')
  if (-not $m.Success) { return $s }
  $span = $m.Value
  $pre = $s.Substring(0, $m.Index)
  $post = $s.Substring($m.Index + $m.Length)
  $best = $null
  foreach ($f in $modelFiles) {
    if (-not $f) { continue }
    $mm = [regex]::Match($span, '[\\/](' + [regex]::Escape($f) + ')(?![\w.-])', "IgnoreCase, RightToLeft")
    if ($mm.Success -and (-not $best -or $mm.Index -gt $best.Index)) { $best = $mm }
  }
  if ($best) {
    $g = $best.Groups[1]
    return $pre + "<path>\" + $g.Value + (Hide-Paths ($span.Substring($g.Index + $g.Length) + $post) $modelFiles)
  }
  $ws = $(if ($post) { [regex]::Match($span, '\s*$').Value } else { "" })
  return $pre + "<path>" + $ws + (Hide-Paths $post $modelFiles)
}
function Hide-Names([string]$s, $hideList, $nameList, [string[]]$modelFiles) {
  foreach ($h in $hideList) {  # (any casing and either slash: "c:\users\name", "C:/Users/Name")
    $s = [regex]::Replace($s, (Get-NamePattern $h[0]), $h[1].Replace('$', '$$'), "IgnoreCase")
  }
  # (then any absolute path left, a drive's or a share's, e.g. the expert store naming another model's files or a
  #  build's .tmp: Hide-Paths. Before the names: a name inside such a path would otherwise split it)
  $s = Hide-Paths $s $modelFiles
  foreach ($n in $nameList) {
    $s = [regex]::Replace($s, "(?<![A-Za-z0-9])" + (Get-NamePattern $n[0]) + "(?![A-Za-z0-9])", $n[1], "IgnoreCase")
  }
  return $s
}
# (gpu_guard's own lines: its command line holds the full paths; anywhere in a line, since its 500 ms copies can glue
#  a fragment of output in front of one)
function Test-Dropped([string]$l) { return $l -match 'gpu_guard: ' }

if ($SelfTest) {
  # fixed cases, checked against the expected text: a user name with a letter outside ASCII (also as mangled), the 8.3
  # temporary folder, this folder under "OneDrive - <company>", another model's file, a gpu_guard line, a short name
  $tp = "C:\Users\Jos" + [char]0xE9 + "\AppData\Local\Temp"
  $hl = New-HideList @("D:\llm\qwen") ("C:\Users\Jos" + [char]0xE9 + "\OneDrive - Contoso\inferzero") @($tp, "C:\Users\JOS~1\AppData\Local\Temp") @(("C:\Users\Jos" + [char]0xE9), "C:\Users\JOS~1")
  $nl = Get-Names ("Jos" + [char]0xE9) @("DESKTOP-AB1", "desktop-ab1.corp.example")
  $mf = @("Qwen3.5-4B-UD-Q4_K_XL.gguf")
  $cases = @(
    @(("cache at C:\Users\Jos" + [char]0xE9 + "\.inferzero\kernels"), "cache at ~\.inferzero\kernels"),
    @(("cache at c:/users/JOS" + [char]0xC9 + "/.inferzero/kernels"), "cache at ~/.inferzero/kernels"),
    @("cache at C:\Users\Jos?\.inferzero\kernels", "cache at ~\.inferzero\kernels"),
    @("built in C:\Users\JOS~1\AppData\Local\Temp\b1.tmp", "built in <temp>\b1.tmp"),
    @(("runz at C:\Users\Jos" + [char]0xE9 + "\OneDrive - Contoso\inferzero\bin\runz.exe"), "runz at <inferzero>\bin\runz.exe"),
    @("expert file E:\other\Flash-Next-IQ3_S.gguf.experts, model D:\llm\qwen\Qwen3.5-4B-UD-Q4_K_XL.gguf", "expert file <path> <model dir>\Qwen3.5-4B-UD-Q4_K_XL.gguf"),
    @("a copy F:\backup\Qwen3.5-4B-UD-Q4_K_XL.gguf", "a copy <path>\Qwen3.5-4B-UD-Q4_K_XL.gguf"),
    @(("Jos" + [char]0xE9 + " on DESKTOP-AB1 (desktop-ab1.corp.example), \\fileserver\share\x"), "<user> on <host> (<host>), <path>"),
    @("Driver D on desktop-ab1", "Driver D on <host>"),
    @(("also D:\data\Jos" + [char]0xE9 + "\x and E:\DESKTOP-AB1\y"), "also <path>"),
    @("removed E:\My Models\private run.gguf.tmp (3.1 GB)", "removed <path>"),
    @("memory (--ctx auto): device 20.42 GiB = encoders / speech 1.05 + MTP / drafter 0.00, 2 / 3 runs, 1024.0 tokens/s",
      "memory (--ctx auto): device 20.42 GiB = encoders / speech 1.05 + MTP / drafter 0.00, 2 / 3 runs, 1024.0 tokens/s"),
    @("a copy F:\back up\Qwen3.5-4B-UD-Q4_K_XL.gguf, done", "a copy <path>\Qwen3.5-4B-UD-Q4_K_XL.gguf, done"),
    @("from \\nas\My Share\x.gguf.experts to G:\a b\c", "from <path>"))
  $bad = 0
  foreach ($c in $cases) {
    $o2 = Hide-Names $c[0] $hl $nl $mf
    $ok = $o2 -ceq $c[1]
    if (-not $ok) { $bad++ }
    Write-Host "$(if ($ok) { 'ok   ' } else { 'WRONG' }) in:  $($c[0])"
    Write-Host "      out: $o2$(if (-not $ok) { "   (expected: $($c[1]))" })"
  }
  $g = "gpu_guard: C:\x\runz.exe ... (timeout 1800 s, log C:\Users\x\AppData\Local\Temp\g.log)"
  $okg = Test-Dropped ("tg128: 5 ms" + $g)
  if (-not $okg) { $bad++ }
  Write-Host "$(if ($okg) { 'ok   ' } else { 'WRONG' }) a gpu_guard line, glued to output, is left out"
  # (a user name under 3 characters is left, and only it: the computer name is still hidden, no letter replaced)
  $short = Get-Names "jo" @("DESKTOP-AB1")
  $o3 = Hide-Names "Driver D jo on DESKTOP-AB1" (New-HideList @() "" @() @()) $short @()
  $oks = ($short.Count -eq 1) -and ($o3 -ceq "Driver D jo on <host>")
  if (-not $oks) { $bad++ }
  Write-Host "$(if ($oks) { 'ok   ' } else { 'WRONG' }) short user name: $o3 ($($short.Count) pair)"
  Write-Host "self-test: $(if ($bad) { "$bad WRONG" } else { 'all ok' })"
  if ($bad) { exit 1 }
  return
}

if ($Model.Count -eq 0) { throw "give -Model <a.gguf>[,<b.gguf>]: the models you run (the 27B, the 35B-A3B, Inkling 2 ...)" }
foreach ($m in $Model) { if (-not (Test-Path $m)) { throw "model not found: $m" } }
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found: run setup, or build it (.\build.ps1 zero\runz.cpp zero\serverz.cpp)" }
# (no other GPU job, a server of the user's included: gpu_guard would refuse every step, and the memory step must not
#  stop anything it did not start)
$busy = @(Get-Process runz, serverz, attn_bench, xw_peak, gpu_bench, ref_forward -ErrorAction SilentlyContinue)
if ($busy.Count) {
  Write-Host ("Another GPU job runs: " + (($busy | ForEach-Object { "$($_.ProcessName) ($($_.Id))" }) -join ", ") +
              ". Close it (a server of yours too) and start the kit again.") -ForegroundColor Red
  exit 2
}
if (-not $Out) { $Out = Join-Path (Get-Location) ("inferzero-kit-" + (Get-Date -Format "yyyyMMdd-HHmm") + ".txt") }
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("inferzero-kit-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$sb = New-Object System.Text.StringBuilder
$temps = @($env:TEMP, $env:TMP, [IO.Path]::GetTempPath()) | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') } | Sort-Object -Unique
$tempAll = @($temps) + @($temps | ForEach-Object { Get-ShortPath $_ } | Where-Object { $_ })
$profAll = @($env:USERPROFILE, (Get-ShortPath $env:USERPROFILE)) | Where-Object { $_ }
$modelDirs = @($Model | ForEach-Object { Split-Path (Resolve-Path $_).Path -Parent }) + @($Model | ForEach-Object { Get-ShortPath (Split-Path (Resolve-Path $_).Path -Parent) } | Where-Object { $_ })
$hide = New-HideList $modelDirs $root $tempAll $profAll
$hostsAll = @($env:COMPUTERNAME)
try { $hostsAll += [System.Net.Dns]::GetHostName() } catch {}
try { $hostsAll += [System.Net.Dns]::GetHostEntry([System.Net.Dns]::GetHostName()).HostName } catch {}
$names = Get-Names $env:USERNAME $hostsAll
$modelFiles = @($Model | ForEach-Object { [IO.Path]::GetFileName($_) })
function Add-Line([string]$s) {
  if (Test-Dropped $s) { return }
  $s = Hide-Names $s $hide $names $modelFiles
  [void]$sb.AppendLine($s); Write-Host $s
}

# the defaults: the shell's switches cleared for the runs, put back at the end
$saved = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -match "^(ARC_|INFERZERO_)" -and $_.Name -ne "ARC_DATA_DIR" })) {
  $saved[$v.Name] = $v.Value
  Remove-Item "Env:$($v.Name)"
}
# (driver and runtime switches stay as the user set them: their names, not values, go into the file as a warning)
$rtVars = @(Get-ChildItem Env: | Where-Object { $_.Name -match "^(ZE_|IGC_|NEO)" -and $_.Name -ne "ZE_AFFINITY_MASK" } | ForEach-Object { $_.Name } | Sort-Object)
$hadMask = Test-Path Env:ZE_AFFINITY_MASK
$oldMask = $env:ZE_AFFINITY_MASK
if ($Device) { $env:ZE_AFFINITY_MASK = $Device }

# one runz run through gpu_guard: its lines matching $keep go into the file under a heading
function Invoke-Step([string]$title, [string]$m, [string[]]$a, [hashtable]$envs, [string]$keep, [int]$t = 1800) {
  Add-Line ""
  Add-Line "## $title"
  foreach ($e in $envs.Keys) { Set-Item "Env:$e" $envs[$e] }
  # (Windows PowerShell 5.1 with "Stop" ends the script at a child's first stderr line: "Continue" here, so a failed
  #  run is written as its exit code and last lines. The console in UTF-8 for the run, so gpu_guard's copy of the
  #  output reaches this script as it wrote it and a name outside ASCII is matched. -ExecutionPolicy Bypass: a pwsh 7
  #  caller's 5.1 child would otherwise refuse gpu_guard.ps1)
  $eap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $enc = [Console]::OutputEncoding
  try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false } catch {}
  $o = & powershell -NoProfile -ExecutionPolicy Bypass -File $guard -TimeoutS $t -- $runz $m @a 2>&1 | Out-String
  $code = $LASTEXITCODE
  try { [Console]::OutputEncoding = $enc } catch {}
  $ErrorActionPreference = $eap
  foreach ($e in $envs.Keys) { Remove-Item "Env:$e" -ErrorAction SilentlyContinue }
  Add-Line ("runz <model> " + ($a -join " ") + $(if ($envs.Count) { "  (" + (($envs.Keys | ForEach-Object { "$_=$($envs[$_])" }) -join " ") + ")" } else { "" }))
  foreach ($l in ($o -split "`r?`n")) { if ($l -match $keep) { Add-Line $l.TrimEnd() } }
  if ($code -ne 0) { Add-Line "(exit code $code)"; foreach ($l in (($o -split "`r?`n") | Where-Object { -not (Test-Dropped $_) } | Select-Object -Last 12)) { Add-Line $l.TrimEnd() } }
}
# the processes started by one process, its children's children too
function Get-Descendants([int]$ppid) {
  $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
  $out = New-Object System.Collections.ArrayList
  $todo = New-Object System.Collections.Queue
  $todo.Enqueue($ppid)
  while ($todo.Count) {
    $q = $todo.Dequeue()
    foreach ($c in ($all | Where-Object { $_.ParentProcessId -eq $q })) { [void]$out.Add($c); $todo.Enqueue([int]$c.ProcessId) }
  }
  return , $out
}

Write-Host "The kit starts a server of its own for its memory step and stops only that one."
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
  if ($rtVars.Count) { Add-Line "warning: driver / runtime variables set (kept for the runs): $($rtVars -join ' ')" }
  $runs = $(if ($Quick) { 1 } else { 2 })
  foreach ($m in $Model) {
    Add-Line ""
    Add-Line "# model: $([IO.Path]::GetFileName($m)) ($([math]::Round((Get-Item $m).Length / 1GB, 2)) GB)"
    # A: the plan log and the start lines (gemm-plan anywhere in a line: a glued fragment must not hide one)
    Invoke-Step "A. kernels (ARC_GEMM_PLAN=1)" $m @("--bench-pp", "512", "--n", "16") @{ ARC_GEMM_PLAN = "1" } "(gemm-plan|^inferzero:|^loaded:)"
    # E: speed
    for ($i = 1; $i -le $runs; $i++) {
      Invoke-Step "E. speed, prompt 512 + generation 128 (run $i)" $m @("--bench-pp", "512", "--n", "128") @{} "^(pp\d|tg\d|prompt:|generation:)"
      Invoke-Step "E. speed, prompt 4096 (run $i)" $m @("--bench-pp", "4096", "--n", "1") @{} "^(pp\d|prompt:)"
    }
    # G: launches and the kernels' counts
    Invoke-Step "G. launches a generated token (ARC_LAUNCH_STATS=1)" $m @("--bench-pp", "512", "--n", "32") @{ ARC_LAUNCH_STATS = "1" } "^(launches:|tg\d)"
    Invoke-Step "G. generation profile (--prof-gen)" $m @("--bench-pp", "512", "--n", "32", "--prof-gen") @{} "(launches\)|tg steps)"
    Invoke-Step "G. prompt profile (--prof)" $m @("--bench-pp", "512", "--n", "1", "--prof") @{} "launches\)"
    # F: --ctx auto in the server: its memory line, then the server it started is stopped (by process, not by name)
    Add-Line ""
    Add-Line "## F. memory (serverz --ctx auto)"
    $log = Join-Path $tmp "serverz.log"
    Remove-Item $log, "$log.out" -ErrorAction SilentlyContinue
    $p = Start-Process powershell -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $guard, "-TimeoutS", "900", "--",
                                                  $serverz, "-m", $m, "--port", "18999", "--ctx", "auto") -PassThru -WindowStyle Hidden `
                                -RedirectStandardOutput "$log.out" -RedirectStandardError $log
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt 840) {
      Start-Sleep -Milliseconds 500
      if ((Test-Path $log) -and (Select-String -Path $log, "$log.out" -Pattern "listening on" -Quiet -ErrorAction SilentlyContinue)) { break }
      if ($p.HasExited) { break }
    }
    foreach ($c in (Get-Descendants $p.Id)) { if ($c.Name -eq "serverz.exe") { Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue } }
    $p.WaitForExit(30000) | Out-Null
    # (each line once: gpu_guard copies the server's lines to both of its outputs)
    $seenF = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($f in @($log, "$log.out")) {
      if (Test-Path $f) {
        foreach ($l in (Get-Content -Encoding UTF8 $f)) {
          if ($l -notmatch "memory \(--ctx auto\)|--ctx auto|loaded|context|error|failed") { continue }
          $l2 = ($l -replace '^\S+ \S+ [IWE] ', '').TrimEnd()
          if ($seenF.Add($l2)) { Add-Line $l2 }
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
