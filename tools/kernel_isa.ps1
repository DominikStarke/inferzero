# Compiles one OpenCL kernel for Meteor Lake with Intel's offline compiler (ocloc, from the oneAPI DPC++/C++ compiler
# install) and summarizes its GPU assembly: instruction count, DP4a, memory messages, scratch (spill) traffic.
# The PowerShell port of the former kernel_isa.py (2026-11-21, the user: no Python in the repository); the same
# summary. Windows PowerShell 5.1 or 7.
#   powershell -File tools\kernel_isa.ps1 FILE.cl KERNEL -Options "BUILD OPTIONS" [-OutDir out_dir]
# (-Options named: build options start with a dash, which PowerShell would otherwise take for a parameter)
# The .asm is kept in out_dir (default: a new folder in %TEMP%) for reading. $env:OCLOC: ocloc's path if not the one
# below. ocloc carries its own compiler version, which can differ slightly from the one in the graphics driver.
param([Parameter(Mandatory = $true)][string]$File, [Parameter(Mandatory = $true)][string]$Kernel,
  [Parameter(Mandatory = $true)][string]$Options, [string]$OutDir = "")
$ErrorActionPreference = "Stop"
$ocloc = if ($env:OCLOC) { $env:OCLOC } else { 'C:\Program Files (x86)\Intel\oneAPI\ocloc\2026.1\bin\ocloc.exe' }
$src = (Resolve-Path -LiteralPath $File).Path
$out = if ($OutDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir) } else { Join-Path ([IO.Path]::GetTempPath()) ("kisa_" + [Guid]::NewGuid().ToString("N").Substring(0, 8)) }
New-Item -ItemType Directory -Force $out | Out-Null
$old = $ErrorActionPreference; $ErrorActionPreference = "Continue"  # (5.1: a native program's stderr lines are not errors here)
try {
  $c = @(& $ocloc compile -file $src -device mtl -options $Options -output k -out_dir $out 2>&1 | ForEach-Object { "$_" })
  $rc = $LASTEXITCODE
} finally { $ErrorActionPreference = $old }
$notes = @($c | Where-Object { $_.Contains($Kernel) -or $_.ToLower().Contains("error") })
if ($rc -ne 0) {
  if ($notes.Count) { $notes } else { $c | Select-Object -Last 20 }
  exit 1
}
$dis = Join-Path $out "dis"
New-Item -ItemType Directory -Force $dis | Out-Null
Push-Location $dis
try {
  $ErrorActionPreference = "Continue"
  & $ocloc disasm -file (Join-Path $out "k_mtl.bin") -dump . 2>&1 | Out-Null
} finally { $ErrorActionPreference = $old; Pop-Location }
$asm = Join-Path $dis ".text.$Kernel.asm"
if (-not (Test-Path -LiteralPath $asm)) {
  "kernel not found in the binary: " + ((Get-ChildItem -LiteralPath $dis -Force | Where-Object { $_.Name.StartsWith(".text.") } | ForEach-Object { $_.Name }) -join ", ")
  exit 1
}
$lines = @([IO.File]::ReadAllLines($asm, (New-Object Text.UTF8Encoding $false)) | Where-Object { $_.Trim() -and -not $_.Trim().EndsWith(":") })
$instr = @($lines | Where-Object { -not $_.Trim().StartsWith("//") })
function Count-Of([string]$pat) { @($instr | Where-Object { $_ -match $pat }).Count }
"$Kernel [$Options]"
foreach ($n in $notes) { "  ocloc: " + $n.Trim() }
"  instructions {0} | dp4a {1} | send (memory) {2} | scratch (spill) {3} | nop/sync {4}" -f $instr.Count, (Count-Of '\bdp4a\b'), (Count-Of '\bsend'), (Count-Of 'scratch'), (Count-Of '\bnop\b|sync\.')
"  asm: $asm"
