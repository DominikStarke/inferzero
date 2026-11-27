# Which kinds of PC this one is, from Windows' display adapters: the words of scripts\options.tsv's best_for column.
# Dot-sourced by zero-server.ps1 (zero-server options) and zero-launcher.ps1 (the options section), so the two agree.
# Windows PowerShell 5.1 or 7.
#   igpu-xe2      Lunar Lake / Panther Lake iGPUs (Arc 130V / 140V, B370 / B390: Xe2 / Xe3, 16-lane XMX)
#   igpu-ptl      Panther Lake's B370 / B390 too (besides igpu-xe2)
#   igpu-140t     the Arc 140T (Xe-LPG Plus, 8-lane XMX: the 140T preset)
#   igpu-xe-lpg   Xe-LPG iGPUs by PCI device ID 0x7D00-0x7DFF, as the engine's zrt::xe_lpg (Meteor Lake's Arc Graphics,
#                 Arrow Lake-S's "Intel(R) Graphics"), the 140T above excepted
#   igpu-xe-lp    any other Intel iGPU (Tiger / Alder / Raptor Lake's Iris Xe and UHD: no H8 there, no NPU)
#   alchemist     discrete Arc A-series (A770, A750, A580, ...)
#   alchemist-a770  an A770 too (the A-series card with 16 GB; best-setups.tsv's need a770)
#   b-series      discrete Arc B-series (B580, B570, Pro B50 / B60 / B65 / B70)
#   multi-gpu     two or more discrete Arc cards
# (the iGPU names first: Panther Lake's B370 / B390 have a B and three digits too, and are no discrete card; the
#  Researcher's review, 2026-11-21)
function Get-PcKinds {
  $k = @(); $discrete = 0
  $vcs = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "Intel" })
  foreach ($vc in $vcs) {
    $g = "$($vc.Name)"
    if ($g -match 'B3[0-9]0|140V|130V|Xe2') {
      $k += "igpu-xe2"
      if ($g -match 'B3[0-9]0') { $k += "igpu-ptl" }  # (Panther Lake's B370 / B390 alone: best-setups.tsv's rows for it, 2026-11-22)
    }
    elseif ($g -match '140T') { $k += "igpu-140t" }
    elseif ($g -match 'Arc\S* A\d{3}') {
      $k += "alchemist"; $discrete++
      if ($g -match 'A770') { $k += "alchemist-a770" }  # (best-setups.tsv's need a770: the only Alchemist with 16 GB, #99)
    }
    elseif ($g -match 'Arc\S* (Pro )?B\d{2,3}\b') { $k += "b-series"; $discrete++ }
    elseif ($g -match 'Arc|Graphics') {
      # (by the PCI device ID, not the name, 2026-11-27: "Intel(R) Iris(R) Xe Graphics" / "UHD Graphics" matched
      #  'Graphics' and were taken for Meteor Lake, with its options and best-setups.tsv's NPU row on PCs without one;
      #  Arrow Lake-S says only "Intel(R) Graphics". 0x7Dxx: Xe-LPG, as the engine's zrt::xe_lpg)
      $dev = if ("$($vc.PNPDeviceID)" -match 'DEV_([0-9A-Fa-f]{4})') { $Matches[1].ToUpper() } else { "" }
      # (0x7D51 / 0x7DD1: Arrow Lake-H's 140T, often named only "Intel(R) Arc(TM) Graphics": its own kind, as the
      #  engine's 140T preset; the Researcher's review, 2026-11-27)
      if ($dev -eq '7D51' -or $dev -eq '7DD1') { $k += "igpu-140t" }
      elseif ($dev -match '^7D') { $k += "igpu-xe-lpg" } else { $k += "igpu-xe-lp" }
    }
  }
  if ($discrete -gt 1) { $k += "multi-gpu" }
  # multi-same: two or more discrete Arc cards, all of one PCI device ID (tensor parallel takes only GPU 0's own: a
  # B60 + B70 or an A770 + A750 is refused without --split layer; the Researcher's review, 2026-11-22)
  if ($discrete -gt 1) {
    $ids = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match "Intel" -and $_.Name -notmatch 'B3[0-9]0|140V|130V|Xe2|140T' -and $_.Name -match 'Arc\S* (A\d{3}|(Pro )?B\d{2,3}\b)' } |
      ForEach-Object { if ("$($_.PNPDeviceID)" -match 'DEV_([0-9A-Fa-f]{4})') { $Matches[1].ToUpper() } else { "?" } })
    if ($ids.Count -gt 1 -and @($ids | Select-Object -Unique).Count -eq 1 -and $ids[0] -ne "?") { $k += "multi-same" }
  }
  return @($k | Select-Object -Unique)
}
# the discrete Arc cards (the launcher's Multi-GPU choice splits a model over them); 1 when there are none
function Get-DiscreteGpuCount {
  $n = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object {
      $_.Name -match "Intel" -and $_.Name -notmatch 'B3[0-9]0|140V|130V|Xe2|140T' -and $_.Name -match 'Arc\S* (A\d{3}|(Pro )?B\d{2,3}\b)' }).Count
  return [math]::Max(1, $n)
}
