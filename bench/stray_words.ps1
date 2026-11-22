# stray_words: counts the words from other languages in Japanese answers (2026-11-21, for issue #4's stray words: a
# Japanese answer with a Korean, Hebrew or Swedish word in it), so runs compare by number, not by eye.
#   .\bench\stray_words.ps1 -Path answers\*.txt          one answer per file (UTF-8)
#   .\bench\stray_words.ps1 -Path a.txt,b.txt -Show       and each stray piece with the text around it
# Counted per answer and per 1,000 characters:
#   script  characters of scripts no Japanese text uses: Hangul, Hebrew, Cyrillic, Greek, Arabic, Thai, Devanagari;
#   latin   words of 4+ Latin letters inside a Japanese sentence (one with kana), such as "Swedish" in the middle of
#           a Japanese line (code, names and English answers are not Japanese sentences and are not counted).
#   simp    Simplified Chinese characters that Japanese text cannot hold (2026-11-22, issue #4): not in code page 932
#           (Shift-JIS, Windows' Japanese set) and in bench/simplified_han.txt (manoreken2's list of Simplified forms,
#           such as U+6743 / U+8FD9 / U+8D75). Characters Japanese shares with Chinese (U+4E0E, U+5199) are in CP932
#           and never counted.
#   outside CP932  Han characters outside CP932 that are not on that list: rare characters Japanese and Chinese share, such
#           as the surname U+90A2 or U+6C10 (in JIS X 0213, so Japanese fonts show them), or other Simplified forms the
#           list lacks. Shown apart, not called Simplified.
#           When a file has "=== thinking" and "=== answer" lines (a saved server answer), simp and "outside CP932" count the
#           answer part and simp_think the thinking part apart: the thinking is often English with Chinese names.
# Not counted: Chinese written with characters Japanese also uses (the two share them; telling them apart needs a
# model, not a character table).
param(
  [Parameter(Mandatory = $true)][string[]]$Path,
  [switch]$Show
)
$ErrorActionPreference = "Stop"
# (CP932 with an empty replacement: a character it cannot hold encodes to 0 bytes. PowerShell 7 (.NET) needs the code
#  pages provider registered first; Windows PowerShell 5.1 (.NET Framework) has CP932 built in and no such type)
try { [Text.Encoding]::RegisterProvider([Text.CodePagesEncodingProvider]::Instance) } catch {}
$cp932 = [Text.Encoding]::GetEncoding(932, (New-Object Text.EncoderReplacementFallback("")), (New-Object Text.DecoderReplacementFallback("")))
$one = New-Object char[] 1
# (the Simplified forms outside CP932: bench/simplified_han.txt, hex code points, '#' lines are notes)
$simpSet = New-Object 'System.Collections.Generic.HashSet[int]'
$listFile = Join-Path $PSScriptRoot "simplified_han.txt"
if (-not (Test-Path $listFile)) { throw "stray_words: $listFile not found (it ships next to this script)" }
foreach ($l in [IO.File]::ReadAllLines($listFile)) {
  if ($l -match '^\s*#') { continue }
  foreach ($h in ($l -split '\s+')) { if ($h) { [void]$simpSet.Add([Convert]::ToInt32($h, 16)) } }
}
function Test-NotJapaneseHan([char]$ch) {
  $c = [int]$ch
  if (-not (($c -ge 0x4E00 -and $c -le 0x9FFF) -or ($c -ge 0x3400 -and $c -le 0x4DBF))) { return $false }
  $one[0] = $ch
  return $cp932.GetByteCount($one) -eq 0
}
# (the Han characters outside CP932 in s: simp the Simplified ones on the list, "outside CP932" the others)
function Measure-Simp([string]$s, [string]$where) {
  $r = @{ simp = 0; rare = 0 }
  for ($i = 0; $i -lt $s.Length; $i++) {
    if (Test-NotJapaneseHan $s[$i]) {
      $k = if ($simpSet.Contains([int]$s[$i])) { "simp" } else { "rare" }  # (shown as "outside CP932")
      $r[$k]++
      if ($Show) {
        $a = [Math]::Max(0, $i - 12)
        $script:pieces += "    [$k $where] " + $s[$i] + " in ..." + $s.Substring($a, [Math]::Min(30, $s.Length - $a)).Replace("`n", " ") + "..."
      }
    }
  }
  return $r
}
$ranges = @(
  @{ n = "Hangul"; r = @(@(0xAC00, 0xD7A3), @(0x1100, 0x11FF), @(0x3130, 0x318F)) },
  @{ n = "Hebrew"; r = @(, @(0x0590, 0x05FF)) },
  @{ n = "Cyrillic"; r = @(, @(0x0400, 0x04FF)) },
  @{ n = "Greek"; r = @(, @(0x0370, 0x03FF)) },
  @{ n = "Arabic"; r = @(, @(0x0600, 0x06FF)) },
  @{ n = "Thai"; r = @(, @(0x0E00, 0x0E7F)) },
  @{ n = "Devanagari"; r = @(, @(0x0900, 0x097F)) }
)
function Script-Of([int]$c) {
  foreach ($s in $ranges) { foreach ($r in $s.r) { if ($c -ge $r[0] -and $c -le $r[1]) { return $s.n } } }
  return $null
}
$kana = [regex]'[\u3040-\u30FF]'
$latin = [regex]'[A-Za-z\u00C0-\u024F]{4,}'
$files = @(foreach ($p in $Path) { Get-ChildItem -Path $p -File -ErrorAction Stop })
if (-not $files.Count) { throw "no files: $Path" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$rows = @(); $tc = 0; $ts = 0; $tl = 0; $tsimp = 0; $trare = 0; $nsimp = 0
foreach ($f in $files) {
  $t = [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::UTF8)
  $chars = 0; $script = 0; $by = @{}; $pieces = @()
  for ($i = 0; $i -lt $t.Length; $i++) {
    if ([char]::IsWhiteSpace($t[$i])) { continue }
    $chars++
    $s = Script-Of ([int]$t[$i])
    if ($s) {
      $script++; $by[$s] = 1 + [int]$by[$s]
      if ($Show -and ($i -eq 0 -or -not (Script-Of ([int]$t[$i - 1])))) {
        $a = [Math]::Max(0, $i - 12); $pieces += "    [$s] ..." + $t.Substring($a, [Math]::Min(36, $t.Length - $a)).Replace("`n", " ") + "..."
      }
    }
  }
  $lat = 0
  foreach ($sent in ($t -split '[\u3002\uFF01\uFF1F!?\n]')) {  # (escapes: Windows PowerShell 5.1 reads this file as ANSI)
    if (-not $kana.IsMatch($sent)) { continue }
    foreach ($m in $latin.Matches($sent)) {
      $lat++
      if ($Show) { $pieces += "    [latin] " + $m.Value + " in: " + $sent.Substring(0, [Math]::Min(60, $sent.Length)).Trim() }
    }
  }
  # (simp / outside CP932: the answer part, and the thinking part apart, when the file marks them; else the whole file)
  $mt = [regex]::Match($t, '(?im)^\s*===\s*thinking[^\n]*\n')
  $ma = [regex]::Match($t, '(?im)^\s*===\s*answer[^\n]*\n')
  $simpT = ""
  if ($ma.Success) {
    $ans = $t.Substring($ma.Index + $ma.Length)
    if ($mt.Success -and $mt.Index -lt $ma.Index) { $simpT = [string](Measure-Simp $t.Substring($mt.Index + $mt.Length, $ma.Index - $mt.Index - $mt.Length) "thinking").simp }
  } else {
    $ans = $t
  }
  $sr = Measure-Simp $ans "answer"
  $per = if ($chars) { 1000.0 * ($script + $lat) / $chars } else { 0 }
  $rows += [pscustomobject]@{ answer = $f.Name; chars = $chars; script = $script; latin = $lat; simp = $sr.simp; "outside CP932" = $sr.rare
    simp_think = $simpT; per1000 = [string]::Format($inv, "{0:F2}", $per); which = (($by.Keys | Sort-Object | ForEach-Object { "$_ $($by[$_])" }) -join ", ") }
  $tc += $chars; $ts += $script; $tl += $lat; $tsimp += $sr.simp; $trare += $sr.rare; if ($sr.simp) { $nsimp++ }
  if ($Show -and $pieces.Count) { Write-Output ("{0}:" -f $f.Name); $pieces | Select-Object -First 20 | ForEach-Object { Write-Output $_ } }
}
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
$clean = @($rows | Where-Object { $_.script -eq 0 -and $_.latin -eq 0 }).Count
Write-Output ([string]::Format($inv, "all {0} answers: {1} characters, {2} of other scripts, {3} Latin words in Japanese sentences: {4:F2} per 1,000 characters; {5} of {0} answers clean",
  $rows.Count, $tc, $ts, $tl, $(if ($tc) { 1000.0 * ($ts + $tl) / $tc } else { 0 }), $clean))
Write-Output ([string]::Format($inv, "Simplified Chinese characters (not in CP932, on bench\simplified_han.txt), in the answers: {0} in {1} of {2} answers; outside CP932 (not on the list, so not counted as Simplified: CP932 alone is too narrow to tell): {3}",
  $tsimp, $nsimp, $rows.Count, $trare))
