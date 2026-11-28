# src/arc_switches.h is current: every ARC_* name a program reads is in it (scripts/gen-switch-list.ps1 -Check), so
# serverz's start line marks only names nothing reads. Run it again after adding a switch.
$ErrorActionPreference = "Stop"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\gen-switch-list.ps1") -Check
exit $LASTEXITCODE
