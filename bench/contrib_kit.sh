#!/usr/bin/env bash
# The contributor kit (docs/gemm-inventory.md table J; the user, 2026-11-27): one command on a contributor's card that
# writes one file to attach to the GitHub issue, with what the shared GEMM layer's inventory needs from that card:
#   A  which kernel and options each matrix product takes (ARC_GEMM_PLAN=1), and the start-up lines (XMX, H8, spills);
#   E  the speed: prompt 512 and 4,096 tokens, generation 128 tokens, two runs each;
#   G  the launches a generated token (ARC_LAUNCH_STATS=1) and the kernels' launch counts (--prof / --prof-gen);
#   F  what --ctx auto picks and the GPU memory the server reports for it.
#   bash bench/contrib_kit.sh --model <a.gguf>[,<b.gguf>] [--device N] [--out <file>] [--quick]
# --device N: which GPU (ZE_AFFINITY_MASK). --quick: one speed run each instead of two. About 10-20 minutes a model.
# bench/contrib_kit.ps1 is the same for Windows; the two must stay the same (change both together). Every GPU run goes
# through bench/gpu_guard.sh. The runs use the defaults: the shell's ARC_* / INFERZERO_* switches are cleared for them
# (but ARC_DATA_DIR), their names in the file. Nothing personal is written: no user or host name, no paths but the
# model's file name.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz" SERVERZ="$ROOT/bin/serverz" GUARD="$ROOT/bench/gpu_guard.sh"
MODELS="" DEVICE="" OUT="" QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODELS="${MODELS:+$MODELS,}$2"; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --quick) QUICK=1; shift;;
    *) echo "unknown option: $1 (--model a.gguf[,b.gguf] [--device N] [--out F] [--quick])" >&2; exit 1;;
  esac
done
[ -n "$MODELS" ] || { echo "give --model <a.gguf>[,<b.gguf>]: the models you run (the 27B, the 35B-A3B, Inkling 2 ...)" >&2; exit 1; }
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
for m in "${MODEL_LIST[@]}"; do [ -f "$m" ] || { echo "model not found: $m" >&2; exit 1; }; done
[ -x "$RUNZ" ] || { echo "bin/runz not found: build it (./build.sh runz serverz)" >&2; exit 1; }
[ -n "$OUT" ] || OUT="$PWD/inferzero-kit-$(date '+%Y%m%d-%H%M').txt"
TMP=$(mktemp -d)
: > "$OUT"

# (no user name in the file: each model's folder becomes <model dir> and the home folder ~, in any casing; then the
#  user and host names wherever they stand as a word, names under 3 characters left alone. One sed script, built here)
HIDE_SED="$TMP/hide.sed"
: > "$HIDE_SED"
sed_esc() { printf '%s' "$1" | sed 's/[][\/.^$*+?(){}|]/\\&/g'; }
for m in "${MODEL_LIST[@]}"; do
  printf 's/%s/<model dir>/gI\n' "$(sed_esc "$(cd "$(dirname "$m")" && pwd)")" >> "$HIDE_SED"
done
# (the home folder only where a path starts: a short one, /root, would match inside /data/root/x too)
PATH_START='(^|[[:space:]"'"'"'(=:])'
[ -n "${HOME:-}" ] && printf 's/%s%s/\\1~/gI\n' "$PATH_START" "$(sed_esc "$HOME")" >> "$HIDE_SED"
for nm in "${USER:-$(id -un 2>/dev/null)}:<user>" "$(hostname 2>/dev/null || uname -n):<host>"; do
  n="${nm%:*}" r="${nm##*:}"
  [ ${#n} -ge 3 ] || continue
  # (twice: a match takes the character before the next one, as in "/name/name")
  printf 's/(^|[^A-Za-z0-9])%s([^A-Za-z0-9]|$)/\\1%s\\2/gI\n' "$(sed_esc "$n")" "$r" >> "$HIDE_SED"
  printf 's/(^|[^A-Za-z0-9])%s([^A-Za-z0-9]|$)/\\1%s\\2/gI\n' "$(sed_esc "$n")" "$r" >> "$HIDE_SED"
done
add() {  # LINE: into the file and to the console, the names hidden
  printf '%s\n' "$1" | sed -E -f "$HIDE_SED" | tee -a "$OUT"
}

# the defaults: the shell's switches cleared for the runs (this script's own environment only)
CLEARED=""
for v in $(env | sed -n 's/^\(ARC_[A-Za-z0-9_]*\|INFERZERO_[A-Za-z0-9_]*\)=.*/\1/p'); do
  [ "$v" = ARC_DATA_DIR ] && continue
  CLEARED="${CLEARED:+$CLEARED }$v"
  unset "$v"
done
[ -n "$DEVICE" ] && export ZE_AFFINITY_MASK="$DEVICE"
trap 'pkill -x serverz -P $$ 2>/dev/null; rm -rf "$TMP"' EXIT

step() {  # TITLE KEEP_REGEX ENV(k=v, or -) ARGS...: one runz run through gpu_guard, its lines matching KEEP
  local title="$1" keep="$2" envs="$3"; shift 3
  local m="$1"; shift
  add ""
  add "## $title"
  local o code
  if [ "$envs" = - ]; then
    o=$("$GUARD" --timeout 1800 -- "$RUNZ" "$m" "$@" 2>&1); code=$?
    add "runz <model> $*"
  else
    o=$(env "$envs" "$GUARD" --timeout 1800 -- "$RUNZ" "$m" "$@" 2>&1); code=$?
    add "runz <model> $*  ($envs)"
  fi
  while IFS= read -r l; do add "$l"; done < <(printf '%s\n' "$o" | grep -E "$keep")
  if [ $code -ne 0 ]; then
    add "(exit code $code)"
    while IFS= read -r l; do add "$l"; done < <(printf '%s\n' "$o" | tail -12)
  fi
}

echo "Close any serverz you run before starting: the kit starts one for its memory step and stops it."
REL="(not a package)"
[ -f "$ROOT/bin/PREBUILT.txt" ] && REL=$(head -1 "$ROOT/bin/PREBUILT.txt")
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)
add "# inferzero contributor kit, $(date '+%Y-%m-%d %H:%M')"
add "release: $REL${COMMIT:+, commit $COMMIT}"
add "OS: $(uname -sr)"
add "CPU: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -1); RAM: $(awk '/MemTotal/ {printf "%.0f", $2 / 1048576}' /proc/meminfo 2>/dev/null) GB"
while IFS= read -r g; do add "GPU: $g"; done < <(lspci -nn 2>/dev/null | grep -Ei 'vga|display|3d' | grep -i intel | sed 's/^[0-9a-f:.]* //')
# (the driver on Linux is the compute runtime: its packages' versions, whichever package manager the system has; the
#  engine's own "kernels compiled by ... driver" line is in section A too)
RT=$( { dpkg-query -W -f='${Package} ${Version}\n' intel-opencl-icd libze-intel-gpu1 intel-level-zero-gpu libze1 libigc2 libigc1 2>/dev/null \
        || rpm -q intel-opencl intel-level-zero-gpu level-zero intel-igc-core 2>/dev/null \
        || pacman -Q intel-compute-runtime level-zero-loader intel-graphics-compiler 2>/dev/null; } | grep -v "not installed\|no packages found\|was not found")
if [ -n "$RT" ]; then while IFS= read -r l; do add "runtime: $l"; done <<< "$RT"; else add "runtime: (no package versions found)"; fi
if [ -n "$DEVICE" ]; then add "device: ZE_AFFINITY_MASK=$DEVICE"; else add "device: the default"; fi
add "switches cleared for the runs: ${CLEARED:--}"
RUNS=2; [ $QUICK = 1 ] && RUNS=1
for m in "${MODEL_LIST[@]}"; do
  add ""
  add "# model: $(basename "$m") ($(awk -v b="$(stat -c %s "$m")" 'BEGIN {printf "%.2f", b / 1073741824}') GB)"
  step "A. kernels (ARC_GEMM_PLAN=1)" '^(gemm-plan|inferzero:|loaded:)' ARC_GEMM_PLAN=1 "$m" --bench-pp 512 --n 16
  for i in $(seq 1 $RUNS); do
    step "E. speed, prompt 512 + generation 128 (run $i)" '^(pp[0-9]|tg[0-9]|prompt:|generation:)' - "$m" --bench-pp 512 --n 128
    step "E. speed, prompt 4096 (run $i)" '^(pp[0-9]|prompt:)' - "$m" --bench-pp 4096 --n 1
  done
  step "G. launches a generated token (ARC_LAUNCH_STATS=1)" '^(launches:|tg[0-9])' ARC_LAUNCH_STATS=1 "$m" --bench-pp 512 --n 32
  step "G. generation profile (--prof-gen)" '(launches\)|tg steps)' - "$m" --bench-pp 512 --n 32 --prof-gen
  step "G. prompt profile (--prof)" 'launches\)' - "$m" --bench-pp 512 --n 1 --prof
  # F: --ctx auto in the server: its memory line, then the server is stopped
  add ""
  add "## F. memory (serverz --ctx auto)"
  LOG="$TMP/serverz.log"
  : > "$LOG"
  "$GUARD" --timeout 900 -- "$SERVERZ" -m "$m" --port 18999 --ctx auto > "$LOG" 2>&1 &
  GPID=$!
  for _ in $(seq 1 1680); do
    grep -q "listening on" "$LOG" 2>/dev/null && break
    kill -0 "$GPID" 2>/dev/null || break
    sleep 0.5
  done
  pkill -x serverz 2>/dev/null  # (gpu_guard refused to start while another serverz ran, so this one is ours)
  pkill -P "$GPID" 2>/dev/null  # (and whatever gpu_guard started, should the server's name differ)
  kill "$GPID" 2>/dev/null
  wait "$GPID" 2>/dev/null
  while IFS= read -r l; do add "$(printf '%s' "$l" | sed -E 's/^[^ ]+ [^ ]+ [IWE] //')"; done \
    < <(grep -v '^gpu_guard:' "$LOG" | grep -E 'memory \(--ctx auto\)|--ctx auto|loaded|context|error|failed')
done
echo
echo "Done. Please attach this file to the GitHub issue: $OUT"
