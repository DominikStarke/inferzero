#!/usr/bin/env bash
# The contributor kit (docs/gemm-inventory.md table J; the user, 2026-11-27): one command on a contributor's card that
# writes one file to attach to the GitHub issue, with what the shared GEMM layer's inventory needs from that card:
#   A  which kernel and options each matrix product takes (ARC_GEMM_PLAN=1), and the start-up lines (XMX, H8, spills);
#   E  the speed: prompt 512 and 4,096 tokens, generation 128 tokens, two runs each;
#   G  the launches a generated token (ARC_LAUNCH_STATS=1) and the kernels' launch counts (--prof / --prof-gen);
#   F  what --ctx auto picks and the GPU memory the server reports for it.
#   bash bench/contrib_kit.sh --model <a.gguf>[,<b.gguf>] [--device N] [--out <file>] [--quick] [--self-test]
# --device N: which GPU (ZE_AFFINITY_MASK). --quick: one speed run each instead of two. About 10-20 minutes a model.
# --self-test: the name hiding over fixed sample lines, no GPU run; exit 1 on any line not as expected.
# bench/contrib_kit.ps1 is the same for Windows; the two must stay the same (change both together). Every GPU run goes
# through bench/gpu_guard.sh. It starts only when no other GPU job runs (a server of yours included: close it first),
# and stops only the server it started. The runs use the defaults: the shell's ARC_* / INFERZERO_* switches are cleared
# for them (but ARC_DATA_DIR), their names in the file.
# Nothing personal is written (the file is posted publicly): the models' folders become <model dir>, this folder
# <inferzero>, the temporary folder <temp> and the home folder ~ (in any casing), the user and host names <user> /
# <host> wherever they stand as a word (a name with letters outside ASCII also as a console may mangle it), any other
# absolute path <path>/<model file> for a given model's file or <file>, and gpu_guard's own lines are left out.
# Needs GNU sed (its I flag), as every Linux the engine runs on has.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz" SERVERZ="$ROOT/bin/serverz" GUARD="$ROOT/bench/gpu_guard.sh"
MODELS="" DEVICE="" OUT="" QUICK=0 SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODELS="${MODELS:+$MODELS,}$2"; shift 2;;
    --device) DEVICE=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --quick) QUICK=1; shift;;
    --self-test) SELFTEST=1; shift;;
    *) echo "unknown option: $1 (--model a.gguf[,b.gguf] [--device N] [--out F] [--quick] [--self-test])" >&2; exit 1;;
  esac
done
TMP=$(mktemp -d)
GPID=""
# (on any exit: the server this kit started, if it still runs, and the temporary folder)
cleanup() { [ -n "$GPID" ] && stop_server "$GPID"; rm -rf "$TMP"; }
trap cleanup EXIT

# ---- the hiding: one sed script, built from the folders and names (byte-wise, LC_ALL=C, so a letter outside ASCII is
# matched as 1-4 bytes of anything but a separator: a narrow-string program's "é" may come out as "?" or U+FFFD)
sed_esc() { printf '%s' "$1" | LC_ALL=C sed 's/[][\/.^$*+?(){}|#]/\\&/g'; }
name_pat() {  # TEXT: a regex for it, ASCII escaped, each byte outside ASCII one run of 1-4 non-separator bytes
  local s="$1" out="" c i
  local LC_ALL=C
  for ((i = 0; i < ${#s}; i++)); do
    c="${s:i:1}"
    if [ "$(printf '%d' "'$c")" -ge 128 ] 2>/dev/null || [ "$(printf '%d' "'$c")" -lt 0 ] 2>/dev/null; then
      case "$out" in *'[^/[:space:]]{1,4}') ;; *) out="$out"'[^/[:space:]]{1,4}';; esac  # (one run per letter's bytes)
    else
      out="$out$(sed_esc "$c")"
    fi
  done
  printf '%s' "$out"
}
PATH_START='(^|[[:space:]"'"'"'(=:])'  # (a folder only where a path starts: /root must not match inside /data/root/x)
build_hide() {  # SED_FILE MODEL_DIRS(;) ROOT TEMPS(;) HOME USER HOSTS(;) MODEL_FILES(;)
  local f="$1" mdirs="$2" root="$3" temps="$4" home="$5" user="$6" hosts="$7" mfiles="$8" d n r
  local IFS=';'  # (the lists' separator, this function's own: the caller's "$*" and word splitting stay as they were)
  : > "$f"
  # (the folders, the longest first, so one inside another (the temporary folder inside home) wins)
  { for d in $mdirs; do [ -n "$d" ] && printf '%s\t<model dir>\n' "$d"; done
    [ -n "$root" ] && printf '%s\t<inferzero>\n' "$root"
    for d in $temps; do [ -n "$d" ] && printf '%s\t<temp>\n' "${d%/}"; done
    [ -n "$home" ] && printf '%s\t~\n' "${home%/}"; } |
    awk -F'\t' '{ print length($1) "\t" $0 }' | sort -rn | cut -f2- |
    while IFS=$'\t' read -r d r; do
      printf 's#%s%s#\\1%s#gI\n' "$PATH_START" "$(name_pat "$d")" "$r" >> "$f"
    done
  # (then any absolute path left: <path>/<name> for a given model's file, else <file>. Before the names: a name inside
  #  such a path would otherwise end the path's match at its "<user>", leaving "<file><user>/x")
  for n in $mfiles; do [ -n "$n" ] &&
      printf 's#%s/[^[:space:]"'"'"'|,;()<>]*/(%s)([^A-Za-z0-9_.-]|$)#\\1<path>/\\2\\3#g\n' "$PATH_START" "$(sed_esc "$n")" >> "$f"; done
  printf 's#%s/[^[:space:]"'"'"'|,;()<>]+#\\1<file>#g\n' "$PATH_START" >> "$f"
  # (the user and host names as words, the longest first; under 3 characters left; twice, since a match takes the
  #  character before the next one, as in "name.name")
  { printf '%s\t<user>\n' "$user"; for n in $hosts; do printf '%s\t<host>\n' "$n"; done; } |
    awk -F'\t' 'length($1) >= 3 && !seen[tolower($1)]++ { print length($1) "\t" $0 }' | sort -rn | cut -f2- |
    while IFS=$'\t' read -r n r; do
      printf 's#(^|[^A-Za-z0-9])%s([^A-Za-z0-9]|$)#\\1%s\\2#gI\n' "$(name_pat "$n")" "$r" >> "$f"
      printf 's#(^|[^A-Za-z0-9])%s([^A-Za-z0-9]|$)#\\1%s\\2#gI\n' "$(name_pat "$n")" "$r" >> "$f"
    done
}
hide() { LC_ALL=C sed -E -f "$HIDE_SED"; }
# (gpu_guard's own lines: its command line holds the full paths; anywhere in a line, since its copies can glue a
#  fragment of output in front of one)
dropped() { case "$1" in *"gpu_guard: "*) return 0;; esac; return 1; }

if [ $SELFTEST = 1 ]; then
  HIDE_SED="$TMP/self.sed"
  E=$'\xc3\xa9' EU=$'\xc3\x89'  # (é, É)
  build_hide "$HIDE_SED" "/data/llm/qwen" "/home/jos$E/Work - Contoso/inferzero" "/home/jos$E/.cache/tmp" "/home/jos$E" \
             "jos$E" "DESKTOP-AB1;desktop-ab1.corp.example" "Qwen3.5-4B-UD-Q4_K_XL.gguf"
  bad=0
  check() {  # IN EXPECTED
    local o; o=$(printf '%s\n' "$1" | hide)
    if [ "$o" = "$2" ]; then echo "ok    in:  $1"; echo "      out: $o"; else bad=$((bad + 1)); echo "WRONG in:  $1"; echo "      out: $o   (expected: $2)"; fi
  }
  check "cache at /home/jos$E/.inferzero/kernels" "cache at ~/.inferzero/kernels"
  check "cache at /HOME/JOS$EU/.inferzero/kernels" "cache at ~/.inferzero/kernels"
  check "cache at /home/jos?/.inferzero/kernels" "cache at ~/.inferzero/kernels"
  check "built in /home/jos$E/.cache/tmp/b1.tmp" "built in <temp>/b1.tmp"
  check "runz at /home/jos$E/Work - Contoso/inferzero/bin/runz" "runz at <inferzero>/bin/runz"
  check "expert file /data/other/Flash-Next-IQ3_S.gguf.experts, model /data/llm/qwen/Qwen3.5-4B-UD-Q4_K_XL.gguf" \
        "expert file <file>, model <model dir>/Qwen3.5-4B-UD-Q4_K_XL.gguf"
  check "a copy /mnt/backup/Qwen3.5-4B-UD-Q4_K_XL.gguf" "a copy <path>/Qwen3.5-4B-UD-Q4_K_XL.gguf"
  check "jos$E on desktop-ab1 (desktop-ab1.corp.example)" "<user> on <host> (<host>)"
  check "Driver D on DESKTOP-AB1" "Driver D on <host>"
  check "also /data/jos$E/x and /mnt/DESKTOP-AB1/y" "also <file> and <file>"
  # (the caller's word splitting is untouched by the hiding's lists)
  w="a b"; set -- $w; if [ $# = 2 ]; then echo "ok    IFS kept"; else bad=$((bad + 1)); echo "WRONG IFS changed"; fi
  if dropped "tg128: 5 msgpu_guard: /x/runz ... (timeout 1800 s, log /tmp/g.log)"; then echo "ok    a gpu_guard line, glued to output, is left out"
  else bad=$((bad + 1)); echo "WRONG a gpu_guard line is kept"; fi
  # (a user name under 3 characters is left, and only it: the host is still hidden, no letter replaced)
  build_hide "$HIDE_SED" "" "" "" "" "jo" "DESKTOP-AB1" ""
  check "Driver D jo on DESKTOP-AB1" "Driver D jo on <host>"
  if [ $bad = 0 ]; then echo "self-test: all ok"; else echo "self-test: $bad WRONG"; exit 1; fi
  exit 0
fi

[ -n "$MODELS" ] || { echo "give --model <a.gguf>[,<b.gguf>]: the models you run (the 27B, the 35B-A3B, Inkling 2 ...)" >&2; exit 1; }
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
for m in "${MODEL_LIST[@]}"; do [ -f "$m" ] || { echo "model not found: $m" >&2; exit 1; }; done
[ -x "$RUNZ" ] || { echo "bin/runz not found: build it (./build.sh runz serverz)" >&2; exit 1; }
# (no other GPU job, a server of the user's included: gpu_guard would refuse every step, and the memory step must not
#  stop anything it did not start)
BUSY=$(pgrep -l -x 'runz|serverz|attn_bench|xw_peak|gpu_bench|ref_forward' 2>/dev/null | tr '\n' ' ')
if [ -n "$BUSY" ]; then echo "Another GPU job runs: $BUSY. Close it (a server of yours too) and start the kit again." >&2; exit 2; fi
[ -n "$OUT" ] || OUT="$PWD/inferzero-kit-$(date '+%Y%m%d-%H%M').txt"
: > "$OUT"
HIDE_SED="$TMP/hide.sed"
MDIRS="" MFILES=""
for m in "${MODEL_LIST[@]}"; do MDIRS="$MDIRS;$(cd "$(dirname "$m")" && pwd)"; MFILES="$MFILES;$(basename "$m")"; done
TEMPS="${TMPDIR:-}"; [ -d /tmp ] && TEMPS="$TEMPS;/tmp"; [ -d /var/tmp ] && TEMPS="$TEMPS;/var/tmp"
HOSTS="$(hostname 2>/dev/null || uname -n);$(hostname -f 2>/dev/null)"
build_hide "$HIDE_SED" "$MDIRS" "$ROOT" "$TEMPS" "${HOME:-}" "${USER:-$(id -un 2>/dev/null)}" "$HOSTS" "$MFILES"
add() {  # LINE: into the file and to the console, the names hidden; gpu_guard's lines left out
  dropped "$1" && return 0
  printf '%s\n' "$1" | hide | tee -a "$OUT"
}

# the defaults: the shell's switches cleared for the runs (this script's own environment only)
CLEARED=""
for v in $(env | sed -n 's/^\(ARC_[A-Za-z0-9_]*\|INFERZERO_[A-Za-z0-9_]*\)=.*/\1/p'); do
  [ "$v" = ARC_DATA_DIR ] && continue
  CLEARED="${CLEARED:+$CLEARED }$v"
  unset "$v"
done
# (driver and runtime switches stay as the user set them: their names, not values, go into the file as a warning)
RTVARS=$(env | sed -n 's/^\(ZE_[A-Za-z0-9_]*\|IGC_[A-Za-z0-9_]*\|NEO[A-Za-z0-9_]*\)=.*/\1/p' | grep -v '^ZE_AFFINITY_MASK$' | sort | tr '\n' ' ')
[ -n "$DEVICE" ] && export ZE_AFFINITY_MASK="$DEVICE"

# the processes started by one process, its children's children too
descendants() {
  local c
  for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done
}
stop_server() {  # GPID: only the serverz this kit's gpu_guard started (by process, not by name), then the guard
  local p
  for p in "$1" $(descendants "$1"); do [ "$(ps -o comm= -p "$p" 2>/dev/null)" = serverz ] && kill "$p" 2>/dev/null; done
  wait "$1" 2>/dev/null
}

step() {  # TITLE KEEP_REGEX ENV(k=v, or -) ARGS...: one runz run through gpu_guard, its lines matching KEEP
  local title="$1" keep="$2" envs="$3"; shift 3
  local m="$1"; shift
  add ""
  add "## $title"
  local o code
  if [ "$envs" = - ]; then
    o=$(bash "$GUARD" --timeout 1800 -- "$RUNZ" "$m" "$@" 2>&1); code=$?
    add "runz <model> $*"
  else
    o=$(env "$envs" bash "$GUARD" --timeout 1800 -- "$RUNZ" "$m" "$@" 2>&1); code=$?
    add "runz <model> $*  ($envs)"
  fi
  while IFS= read -r l; do add "$l"; done < <(printf '%s\n' "$o" | grep -E "$keep")
  if [ $code -ne 0 ]; then
    add "(exit code $code)"
    while IFS= read -r l; do add "$l"; done < <(printf '%s\n' "$o" | grep -v 'gpu_guard: ' | tail -12)
  fi
}

echo "The kit starts a server of its own for its memory step and stops only that one."
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
[ -n "$RTVARS" ] && add "warning: driver / runtime variables set (kept for the runs): $RTVARS"
RUNS=2; [ $QUICK = 1 ] && RUNS=1
for m in "${MODEL_LIST[@]}"; do
  add ""
  add "# model: $(basename "$m") ($(awk -v b="$(stat -c %s "$m")" 'BEGIN {printf "%.2f", b / 1073741824}') GB)"
  # (gemm-plan anywhere in a line: a glued fragment must not hide one)
  step "A. kernels (ARC_GEMM_PLAN=1)" '(gemm-plan|^inferzero:|^loaded:)' ARC_GEMM_PLAN=1 "$m" --bench-pp 512 --n 16
  for i in $(seq 1 $RUNS); do
    step "E. speed, prompt 512 + generation 128 (run $i)" '^(pp[0-9]|tg[0-9]|prompt:|generation:)' - "$m" --bench-pp 512 --n 128
    step "E. speed, prompt 4096 (run $i)" '^(pp[0-9]|prompt:)' - "$m" --bench-pp 4096 --n 1
  done
  step "G. launches a generated token (ARC_LAUNCH_STATS=1)" '^(launches:|tg[0-9])' ARC_LAUNCH_STATS=1 "$m" --bench-pp 512 --n 32
  step "G. generation profile (--prof-gen)" '(launches\)|tg steps)' - "$m" --bench-pp 512 --n 32 --prof-gen
  step "G. prompt profile (--prof)" 'launches\)' - "$m" --bench-pp 512 --n 1 --prof
  # F: --ctx auto in the server: its memory line, then the server it started is stopped (by process, not by name)
  add ""
  add "## F. memory (serverz --ctx auto)"
  LOG="$TMP/serverz.log"
  : > "$LOG"
  bash "$GUARD" --timeout 900 -- "$SERVERZ" -m "$m" --port 18999 --ctx auto > "$LOG" 2>&1 &
  GPID=$!
  for _ in $(seq 1 1680); do
    grep -q "listening on" "$LOG" 2>/dev/null && break
    kill -0 "$GPID" 2>/dev/null || break
    sleep 0.5
  done
  stop_server "$GPID"
  GPID=""
  while IFS= read -r l; do add "$(printf '%s' "$l" | sed -E 's/^[^ ]+ [^ ]+ [IWE] //')"; done \
    < <(grep -v 'gpu_guard: ' "$LOG" | grep -E 'memory \(--ctx auto\)|--ctx auto|loaded|context|error|failed' |
        sed -E 's/^[^ ]+ [^ ]+ [IWE] //' | awk '!seen[$0]++')  # (each line once, as the .ps1)
done
echo
echo "Done. Please attach this file to the GitHub issue: $OUT"
