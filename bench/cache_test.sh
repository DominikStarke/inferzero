#!/usr/bin/env bash
# Checks the server's two disk caches on a split model (zero/mgpu/engine_tp.h save_state_mirrored): the system-prompt
# cache must make a second start's long system prompt cheap, and a saved conversation must be found again by its
# tokens. Each phase starts its own serverz, asks one or two questions over HTTP and reads the server's log, so the
# numbers are the ones the server reports ("cached N", "prompt ... s").
#   bash bench/cache_test.sh --model M.gguf [--gpus 2] [--ctx 8192]
# Every start waits for "listening" with a deadline and is killed by its own PID; a port in use stops the run at once
# instead of waiting. Nothing here writes into the repository: the caches go to a temp folder (ARC_DATA_DIR).
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVERZ="$ROOT/bin/serverz"
GPU_PROGRAMS="serverz runz zero zdec ref_forward"

die() { printf '%s\n' "$*" >&2; exit 2; }

MODEL=""
GPUS=2
CTX=8192
PORT=8200
PHASE=all     # prefix | session | all   # a system prompt of about this many tokens (the reason the caches exist: long, identical prefixes)
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --ctx) CTX=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --phase) PHASE=$2; shift 2;;
    --help|-h) sed -n '2,20p' "$0"; exit 0;;
    *) die "unknown option: $1 (--help)";;
  esac
done
[ -n "$MODEL" ] || die "--model M.gguf is needed"
[ -f "$MODEL" ] || die "no such model: $MODEL"
[ -x "$SERVERZ" ] || die "bin/serverz is not built (build.sh serverz)"
command -v curl >/dev/null || die "curl is needed"

# (another GPU program on the GPUs this test uses stops it: two programs on one card spoil each other. A server on
#  other GPUs of the same machine is left alone, as --gpus 0,1 and --gpus 2,3 side by side are fine)
gpu_programs_running() {  # the PIDs of serverz / runz, if any
  local c n p
  for c in /proc/[0-9]*/comm; do
    [ -r "$c" ] || continue
    IFS= read -r n 2>/dev/null < "$c" || continue
    n="${n,,}"
    for p in $GPU_PROGRAMS; do
      [ "$n" = "$p" ] && { printf '%s ' "${c#/proc/}"; break; }
    done
  done
}
for pid in $(gpu_programs_running | tr -d ' ' | sed 's|/comm||g'); do
  mask=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^ZE_AFFINITY_MASK=//p')
  if [ -z "$mask" ]; then
    die "a GPU program runs without ZE_AFFINITY_MASK (pid $pid): its GPUs may include this test's"
  fi
  for g in $(printf '%s' "$mask" | tr ',' ' '); do
    case " $(seq -s' ' 0 $((GPUS - 1))) " in
      *" $g "*) die "a GPU program uses GPU $g (pid $pid, ZE_AFFINITY_MASK=$mask): stop it, or give this test other --gpus" ;;
    esac
  done
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/izcache.XXXXXX") || die "cannot make a temp folder"
DATA="$WORK/data"
SYS="$WORK/system.txt"
: > "$SYS"
for _ in $(seq 1 200); do
  printf 'You are a helpful assistant that answers concisely and accurately. ' >> "$SYS"
done

SPID=""
cleanup() {
  if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; fi
  rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3<&- 3>&-; return 0; } || return 1; }

# start_server PORT LOG: serverz in the background; 0 once it listens, 1 on a port in use or no "listening" in time
start_server() {
  local port=$1 log=$2
  port_busy "$port" && { printf 'port %s is in use (another server?)\n' "$port" >&2; return 1; }
  ARC_DATA_DIR="$DATA" ZE_AFFINITY_MASK=$(seq -s, 0 $((GPUS - 1))) ARC_GEMV_GU=1 \
    timeout 900 "$SERVERZ" -m "$MODEL" --ctx "$CTX" --slots 1 --gpus "$GPUS" \
      --mtp --mtp-drafts 3 --driver-reserve-gb 0.75 --chunk 1024 \
      --prefix-cache-gb 2 --session-cache-gb 2 --port "$port" --host 127.0.0.1 > "$log" 2>&1 &
  SPID=$!
  local i
  for i in $(seq 1 120); do
    grep -q "listening" "$log" && return 0
    kill -0 "$SPID" 2>/dev/null || { printf 'the server stopped while loading (see %s)\n' "$log" >&2; return 1; }
    sleep 1
  done
  printf 'the server did not listen within 120 s (see %s)\n' "$log" >&2
  return 1
}
stop_server() {
  [ -z "$SPID" ] && return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null
  SPID=""
  sleep 3
}
# ask PORT REQUESTFILE: one chat completion, the content printed
ask() {
  curl -s --max-time 600 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' -d @"$2"
}
# the server's own line for the last request: "prompt N tok (cached M) ..."
prompt_line() { grep -oE 'prompt [0-9]+ tok \(cached [0-9]+\) [0-9.]+ s' "$1" | tail -1; }

req() {  # req FILE SYSTEMFILE USER: one request with the long system prompt
  printf '{"model":"x","messages":[{"role":"system","content":"%s"},{"role":"user","content":"%s"}],"max_tokens":32,"temperature":0}' \
    "$(cat "$2")" "$3" > "$1"
}

printf 'model %s, %d GPU(s), ctx %d, caches in %s\n\n' "$(basename "$MODEL")" "$GPUS" "$CTX" "$WORK"
req "$WORK/q1.json" "$SYS" "Name the capital of France."
req "$WORK/q2.json" "$SYS" "And of Germany."

fail=0
if [ "$PHASE" = all ] || [ "$PHASE" = prefix ]; then
# ---- phase 1: the system-prompt cache across two starts
printf '== system-prompt cache\n'
start_server "$PORT" "$WORK/a.log" || die "phase 1: the server did not start"
ask "$PORT" "$WORK/q1.json" > "$WORK/a1.json"
A1=$(prompt_line "$WORK/a.log")
sleep 20
stop_server
start_server $((PORT + 1)) "$WORK/b.log" || die "phase 1: the second start did not start"
ask $((PORT + 1)) "$WORK/q1.json" > "$WORK/b1.json"
B1=$(prompt_line "$WORK/b.log")
stop_server
printf '  start 1 (cold): %s\n' "${A1:-no reading}"
printf '  start 2 (warm): %s\n' "${B1:-no reading}"
case "$B1" in
  *"(cached 0)"*|"") printf '  FAIL the second start did not use the cache\n'; fail=1;;
  *) printf '  OK the system prompt came from the cache\n';;
esac
cmp -s <(sed -e 's/.*"content":"//' -e 's/".*//' "$WORK/a1.json") <(sed -e 's/.*"content":"//' -e 's/".*//' "$WORK/b1.json") \
  && printf '  OK the answer is the same\n' || { printf '  FAIL the answers differ\n'; fail=1; }
[ -n "$(find "$DATA" -name '*.arcp' 2>/dev/null)" ] && printf '  OK a cache file was written\n' \
  || { printf '  FAIL no cache file\n'; fail=1; }

fi
if [ "$PHASE" = all ] || [ "$PHASE" = session ]; then
# ---- phase 2: a saved conversation found again (needs >= 1024 cached tokens)
printf '\n== session cache\n'
start_server $((PORT + 2)) "$WORK/c.log" || die "phase 2: the server did not start"
ask $((PORT + 2)) "$WORK/q1.json" > /dev/null
ask $((PORT + 2)) "$WORK/q2.json" > /dev/null   # the second request saves the first conversation
sleep 25
NARC=$(find "$DATA" -name '*.arcs' 2>/dev/null | wc -l)
printf '  saved conversations: %s\n' "$NARC"
[ "$NARC" -gt 0 ] && printf '  OK a conversation was saved\n' || { printf '  FAIL none saved (cached tokens below 1024?)\n'; fail=1; }
stop_server
start_server $((PORT + 3)) "$WORK/d.log" || die "phase 2: the third start did not start"
ask $((PORT + 3)) "$WORK/q1.json" > /dev/null
D1=$(prompt_line "$WORK/d.log")
printf '  after a restart, the same conversation: %s\n' "${D1:-no reading}"
stop_server
fi

printf '\n%s\n' "$([ "$fail" = 0 ] && echo 'all checks passed' || echo 'SOME CHECKS FAILED')"
exit "$fail"
