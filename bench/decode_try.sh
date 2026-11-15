#!/usr/bin/env bash
# bench/decode_try.ps1 is the same script for Windows; the two must stay the same (change both together).
# Token generation settings on this GPU (made for GPUs with memory of their own, e.g. Arc Pro B60): the same greedy
# answer to one coding question plain, with the model's MTP block, with a DFlash2 drafter, and with the GEMV's
# tuning switches, each the best of 2 runs. Paste the whole output into the GitHub issue.
#   bash bench/decode_try.sh --models ~/hf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--dflash <DFlash2 drafter folder>] [--n 256]
#   (several models: --models a.gguf b.gguf; --ctx N sets the long-context rows' prompt length)
# Runs (each: bin/runz <model> --text-file <question> --n N):
#   plain          no drafts
#   mtp            --mtp (the model's own MTP block, when it has one)
#   mtp7           --mtp --mtp-drafts 7
#   dflash         --dflash DIR --draft auto (with --dflash)
#   GEMV_GRF=256   plain, the single-token GEMV with 256 registers per thread (ARC_GEMV_GRF=256)
#   GVROWS=1/2/4   plain, rows per sub-group of the codebook GEMV (ARC_GVROWS; default: by matrix size)
#   mtp MT_GRF=256 mtp, the 2-8 token GEMV of the verification with 256 registers, as before 2026-11-03 (ARC_GEMV_MT_GRF=256)
#   mtp MT_GRF=128 the same with 128 registers (the discrete cards' default; iGPUs default to 256)
#   GEMV_PF=0      plain, without the single-token GEMV issuing all its rows' reads of a step first (ARC_GEMV_PF=0;
#                  on by default on discrete cards: dense models since 2026-11-08, MoE models since 2026-11-12, a
#                  B60's 35B 79.9 without it, 88.9 with)
#   GEMV_DB=1      plain, the single-token GEMV reading one window ahead instead of the prefetch (ARC_GEMV_DB=1,
#                  2026-11-15; bit-identical); tokens checked against plain
#   GV_IQ3S=1      plain, IQ3_S weights through gemv_v (straight into registers) instead of gemv's local window
#                  (ARC_GV_IQ3S=1); also with GVROWS=2 / 4 (ARC_GVROWS: its rows per sub-group)
#   BATCH=16/32/128  plain, kernels recorded into command lists of up to n kernels, each appended to the immediate
#                  list when full (ARC_BATCH=n; off by default); also "mtp BATCH=32"
#   XALL=0         MoE models: without skipping the host's per-layer reads of expert ids when every expert is
#                  cached (ARC_XALL: on by default on discrete cards since 2026-11-09, a B60's 35B 78.8 -> 88.1);
#                  also with GEMV_PF=1 (both as before 2026-11-09) and "mtp XALL=0". Their tokens are checked against plain.
#   WIDE_ROWS=0    plain, the row kernels that quantize for the next GEMV (swiglu_quant, rmsnorm_quant,
#                  gated_norm_quant, attn_gate_quant) in one work-group per row, as before 2026-11-10
#                  (ARC_WIDE_ROWS=0); tokens checked against plain
#   SMALL_WG=1     plain, the GEMV of matrices with few rows (DeltaNet's alpha / beta, MoE routers) in work-groups
#                  of 64 (ARC_GEMV_SMALL_WG=1; the default 2026-11-10 to 11-12, off since: the B60's 35B 30 % slower
#                  with it); tokens checked against plain
#   GEMV_GU=1      dense models: plain, the FFN gate and up GEMVs in one launch instead of two (ARC_GEMV_GU=1,
#                  QUEUE 33.5; bit-identical; MoE models take their own FFN, so it equals plain there); tokens checked
#   (at the end, per model: a --prof-gen table, the GPU time of a generation step by kernel)
#   MOE_GR_GU=2/4, MOE_GR_DN=2/8   MoE models: rows per sub-group of the experts' gate / up and down GEMVs for one
#                  token (ARC_MOE_GR_GU, default 1; ARC_MOE_GR_DN, default 4)
# Added 2026-11-13 (a GEMV sweep for the B60 / B70, issue #12):
#   GVROWS=8       plain, 8 rows per sub-group (the most ARC_GVROWS takes)
#   GV4=0          plain, the codebook types (IQ1 / IQ2 / IQ3_XXS) at 2 rows per sub-group instead of 4 (ARC_GV4=0)
#   GV_IQ3S=1 GVROWS=1/8   the IQ3_S path at the ends of the range
#   mtp MOE_GR_DN_S=1/2/8, MOE_GRF=128   MoE models: rows per sub-group of the experts' down GEMV for 2-8 tokens (the
#                  verification; default 4), the experts' GEMVs with the default register file (ARC_MOE_GRF=128)
#   long context (--ctx, default 16384; 0 skips it): the prompt of that length, then 64 tokens, best of 2
#   (runz --bench-pp N --n 64): plain, ATTN_DEC_GRF=128 (the generation attention with the default register file),
#   ATTN_DEC_SG=256/1024 (the sub-groups its key split aims for, default 512), ATTN_DEC_SMAX=128/512 (keys per split
#   at most, default 256), and a --prof-gen table there
# Needs bash 4.4 or newer and the standard tools (awk, sed, grep, sha256sum, coreutils); runs bin/runz and bin/mgpu_probe.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz"
PROBE="$ROOT/bin/mgpu_probe"
VARS=(ARC_GEMV_GRF ARC_GEMV_DB ARC_GVROWS ARC_GEMV_MT_GRF ARC_GEMV_PF ARC_GV_IQ3S ARC_MOE_GR_GU ARC_MOE_GR_DN ARC_BATCH ARC_XALL
      ARC_WIDE_ROWS ARC_GEMV_SMALL_WG ARC_GEMV_GU ARC_GV4 ARC_MOE_GR_DN_S ARC_MOE_GRF ARC_ATTN_DEC_GRF
      ARC_ATTN_DEC_SG ARC_ATTN_DEC_SMAX)

usage() {
  # (the comment at the top of this file, without its "# ")
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}
die() { echo "$*" >&2; exit 1; }
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# --- the options
MODELS=()
DFLASH=""
N=256
CTX=16384
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --models|--model)
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do MODELS+=("$1"); shift; done
      ;;
    --models=*|--model=*) MODELS+=("${1#*=}"); shift ;;
    --dflash) [ $# -ge 2 ] || die "--dflash needs a folder"; DFLASH="$2"; shift 2 ;;
    --dflash=*) DFLASH="${1#*=}"; shift ;;
    --n) [ $# -ge 2 ] || die "--n needs a number"; N="$2"; shift 2 ;;
    --n=*) N="${1#*=}"; shift ;;
    --ctx) [ $# -ge 2 ] || die "--ctx needs a number"; CTX="$2"; shift 2 ;;
    --ctx=*) CTX="${1#*=}"; shift ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ ${#MODELS[@]} -gt 0 ] || { echo "--models is required (one or more .gguf files; see --help)" >&2; exit 2; }
is_int "$N" || die "--n takes a whole number: $N"
is_int "$CTX" || die "--ctx takes a whole number: $CTX"
[ -f "$RUNZ" ] || die "bin/runz not found (run from the inferzero folder)"
[ -f "$PROBE" ] || die "bin/mgpu_probe not found"

TMP="$(mktemp -d)" || die "cannot make a temporary folder"
trap 'rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM
OUT="$TMP/out.txt"

# (env -u for every variable of the script: each run starts from the launcher's environment without them)
UNSET=()
for v in "${VARS[@]}"; do UNSET+=(-u "$v"); done

# (Qwen chat format, thinking off: the answer starts right away)
Q="$TMP/inferzero-decode-try.txt"
printf '%s' $'<|im_start|>user\nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using '\
$'a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>\n<|im_start|>assistant\n'\
$'<think>\n\n</think>\n\n' > "$Q"

# one runz call: $1 the variables to set ("VAR=value VAR=value", or ""), then runz's arguments; the output (stdout and
# stderr) goes to $OUT
run_runz() {
  local -a e=()
  read -ra e <<< "$1"; shift
  env "${UNSET[@]}" "${e[@]}" "$RUNZ" "$@" > "$OUT" 2>&1
}
first_error() { grep -aiE 'error|cannot|failed|exception' "$1" | head -n 1; }
trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
fnum() { LC_ALL=C awk -v x="$1" -v f="%$2.$3f" 'BEGIN { printf f, x }'; }  # number width decimals
ftrue() { LC_ALL=C awk "BEGIN { exit !($1) }"; }  # a comparison of numbers

# the best of 2 runs of the question; $1 the variables, $2 the model, then extra arguments; sets R_TPS R_NOTE R_MOE R_IDS
invoke_run() {
  local envs="$1" model="$2"; shift 2
  local i tps note ids moe a b found=0
  for i in 1 2; do
    run_runz "$envs" "$model" --text-file "$Q" --n "$N" "$@"
    tps="$(grep -aoE 'generation: [0-9]+ tokens in [0-9.]+ ms \([0-9.]+ t/s\)' "$OUT" | head -n 1 |
           sed -E 's/.*\(([0-9.]+) t\/s\)$/\1/')"
    if [ -z "$tps" ]; then
      R_TPS=0; R_NOTE="failed: $(first_error "$OUT")"; R_MOE=0; R_IDS=""
      return
    fi
    note=""
    a="$(grep -aoE 'MTP: [0-9]+ drafts, [0-9]+ accepted \([0-9.]+ %\), [0-9.]+ tokens per main-model pass' "$OUT" | head -n 1)"
    if [ -n "$a" ]; then
      b="$(sed -E 's/.*accepted \(([0-9.]+) %\), ([0-9.]+) tokens per.*/\1 % accepted, \2 tokens per pass/' <<< "$a")"
      note="$b"
    fi
    ids="$(grep -aoE 'greedy:[ 0-9]*' "$OUT" | head -n 1)"
    ids="$(trim <<< "${ids#greedy:}")"
    if grep -aqi 'expert cache' "$OUT"; then moe=1; else moe=0; fi
    if [ "$found" = 0 ] || ftrue "$tps > $R_TPS"; then
      R_TPS="$tps"; R_NOTE="$note"; R_MOE="$moe"; R_IDS="$ids"; found=1
    fi
  done
}

line() { printf '%-22s %s tok/s  %s\n' "$1" "$(fnum "$R_TPS" 8 2)" "$R_NOTE"; }  # what (the result in R_*)
# (switches that must not change the answer: their tokens against plain's)
same() {
  local n
  if ftrue "$R_TPS == 0"; then n="$R_NOTE"
  elif [ "$R_IDS" = "$PLAIN_IDS" ]; then n="same tokens as plain"
  else n="TOKENS DIFFER from plain"; fi
  printf '%-22s %s tok/s  %s\n' "$1" "$(fnum "$R_TPS" 8 2)" "$n"
}
run_line() { local what="$1"; shift; invoke_run "$@"; line "$what"; }
run_same() { local what="$1"; shift; invoke_run "$@"; same "$what"; }
prof_lines() {  # the --prof-gen table's lines of the file $1
  grep -aiE '^tg[0-9]+:|tg steps|weights read|GPU time by kernel|^[[:space:]]+[[:alnum:]_]+[[:space:]]+[0-9.]+ ms' "$1" |
    sed -e 's/[[:space:]]*$//'
}
# generation after the --ctx prompt: best of 2 (prints "<tok/s> tok/s" or "failed: ..."); with "prof" one run with
# --prof-gen, its output left in $OUT
tg() {
  local envs="$1" model="$2" prof="${3:-}" best="0.0" i n=2 m
  local -a a=(--bench-pp "$CTX" --n 64)
  if [ -n "$prof" ]; then n=1; a+=(--prof-gen); fi
  for ((i = 0; i < n; ++i)); do
    run_runz "$envs" "$model" "${a[@]}"
    m="$(grep -aoE 'tg[0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$OUT" | head -n 1 | awk '{ print $(NF-1) }')"
    if [ -n "$m" ] && ftrue "$m > $best"; then best="$m"; fi
  done
  [ -n "$prof" ] && return
  if ftrue "$best == 0"; then echo "failed: $(first_error "$OUT")"; return; fi
  echo "$(fnum "$best" 8 2) tok/s"
}

"$PROBE" 2>&1 | grep -aiE '^[[:space:]]+GPU [0-9]+:' | trim
echo "generation of $N tokens (greedy), best of 2 runs"
for model in "${MODELS[@]}"; do
  if [ ! -e "$model" ]; then echo "skipped (not found): $model"; continue; fi
  echo
  echo "=== $(basename -- "$model")"
  invoke_run "" "$model" --n 8  # (warm-up: the first run after a start is slower)
  invoke_run "" "$model"
  PLAIN_IDS="$R_IDS"; PLAIN_MOE="$R_MOE"
  line "plain"
  # (2026-11-10: the wide row kernels, on by default: without them; the small GEMV work-groups, off by default
  #  since 2026-11-12: with them)
  run_same "WIDE_ROWS=0" "ARC_WIDE_ROWS=0" "$model"
  run_same "SMALL_WG=1" "ARC_GEMV_SMALL_WG=1" "$model"
  run_same "GEMV_GU=1" "ARC_GEMV_GU=1" "$model"
  # (meant bit for bit, so checked bit for bit: every one of 32 steps' logits against plain's)
  lg=()
  for gu in 0 1; do
    d="$TMP/inferzero-decode-try-gu$gu.bin"
    if [ "$gu" = 1 ]; then e="ARC_GEMV_GU=1"; else e=""; fi
    run_runz "$e" "$model" --text-file "$Q" --n 32 --dump-logits "$d"
    if [ -f "$d" ]; then lg+=("$(sha256sum < "$d" | awk '{ print $1 }')"); else lg+=(""); fi
    rm -f "$d"
  done
  if [ -z "${lg[0]}" ] || [ -z "${lg[1]}" ]; then r="not written"
  elif [ "${lg[0]}" = "${lg[1]}" ]; then r="identical to plain (32 steps)"
  else r="DIFFER from plain"; fi
  echo "GEMV_GU=1 logits       $r"
  run_line "mtp" "" "$model" --mtp
  run_line "mtp7" "" "$model" --mtp --mtp-drafts 7
  if [ -n "$DFLASH" ]; then run_line "dflash" "" "$model" --dflash "$DFLASH" --draft auto; fi
  run_line "GEMV_GRF=256" "ARC_GEMV_GRF=256" "$model"
  for g in 1 2 4 8; do run_line "GVROWS=$g" "ARC_GVROWS=$g" "$model"; done
  run_same "GV4=0" "ARC_GV4=0" "$model"
  run_line "mtp MT_GRF=256" "ARC_GEMV_MT_GRF=256" "$model" --mtp
  run_line "mtp MT_GRF=128" "ARC_GEMV_MT_GRF=128" "$model" --mtp
  # (the prefetch is on by default on discrete cards, dense and MoE models: the row without it)
  run_line "GEMV_PF=0" "ARC_GEMV_PF=0" "$model"
  run_same "GEMV_DB=1" "ARC_GEMV_DB=1" "$model"
  run_line "GV_IQ3S=1" "ARC_GV_IQ3S=1" "$model"
  for g in 1 2 4 8; do run_line "GV_IQ3S=1 GVROWS=$g" "ARC_GV_IQ3S=1 ARC_GVROWS=$g" "$model"; done
  for b in 16 32 128; do run_same "BATCH=$b" "ARC_BATCH=$b" "$model"; done
  run_line "mtp BATCH=32" "ARC_BATCH=32" "$model" --mtp
  if [ "$PLAIN_MOE" = 1 ]; then  # (MoE models: the experts' per-tile GEMVs of one token, rows per sub-group)
    # (XALL is on by default on discrete cards since 2026-11-09: the rows without it)
    run_same "XALL=0" "ARC_XALL=0" "$model"
    run_same "XALL=0 GEMV_PF=0" "ARC_XALL=0 ARC_GEMV_PF=0" "$model"
    run_line "mtp XALL=0" "ARC_XALL=0" "$model" --mtp
    for g in 2 4; do run_line "MOE_GR_GU=$g" "ARC_MOE_GR_GU=$g" "$model"; done
    for g in 2 8; do run_line "MOE_GR_DN=$g" "ARC_MOE_GR_DN=$g" "$model"; done
    for g in 1 2 8; do run_line "mtp MOE_GR_DN_S=$g" "ARC_MOE_GR_DN_S=$g" "$model" --mtp; done
    run_same "MOE_GRF=128" "ARC_MOE_GRF=128" "$model"
  fi
  # (generation after a long prompt: the attention's share grows with the context)
  if [ "$CTX" -gt 0 ]; then
    echo "--- after a $CTX-token prompt: 64 tokens, best of 2"
    for c in "plain|" "ATTN_DEC_GRF=128|ARC_ATTN_DEC_GRF=128" "ATTN_DEC_SG=256|ARC_ATTN_DEC_SG=256" \
             "ATTN_DEC_SG=1024|ARC_ATTN_DEC_SG=1024" "ATTN_DEC_SMAX=128|ARC_ATTN_DEC_SMAX=128" \
             "ATTN_DEC_SMAX=512|ARC_ATTN_DEC_SMAX=512"; do
      printf '%-22s %s\n' "ctx ${c%%|*}" "$(tg "${c#*|}" "$model")"
    done
    echo "--- generation profile after the $CTX-token prompt (plain)"
    tg "" "$model" prof
    prof_lines "$OUT"
  fi
  # (where a generation step's GPU time goes, per kernel, plain: --prof-gen over 32 steps after a 128-token prompt)
  echo "--- generation profile (plain, 32 steps)"
  run_runz "" "$model" --bench-pp 128 --n 32 --prof-gen
  prof_lines "$OUT"
  # (kernel launches and host waits per token: plain, and for MoE models with ARC_XALL=0)
  echo "--- launches per token"
  runs=("plain|")
  if [ "$PLAIN_MOE" = 1 ]; then runs+=("XALL=0|ARC_XALL=0"); fi
  for c in "${runs[@]}"; do
    run_runz "ARC_LAUNCH_STATS=1 ${c#*|}" "$model" --bench-pp 128 --n 32
    grep -aiE '^launches:|expert cache' "$OUT" | trim | while IFS= read -r l; do echo "${c%%|*}: $l"; done
  done
done
