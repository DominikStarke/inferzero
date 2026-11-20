#!/usr/bin/env bash
# Sweeps every kernel and tile setting a user can change, on one GPU, from the smallest value to the largest: made for
# an Intel Data Center GPU Max 1550 stack (Xe-HPC, HBM, issue #48), whose settings were never tuned (the defaults are
# the B-series ones). Any discrete card works. Paste the whole output into the GitHub issue.
#   bash bench/xehpc_sweep.sh --model ~/hf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--moe-model ~/hf/<35B-A3B or Flash Next>.gguf]
#   [--device 0]   which GPU (ZE_AFFINITY_MASK; with a flat hierarchy every stack is one GPU)   [--quick]  fewer values
# Phases: G generation (one token at a time: --bench-pp 128 --n 128), P the prompt (--bench-pp 4096), M the MoE model's
# settings (with --moe-model). Each setting is changed alone from the defaults, every value from the smallest to the
# largest; a value wins when it is 2 % faster than the defaults. Then H checks each winner's logits against the defaults
# (a short prompt, --dump-logits, sha256: "same" or "differs": a differing one needs a KL check before it can be a
# default), and C runs all of a phase's winners together, with the generation and 512 / 4K / 16K prompts.
# Prompt chunks stay at most 4096 tokens. Needs bash 4.4+, coreutils (sha256sum), awk, grep; runs bin/runz.
# Expect ~60-90 minutes for the 27B (each prompt run loads the model once).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz"
die() { echo "$*" >&2; exit 1; }
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

MODEL=""; MOE_MODEL=""; DEVICE=""; QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) [ $# -ge 2 ] || die "--model needs a file"; MODEL="$2"; shift 2 ;;
    --moe-model) [ $# -ge 2 ] || die "--moe-model needs a file"; MOE_MODEL="$2"; shift 2 ;;
    --device) [ $# -ge 2 ] || die "--device needs a number"; DEVICE="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done
[ -n "$MODEL" ] || die "--model is needed (--help)"
[ -f "$MODEL" ] || die "not found: $MODEL"
[ -z "$MOE_MODEL" ] || [ -f "$MOE_MODEL" ] || die "not found: $MOE_MODEL"
[ -x "$RUNZ" ] || die "bin/runz not found: build it first (build.sh)"
[ -z "$DEVICE" ] || [[ "$DEVICE" =~ ^[0-9]+$ ]] || die "--device: a number"
command -v sha256sum > /dev/null || die "sha256sum not found"

TMP="$(mktemp -d)" || die "cannot make a temporary folder"
trap 'rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM
OUT="$TMP/out.txt"
PROMPT="$TMP/prompt.txt"
head -c 9000 "$ROOT/README.md" > "$PROMPT" 2> /dev/null || printf 'The quick brown fox. %.0s' {1..200} > "$PROMPT"

# --- the settings, each with its values from the smallest to the largest (the B-series default marked in the notes)
# generation (one token): name|values|quick values
GEN=(
  "ARC_GVROWS|1 2 4|1 2 4"                 # rows per sub-group of the codebook GEMV (by matrix size)
  "ARC_GEMV_GRF|128 256|128 256"           # the GEMV's register file [128]
  "ARC_GEMV_WAVE|0 1|0 1"                  # [1]
  "ARC_GEMV_DB|0 1|0 1"                    # window read one ahead [1 for dense]
  "ARC_GEMV_PF|0 1|0 1"                    # all rows' reads issued first [MoE: 1]
  "ARC_GEMV_SMALL_WG|0 1|0 1"              # small matrices' work-groups [0]
  "ARC_GEMV_GU|0 1|0 1"                    # gate + up in one launch [0]
  "ARC_GV_IQ3S|0 1|0 1"                    # IQ3_S through gemv_v [0]
  "ARC_IQ3S_TILE|0 1|0 1"                  # IQ3_S repacked as planes at load [0]
  "ARC_WIDE_ROWS|0 1|0 1"                  # row kernels over many work-groups [1]
  "ARC_BATCH|0 8 16 32 64 128|0 32 128"    # kernels per submitted list [0]
  "ARC_ATTN_DEC_GRF|128 256|128 256"       # generation attention registers [256]
  "ARC_ATTN_DEC_SG|128 256 512 1024 2048|256 512 2048"  # sub-groups the key split aims for [512]
  "ARC_ATTN_DEC_SMAX|64 128 256 512 1024|128 256 1024"  # most keys per split [256]
)
# the prompt (4K tokens)
PP=(
  "ARC_GEMM_XW_WGX|1 2 4 8|1 4 8"          # column sub-groups per work-group [4]
  "ARC_GEMM_XW_LOCK|0 1|0 1"               # the work-group in step [0]
  "ARC_GEMM_XW_2D|0 1 2|0 1"               # 2D block loads / stores [1]
  "ARC_GEMM_XW_PF|0 1 2 3|0 1 3"           # prefetch: 0 off, 1 weights, 3 both [3 with 2D]
  "ARC_GEMM_XW_PFD|1 2 4 8|1 2 4"          # prefetch distance in chunks [2]
  "ARC_GEMM_SPLIT|0 9 17 34 68|0 17 34"    # superblocks per split-K piece [34] (changes the sums: KL)
  "ARC_GEMM_GRF|128 256|128 256"           # [128]
  "ARC_H8_GRF|128 256|128 256"             # [128]
  "ARC_H8_NORM|0 1|0 1"                    # [1]
  "ARC_H8_SWI|0 1|0 1"                     # [1]
  "ARC_H8_REUSE|0 1|0 1"                   # [1]
  "ARC_H8_FUSE|0 1|0 1"                    # [1]
  "ARC_ATTN_XMX|0 1|0 1"                   # [1]
  "ARC_ATTN_FX|0 1 2|0 1"                  # flash attention on XMX [1]
  "ARC_GDN_L1PF|0 1 2 4 8 16|0 2 8"        # DeltaNet prefetch distance [2]
  "ARC_GDN_PF|0 2|0 2"                     # DeltaNet operands in registers [0]
  "ARC_GDN_CHUNK|0 1 2 3|0 3"              # chunked DeltaNet [0] (changes the sums: KL)
  "ARC_W8_PARTIAL_GB|0 auto|0 auto"        # decoded weights kept resident [auto] (VRAM)
  "--chunk|1024 2048 4096|2048 4096"       # prompt chunk (at most 4096)
)
# the MoE model (with --moe-model): prompt 4K and generation
MOE=(
  "ARC_MOE_GRF|128 256|128 256"            # [128]
  "ARC_MOE_CG|1 2 4 8|1 2 4"               # [1]
  "ARC_MOE_WY|1 2 4 8|1 2 4"               # [1]
  "ARC_MOE_GR_GU|1 2 4|1 2 4"              # one token: rows per sub-group, gate / up [1]
  "ARC_MOE_GR_DN|1 2 4 8|1 4 8"            # down [4]
  "ARC_MOE_GEMM_PAD|0 1|0 1"               # padded experts (Flash Next) [1]
  "ARC_XALL|0 1|0 1"                       # [1]
)

# every setting the script may set: unset for each run, so each starts from the defaults
UNSET=()
for item in "${GEN[@]}" "${PP[@]}" "${MOE[@]}"; do n="${item%%|*}"; [[ "$n" == ARC_* ]] && UNSET+=(-u "$n"); done
[ -n "$DEVICE" ] && export ZE_AFFINITY_MASK="$DEVICE"

# runs runz with settings "NAME=value ..." ($2) on model $1 and the arguments after; output to $OUT
run() {
  local model="$1" sset="$2"; shift 2
  local -a e=() a=() kv
  for kv in $sset; do
    if [[ "$kv" == --chunk=* ]]; then a+=(--chunk "${kv#*=}"); else e+=("$kv"); fi
  done
  env "${UNSET[@]}" "${e[@]}" "$RUNZ" "$model" "$@" "${a[@]}" > "$OUT" 2>&1
}
best_pp() { grep -aoE 'pp[0-9]+ run [0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$OUT" |
            LC_ALL=C awk '{ v = $(NF-1) + 0; if (v > m) m = v } END { printf "%.1f", m + 0 }'; }
tg_of() { grep -aoE 'tg[0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$OUT" | tail -n 1 |
          LC_ALL=C awk '{ printf "%.2f", $(NF-1) + 0 }'; }
gbs_of() { grep -aoE 'weights read[^|]*' "$OUT" | head -n 1; }
err_of() { grep -aiE 'error|cannot|failed|exception' "$OUT" | head -n 1 | cut -c1-120; }
faster() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > 0 && b > 0 && a >= b * 1.02) }'; }
pct() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { if (b > 0) printf "%+.1f %%", (a / b - 1) * 100 }'; }

measure() {  # model kind sset -> R (tok/s)
  local model="$1" kind="$2" sset="$3"
  if [ "$kind" = gen ]; then run "$model" "$sset" --bench-pp 128 --n 128; R="$(tg_of)"
  else run "$model" "$sset" --bench-pp 4096 --n 0; R="$(best_pp)"; fi
  [ -n "$R" ] && [ "$R" != "0.0" ] && [ "$R" != "0.00" ] || R=0
}
hash_of() {  # model sset -> H (sha256 of the logits of a short prompt and 8 tokens)
  run "$1" "$2" --text-file "$PROMPT" --n 8 --dump-logits "$TMP/l.bin"
  H="$(sha256sum "$TMP/l.bin" 2> /dev/null | cut -c1-16)"; [ -n "$H" ] || H="failed"
}

declare -a WINNERS
sweep() {  # title model kind list-name
  local title="$1" model="$2" kind="$3"; local -n list="$4"
  local item name vals v base_r unit
  unit=$([ "$kind" = gen ] && echo "tok/s (tg128)" || echo "tok/s (pp4096)")
  echo; echo "=== $title: one setting at a time, $unit; the base is the defaults"
  measure "$model" "$kind" ""; base_r="$R"
  printf '%-28s %10s\n' "defaults" "$base_r"
  [ "$kind" = gen ] && echo "  $(gbs_of)"
  WINNERS=()
  for item in "${list[@]}"; do
    name="${item%%|*}"; vals="${item#*|}"
    if [ "$QUICK" = 1 ]; then vals="${vals#*|}"; else vals="${vals%%|*}"; fi
    [ "$kind" = gen ] && [ "$name" = "--chunk" ] && continue
    # (only each setting's fastest winning value goes on to H and C: two values of one setting in C's environment left
    #  the last one listed in force, not the best; hibagus, issue #48, 2026-11-20)
    local best_kv="" best_r=0
    for v in $vals; do
      local kv="$name=$v"; [ "$name" = "--chunk" ] && kv="--chunk=$v"
      measure "$model" "$kind" "$kv"
      local note=""; [ "$R" = 0 ] && note="failed: $(err_of)"
      if faster "$R" "$base_r"; then
        note="WINNER"
        if LC_ALL=C awk -v a="$R" -v b="$best_r" 'BEGIN { exit !(a > b) }'; then best_kv="$kv"; best_r="$R"; fi
      fi
      printf '%-28s %10s  %-9s %s\n' "$kv" "$R" "$(pct "$R" "$base_r")" "$note"
    done
    [ -n "$best_kv" ] && WINNERS+=("$best_kv")
  done
  [ ${#WINNERS[@]} -gt 0 ] && echo "(each setting's best winner, for H and C: ${WINNERS[*]})"
  # (the defaults again: a card that warms or clocks down over the sweep makes late settings look better or worse;
  #  a drift of a few % means a winner of that size needs an alternated A/B before it counts)
  measure "$model" "$kind" ""
  printf '%-28s %10s  %-9s %s\n' "defaults again (drift)" "$R" "$(pct "$R" "$base_r")" ""
}
check_and_combine() {  # model kind
  local model="$1" kind="$2" w ref all=""
  [ ${#WINNERS[@]} -gt 0 ] || { echo "(no setting 2 % faster than the defaults)"; return; }
  echo "--- H. the winners' logits against the defaults (a short prompt and 8 tokens)"
  hash_of "$model" ""; ref="$H"
  for w in "${WINNERS[@]}"; do hash_of "$model" "$w"; printf '%-28s %s\n' "$w" "$([ "$H" = "$ref" ] && echo same || echo "differs ($H): needs a KL check")"; all+="${all:+ }$w"; done
  echo "--- C. all winners together: $all"
  for s in "" "$all"; do
    local label=$([ -z "$s" ] && echo defaults || echo winners)
    run "$model" "$s" --bench-pp 128 --n 128; local g="$(tg_of)"
    local p512 p4k p16k
    run "$model" "$s" --bench-pp 512 --n 0; p512="$(best_pp)"
    run "$model" "$s" --bench-pp 4096 --n 0; p4k="$(best_pp)"
    if [ "$QUICK" = 1 ]; then p16k="-"; else run "$model" "$s" --bench-pp 16384 --n 0; p16k="$(best_pp)"; fi
    printf '%-9s tg128 %8s   pp512 %9s   pp4096 %9s   pp16384 %9s\n' "$label" "$g" "$p512" "$p4k" "$p16k"
  done
}

echo "inferzero xehpc_sweep: $(basename -- "$MODEL")${MOE_MODEL:+, $(basename -- "$MOE_MODEL")}; device ${DEVICE:-default}; $(date -u '+%Y-%m-%d %H:%M UTC')"
run "$MODEL" "" --bench-pp 512 --n 1  # (warm-up: kernels compiled, caches filled)
grep -aE '^inferzero: GPU|^loaded:' "$OUT" | head -n 3
git -C "$ROOT" log -1 --format='commit %h %ad' --date=short 2> /dev/null

sweep "G. generation" "$MODEL" gen GEN; check_and_combine "$MODEL" gen
sweep "P. the prompt" "$MODEL" pp PP; check_and_combine "$MODEL" pp
if [ -n "$MOE_MODEL" ]; then
  run "$MOE_MODEL" "" --bench-pp 512 --n 1
  sweep "M. MoE generation" "$MOE_MODEL" gen MOE; check_and_combine "$MOE_MODEL" gen
  sweep "M. MoE prompt" "$MOE_MODEL" pp MOE; check_and_combine "$MOE_MODEL" pp
fi
echo; echo "done: paste everything above into the issue"
