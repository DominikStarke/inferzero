#!/usr/bin/env bash
# gpu_sweep: every kernel and tile setting a user can change, measured on one GPU with a real model, for any Intel GPU
# (Arc iGPUs, A-series, B-series, Data Center GPU Max). bench/gpu_sweep.ps1 is the same for Windows; the two must stay
# the same (change both together). Paste the whole output into the GitHub issue.
#   bash bench/gpu_sweep.sh --model <Qwen3.8 27B or another dense gguf> [--moe-model <35B-A3B or Flash Next gguf>]
#        [--dflash <DFlash2 drafter folder>] [--device N] [--quick] [--phases GPDTKM]
# It covers the prompt GEMM on DP4a and XMX (W8, H8, f16), the XMX tiles, the GEMVs, attention, DeltaNet, MoE and
# drafting; bench/f33_sweep.ps1 does the chunked DeltaNet's and XMX attention's KL / emulation checks.
# Phases (all by default, in this order; --phases picks some):
#   G  generation (--bench-pp 128 --n 128, tg128): each setting alone from the defaults, every value from the smallest
#      to the largest;
#   P  the prompt (--bench-pp 4096): the same for the prompt's settings;
#   D  drafting on a coding question (256 tokens): plain, the model's MTP block with 2-7 drafts, a DFlash2 drafter
#      (--dflash), each one's text checked against plain (drafts are verified: it must be the same);
#   M  the MoE model's settings (with --moe-model), generation and prompt;
#   then for G / P / M:
#      H  the winners' logits against the defaults (a short prompt, --dump-logits, sha256): "same", or "differs: needs
#         a KL check" (such a value can't be a default without one);
#      C  the best combination: only each setting's best value (a setting with two winning values keeps the faster),
#         added one at a time from the largest gain down, each kept only when the run gets faster still; then the
#         defaults against the combination at tg128 and 512 / 4K / 16K prompts;
#   T  the XMX tile per GEMM kind (the prompt's --prof per gemm_* kernel for each tile setting, two rounds in alternating
#      order): which tile each of the model's GEMMs wants (down: few columns, a long K; up / gate: many columns);
#   K  where the time goes: --prof (the prompt) and --prof-gen (generation) by kernel, the defaults against the best
#      combination, the largest kernels first, with each kernel's change.
# A value wins when it is 2 % faster than the defaults, and the defaults are measured again after each sweep (a drift
# of a few % means a small win needs an alternated A/B before it counts). Prompt chunks stay at most 4096 tokens.
# [--device N]: which GPU (ZE_AFFINITY_MASK; with a flat hierarchy every stack of a Max 1550 is one GPU).
# Needs bash 4.4+, coreutils (sha256sum), awk, grep; runs bin/runz. Expect ~1.5-2.5 hours for the 27B (--quick: ~45 min).

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz"
die() { echo "$*" >&2; exit 1; }
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; }

MODEL=""; MOE_MODEL=""; DEVICE=""; QUICK=0; PHASES="GPDTKM"; DFLASH=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) [ $# -ge 2 ] || die "--model needs a file"; MODEL="$2"; shift 2 ;;
    --moe-model) [ $# -ge 2 ] || die "--moe-model needs a file"; MOE_MODEL="$2"; shift 2 ;;
    --device) [ $# -ge 2 ] || die "--device needs a number"; DEVICE="$2"; shift 2 ;;
    --phases) [ $# -ge 2 ] || die "--phases needs letters (GPDTKM)"; PHASES="$2"; shift 2 ;;
    --dflash) [ $# -ge 2 ] || die "--dflash needs a folder"; DFLASH="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done
[ -n "$MODEL" ] || die "--model is needed (--help)"
[ -f "$MODEL" ] || die "not found: $MODEL"
[ -z "$MOE_MODEL" ] || [ -f "$MOE_MODEL" ] || die "not found: $MOE_MODEL"
[ -x "$RUNZ" ] || die "bin/runz not found: build it first (build.sh)"
[ -z "$DFLASH" ] || [ -d "$DFLASH" ] || die "not a folder: $DFLASH"
[ -z "$DEVICE" ] || [[ "$DEVICE" =~ ^[0-9]+$ ]] || die "--device: a number"
command -v sha256sum > /dev/null || die "sha256sum not found"
has() { [[ "$PHASES" == *"$1"* ]]; }

TMP="$(mktemp -d)" || die "cannot make a temporary folder"
trap 'rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM
OUT="$TMP/out.txt"
PROMPT="$TMP/prompt.txt"
head -c 9000 "$ROOT/README.md" > "$PROMPT" 2> /dev/null || printf 'The quick brown fox. %.0s' {1..200} > "$PROMPT"

# --- the settings: name|values|quick values, each list from the smallest value to the largest
GEN=(
  "ARC_GVROWS|1 2 4|1 2 4"                 # rows per sub-group of the codebook GEMV
  "ARC_GEMV_GRF|128 256|128 256"           # the GEMV's register file
  "ARC_GEMV_MT_GRF|128 256|128 256"        # the 2-8 token GEMV's (verification, batches)
  "ARC_GEMV_WAVE|0 1|0 1"
  "ARC_GEMV_DB|0 1|0 1"                    # the window read one ahead
  "ARC_GEMV_PF|0 1|0 1"                    # all rows' reads issued first
  "ARC_GEMV_SMALL_WG|0 1|0 1"              # small matrices' work-groups
  "ARC_GEMV_GU|0 1|0 1"                    # gate + up in one launch
  "ARC_GV_IQ3S|0 1|0 1"                    # IQ3_S through gemv_v
  "ARC_IQ3S_TILE|0 1|0 1"                  # IQ3_S repacked as planes at load
  "ARC_WIDE_ROWS|0 1|0 1"                  # row kernels over many work-groups
  "ARC_BATCH|0 8 16 32 64 128|0 32 128"    # kernels per submitted list
  "ARC_ATTN_DEC_GRF|128 256|128 256"       # generation attention's registers
  "ARC_ATTN_DEC_SG|128 256 512 1024 2048|256 512 2048"  # sub-groups the key split aims for
  "ARC_ATTN_DEC_SMAX|64 128 256 512 1024|128 256 1024"  # most keys per split
)
PP=(
  "ARC_NO_XMX|0 1|0 1"                     # 1: DP4a only (the prompt GEMMs without XMX)
  "ARC_XMX_FORCE|0 1|0 1"                  # 8-lane iGPUs: XMX without the speed check
  "ARC_H8|0 1|0 1"                         # the Hadamard-rotated int8 GEMM (changes the sums: KL)
  "ARC_H8_MIN|256 1024 100000|256 100000"  # 16-lane iGPUs: the smallest chunk that takes H8
  "ARC_140T|0 1|0 1"                       # the Arc 140T / Alchemist preset
  "ARC_140T_RES|0 1|0 1"                   # its decoded weights resident for short chunks (VRAM)
  "ARC_X16|0 1|0 1"                        # the f16 XMX GEMM for W8 weights
  "ARC_GEMM_TM|16 32|16 32"                # the W8 / DP4a GEMM: tokens per sub-group
  "ARC_GEMM_RG|1 2 4 8|2 4 8"              # token rows swept together per weight slice
  "ARC_GEMM_WGY|2 4 8 16|4 8 16"           # most sub-groups per work-group
  "ARC_GEMM_CG|1 2 4|1 2 4"                # column groups per work-group
  "ARC_GEMM_WPF|0 2 4 8|0 4"               # the weights prefetched n chunks ahead
  "ARC_GEMM_2D|0 1|0 1"                    # its activations by 2D block loads
  "ARC_GEMM_2D_PF|0 1|0 1"                 # and their prefetch
  "ARC_GEMM_XW_WGX|1 2 4 8|1 4 8"          # column sub-groups per work-group
  "ARC_GEMM_XW_LOCK|0 1|0 1"               # the work-group in step
  "ARC_GEMM_XW_2D|0 1 2|0 1"               # 2D block loads / stores
  "ARC_GEMM_XW_PF|0 1 2 3|0 1 3"           # prefetch: 0 off, 1 weights, 2 activations, 3 both
  "ARC_GEMM_XW_PFD|1 2 4 8|1 2 4"          # prefetch distance in chunks
  "ARC_GEMM_XW_GU|0 1|0 1"                 # gate + up in one launch
  "ARC_GEMM_XW_NC_E|1 2|1 2"               # 8 lanes: the n_embd-wide GEMMs' tile width (2: 32 x 32)
  "ARC_GEMM_SPLIT|0 9 17 34 68|0 17 34"    # superblocks per split-K piece (changes the sums: KL)
  "ARC_GEMM_GRF|128 256|128 256"
  "ARC_H8_GRF|128 256|128 256"
  "ARC_H8_NORM|0 1|0 1"
  "ARC_H8_SWI|0 1|0 1"
  "ARC_H8_REUSE|0 1|0 1"
  "ARC_H8_FUSE|0 1|0 1"
  "ARC_ATTN_XMX|0 1|0 1"
  "ARC_ATTN_FX|0 1|0 1"                    # flash attention on 16-lane XMX
  "ARC_ATTN_X8|0 1|0 1"                    # flash attention on 8-lane XMX
  "ARC_GDN_L1PF|0 1 2 4 8 16|0 2 8"        # DeltaNet prefetch distance
  "ARC_GDN_PF|0 2|0 2"                     # DeltaNet operands in registers
  "ARC_GDN_CHUNK|0 1 2 3|0 3"              # chunked DeltaNet (changes the sums: KL)
  "ARC_W8_PARTIAL_GB|0 auto|0 auto"        # decoded weights kept resident (VRAM)
  "--chunk|1024 2048 4096|2048 4096"       # prompt chunk (at most 4096)
)
MOE=(
  "ARC_MOE_GRF|128 256|128 256"
  "ARC_MOE_CG|1 2 4 8|1 2 4"
  "ARC_MOE_WY|1 2 4 8|1 2 4"
  "ARC_MOE_GR_GU|1 2 4|1 2 4"              # one token: rows per sub-group, gate / up
  "ARC_MOE_GR_DN|1 2 4 8|1 4 8"            # down
  "ARC_MOE_TM|16 32|16 32"                 # tokens per expert tile
  "ARC_MOE_RG_XMX|0 1|0 1"                 # the expert GEMMs on XMX from the raw rows
  "ARC_MOE_GEMM_PAD|0 1|0 1"               # padded experts (Flash Next)
  "ARC_XALL|0 1|0 1"
  "ARC_XCPU|0 1|0 1"                       # missed experts on the CPU (changes the sums: KL)
)
TILES8=("32x16 (default)|" "32x32|ARC_GEMM_XW_NC=2" "16x32|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=2" "32x16 WGX 4|ARC_GEMM_XW_WGX=4"
        "16x64|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=4" "64x16|ARC_GEMM_XW_TM=64" "32x16 WGX 1|ARC_GEMM_XW_WGX=1"
        "32x32 WGX 4|ARC_GEMM_XW_NC=2 ARC_GEMM_XW_WGX=4")
TILES16=("32x64 (default)|" "32x32|ARC_GEMM_XW_NC=2" "32x96|ARC_GEMM_XW_NC=6" "32x128|ARC_GEMM_XW_NC=8"
         "16x128|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=8" "64x64|ARC_GEMM_XW_TM=64" "32x64 WGX 1|ARC_GEMM_XW_WGX=1"
         "32x64 WGX 2|ARC_GEMM_XW_WGX=2")

# every variable the script may set is unset for each run, so each starts from the defaults
UNSET=()
for item in "${GEN[@]}" "${PP[@]}" "${MOE[@]}"; do n="${item%%|*}"; [[ "$n" == ARC_* ]] && UNSET+=(-u "$n"); done
UNSET+=(-u ARC_GEMM_XW_TM -u ARC_GEMM_XW_NC)
[ -n "$DEVICE" ] && export ZE_AFFINITY_MASK="$DEVICE"

# progress on stderr (the results on stdout stay clean to paste): the phase, its step of the steps planned, the time so
# far and an estimate of the rest of the phase (its steps so far at their average time). In a terminal one line updated
# in place, else one line per step.
PH_NAME=""; PH_STEP=0; PH_STEPS=0; PH_T0=0; ALL_T0=$SECONDS
phase_start() { PH_NAME="$1"; PH_STEPS="$2"; PH_STEP=0; PH_T0=$SECONDS; }
fmt_t() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }
progress() {  # what this run is
  PH_STEP=$((PH_STEP + 1))
  local el=$((SECONDS - PH_T0)) left="" steps=$PH_STEPS
  [ "$PH_STEP" -gt "$steps" ] && steps=$PH_STEP
  [ "$PH_STEP" -gt 1 ] && left=", ~$(fmt_t $((el * (steps - PH_STEP + 1) / (PH_STEP - 1)))) left in it"
  local msg; msg="$(printf '[%s %d/%d] %s | %s in this phase%s, %s in all' "$PH_NAME" "$PH_STEP" "$steps" "${1:0:40}" \
                    "$(fmt_t "$el")" "$left" "$(fmt_t $((SECONDS - ALL_T0)))")"
  if [ -t 2 ]; then printf '\r\033[K%s' "$msg" >&2; else printf '%s\n' "$msg" >&2; fi
}
progress_done() { [ -t 2 ] && printf '\r\033[K' >&2; return 0; }

# runs runz on model $1 with settings "NAME=value ..." ($2) and the arguments after; its output in $OUT
run() {
  local model="$1" sset="$2"; shift 2
  progress "${sset:-defaults} ${*:1:4}"
  local -a e=() a=() kv
  for kv in $sset; do
    if [[ "$kv" == --chunk=* ]]; then a+=(--chunk "${kv#*=}"); else e+=("$kv"); fi
  done
  env "${UNSET[@]}" "${e[@]}" "$RUNZ" "$model" "$@" "${a[@]}" > "$OUT" 2>&1
  progress_done
}
# the values a phase's sweep will run (for its progress)
count_values() {  # list-name kind
  local -n l="$1"; local n=0 item vals
  for item in "${l[@]}"; do
    [ "$2" = gen ] && [ "${item%%|*}" = "--chunk" ] && continue
    vals="${item#*|}"; if [ "$QUICK" = 1 ]; then vals="${vals#*|}"; else vals="${vals%%|*}"; fi
    set -- $vals; n=$((n + $#))
  done
  echo "$n"
}
best_pp() { grep -aoE 'pp[0-9]+ run [0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$OUT" |
            LC_ALL=C awk '{ v = $(NF-1) + 0; if (v > m) m = v } END { printf "%.1f", m + 0 }'; }
tg_of() { grep -aoE 'tg[0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$OUT" | tail -n 1 |
          LC_ALL=C awk '{ printf "%.2f", $(NF-1) + 0 }'; }
gbs_of() { grep -aoE 'weights read[^|]*' "$OUT" | head -n 1; }
err_of() { grep -aiE 'error|cannot|failed|exception' "$OUT" | head -n 1 | cut -c1-120; }
gain() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { if (a > 0 && b > 0) printf "%.4f", a / b; else print 0 }'; }
ge() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }
pct() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { if (b > 0) printf "%+.1f %%", (a / b - 1) * 100 }'; }
measure() {  # model kind sset -> R (tok/s)
  if [ "$2" = gen ]; then run "$1" "$3" --bench-pp 128 --n 128; R="$(tg_of)"
  else run "$1" "$3" --bench-pp 4096 --n 0; R="$(best_pp)"; fi
  [ -n "$R" ] && [ "$R" != "0.0" ] && [ "$R" != "0.00" ] || R=0
}
hash_of() {  # model sset -> H
  run "$1" "$2" --text-file "$PROMPT" --n 8 --dump-logits "$TMP/l.bin"
  H="$(sha256sum "$TMP/l.bin" 2> /dev/null | cut -c1-16)"; [ -n "$H" ] || H="failed"
}
# the kernel table of the last run ("  name  ms  share%  (n launches)") -> name<TAB>ms lines
prof_rows() { LC_ALL=C awk '/^GPU time by kernel/ { on = 1; next } on && /^[ \t]+[A-Za-z0-9_]+[ \t]+[0-9.]+ ms/ { print $1 "\t" $2 }' "$OUT"; }

# --- G / P / M: each setting alone; WINNERS holds each setting's best value only, as "gain<TAB>NAME=value"
declare -a WINNERS
sweep() {  # title model kind list-name
  local title="$1" model="$2" kind="$3"; local -n list="$4"
  local item name vals v base_r unit best_v best_r kv
  unit=$([ "$kind" = gen ] && echo "tok/s (tg128)" || echo "tok/s (pp4096)")
  echo; echo "=== $title: one setting at a time, $unit; the base is the defaults"
  phase_start "${title%%.*}" $(($(count_values "$4" "$kind") + 2))
  measure "$model" "$kind" ""; base_r="$R"; BASE_R="$R"
  printf '%-28s %10s\n' "defaults" "$base_r"
  [ "$kind" = gen ] && echo "  $(gbs_of)"
  WINNERS=()
  for item in "${list[@]}"; do
    name="${item%%|*}"; vals="${item#*|}"
    if [ "$QUICK" = 1 ]; then vals="${vals#*|}"; else vals="${vals%%|*}"; fi
    [ "$kind" = gen ] && [ "$name" = "--chunk" ] && continue
    best_v=""; best_r=0
    for v in $vals; do
      kv="$name=$v"; [ "$name" = "--chunk" ] && kv="--chunk=$v"
      measure "$model" "$kind" "$kv"
      local note=""; [ "$R" = 0 ] && note="failed: $(err_of)"
      if ge "$R" "$(LC_ALL=C awk -v b="$base_r" 'BEGIN { print b * 1.02 }')"; then
        note="wins"
        if ge "$R" "$best_r" && [ "$R" != "$best_r" ]; then best_r="$R"; best_v="$kv"; fi
      fi
      printf '%-28s %10s  %-9s %s\n' "$kv" "$R" "$(pct "$R" "$base_r")" "$note"
    done
    [ -n "$best_v" ] && WINNERS+=("$(gain "$best_r" "$base_r")	$best_v")
  done
  measure "$model" "$kind" ""
  printf '%-28s %10s  %-9s %s\n' "defaults again (drift)" "$R" "$(pct "$R" "$base_r")" ""
}
BEST_SET=""
check_and_combine() {  # model kind
  local model="$1" kind="$2" ref w line cur cur_r kv
  BEST_SET=""
  [ ${#WINNERS[@]} -gt 0 ] || { echo "(no setting 2 % faster than the defaults)"; return; }
  echo "--- H. each setting's best value: its logits against the defaults (a short prompt and 8 tokens)"
  phase_start "H" $((${#WINNERS[@]} + 1))
  hash_of "$model" ""; ref="$H"
  for line in "${WINNERS[@]}"; do
    kv="${line#*	}"; hash_of "$model" "$kv"
    printf '%-28s %s\n' "$kv" "$([ "$H" = "$ref" ] && echo same || echo "differs ($H): needs a KL check")"
  done
  echo "--- C. the best combination: each setting's best value, added from the largest gain down, kept when faster"
  phase_start "C" $((${#WINNERS[@]} + 1 + 2 * (QUICK == 1 ? 3 : 4)))
  cur=""; measure "$model" "$kind" ""; cur_r="$R"
  printf '%-40s %10s\n' "defaults" "$cur_r"
  while IFS= read -r line; do
    kv="${line#*	}"
    measure "$model" "$kind" "${cur:+$cur }$kv"
    if ge "$R" "$(LC_ALL=C awk -v b="$cur_r" 'BEGIN { print b * 1.01 }')"; then
      printf '%-40s %10s  %-9s kept\n' "+ $kv" "$R" "$(pct "$R" "$cur_r")"; cur="${cur:+$cur }$kv"; cur_r="$R"
    else
      printf '%-40s %10s  %-9s left out\n' "+ $kv" "$R" "$(pct "$R" "$cur_r")"
    fi
  done < <(printf '%s\n' "${WINNERS[@]}" | sort -t '	' -k1,1 -g -r)
  BEST_SET="$cur"
  echo "the best combination: ${cur:-the defaults}"
  local g p512 p4k p16k label s
  local -a both=("defaults|")
  [ -n "$cur" ] && both+=("best|$cur")
  for s in "${both[@]}"; do
    label="${s%%|*}"; s="${s#*|}"
    run "$model" "$s" --bench-pp 128 --n 128; g="$(tg_of)"
    run "$model" "$s" --bench-pp 512 --n 0; p512="$(best_pp)"
    run "$model" "$s" --bench-pp 4096 --n 0; p4k="$(best_pp)"
    if [ "$QUICK" = 1 ]; then p16k="-"; else run "$model" "$s" --bench-pp 16384 --n 0; p16k="$(best_pp)"; fi
    printf '%-9s tg128 %8s   pp512 %9s   pp4096 %9s   pp16384 %9s\n' "$label" "$g" "$p512" "$p4k" "$p16k"
  done
}

# --- D: drafting, on a coding question (Qwen chat format, thinking off: the answer starts at once)
QUESTION="$TMP/question.txt"
printf '%s' $'<|im_start|>user\nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using '\
$'a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>\n<|im_start|>assistant\n'\
$'<think>\n\n</think>\n\n' > "$QUESTION"
gen_tps() { grep -aoE 'generation: [0-9]+ tokens in [0-9.]+ ms \([0-9.]+ t/s\)' "$OUT" | tail -n 1 | grep -oE '[0-9.]+ t/s' | cut -d' ' -f1; }
text_hash() { awk '/^----- text -----$/ { on = 1; next } /^----------------$/ { on = 0 } on' "$OUT" | sha256sum | cut -c1-12; }
drafting() {  # model
  local model="$1" ref t h d
  echo; echo "=== D. drafting: one coding question, 256 tokens, the best of 2 runs (tok/s); every text against plain"
  local nd=8; [ -n "$DFLASH" ] && nd=10
  phase_start "D" $((nd * 2))
  local -a RUNS=("plain|" "mtp (default drafts)|--mtp")
  for d in 2 3 4 5 6 7; do RUNS+=("mtp, $d drafts|--mtp --mtp-drafts $d"); done
  [ -n "$DFLASH" ] && RUNS+=("dflash (auto)|--dflash $DFLASH --draft auto" "dflash only|--dflash $DFLASH --draft dflash")
  ref=""
  for x in "${RUNS[@]}"; do
    local name="${x%%|*}" args="${x#*|}" best=0
    for rep in 1 2; do
      # shellcheck disable=SC2086
      run "$model" "" --text-file "$QUESTION" --n 256 --show $args
      t="$(gen_tps)"; [ -n "$t" ] || t=0
      LC_ALL=C awk -v a="$t" -v b="$best" 'BEGIN { exit !(a > b) }' && best="$t"
    done
    h="$(text_hash)"; [ -z "$ref" ] && ref="$h"
    local acc; acc="$(grep -aoE 'MTP: [0-9]+ drafts, [0-9]+ accepted \([0-9.]+ %\)' "$OUT" | tail -n 1)"
    printf '%-24s %8s tok/s  %-9s %s%s\n' "$name" "$best" "$( [ "$name" = plain ] && echo "" || pct "$best" "$PLAIN_TPS" )" \
      "$( [ "$h" = "$ref" ] && echo "same text" || echo "TEXT DIFFERS from plain" )" "${acc:+; $acc}"
    [ "$name" = plain ] && PLAIN_TPS="$best"
  done
}
PLAIN_TPS=0

# --- T: the XMX tile per GEMM kind
tiles() {  # model
  local model="$1" r i j N name e
  local -a SETS
  if [ "$LANES8" = 1 ]; then SETS=("${TILES8[@]}"); else SETS=("${TILES16[@]}"); fi
  [ "$QUICK" = 1 ] && SETS=("${SETS[@]:0:4}")
  echo; echo "=== T. the XMX tile per GEMM kind (the prompt's --prof, ms; the lower of 2 rounds, alternating order)"
  phase_start "T" $((1 + 2 * ${#SETS[@]}))
  local RES="$TMP/tiles.tsv"; : > "$RES"; N=${#SETS[@]}
  run "$model" "" --bench-pp 4096 --n 0  # (untimed: the first measured setting ran ~8 % slow without it)
  for ((r = 1; r <= 2; r++)); do
    for ((i = 0; i < N; i++)); do
      j=$i; [ $r = 2 ] && j=$((N - 1 - i))
      name="${SETS[$j]%%|*}"; e="${SETS[$j]#*|}"
      run "$model" "$e" --bench-pp 4096 --n 0 --prof
      prof_rows | awk -v s="$name" -F'\t' '$1 ~ /^gemm_/ { print s "\t" $1 "\t" $2 }' >> "$RES"
      printf '%s\t@pp\t%s\n' "$name" "$(best_pp)" >> "$RES"
      grep -aE 'H8 GEMMs on XMX with|gemm_xw.* differs|not supported' "$OUT" | head -n 1 |
        awk -v s="$name" '{ sub(/^[ \t]+/, ""); print s "\t@note\t" $0 }' >> "$RES"
    done
  done
  LC_ALL=C awk -F'\t' -v sets="$(printf '%s\n' "${SETS[@]%%|*}")" '
    BEGIN { n = split(sets, S, "\n"); while (n > 0 && S[n] == "") n-- }
    $2 == "@pp" { if (!(($1) in pp) || $3 + 0 > pp[$1]) pp[$1] = $3 + 0; next }
    $2 == "@note" { note[$1] = $3; next }
    { k = $2; if (!(k in seen)) { seen[k] = 1; K[++nk] = k }
      key = $1 SUBSEP k; if (!(key in ms) || $3 + 0 < ms[key]) ms[key] = $3 + 0 }
    END {
      printf "%-16s", "GEMM"; for (i = 1; i <= n; i++) printf " %15s", S[i]; printf "   fastest\n"
      for (q = 1; q <= nk; q++) {
        k = K[q]; printf "%-16s", k; best = ""; bv = 1e30
        for (i = 1; i <= n; i++) { key = S[i] SUBSEP k
          if (key in ms) { printf " %15.1f", ms[key]; if (ms[key] < bv) { bv = ms[key]; best = S[i] } } else printf " %15s", "-" }
        d = S[1] SUBSEP k
        if (d in ms) { sd += ms[d]; sb += bv; printf "   %s (%+.1f %%)\n", best, 100 * (bv / ms[d] - 1) } else printf "   %s\n", best
      }
      printf "%-16s", "prompt tok/s"; for (i = 1; i <= n; i++) printf " %15s", (S[i] in pp) ? sprintf("%.1f", pp[S[i]]) : "failed"
      printf "\nall GEMMs at the default: %.1f ms; each GEMM at its fastest tile: %.1f ms (%+.1f %%)\n", sd, sb, (sd > 0 ? 100 * (sb / sd - 1) : 0)
      for (i = 1; i <= n; i++) printf "  %-16s %s\n", S[i], (S[i] in note) ? note[S[i]] : "(no tile line)"
    }' "$RES"
}

# --- K: where the time goes, the defaults against a settings set
kernels() {  # model what(prompt|generation) sset label
  local model="$1" what="$2" sset="$3" label="$4"
  local -a a
  if [ "$what" = prompt ]; then a=(--bench-pp 4096 --n 0 --prof); else a=(--bench-pp 128 --n 64 --prof-gen); fi
  run "$model" "" "${a[@]}"; prof_rows > "$TMP/k0.tsv"
  local r0="$( [ "$what" = prompt ] && best_pp || tg_of )"
  if [ -n "$sset" ]; then run "$model" "$sset" "${a[@]}"; prof_rows > "$TMP/k1.tsv"; else cp "$TMP/k0.tsv" "$TMP/k1.tsv"; fi
  local r1="$( [ "$what" = prompt ] && best_pp || tg_of )"
  echo "--- K. $label, $what by kernel (ms of the last profiled $( [ "$what" = prompt ] && echo "4K prompt" || echo "generation" )); defaults $r0 tok/s, best $r1 tok/s while profiled (slower than unprofiled runs)"
  LC_ALL=C awk -F'\t' '
    FNR == NR { a[$1] = $2; t0 += $2; if (!($1 in o)) { o[$1] = 1; K[++n] = $1 } next }
    { b[$1] = $2; t1 += $2; if (!($1 in o)) { o[$1] = 1; K[++n] = $1 } }
    END {
      printf "%-22s %10s %7s %10s %7s %9s\n", "kernel", "defaults", "share", "best", "share", "change"
      for (i = 1; i <= n && i <= 25; i++) { k = K[i]; x = a[k] + 0; y = b[k] + 0
        printf "%-22s %10.1f %6.1f%% %10.1f %6.1f%% %+8.1f%%\n", k, x, (t0 ? 100 * x / t0 : 0), y, (t1 ? 100 * y / t1 : 0), (x ? 100 * (y / x - 1) : 0) }
      if (n > 25) printf "(%d more kernels)\n", n - 25
      printf "%-22s %10.1f %7s %10.1f %7s %+8.1f%%\n", "total", t0, "", t1, "", (t0 ? 100 * (t1 / t0 - 1) : 0)
    }' "$TMP/k0.tsv" "$TMP/k1.tsv"
}

echo "inferzero gpu_sweep: $(basename -- "$MODEL")${MOE_MODEL:+, $(basename -- "$MOE_MODEL")}; device ${DEVICE:-default}; phases $PHASES$([ "$QUICK" = 1 ] && echo ", quick"); $(date -u '+%Y-%m-%d %H:%M UTC')"
git -C "$ROOT" log -1 --format='commit %h %ad' --date=short 2> /dev/null
run "$MODEL" "" --bench-pp 1024 --n 1  # (warm-up: kernels compiled, caches filled)
grep -aE '^inferzero: (GPU|XMX|H8|the H8|prompt attention)|^loaded:' "$OUT" | head -n 8
LANES8=1; grep -aq 'no 8-lane sub-groups' "$OUT" && LANES8=0
XMX=1; grep -aqE 'XMX on for the prompt GEMMs|H8 GEMMs on XMX' "$OUT" || XMX=0

GEN_BEST=""; PP_BEST=""
if has G; then sweep "G. generation" "$MODEL" gen GEN; check_and_combine "$MODEL" gen; GEN_BEST="$BEST_SET"; fi
if has P; then sweep "P. the prompt" "$MODEL" pp PP; check_and_combine "$MODEL" pp; PP_BEST="$BEST_SET"; fi
if has D; then drafting "$MODEL"; fi
if has T; then
  if [ "$XMX" = 1 ]; then tiles "$MODEL"; else echo; echo "=== T. skipped: this GPU's log shows no XMX prompt GEMMs (no tiles to choose)"; fi
fi
if has K; then
  echo; echo "=== K. where the time goes"
  phase_start "K" 4
  kernels "$MODEL" prompt "$PP_BEST" "the prompt's best combination"
  kernels "$MODEL" generation "$GEN_BEST" "generation's best combination"
fi
if has M && [ -n "$MOE_MODEL" ]; then
  run "$MOE_MODEL" "" --bench-pp 1024 --n 1
  sweep "M. MoE generation" "$MOE_MODEL" gen MOE; check_and_combine "$MOE_MODEL" gen; MG="$BEST_SET"
  sweep "M. MoE prompt" "$MOE_MODEL" pp MOE; check_and_combine "$MOE_MODEL" pp; MP="$BEST_SET"
  if has K; then
    kernels "$MOE_MODEL" prompt "$MP" "the MoE prompt's best combination"
    kernels "$MOE_MODEL" generation "$MG" "MoE generation's best combination"
  fi
fi
echo; echo "done: paste everything above into the issue"
