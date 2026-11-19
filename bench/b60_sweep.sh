#!/usr/bin/env bash
# bench/b60_sweep.ps1 is the same script for Windows; the two must stay the same (change both together).
# Sweeps the prompt GEMM's settings on a discrete Arc card (made for the Pro B60, issue #4) around the two best paths
# so far: the default (XMX with 2D loads and their prefetch: 764 tok/s on the 27B, 4K prompt) and H8 (ARC_H8=1,
# Hadamard-rotated int8 with one scale per column and token: 918 tok/s). Each setting is changed alone from each base,
# the winners are put together, then measured at prompt sizes of real use, and their accuracy is checked against the
# default by KL. With --moe-model also the 35B-A3B's expert GEMM shapes (ARC_MOE_CG / ARC_MOE_WY) and H8 there.
# Paste the whole output into the GitHub issue.
#   bash bench/b60_sweep.sh --model ~/hf/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [--moe-model ~/hf/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf]
#   [--tokens 4096]  phase A's prompt length   [--quick]  fewer values, no 16K row   [--gpus 2 4] (or 2,4)  phase E's GPU counts
# A setting wins only 3 % ahead of its base; "best" is the faster of all winners together and the single best change.
# Settings (defaults on discrete cards in brackets):
#   TM      ARC_GEMM_TM, tokens per sub-group [16]      WGY    ARC_GEMM_WGY, sub-groups per work-group [8]
#   RG      ARC_GEMM_RG, rows per weight slice [4]       CG     ARC_GEMM_CG, column groups per work-group [1]
#   GRF     ARC_GEMM_GRF, registers per thread [128]     SPLIT  ARC_GEMM_SPLIT, superblocks per split-K piece [17; H8 34]
#   X2D     2D loads: pf (with the prefetch) / on / off [pf]
#   WPF     ARC_GEMM_WPF, the weights prefetched n 64-value chunks ahead, 0 = off [0]
#   FUSE    ARC_H8_FUSE, H8's weight rotation in the decode pass (1) or after it (0) [1]; H8 only, bit-identical
#   MOECG / MOEWY  ARC_MOE_CG / ARC_MOE_WY, sub-groups per work-group of the expert GEMM [1 / 1] (35B only)
# Needs bash 4.4 or newer and the standard tools (awk, sed, grep, coreutils); runs bin/runz and bin/mgpu_probe.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz"
PROBE="$ROOT/bin/mgpu_probe"
VARS=(ARC_H8 ARC_GEMM_TM ARC_GEMM_WGY ARC_GEMM_RG ARC_GEMM_CG ARC_GEMM_GRF ARC_GEMM_SPLIT ARC_GEMM_2D
      ARC_GEMM_2D_PF ARC_MOE_CG ARC_MOE_WY ARC_GEMM_WPF ARC_H8_FUSE)

usage() {
  # (the comment at the top of this file, without its "# ")
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}
die() { echo "$*" >&2; exit 1; }
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# --- the options
MODEL=""
MOE_MODEL=""
TOKENS=4096
GPUS=(2 4)
QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) [ $# -ge 2 ] || die "--model needs a file"; MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --moe-model) [ $# -ge 2 ] || die "--moe-model needs a file"; MOE_MODEL="$2"; shift 2 ;;
    --moe-model=*) MOE_MODEL="${1#*=}"; shift ;;
    --tokens) [ $# -ge 2 ] || die "--tokens needs a number"; TOKENS="$2"; shift 2 ;;
    --tokens=*) TOKENS="${1#*=}"; shift ;;
    --gpus|--gpus=*)
      # (--gpus 2 4 and --gpus 2,4 both work, as -Gpus 2,4 does in the .ps1)
      GPUS=()
      if [[ "$1" == --gpus=* ]]; then list=("${1#*=}"); shift; else shift; list=(); while [ $# -gt 0 ] && [[ "$1" != -* ]]; do list+=("$1"); shift; done; fi
      for item in "${list[@]}"; do
        IFS=, read -ra parts <<< "$item"
        for g in "${parts[@]}"; do
          g="${g//[[:space:]]/}"
          [ -n "$g" ] || continue
          is_int "$g" || die "--gpus takes whole numbers: $g"
          GPUS+=("$g")
        done
      done
      [ ${#GPUS[@]} -gt 0 ] || die "--gpus needs one or more GPU counts"
      ;;
    --quick) QUICK=1; shift ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "--model is required (see --help)" >&2; exit 2; }
is_int "$TOKENS" || die "--tokens takes a whole number: $TOKENS"
[ -f "$RUNZ" ] || die "bin/runz not found (run from the inferzero folder)"
[ -e "$MODEL" ] || die "model not found: $MODEL"
[ -f "$PROBE" ] || die "bin/mgpu_probe not found"

TMP="$(mktemp -d)" || die "cannot make a temporary folder"
trap 'rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM
OUT="$TMP/out.txt"

# (env -u for every variable of the script: each run starts from the launcher's environment without them)
UNSET=()
for v in "${VARS[@]}"; do UNSET+=(-u "$v"); done

# --- setting sets: words "NAME=value" separated by spaces, e.g. "H8=on CG=2"
set_get() {  # set name -> value ("" when not in the set)
  local kv; local -a kvs
  read -ra kvs <<< "$1"
  for kv in "${kvs[@]}"; do [ "${kv%%=*}" = "$2" ] && { printf '%s' "${kv#*=}"; return; }; done
}
set_put() {  # set name value -> the set with name=value
  local kv out=""; local -a kvs
  read -ra kvs <<< "$1"
  for kv in "${kvs[@]}"; do [ "${kv%%=*}" = "$2" ] || out+="${out:+ }$kv"; done
  printf '%s' "${out:+$out }$2=$3"
}
desc() {  # "defaults", or the set's NAME=value sorted by name
  if [ -z "$1" ]; then echo "defaults"; else tr ' ' '\n' <<< "$1" | LC_ALL=C sort -t '=' -k1,1 | paste -sd' ' -; fi
}
# the environment of a setting set (name -> value as in the table above), one VAR=value per line
env_of() {
  local kv k v; local -a kvs
  read -ra kvs <<< "$1"
  for kv in "${kvs[@]}"; do
    k="${kv%%=*}"; v="${kv#*=}"
    case "$k" in
      H8) if [ "$v" = "on" ]; then echo "ARC_H8=1"; else echo "ARC_H8=0"; fi ;;  # (off: ARC_H8=0, since H8 is on by default for Qwen dense models)
      TM) echo "ARC_GEMM_TM=$v" ;;
      WGY) echo "ARC_GEMM_WGY=$v" ;;
      RG) echo "ARC_GEMM_RG=$v" ;;
      CG) echo "ARC_GEMM_CG=$v" ;;
      GRF) echo "ARC_GEMM_GRF=$v" ;;
      SPLIT) echo "ARC_GEMM_SPLIT=$v" ;;
      X2D) if [ "$v" = "off" ]; then echo "ARC_GEMM_2D=0"; else echo "ARC_GEMM_2D=1"; fi
           if [ "$v" = "pf" ]; then echo "ARC_GEMM_2D_PF=1"; else echo "ARC_GEMM_2D_PF=0"; fi ;;
      WPF) echo "ARC_GEMM_WPF=$v" ;;
      FUSE) echo "ARC_H8_FUSE=$v" ;;
      MOECG) echo "ARC_MOE_CG=$v" ;;
      MOEWY) echo "ARC_MOE_WY=$v" ;;
    esac
  done
}
# one runz call with the settings $1 and the arguments after it; its output (stdout and stderr) goes to $OUT
run_runz() {
  local sset="$1"; shift
  local -a e=()
  mapfile -t e < <(env_of "$sset")
  env "${UNSET[@]}" "${e[@]}" "$RUNZ" "$@" > "$OUT" 2>&1
}
first_error() { grep -aiE 'error|cannot|failed|exception' "$1" | head -n 1; }
trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
fnum() { LC_ALL=C awk -v x="$1" -v f="%$2.$3f" 'BEGIN { printf f, x }'; }  # number width decimals
ftrue() { LC_ALL=C awk "BEGIN { exit !($1) }"; }  # a comparison of numbers

# a prompt benchmark: best of 3 passes, and the path the engine took; sets R_TPS and R_PATH
bench() {
  local model="$1" sset="$2" pp="$3" chunk="${4:-0}" gpus="${5:-1}" tps path
  local -a a=("$model" --bench-pp "$pp" --n 1)
  [ "$chunk" -gt 0 ] && a+=(--chunk "$chunk")
  [ "$gpus" -gt 1 ] && a+=(--gpus "$gpus")
  run_runz "$sset" "${a[@]}"
  # (the best pass as runz printed it; as in the .ps1, no pass, or one pass at 0, counts as failed)
  tps="$(grep -aoE 'pp[0-9]+ run [0-9]+: [0-9.]+ ms[[:space:]]+->[[:space:]]+[0-9.]+ tokens/s' "$OUT" |
         LC_ALL=C awk '{ v = $(NF-1) + 0; n++; if (n == 1 || v > m) { m = v; s = $(NF-1) } }
                       END { if (n > 0 && !(n == 1 && m == 0)) print s }')"
  if grep -aqi 'XMX on for the prompt GEMMs' "$OUT"; then path="XMX"
  elif grep -aqi 'XMX results differ' "$OUT"; then path="DP4a (XMX differed!)"
  else path="DP4a"; fi
  if grep -aqi 'with 2D block loads' "$OUT"; then path+=" + 2D"
  elif grep -aqiE '2D block loads \(ARC_GEMM_2D\) do not build or differ' "$OUT"; then path+=" (2D differed!)"; fi
  if grep -aqi 'H8 on for the prompt GEMMs' "$OUT"; then path+=" + H8"; fi
  if [ -z "$tps" ]; then R_TPS=0; R_PATH="failed: $(first_error "$OUT")"; return; fi
  R_TPS="$tps"; R_PATH="$path"
}
line() { printf '%-40s %s tok/s  %s\n' "$1" "$(fnum "$2" 9 1)" "$3"; }  # what tps path
bench_line() { local what="$1"; shift; bench "$@"; line "$what" "$R_TPS" "$R_PATH"; }

# (in this order)
GRID_KEYS=(TM WGY RG CG GRF SPLIT X2D WPF FUSE)
grid_vals() {
  case "$1" in
    TM) echo "16 32" ;;
    WGY) if [ "$QUICK" = 1 ]; then echo "4 8 16"; else echo "2 4 8 16"; fi ;;
    RG) if [ "$QUICK" = 1 ]; then echo "2 4 8"; else echo "1 2 4 8"; fi ;;
    CG) echo "1 2 4" ;;
    GRF) echo "128 256" ;;
    SPLIT) if [ "$QUICK" = 1 ]; then echo "0 17 34"; else echo "0 9 17 34"; fi ;;
    X2D) echo "pf on off" ;;
    WPF) if [ "$QUICK" = 1 ]; then echo "0 4"; else echo "0 2 4 8"; fi ;;
    FUSE) echo "1 0" ;;
  esac
}
declare -A DEFAULTS=([TM]=16 [WGY]=8 [RG]=4 [CG]=1 [GRF]=128 [SPLIT]=17 [X2D]=pf [WPF]=0 [FUSE]=1)

probe_gpus() { env "$@" "$PROBE" 2>&1 | grep -aiE '^[[:space:]]+GPU [0-9]+:' | trim; }

probe_gpus
echo "model: $(basename -- "$MODEL"); phase A prompt: $TOKENS tokens; tok/s = best of 3 passes"
bench "$MODEL" "" 512  # (warm-up: the first run after a start is slower)
declare -A TOGETHER
for base in "default (no H8)|H8=off" "H8|H8=on"; do
  bname="${base%%|*}"; bset="${base#*|}"
  echo
  echo "=== A. $bname: one setting at a time"
  bench "$MODEL" "$bset" "$TOKENS"; b_tps="$R_TPS"
  line "base" "$R_TPS" "$R_PATH"
  # (H8 runs at 4 column groups by default since 2026-11-08: its base is CG 4, so CG 1 and 2 are the changes; and at
  #  split-K pieces of 34 superblocks since 2026-11-15 (engine_h8.h h8_splits): its base is SPLIT 34, 17 a change)
  declare -A bdef=()
  for p in "${!DEFAULTS[@]}"; do bdef[$p]="${DEFAULTS[$p]}"; done
  h8on=0
  if [ "$(set_get "$bset" H8)" = "on" ]; then h8on=1; bdef[CG]=4; bdef[SPLIT]=34; fi
  best_set="$bset"
  single="$bset"; single_t="$b_tps"
  for p in "${GRID_KEYS[@]}"; do
    if [ "$p" = "FUSE" ] && [ "$h8on" = 0 ]; then continue; fi  # (H8's switch: nothing to measure without it)
    bv="${bdef[$p]}"; bt="$b_tps"
    for v in $(grid_vals "$p"); do
      [ "$v" = "${bdef[$p]}" ] && continue  # (the base: measured above)
      s="$(set_put "$bset" "$p" "$v")"
      bench "$MODEL" "$s" "$TOKENS"
      line "$p=$v" "$R_TPS" "$R_PATH"
      # (a winner must be 3 % ahead of the base, past the run-to-run drift)
      if ftrue "$R_TPS > $bt * 1.03 && $R_TPS > $b_tps * 1.03"; then bt="$R_TPS"; bv="$v"; fi
      if ftrue "$R_TPS > $single_t && $R_TPS > $b_tps * 1.03"; then single_t="$R_TPS"; single="$s"; fi
    done
    if [ "$bv" != "${bdef[$p]}" ]; then best_set="$(set_put "$best_set" "$p" "$bv")"; fi
  done
  unset bdef
  # (winners alone do not always add up: settings of the same work-group can work against each other, so the single
  #  best change is kept too, and whichever of the two is faster is "best")
  bench "$MODEL" "$best_set" "$TOKENS"; r_tps="$R_TPS"
  line "best of each together" "$R_TPS" "$R_PATH"
  echo "  ($(desc "$best_set"))"
  bench_line "base again (drift)" "$MODEL" "$bset" "$TOKENS"
  echo "  single best change: $(desc "$single")"
  if ftrue "$r_tps >= $single_t"; then TOGETHER[$bname]="$best_set"; else TOGETHER[$bname]="$single"; fi
  echo "  taken as best: $(desc "${TOGETHER[$bname]}")"
done

echo
echo "=== B. prompt sizes of real use (default chunks unless noted)"
SET_NAMES=("default" "no H8" "H8" "best without H8" "best with H8")
declare -A SETS=(["default"]="" ["no H8"]="H8=off" ["H8"]="H8=on" ["best without H8"]="${TOGETHER["default (no H8)"]}"
                 ["best with H8"]="${TOGETHER["H8"]}")
SIZES=("512|0|512 (a chat turn)" "4096|0|4096 (a document)" "16384|0|16384 (a long prompt)"
       "16384|4096|16384 in 4096-token chunks")
if [ "$QUICK" = 1 ]; then SIZES=("${SIZES[@]:0:2}"); fi
for z in "${SIZES[@]}"; do
  IFS='|' read -r zpp zchunk zname <<< "$z"
  for n in "${SET_NAMES[@]}"; do
    bench_line "$(printf '%-27s %-16s' "$zname" "$n")" "$MODEL" "${SETS[$n]}" "$zpp" "$zchunk"
  done
done

echo
echo "=== C. accuracy against the plain path (no H8): KL over 4 perplexity chunks of USAGE.md"
KL="$TMP/b60_sweep_base.kl"
TEXT="$ROOT/USAGE.md"
run_runz "H8=off" "$MODEL" --perplexity "$TEXT" --ppl-chunks 4 --ppl-kl-save "$KL"  # (the plain path, as before H8 was a default)
l="$(grep -aiE '^Final|perplexity|PPL' "$OUT" | tail -n 1)"
if [ -n "$l" ]; then echo "no H8: $(trim <<< "$l")"; fi
for n in "default" "H8" "best without H8" "best with H8"; do
  run_runz "${SETS[$n]}" "$MODEL" --perplexity "$TEXT" --ppl-chunks 4 --ppl-kl "$KL"
  l="$(grep -aiE 'KL vs base' "$OUT" | tail -n 1)"
  if [ -n "$l" ]; then l="$(trim <<< "$l")"; else l="no KL line"; fi
  ppl="$(grep -aoE 'perplexity: [0-9.]+' "$OUT" | head -n 1)"
  if [ -n "$ppl" ]; then ppl="${ppl#perplexity: }"; else ppl="?"; fi
  printf '%-18s %s  perplexity %s  (%s)\n' "$n" "$l" "$ppl" "$(desc "${SETS[$n]}")"
done
rm -f "$KL"

if [ -n "$MOE_MODEL" ]; then
  echo
  echo "=== D. $(basename -- "$MOE_MODEL"): the expert GEMM's work-group shapes, 16384-token prompt"
  if [ ! -e "$MOE_MODEL" ]; then echo "skipped (not found): $MOE_MODEL"
  else
    bench "$MOE_MODEL" "" 512
    for m in "defaults (1 x 1)|" "MOECG=2|MOECG=2" "MOECG=4|MOECG=4" "MOEWY=2|MOEWY=2" "MOEWY=4|MOEWY=4" "MOEWY=8|MOEWY=8" \
             "MOECG=2 MOEWY=4|MOECG=2 MOEWY=4" "MOECG=4 MOEWY=4|MOECG=4 MOEWY=4" "H8|H8=on"; do
      bench_line "${m%%|*}" "$MOE_MODEL" "${m#*|}" 16384
    done
    bench_line "defaults again (drift)" "$MOE_MODEL" "" 16384
  fi
fi

echo
if [ -n "$MOE_MODEL" ]; then e_moe=", the 35B at 16384"; else e_moe=""; fi
echo "=== E. several GPUs (--gpus), the 27B at 4096 and 16384 tokens$e_moe"
NGPU="$(probe_gpus "${UNSET[@]}" | grep -c .)"  # (as the .ps1: without the variables of the script)
for g in "${GPUS[@]}"; do
  if [ "$g" -gt "$NGPU" ]; then echo "skipped --gpus $g ($NGPU GPUs found)"; continue; fi
  for n in "default" "best with H8"; do
    for pp in 4096 16384; do
      bench_line "$(printf -- '--gpus %s 27B %-14s %5s' "$g" "$n" "$pp")" "$MODEL" "${SETS[$n]}" "$pp" 0 "$g"
    done
  done
  if [ -n "$MOE_MODEL" ] && [ -e "$MOE_MODEL" ]; then
    bench_line "$(printf -- '--gpus %s 35B default        16384' "$g")" "$MOE_MODEL" "" 16384 0 "$g"
    bench_line "$(printf -- '--gpus %s 35B MOEWY=4        16384' "$g")" "$MOE_MODEL" "MOEWY=4" 16384 0 "$g"
  fi
done
