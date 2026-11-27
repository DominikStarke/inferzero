# GEMM inventory: the contract the shared GEMM layer must meet

Phase 0 of the move to one shared GEMM layer ([docs/design-shared-gemm.md](design-shared-gemm.md)), approved by the
user on 2026-11-27. The direction:
1. collect the fastest GEMM and GEMV forms, their scaling forms included;
2. the new route keeps every guard, bug fix and proven try / catch of the old paths;
3. Inkling 2 Mini is the first model on it, then its optimization;
4. then every model, the encoders, the tensor and the layer split, one at a time, the old path kept as a fallback
   until each is proven.

**The rule:** every row has its source and its proof, or says **"unproven"** and why. Nothing is left blank.

## A. Every GEMM / GEMV form in use (fused launches included: a fused form names what it fuses, see K)

Generated from the engine's own launch log (`ARC_GEMM_PLAN=1`): one line per distinct (weight role, weight type, row
class: one token / 2-8 rows / prompt chunk, host or resident) giving the kernel, its `-D` options, grid, local size and
caller. The source is one short run per model: a prompt of more than one chunk, 8 generated tokens, and the 2-8-row
passes. The laptop shows only Xe-LPG. Other GPU kinds' rows come from the contributor kit (J), and stay "unproven:
plan not seen on this kind" until then.

| Role | Type | Rows | Host / resident | GPU kind | Kernel and `-D` | Grid / local | Chosen at | Fastest measured form (source) | Correctness proof |
|---|---|---|---|---|---|---|---|---|---|

**Forms missing, a faster one measured elsewhere** (checked against the code 2026-11-27):

| Shape | Ours | Faster, measured | Source |
|---|---|---|---|
| MoE experts in prompt chunks | Inkling `moe_down` 73.8 ms a visit (125H) | oneDNN batched 6.9 ms | plan-2026-11-30.md section 1 |
| the 35B's experts | 23.4 / 13.1 ms | oneDNN 13.5 / 7.9 (3.5x with decode_w8) | plan section 1 |
| thin GEMM, Inkling's controller 512 x 64 x 2304 | 10.2 ms | oneDNN 0.06 ms | plan section 1; `fix/ink-ctrl-gemm` unmerged |
| NVFP4 experts (unsloth's Inkling) | B70 `moe_down` 2,275 ms | the same model as IQ4_NL: 238 ms | hibagus, #92 (2026-11-24) |
| dense int8 prompt GEMM, DP4a iGPU | H8R 8.7 TOPS, W8 8.1 | oneDNN 12.1 TOPS (per-row scales; its kernel at that shape: unknown) | before-after.md, 2026-11-26 |
| one-token Q4_K GEMV, 64-EU Arrow Lake-S | 25-28 GB/s | oneDNN u4 GEMV 60 GB/s | jac2424, #97 (2026-11-26) |
| one-token Q5_K GEMV, 125H | 58.6-61 GB/s (gemv_bench 69-70) | Q6_K's path 67-68.5 (bench 76.4) | before-after.md, 2026-11-26 |
| 8-lane XMX prompt GEMM, A770 | 27B 4K ~1,773 tok/s | target 2,200 | QUEUE.md; before-after.md |
| f16 GEMM on DP4a iGPUs | none fast | oneDNN f16 3.6 TOPS at 512 x 2560 -> 9216 (125H, 2026-11-26) | the Researcher's run; not yet in before-after |
| the 27B on the fly, 125H | gate / up / down 1.55-1.6x, qkv 1.56x behind | oneDNN 3.14.2 | plan section 1 |
| Flash Next on the 125H | dense 1.7-2.1x, experts 5.7-7x behind | oneDNN 3.14.2 | plan section 1 |
| A770 int8 GEMM | ~65 % of peak | oneDNN 72-79 % | QUEUE.md |
| B60 under Windows | 157.6 TOPS | oneDNN 170-183 | QUEUE.md |
| B70 16-lane XMX GEMM | xw_peak's best 232 TOPS (`L128` not measured in the model) | oneDNN 233-237 | QUEUE.md; before-after.md |
| 8-lane XMX expert tiles | 1.9x / 3.2x slower than our own DP4a | | QUEUE.md |
| Arc 140T prompt GEMM | ~33 % of its XMX int8 peak | | CONTRIBUTING.md |

## B. Every guard and invariant

Two kinds of row:
- **guard:** a check that refuses or falls back, proven by a test that triggers it;
- **invariant:** a rule the code must keep, proven by a hash that shows it holds.

The code side is read from the choice functions' conditions (matmul1, gemv_gu, gemv_rows, h8_takes /
h8_igpu_rows_ok, layer_moe's branch order, p140_take, the xe2 defaults per kind), not their comments. The history side
comes from CHANGELOG.md, docs/before-after.md, docs/audit-*.md and BUGS.md. The two are merged with each row's source.

| Kind | What it protects | Where (file:line) | Added by (commit) | Proof (test or hash) | Source |
|---|---|---|---|---|---|

## C. The golden test set

Per model, logits hashes (`runz --dump-logits`) with the shared layer on and off (`ARC_GEMM_SHARED`), equal bit for
bit. Each set is recorded with its build and driver version, since the f32 parts can move with a driver's compiler.
1. a prompt longer than one chunk, then 16 generated tokens;
2. a short prompt (under 8 tokens): the 2-8-row paths;
3. drafting on (MTP / DFlash2 / DSpark) where the model has it;
4. two slots batched against each alone;
5. the memory forms: host-memory weights forced, resident W8 against decoded per chunk;
6. the splits under `ARC_TP_SIM`: `--gpus 2` and `3` (its own hash: not bit-identical to one GPU) and `--split layer`
   (bit-identical to one GPU);
7. the switches that change kernels (`ARC_H8=0`, `ARC_BATCH`, `ARC_GEMV_GU`, ...), one hash each.

The DPAS emulations (`ARC_MOE_RG_XEMU`, `ARC_GDN_XEMU`, the attention one) cover moe_rg, DeltaNet and attention on
the laptop. `gemm_x8`, `gemm_xw` and `gemm_x16` stay contributor-only.

## D. Routes

One row per model x feature x setup: **works** (its proof), **refused** (the refusal's file:line), or **unknown**.
Filled from the Developer's check of 2026-11-27 against main 01617e2.

| Model | Feature / setup | State | Where / proof |
|---|---|---|---|

## E. Speed baseline

Same-release pairs only: the release before a migration step and the step itself, on the same card. Old issue
numbers are context. Models: the 4B, the 27B IQ3_S, the 35B-A3B, Flash Next, Gemma 4 12B and E4B, Inkling 2 Mini.
Lengths: pp512, pp4096, tg128 after 512; pp16384 for the 35B and Inkling. Three A B B A runs per model and length.
**A step is blocked** when its median is below the baseline by more than max(2 %, 2 x that model's own spread), on two
A B B A rounds.

| Model | Card | Release / driver | pp512 | pp4096 | tg128 | pp16384 | Spread |
|---|---|---|---|---|---|---|---|

## F. Memory plan baseline

`--ctx auto`'s choice and its memory line per model. On the laptop at `ARC_GPU_MEM_GB` = 6 / 8 / 12 / 16; on discrete
cards from the kit. **Proof:** the dry count against the real allocation after load, on the same run. Any gap is an
uncounted buffer the layer must add (first suspects: the JIT decode areas, H8's areas, the resident W8 partial share).

| Model | Card / cap | Context chosen | Memory line | Dry count vs real | Gap |
|---|---|---|---|---|---|

## G. Launches and start-up

Launches per generated token (`ARC_LAUNCH_STATS=1`) and per 512-row chunk (`--prof`), the warm-up's time, and the
total start, after an update (a cold kernel cache). Before a cold-cache run with `ARC_DATA_DIR`, check the real hang
marker by hand: that variable moves the marker too. The per-model budget file `tests/launch_budget.txt` is new.

| Model | Card | Launches a token | Launches a 512 chunk | Warm-up (cold) | Start (cold) |
|---|---|---|---|---|---|

## H. Switches that steer GEMM / GEMV / expert choices

349 distinct `getenv("ARC_...")` names (2026-11-27). Each is listed with the function that reads it and the member it
sets. It's classified by three existing lists: `scripts/options.tsv` (user-facing), `gpu_sweep`'s settings (tuned by
contributors), and xe2-tuning.md's defaults per kind. A switch in none of the three is test-only or forgotten: a
candidate to retire, each checked. This is the seed of the switch registry (plan section 6).

| Switch | Read in | Sets | Steers | Default where | Class |
|---|---|---|---|---|---|

## I. Dead paths

A form is a deletion candidate only if all three hold:
- it never appears in any GPU kind's plan log (A);
- it isn't one of B's guards (a fallback, a low-memory form, a failure path);
- no `options.tsv` row or `gpu_sweep` setting reaches it.

It's deleted **during** migration, when its route reaches "old path deleted", with C's hashes equal. Not before.

| Form | Never chosen on | Not a guard because | Not reachable because | Deleted in |
|---|---|---|---|---|

## J. The contributor kit

`bench/contrib_kit.ps1` / `.sh`, through `gpu_guard`, 10-20 minutes, one file to attach:
- the device, driver and release;
- A's plan log;
- E's pp512 / pp4096 / tg128 x 2;
- G's launch counts;
- F's `--ctx auto` memory line;

for whichever of E's models the contributor has. The first ask goes to skipzx-lab (A770, B60), hibagus (B70),
manoreken2 (4 x B60) and LukaszBrzyszkiewicz (B70, A380) once it's merged.

## K. Fusions: what is fused today, and where fused GEMMs go

Today's fused forms are where most of the per-GPU tuning lives, so the shared layer must carry each one as a fusion
rule, not as a separate kernel next to it. One row per fused form in use:
- **fused GEMMs:**
  - gate + up in one launch: `gemv_gu` / `-DGU2`, `ARC_GEMM_XW_GU`;
  - stacked projections (qkv + z, Inkling's q / rel / gate / k / v, the MoE router + shared gate / up);
  - SwiGLU in the up GEMM's epilogue (`ARC_H8_SWI`);
  - the MoE experts' gate / up with situ;
  - `ink_ctrl_fused`;
- **fused prologues:** the norm and H8 rotation in one kernel (`rmsnorm_h8r`, `ARC_H8_NORM`); one H8 activation
  shared by several GEMMs (`ARC_H8_REUSE`); `attn_gate_h8r`; the quantization kernels that write H8's float copy
  (`-DH8F`);
- **fused epilogues and other steps:** the DeltaNet conv / gates fused into the recurrence (`-DCONV_UPD`,
  `-DFUSE_GATES`); the outputs rotated for H8 by the kernels that make them (`ARC_H8_OUT`); the KV store and norms in
  attention's preparation; the multi-GPU sums' fused forms (`tp.cl`).

| Fused form | What it replaces (the unfused chain) | GPU kinds where it's on | Where it's chosen | Bit-identical to the chain? (proof) | Spill status per family |
|---|---|---|---|---|---|

**Where fused GEMMs go in the shared layer** (design section 3.12):
- A fused GEMM is **not a kernel of its own**. It's the shared GEMM stage plus a prologue (`Norm`, `Quant`, the H8
  rotation) and an epilogue (`Act`, `Add`, `Combine`, `Softcap`, quantization for the next GEMM), generated from the
  layer's chain.
- **Which neighbouring steps join is a fusion rule** in each GPU kind's catalog, with its measurement, not code spread
  through `engine_h8.h` and `xe2/`.
- **Every fused form must equal its unfused chain bit for bit**, checked by `gemm_check` per rule. So a fusion choice
  changes only speed, never output.
- **New kernels are only fusion kernels** (the user, 2026-11-26, #92): a model or a change adds a fusion rule, or a
  step no kernel covers, never another GEMM.
- **Grouped products** (one launch over weights side by side, the design's "way 1") and **horizontal fusion**
  (independent small steps in one launch, "way 6") are fusion rules too.

**Per model** (the user, 2026-11-27: "for each models"). For each model:
- which of the fused forms above it takes today, per GPU kind, from its plan log (A);
- its launches a token now (G);
- its floor, the launches left when every legal fusion is taken (design section 3.7);
- its planned chain. The floors and chains are from design section 3.7, and the "now" counts are to be measured in G.

| Model | Fused forms today (from A) | Launches a token now | Floor | Planned chain, per layer |
|---|---|---|---|---|
| Qwen3.5 4B / 9B (dense, 3 DeltaNet : 1 attention) | to fill | 4B ~426 (125H `--prof-gen`, 2026-11-26) | ~5 a layer: 4B ~160 | DeltaNet: [norm -> qkvz + ab, conv / state / gates in the epilogue] · [recurrence] · [gated norm -> ssm_out + residual] · [norm -> gate / up, SwiGLU] · [down + residual]; attention: [norm -> q / k / v, q / k norm, RoPE, KV store] · [attention] · [sigmoid gate -> o + residual] · the FFN's 2 |
| Qwen3.8 27B | to fill | ~710 | **321** | as the 4B, 48 DeltaNet + 16 attention layers |
| Qwen3.6 35B-A3B (MoE) | to fill | ~620 | **241** | the mixer's 3, then [norm -> router + shared gate / up] · [top-k -> experts' gate / up, the shared down grouped in] · [experts' down]; the combine in the next layer's prologue |
| Flash Next | to fill | ~1,050 | **~483** | the hc mixer's 2 global steps per block, then the mixer's or the MoE's 3; `hc_combine` in the block's last epilogue |
| DeepSeek-V4.1 | to fill | ~1,900 | **~415** | the design's 10-13 steps a layer (mix, Sinkhorn + norm -> q_a / kv grouped, attention, out_a in one launch, out_b + expand, the MoE's 3) |
| Gemma 4 dense (31B) | to fill | ~840 | **301** | as Qwen3 dense; the post-attention norm + residual + FFN norm in gate / up's prologue |
| Gemma 4 26B MoE | to fill | to measure | 6 a layer | qkv · attention · o · [dense FFN gate / up + router] · [experts' gate / up + dense down] · [experts' down] |
| Inkling 2 Mini | to fill | **1,356** | **385** | every visit: [mix + norm -> q / gate / rel / k / v / sink / kcp / vcp grouped, K / V convs in the epilogue] · [attention] · [o + post-conv + residual] · [FFN mix + norm -> router] · [top-k -> experts' gate / up, situ] · [experts' down]; later passes add the controllers' 4 |
| the encoders (vision, audio, speech) | to fill | to measure | to work out | per encoder, once their GEMMs move to the layer |

The gap between "now" and the floor is each model's fusion work. Inkling's is the largest (3.5x), which is one reason it
goes first.

Phase 0 records each of today's fused forms in this table with its proof. Phase 1 carries each one as a rule over the
same kernels, bit-identical. The new fusions (the design's ways 1-8: fewer launches toward each model's floor) come
after a model is on the layer, Inkling first.

## Exit rule for Phase 0

1. every row of A-K has a source and a proof, or says "unproven";
2. the silent refusals are fixed (Inkling on `--split layer`, Gemma 4's `--dflash`), with the five stale comments;
3. C runs green twice in a row on the same build and driver, and E's baseline is taken on that build;
4. F shows no uncounted buffer, or lists the ones the layer must count;
5. only the discrete kinds' A / E / F / G rows may stay "unproven" until the kit comes back. Phase 1 then starts with
   what the laptop can check: the host layer over unchanged kernels (the design's section 6.1);
6. the user signs off.

Phase 0's code, smallest first, each its own branch with C's hashes equal:
1. the silent refusals and comments;
2. `ARC_GEMM_PLAN=1`;
3. C's harness;
4. the kit;
5. optionally, a cache-only override and a compile counter.
