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

How to read the log:
- **Memory column:** a GEMV line says "raw" or "host raw", the rows it reads. A GEMM line says where its W8 comes from:
  "w8 resident", "w8 jit" or "host". "-" is a step with no weight.
- **One summary per model:** the "gemm-plan summary (freed)" line that runz prints at its end. C's harness runs runz, so
  that's the line it collects.
- **serverz:** prints "gemm-plan summary (after the warm-up)" after start-up. That holds only the routes the warm-up
  took, so it's partial. A stopped server prints no final summary: it ends without its destructors. A model switch
  prints the old model's "(freed)" summary.
- **A split model:** each shard logs its own lines and summary ("shard r"). serverz's warm-up summary is shard 0's or
  the last stage's.
- **Not in the log yet:** Whisper's own launcher, the NPU and `engine_tp.h`'s one direct launch. They bypass
  `Engine::run`, so they're listed by hand until their migration turns.
- **Steps that aren't GEMMs** (norm, quant, swiglu, rot) give no model-file line ("from") until the C-harness commit.

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

### The code side (the Developer, 2026-11-28, read at main 6fb85d7)

Read from the code's conditions, not the comments. The findings marked "checked" were re-read by hand, and the Researcher re-read findings 1 and 4. "golden" is tests\gemm_golden.ps1 and its kind; "run_tests" is tests\run_tests.ps1. "(no comment)" means no comment says why. A row whose test is "none" is unproven: Phase 1 must not move it without adding one, or keep it unchanged and prove it by the golden hashes.

#### Findings (not changed: Phase 0 is inventory; each goes to a fix after Phase 0 or with its migration turn)

1. **XMX verdict cache per device only (checked).** engine.h:489-494 `xmx_by_dev[cl_.device_id]`; the speed check is skipped when `p140_` is on (engine_setup.h:857). In one server process on an Arc 140T, the first model loaded decides XMX for every later model on that GPU, its speed check included or not.
2. **"512-byte aligned" views are a comment only.** engine_stream.h:563 says so (rgu_rows rounded to 128 rows, engine_sessions.h:858); nothing checks it, and zrt's clCreateSubBuffer checks bounds only (clrt.h:1044-1052). The qz and Inkling stack views have no alignment rule.
3. **Staged W8 written without a size check.** matmul1 / h8_prepare decode into `stage_q_[role]` relying on `stage_need_*` / `ensure_stage` having covered every weight (engine.h:1722-1726; engine_alloc.h:207-209; engine_sessions.h:862-870).
4. **The spill warning is said once per name + options, not per GPU (checked).** clrt.h:1762-1765: the same kernel spilling on a second GPU kind of a layer split is not reported again.
5. **CL_CHECK exits with _Exit(1) (checked)**, no destructors, no marker (clrt.h:89-97): a CL error is neither a hang (3) nor a lost device (4).
6. engine.h:1729 (`stage_packed_` reset) is mis-indented under the `if` above it (cosmetic; the logic is what the braces say).

#### B.1 zero/engine.h

| kind | protects | file:line | test |
|---|---|---|---|
| guard | head softcap epilogue only for the head, softcap > 0, not `q2_nocap_` | 1309 | run_tests 4c |
| invariant | two-pass: the head's softcap applies to the sum of both passes | 1413-1414 | run_tests 4c |
| guard | fused two-pass GEMV (`q2_fuse_`) only at T <= 4 | 1416 | golden short / mtp |
| guard | no second pass when `!q2mm_ \|\| q2_off_`; the drafter forces one pass (engine_drafters.h:71-72) | 1420 | run_tests 4c |
| guard | two-pass with a fused epilogue (acc > 1) throws | 1421 | none |
| invariant | second pass reads aq_/asc_/absum_ at q2 offsets, skipped when fused (buffers x2: engine_alloc.h:10, 42, 44) | 1422-1430, 1395-1401 | run_tests 4c |
| guard | two-pass head over > 8 rows throws | 1434 | none |
| invariant | `q2mm_` decided once (q2_init): Gemma but E2B/E4B, Flash Next with ARC_FN_Q2, Qwen3 unless ARC_Q3_Q2=0 | 1387-1392 | run_tests 4c |
| invariant | `gemv_v_ok` type list (IQ3_S excluded: "measured" only) | 1320-1321 | none |
| guard | ARC_GVROWS outside 1-8 ignored; forced count only if <= N | 1330-1331, 1338 | none |
| invariant | Xe-HPC: 1 row unless ARC_GVROWS | 1335 | none |
| invariant | gemv_v rows: N > 4096 or K >= 8192 -> 4, but IQ4_XS / Q6_K / Q4_0 | 1343-1344 | golden base |
| guard | tiled IQ3_S never on gemv_v | 1451 | none |
| guard | K4 rule: !q2mm_, not the head; default Q5_K, N >= 8192, N >= 2K, iGPU, no XMX | 1461-1464 | none |
| guard | ARC_GV_SMALL: iGPU, no XMX, Xe-LPG <= 640 threads (or SMALL_ANY) | 1474-1476 | none |
| guard | GV_SMALL=2 widening needs gv_k4 != 0, !q2mm_, not the head | 1477-1478 | none |
| invariant | Alchemist (sg8 + XMX + discrete + unbatched): vk types 1 row | 1488 | none |
| guard | ARC_GEMV_ROWS / gv_small overrides only for !vk and N <= 65536 | 1492-1493 | none |
| invariant | default rows 1 / 2 / 4 / 2 by N, then xe2_gemv_rows | 1495 | golden base / short |
| invariant | host_gemv_mb: 0 on an iGPU, 4 MB discrete, per call | 1499-1502 | golden host |
| guard | gemv_gu: env, T == 1, !q2mm_, both raw, same type/N/K, not the head | 1518-1520 | golden fusion GEMV_GU=1 |
| guard | gemv_gu: iq3t must match | 1521 | none |
| guard | gemv_gu: host weights refused | 1522 | golden host |
| guard | gemv_gu: N % grows refused | 1528 | none |
| invariant | gemv_gu kernel / rows cached on g at first use, never re-evaluated | 1523-1534 | none |
| guard | gemv split-K: refused when off, ARC_GEMV_ROWS set, no thread count, N > 65536, MoE | 1556-1557 | none |
| guard | SPLITK_FORCE: power of two 2-8, <= steps | 1573 | none |
| guard | host-CPU GEMV (ARC_HOST_CPU) only with act32 | 1593 | none |
| invariant | one-token kernel cached per weight (g1k[q2_fuse_], g1rows, g1vk, g1srows, g1split) | 1602-1626 | none |
| guard | GV_PROBE never on vk types or the head | 1613 | none |
| guard | split-K GEMV excluded for fused two-pass, iq3t, probe, head, GEMV_DEEP | 1617-1619 | none |
| guard | fused two-pass GEMV with an epilogue throws | 1631 | none |
| guard | host pieces only if host, not fused, split 1, host_mb > 0, rows > host_mb; whole row groups | 1646-1648 | golden host |
| invariant | GEMV grid / WG 256 for gemv_v or split | 1660-1661 | none |
| invariant | 2-8 rows: vk adds IQ3_S (untiled) and Qwen3 Q8_0 fused | 1682 | golden mtp |
| invariant | 2-8 rows: Q6_K 1, T < 6 -> 4 / 2, then xe2_gemv_mt_rows | 1689-1697 | golden mtp |
| guard | 2-8 rows two-pass fuse only when q2_fuse_ && vk && a <= 1 | 1696 | run_tests 4c |
| guard | GEMM: ldc % 4 or c_off % 4 throws | 1714-1716 | none |
| guard | h8_takes tested first on the GEMM path | 1719 | golden option H8=0 |
| invariant | staged W8 re-decoded unless owner and not the H8 form | 1725-1730 | golden base |
| guard | x16 GEMM tried first, falls back when it returns false | 1732 | none |
| invariant | 140T preset tile choice (32 tokens / cg 2; 16 / cg 1; else xe2_cg_for) | 1735-1737, 1743 | xmx_check mode 1 |
| guard | GEMM shape Mp % tm, N % 16, K % 256 throws | 1745-1747 | none |
| invariant | ARC_GEMM_SPLIT: 0-4096, default 17; only a != 2 and K > 8192; pieces 2+ accumulate | 1751, 1754, 1759 | none |
| invariant | wgy doubles while it divides Mp / tm, up to ARC_GEMM_WGY 8 (no comment) | 1756 | none |
| guard | decode_w8: N % 16 throws | 1277 | none |
| invariant | decode_w8 WG halves until it divides N; vb needs lx >= 64 and Q8_0 / Q4_0 (no comment on 64) | 1286-1287 | none |
| guard | decode_w8_sl: no requant, nsb <= 16, N % 16, <= 32 KB, not IQ1_M / IQ1_S (no comment on IQ1 / 32 KB) | 1291-1292 | none |
| guard | short-prompt 8-row passes only when !keep_rows, !w8_resident, !stream_, n <= few, max_chunk >= 64 | 752-753 | golden short |
| invariant | Mp: >= 128 rounded to 128 (capped at max_chunk), else to 16 | 758, 802-805 | none |

#### B.2 zero/engine_h8.h

| kind | protects | file:line | test |
|---|---|---|---|
| guard | h8_init: (ARC_H8 \|\| p140_ \|\| h8_auto_ \|\| h8_dp4a_) && !q2mm_ && !npu | 84 | golden option |
| guard | f32 source only when H8R, not ARC_H8_INT, !hc, !gemma | 77, 85-88 | none |
| invariant | h8_block: largest power of two <= cap dividing K, >= 256 | 108-116 | none |
| guard | h8_ok: role 0-10, qtype Q8_0 / 100-103, K % 256, N % 16; H8R K / hb <= 32, K <= 130000 | 117-121 | none |
| guard | h8_takes: T > 8, !q2mm_, h8_ok, 140T take rule, resident or h8_igpu_rows_ok | 265-267 | golden base / option |
| invariant (A1) | resident W8 always takes H8 (rotated in place); the plain GEMM never reads a rotated copy | 266, 276-278 | h8_e2e_check |
| guard (spec_) | h8_dp4a_ without ARC_H8: not in verification passes for JIT weights | 285 | golden mtp |
| guard | 16-lane-XMX iGPU: H8 from Mp >= ARC_H8_MIN (256) | 286-288 | none |
| guard | h8_swi_ok: env, H8R, f32 copy, h8_takes(down) | 290-293 | golden fusion H8_SWI=0 |
| guard | h8_norm_env defaults: p140_ \|\| discrete non-Xe-HPC \|\| (h8_dp4a_ && hb <= 512) | 319-323 | golden fusion H8_NORM=0 |
| guard | rmsnorm_h8 falls back on any of 11 conditions (no share scope, wide rows, K / hb > 16, row > 60 KB ...) | 331-332 | golden fusion |
| invariant | h8_pre_live_ cleared by rmsnorm_quant / ~H8Share / any rotation; used only if Mp and K match | 255, 347, 464-465 | golden fusion H8_NORM=0 |
| guard | h8_out_env default only for discrete sg8 (Alchemist) | 358-361 | none |
| guard | h8_out_ok: rmsnorm_h8's row limits | 362-366 | none |
| invariant | rotation reuse only if K, M, act32 and the source match | 467-470 | golden fusion H8_REUSE=0 |
| guard | H8 GEMM without h8_act in its call throws (h8_input_check) | 456-459, 589, 610 | none |
| guard | h8_act with an f32 source without H8R throws | 485 | none |
| guard | h8_e2e_check A/B/A on layer 0's qkv; skipped for MoE / hc / gemma / ds41 / inkling, ARC_H8_E2E=0, dry | 393-455 (engine.h:605) | itself |
| invariant | e2e Restore keeps resident rotated entries | 427-441 | none |
| guard | h8_dec_ok: one-pass decode+rotate type list, hb >= 512, N % 16 (no comment on IQ1) | 193-202 | none |
| guard | h8_prepare: a resident weight asked for packed order throws | 498 | none |
| invariant (A1) | resident copy rotated once; p140_res_ keeps them unrotated; w8_part_auto erases the mark on a fresh decode | 497-506; engine_load.h:260 | h8_e2e_check (#80) |
| invariant | staged H8 re-decoded unless owner, H8 flag and packed order match | 514 | golden base |
| guard | H8SB scales buffer grown when small | 542-544 | none |
| invariant | h8_splits: 17 superblocks iGPU / 140T, 34 discrete; never with a == 2 (decides bits) | 567-572 | none |
| guard | ARC_H8_Q2 with fused SwiGLU throws | 579 | none |
| guard | H8R int32: a piece over 133,000 values throws | 593-598 | none |
| guard | packed operands not taken by gemm_xw8: throws | 617-618 | xw_check |
| guard | H8 GEMM shape throws | 625-626 | none |

#### B.3 zero/xe2/engine_xe2.h and the setup (engine.h)

| kind | protects | file:line | test |
|---|---|---|---|
| guard | h8_dp4a_wanted: not with ARC_H8, XMX, discrete, non-Xe-LPG, moe/gemma/hc/qwen3/ds41/inkling, !h8_row; resident yes, else gate Q4_K / Q5_K | xe2:60-68 | none |
| guard | h8_auto_wanted: XMX, 16 lanes, not moe/gemma/hc/qwen3 | xe2:69-72 | golden option |
| guard | p140_wanted: XMX (or test), sg8, resident rule, not moe/gemma/hc/qwen3, H8R | xe2:79-92 | xmx_check mode 1 |
| guard | p140_take: qtype 100-103, Mp >= 512 (768 res), Mp % 32, N % 32 (no comment on N % 32) | xe2:30-32 | xmx_check 140T |
| invariant | setup order q2_init -> p140_ -> h8_auto_ -> h8_dp4a_ -> h8_init -> p140_ &= h8_ | engine.h:455-465 | none |
| invariant | h8_cg_for: 4 groups with h8_auto_ on discrete when (N/16) % 4 == 0 | xe2:75-78 | xmx_check mode 1 |
| invariant | xehpc_on / xe2_g21 / t1280 / moe_grf: per engine | xe2:133-137, 150-162, 217-227 | none |
| invariant | xe2_cg_for only with xe2_ or ARC_GEMM_CG, (N/16) % cg | xe2:230-240 | xmx_check (cg <= 4) |
| invariant | xe2_gemv_rows / mt_rows / wg rules | xe2:309-352 | golden mtp (mt) |
| guard | 8-lane iGPU: XMX off unless FORCE / TRY / p140_ | engine.h:474-479 | none |
| guard | xmx_check mode 0 once per device_id; 2D retried without; sets xmx_ / slow. **Finding 1** | engine.h:487-531 | start-up |
| guard | xmx_check(1) for H8 keyed by device, qtype, cg, 140t, env, 2d; failure drops p140_ / h8_auto_ / h8_, keeps XMX | engine.h:536-551 | start-up |
| guard | p140_ / h8_auto_ without xmx_: off | engine.h:554-561 | none |
| invariant | H8 off after checks: h8f_off_ dropped, kernels rebuilt | engine.h:566-570 | none |
| guard | x16: wanted rule; x16_check once per device; shape refusals | xe2:823-847, 882, 911-1003 | x16_check |
| guard | xw_init / xw_check per key; fall back DPASW -> split barrier -> C_RND -> 2D -> off | engine_xw.h:1060-1198 | xw_check |
| guard | rgx_first_use: DP4a vs XMX copy byte compare per type key; differs or no room -> DP4a for that type, layer back to decode + GEMM | engine_moe_xw.h:78-127, 43-53; engine_moe.h:614-616 | itself |
| guard | rgx_kernel build failure -> DP4a (soft_fail) | engine_moe_xw.h:55-66 | none |

#### B.4 zero/engine_setup.h

| kind | protects | file:line | test |
|---|---|---|---|
| guard | gemm_self_check: Q8_0 64 x 1024 vs CPU; rel > 1e-3 -> warning + driver_warning, never a refusal; ARC_SELF_CHECK=0 skip, =2 fake | 612-694 (engine.h:472) | ARC_SELF_CHECK=2 |
| guard | xmx_check mode 0: qtypes x act32 x store / split / add / swiglu, bit for bit vs DP4a; MoE tiles | 735-780 | start-up |
| guard | MoE x8 failure on sg8: moe_x8_ok_ off, dense XMX kept | 781-813 | none |
| guard | mode 1 140T cases | 817-846 | start-up |
| guard | speed check: XMX must be 3 % faster on 512 x 17408 x 5120; skipped with FORCE / p140_ / TRY (**finding 1**) | 857-907 | start-up |
| guard | build exceptions -> DP4a; GpuWaitError rethrown | 909-916 | none |
| invariant | p140_res_ only with ARC_140T_RES, resident, not TP, discrete | 437-440 | none |

#### B.5 zero/engine_moe.h (layer_moe, line 8)

| kind | protects | file:line | test |
|---|---|---|---|
| invariant | q2on / q2f / q2g (fused per tile, two-pass on the GEMM) | 99-102 | run_tests 4c 26B |
| invariant | tile size rules; ARC_MOE_TM_SMALL in {0,1,2,4,8,16} | 87-94, 106 | none |
| invariant | max_tiles grid, extra WGs exit | 167 | none |
| guard | gemm_ok: moe_group_, Fp == F, moe_gemm_layer_ok, !tp_moe, !ds41 | 196 | run_tests 35B |
| guard | rg_ok: moe_std, tm 16, T >= rg_min or q2g, F % 64, layer ok, !ds41 (no comment on tm / F % 64 here) | 197 | run_tests 35B |
| invariant | gate/up order: moe_rg -> moe_gemm -> per tile gateup_q -> GSQ legacy | 212-219, 447-458 | run_tests 35B; golden 35B |
| invariant | down order: moe_rg -> moe_gemm (tm 16, T >= 1152) -> per tile down_q -> GSQ legacy (no comment on tm 16) | 215-216, 470-481 | same |
| guard | moe_gemm_part_ok: W8 type, not Q8_0 / Q4_0, variant < 2 | 527-533 | none |
| guard | moe_gemm_pad_on: discrete 16-lane, padded F, not gemma | 544-554 | none |
| guard | moe_rg_type list | 562-570 | none |
| guard | XCPU path: xc_, T == 1, !use_gemm, moe_std, xsplit, !q2g, (!tp_moe \|\| xsplit_tp) | 337 | none |
| guard | xsplit_tp off unless ARC_XSPLIT_TP=1 (F30) | 523-526 | none |
| guard | xcpu_layer_ok: discrete 16-lane default, moe_std, not gemma / inkling | xe2/engine_xcpu.h:50-55, 72-79 | none |
| invariant | xgpu: GPU tiles skipped when the CPU took every expert | 416, 452, 460, 475 | none |
| invariant | moe_rg two-pass then moe_gelu_mul | 597-629 | run_tests 4c 26B |
| invariant | moe_gemm sub-buffers per group (+kRawSlack, clamped) | 640-648 | none |

#### B.6 Layout contracts

| kind | protects | file:line | test |
|---|---|---|---|
| invariant | kRawSlack = 1024 bytes after the last superblock (GEMV reads past a row) | engine_load.h:98, 134, 178 | none |
| invariant | superblocks padded to 8 bytes | src/quants.h:857; engine_load.h:131 | test_quant_rows (CPU) |
| guard | IQ3_S tiling only K % 512, bp 112, nsb even, ARC_IQ3S_TILE; reported to w.iq3t | engine_load.h:132-133, 278 | none |
| invariant | host flag: up_host_ copied by every upload_padded caller | engine_load.h:137-142, 279; engine_load_gemma.h:142, 158, 254, 265, 270; engine_stream.h:574; engine_drafters.h:751; engine_inkling.h:273 | golden host |
| invariant | views copy their stack's iq3t / host by hand | engine_load.h:450-455; engine_stream.h:583; engine_inkling.h:290-291 | golden host |
| invariant | view size N * sbr + kRawSlack | engine_load.h:439; engine_stream.h:588; engine_inkling.h:277 | none |
| invariant | "512-byte aligned" views: comment only (**finding 2**) | engine_stream.h:563 | none |
| invariant | w8_bytes_q = N*K, w8_bytes_h = N/8 * K/256 * 320 | src/quants.h:832-833 | none |
| invariant | stage_need_q_/h_ max per role; ensure_stage for later weights; no size check at the write (**finding 3**) | engine_load.h:285-286, 311-316; engine_stream.h:584-585; engine_alloc.h:207-209; engine_sessions.h:862-870; engine.h:1722-1726 | none |
| invariant | H8SB scales sized stage_need_h_ / 8 + 64 | engine_h8.h:135 | none |

#### B.7 Process-wide statics tied to a device

None: no `static const` initializer calls zrt::integrated(), xe_lpg(), cl_.xmx or a device id (single-line and multi-line static lambdas in zero/*.h, zero/xe2/*.h, zero/mgpu/*.h). Caches keyed by device: xmx_by_dev (engine.h:488-489, **finding 1**), h8_known (537-538), x16 known (xe2:834-835), the xw maps (engine_xw.h:1060-1079), first_for_device (engine.h:1245-1246). Not keyed by device: the spill warning's set (clrt.h:1762-1765, **finding 4**), once_flag messages (xe2:154, engine_moe_xw.h:122).

#### B.8 Cache keys

| kind | protects | file:line | test |
|---|---|---|---|
| invariant | kernel binary = fnv1a(src, opts, device_id = CL name + driver); a damaged binary is recompiled | clrt.h:1643, 1412, 1708-1716 | none |
| invariant | warm marker = fnv1a(sources, model path, warm_key_extra (all ARC_*), split_key, mtp / dflash / drafts / max_chunk / device_id, "w3") | engine.h:994-1033 | none |
| invariant | split_key: "\|tp<n>\|fnv(tp.cl)" or "\|pp<n>" + each stage's range@gpu_id | tools/server_load.h:393-411 | none |

#### B.9 Failure paths

| kind | protects | file:line | test |
|---|---|---|---|
| guard | CL_CHECK: message + _Exit(1) (**finding 5**) | clrt.h:89-97 | none |
| guard | check_marker: a marker from this boot (< 12 h) -> _Exit(3); ARC_IGNORE_HANG warns | gpu_fail.h:161-196; clrt.h:1481 | gpu_guard |
| guard | gpu_fatal: Hang 3 / Lost 4 write the marker; ze_exit 0x70000001 -> 4, else 5 | gpu_fail.h:29, 218-241, 262-264 | gpu_guard |
| guard | ARC_GPU_WAIT_S >= 5 s; a longer wait -> Hang | clrt.h:371-383, 409-411 | gpu_guard |
| guard | per-kernel wait (ARC_ZE_WATCH / SYNC_EACH) -> Hang | clrt.h:1165-1169, 1320-1329 | none |
| guard | ARC_GPU_FAULT_TEST=hang / lost | clrt.h:398-404; gpu_fail.h:268-276 | manual |
| guard | launch: g == 0 none; g > 2^32 refused; failed append -> ze_exit | clrt.h:1251-1255, 1308-1313 | none |
| invariant | run(..., nullptr): suggestion shrunk until it divides; a given lws that does not divide -> -54 | clrt.h:1260-1280 | none |
| guard | spill warning once per name + opts (**finding 4**) | clrt.h:1753-1770 | none |
| guard | cancel: between chunks; per layer only when !w8_resident; layer-major start and after each layer (TP barrier); never inside matmul or generation | engine.h:745-766, 806-862; engine_sessions.h:221-238 | none |
| guard | ARC_FAIL_PASS: the N-th verify / batched pass throws after writing its state | engine_slots.h:386-391; tools/server_load.h:431 | tests/fail_inject.ps1 |

Not tested by any dedicated test: every logic_error / runtime_error shape refusal in matmul1 and h8_matmul.

### The history side

To come (the Researcher): each row's commit and the bug or audit that added it, from CHANGELOG.md, docs/before-after.md, docs/audit-*.md and BUGS.md, merged into the rows above.

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

**The harness:** `tests/gemm_golden.ps1` (merged 2026-11-28, 5d63cc2) runs these kinds through `bench\gpu_guard.ps1`
and writes `results.tsv` to a temp folder. A row is one of:
- ok;
- rec (its own hash, recorded);
- DIFFERS (a path that must equal base doesn't);
- CHANGED (unlike `-Baseline`);
- SHORT (fewer than 16 tokens generated);
- FAIL / ERROR.

The shell's `ARC_*` / `INFERZERO_*` / `ZE_*` variables are cleared for the runs, except `ARC_DATA_DIR` and
`ZE_AFFINITY_MASK`. Until the shared layer exists, it compares fusion off / on and the paths that must equal base.
Phase 1 adds `ARC_GEMM_SHARED` on / off.

**The laptop's baseline** (125H Xe-LPG, driver 32.0.101.9175, main 204a3e6 + the harness; 54 runs, green twice in a
row, every hash equal between the two):

| Model | Base hash | Kinds run | Notes |
|---|---|---|---|
| Qwen3.5 4B UD-Q4_K_XL | 70906E39A9EE61BB | all | host, layer and the six fusion switches equal base; slots OK |
| Qwen3.5 9B UD-Q4_K_XL | 5E65E3BD605C2168 | all | the same |
| Qwen3.8 27B GSQ IQ3_S (MTP) | 5C199FD26C0DEF20 | all | the same. `ARC_H8=0` also gives the base hash: the 27B takes no H8 on the iGPU, so `ARC_H8_NORM` / `H8_REUSE` / `H8_SWI` check nothing there; `GEMV_GU`, `GDN_FUSE2` and `BATCH` still do |
| Qwen3.6 35B-A3B UD-IQ4_XS | EF37993715CC7D39 | no host / fusion / option (MoE) | layer equals base; slots OK |
| Inkling 2 Mini UD-IQ4_XS | 8C9B0CF29F635EFC | base, short | it refuses the splits and MTP, batches no slots, and is MoE |

A baseline holds only for its build and driver. After a driver update, record a new one before comparing.

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

**The laptop, no cap** (the Developer, 2026-11-28; 125H with 20.42 GiB of shared memory, driver 32.0.101.9175, main
6fb85d7 + fix/plan-memory). The real allocation is zrt's own sum of every `zeMemAllocDevice`, printed with
`ARC_GEMM_PLAN=1`'s summary ("gemm-plan memory"): serverz after its warm-up, runz at its end. GiB throughout.

| Model | `--ctx auto` plan (weights + buffers + context) | Context chosen | serverz loaded | After the warm-up: counted / allocated | runz (ctx 1,040) at its end: counted / allocated |
|---|---|---|---|---|---|
| Qwen3.5 4B | 6.61 + 0.09 + 4.13 = 10.83 | 262,144 | 10.84 | 10.849 / 10.849 | 6.771 / 6.771 |
| Qwen3.5 9B | 12.54 + 0.14 + 4.13 = 16.81 | 262,144 | 16.80 | 16.810 / 16.810 | 5.616 / 5.616 |
| Qwen3.8 27B GSQ IQ3_S | 10.94 + 1.69 + 5.53 = 18.16 | 163,840 | 18.16 | 18.159 / 18.159 | 11.728 / 11.728 |
| Qwen3.6 35B-A3B | 14.80 + 0.72 + 2.76 = 18.28 | 262,144 | 18.19 | 18.189 / 18.190 | 16.867 / 16.868 |
| Inkling 2 Mini | 13.39 + 0.29 + 6.16 = 19.84 | 188,416 | 19.14 | 19.138 / 19.139 | 15.264 / 15.265 |

- **No uncounted buffer on any of the five:** counted equals allocated within 0.001 GiB, after the warm-up's prompt too.
  H8's areas, the staging and the buffers made at the first prompt are all in `gpu_bytes`.
- **The plan against the load:** equal on the 4B, 9B and 27B. Two models plan high, which is safe but costs context:
  the 35B by 0.09 GiB, Inkling by 0.70 GiB. So Inkling's `--ctx auto` picks less context than fits.
- **The 27B's plan line says "left -0.59":** its parts (kept free 1.17 + snapshots 0.29 + encoders 1.39 + weights
  10.94 + buffers 1.69 + context 5.53 = 21.01) exceed the 20.42 GiB device, yet the load fits (18.16 real). The
  printed line and the fit check count the encoders or the kept-free room differently. To look at after Phase 0.
- **Still to do:** the same under `ARC_GPU_MEM_GB` = 6 / 8 / 12 / 16 on the laptop, and the discrete cards from the
  kit.

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

### The code side (the Developer, 2026-11-28, read at main 6fb85d7)

Each fused form in use: the unfused chain it replaces, its switch and default, where it's chosen, whether the laptop's plan logs saw it (A), and its proof from the golden set's fusion rows (C: each must equal the model's base hash). "Taken on the laptop" is the Xe-LPG kind only. Other kinds follow the code's defaults and stay unproven until the kit. Spill status per family comes from the kit's start lines (A).

#### Fused GEMMs

| fused form | replaces | switch (default) | chosen at | taken on the laptop | proof |
|---|---|---|---|---|---|
| gate+up GEMV (`gemv` -DGU2) | the gate and the up GEMVs, two launches | `ARC_GEMV_GU=1` (off: opt-in) | engine.h:1517 (gemv_gu's rules: B 1) | no (opt-in) | golden fusion ARC_GEMV_GU=1 = base on the 4B, 9B, 27B |
| gate+up+SwiGLU GEMM (`gemm_xw` -DXW_GU, role gemm_gateup) | the gate GEMM, the up GEMM and swiglu_quant | `ARC_GEMM_XW_GU` (the default per kind: discrete XMX) | engine_qwen.h:499; xe2/engine_xw.h:231 | no (no XMX) | unproven on the laptop: xw_check at start (B 3), the kit on XMX cards |
| qkv+z stacked GEMV / GEMM (`gemm_qkvz`) | the qkv and the z products, two launches | none: taken when attn_qkv and attn_gate share a type and the shapes fit | engine_load.h:405-455 (the stack), engine_qwen.h:22 | yes: 27B, 35B (the 4B / 9B UD files mix types: two products) | no off switch, so no A/B hash; exact by construction (the same rows and kernel over one buffer) |
| router+shared gate/up GEMV (`gemm_router_sh`, Layer::rgu) | the router, the shared expert's gate and up: three GEMVs | none: taken when the shared gate / up are Q8_0, E % 256, not w8_resident on a dense model | engine_stream.h:560-595, engine_moe.h:65 | yes: 35B | no off switch, no A/B hash; exact by construction |
| both int8 passes in one weight read (`gemv` -DQ2F) | the two-pass GEMV run twice | none: two-pass models (q2mm_: Gemma, Qwen3) at T <= 4 | engine.h:1415-1416, 1454, 1623 | no (no two-pass model on the laptop) | unproven here: needs a Gemma 4 / Qwen3 file; run_tests 4c covers Gemma's KL |
| expert decode + GEMM in registers (`moe_rg`) | decode_w8 of the experts, then the expert GEMM | none to turn off; ARC_MOE_RG_MIN / _MAX move its row range (320 / 1280) | engine_moe.h:188-197 (rg_ok), 212-219, 584 | yes: 35B, Inkling | no A/B in golden (another kernel path: exactness to be shown) |
| Inkling's controller gain + forget + apply (`ink_ctrl_fused`) | three steps, separate launches | `ARC_INK_FUSE=0` turns it off (on by default) | engine_inkling.h:727, 818-824 | yes: Inkling | **not in golden**: add ARC_INK_FUSE=0 to its fusion rows (Inkling has none today) |

#### Fused prologues (norm, quantization, rotation)

| fused form | replaces | switch (default) | chosen at | taken on the laptop | proof |
|---|---|---|---|---|---|
| norm+rot (`rmsnorm_h8r`) | rmsnorm_quant (with its f32 copy) then h8r_act's rotation | `ARC_H8_NORM` (p140_, discrete non-Xe-HPC, or h8_dp4a_ with hb <= 512) | engine_h8.h:319-332 | yes: 4B (9B no: hb 1024) | golden ARC_H8_NORM=0 = base on the 4B (taken); the 9B and 27B rows pass without taking it |
| one rotation shared by the GEMMs of one input | a rotation before each GEMM | `ARC_H8_REUSE` (on) | engine_h8.h:238-255, 467-470 | yes: 4B, 9B | golden ARC_H8_REUSE=0 = base on the 4B, 9B |
| swiglu+rot (the down GEMM's rotation reads gate / up) | swiglu_quant, then the rotation of its output | `ARC_H8_SWI` (on where H8R has its f32 copy) | engine_h8.h:290-293, engine_qwen.h:511 | yes: 4B, 9B | golden ARC_H8_SWI=0 = base on the 4B, 9B |
| attn gate+rot / gated norm+rot (`attn_gate_h8r`, `gated_norm_h8r`) | the gate / gated norm quant, then the rotation for o / ssm_out | `ARC_H8_OUT` (default: discrete 8-lane, Alchemist) | engine_h8.h:347-375, engine_qwen.h:105, 470 | no | unproven on the laptop (not taken by default; ARC_H8_OUT=1 not in golden) |
| quant+f32 copy (`-DH8F`) | a separate f32 copy of the rows H8R rotates from | none: on whenever H8R takes its f32 source (h8_float: not ARC_H8_INT, not hc / gemma) | engine.h:562, engine_h8.h:77, 87, 475 | yes: 4B, 9B | inside every H8 run: golden base of the 4B / 9B; no A/B of its own |
| norm+quant (`rmsnorm_quant`), swiglu+quant, gated norm+quant, attn gate+quant | the norm / activation, then a quantization kernel | none: the only form there is | engine.h:1768-1780; engine_qwen.h:114, 480, 520 | yes: every model | no unfused form exists to compare; they are the reference chain |
| add+norm(+quant) (`add_norm_quant`, `add_rmsnorm`) | the residual add, then the norm (and its quantization) | none | engine.h:1911-1992, engine_gemma.h:169 | no (Gemma 4's sandwich norms; the drafter's add_rmsnorm) | unproven on the laptop (no Gemma file) |
| q/k norm + RoPE + KV store (`attn_prep`, `attn_prep_g`) | the per-head norms, RoPE, the cache writes (and the int8 copies for attn_q8) | none | engine.h:1959; engine_gemma.h:39 | yes: the Qwen models | no unfused form; the reference |

#### Fused epilogues and other steps

| fused form | replaces | switch (default) | chosen at | taken on the laptop | proof |
|---|---|---|---|---|---|
| conv + state update (`gdn_conv` -DCONV_UPD) | the short conv, then the conv history update | `ARC_GDN_FUSE` (on) | engine_load.h:36; engine_qwen.h:39, 51-52 (one token, not in verification passes) | yes: the Qwen models | **not in golden**: add ARC_GDN_FUSE=0 |
| recurrence + gates (`gdn_recur` / `gdn_chunk` -DFUSE_GATES) | gdn_gates, then the recurrence | `ARC_GDN_FUSE` (on; not in verification passes: `!spec_`) | engine_load.h:56; engine_qwen.h:39, 77, 89 | yes: the Qwen models | **not in golden**: as above |
| two recurrence steps in one sweep (`gdn_recur` -DGDN_FUSE2) | the second pass of token t and the first of t + 1 | `ARC_GDN_FUSE2=1` (off: opt-in; not with ARC_GDN_PF) | engine_load.h:50-54 | no (opt-in) | golden ARC_GDN_FUSE2=1 = base on the 4B, 9B, 27B |
| the multi-GPU sums' fused forms (tp.cl) | a sum, then the residual add / norm | per split (choose_sums, ARC_TP_*) | zero/mgpu/engine_tp.h | yes under ARC_TP_SIM (tp2 / tp3) | golden tp2 / tp3: their own hashes, stable over two runs; not against one GPU (not bit-identical by design) |

**Gaps in the proof** (to close in Phase 0):
- **Off switches the golden set doesn't test yet:** `ARC_GDN_FUSE=0` (conv + state update, recurrence + gates; every Qwen model) and `ARC_INK_FUSE=0` (Inkling's controller). Both are being added as fusion rows.
- **`moe_rg` has no off switch.** Its exactness against decode_w8 + the expert GEMM isn't shown anywhere: a golden row that moves its row range out of reach (`ARC_MOE_RG_MIN`) is to be added for the 35B and Inkling.
- **Not taken on the laptop:** `XW_GU` and `H8_OUT` (XMX / Alchemist: from the kit); `Q2F` and `add_norm_quant` (they need a Gemma 4 or Qwen3 file).
- **No switch, exact by construction** (the same rows and kernel over one buffer) or the reference chain itself: qkv+z, router+shared gate / up, `-DH8F`, the norm+quant family, `attn_prep`.

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

| Model | Fused forms today (from A) | Launches a token now | Floor (ways 1-8; the lowest, ways 1-12: design 3.7a) | Planned chain, per layer |
|---|---|---|---|---|
| Qwen3.5 4B / 9B (dense, 3 DeltaNet : 1 attention) | Xe-LPG: 4B attn gate+quant; conv+state update; gated norm+quant; norm+quant; norm+rot; q/k norm+rope+kv store; quant+f32 copy; recurrence+gates; swiglu+quant; swiglu+rot; 9B the same but no norm+rot (its rows get a rotation block of 1,024, above the default's 512) | 4B ~426 (125H `--prof-gen`, 2026-11-26) | ~5 a layer: 4B ~160 | DeltaNet: [norm -> qkvz + ab, conv / state / gates in the epilogue] · [recurrence] · [gated norm -> ssm_out + residual] · [norm -> gate / up, SwiGLU] · [down + residual]; attention: [norm -> q / k / v, q / k norm, RoPE, KV store] · [attention] · [sigmoid gate -> o + residual] · the FFN's 2 |
| Qwen3.8 27B | Xe-LPG: no H8 forms (no H8 on the iGPU); attn gate+quant; conv+state update; gated norm+quant; norm+quant; q/k norm+rope+kv store; recurrence+gates; swiglu+quant; qkv+z | ~710 | **321** | as the 4B, 48 DeltaNet + 16 attention layers |
| Qwen3.6 35B-A3B (MoE) | Xe-LPG: as the 27B, plus decode+expert gemm (moe_rg) and router+shared gate/up | ~620 | **241** | the mixer's 3, then [norm -> router + shared gate / up] · [top-k -> experts' gate / up, the shared down grouped in] · [experts' down]; the combine in the next layer's prologue |
| Flash Next | to fill | ~1,050 | **~483** | the hc mixer's 2 global steps per block, then the mixer's or the MoE's 3; `hc_combine` in the block's last epilogue |
| DeepSeek-V4.1 | to fill | ~1,900 | **~415** | the design's 10-13 steps a layer (mix, Sinkhorn + norm -> q_a / kv grouped, attention, out_a in one launch, out_b + expand, the MoE's 3) |
| Gemma 4 dense (31B) | to fill | ~840 | **301** | as Qwen3 dense; the post-attention norm + residual + FFN norm in gate / up's prologue |
| Gemma 4 26B MoE | to fill | to measure | 6 a layer | qkv · attention · o · [dense FFN gate / up + router] · [experts' gate / up + dense down] · [experts' down] |
| Inkling 2 Mini | Xe-LPG: controller; decode+expert gemm; norm+quant; swiglu+quant (the fewest) | **1,356** | **385** | every visit: [mix + norm -> q / gate / rel / k / v / sink / kcp / vcp grouped, K / V convs in the epilogue] · [attention] · [o + post-conv + residual] · [FFN mix + norm -> router] · [top-k -> experts' gate / up, situ] · [experts' down]; later passes add the controllers' 4 |
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
