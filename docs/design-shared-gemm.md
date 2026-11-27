# Design: one shared GEMM layer for every model

Status: draft for the user's review (2026-11-24). Nothing is built yet.

## 1. Why

Every matrix product in inferzero picks its own kernel today:
- dense layers through `matmul` / `matmul1` (engine.h:1392, 1525);
- MoE experts through `layer_moe` (engine_moe.h:8);
- audio, vision and Whisper through wrappers of their own;
- f32 / bf16 products through more kernels again.

So a faster GEMM helps only the paths that call it.

| What got faster | What stayed behind, and why |
|---|---|
| The dense tile GEMM (gemm.cl, XMX, H8R, wide tile) | Inkling's experts ran per tile (`moe_gateup_q`) because `moe_gemm_layer_ok` wanted every layer to pass. Measured on the 125H, 4K prompt: the tile GEMM is 3.4x faster (10.3 s → 3.0 s) |
| The int8 tile GEMM | Gemma 4's audio encoder has no int8 path at all: `vit_gemm_f16` only (audio_gemma4a.h:403) |
| The engine's split-K and grid rules | Whisper has a copy of them (whisper_engine.h:519), which doesn't follow later changes |
| Fused decode inside the kernel (`moe_rg`) | The expert tile GEMM still decodes every expert to W8 first: `decode_w8` took 7.0 s of the same 4K prompt and ate the GEMM's gain |

Two more problems:
- **Routing by card, not by hardware.** Choices are written as card checks: `p140_` (the Arc 140T preset), `sg8 && xmx && !integrated()` (Alchemist), `xehpc_on()` (Data Center Max), `zrt::integrated()`. A new card needs new branches.
- **About 150 `ARC_*` switches** steer GEMM and GEMV choices (the list is in section 10). Several do the same thing in different paths, for example the register file: `ARC_GEMM_GRF`, `ARC_MOE_GRF`, `ARC_GEMV_GRF`, `ARC_H8_GRF`, `ARC_X16_GRF`, `ARC_GEMM_XW_GRF`.

The user's worry: a change for one model can move another model's KL. A shared GEMM must not make that worse. Section 4 is how it avoids that.

## 2. Goals and non-goals

Goals:
1. One GEMM layer that every matrix product goes through: dense, experts, GEMV, the two-pass (Q2) residual, the multi-GPU shards, audio, vision, Whisper.
2. Kernels picked from the GPU's hardware traits and the problem's shape, not from card IDs. A card we've never seen gets a sensible plan without a code change.
3. A new or faster kernel is written once, and every model that can use it does.
4. Model-specific code stays in its own files. A shared GEMM change can move a model's logits only if that model allowed it (section 4).
5. Every step of the migration is bit-identical for every model, and checked by hashes.

Non-goals:
- No change of numbers during the migration. New kernels come after it, each behind the layer.
- Not a general BLAS. It covers the products inferzero has: int8 activations against GGUF weight types, plus the f16 / f32 / bf16 products that exist today.
- Attention, DeltaNet, Inkling's controller and other non-GEMM kernels stay where they are.

## 3. The design

The architecture is oneDNN's GPU GEMM design (its `gemmstone` generator), mapped onto our files in 3.0. The sections after it describe each part, from the model's call down to the kernel.

### 3.0 oneDNN's GEMM design, mapped

This section is from a read of oneDNN's main branch on 2026-11-25 (`src/gpu/intel/gemm/jit/`). oneDNN splits a GEMM into five pieces, and we keep the same split:
1. a problem description;
2. a strategy;
3. a catalog of strategies per hardware generation;
4. a selection model;
5. a generator that turns a strategy into a kernel.

Its attention kernels reuse the same pieces as microkernels.

| oneDNN | What it is there | Ours | Section |
|---|---|---|---|
| **Problem** (`gemmstone/problem.hpp`) | the types of A, B, C and the accumulator; layouts and alignment; batch; A / B scales and zero points, with their group sizes; the post-op chain | `GemmOp`. Our epilogues are a post-op chain in oneDNN's kinds: eltwise (SiLU, GELU, sigmoid, softcap, tanh), binary (the residual add, the gate's multiply), sum, and output scales. Our model-specific steps (situ, the controller update, the expert combine with row weights) are extra kinds in the same chain. The norm / quantization prologue is an addition oneDNN's matmul doesn't have (it only takes source scales and zero points) | 3.1 |
| **Strategy** (`gemmstone/strategy.hpp`, `strategy_parser.cpp`) | one parameter set: the register tile (unroll M × N); the K unroll; A / B access (block 2D loads, SLM copies, prefetch distance); the work-group shape; k-parallel (split-K) with its own reduction, in three forms (global, local, variable); the GRF mode; double buffering; remainder handling; the inner-product form (DP4a, DPAS, `dpasw`). The catalog stores each one as a short string, for example `ab4 as8 ab l4 ca1 wg 2x8 int sr` with unroll {32, 16, 8}, work-group {2, 8, 1} and GRF 128 | the stage library's parameters (3.10). Each catalog entry is one strategy string in our own documented format, with one token per stage parameter. `ARC_GEMM_PLAN=1` prints it per weight, the way oneDNN's verbose output does | 3.10 |
| **Catalog** (`selector/db/kernel.db`: ~1,780 entries, 1 MB) | per entry: a selector (hardware, kernel kind, types, layouts), restrictions (alignment, size ranges), the strategy string, driver info (sub-group size, GRF, unroll, work-group), and a model | `zero/gemm/catalog_<arch>.h`, per architecture, each entry with its measurement | 3.5 |
| **Selection model** (`selector/kernel_evaluator.cpp`, `kernel_selector.cpp`) | each entry carries fitted parameters. The evaluator estimates the time from the waves of hardware threads (full and partial, against the GPU's thread capacity), a fixed cost and a cost per wave. It also picks the k-slicing that gets close to 1 or 2 full waves. The selector takes the lowest estimate among the matching entries. A plain priority (`W`) is used for some entries | the same form: each catalog entry carries a small fitted model (fixed cost, cost per full wave, cost per partial wave). The planner ranks the candidates by it, and the tuning cache then measures the top few on the user's card. The fit is made by a C++ tool (`bench/gemm_fit.cpp`) from `gemm_bench` runs. oneDNN's fitting tool isn't in its public tree | 3.5 |
| **Generator** (nGEN, `generator/`) | writes the GPU's machine code directly from the strategy, per hardware generation, without IGC | **OpenCL C from stages, compiled by IGC** (3.12). This is our choice, see below | 3.10, 3.12 |
| **Microkernels** (`generator/microkernel/`: `fuser.cpp`, `shim.cpp`; `selector/db/ukernel_*.db`) | GEMM building blocks with their own catalogs, called from inside other kernels. oneDNN's SDPA (attention) is OpenCL C that calls them. nGEN's code is patched into the IGC-compiled binary by the fuser | the stage library's tile functions (`tile.h.cl`), called by attention, DeltaNet, the controller and the expert tiles instead of their own inner products. A microkernel's shape class has catalog entries like a GEMM's | 3.6 |

**Why we generate OpenCL C, not machine code.**
- Every kernel in the project is OpenCL C compiled by the driver, so one toolchain and one kernel cache cover everything.
- The spill, size and bit-for-bit checks (3.8, 3.12) work on any user's driver.
- Direct machine code needs a code generator per family (Xe-LPG, Xe-HPG, Xe2, Xe3), each checkable for speed only on contributors' cards.

The cost: IGC is sensitive to how code is written (3.10's pitfalls), and oneDNN's code reaches register layouts IGC won't produce. So a machine-code backend stays an option for later: the "assembly" route, for one or two hot strategies, keyed per family, behind the same strategy string and the same `gemm_check`. It isn't the default. If it's ever built, oneDNN's nGEN is a C++ header library under Apache 2.0, and the fuser shows how such code goes into an OpenCL kernel. We would vendor it rather than write our own.

**Licence and reuse.** oneDNN is Apache 2.0, the same as this project, and NOTICE already credits it for the Alchemist int8 GEMM shape.
- We reuse the design, plus the catalog's tile, work-group, K-step and GRF values as starting points for our catalog entries. Each entry that takes oneDNN's values says so.
- We reuse no code for now. If nGEN or gemmstone code is vendored later, it keeps its licence headers and gets a NOTICE line.
- The no-Python rule holds: the parts named here are C++, and I found no Python under oneDNN's GEMM folders. Our fitting and checking tools are C++ (`gemm_fit`, `gemm_check`).

**Where we differ from oneDNN, on purpose.**
- **Exactness first.** oneDNN picks k-parallel freely. We treat it as an approximate strategy that a model must allow (section 4).
- **Weight types.** Our weights are GGUF types with their own decoders (3.4). oneDNN knows int4 / int8 with scale groups.
- **Our prologue and fusion chain** (3.7, 3.12).

### 3.1 The operation: `GemmOp`

One struct describes any product:

| Field | Values |
|---|---|
| Shape | M (rows / tokens), N, K |
| Activations | int8 `aq_` with per-256 or per-32 scales (`asc_`, ACT32) and `absum_`; or f16; or f32 |
| Weights | the GGUF type and its layout: `Raw` (GGUF rows, `iq3t` tiled), `W8` (resident or staged, variants 100-103 and the per-32 Q8_0 form), `H8R`, `XwPacked`, `F16`, `F32`, `BF16` |
| Batching | `Dense`; `ExpertTiles` (the tile table, the row map `stok_`, the row weights `swt_`, the expert cache table `xtab` and offsets); `TwoPass` (the Q2 residual planes, `q2_aq_ / q2_sc_ / q2_ab_`) |
| Output | C, ldc, c_off |
| Epilogue | `Store`, `Add`, `SwiGLU` (today's acc 2), `GeluMul`, `Softcap`, `RowScale`, `Situ` (Inkling's per-expert scales, with its parameters) |
| Exactness | `Exact` (must equal the reference order, section 4) or the model's list of allowed approximations |

A model calls one of three helpers, all built on `gemm::run(op)`:
- `linear(name, w, x)`: dense layers (today's `matmul`);
- `linear_gu(name, wg, wu, x)`: a stacked or fused gate and up (today's `gemv_gu`, `xw_gu_matmul`, acc 2);
- `experts(name, L, part, tiles)`: one MoE part (today's `layer_moe` gate / up and down).

No model file names a kernel.

### 3.2 The GPU's traits

`GpuTraits` is read once from the driver at start, with a few measurements:

| Trait | Where it comes from | Why it matters |
|---|---|---|
| Matrix unit: none / DPAS 8-lane / DPAS 16-lane, and `dpasw` | device properties (Level Zero / OpenCL extensions) | the 125H has none; the 140T and Alchemist have 8-lane; Xe2 has 16-lane |
| Sub-group sizes | device properties | 8 or 16; sets the tile shapes |
| Register file: bytes per register and 128 / 256 modes | the device's EU properties | Xe-HPG's 32-byte registers hold half as much |
| Cores (Xe cores / EUs), threads per EU, clock | device properties | how many tiles fill the GPU |
| SLM per work-group, L2 / L3 size | device properties | the staging size of a tile |
| 2D block loads | extensions | `X2D` / `XW_2D` |
| Integrated or discrete; memory bandwidth | properties + a short copy test, cached | the GEMV choices and the CPU's share |
| Driver and compiler version | the driver | some forms spill on old drivers |

Today's card branches become trait checks. A small table in `zero/gemm/presets.h` keeps the measured exceptions (the 140T's resident-weight rule, Alchemist's one row per sub-group). Each entry is a plan override keyed by device ID and shape class, with a comment naming the measurement and issue. A preset can only choose between strategies that are already exact for that card. It never adds code.

### 3.3 Strategies: today's kernels, each described once

Each strategy says:
- which traits and operations it supports;
- whether it keeps the reference order (exact);
- its parameter space (tile, work-group, K step, register file);
- its spill status per family from ocloc, recorded with the strategy so the planner never picks a spilling form when a spill-free one that measured as fast exists.

| Strategy | Today's kernels | Typical use |
|---|---|---|
| `gemv` | decode.cl `gemv`, `gemv_v` (+ `-DGU2`, `-DSPLITK`, `-DQ2F`) | T = 1 |
| `gemv_mt` | `gemv_mt`, `gemv_mt_v` | T 2-8 |
| `tile_dp4a` | gemm.cl `gemm` (+ `-DMOE_TILES`) | prompts without XMX (the 125H); also the reference |
| `tile_xmx16` | gemm.cl `-DXMX` (+ `X2D`, `WPF`) | Xe2 prompts |
| `tile_xmx8` | gemm_x8.cl | 140T, Alchemist |
| `wide_h8r` | gemm_xw.cl / gemm_xw8.cl, with H8R weights | discrete prompt GEMMs |
| `tile_f16` | gemm_x16.cl (not exact) | opt-in |
| `expert_small` | decode_moe.cl `moe_gateup_q` / `moe_down_q` | experts in generation and small chunks |
| `expert_fused` | `moe_rg` (decode inside the kernel, + `RG_XMX`) | expert prompts where the decode-first copy costs too much |
| `f32_small` | `gemv_bf16`, `gemv_f32`, `hc_gemm_f32`, `ink_mm_f32(_sg)` | the f32 / bf16 matrices that exist today |
| `f16_encoder` | ops_vit.cl `vit_gemm_f16` | encoder layers kept in f16 |
| `cpu` | `host_cpu_gemv`, `xcpu_run` | ARC_HOST_CPU, expert misses in RAM |
| `npu` | `npu_ffn` | ARC_NPU |

Expert tiles become a batching mode of the same tile strategies, not separate code. `tile_dp4a` with `ExpertTiles` is today's `moe_gemm`. A faster tile kernel therefore speeds up dense and expert layers at once.

### 3.4 Weight loaders: one decoder per type

Today each type is decoded in several places:
- `decode_w8_sb / _sl / _vb`;
- `h8r_decode`;
- `moe_rg`'s per-row decode;
- the per-tile kernels' decode;
- `w8_requant`.

The design keeps one decoder per GGUF type in `zero/kernels/gemm/load_<type>.cl`, used two ways:
- **Ahead** (today's `decode_w8`): decode to a W8 / H8R area once, for weights read many times (dense layers, resident weights).
- **Fused**: the tile kernel's weight-load step calls the same decoder on each K block as it brings it into registers or SLM. No W8 copy is made. This is what oneDNN does with compressed int4 weights, and it is the fix for the 7.0 s `decode_w8` in Inkling's 4K prompt.

Because the decoder is shared, "ahead" and "fused" give the same int8 values and scales, and the GEMM's result is the same bit for bit. NVFP4 gets a per-16 scale form in the tile kernels, so it stays exact per 16 as in generation, instead of today's per-32 fold.

### 3.5 The planner and the tuning cache

`plan = planner(traits, op)` picks a strategy and its parameters.

**A hand-picked catalog per architecture, as oneDNN has.** oneDNN doesn't derive its kernels from traits alone. Its GEMM picks from a catalog of strategies chosen by hand and by offline tuning for each hardware generation; the 125H's `gemm OOIS NNN 32 32 ... wg 8x4 k64 grf256` is one such entry. We do the same, keyed by architecture, not by card:

| Architecture | Our cards |
|---|---|
| Xe-LPG | Core Ultra 1 iGPUs (the laptop's 125H) |
| Xe-LPG+ | Core Ultra 2 H iGPUs (Arc 140T) |
| Xe-HPG | Alchemist (A770, A750, A380) |
| Xe2-LPG | Lunar Lake iGPUs (Arc 140V) |
| Xe2-HPG | Battlemage (B580, B60, B70) |
| Xe3-LPG | Panther Lake iGPUs (B390) |
| Xe-HPC | Data Center Max (one user; presets only, no catalog of its own yet) |

`zero/gemm/catalog_<arch>.h` lists, per shape class and weight type, the strategy and parameters chosen by hand from measurements: tile, work-group, K step, register file, submission form. Each entry names its measurement (card, date, issue). Entries follow oneDNN's form (3.0):
- a selector (shape class, weight type, layout);
- restrictions (alignment, the M / N / K ranges it was measured on);
- the strategy string;
- a small fitted model.

Where an entry starts from oneDNN's catalog values, it says so.
- A few architectures cover every card: a B60 and a B70 share the Xe2-HPG catalog and differ only in traits the parameters scale with (cores).
- A new card of a known architecture needs no new entry.
- A new architecture gets a catalog of its own, starting from its nearest relative and from `gemm_check`'s offline numbers.

The tuning cache (below) only picks among the catalog's few candidates for that entry, so a hand-chosen form always stays reachable and the cache never invents one.

The order of the decision:
1. **Rules and the model.** The architecture's catalog entries that match the shape class and type, ranked by each entry's fitted model as oneDNN's evaluator does: waves of hardware threads (full and partial) against the GPU's capacity, a fixed cost, and the choice of k-slicing for exact strategies that allow it. For an unknown architecture, the supported strategies in an order set by traits. The shape classes: T = 1, T 2-8, T 9-255, T ≥ 256, and expert tiles by tile count.
2. **Presets** (3.2), if one matches.
3. **The tuning cache.** For prompt shapes, the fastest of a few candidate tilings is measured on first use. It's saved in `%USERPROFILE%\.inferzero\tune\<device uuid>-<driver>.txt`, next to the kernel cache. Only exact candidates are tuned, so tuning never changes a result, only the speed. It starts opt-in (`ARC_GEMM_TUNE=1`) and becomes the default once contributors' cards show it is stable.

The plan is computed once per weight and shape class and cached on the weight, as `g1k` / `guk` are today. The hot path does no string building or map lookups beyond today's.

Switches:
- `ARC_GEMM_PLAN=1` prints each plan once: the layer, the strategy, its parameters, exact or not, and why.
- `ARC_GEMM_FORCE=<strategy>[:params][@pattern]` forces a plan for testing, for all products or the ones matching a name pattern.
- Today's switches stay as aliases for one release, each translated to a FORCE or to a planner parameter and saying so once. Then they go. Model-specific ones like `ARC_INK_DOWN_TILE` become planner rules ("expert tiles, no XMX, T > 8: `expert_small` per tile"), not model branches.

### 3.6 Attention, DeltaNet and the controller

Each of these has matrix products inside, but in different forms:

| Kernel | Its products | Where it goes |
|---|---|---|
| Inkling's controller (`ink_ctrl_fused`, `ink_ctrl_fused_t`) | one small GEMM (a 512-row chunk × 64 × 2304) plus a tanh update | the shared layer, as a GEMM with an epilogue of its own. oneDNN did this shape in 0.06 ms against our 10.2 ms |
| The MoE router, Flash Next's `hc_gemm_f32`, `ink_mm_f32` | plain f32 / bf16 products | the shared layer (`f32_small`) |
| Prompt attention (`attn_fx`, `ink_attn_pp` / `_ps`, `attn_q8`) | Q·Kᵀ and P·V, with the softmax between them in registers (flash attention) | stays its own kernel, since splitting it into two GEMMs would write the score matrix to memory. It uses the shared building blocks below |
| Decode attention | one query row against the cache: a GEMV with a softmax | its own kernel, shared building blocks |
| Chunked DeltaNet (`ops_gdn.cl`, `gdn_x.cl`) | chunk-sized matrix products inside a recurrence | its own kernel, shared building blocks |

The shared building blocks are a kernel include, `zero/kernels/gemm/tile.h.cl`. It holds:
- loading a tile (2D block loads where the GPU has them);
- the inner product on DP4a or XMX (8- and 16-lane);
- register-tile accumulation;
- the online-softmax merge.

The tile sizes come from the same `GpuTraits` and planner (a `fused` shape class), so attention and DeltaNet get the same per-GPU tuning as the GEMMs. When the tile code gets faster, these kernels get it too. Their numbers still change only with their own kernel, so their KL stays the model's own.

**DeltaNet in detail** (zero/engine_qwen.h `layer_gdn`). It has three parts that can go further.

*1. Decode: 6-7 launches a layer today, against a floor of 3 (3.7).* Today, with `ARC_GDN_FUSE` on, a DeltaNet layer runs:
1. `rmsnorm_quant`;
2. `gemm_qkvz` (qkv and z stacked; alpha / beta stacked in only when they are Q8_0, otherwise a separate `gemm_ab` or `gemv_bf16`);
3. `gdn_conv` (`-DCONV_UPD` stores the history);
4. `gdn_recur` (`-DFUSE_GATES`);
5. `gated_norm_quant`;
6. `gemm_ssm_out`.

At the floor it's three launches:
- **[the norm in the prologue → qkvz and ab as one grouped GemmOp, the conv, its state update and the gates in the epilogue].** All of them are per channel or per head. When ab is F32 / BF16, the group needs either a mixed-type group (each weight its own loader, 3.4) or ab requantized to Q8_0. Requantizing changes its numbers and needs a KL check, so the mixed-type group comes first.
- **[the recurrence, q / k l2-norm in its prologue].** A work-group owns whole heads, so the norm is local.
- **[the gated norm + quantization in ssm_out's prologue].**

The 27B's 48 DeltaNet layers save 3-4 launches each, ~150-190 a token. That's the largest single item in its gap to the floor of 321. Verification passes (drafting) keep today's split forms, which save `conv_pre / qkvin / xin / abin` for the replay, until the replay reads those from the fused kernels' outputs.

*2. Prompts: the recurrence is 14-16 % of a 27B 4K prompt.* `gdn_recur` takes 244 ms on the B70, 290 ms on the B60 and 315-344 ms on the A770.

The chunked form (WY: products inside a chunk, then the state's update between chunks as GEMMs) should win, since the sequential update waits on each token's sums. Ours loses everywhere:
- `gdn_chunk_x` on the B60: 886 ms against `gdn_recur`'s 290, so the prompt drops from 1,087 to 918 tok/s (11-13);
- on the B70: -3 % at 4K, -2.5 % at 16K (11-15);
- on the 125H: 10-14x slower (engine_qwen.h).

So `ARC_GDN_CHUNK=3` stays off. It was written by hand, outside the GEMM work. In this design the chunked form becomes a client of the shared layer: its products inside a chunk (Q·Kᵀ, the attention-like product) and the state's update over a chunk become microkernel calls (3.0). The tile strategy comes from the catalog (`dpasw` on Alchemist, 16 lanes on Xe2 / Xe3, DP4a or f16 on the 125H), not from a separate kernel. This is the migration's DeltaNet step, measured against `gdn_recur` on each family. `gdn_recur` stays the default on a family until the chunked form beats it there, and the KL is checked against `gdn_recur`: today's chunked form measured 0.00102.

*3. Smaller items.*
- **State precision.** The decode state is f32, read and written every token: ~150 MB each way a token on the 27B, about 3 % of its bytes. An f16 / bf16 store halves that, but the state accumulates, so it needs the KL check over long generations (hundreds of positions, the maximum reported). It would be opt-in until that passes.
- **`ARC_GDN_FUSE2`** (one sweep a token, opt-in since 11-22) measured slower where it was tried:
- the A770: `gdn_recur` 338 → 392 ms at 4K (#45);
- the 125H: 37.2 → 40.6 ms on 512 tokens.

It's still unmeasured on B-series, so it goes into the DeltaNet strategies' candidates for the tuning cache. It's bit for bit equal to `gdn_recur` (the 4B's logits byte for byte, 11-22), so the cache may pick it freely.

### 3.7 Fewer launches

Measured costs per launch:

| GPU | Cost per launch | Effect |
|---|---|---|
| The 125H | ~1.5-2 µs for tiny kernels (replay_probe); ~7 µs idle between dependent kernels in the real models | Inkling's ~970 launches above its floor cost ~7 ms of a ~115 ms token, so fewer launches help the laptop too (3.7a) |
| B60 under Windows | 2.7 µs | at 30-60 tok/s, launches are a real share of a token |

**The floor: one launch per global dependency.** A launch can't be avoided only where every work-group's output must be complete before any work-group of the next step starts. That happens where the next step reads a whole result: a reduction over a whole row, over all the keys, or over all the experts. Everything else joins a neighbouring launch. Per decode step:

| Layer kind | Its launches at the floor | Count |
|---|---|---|
| Attention + dense FFN | [norm + q / k / v and every other product of the layer input, with q / k norm, RoPE and the KV store in the epilogue] · [attention] · [output product + residual] · [gate / up + SwiGLU] · [down + residual] | 5 |
| Attention + MoE | the first three as above · [router + top-k, with the shared expert's gate / up grouped in] · [the experts' gate / up] · [the experts' down] | 6 |
| DeltaNet + FFN | [norm + the input products, the short conv in the epilogue] · [the recurrence, q / k l2-norm in its prologue, the gated norm in its epilogue] · [output product + residual] · the FFN's 2 (dense) or 3 (MoE) | 5 or 6 |
| Once a token | [final norm + lm_head]. Today the logits are read back and sampled on the host, and the embedding row is decoded on the host (no launch). A GPU argmax / top-k can run inside lm_head's launch by way 7, which also saves the logits' read (0.8 MB a token for Inkling's 201K vocabulary) | 1 |

Three of these points aren't obvious:
- **The router is its own point.** Every expert tile needs the top-k, and the top-k needs all the router's logits. Computing the router again in each work-group would read its weights (0.5-1 MB) once per work-group, which costs more than a launch.
- **A dense FFN's down is its own point.** Its K is the whole intermediate row.
- **The expert combine needs no point.** The next layer's prologue sums each row's k expert outputs in a fixed expert order, together with the residual. Float atomics in down's epilogue would be inexact.

**The floor for every model.** These counts come from a read of each family's decode chain on 2026-11-25 (one token, the iGPU's default path). Today's counts are approximate, because stacking depends on the weight types at load; `test_launch_budget.ps1`'s first run records the real ones.

| Family | The floor's launches a layer | Floor a token | Today a layer | Today a token |
|---|---|---|---|---|
| Qwen3.5 / 3.8 dense (4B, 9B, 27B): 3 DeltaNet to 1 attention | DeltaNet: [norm → qkvz + ab, conv / state / gates in the epilogue] · [recurrence] · [gated norm → ssm_out + residual] · [norm → gate / up, SwiGLU] · [down + residual]. Attention: [norm → q / k / v, q / k norm, RoPE, KV store] · [attention, combine by way 7] · [sigmoid gate → o + residual] · the FFN's 2 | 5 / 5. **27B (48 + 16 layers): 321** | 11-12 / 11-13 | 27B ~710 (`129 rmsnorm_quant` a token, engine.h:1829) |
| Qwen3.6 35B-A3B (40 layers, 30 DeltaNet + 10 attention) | the mixer's 3, then the MoE: [norm → router + the shared expert's gate / up] · [top-k in the prologue → the experts' gate / up, with the shared down grouped in by way 6] · [the experts' down]; the combine and the shared gate's sigmoid go into the next layer's prologue | 6. **241** | 15-16 | ~620 |
| Qwen3 dense, Qwen3-ASR's decoder | [norm → q / k / v] · [attention] · [o + residual] · [norm → gate / up] · [down + residual] | 5 | 11 | |
| Gemma 4 dense (31B: 60 layers) | as Qwen3 dense. The post-attention norm, residual and FFN norm (`add_norm_quant`) go into gate / up's prologue (two row reductions in a row, both per work-group). The post-FFN norm and out_scale go into the next layer's prologue. KV-shared layers skip k / v inside the same launch | 5. **31B: 301** | 14 | ~840 |
| Gemma 4 26B MoE | qkv · attention · o · [the dense FFN's gate / up + the router, each with its own norm as a per-weight prologue] · [the experts' gate / up + the dense down, by way 6] · [the experts' down] | 6 | ~26 | |
| Gemma 4 E2B / E4B (per-layer embeddings) | 5, plus 2 for the PLE: [`ple_gate`, which reads the layer's whole output] · [`ple_proj`, which reads the gate's whole output]; `gemm_ple_model` once a token | 7 (+1 a token) | ~18 | |
| Flash Next (48 layers, 3 DeltaNet to 1 attention, hc 4 streams) | before each of the two blocks, the hc mixer is 2 global steps that can't be prologues: `hc_down` (K = 10,240, 6.9 MB of weights) and `hc_up` (10.5 MB), both too large to repeat per work-group. Then the block: the mixer's 3 or the MoE's 3. `hc_combine` goes into the block's last epilogue | 10. **~483**, plus 1 an attention layer for QSA past ~2K positions (scores + top-k in one launch by way 7) and 2 for the PLE at layer 1 | 21-25 | ~1,050 |
| DeepSeek-V4.1 (40 layers, hc 4) | [mix product (K = 20,480 → 24)] · [Sinkhorn + collapse + norm in the prologue → q_a, kv and, where present, the compressor and indexer projections, grouped] · [q_a's norm → q_b (+ the indexer's q_b), RoPE in the epilogue] · [attention, combine by way 7] · [the 8 out_a groups in one launch by way 6] · [out_b, expand in the epilogue] · [FFN mix product] · [router + the shared gate / up] · [the experts' gate / up + the shared down] · [the experts' down]. Index-source layers add [scores + top-k] (+1-2); a KV source with a complete group adds [the compressed row's norm → ix_k] (+1); the Engram layers (1, 14) add 1-2 | 10 (plain), 11-13 (index / KV / Engram). **~415** | ~45-50 | ~1,900 |
| Inkling 2 Mini (24 layers; 6-17 run 3 times: 48 visits) | every visit: [the attention mix + norm in the prologue → q / gate / rel / k / v / sink / kcp / vcp grouped, the K / V convs in the epilogue] · [attention, the clips and norms in its prologue when a tile isn't whole heads] · [o + post-conv + residual] · [the FFN mix + norm → router] · [top-k → the experts' gate / up, situ in the epilogue] · [the experts' down]; the combine, the FFN conv and the residual go into the next prologue. The 24 later-pass visits (controllers on, `pass > 0`) add 4: [the q / k / v controllers' down, grouped (K = 2E → 64)] · [their gain / forget, grouped, the update in qkv's prologue], and the same two for the FFN controller before the router | 6 / 10. **385** (24 × 6 + 24 × 10 + 1) | 19 (sliding), ~25 (global), ~38 (later pass) | 1,356 (125H profile; docs/inkling2.md: 1,412-1,492) |
| Whisper's decoder | [LN → qkv, KV store] · [self-attention] · [out + residual] · [LN → cross q] · [cross-attention, combine by way 7] · [cross out + residual] · [LN → fc1, GELU] · [fc2 + residual] | 8 | 16 | |
| Encoders (Whisper, ViT, Gemma's audio), per layer per window or image | [LN → qkv] · [attention] · [out + residual] · [LN → fc1 / up, activation] · [down + residual] | 5 | ~10 | (once per input, not per token) |

The gap is largest where today's chain has the most norm, quantization and elementwise launches: DeepSeek (~4.6x its floor), Inkling (3.5x), Flash Next (2.2x), the 27B (2.2x), Gemma (2.8x). On the 125H, replay_probe's tiny kernels cost ~1.5-2 µs a launch, but in the real models the GPU idles ~7 µs between dependent kernels (xe2-tuning.md: the 35B 5.2 ms of a 51.8 ms token over 692 launches), so Inkling's gap of ~970 launches is ~7 ms of a ~115 ms token. On the laptop the floor is about fewer passes over memory (each fused norm, quantization and elementwise step reads and writes its row once less), more than about launch cost. On the B60 (2.7 µs under Windows) the same gap is ~2.6 ms of a ~35 ms token. q / k norm in the q / k / v epilogue needs a work-group's N tile to cover whole heads (head size 128 / 256). The planner either picks such a tile or moves the norm into the attention's prologue. A split decode attention (split over the keys) stays one launch only with way 7 below.

`ARC_GEMM_PLAN=1` prints three numbers per model, so each fusion shows how much is left:
- today's launches a token;
- the floor, computed from the layer chain (3.12's graph knows the dependencies);
- the gap between them.

The shared layer gets toward the floor in eight ways. Ways 1-4, 6 and 8 keep each value's operations in the reference order, so they are exact. Ways 5 and 7 are exact under the conditions given with them.

1. **Grouped products.** Products that read the same input become one launch over the weights side by side: N concatenated, each output part placed by `c_off`. Examples: q / k / v; gate and up; Inkling's q / rel / gate / sink / k / kcp / v / vcp (Basalt does all eight in one); the shared expert next to the router. `GemmOp` takes a list of weights with one input. The planner groups them when their types and K match, and the per-weight output keeps its own offset and epilogue.
2. **Prologues.** The step that makes the GEMM's input can run inside the GEMM's activation load: the RMSnorm, int8 quantization and the activation scales (today separate `*_rmsnorm_quant` / `quant` launches). This fits when one GEMM, or one group, reads that input. When several read it, one quantization launch stays shared.
3. **Epilogues.** The steps after the GEMM go into its store: SwiGLU (already acc 2), the residual add, the gate's sigmoid, Inkling's controller update, the expert combine with its row weights, the softcap.
4. **One launch per MoE part.** Every expert's tiles run in one launch, as `moe_gemm` does now. The routing steps (route, scan, scatter: 3 launches a layer) merge into one kernel where the expert count allows.
5. **Redundant prologues.** When several launches read one normalized row, each work-group computes the row's RMS (and its activation scales) itself, and the shared norm / quantization launch that way 2 keeps goes away. At decode that's one small reduction a work-group, which costs nothing. In a prompt chunk each work-group already reads its rows' whole K, so the extra cost is a second pass over its activation rows. It's bit-identical to today only if each work-group reduces in the same order as today's norm kernel. Folding 1/RMS into the output's row scale instead (the int8 values don't depend on the RMS when the scales are per block) is faster, but it changes rounding, so it's a KL-checked change.
6. **Horizontal fusion.** Independent small steps at the same dependency level go into one launch, each op with its own range of work-groups. Examples: Inkling's four controllers; the KV store with the conv-state update; the q / k / r clips with V's norm; the expert scale tables.
7. **"The last work-group finishes".** This is a reduction across work-groups without a second launch and without a barrier across work-groups. Each work-group writes its part, issues a memory fence, and increments a counter atomically. That needs device-scope atomics: the kernels already build as `-cl-std=CL3.0` (engine_setup.h:485), and the build checks the driver's `__opencl_c_atomic_scope_device` feature. The work-group that sees the last count does the follow-up step. Uses: the split attention's combine, the sampling over the vocabulary, a residual + norm that can't be a prologue. No work-group ever waits for another, so this can't hang, unlike the megakernel. Rules:
   - One counter per output tile (per head, per row block), not one per launch, so the follow-up stays parallel. One work-group normalizing a whole prompt chunk would be serial.
   - The follow-up sums the parts in index order, not arrival order, so the result is exact.
   - The follow-up resets its counter. The engine also clears the counters at load and after a device-lost error, since a run that stopped halfway leaves them wrong.
8. **Small elementwise steps never get a launch of their own.** Conv taps, clips, the gate's sigmoid, softcap and the mask go into the producer's epilogue or the consumer's prologue. The plan checks this: an elementwise step with its own launch is reported as a fusion miss in `ARC_GEMM_PLAN`'s output.

Not proposed:
- A persistent kernel per layer ("megakernel"): it needs barriers across work-groups, and a hung barrier froze the laptop on 2026-11-23. Way 7 gets the cross-work-group reductions without one.
- Command-list replay: it lowers the cost per launch, not the count. It measured as a gain on B-series only (replay_probe), so it stays a B-series option outside this layer (3.9).

#### 3.7a The lowest number of launches (the user, 2026-11-27; checked against the code by the Developer the same day)

**Why it matters on the laptop too:** in the real models, the GPU idles ~7 µs between dependent kernels on the
125H. The 35B spends 5.2 ms of a 51.8 ms token on that over 692 launches, the 27B 6.8 of 208.7 over 933
(xe2-tuning.md). replay_probe's 1.5-2 µs is for tiny kernels, not a model's. So Inkling's ~970 launches above its
floor cost ~7 ms of a ~115 ms token, and the lowest count below would save ~7 ms against today.

The floor above is what ways 1-8 reach. The lowest count possible without a persistent kernel is set by the true
all-to-all dependencies alone: a step that needs **every** value of the step before it. Four more ways remove the
boundaries that aren't of that kind. Each is exact **only under the conditions given**, proven per GPU kind by
`gemm_check`, which repeats every fused form many times at random sizes.

9. **A work-group owns its heads, from the projections to the mixer** (one token, and drafting's few rows).
   - **DeltaNet:** one work-group owns a k-head group: its v-heads x 128 state columns, within the 1,024-item limit
     (e.g. 3 x 128 = 384). It computes that group's q / k / v / z / beta / gate rows as GEMV slices, then the
     recurrence. A smaller group would recompute q / k per v-head (1.3 MB of int8 weights a group at E = 5,120), far
     more than a launch costs. The 384-item group is the first form to build and check by ISA for spills.
   - **Attention at short contexts** (one work-group per head is enough): the same, with the new token's KV store
     head-local in the owning group.
   - **Attention past short contexts:** the decode attention splits over the keys, and every key-split work-group
     needs the head's whole q. That's an all-to-all, so it stays a launch of its own there.
   - **Exact only if** each output row is summed exactly as today's GEMV form for that type and GPU kind sums it
     (`gemv` vs `gemv_v`, rows per work-group, the K split across lanes, the sub-group size 8 or 16), and its prologue
     norm meets 10's conditions. Each fused variant is restricted to the GEMV forms its kind uses.
   - **GQA's few KV heads:** a K / V group split over K across work-groups and joined by way 7, in index order. The
     mixer then runs in the last work-group of each KV group, which is serial. That's fine for DeltaNet's fixed cost a
     head; for attention it holds only at short contexts.
10. **The next norm's statistics from the producer's last work-group** (way 7). Exact only if the last work-group
    reproduces `rmsnorm_quant`'s order:
    - the same 256 items;
    - the same strided loop;
    - `work_group_reduce_add`;
    - the same SIMD width, forced on both kernels with `intel_reqd_sub_group_size`. Forcing it may itself change
      `rmsnorm_quant`'s current width, which is a check of its own.

    The wide-rows path (a work-group per 16 superblocks) has a different order, and way 10 matches whichever path
    that T takes.
11. **Top-k in the router's last work-group** (way 7), for one token and drafting's few rows. Exact if it keeps
    today's softmax / sigmoid order, the index-order tie-break and the weights' normalization. Prompt chunks still
    need the scan / scatter of rows to experts: the counts per expert depend on every row.
12. **The head and argmax in one launch:** a fast path for unconstrained greedy and plain sampling, and drafting's
    argmax. With a tool-call constraint, a JSON prefix, logit bias, penalties, top-p, logprobs or `--dump-logits`, the
    logits still come back to the host as today.

**What can't go**, each a step that reads every value of the one before:
- attention / DeltaNet output -> the output projection (its K is every head);
- the output projection + residual -> gate / up (the whole row);
- SwiGLU -> down (the whole intermediate row);
- the router -> the experts (the routes);
- the experts' gate / up -> the experts' down (each expert's whole intermediate row);
- q -> the key-split attention, past short contexts;
- Inkling's controllers: the down (K = 2E -> 64) and the gain / forget feeding qkv's prologue, 2 a controller.
  Redoing gain / forget per work-group would cost E x 64 int8 weights each, times every work-group;
- Flash Next's `hc_down` / `hc_up`: each reads all 4 streams or the whole block;
- DeepSeek's mix products (K = 20,480);
- drafting's chain: each MTP draft needs the previous draft's argmax over the vocabulary;
- between tokens: the sampled token -> the next token's embedding.

**Not boundaries** (head- or channel-local, so prologue or epilogue): q / k norm, the short convs and their states,
the KV store at short contexts, Inkling's mixes and post-convs.

Joining any of the "can't go" steps inside one launch needs a barrier across work-groups, the persistent-kernel form
rejected above because a hung barrier froze the laptop.

**The lowest count, one generated token:**
- **4** a DeltaNet layer with a dense FFN, and an attention layer at short contexts;
- **5** an attention layer past short contexts, and a layer with a MoE FFN;
- **~2** a token (the embedding with the first norm, the head).

| Model | Today | Floor (ways 1-8) | Lowest (ways 1-12) | Past short contexts |
|---|---|---|---|---|
| Qwen3.5 4B (32 layers) | ~426 | ~160 | 130 | 138 |
| Qwen3.8 27B (64) | ~710 | 321 | 258 | 274 |
| Qwen3.6 35B-A3B (40, MoE) | ~620 | 241 | 202 | 212 |
| Flash Next (48, hc) | ~1,050 | ~483 | 434 (9 a layer + 2) | ~446, ~448 with the PLE at layer 1 |
| Inkling 2 Mini (48 visits) | 1,356 | 385 | **338** (24 first-pass x 5 + 24 later-pass x 9 + 2) | +1 per global-attention visit (their number: to read from the chain) |

Flash Next's count follows 3.7's description of `hc_down` / `hc_up` as two global steps a block, not yet re-read in
`ops_flashnext.cl`.

**What cuts the cost per generated token further:**
- **Drafting:** per generated token it's (the main pass + the draft chain) / the accepted tokens.
  - MTP: k serial drafts, each about one MTP layer (~5) + the head (1). 7 drafts add ~42 to the 27B's 258.
  - DFlash2: one drafter pass for all its drafts. DSpark: its own chain.
- **Batched slots:** today they share only the GEMMs. `forward_batch` runs the mixer per slot, on its rows and state,
  so attention / recurrence launches multiply by the slot count. They'd be shared only once the mixer kernels take
  every slot's rows with per-slot state pointers in one launch, a kernel change of its own.
- **Lists of launches:** a step can't be recorded once and replayed as it is, because its arguments (positions,
  lengths) and the experts' grids change every token. What exists is `ARC_BATCH`: each step recorded again in lists of
  up to n. A true replay needs the per-token arguments read from a device buffer, and the expert grids at their
  maximum with an early exit. It's a per-GPU-kind choice:
  - B-series: replay_probe halved the cost a launch (B70 1.19 -> 0.57-0.63 µs, B60 -49 %);
  - A770: only the barrier-after form gained (-4 %);
  - the laptop: tiny kernels 1.36 -> 1.07 ms, large ones 1.95 -> 2.01 ms (+3 %), so it stays off on the iGPU.

**Safety:**
- **No hang:** way 7 never waits, so there's no barrier to hang. It needs release fences in the writers and a
  device-scope acquire in the last work-group. Intel's per-subslice L1 isn't coherent, so a missing acquire gives rare
  stale reads, not a hang. That's why `gemm_check` repeats each fused form many times at random sizes, since one hash
  can't prove it.
- **Device-scope atomics** (`__opencl_c_atomic_scope_device`) are checked per family at build, with the unfused chain
  as the fallback where they're missing (Xe-LP, older drivers): unknown per family until checked.
- **Counters** are reset at load and after a device-lost error. A cancelled pass stops between launches, so a fused
  launch always completes and its counters stay consistent.
- **The real per-kernel limit is Windows' TDR, ~2 s.** Ways 9-12 at T <= 8 are tiny, but any fused prompt-chunk launch
  must stay well under 2 s on the slowest family (Xe-LP, the A380, weights in host memory over PCIe). #99's hang was
  one long launch.
- **Cancel points:** fewer launches within a token is fine. The cancel check stays between layers in prompt chunks.

Ways 9-12 come after Inkling is on the layer and ways 1-8 have brought it to its floor, not before.

`ARC_GEMM_PLAN=1` also prints the launches a token by kind (GEMM, prologue, other), so each step's gain is visible. The goal is each model's floor in the table above, not a fixed number. `tests/launch_budget.txt` gets one row per model in that table: today's count, the floor, and the rule (1-8) that closes each part of the gap. The 2026-11-24 target (Inkling under 600 launches a token, against 1,356 today) is a first stop on the way.

### 3.8 Budgets that stay low

The structure makes few launches, few passes over the activations and small kernels possible, but only a check keeps them that way as features are added. Four budgets, each checked automatically:

| Budget | What it counts | How it's checked |
|---|---|---|
| Launches a token | kernel launches per generated token and per prompt chunk, by model | `ARC_GEMM_PLAN` / `--prof-gen` counts. `tests/test_launch_budget.ps1` runs each model's short run and fails when a count exceeds its row in `tests/launch_budget.txt`. Each row holds today's count and the model's floor (3.7). The count is lowered toward the floor as fusions land, and the test prints the gap. Runs on the laptop and on contributors' cards |
| Host work a token | the CPU's µs per token in the engine: argument setting, plan lookups, list appends | the plan is cached per weight; arguments that don't change are set once at plan time, and only the changing ones (row count, offsets) per launch. `--prof-gen` reports host µs a token next to GPU µs, with a budget row per model |
| Activation bytes | the bytes of intermediates written and read between kernels (a norm's output, quantized activations, gate / up's output, the attention's output, the MoE's gathered rows and combine), per prompt chunk and per token, by model | `ARC_GEMM_PLAN=1` computes it from the plan, and `launch_budget.txt` gets a column for it. Its floor is the bytes left when every legal fusion is taken (3.14). For prompts this matters more than launches: a 4K chunk of the 27B spends ~1.9 ms on launches but moves ~110 GB of intermediates that fusion can remove |
| Kernel size | each kernel form's instruction count, spills and private memory, per GPU family | `gemm_check --offline`: compiles every strategy's forms with ocloc for mtl, arl-h, dg2, bmg-g21, bmg-g31, and fails on a spill (unless the plan has a measured exception, as for the 125H prompt tiles) or on an instruction count above its budget (oneDNN's 812 KB kernel was larger than the instruction cache). Needs no GPU, so it can run on GitHub's Linux runners for every pull request |

**Keeping instruction counts low.** The kernel-size row is the one that grows unseen, so the design adds rules and two measurements.

The rules:
1. **One form per build.** Every choice that is known when the kernel is built (type, tile, epilogue, prologue, batching) is a `-D` define, so a form contains only its own code. No kernel switches between types or epilogues at run time. Today's `moe_rg` is ~3,000 instructions because one build carries several paths.
2. **Unroll only the register tile.** The tile's M × N inner product is unrolled; the K loop and the expert / tile loops are not. Full unrolling is what makes oneDNN's 125H kernel 812 KB, larger than the instruction cache.
3. **Loaders as shared functions.** Each type's decoder (3.4) is one function, called from every strategy. It isn't inlined into the unrolled tile more than once per K step, so adding a weight type adds its decoder once, not into every tile copy.
4. **A budget per strategy and family:** instruction count and binary bytes in `tests/kernel_budget.txt`, set from the first measured form. The icache size is a hard limit: a hot kernel's loop must fit in it.

The measurements:
- **Offline:** `gemm_check --offline` disassembles each form with ocloc and counts instructions per family, failing above budget (above). This runs on GitHub's runners.
- **At run time, on every user's driver:** next to the default-on spill warning (clrt.h), each kernel's binary size (`zeModuleGetNativeBinary`) is logged once with `ARC_ZE_KPROPS=1`, plus a warning by default when a hot kernel's binary exceeds the GPU's instruction cache. So contributors' logs show it without being asked, as spills do now.

A change that raises a budget must say so in its commit and in before-after.md, with the measurement that justifies it. That's the same rule as a KL change.

### 3.9 Submission and batched decoding

**How launches reach the GPU.** Three forms exist today:
- **The immediate command list:** each launch is submitted as it's made. It's the default, and the fastest on the 125H (1.51 µs a kernel).
- **Batched regular lists** (`ARC_BATCH=N`): launches collected and submitted N at a time. On by default for Alchemist.
- **Recorded lists replayed:** the barrier form, and the mutable form (`fix/replay-*`), measured as a gain on B-series only (B60 1.38 → 0.67 µs on Linux, 2.7 → 0.66 µs on Windows). The A770 was slower in every in-order form; the 125H was +3 % with replay.

These become one submission layer under `gemm::run` and the other kernels (`zero/submit.h`), chosen like a GEMM plan:
- by traits (integrated or discrete; the driver's support for mutable lists);
- then by a measured preset: B-series replay with barriers, Alchemist batched, iGPUs immediate.

`replay_probe`'s numbers decide the preset, and a new card runs the probe once and caches the answer like the tuning cache. Kernels never call the list API themselves. That keeps one place that knows how a token's launches are recorded, so replay works for every model and not only for Inkling's `fix/replay-inkling`. The host budget in 3.8 counts this layer's work too.

**Several sequences at once** (the server's slots, `--bench-batch`). Their rows stack into one `GemmOp`: M = the sum of the rows of all sequences in the step. One GEMM or GEMV launch then serves every sequence, and the planner's shape classes (T = 1, 2-8, 9-255) pick the kernel by the total rows. Two sequences decoding one token each take `gemv_mt` with T = 2, not two `gemv` launches. Expert tiles already gather rows across tokens, and they gather across sequences the same way. Only the attention stays per sequence, since each has its own cache, and it runs as one launch over the sequences with their offsets. Each sequence's results are the same as when it runs alone only for exact strategies whose sums don't depend on the other rows. Every exact strategy is like that (a row's sums use only that row), so batching never changes a sequence's output.

### 3.10 Turning hand-tuned kernels into one generic kernel

This is how oneDNN's GEMM generator and CUTLASS work: the kernel is written once, as stages, and every hand-tuned trick becomes a compile-time parameter of one stage. The stage parameters below are oneDNN's strategy fields (3.0): the register tile, K unroll, A / B access, prefetch, work-group, k-parallel, GRF, double buffering, remainders and the inner-product form. Each kernel form is written as one strategy string.

**The stages of a tile GEMM:**

| Stage | Hand-tuned choices it absorbs (today's switches) |
|---|---|
| Work split | sub-group tile M × N (`TM`, `XW_TM`, `XW_NC`), sub-groups per work-group (`CG`, `WGY`, `XW_WGX`), group order (`XW_WGL`, `XS_BG`) |
| Load A / B | plain or 2D block loads (`X2D`, `XW_2D`), through SLM or straight to registers (`XW_SLM`), packed layouts (`XW_PACKA`, `XW_PACKW`), the weight loader per type (3.4, ahead or fused) |
| Prefetch | distance and level (`X2D_PF`, `WPF`, `XW_PF`, `XW_PFD`, `XW_PFL`) |
| Pipeline | double buffering (`XW_DB`, `XW_DB2`), K step (`k64`, `XW_K128`) |
| Inner product | DP4a, DPAS 8 / 16 lanes, `dpasw` (`XS_DPASW`), both column halves in one sub-group (`X8W`) |
| Accumulate | register tile layout, the order of block sums (fixed: the reference order, section 4) |
| Epilogue | store, add, SwiGLU, situ, combine, softcap (`XW_EPI2`) |
| Register file | 128 / 256 (`ARC_GEMM_GRF`, ...) |
| Remainders | M / N / K edges: masked loads or a remainder form (today each kernel handles them its own way) |
| k-parallel | none (exact), or split-K in oneDNN's three forms: global with a second reduction, local inside the work-group, variable. Approximate unless the reduction keeps the K order (section 4) |

Written this way, each of today's hand-tuned kernels is one parameter set:
- `gemm_xw8`'s 140T form is "8-lane DPAS, `dpasw`, SLM for A, packed W, double buffer, tile 32 × 32, work-group 4 × 8";
- oneDNN's 125H kernel is "DP4a, A / B through SLM, tile 32 × 32, work-group 8 × 4, K 64, 256 registers".

The parameter sets become the catalog entries in 3.5.

**The conversion, one kernel at a time:**
1. **Take the best hand kernel of a family as the base**: gemm.cl for DP4a, gemm_xw8 for 8-lane XMX, gemm_xw for 16-lane. Don't start from a clean rewrite.
2. **Lift one trick at a time into a stage parameter.** After each lift, compile the generic kernel with the hand kernel's parameters and compare three things:
   - the output, bit for bit (`gemm_check`);
   - the code: instruction count, send messages, spills and private memory per family, with ocloc;
   - the speed, within about 2 %, on a card of that family.
   A lift that changes the code is reworked until it doesn't.
3. **Tricks that don't fit a parameter become a stage variant**: another implementation of that one stage, chosen by a define. A register layout only `dpasw` needs is an example. The rest of the kernel stays shared.
4. **Delete the hand kernel** only when the generic one matches it on all three counts for every form the catalog uses.
5. **Explore combinations the hand kernels never tried**: the tuning cache's candidates. For example, gemm_xw8's SLM staging with the DP4a inner product is the oneDNN shape for the 125H.

**Pitfalls, from this project's history:**
- The compiler (IGC) is sensitive to how code is written: a loop that is a macro in one kernel and an inline function in another can compile to different code. Compare the ISA, not only the speed.
- No private arrays indexed at run time. `decode_w8_sb` keeps up to 8.7 KB a thread in private memory on the A770 (hansong80, #45, 2026-11-24).
- Unroll only the register tile (3.8). Keep loads uniform across the sub-group, so the compiler doesn't hoist a whole loop's loads into registers. Hoisting is how the first Q-block attention spilled 495-929 B.
- The XMX families can only be measured on contributors' cards. Their conversions wait for those runs. On the laptop, only the DP4a kernel can be converted and measured.

### 3.11 Fusion differs per card, so it's part of the plan

Which kernels are fused together already depends on the GPU, and the measurements show why:

| Fusion today | Where it's on | Measured |
|---|---|---|
| Gate and up in one XMX GEMM (`xw_gu`, `ARC_GEMM_XW_GU`) | discrete cards; off on iGPUs | `xw_gu_env`, xe2/engine_xw.h:239 |
| The norm and the H8 rotation in one kernel (`rmsnorm_h8r`, `ARC_H8_NORM`) | discrete cards and the 140T preset; off on iGPUs and on a Data Center Max | 140T 27B 4K 368.3 → 382.1 tok/s (#5); B70 2,890 → 2,972 (#27); B60 no gain before its 128-register default (engine_h8.h:306) |
| One H8 activation shared by the GEMMs that read it (`ARC_H8_REUSE`) | everywhere except a Max | engine_h8.h:242 |
| The XW double buffer (`ARC_GEMM_XW_DB`) | Alchemist, only for 32 × 16 tiles | A770: 32 × 16 GEMMs -4 to -10 %, but 16 × 32 ones +72-139 % (#45) |
| SwiGLU's output straight into the down GEMM's rotation (`ARC_H8_SWI`) | where H8 rows take the down projection | engine_h8.h:283 |
| Gate and up in one GEMV launch (`ARC_GEMV_GU`) | off (unmeasured gain) | engine.h:1454 |

So fusion can't be a fixed property of the code. The same fusion helps one architecture and costs another, sometimes even between two shapes on one card (the A770's double buffer). The design keeps those per-card choices, but as data:
- **A layer is a small chain of steps:** norm → quantize / rotate → GEMMs → activation → GEMM → residual add. The model file describes its chain once, and it's the same for every GPU.
- **Each architecture's catalog (3.5) has fusion rules:** which neighboring steps run as one kernel, for which shape class. For example, Xe2-HPG: "norm + rotate: fused; gate + up: one GEMM; down's input from SwiGLU: fused". Xe-HPG: "double buffer for 32 × 16 tiles only". Each rule names its measurement, as the table above does.
- **Card exceptions stay possible:** a rule keyed by device ID (the Max's "no norm fusion") sits in `presets.h`, with its measurement. They are kept, as table rows rather than `if` branches spread through engine_h8.h and xe2/.
- **Every fused form must equal its unfused chain bit for bit,** as today's do (the comments above say "bit for bit", "hashes equal"). `gemm_check` tests each fusion rule against the unfused steps, so a fusion choice can change speed per card but never a model's output.
- **The tuning cache can try a fusion on or off for a shape,** like a tile, since both forms are exact. That's how the A770's "32 × 16 yes, 16 × 32 no" would have been found automatically.

### 3.12 Fusion compiled at run time (JIT)

Kernels are already compiled at run time: OpenCL C through the driver, with `-D` options, cached in `%USERPROFILE%\.inferzero\kernels` (zero/clrt.h). So fused kernels can be generated for each layer chain when a model loads, instead of being written by hand for each combination. This section makes fusion the center of the design: the GEMM is one stage of a generated kernel, not a fixed kernel with switches.

**1. The layer chain (a small graph).** Each model file describes its layers as a chain of steps, once:

| Step | Examples |
|---|---|
| `Norm` | RMSnorm, gated norm, Gemma's norms |
| `Quant` | int8 per 32 or per 256, the H8 rotation |
| `Gemm` | one or a group of weights reading the same input (q / k / v, gate + up, Inkling's eight) |
| `Act` | SwiGLU, GELU-mul, the sigmoid gate, Inkling's situ and controller update |
| `Add` | residual add |
| `Combine` | the experts' weighted sum |
| `Softcap`, `Store` | the head, the output |

**2. Fusion rules: what may join.** Legal by construction:
- **Element-wise steps after a GEMM go into its epilogue:** `Act`, `Add`, `Combine`, `Softcap`.
- **Quantization in the producer's epilogue.** A GEMM's tile can write its output already as int8 with per-32 scales for the next GEMM, when its tile covers whole 32-value blocks of a row (N a multiple of 32). This removes most `quant` / `*_quant` launches, and it gives the same values as quantizing afterwards: each block's max and its rounding see the same 32 numbers.
- **A norm in the next GEMM's prologue,** when the row's statistics are ready. The sum of squares needs the whole row, so it is either a tiny pass of its own or computed in the previous row-wise step (the residual add), in the same order as today's `rmsnorm_quant`. Never with atomics, whose order changes.
- **Not fused:** across layers (no persistent kernel); through a reduction whose order would change; attention and DeltaNet internals (3.6).

**3. Choosing what to fuse: per architecture.** The catalog's fusion rules (3.11) say which legal fusions to use per shape class. Where no rule exists, the tuning cache times the fused and unfused forms on first use, since both are exact. That's how "XW double buffer: 32 × 16 yes, 16 × 32 no" on the A770 would be found without a hand rule.

**4. Generating the kernel.** Unlike oneDNN, which writes machine code through nGEN, we generate OpenCL C and let IGC compile it (3.0 gives the reasons; a machine-code backend for hot strategies stays an option for later). Each step is a function in a stage library, `zero/kernels/gemm/stage_*.cl`: the norm, the quantizer, the tile loads, the inner product, each epilogue. That's the same code the unfused kernels use. The generator writes a short kernel that calls the chosen stages in order, with the catalog's parameters as constants. Only structure is compiled in: types, tile, fusion, epilogue. Sizes (M, offsets) stay run-time arguments, so one kernel serves every chunk length.

**5. Compiling, checking, caching:**
- **Key:** a hash of the generated source, the options, the device and the driver. Today's kernel cache stores the binary under that key.
- **At load:** the planner knows the model's whole set of chains, so every fused kernel the model needs is generated and compiled in parallel while the weights load, with a `progress:` line. The first start is slower; later starts read the cache. mejoudeh measured a slow first start on .04 (serverz with Gemma: 116 s against 13 s), so compile time gets a budget of its own in 3.8, and the launcher can warm the cache in the background.
- **Checks before first use**, each one falling back to the unfused chain:
  - spills, private memory and binary size from `zeKernelGetProperties` (3.8);
  - a bit-for-bit comparison of the fused kernel against the unfused steps on a few real rows of the model, under the GPU watchdog (`ARC_GPU_WAIT_S`), like today's start-up checks (`xw_check`, `xmx_check`).
  The result is stored in the cache, so a failed fusion isn't retried on that driver.
- **Barriers:** generated code uses only the barriers inside known stage functions, never new ones. A hung kernel froze the laptop on 2026-11-23, and new barrier kernels need the watchdog.

**6. What it gives:**
- Fusions the hand code never had, for every model at once: quantization in the producer's epilogue, every model's gate + up, the norm into the GEMM on every architecture where it measures as faster.
- Fewer launches without hand-written fused kernels. Inkling's 1,356 launches a token shrink mostly by removing separate norm / quant / add / gate launches.
- Per-card fusion as data and measurement: no fused-kernel files per card.
- Exactness unchanged: the stages are the same code as the unfused kernels, in the same order, and every fused kernel is checked against them before use.

**Limits and risks:**

| Limit | Answer |
|---|---|
| Compile time on first start | parallel compiles at load, a compile budget, the launcher warming the cache |
| A driver miscompiles a generated kernel | the bit-for-bit check before use, the fallback, and the stored "bad" mark |
| Harder to read than a hand kernel | `ARC_GEMM_PLAN=1` prints each chain's plan, and `ARC_FUSE_DUMP=dir` writes the generated sources for review and for ocloc |
| The offline spill check (3.8) can't know every generated combination | it compiles the combinations the shipped models use for each family; the run-time check covers the rest |

### 3.13 Looped layers (Inkling 2), and how Basalt runs them

Inkling 2 Mini runs its core layers [6, 18) three times a token: 6 + 36 + 6 = 48 layer visits. The controllers run on the 24 visits of passes 2 and 3 (`pass > 0`). Today it's slow:
- the B60: 24.6-31.5 tok/s generation, ~315 tok/s prompt;
- Basalt (MIT, CUDA) on an RTX 5060 Ti of about the same bandwidth: 64.8 / 1,803 tok/s.

The 125H's prompt profile shows `moe_down` at 46 % (`moe_rg`), `ink_attn` 15.5 % and `ink_ctrl_fused` 12.8 % (10.2 ms for a product oneDNN does in 0.06 ms). On the B60, `ink_mix` is 8.7 %.

What follows is from a read of Basalt's code on 2026-11-25 (github.com/jesdga95/basalt: `src/models/inkling2/session.cpp`, `walk.cpp`, `pack.hpp`, `kernels/glue.cu`, `prompt.cpp`).

**How Basalt runs the loop, and what we take:**

| | Basalt | Ours today | In this design |
|---|---|---|---|
| **Planning and replay** | one CUDA graph per window shape (rows, key bucket, head), covering all 48 visits unrolled (session.cpp:1523-1559, 1614-1657). There is no graph per pass and no graph update: each visit has its own cache slot (pack.hpp:15-17, 74-77), so every pointer is fixed in the graph. Only positions and tokens change, through a device buffer, with the key bound rounded up to a power of two | 1,356 launches a token, one at a time; `fix/replay-inkling` records with mutable arguments (B-series) | the plan is made once per visit at load. Visits of one physical layer share the weights' plans and differ only in their slot's pointers. On B-series, a token is recorded once per shape bucket, as Basalt does, with positions in a device buffer, so a replay needs no argument updates. On iGPUs the immediate list stays (replay measured slower on the 125H) |
| **The bank and messages** | at a pass's first core layer, one `mix_norm` launch mixes the 2 bank states for all 12 layers × 4 streams = 48 queries at once (session.cpp:1632-1641). Each block recomputes the states' RMS, the query dots and the softmax (glue.cu:111-168, 244-286), so the RMS is repeated 48 times per row inside that one launch. Its prompt path mixes per layer (prompt.cpp:514-520) | `ink_mix` for each later visit's message, plus the attention and FFN mixes per visit | decode: the message mix is a prologue (of the controllers' down), with the RMS recomputed per work-group as Basalt does. Repeating it is cheap at one row. Prompt: the bank is fixed for a whole pass, so its RMS per row is computed once per pass and kept, instead of in each of the 12 layers' mixes. That's a step Basalt doesn't take |
| **The pass boundary** | no copies: buffer roles alone (walk.cpp:21-44, "a forward copies nothing"). Each pass reads the prelude's blocks and output again | `ink_copy` for the prelude and the bank | buffer roles, as Basalt; the copies go |
| **Controllers** | 3 kernels a call: down (`[rms(dest) \| rms(msg)]` → 64, the RMS scales inside), up (`[g \| f]`, 2 × 2304 outputs in one product), blend + RMS norm + q8_1 quantization (glue.cu:290-472). The q / k / v controllers are one call (streams on `gridDim.y`), the FFN's another, so 6 kernels a later visit. The projection then reads the three blended inputs in one segmented launch (`native_q8_planes_mmvq_segments`, session.cpp:1257-1267) | 4 controllers × 3 launches (`ink_rms2`, `ink_ctrl_down`, `ink_ctrl_fused`) = 12 a later visit, and separate k / v products | 4 a later visit: [down, grouped over q / k / v] · [up as one `[g \| f]` product, grouped] for q / k / v, and the same two for the FFN. The blend, its RMS and the quantization go into the next product's prologue, one per work-group, since `g` and `f` are complete by then. The projection takes 3 inputs in one GemmOp (a segmented group). In a prompt, down and up are real GEMMs through the shared layer, replacing `ink_ctrl_fused`'s 10.2 ms |
| **Experts across passes** | no reuse of choices or weights between a token's passes: every visit routes again (session.cpp:1391-1412). Each visit's first 8 experts are recorded, and the next window's same visit reads them ahead into L2 on a side stream (ops.cu:493-496; session.cpp:1338-1346). At 1-2 rows, the next visit's router runs early on this visit's MLP input, only to prefetch (:1396-1404). The kernel is chosen by the layer's weight type (`native_expert_layout`): layers 0-19 NVFP4, 20-23 MXFP8 | one expert path chosen for the model; Q8_0 / BF16 layers 20-23 can drop the whole model to per-tile (plan-2026-11-29, item 1) | the expert strategy is planned per layer from its weight types (3.5), never per model. Prefetch only where experts stream (discrete cards, I1.7): in visit order, from the previous window's same-visit record or an early router run. Prefetching never changes which experts are used, so it's not a routing bias |
| **Keeping data hot across passes** | no L2 persistence setting. `mix_norm` has an extra block that prefetches each visit's older KV cells and conv-ring entries into L2 (glue.cu:214-242). Every small kernel loads its weights before waiting on the previous kernel (programmatic dependent launch) | none | KV / conv prefetch as a stage of the mix prologue, measured per family. Level Zero has no dependent-launch equivalent. On B-series, replay removes the launch gaps instead |
| **Rows** | draft / verify rows go through all passes together: up to 16 rows a window (`kDecodeRows`), products 8 columns a launch | verify passes cost 1.3-1.5x a plain token on the B60 | a window's rows are one GemmOp's M through every visit (3.9), with `gemv_mt` for 2-8 rows |

**Basalt isn't at the floor either.** From its code it runs ~9-10 kernels a visit, plus 6 controller kernels on later visits: an estimate of ~480 a token, not measured. Its speed comes from three things:
- the graph, which removes launch gaps;
- dependent launch, which overlaps the next kernel's weight loads;
- fast kernels: the 8-way fused projection, FP4 / MMQ prompt GEMMs, and the expert kernels.

The kernels are what we can take everywhere; the launch gaps only matter on discrete cards.

**The floor per visit, and where each launch goes** (today's counts from the code read, 3.7):

| Visit | Today | Floor | What closes the gap |
|---|---|---|---|
| Sliding, no messages | 19 | 6 | the two mixes and their two norms into prologues (−4, ways 2 / 5); sink into the grouped projection (−1, way 1); the K / V convs into the epilogue (−2, way 8); `ink_rows_kv` into the attention's prologue (−1); `quant_rows` into o's prologue (−1); route and `moe_quant` into the experts' prologue / epilogue (−2); `ink_moe_out` into the next visit's prologue (−1); the block copies by buffer roles |
| Global (layers 5, 11, 17, 23) | ~25 | 6 | the above, plus the 3 query norms and the separate k / kcp / v / vcp products into the one grouped projection |
| A later-pass visit (controllers on) | ~38 | 10 | the above, plus the controllers 12 → 4 and the message mix into the controllers' prologue |
| A token | 1,356 | 385 | |

As elsewhere, the gain on the 125H is mostly fewer passes over the rows, though launch cost counts too (~7 ms of the gap at ~7 µs a launch). The biggest items for the laptop are the kernels themselves:
- `moe_down` through the shared expert GEMM;
- the controllers as real GEMMs in prompts;
- the query-block attention (3.6), with `ARC_INK_ATTN_QB` still to measure.

### 3.14 Fusion as a goal: the bytes between kernels

Fewer launches (3.7) matters most in generation. In prompts, what costs time is each pass over a T × E activation: one f32 copy of the 27B's hidden state at 4K is 4,096 × 5,120 × 4 = 84 MB. So fusion is a goal of its own, with the activation-bytes budget (3.8), not only a way to fewer launches.

**Two rules for prompt fusion:**
- **A step that needs a reduction over the whole row (a norm, Inkling's mixes) doesn't go into the next GEMM's prologue in a prompt.** There, every column tile of the GEMM would read the f32 rows again and redo the reduction. Instead it joins the pass that already reads or writes those rows: the producer's store, or one combined norm + quantization pass. At decode (one row) the prologue form is fine (3.7, way 5).
- **Element-wise steps** go into the producer's epilogue or the consumer's load (way 8) in prompts too.

**Exactness.** Skipping an f32 store and reading the value back gives the same bits: an f32 store is lossless. So a fusion that computes the same operations in the same order is exact. The catch is `-cl-mad-enable` (all our builds): when a multiply and an add that used to be split by a store end up in one kernel, the compiler may fuse them into one rounding, and the bits change. The project already guards against that where it mattered (`ops_inkling.cl:233` reads the stored value back as volatile; gemm_xw8.cl:760 is the split-K case). Each fused form therefore keeps the reference's rounding points explicitly, and `gemm_check` and the dump-logits hashes prove it. A fusion that changes the operations is approximate and goes through the KL rule. Example: quantizing x·w per block without the RMS and applying 1/RMS as a row scale afterwards (way 5).

**The prompt fusion list, by bytes saved.** These are estimates from the shapes, T = 4,096, as DRAM bytes if none of it stays in cache, so they're upper bounds:

| Fusion | Today | Saved per 4K chunk | Exact? |
|---|---|---|---|
| Gate and up as one GEMM, with SwiGLU / GeGLU / situ and the int8 quantization in its epilogue | the gate GEMM writes f32 gate, the up GEMM reads it and writes f32 silu(g)·u, `swiglu_quant` reads that and writes int8 (engine_qwen.h:491-523) | 27B: ~1.14 GB a layer, **~73 GB** | yes, with the rounding points kept |
| Inkling's and the 35B's expert gate / up: situ / SwiGLU and the quantization in the expert GEMM's epilogue (no `moe_quant`) | the expert gate / up writes f32 rows (T × 8 × F), `moe_quant` reads and quantizes them | Inkling (F 1,152, 48 visits): **~14.5 GB** (`moe_quant` 241 ms on the 125H); 35B (F 512): ~5.4 GB | yes |
| DeltaNet's conv, SiLU and q / k l2-norm in the recurrence's load (it walks tokens in order per head, so the causal conv fits there, with no halo between GEMM tiles) | `gdn_conv` reads the qkv rows and writes them back convolved | 27B: ~0.34 GB a layer, **~16 GB** | yes |
| Residual add + next norm + quantization in one pass | the GEMM's epilogue adds the residual and writes x (f32); `rmsnorm_quant` reads x again | 27B: one 84 MB read per norm, 128 norms, **~10.7 GB**. Exact form: the producer's epilogue writes x and a per-row partial sum of squares; the last work-group of a row block (way 7) computes the RMS and quantizes the block from L2, where it was just written. The faster RMS-free form is approximate | exact form: yes |
| The gated norm + quantization in the recurrence's epilogue (a work-group owns its head) | `gdn_recur` writes f32, `gated_norm_quant` reads it | 27B: ~0.2 GB a layer, **~9.6 GB** | yes |
| The attention's output gate + quantization in its own epilogue / combine (Basalt writes q8_1 there) | the attention writes f32, `attn_gate_quant` reads it and the gate | 27B: ~0.2 GB a layer, ~3.2 GB | yes |
| Inkling's mixes + norm + quantization as one pass, the post-convs in the epilogues | `ink_mix` writes f32, the norm reads it; `ink_conv1` after o | ~3.6 GB each | yes |
| The MoE gather in the expert GEMM's A load (already: the row map `stok_`), the combine with its row weights in the next pass | `moe_combine` reads the T × 8 × E outputs and writes x | small: the 8 partial rows are read either way; it saves the x write and read (~68 MB a 35B layer) | yes (fixed expert order) |

Total for the 27B: ~110 GB a 4K chunk. On the B60 (456 GB/s) that's at most ~0.25 s of a ~2.3-3.8 s chunk, 7-10 %. On the 125H (~100 GB/s) it's ~1.1 s. Prompts stay GEMM-bound, so the oneDNN-style tile (6, step 7) is still the larger gain, but fusion is the next one. The first three rows are the big items for every model at once.

### 3.15 Multi-GPU: the tensor split and the layer split in the same layer

The shared layer is designed for several GPUs from the start, so the split isn't built twice. Today's multi-GPU code becomes rows in the route table (6.0), not a second implementation:
- the tensor split: `zero/mgpu/engine_tp.h`, `engine_tp_shard.h`, `tp_mirror.h`, `zero/kernels/tp.cl`;
- the layer split: `zero/mgpu/engine_pp.h`;
- their designs: docs/multi-gpu.md, docs/tiered-placement.md.

**1. The tensor split: a GemmOp carries its sharding.** Every GPU keeps the whole residual stream and computes the norms itself, as today (multi-GPU.md, "How it is split"):

| Sharding | Products | What follows |
|---|---|---|
| Column split (N) | q / k / v with their kv-head groups; DeltaNet's qkvz / ab by k-head with the v-heads that read them; gate / up by FFN column; the head by vocabulary rows | nothing: each shard's output is complete for its columns (the logits are joined on the host) |
| Row split (K) | `attn_output`, `ssm_out`, `ffn_down` | partial sums, added over the GPUs before they reach the residual stream |
| Expert split | the routed experts `[e0, e1)` per GPU (the 35B, Flash Next, DeepSeek); every GPU routes every token itself, and only its own (token, expert) pairs get tiles (`tp_moe_mask` / `tp_moe_scatter` / `tp_moe_tiles1`) | the experts' parts, plus the shared expert on GPU 0, summed (`tp_moe_combine`; for hc models, then `tp_reduce_hc`) |

**The device sum is a post-op kind of its own,** after the row-split and expert-split products. Its forms are today's, chosen by the submission layer (3.9) per box:
- **generation:** the parts copied to pinned host memory, a barrier, and each GPU adding every part in rank order (`tp_add`); with `ARC_TP_EVSUM`, events instead of the host barrier; with `ARC_TP_P2P`, direct copies where the kernel allows peer access;
- **prompt chunks** (64 rows or more): the f16 reduce-scatter (`tp_half`, `tp_rs_sum`, `tp_rs_add`), in pieces (`ARC_TP_PIPE`).

The sum fuses with its neighbours the same way as one GPU's steps:
- `tp_half` goes into the producer's epilogue;
- `tp_rs_sum` joins the residual + norm + quantization pass (3.14);
- at decode, `tp_add` goes into the next product's prologue, still adding the parts in rank order.

The floor (3.7) adds no kernel for a sum at decode, only its copies. A prompt chunk keeps one launch a sum (`tp_rs_sum`). The floor table gets a tensor-split column: sums per layer × the copy commands each one costs. That's 2 sums a layer for the Qwen models (the mixer's output and the FFN's, or the MoE's), and 1 for Flash Next and DeepSeek, whose attention isn't split. The launch and activation-byte budgets (3.8) count per shard plus the sums.

**2. The layer split: the planner puts steps of the layer chain on devices.** Each stage is a run of whole layers on one GPU (today GPU 1 first, GPU 0 last with the norm and head). A stage boundary is an activation copy (through host memory today, ~20 KB a token for the 27B). Rules:
- **No fusion across a stage boundary.** The fusion rules (3.11, 3.12) see the boundary as a store.
- **A looped model keeps a physical layer's passes on one device.** Inkling's three passes over a core layer share its weights, and each pass has its own cache slot (3.13), so the slots stay with the layer. The stage cut falls between physical layers, never between passes.
- The expert cache, RAM tier and `--ctx auto` per stage stay as today.

**3. Per-device catalogs.** Each GPU of a mixed box (a B60 + an A770, #57 / #58) plans with its own family's catalog, tuning cache and presets: the same tuned choices as when it runs alone. In a layer split each stage already compiles for its own GPU (`rcl::device_for`).
- **The tensor split stays one kind for now:** its shards share GPU 0's kernels.
- **What the design adds:** exact strategies give the same bits on every family (section 4). So a tensor split over unlike cards could later let each shard run its own family's exact strategy, and still produce the same partial sums. That's only possible, not planned. The layer split is the route for unlike cards.

**4. Overlap.** The sums beside the next chunk's work (`ARC_TP_OVERLAP`, the layer-major segments) and the hand-off's overlap belong in the submission layer (3.9), per device. Kernels never manage their own overlap.

**5. Exactness, per split** (from multi-GPU.md's checks):

| Split | Against one GPU | Against today's split |
|---|---|---|
| Tensor split, generation (f32 sums) | **not bit-identical**: the parts are added in another order than one GPU's single sum. KL 0.0007-0.001, top-1 the same | the same bits, as long as each part is computed by an exact strategy and the parts are added in rank order. That's what `ARC_TP_P2P`, `ARC_TP_PIPE`, `ARC_TP_OVERLAP` and `ARC_TP_EVSUM` keep today |
| Tensor split, prompts (f16 reduce-scatter) | approximate: KL 0.0016-0.0018 mean, 0.0035 max on the 27B (`ARC_TP_SUM=f32` gives 0.0009-0.0012) | the same bits, with the same rule |
| Layer split, cards of one kind | **bit for bit** (the same kernels on the same rows in the same order) | the same |
| Layer split, unlike cards | a KL check (each stage runs its own forms) | the same; and with exact strategies on both stages, bit for bit against one GPU |

So "exact" in a split means equal to today's split of that kind, and the split's own KL against one GPU stays measured as now. Batching several sequences stays exact per shard, as on one GPU (3.9). The checks run on the laptop under `ARC_TP_SIM=1`, as today.

**6. Route-table rows** (6.0), each moved and deleted like the others:
- the device sums (`tp_add`, `tp_half`, `tp_rs_*`, the P2P path);
- the shard setup (`engine_tp_shard.h`);
- `tp_mirror.h`;
- the layer split (`engine_pp.h`);
- the expert split (`tp_moe_*`).

They're measured on the owners' boxes (manoreken2's 4 B60s, Daleang's 2 B70s, MaineTim, skipzx-lab's B60 + A770, hansong80's 2 A770s, catduckgnaf's 2 B50s). Every request asks for 1, 2, 3 and 4 GPUs.

**7. Not in scope:** the tensor split and the layer split combined in one model (a hybrid). The design supports each on its own, as the user decided earlier.

### 3.16 Generation: the same oneDNN design and fusion for one row to 16

Generation gets the same treatment as prompts: catalog strategies and fusion. That covers one token, the verify passes of drafting (2-16 rows), and the server's batched sequences.

**1. GEMV and small M are catalog strategies, not separate kernels.** oneDNN's evaluator (3.0) turns to k-parallel forms when M × N gives too few threads to fill the GPU, which is the GEMV case. In this design these are all strategies of the shared layer:
- `gemv`, `gemv_v`, `gemv_mt`, `gemv_mt_v`, with today's `-DGU2`, `-DSPLITK` and `-DQ2F` forms;
- the gate + up GEMV (`gemv_gu`);
- the head's GEMV.

They're chosen by the shape classes (T = 1, 2-8, 9-16), with strategy strings, catalog entries per family and a fitted model. Their target is different from a prompt's: a GEMV is bound by memory, so its model and its budget measure the **share of the measured bandwidth** per weight type, not TOPS. The B70 reaches 88 % of its measured 590 GB/s on the 27B (xe2-tuning.md:871). The Arc 140T (a Core Ultra 9 285H) reaches ~84 % of its memory's on the 27B (#5, release 2026.11.18.07, site/results.html). Split-K over K stays approximate, as in section 4, unless its parts are added in K order.

**2. Generation fusions.** At one row the activations are tiny, but the KV cache, the partial sums and the launches aren't:
- the norm + quantization in the GEMV's prologue, recomputed per work-group (way 5);
- gate + up as one GEMV with SwiGLU / GeGLU / situ in its epilogue; the residual add in o's, ssm_out's and down's epilogues (already there through `tp_out`);
- **the KV store in the q / k / v epilogue:** K and V written straight into the cache, with RoPE, the q / k norm, and Inkling's clips and K / V convs. This needs a tile that covers whole heads, otherwise these steps go into the attention's prologue. Either way `attn_prep`, `ink_rows_kv` and `ink_conv1` K / V get no launch of their own;
- the expert GEMVs with the routing weights and scales in their epilogues; `moe_quant` into gate / up's epilogue; the combine into the next prologue;
- DeltaNet as in 3.6.

**3. Decode attention is a microkernel client** (3.6), as oneDNN's SDPA is:
- **Split-K over key slices** (flash decoding), with the combine done by the last work-group (way 7) in a fixed order. Basalt sizes the slices from each row's own key count, so a row never depends on the other rows in the call. We do the same.
- **The q·K and P·V products** use the stage library's tile functions: XMX where the family has it and a slice is long enough, DP4a / f16 otherwise.
- **Fused in:** the bias, sink, softcap and window mask (Inkling's relative bias, Gemma's sliding window); the output gate; and the output's quantization for o in the combine (Basalt's q8_1 there).
- **The KV format** (f16, int8 as Basalt's `--kv int8`) is a strategy parameter. int8 changes the numbers, so it's approximate and needs each model's KL check over long generations.
- **One kernel serves all rows:** one token, the verify rows (T up to 16) and the batched sequences, each with its own cache offsets.

The new split decode attention (`fix/ink-attn-dec`) is the first form. It's still waiting for its KL check against f32.

**4. Budgets for generation** (3.8), printed by `ARC_GEMM_PLAN` and kept in `launch_budget.txt`:
- launches a token, against the floor table (3.7);
- each GEMV strategy's share of the measured bandwidth, per weight type and family;
- the KV bytes a token at a given context, per KV format.

### 3.17 Inkling 2 on the new design, end to end

This pulls together 3.6, 3.7, 3.13, 3.14 and 3.16 for one model. Inkling 2 Mini, the UD-IQ4_XS file:
- 24 layers, E 2,304, 24 q / 8 kv heads × 128, no RoPE;
- a window of 1,024, with layers 5, 11, 17 and 23 global;
- 128 experts with 8 used, F 1,152;
- the core [6, 18) runs 3 times: 48 visits a token;
- the experts are NVFP4 in layers 0-19, and Q5_K / Q5_0, Q4_K / Q5_0 and IQ4_XS / IQ4_NL in layers 20-23.

**The model file's chain for one visit.** What the model file says; the planner decides the rest:

| Step | GemmOp / kernel | Prologue | Epilogue |
|---|---|---|---|
| 1 | **projection:** q / gate / rel / sink / k / kcp / v / vcp as one grouped GemmOp (3 inputs on a later pass, a segmented group) | the attention mix over the blocks + RMSnorm + int8; on a later pass, the controllers' blend for q / k / v | the KV store: clips, norms, K / V convs (taps from kcp / vcp on global layers), written into this visit's cache slot |
| 2 | **attention** (decode: split over key slices; prompt: the query-block form) | anything step 1's tile couldn't cover (whole heads) | gate; quantized for o; way 7 combine at decode |
| 3 | **o** (row-wise, + residual) | — | post-conv + residual; the block push by buffer role (no copy) |
| 4 | **router** (f32, 128 × 2,304) | the FFN mix + RMSnorm; on a later pass, the FFN controller's blend | — |
| 5 | **experts' gate / up** (ExpertTiles) | top-k from step 4's logits, the gather by row map | situ, the gate-up scale, int8 quantization (no `moe_quant`) |
| 6 | **experts' down** (ExpertTiles; the down scale folded into the weight) | — | — (the combine with row weights, the FFN conv and the residual go into the next visit's prologue in a fixed expert order) |
| later pass only | **controllers:** [down for q / k / v grouped, K = 2E → 64] · [up as `[g \| f]`, grouped]; the same two for the FFN controller before step 4 | rms(dest), rms(msg), the message from the bank mix | — (the blend goes into step 1's or step 4's prologue) |

That's 6 launches a visit, plus 4 on a later pass. At a token: 24 × 6 + 24 × 10 + 1 (final mix + norm → lm_head, the logit mask in its epilogue) = **385**, against 1,356 today.

**The strategies it hits, per family.** These are proposals from 3.5 / 3.10; each is chosen by the catalog and the model:

| Step | 125H (Xe-LPG, the laptop) | B-series (Xe2-HPG) | A770 / 140T (8-lane XMX) |
|---|---|---|---|
| Projection, prompt | `tile_dp4a`, oneDNN's Xe-LPG entry (32 × 32, work-group 8 × 4, k64, 256 GRF) | `tile_xmx16` / `wide_h8r` | `tile_xmx8` / the xw8 form with `dpasw` |
| Projection, decode / verify | `gemv` (T = 1), `gemv_mt` (2-8) | the same, bandwidth-judged | the same |
| Experts, prompt | `tile_dp4a` with ExpertTiles and the fused loader: NVFP4 exact per 16 (Q8_16); Q5_K / Q4_K / IQ4_XS through W8 / Q8_0, chosen per layer, never per model | `tile_xmx16` with ExpertTiles | `tile_xmx8` with ExpertTiles, the big-tile form (today's `gemm_x8` tiles lose to DP4a on the A770) |
| Experts, decode / small chunks | `expert_small` (`moe_gateup_q` / `moe_down_q` with GU2 at T ≤ 8) | `expert_small` | `expert_small` |
| Controllers | decode: `f32_small`; prompt: the tile strategy as real GEMMs (today `ink_ctrl_fused`: 10.2 ms against oneDNN's 0.06 for the shape) | the same | the same |
| Attention | decode: the split form on DP4a / f16 tiles; prompt: the query-block form (`ARC_INK_ATTN_QB`, still to measure) | XMX tiles for long slices | XMX8 tiles |
| Submission | immediate | the token recorded once per shape bucket and replayed (3.13) | measured per card (the A770 was slower in every in-order form) |

**Its fusions and what they save** (3.14), at a 4K prompt:
- the experts' situ + quantization in the epilogue: ~14.5 GB (today `moe_quant` takes 241 ms on the 125H);
- the fused weight load instead of `decode_w8`: 3.7 s of the 25.6 s 4K prompt on the 125H (14.6 %);
- the mixes + norms + post-convs: ~3.6 GB each;
- the controllers as real GEMMs: ~2.4 s of `ink_ctrl_fused`.

**What the laptop's 4K prompt would lose,** from the 2026-11-24 profile (pp4096 at 160 tok/s, 25.6 s):

| Kernel today | Time | Its new route |
|---|---|---|
| `ink_attn` | 8.5 s | the query-block attention on tile functions |
| `decode_w8` | 3.7 s | removed by the fused loader |
| `moe_gateup` | 3.3 s | the ExpertTiles DP4a tile |
| `ink_ctrl_fused` | 2.4 s | GEMMs |
| `moe_down` | 1.8 s | the ExpertTiles DP4a tile |
| `moe_combine` + `moe_quant` | 0.5 s | prologue / epilogue |

No speed-up is promised here. Each step is measured as its route-table row moves (6.0).

**The route rows that retire its kernels** (6.0):
- `ink_mix`, `ink_rmsnorm(_quant)`, `ink_rms2(_quant)`, `ink_conv1`, `ink_rows_kv`, `ink_copy`, `ink_moe_out` → prologues and epilogues;
- `ink_qkv_stack`, `ink_sink`, `ink_o`, `ink_mm_f32(_sg)`, `ink_ctrl_down`, `ink_ctrl_fused` → grouped GemmOps on catalog strategies;
- `ink_attn` (`_pp`, `_ps`, `_qb`, the decode form) → the attention on tile functions;
- the `ARC_INK_*` switches → planner rules;
- `moe_rg`, the per-tile path and `moe_gemm`, as used by Inkling → ExpertTiles.

Each row needs its bit-for-bit or KL check: Inkling's KL ≤ 0.015 and the same top token ≥ 98 % excluding near-ties, over hundreds of positions, with the maximum reported.

## 4. Exactness: how a shared GEMM keeps models apart

**The reference order.** For every weight type the layer defines one way to compute an output:
- the int8 × int8 products of a block summed in int32, which is exact in any order;
- each block's sum multiplied by its scales in one fixed form;
- the blocks added in K order in f32;
- the epilogue applied once at the end.

gemm.cl's DP4a tile already works this way, and XMX already matches it bit for bit (gemm.cl:5-6, gemm_x8.cl:4, the `xw_check` family). So the reference is the code we have, not a new one.

**Exact strategies.** Any tiling, work-group shape, register file or XMX form that keeps the reference order is `exact`. It gives the same bits as every other exact strategy, so the planner may switch between them freely, and a faster exact kernel needs no KL run.

**Approximate strategies** carry `exact=false`:
- split-K (ARC_GEMM_SPLIT, engine.h:1653);
- GEMV split-K (engine.h:1484);
- the f16 XMX GEMM (gemm_x16.cl:17);
- H8's pieces (engine_h8.h:548);
- the NVFP4 per-32 fold;
- the padded-F expert GEMM (engine_moe.h:536).

A model uses an approximate strategy only if its model file lists it, for example `allow = {split_k, h8_pieces}` in engine_qwen.h, after that model's KL check. So a new approximation can't reach another model silently. Today's defaults become each model's initial allow-list, so nothing changes on day one.

**Accuracy classes: the activation precision a model needs.** Exactness (above) says whether a strategy keeps the reference order. A second rule says whether the reference itself is precise enough for a model. Gemma 4 is the case that forces it. Its KL against f32 over 480 tokens (Gemma 4 12B, before-after.md 2026-11-16):

| Activations | KL |
|---|---|
| two int8 passes | 0.00001 |
| one int8 pass | 0.064 |
| one H8 pass | 0.155 |
| `REF_HAD` (the reference with a Hadamard rotation) | 0.549 / 0.267 |

One int8 rounding of Gemma's activations is too coarse, whatever the scales. Today scattered guards keep it off by hand: `!hp.gemma` at engine_h8.h:77 and :379, and at xe2/engine_xe2.h:45 and :63. The design makes it a rule the planner enforces:

- **Each model file declares its class.** It's per model, not per family: Gemma 4 has models in both classes.
  - **"two int8 passes or better"** (q2 two-pass int8, f32 activations, or f16 where that model's KL run places it, below):
    - Gemma 4 12B: two passes 0.0002, one pass 0.021 (0.064 in the H8 test of 11-16);
    - Gemma 4 26B-A4B: two passes 0.0042, one pass 0.0985;
    - Gemma 4 31B: two-pass today, as the non-PLE Gemma models are (engine.h:1371-1376);
    - Qwen3 dense and Qwen3-ASR's text model, two-pass today (one pass measured 0.0064 prompt / 0.0093 decode, two passes < 0.00001, engine.h:1373). The speech limit of 0.01 leaves one pass no margin.
  - **"one int8 pass"**, with no H8 until measured:
    - Gemma 4 E4B and E2B (the per-layer-embedding models), one pass by design since 10-23. E4B's KL: 0.00075 prompt, 0.00093 generation, 476 / 474 of 480 top tokens (before-after.md:810). `g4_one` applies it to every Gemma with PLE (engine.h:1367-1372); E2B has no KL row of its own, so it gets one in its turn.
  - **"one int8 pass"**: Qwen3.5 / 3.6 / 3.8 dense and MoE, Inkling, DeepSeek;
  - **Flash Next:** "one int8 pass", with its own exceptions as today (no H8, `!hp.hc`), and its `--fast` mode as an allowed approximation.
- **Each strategy declares what it delivers:** H8 one pass, int8 one pass, int8 two passes, f16 (`gemm_x16`, XMX f16), f32.
- **f16 is a class of its own, not ranked by rule.** Where it stands against two int8 passes depends on the model:
  - Two passes leave an error of about the block's max / 127 / 254 (~3e-5 of the block's largest value).
  - f16 is ~5e-4 relative per value: better for small values, worse for large ones.
  - The record has f16 only for Gemma's attention keys (11-11) and the NPU's FFNs (10-25), never for the GPU GEMMs' activations.
  
  So a model file accepts f16 only after its own KL run against f32 over hundreds of positions (for Gemma: the 12B and the 26B). If f16 passes, f16 on XMX (`tile_f16`: one pass instead of two) is likely Gemma's fastest route on XMX cards, so it's the first thing Gemma's turn (6.0) measures.
- **The planner refuses, at plan time, any strategy below the model's class,** whatever the tuning cache measured as faster. No family catalog, preset or `ARC_*` switch overrides it. Overriding takes a change to the model's description, with a KL run recorded in before-after.md.
- **H8 is never a candidate for a two-pass model.** It's one pass by construction, unless a two-pass H8 form is built and passes that model's KL. This covers the paths that turn H8 on automatically: the 140T preset, the A770's and B-series' XMX defaults, `ARC_H8_NORM` and `ARC_H8_REUSE`. A Gemma load on those cards goes through the same class check, so it can't pick H8 up.
- **Gemma's XMX routes:** two-pass int8 on XMX (each pass an exact strategy, so the pair keeps today's two-pass numbers bit for bit), or f16 activations on XMX (the `tile_f16` route). Both are candidates for Gemma's catalog entries. Each is measured for KL against f32 over hundreds of positions, and for speed on each family, before it becomes Gemma's default there.
- **The shared routes carry their class from day one.** The first model moved (6.0) brings the dense tile and the GEMV with their delivered precision declared, and two-pass forms where they exist. So Gemma's turn doesn't inherit a one-pass default. The encoders' turn (Qwen3-ASR's two-pass text model) exercises the class check before Gemma's.
- **Re-checks:** when a shared route that Gemma uses moves, Gemma is re-checked on every family that runs it (its hashes, plus its KL where the route is approximate), as a model whose KL has broken before.

**Checks.**
- `tools/gemm_check.cpp` (new, C++): every strategy × weight type × shape class this GPU supports, on random data, against the reference. Bit for bit for exact strategies, an error bound for approximate ones. It needs no model file and runs in minutes, so contributors can run it on their cards. It replaces the scattered start-up checks (`xmx_check`, `xw_check`, `x16_check`, `rgx_first_use`) with one tool. The start-up checks stay until it has run on every family.
- Per migration step: `runz --dump-logits` hashes before and after on the 27B, the 35B, Flash Next, Gemma 4, Inkling 2 Mini, Qwen3-VL's vision, Qwen3-ASR and Whisper.

## 5. Source layout

```
zero/gemm/
  gemm_op.h        GemmOp, layouts, epilogues
  traits.h         GpuTraits from the driver
  presets.h        measured per-card overrides
  planner.h        rules, presets, tuning cache, ARC_GEMM_PLAN / FORCE
  tune.h           the tuning cache file
  strategies/      one host file per strategy (gemv.h, tile.h, wide_h8r.h, expert.h, f32.h, cpu.h, npu.h)
  weights.h        the ahead / fused weight preparation (today's decode_w8, h8_prepare, xw packing)
zero/kernels/gemm/
  load_<type>.cl   one decoder per GGUF type
  gemv.cl, tile_dp4a.cl (gemm.cl), tile_x8.cl, wide_xw.cl, wide_xw8.cl, tile_f16.cl, expert_small.cl, expert_fused.cl
tools/gemm_check.cpp
bench/gemm_fit.cpp   fits each catalog entry's model from gemm_bench runs (3.0, 3.5)
```

- The model files (`engine_qwen.h`, `engine_gemma.h`, `engine_moe.h` routing only, `engine_inkling.h`, `engine_flashnext.h`, `engine_ds41.h`, the drafters) call `linear`, `linear_gu` and `experts`.
- The encoders (`audio_*`, `vision_*`) call `linear`.
- Whisper drops its own split-K and grid and calls the shared layer through a thin adapter for its weights.
- `zero/mgpu/` shrinks to the shard threads and the device-sum forms (3.15). The sharding itself is a field of `GemmOp`, and the stages of a layer split come from the planner, so shards and stages get every new kernel and fusion too.
- `zero/xe2/` shrinks to traits, presets and the XMX strategies' host code. "Code only for discrete cards in `zero/xe2/`" stays true, now as strategies.

## 6. Migration, step by step

Each step:
- is its own branch, built and hashed;
- is bit-identical for every model;
- merges into main only after its checks pass on the laptop and, for code that runs only on discrete cards, on contributors' cards.

| Step | What | Check |
|---|---|---|
| 0 | `gemm_check` against today's kernels; record the model hashes on the laptop | the baseline |
| 1 | `GemmOp` and `gemm::run`, wrapping today's `matmul` / `matmul1` unchanged; dense model calls move to `linear` / `linear_gu` | hashes equal |
| 2 | Experts: `layer_moe`'s choices (per tile, `moe_rg`, `moe_gemm`, GSQ legacy, CPU) move into the planner as strategies with `ExpertTiles` | hashes equal, Inkling's switches included |
| 3 | Encoders and Whisper through the layer, keeping each one's current kernels | hashes equal |
| 4 | Traits and presets replace the card checks, with the same pick per card | the laptop (iGPU) now; discrete cards by contributors: `gemm_check` + hashes |
| 5 | `ARC_GEMM_PLAN` / `FORCE`; old switches become aliases | hashes equal; a table of each alias in docs |
| 6 | The tuning cache, exact candidates only, opt-in | hashes equal with it on and off |
| 7 | New kernels behind the layer, each its own row in before-after.md (below) | per kernel: `gemm_check`, spills, speed |
| 8 | The layer chains in the model files and the JIT fusion compiler (3.12), with today's fused kernels as its first rules | each fused kernel equal to its unfused chain bit for bit; launches a token and compile time against their budgets |

Step 7's first kernels, all for the laptop and all reaching every model at once:
- **The fused weight load for expert tiles:** no `decode_w8` area, NVFP4 per 16. Inkling's 4K prompt spends 7.0 s in `decode_w8`.
- **The oneDNN-style DP4a tile for Xe-LPG:** 32 × 32 per sub-group, work-groups of 8 × 4, K 64, 256 registers. oneDNN reaches ~9.6 TOPS on the 125H against our ~6.0 on the 27B's FFN, and the same kernel serves the 27B, the 35B, Flash Next, Gemma and Inkling.
- **An int8 path for Gemma 4's audio encoder.** This changes its numbers, so it needs the speech KL < 0.01.

The order puts steps 1-2 first: they unlock the laptop's gains (step 7) without waiting for steps 3-6. Each model of the model order (6.0) goes through these steps for its own routes; the first model also builds the shared parts.

### 6.0 Moving old routes to new ones, until the old are deleted

Every path that exists today gets a row in a route table (`docs/routes.md`): its new route in the shared layer, its status, and the measurements that moved it. There are no permanent dual paths. Each row ends with the old route deleted, with its host code, its kernels, its `ARC_*` switches and their docs.

**Status, in order:** `old default` → `both` (the new route built beside the old one, behind a switch) → `new default` (the old one kept for one release as a fallback switch) → `old deleted`.

**A row moves to `new default` only when:**
- the new route is bit-identical where it claims to be exact (`--dump-logits` hashes on the laptop, and `gemm_check` on each family), or passes the KL rule where it doesn't;
- its speed is at or above the old route's on every family it serves, measured the same way before and after (before-after.md), on contributors' cards for the discrete families;
- its kernels pass the spill and size budgets (3.8).

**Release cadence.** Releases go out once a day, so "one release" in this design means about a day. That covers the old route's fallback switch and the switch aliases.

**Deadlines.** Each row gets a deadline when its new route is built: **two releases, about two days at one release a day** (the user, 2026-11-25). If a family's measurements haven't come by then, the row is decided on the data there is:
- the new route becomes the default where it was measured;
- the old one stays only for the family that hasn't measured, as a preset (3.2) naming the missing measurement, until the next deadline.

So an unmeasured family can delay a deletion but can't keep a dual path forever.

**The route table, first version:**

| Old route (kernels; host; switches) | New route | Who measures |
|---|---|---|
| gemm.cl DP4a tile (`gemm`, `-DMOE_TILES`); `matmul1`; `ARC_GEMM_*` tile switches | `tile_dp4a` staged (3.10), the oneDNN Xe-LPG entry | the laptop |
| decode.cl `gemv` / `gemv_v` / `gemv_mt` / `gemv_mt_v` (+ `GU2`, `SPLITK`, `Q2F`); `ARC_GEMV_*` | `gemv`, `gemv_mt` (merged where one form is as fast) | the laptop, then every family |
| gemm.cl `-DXMX` (+ `X2D`, `WPF`) | `tile_xmx16` | B-series, B390 |
| gemm_x8.cl | `tile_xmx8` | 140T, Alchemist |
| gemm_xw.cl / gemm_xw8.cl and the H8R forms; zero/xe2/engine_xw.h; `ARC_GEMM_XW_*`, `ARC_H8_*` | `wide_h8r` / the XMX strategies, fusion rules as catalog data (3.11) | B-series, Alchemist, 140T |
| gemm_x16.cl; `ARC_GEMM_X16` | `tile_f16` (approximate), or deleted if it never beats int8 | 140T |
| moe.cl `moe_gateup` / `moe_down` (GSQ legacy) | deleted if no supported model uses it | (check) |
| `moe_rg` (+ `RG_XMX`), the per-tile `moe_gateup_q` / `moe_down_q`, `moe_gemm`; `layer_moe`'s choices; `ARC_MOE_*`, `ARC_INK_DOWN_TILE`, `ARC_INK_MOE_GEMM` | `ExpertTiles` on the tile strategies, `expert_small`, the fused weight load (3.4), planned per layer type | the laptop (Inkling, 35B), then every family |
| `decode_w8_sb / _sl / _vb`, `h8r_decode`, `w8_requant` | one loader per type, ahead or fused (3.4) | the laptop |
| `gdn_recur` (+ `FUSE_GATES`, `GDN_FUSE2`), `gdn_conv`, `gated_norm_quant`; `gdn_chunk` / `gdn_chunk_x` / `_x8`; `ARC_GDN_*` | the DeltaNet chain at the floor (3.6), the chunked form on microkernels | the laptop, then every family |
| `ink_*` kernels (`ink_mix`, `ink_ctrl_*`, `ink_conv1`, `ink_rows_kv`, `ink_mm_f32(_sg)`, `ink_moe_out`, `ink_copy`); `ARC_INK_*` | the layer chain with prologues / epilogues (3.13), the controllers as GEMMs | the laptop |
| the generation GEMVs: `gemv_gu`, the head's GEMV, `gemv_bf16` / `gemv_f32` at T ≤ 16 | the small-M strategies (3.16), judged by their share of the bandwidth | the laptop, then every family |
| `attn_prep`, `attn_prep_g`, `ink_rows_kv`, the K / V `ink_conv1`, `kv_store` | the KV store and its steps in the q / k / v epilogue or the attention's prologue (3.16) | the laptop |
| attention: `attn_split` / `attn_dsplit` / `attn_decode` / `attn_combine`, the new split decode attention (`fix/ink-attn-dec`), `attn_fx`, `attn_q8`, `attn_fa_g(c)`, `ink_attn(_pp / _ps / _qb)`, `ds_attn_*` | the attention kernels on the shared tile functions (3.6), one per form, combine by way 7 | the laptop, then every family |
| norm / quant / elementwise launches (`rmsnorm_quant`, `swiglu_quant`, `quant_rows`, `moe_quant`, `attn_gate_quant`, `add_norm_quant`, `softcap_rows`, ...) | prologues and epilogues (3.7, 3.14); each is deleted when no chain uses it | the laptop |
| `gemv_bf16`, `gemv_f32`, `hc_gemm_f32`, `vit_gemm_f16`, Whisper's own GEMM | `f32_small`, `f16_encoder` through the shared layer | the laptop |
| the submission forms (immediate, `ARC_BATCH`, replay) | one submission layer (3.9) | per family |
| the tensor split's sums (`tp_add`, `tp_half`, `tp_rs_sum / _add / _add_r`, P2P, `ARC_TP_*`), the shard setup (`engine_tp_shard.h`), `tp_mirror.h` | the device-sum post-op and the GemmOp's sharding (3.15) | the multi-GPU boxes, 1 / 2 / 3 / 4 GPUs |
| the expert split (`tp_moe_mask / _scatter / _tiles1 / _combine`, `tp_reduce_hc`) | expert sharding of `ExpertTiles` | the multi-GPU boxes |
| the layer split (`engine_pp.h`, its hand-off) | stages planned from the layer chain, per-device catalogs (3.15) | the multi-GPU boxes, the mixed B60 + A770 box |

**Order: model by model** (the user, 2026-11-25). Each model is moved whole onto the new path, and its old routes are retired, before the next one starts:

| # | Model | What it brings into the shared layer |
|---|---|---|
| 1 | **Inkling 2 Mini, the pilot** | the catalog, the planner, the budgets and the route table themselves; the dense tile and GEMV strategies; ExpertTiles with the fused loader (NVFP4 per 16, the Q5_K / Q4_K / IQ4_XS layers); the controllers as GEMMs; its attention on tile functions; the looped-layer plan (3.13, 3.17). Why first: the largest gap (1,356 → 385 launches; 2-6x behind Basalt), and its correctness path (I1.5) as the KL reference |
| 2 | **Qwen 27B (dense)** | the DeltaNet chain (3.6), the attention with q / k norm and RoPE, the H8 / XMX dense strategies on discrete cards. The most-used model, measured on the most cards |
| 3 | **Qwen3.6 35B-A3B (MoE)** | the 35B's expert path, the shared expert, larger chunks, the expert split over GPUs |
| 4 | **Qwen3.8 Flash Next** (the user: Flash Next before Gemma 4) | the hc mixers, QSA, the precise / `--fast` modes. Its tensor split moves onto 3.15's sharding (it splits today, with the experts by range and the hc reduce). The layer split, which today is Qwen-only ("not Gemma, DeepSeek or Flash Next yet", engine_load.h:718), could be added in this turn too; that's a proposal for the user's review |
| 5 | **The encoders** (Whisper, Qwen3-ASR, the vision towers) and the remaining f32 / f16 GEMMs (the user: the encoders after Flash Next) | the f16 encoder strategy; Whisper's own GEMM retired. Qwen3-ASR's text model is in the "two int8 passes or better" class (section 4), and the f32 / f16 GEMMs declare their own classes, so this is the first turn that runs the planner's class check with a two-pass model, before Gemma's. Speech models stay under KL 0.01 |
| 6 | **Gemma 4** (12B / 31B dense, 26B MoE, E2B / E4B) | sliding-window attention, GeGLU, the sandwich norms, the PLE. Its classes per model (section 4): 12B, 26B and 31B "two int8 passes or better", with no H8 or one-pass candidate on any family; E4B / E2B "one int8 pass", with no H8 until measured. The two-pass models' XMX routes are two-pass int8, or f16 activations. f16 is measured first: a KL run for the 12B and the 26B against f32 over hundreds of positions, then speed per family. It's re-checked whenever a shared route it uses moves, also before its own turn. **Multi-GPU arrives in this turn** (today "Gemma does not split yet", and the layer split is Qwen-only), built from 3.15, not separately. Details below the table |
| 7 | **DeepSeek-V4.1** | the compressor / indexer, the grouped output, Engram |

**Gemma 4's multi-GPU, in its turn** (3.15's sharding, no second implementation):
- **Tensor split:**
  - q / k / v by kv-head group, o by rows, gate / up (GeGLU) by FFN column, down by rows;
  - the 26B's experts by range, as the 35B's;
  - the head by vocabulary, with softcap in the head's epilogue on each shard. Softcap is per logit, so it can run before the join.
- **Gemma's own pieces:**
  - **Sliding-window and global layers** keep their own KV geometry per shard (the window rings, the global KV with K shared as V on the 31B's global layers).
  - **The shared-KV layers** (`il >= n_layer_kv` read an earlier layer's KV, engine_load_gemma.h:33-41). They must find that KV on their own GPU. In the tensor split that holds if the source and its readers shard by the same kv heads. In the layer split, the source and its readers stay on one stage, or the cut falls where no reader crosses it.
  - **E2B / E4B's per-layer embeddings:** replicated on every shard (the tensor split), or computed on the first stage and handed on with the rows (the layer split).
  - **The sandwich norms** (post-attention, post-FFN, out_scale) join the sums' fused residual + norm pass. The norms then come after the full sum, in the same order as on one GPU.
- **Layer split:** whole layers per stage, with the shared-KV rule above for where the cut can fall.
- **The accuracy class holds per shard:** two int8 passes or better, no H8. The device sum is added in fixed rank order. As for Qwen's tensor split, this isn't bit-identical to one GPU (3.15), so its KL against one GPU and against f32 is measured over hundreds of positions. A layer split of like cards is bit for bit against one GPU.
- **Measured** at 1 / 2 / 3 / 4 GPUs on the owners' boxes: manoreken2's B60s, Daleang's B70s, hansong80's A770s, skipzx-lab, MaineTim, catduckgnaf. The 26B MoE and the 31B dense gain most from a split. Gemma's multi-GPU route rows land in its turn, after Flash Next's and the encoders'.

- **Shared rows move with the first model that needs them.** A dense-GEMM strategy and the submission layer arrive with Inkling. Each later model reuses them and adds only its own steps. When a later model uses a shared route, that route is checked on that model too: its hashes, and its KL where the route is approximate.
- **A model is done when:**
  - its launches and activation bytes are at or near its floor (3.7, 3.14);
  - its KL or bit-identity passes as the rules require;
  - its speed is at or above the old path on every family measured;
  - its old routes are deleted at their deadline.
- **Two things the model order doesn't change:**
  - Every deletion is listed in the CHANGELOG and before-after.md.
  - A shared old route (`gemm.cl`, the GEMVs) is deleted only when no model uses it any more.

**The trade-off of piloting with Inkling** (an open question, 9.6). Inkling is the most complex model: it has experts, loops, controllers and an attention of its own. The pilot therefore builds the most new code at once, and Inkling has few contributor runs to measure it on the discrete cards. Starting with the 27B would build the dense tile, the GEMV and the planner on the model measured on the most cards, and with the simplest chain. Inkling would come second, adding experts, loops and controllers on a proven base. The cost of that order is that Inkling's largest gap waits one model longer. Both can be checked on the laptop for correctness and speed, since both run on the 125H.

**The code shrinks visibly.** `ARC_GEMM_PLAN=1` prints the counts of routes, kernel files and switches. `docs/routes.md` keeps them per release, and they only go down: a change that adds a route without retiring one says why in its commit.

### 6.1 Alternative: rewrite instead of migrate

Rewriting the GEMM code fresh gives a leaner result than wrapping today's code step by step. What makes it possible: the int8 sums are exact. A rewritten kernel that keeps the reference order (section 4) gives the same bits as today's kernels, so a rewrite doesn't have to change any model's numbers. `gemm_check` and the model hashes prove it.

What decides the plan is what can be checked where:

| Part | Rewrite now? | Why |
|---|---|---|
| Host side: `GemmOp`, planner, catalogs, submission, the model calls, the switches | **Yes, in one go** | That's where most of the mess is: about 150 switches and the branches per card in engine.h, engine_moe.h, engine_h8.h and zero/xe2. The kernels and their options stay as they are underneath, so it's checkable on the laptop by hashes for every model, discrete paths included, because the kernel builds don't change |
| The DP4a tile kernel (kernel 2) and the GEMV (kernel 1) | **Yes, fresh** | Written in the staged form of 3.10 with oneDNN's shape for Xe-LPG. Measurable on the laptop, bit for bit against today's gemm.cl and decode.cl |
| The XMX kernels (kernel 3) | **Not until contributors can run it** | Weeks of contributors' runs are in `gemm_xw8` / `gemm_x8` (A770, 140T) and `gemm_xw` (B-series). A fresh kernel is checkable for bits offline, but not for speed. Keep them under the new host until a rewritten kernel 3 matches them on the cards, then delete them in one step per family |
| f16 / f32 kernels (4, 5) | Yes | small, and checkable on the laptop |

**What the rewrite drops**, each confirmed unused or superseded by a measurement first:
- the GSQ legacy MoE kernels (moe.cl `moe_gateup` / `moe_down`), if no supported model uses them;
- per-256 H8 (`ARC_H8_SB`) next to H8R;
- `gemm_x16` if the 140T's f16 measurements never beat int8;
- the duplicate GEMV forms (`gemv` and `gemv_v`, `gemv_mt` and `gemv_mt_v`) merged where one form measures as fast;
- Whisper's own GEMM;
- each switch that no catalog entry or test uses.

**The order with a rewrite:**
1. `gemm_check` and the baseline hashes.
2. The new host in one branch (old kernels underneath; every model's hash equal).
3. The new DP4a tile and GEMV (laptop; bit for bit; speed).
4. The encoders and the f32 kernels.
5. The XMX rewrite when contributors can run it.

It's fewer steps than the migration in 6, but step 2 is one large change. It needs the hash set on every model, plus `gemm_check` and the hashes on contributors' discrete cards before main takes it, since their code paths change on the host side too.

## 7. Risks

| Risk | Answer |
|---|---|
| The migration touches every path, but the laptop can only test the iGPU paths | Steps 1-6 keep the kernels and their options as they are, so the kernel cache keys are the same; discrete paths are checked with `gemm_check` + hashes by contributors before main takes them. Until then they stay on the branch |
| Dispatch overhead in generation (hundreds of GEMVs a token) | The plan is cached on the weight; the call is a pointer and a switch, as today. Checked with `--prof-gen` launch time before / after step 1 |
| A preset table drifts from measurements | Each entry names its measurement and issue; `gemm_check` and `ARC_GEMM_PLAN` show it on any card |
| Switch aliases break users' scripts | One release of aliases with a one-line notice each, and the table in docs |
| A large change is hard to review | One step per branch, each with a diff that only moves calls, until step 7 |
| The tuning cache gives different speeds per run | Exact candidates only; the cache can be deleted; `ARC_GEMM_TUNE=0` turns it off |

## 8. Rules this changes in AGENTS.md

- "Where code goes": GEMM code goes in `zero/gemm/` and `zero/kernels/gemm/`; model files don't pick kernels.
- The KL rule adds: a change to an exact strategy needs `gemm_check` bit for bit plus the model hashes. A new approximate strategy needs a KL check for each model that adds it to its allow-list.
- Card-specific choices go in `presets.h` with their measurement.

## 9. Questions for the user

1. Keep today's `ARC_*` switches as aliases for one release, or drop them with the migration?
2. The tuning cache: opt-in first (proposed), or on from the start?
3. Gemma 4 audio's int8 path changes its numbers: include it in step 7, or leave the encoder in f16?
4. Steps 4-6 touch the discrete paths. Do them now on a branch (proposed), with contributors checking from Sunday, or start them only on Sunday?
5. ~~The route table's deadlines~~: answered on 2026-11-25. Two releases, at one release a day (6.0).
6. The model order (6.0): Inkling 2 as the pilot, as you said, or the 27B first? The 27B has the simplest chain and the most cards measuring it; Inkling has the largest gap. The trade-off is in 6.0. Your concern about Gemma's KL (the H8 / one-pass breakage) is handled whichever model goes first: the accuracy classes (section 4) travel with the shared routes from the first model on.

## 10. Today's GEMM paths, for reference

From a read of the code on 2026-11-24 (line numbers of that tree):
- **Dense:**
  - `matmul` (engine.h:1392): the Q2 two-pass wrapper and softcap.
  - `matmul1` (engine.h:1525):
    - host CPU GEMV;
    - `gemv` / `gemv_v` (T = 1);
    - `gemv_mt` (T 2-8);
    - H8 (T > 8, `h8_takes`);
    - `decode_w8` staging;
    - `x16_matmul` (ARC_X16);
    - `gemm_kernel` (engine_setup.h:511): gemm.cl, or gemm_x8.cl on 8-lane XMX.
  - The 140T preset `p140_take` (xe2/engine_xe2.h:30).
  - Split-K (engine.h:1653).
- **H8 / wide tile:**
  - `h8_matmul` (engine_h8.h:560): prepares H8R weights and activations, then `xw_matmul` (xe2/engine_xw.h:1030) or gemm.cl QTYPE 110 / 111.
  - The fused gate and up: `xw_gu_matmul` (855) and `xw_gu_matmul8` (750).
- **Experts** (`layer_moe`, engine_moe.h:8):
  - per tile: `moe_gateup_q` / `moe_down_q`, decode_moe.cl;
  - `moe_rg` (574), with XMX variants in xe2/engine_moe_xw.h;
  - `moe_gemm` (635): decode_w8 then gemm.cl MOE_TILES;
  - GSQ legacy: moe.cl;
  - CPU misses: xe2/engine_xcpu.h:146;
  - multi-GPU: tp.cl.
- **f32 / bf16:**
  - `gemv_bf16` / `gemv_f32` (the router, Flash Next);
  - `hc_gemm_f32` (ops_flashnext.cl:227);
  - `ink_mm_f32(_sg)` (engine_inkling.h:683).
- **Encoders:**
  - audio_gemma4a.h:403: f16 only, `vit_gemm_f16`;
  - audio_qwen3a.h:407 and vision_qwen3vl.h:339: two-pass `matmul`;
  - vision_gemma4v.h:351: one-pass `matmul`;
  - whisper_engine.h:519: its own GEMV / GEMM and split-K.
- **Switches** that change GEMM / GEMV choices, by family:
  - GEMV: about 16;
  - GEMM: about 25;
  - H8: 14;
  - wide tile: about 35;
  - MoE: about 25;
  - two-pass / f32 / CPU / NPU: about 15;
  - encoders: 5.
- **Checks:**
  - `gemm_self_check`, `xmx_check` (engine_setup.h:592, 674);
  - `x16_check`, `xw_check`, `xw_gu_check(8)`, `rgx_first_use`;
  - bench/gemm_w8_bench.cpp, gemv_bench.cpp, moe_bench.cpp, xw_peak.cpp.
