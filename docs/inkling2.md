# Inkling 2 Mini: design (2026-11-23)

Status: design, to build. The user's targets are **prompt 6,000 tok/s and generation 130 tok/s on a Pro B70 and an A770**
(#83). The model facts below come from the other session's analysis of the GGUF headers (unsloth/Inkling-2-Mini-GGUF:
the mainline Q8_K_XL, the mmproj, the DSpark drafter; all in the laptop's HF cache).

## Work rules for this phase (the user: no GPU and no CPU use)

This phase is **code and documents only**:
- **Nothing runs:** no model run on the GPU or the CPU, no ref_forward or llama.cpp run, no conversion, no benchmark, no
  laptop GPU test.
- **What's allowed:** reading the GGUF headers and llama.cpp's sources, writing code, tests and docs, and short builds or
  ocloc compiles to check that the code compiles. A full rebuild of everything is not needed: only the files you
  touched.
- **What waits:** every run (the tiny model's tests, the KL against llama.cpp, the speed asks) waits for the user's
  go-ahead. Each step below lists the run it waits on.

The Developer works on the A770 at the same time, so this work stays out of its files:
- not `zero/kernels/gemm_xw8.cl`, `zero/xe2/engine_xw.h`, `zero/engine_sessions.h`, `zero/xe2/engine_hostcpu.h` or
  `bench/xw_peak.cpp`;
- new files for Inkling's own code, and small hooks elsewhere, each guarded by `hp.inkling`.

The iGPU path and every other model must stay bit-identical: Inkling's code only runs for its architecture. Commits:
only this work's files, the GitHub noreply address, no Co-Authored-By line, a CHANGELOG entry. No releases; the
Developer cuts them.

## The model (the parts the engine needs)

- **Layers:** `general.architecture = inkling2_loop`: 24 layers, n_embd 2304, 24 query / 8 KV heads of 128, no RoPE.
- **The loop:** layers 6-17 run 3 times a token (`loop.start 6`, `loop.end 18`, `loop.count 3`), 48 layer passes. Each
  pass has its own KV. The loop layers have rank-64 controllers: `loop_{q,k,v,ffn}_ctrl_down`, `_forget`, `_gain`,
  `loop_msg_query`.
- **Attention:**
  - 18 sliding layers (window 1024) and 6 global ones (5, 11, 17, 23);
  - a relative-position bias (`attn_rel`, 16 buckets, `relative_extent 1024`, `relative_clip 8`, `attn_rel_bias`);
  - sinks (`attn_sink_proj`) and an output gate (`attn_gate`);
  - `qk_clip 22.6` and a length-log scale (`log_scale_floor 8192`, `alpha 0.1`);
  - short causal convs on K and V (kernel 4);
  - the global layers add `attn_{k,v}_conv_proj` and q / k / v input norms.
- **The residual stream:** depthwise convs after attention and the FFN (`attn_post_conv`, `ffn_post_conv`), attention
  residuals (`attn_res.block_size 6`, `attn_res_query` / `ffn_res_query` / `output_res_query`), and `token_embd_norm`.
- **MoE in every layer:** 128 experts, top 8, expert FFN 1152, gate and up fused (`ffn_gate_up_exps`). The router is
  F32 with an `exp_probs_b` bias, `expert_weights_scale 8`, and a per-expert f32 `scale`. The activation is "situ"
  (`situ_beta 4`, `situ_linear_beta 25`). No shared expert.
- **The output:** a vocab of 201,024, `final_logit_softcapping 30`, tokenizer pre `inkling2`.
- **Weights in Q8_K_XL:**
  - experts in **NVFP4 (ggml type 40)** in layers 0-19 and BF16 in 20-23;
  - attention and the head in BF16;
  - norms in F32.
- **NVFP4:** E2M1 codes with an E4M3 scale per 16, times the per-expert f32 scale. E2M1 x 2 are MXFP4's integer codes
  (`kvalues_mxfp4`), so the codes are int8 exactly; the scales are not (see I1.1: folded per 32, sometimes rounded).
- **The accuracy reference:** llama.cpp master's `src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp` (plus
  `llama-graph.cpp`, `llama-model.cpp`, `llama-kv-cache.cpp` for the shared parts). PR #32704 itself only re-shares
  the loop's layers after the scale pass. Mirrored here: master `44d5087` (2026-11-23), where inkling2.cpp last
  changed in `19bc41d` and inkling2-loop.cpp in `690b87d` (#32704). The KL runs use a llama.cpp build at or after
  that commit (Unsloth's perplexity 7.653 on Q8_K_XL). Read its code for every operation's exact order and constants;
  the engine must match it, not this summary.

## Accuracy: what differs from the f32 reference by design

- **The NVFP4 per-32 fold (GPU only):**
  - The engine's int8 form has one scale per 32 values, while NVFP4 has one per 16.
  - Where a 32-block's two scales differ by other than a power of two up to 8, the smaller half is rounded, by at
    most d / 2 (1/192 of the block's largest possible value). Elsewhere it is exact.
  - `tools/nvfp4_stats` counts the share of exact blocks in a file from its scale bytes alone.
  - The real file's KL against llama.cpp includes the fold. If it costs too much, the experts' prompt path can decode
    per 16 instead.
  - The I1.5 correctness path (`ink_expert_mv`) reads NVFP4 exactly, so it isolates the graph from the fold.
- **The K / V cache in f16**, as Gemma's V.
- **The BF16 matrices requantized to Q8_0** (the engine's default for float weights), as for every other model.
- **2D weight `.scale` tensors:** llama's `build_lora_mm` multiplies a 2D weight's output by its `<name>.scale` (one
  value) where the file has one. The Q8_K_XL has none on 2D weights (only the experts' per-expert scales), but an
  NVFP4-everywhere file could.
  - ref_forward applies them (`ref_inkling2.h` `mm`, and `output.scale` in `head()`).
  - The engine's loader must too, or refuse the file (I1.5).
  - Activation scales (`.input_scale`) are refused.

## Steps

**I1. Support, written now, run later.**
1. **NVFP4:**
   - `dequant_row` for type 40 in `src/quants.h` (CPU, for ref_forward and the loader);
   - `tools/gguf_dump` reading type 40;
   - the GPU decode to int8 per 16 in a new part of `zero/kernels/decode_moe.cl`, or its own file;
   - a raw NVFP4 GEMV for one token (4.5 bits, as MXFP4's).
   - Waits on: a unit test against llama.cpp's dequant on a few blocks.

   **Written (2026-11-23):**
   - `src/gguf.h`: type 40, blocks of 64 in 36 bytes, so `tools/gguf_dump` and every loader open the file (checked:
     gguf_dump lists the experts as NVFP4, 10,935 MiB of 21,797).
   - `src/quants.h`: `nvfp4_scale` (ggml's `ggml_ue4m3_to_fp32`: 0x00 and 0x7F are 0, the top bit not read, halved),
     `dequant_row` for type 40 (exact, as ggml's `dequantize_row_nvfp4`); the GPU raw superblock (4 blocks: the 16
     scale bytes, then 4 x 32 code bytes, 144 bytes); NVFP4 in the per-32 and small-block type lists.
   - **The per-32 fold, a correction to the analysis:** the engine's int8 form has one f32 scale per 32 values, and
     NVFP4 has a UE4M3 scale per 16. Two halves with the same scale keep their codes; otherwise d = max / 8, the larger
     half's codes x 8 and the other's rounded in integers (`nvfp4_block32_w8`). Exact when the scales differ by a power
     of two up to 8, else at most d / 2 off (1/192 of the block's largest possible value). The Q5_1 precedent: every
     path (decode_w8, the GEMVs, the experts' kernels) decodes through `decode32`, so prompt and generation agree.
   - `zero/kernels/decode.cl`: QT 40 in `decode32` (the same integers as the CPU), so the raw GEMV for one token, the
     expert GEMV (`MOE_Q`) and the prompt's expert tiles (`MOE_RG`) all read NVFP4's 4.5-bit rows. ocloc: compiles for
     mtl and bmg-g31, the plain, `MOE_Q` and `MOE_RG` forms (MXFP4's the same, as a control).
   - `tests/test_quant_rows.cpp --self-test`: the scale decode against ggml's formula for all 256 bytes,
     `dequant_row` against ggml's loop, the GPU layout's addressing, the fold's bound over all 65,536 scale pairs.
   - The per-expert f32 `.scale` tensors (`ffn_gate_up_exps.scale`, `ffn_down_exps.scale`, 128 each) multiply the
     expert's matmul output, as llama.cpp's `build_lora_mm_id` does: applied in the engine's graph (I1.5), not here.
   - **Runs it waits on:** `bin\test_quant_rows.exe --self-test` (CPU, a second); the fold's effect on the model, the
     KL of I1.5 (if it is too large, the experts' prompt path can decode per 16 instead).
2. **The tokenizer:** pre `inkling2` in the tokenizer code, and the chat template.
   - Waits on: token ids against llama.cpp's on a few texts.

   **Written (2026-11-23):**
   - llama.cpp maps pre `inkling2` to GPT-4o's split (`LLAMA_VOCAB_PRE_TYPE_GPT4O`) with `ignore_merges` and no BOS. It
     has no hand-written splitter for that regex: it runs std::regex on its "collapsed" text, where a non-ASCII
     codepoint becomes its class, so `[^a-z]` / `[^A-Z]` tell case apart for ASCII only.
   - `src/tokenizer.h`: `pretokenize_gpt4o` follows that regex alternative by alternative (ECMAScript order, greedy
     with the one backtrack that matters: `U*` gives characters back until `Lo+` can start); `ignore_merges` (a word
     that is a token as a whole is that token); `bos = -1`. Other models' tokenizers unchanged (the flags are set only
     for pre `inkling2`).
   - `src/chat_inkling.h`: the GGUF's template written out natively, because it uses what minja lacks
     (`{% generation %}`, `tojson(sort_keys=..., separators=...)`, `dictsort(case_sensitive=...)`). It renders as Hugging
     Face does (`tojson` = json.dumps with ensure_ascii=False). Tools are declared first in a `tool_declare` system
     message, keys sorted. The thinking effort is a system message before the first non-system message: a name (none
     0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh / max 0.99) or a number, 0.9 by default. Tool calls are
     `<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,"args":{sorted}}`, and an assistant turn ends with
     `<|content_model_end_sampling|>`, token 200006, the end of generation. **There is no generation prompt:** the
     model writes its own `<|message_model|>`. `inkling::parse` turns the output back into reasoning, content and tool
     calls. Images raise an error: the inkling2v projector isn't built.
   - `tests/test_inkling_chat.cpp`: 12 splits worked from the regex by hand, the rendered prompts against the
     template worked by hand (turns, effort, tools, calls, the tool's name from its call), the parser.
   - Not yet: the server's use of it (the format's stop token, streaming the thinking and the calls). That goes with
     the catalog entry (I1.6), once the engine loads the model.
   - **Runs it waits on:** `bin\test_inkling_chat.exe` (CPU, instant); token ids against `llama-tokenize` on the real
     file for a few texts (English, code, CJK, emoji, numbers, contractions, runs of spaces and newlines); one rendered
     conversation against llama.cpp's own rendering (`llama-server` `/apply-template`), since its Jinja engine is the
     reference here.
3. **ref_forward's `inkling2_loop` graph** (f32, `tools/ref_forward.cpp` or a part of its own): every operation above,
   mirroring PR #32704 line by line, the weights dequantized per row as today.
   - Waits on: ref_forward against llama.cpp on a short prompt (KL, top token), then the engine against ref_forward.

   **Written (2026-11-23):** `tools/ref_inkling2.h`, included by `tools/ref_forward.cpp` for `inkling2` and
   `inkling2_loop`. It mirrors llama.cpp master's `src/models/inkling2.cpp` and `inkling2-loop.cpp` (PR #32704 itself
   only re-shares the loop's layers after the scale pass). In f32 with double sums, one sequence from position 0:
   - **The loop:** the prelude (layers 0-5), then the core (6-17) three times, every pass starting from the prelude's
     output and its blocks. A later pass gets each layer's "messages": a 4-query mix over the previous pass's bank
     (its blocks after the first core one) and its output.
   - **Controllers:** they act on the q / k / v / ffn inputs: `x - tanh(forget(g)) x + tanh(gain(g)) msg`, where
     `g = down([rms(x), rms(msg)])`.
   - **Attention residuals:** the stream restarts at every 6th layer, and the old stream joins the blocks. A layer's
     inputs are softmax mixes over the RMS-normed blocks and the stream; the global layers take one mix each for q, k
     and v.
   - **Attention:**
     - q and r times their per-head clip scales, then clipped to an L2 norm of 22.6 / 8; k clipped; v RMS-normed;
     - q and r of the global layers times `1 + 0.1 log(max(1, (pos + 1) / 8192))`;
     - the score is `q.k / sqrt(128) + r.bias[distance] - sink`; past the 1,024-entry table the bias is 0, so the
       score is just minus the sink;
     - a zero sink logit in the softmax; the SWA window is 1,024;
     - the output times `sigmoid(gate)`.
   - **Short convolutions:** on K and V (the global layers add per-head input-dependent taps), on the attention
     output and on the MoE output, each with its residual and `1 / sqrt(5)` (with the taps: `1 / sqrt(9)`).
   - **MoE:** a sigmoid router plus the selection bias, top 8, `w / (sum + 1e-20) x 8`; gate and up fused; each
     expert's two matmul outputs times its `.scale`; situ.
   - **The end:** the output mix, `output_norm`, the head, softcap 30, and the logits cut to the unpadded 200,058
     (ref_forward's new `head()`: a no-op for every other model).
   - Built clean. **Runs it waits on:** ref_forward on the real file, the top 10 and KL against llama.cpp on a short
     prompt. That needs a llama.cpp build with `inkling2_loop` and #32704's fix (merged
     2026-11-21; e.g. release b12648 of 2026-11-23). The local `llama-builds` (b12292, 2026-11-11) predate Inkling
     altogether: it landed on 2026-11-14 (#32217) and the loop variant on 2026-11-20. Then the engine against it.
4. **A tiny synthetic Inkling** (`tools/make_tiny_inkling2.cpp`, after make_tiny_deepseek41): the same tensor names
   and metadata, a few layers, a loop of 2-3, a few experts, random weights. It's for fast correctness tests of the
   engine against ref_forward without the 23 GB file.
   - Waits on: the user's go for CPU / GPU runs.

   **Written (2026-11-23):** `tools/make_tiny_inkling2.cpp` (`make_tiny_inkling2 OUT.gguf [--seed N] [--no-loop]
   [--nvfp4-equal]`). The real file's keys and tensor names, with sizes chosen so that a ~40-token prompt reaches
   every case:
   - 8 layers in residual blocks of 2, the loop core [2, 6) run 3 times;
   - global layers 1, 5 and 7: 5 inside the loop, so a global layer gets messages;
   - head size 128, 6 query heads over 2 KV heads (GQA 3, as 24 / 8);
   - a window of 16 and a relative extent of 32, so distances pass both tables' ends; a log-scale floor of 8;
   - 8 experts, 2 active, FFN 128: NVFP4 with their `.scale` tensors in layers 0-5, BF16 in 6-7 (as Unsloth's split);
   - every other matrix BF16, the router F32;
   - a vocabulary of 500 padded to 512, tokenizer pre `inkling2`.

   `--no-loop` writes the plain `inkling2` architecture. `--nvfp4-equal` makes every NVFP4 block's scales equal, so
   the engine's per-32 fold is exact and any remaining difference is the graph's. Built clean.

   **Runs it waits on:** making the file (CPU, a second), and ref_forward on it (CPU, seconds). Then the engine
   against ref_forward on the laptop through gpu_guard (I1.5), with and without `--nvfp4-equal`. One check of the
   file itself: llama.cpp (a build with Inkling) loads it and gives the same top token as ref_forward.
5. **The engine's graph:**
   - `zero/engine_inkling.h` (hparams, the loader's tensor map, the layer loop with per-pass KV, the forward for one
     token and for a prompt chunk);
   - its kernels in `zero/kernels/ops_inkling.cl`: the relative-bias attention with sinks, qk clip and the log scale;
     the K / V short convs; the residual convs; the attention residuals; the controllers; situ; the softcap;
   - the MoE through the existing standard-GGUF expert path where it fits, with NVFP4 added there.
   - BF16 matrices take the engine's float requant to Q8_0, as other models do.
   - Explicit local sizes where a kernel's local memory or shuffles assume a group shape (AGENTS.md).
   - Waits on: the tiny model's KL on the laptop through gpu_guard, then the real file.

   **Written (2026-11-23): the correctness path.** `zero/engine_inkling.h` (part of the Engine class) and
   `zero/kernels/ops_inkling.cl`. It follows `ref_inkling2.h` step by step:
   - **Weights:**
     - The dense matrices (q, k, v, o, gate, rel, conv taps, the controllers' down) go through the engine's int8
       matmuls, BF16 requantized to Q8_0 as for every model.
     - Matrices whose shapes the GEMM refuses (N not a multiple of 16 or K of 256: the 24-row sinks, the controllers'
       rank-64 gain / forget), and the router, stay f32 (`ink_mm_f32`); the router is f32 in llama.cpp too.
     - A 2D weight with a `.scale` is refused for now, and so are `.input_scale` tensors.
   - **Experts:** read from their raw GGUF rows by `ink_expert_mv`: NVFP4 per 16 exactly, plus BF16, Q8_0 and F32.
     The per-expert `.scale` goes on the gate-up output (in `ink_situ`) and on the down output (`ink_expert_sum`), as
     llama.cpp applies it. So this path does not take the per-32 fold: the fast MoE kernels (I2 / I3) will. The whole
     expert tensors are uploaded raw: the Q8_K_XL needs a 32 GB card (B70, B65), and the laptop runs the tiny file.
   - **Cache slots:** llama.cpp's 48, i.e. the 24 layers and then the loop's later passes. Each has its own f16 K / V
     (the global layers' slots hold the context, the sliding ones a ring of `swa_ring_cap`) and its four
     short-convolution states (K, V, attention output, FFN output), zeroed by `reset()`.
   - **Kernels**, f32 and written for correctness:
     - `ink_mix` (the attention over depth), `ink_rms2` and `ink_ctrl_apply` (the controllers);
     - `ink_conv` and `ink_conv_state`; `ink_rows` (the clips and tau, V's norm); `ink_kv_store`;
     - `ink_attn`: one work-group of 128 per (token, head), the relative-bias table in local memory, online softmax
       from the zero sink, the gate;
     - `ink_route`, `ink_situ`, `ink_expert_sum`, `ink_mask_logits`.
     - Every kernel with local memory takes an explicit size and keeps its barriers in uniform control flow.
   - **The chunk:** `ink_forward` runs the whole graph from `run_layer`'s first layer: the prelude, the passes with
     their bank and messages, the epilogue. The output mix goes into `x_`, and `head()` then applies `output_norm`,
     the head (Q8_0) and the softcap; `ink_mask_logits` sets the padded vocabulary's logits to -inf.
     `runz --kl-ref` compares the unpadded 200,058. The chunk is 512.
   - **Refused:** slots, MTP / DFlash, several GPUs, snapshots (`snapshot()` throws, `restore()` returns false).
   - **Hooks**, each under `hp.inkling`:
     - engine_setup.h `load_hparams`; engine_load.h `build_kernels` / `load_weights`; engine_alloc.h;
     - engine.h `forward` (no layer-major), `run_layer`, `head_rows`, `reset`, `snapshot` / `restore`, `max_chunk`;
     - engine_slots.h `ctx_bytes`; runz's `--kl-ref` vocabulary; `ops_inkling.cl` appended last to
       `KernelParts::kOps`, so every other kernel's source is unchanged.
   - Built: runz and serverz clean; `ops_inkling.cl` compiles alone for mtl, bmg-g31 and dg2-g10, and after the other
     ops parts for mtl.
   - **Runs it waits on:**
     - the tiny file through `bench\gpu_guard.ps1`, runz `--kl-ref` against ref_forward's REF_ALL, with and without
       `--nvfp4-equal` (they should agree, since this path reads NVFP4 exactly);
     - the bit-identity of the other models: `runz --dump-logits` on the 4B and the 35B before and after;
     - the real file on a 32 GB card.
6. **Memory and fit:** ctx_bytes for the loop's per-pass KV (12 loop layers x 3 passes), the plan's estimate through
   the shared helpers (`ring_pos_bytes`, `swa_ring_cap`), and a catalog entry with the drafter.

**I2-I4. Speed, later, each needs cards** (from the analysis): tg on a B70 at ~83 % of its bandwidth, then DSpark
drafting; prompt with NVFP4 read straight in the expert GEMM and chunk 8,192; the A770 needs DSpark and a ~3-bit file or
the RAM tier, since the Q8 file doesn't fit 16 GB.

## Runs waiting for the user's go (none run yet)

| Step | Run | Where | Checks |
|---|---|---|---|
| I1.1 | `bin\test_quant_rows.exe --self-test` | CPU, 1 s | NVFP4 dequant against ggml's formula, the fold's bound |
| I1.1 | `bin\nvfp4_stats.exe <Inkling Q8_K_XL>` | CPU, reads 11 GB of expert bytes | the share of 32-blocks the fold keeps exact |
| I1.2 | `bin\test_inkling_chat.exe` | CPU, 1 s | the split (incl. 200 KB runs of one class), the rendering, the parser (incl. a stream stopped inside a call) |
| I1.2 | token ids against `llama-tokenize` on a few texts | CPU | English, code, CJK, emoji, numbers, contractions, runs of spaces / newlines |
| I1.2 | one conversation against llama-server's `/apply-template` | CPU | the native rendering = llama.cpp's Jinja |
| I1.3 | `ref_forward` on the real file vs llama.cpp (a build with #32704) | CPU, minutes | top 10 and KL on a short prompt |
| I1.4 | `make_tiny_inkling2` + `ref_forward` on it; llama.cpp on it | CPU, seconds | the tiny file loads in llama.cpp, the same top token |
| I1.5 | the engine on the tiny file vs ref_forward, through `bench\gpu_guard.ps1` | laptop iGPU | KL, top token over a few hundred positions; with and without `--nvfp4-equal` |
| I1.5 | the engine on the real file vs ref_forward / llama.cpp | a 32 GB card (B70, B65) | KL <= 0.015 token by token |

## Done for this phase

- I1.1-I1.6 written, compiling (the touched files only), and documented.
- A list of the runs each step waits on, ready for the user's go.
- No run of any kind.
