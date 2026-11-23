# Inkling 2 Mini: design (2026-11-23)

Status: design, to build. The user's targets are **prompt 6,000 tok/s on a Pro B70** (the user, 2026-11-23: "I mean
6000tok/s for B70") **prompt 4,000 tok/s on an A770** (the user: "Use 4,000"), **and generation 130 tok/s on both** (#83). The model facts below come from the other session's analysis of the GGUF headers (unsloth/Inkling-2-Mini-GGUF:
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
- **ggml-org's files** (`ggml-org/Inkling-2-Mini-GGUF`, the conversion arcint uses, #83): the same architecture
  name and tensor names; other types. IQ4_NL (17.6 GB): IQ4_NL experts in layers 0-19, Q8_0 in 20-23 and every other
  matrix, the loop controller F32. Q8_0 (27.8 GB): everything Q8_0. BF16 (52.2 GB): the experts upcast to BF16. The
  engine takes all three since 2026-11-24 (IQ4_NL experts on the streamed path, the default; `ARC_INK_RESIDENT=1`
  refuses them). Checked on the tiny file with those types (`make_tiny_inkling2 --ggml-org`): KL against ref_forward
  0.00020 on the prompt and the decode paths.
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
     After the Researcher's review of 0741c07:
     - `serverz` refuses an Inkling file before any GPU memory is taken. The server snapshots after every prompt, and
       its host snapshot would have let a new conversation continue on another one's conv states and rings.
     - `snapshot_host` keeps nothing and `restore_host` returns false for Inkling.
     - `save_state` / `load_state` and a cancelled chunk's rewind (which must `reset()`, since the whole graph runs
       from layer 0) are in engine_sessions.h; the Developer guards them.
     - `runz`'s one-GPU KL paths (`--kl-ref`, `--kl-decode`, the reference-size test) compare the unpadded vocabulary.
     - Loading also refuses an `output.scale`, more than 256 relative buckets, or an expert FFN not a multiple of 64.
     - `attn_res_query`'s queries now come from its size.
     - tau applies to the global layers whatever `log_alpha`'s sign, as in the reference.
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

   **Written (2026-11-23):**
   - **Memory:**
     - `ink_ctx_cache_bytes(c)` (in `ctx_bytes`) is the 48 slots' f16 K / V: the 8 global slots hold c positions, the
       40 sliding ones a ring of `swa_ring_cap(c, chunk, 0)`.
     - Inkling 2 Mini at 32K: 8 x 32,768 x 1,024 x 2 x 2 = 1.07 GB, plus the rings, 40 x 1,536 x 4 KB = 0.25 GB.
     - The conv states (48 slots x 3 inputs x (1,024 + 1,024 + 2,304 + 2,304) channels x 4 bytes, 3.8 MB), the weights, the raw experts and the chunk's buffers don't grow with c. They
       are counted in `gpu_bytes` at load and alloc, also in a dry run (`--ctx auto`).
   - **Catalog** (`tools/server_catalog.h`):
     - The mainline file is known as `inkling-2-mini` / `inkling`, with the engine's chunk of 512.
     - It is **listed but not supported** by the server yet: `m.supported` stays false, as for DeepSeek-V4.1.
     - The DSpark drafter (`dspark-*.gguf`, GGUF `dflash`) isn't taken as a drafter; it is step I2's work.
   - **What the server still needs**, before `m.supported` (a step of its own after the tiny file's KL passes):
     - **Prompts:** render them with `inkling::render` (src/chat_inkling.h). The effort comes from the request's
       `reasoning_effort` and the tools from `tools`.
     - **Generation:** no generation prompt, since the model writes `<|message_model|>`. Stop at token 200006 and at
       `<|end_message|>` only when the turn ends there.
     - **Streaming:** the thinking (`<|content_thinking|>`) as reasoning, the text as content, and the calls from
       `<|content_invoke_tool_json|>` via `inkling::parse`.
     - **Snapshots:** the prefix cache needs them, i.e. the 48 slots' K / V and their conv states saved and restored
       (`snapshot()` refuses now), and `save_state` for saved conversations (engine_sessions.h, out of this phase's
       files).
   - **Runs it waits on:** `serverz` lists the file as unsupported (no load); `runz --ctx auto` dry-run sizes on the
     real file (no GPU work beyond the dry plan).

**I1.7. Experts streamed, as the 35B-A3B: Inkling on the laptop and on 16 GB cards** (the user, 2026-11-23: "as
this is an MoE model I want it to work like 35B so this laptop can run it"). **Paused (the user, 2026-11-23: "pause inkling"), at this state:**
- **Done, compiling, not run:** step 1, the expert file's builder: `ExpertStore::float_to_q8` writes BF16 / F16 / F32
  experts as Q8_0 records (each row dequantized, padded to whole superblocks, `quantize_rows_q8_0`, `gpu_raw_sb`), with
  the recipe in the file's fingerprint only when set, so every other model's expert file keeps its name and bytes.
  `engine_stream.h` lets Inkling's float experts through and sets the flag. Nothing reaches it yet: Inkling does not take
  the streamed path (`hp.moe` / `stream_`) until steps 2-6.
- **Step 2 done (2026-11-23):** the expert cache follows the execution order. `ink_forward` keeps the order of the physical layers (`ink_order_`) and the one after the current (`ink_next_il_`), so the prefetch after a pass's layer 17 reads layer 6; `ink_seq_` gives the MoE path its token boundary (`all_in` / `ARC_XCPU`'s token count) instead of a layer index going down; `XCache::keep_front` keeps a prompt's experts of a layer that runs again in the forward at the list's front. Dormant until step 5 calls `layer_moe`. Checked: the 4B 709A33CF..., the 35B 1CFAD0B5... and the tiny Inkling FCD10220... the same before and after.
- **Step 3 done (2026-11-23):** the kernels' Inkling forms. `moe.cl -DROUTE_SIG` (with `-DRSCALE`): p = sigmoid(logit), chosen by p + bias, ties to the lower expert, weights p / (sum + 1e-20) x RSCALE, no shared expert (SGATE 0); with `-DESCALE` the down scale joins the weight. `-DMOE_SITU -DSITU_B -DSITU_LB`: situ after the gate-up scale (`esc_gu[e]`, the last kernel argument) in `moe_gateup_q`, `moe_rg` (DP4a and `RG_XMX`) and gemm.cl's `MOE_TILES` up pass. ocloc 26.40: every form builds for mtl, dg2-g10 and bmg-g31. Dormant until step 5 builds them for Inkling; the 4B, the 35B and the tiny Inkling the same bit for bit.
- **Step 4 done (2026-11-23):** Inkling's MoE input and output on the shared path, behind `ink_fast_moe_` (off until step 5). `ink_layer` quantizes its ffn_norm output into `aq_` / `asc_` (`quant_rows`) and keeps its f32 router in `ink_rl_`; `layer_moe` under `hp.inkling` skips the norm, the shared expert and the router GEMM, routes from `ink_rl_` at stride NE with `ink_route_defs` (ROUTE_SIG, ESCALE with the down scales, the bias), builds the gate-up kernels (`moe_gateup_q`, `moe_rg`'s up pass, decode + GEMM's up pass) with `ink_situ_defs` and the gate-up scales as their last argument, and combines into `ink_mo_` (zeroed). The tables default to ones / zeros where a layer has none. `ARC_XCPU` refuses Inkling for now. Nothing runs it yet: the fused gate-up tensor reaches these kernels only through the expert cache's records (steps 5-6).
- **Steps 5-7 done and run (2026-11-23):** streamed is the default (`hp.moe` / `moe_std` set for Inkling, `stream_` on, the expert width padded to whole superblocks as Flash Next's: 1152 in 1280); `layers_[il].ex_type` from the records' types; no raw expert copy on the GPU; `ARC_INK_RESIDENT=1` keeps I1.5's exact path. **The tiny file (`make_tiny_inkling2 --scale 0.3 --all-experts`, 40 positions against ref_forward):** resident (exact) prompt KL 0.00022 mean / 0.00047 max, decode path 0.00021 / 0.00049; streamed prompt 0.00043 / 0.00127, decode path 0.00044 / 0.00122 (the per-32 int8 activations and NVFP4's fold; the top tokens are near ties at ~0.005 in this random file, 35-36 / 40 on both). A cache holding 53 % of the experts (`ARC_XCACHE_GB=0.003`): 3,696 of 4,224 generation lookups missed and were read from the SSD, the logits the same bit for bit as with every expert cached. `ARC_INK_RESIDENT=1`: the same hash as before these steps (FCD10220...).
- **Left:** the real file (the expert file's 14.7 GB on C:, KL on a short prompt against ref_forward / llama.cpp, then speed), `ARC_XCPU` for Inkling, the server's support (I1.6's list) (the loop's `ensure()` per pass and the prefetch in pass order, `ROUTE_SIG`, `MOE_SITU` and the
  gate-up scale table, the combine into `ink_mo_`, the plan and the tiers, `ARC_INK_RESIDENT`), then the tiny file streamed
  against resident (a small cache, a cold start) and the real file.
- **Before it, I1.5 was run on the laptop (2026-11-23):** a crash fixed (`tok_embd_` unset); the tiny file's 3-pass loop with
  all experts at `make_tiny_inkling2 --scale 0.3` against ref_forward, token by token over 200 positions: KL 0.00017 mean,
  0.00089 max, 186 / 200 top tokens. The default random file's larger KL is its saturated router and controllers
  amplifying int8 rounding, and its top-2 near ties (`--all-experts`, `--scale`, `REF_INK_Q8`, `ARC_INK_DUMP` /
  `REF_INK_DUMP` were made to show it). The 4B and the 35B: the same logits before the Inkling commits and after.

Design; not written yet.

Why it fits: the experts are 18.3 GiB of the file's 21.3, and a token uses 8 of 128 per layer. The rest the GPU keeps
for good is ~1.2 GiB (attention, controllers, router, conv taps in Q8_0 / f32, the head in Q8_0) plus the KV cache.
The token embedding table stays in the mapped file and is read on the CPU, as for every model.

| | resident (I1.5) | streamed (I1.7) |
|---|---|---|
| expert bytes on the GPU | all 18.3 GiB, raw | an LRU cache of whatever room is left |
| expert bytes in total | NVFP4 10.7 + BF16 7.6 GiB | NVFP4 10.7 + Q8_0 4.0 = 14.7 GiB (the expert file) |
| the laptop (GPU share ~18 GiB) | does not fit | ~14 GiB of cache at 8K context: ~95 % of the experts, near-zero SSD reads once warm |
| an A770 / B580 / B50 (12-16 GB) | does not fit | 9-13 GiB of cache: 60-90 %, the rest from the SSD / RAM tier |

**What it reuses** (the 35B's path: `zero/expert_store.h`, `zero/xcache.h`, `zero/engine_stream.h`,
`zero/engine_moe.h`, `zero/kernels/moe.cl`, `zero/kernels/decode_moe.cl`):
- **The expert file:** `open_or_build` writes each (layer, expert) record in the GPU raw form next to the model in
  `%USERPROFILE%\.inferzero`, read later with unbuffered I/O, one record per read.
- **The cache:** the LRU pools by record size, with prefetch of the next layer's misses while the GPU works.
  A prompt chunk takes new experts at the back of the list, so a prompt doesn't sweep the cache. The 35B on this
  laptop runs this way today, with its experts on the SSD.
- **Fused gate_up:** expert records already hold gate then up rows (Gemma 4 26B's fused `ffn_gate_up_exps`), and the
  `-DXCACHE` kernels take the two offsets of one record.
- **NVFP4:** the I1.1 `decode32` branch is already in `MOE_Q` (generation) and `MOE_RG` (prompt tiles), with the
  per-32 fold. Its accuracy is checked here, against the exact I1.5 path.

**What is new** (all under `hp.inkling`, other models bit-identical):
1. **The expert file's builder:** BF16 experts (layers 20-23) requantized to Q8_0 as their records are written. The
   builder refuses float experts today ("the experts run in a quantized type only"). 4 layers x 2.04 GB -> ~1.1 GB
   each; KL against the exact path checks it. The builder's 14.7 GB on C: (never D:) is checked against free space
   first, as now.
2. **The loop:** a token runs layers 6-17 three times, all three passes on the same physical layer's experts. The
   cache tables stay per physical layer (24), so pass 2 and 3 mostly hit what pass 1 fetched. `ensure()` runs once
   per pass, which is 48 calls a token instead of 24, each after that pass's routing.
3. **The routing:** a sigmoid router with the selection bias, `w / (sum + 1e-20) x 8`. It becomes a `ROUTE_SIG`
   variant of `moe.cl`'s routing macros (`moe_route1`, `moe_route` / `moe_scan` / `moe_scatter`), as `ROUTE_DS` is
   DeepSeek's, and replaces `ink_route`. The router stays f32 (`ink_mm_f32`) into the routing kernels' logits buffer.
4. **The activation and the per-expert scales:** `-DMOE_SITU` in `decode_moe.cl`, situ instead of SwiGLU in
   `moe_gateup_q` and `moe_rg`'s epilogue.
   - The gate-up `.scale` multiplies both halves before the activation, so the kernels read it by expert id (a 128
     f32 table per layer).
   - The down `.scale` is linear, so it folds into the routing weight (`w x s_down[e]`, as `-DESCALE` does for Gemma).
5. **The input and the output:** the MoE input is Inkling's `ffn_norm(mix + controller)` in f32, quantized by
   `quant_rows` into `aq_` (the engine's path takes it from there). The combine writes into Inkling's own buffer
   (`ink_mo_`), not `x_`, as DeepSeek combines into `hco_`; the FFN's short convolution and the residual follow as in
   I1.5.
6. **The plan:** `plan_weights` / `size_expert_cache()` treat Inkling as a streamed MoE (`hp.moe = hp.moe_std = true`,
   `stream_`). The cache gets what the GPU has left after the resident part and the context (`ctx_bytes`:
   `ink_ctx_cache_bytes`), as for the 35B. `--ctx auto` gives the expert cache its share first and the context the
   rest.
7. **A switch:** `ARC_INK_RESIDENT=1` (or a card where everything fits, with the plan's choice) keeps the I1.5 path,
   which is exact and resident. It stays the accuracy reference for the streamed path on the tiny file.

**From the Researcher's review (2026-11-23):**
- **The loop's passes and misses:** each pass routes on its own input, so passes 2 and 3 may choose experts pass 1
  didn't.
  - On the laptop, with ~95 % cached, that's rare.
  - On a 12-16 GB card (60-90 %), each of the 48 `ensure()` calls a token can wait on the host, the cost QUEUE's
    `ARC_XCPU` host-wait row is about.
  - So the existing `ARC_XCPU` path (the CPU computes a missing expert from its RAM-tier copy instead of the GPU
    waiting for it) takes Inkling's MoE too: its CPU dot products exist for Q8_0 (`zero/xe2/cpu_experts.h`), NVFP4
    gets one (the per-16 exact form), and situ plus the gate-up scale join its SwiGLU on the CPU.
- **Prefetch in pass order:** the 35B prefetches the next *layer's* misses. In the loop, layer 17 of pass 1 is
  followed by layer 6 of pass 2, so the prefetch follows the order of execution (the slot sequence of `ink_forward`),
  not the layer index.
- **The down scale's order:** the combine computes `y = sum_k (w_k x s_down[e_k]) x down_k`. The product
  `w_k x s_down[e_k]` is formed first in f32 and multiplies the expert's output before the sum, as the reference's
  `w x (down x s)` does up to f32 rounding (a scalar product's order). The gate-up scale multiplies each gate and up
  value in the epilogue, before situ.
- **The expert file's name:** its fingerprint includes the record recipe (here: BF16 experts as Q8_0), so a change of
  recipe rebuilds the file instead of reusing a stale one.
- **Accuracy runs:** streamed against I1.5's exact path on the tiny file mixes NVFP4's per-32 fold and the BF16 ->
  Q8_0 requant. One more run with `--nvfp4-equal` (the fold then exact) shows the requant's KL alone.
- **The laptop's room:** 31.7 GB RAM; C: has 77 GB free for the 14.7 GB expert file.
- **Files:** `expert_store.h`, `xcache.h`, `engine_stream.h`, `engine_moe.h`, `moe.cl`, `decode_moe.cl`,
  `xe2/cpu_experts.h`, each change under `hp.inkling`. A change to `plan_weights` / `size_expert_cache` in
  `engine_sessions.h` (the Developer's file) goes through the Researcher.

**The memory tiers** (the user, 2026-11-23: "make sure for it to run with memory tiers design"; the tiers of
[tiered-placement.md](tiered-placement.md), section 1: compute where the bytes are). Inkling takes the 35B's tiers as
they are, with nothing of its own but the record recipe:

| Tier | Holds for Inkling | Who computes from it | Sized by |
|---|---|---|---|
| GPU, resident | attention, controllers, router, conv taps, head (~1.2 GiB); the 48 slots' K / V and conv states; the chunk's buffers | the GPU | the model (fixed) and the context (`ink_ctx_cache_bytes`) |
| GPU expert cache | the most recently used expert records, LRU per record-size pool, per physical layer (the loop's passes share it) | the GPU (`-DXCACHE` kernels) | what the GPU has left after the resident part and the context (`size_expert_cache`) |
| RAM tier (discrete cards) | expert records that miss the GPU cache: pinned host memory the GPU copies from in the background | `ARC_XCPU=1`: the CPU computes a missing expert from its RAM-tier copy while the GPU runs; otherwise copied into a victim slot | automatic: the free RAM less 16 GiB, or half the free RAM leaving 4 GiB; capped by what Windows lets the GPU map of RAM; `--expert-ram-gb` / `ARC_XRAM_GB` override |
| SSD (the expert file) | every expert record, 14.7 GiB, read unbuffered on a miss of both tiers; prefetched in execution order | — | the file: built once on C:, never D: |
| RAM, mapped file | the token embedding table (BF16, 0.9 GB) and the GGUF itself | the CPU (`embed`) | — |

- **On the laptop (iGPU),** the RAM tier is off by design (`!zrt::integrated()` in `engine_stream.h`): the GPU's
  memory is the RAM, so the expert cache itself is the RAM tier. Two tiers: the cache (~14 GiB of the GPU share
  holding ~95 % of the 14.7 GiB of experts) and the SSD (the rest, and the cold start).
- **On a discrete card,** all three:
  - B580 12 GB with 32 GB+ RAM: ~9.5 GiB cached; the automatic tier (~10 GiB on a machine with ~20 GiB free) holds the
    other ~5 GiB, so the SSD is read only at the cold start. On a 16 GB PC the tier's automatic size,
    max(free - 16 GiB, min(free / 2, free - 4 GiB)), leaves little or nothing (a B580 user's 15.4 GiB): there it is
    the GPU cache plus the SSD.
  - A770 16 GB: ~13 GiB cached, the tier holds the other ~2 GiB.
  - B60 / B65 / B70: every expert in VRAM.
  - With `ARC_XCPU=1` a miss in the GPU cache is computed on the CPU from the RAM tier (NVFP4 and Q8_0 CPU dots, situ
    plus the gate-up scale), so the loop's later passes don't wait on a copy.
- **The expert file's place:** next to the GGUF when the model is inside the data folder
  (`%USERPROFILE%\.inferzero`, or `ARC_DATA_DIR` when set), else in the data folder's `zero\models\<gguf name>\`, as
  for the 35B (`zero/expert_store.h`). The SSD tier is only as fast as that drive. A data folder on a slow or USB
  drive makes every miss slow; on this laptop the models and the data folder stay on C: (D: hangs under random
  access).
- **Fill order:** a prompt's experts go into the GPU cache's transient end, and into the RAM tier only where the tier
  can't hold every expert anyway (`ARC_XRAM_FILL`, the default since 2026-11-21). Generation's misses go to the front
  of the LRU, and to the tier.
- **The other tiers of tiered-placement.md** (one model over several GPUs, a layer split, the CPU computing dense rows
  `ARC_HOST_CPU`) stay refused for Inkling in this step, as in I1.5. A layer split would move whole cache slots with
  their physical layer's experts: a later step.
- **Checks in the run list:**
  - on the tiny file, a cache forced small (`ARC_XCACHE_GB`) with a RAM tier (`ARC_XRAM_GB`) and with none, so that
    every tier serves records;
  - the same top tokens across the three: GPU only, GPU + SSD, GPU + RAM + SSD with and without `ARC_XCPU`;
  - the expert statistics line (`expert_stats`: hits per tier, SSD bytes) on the laptop with the real file;
  - one cold start: the first prompt after a restart, the expert file not in Windows' file cache, the SSD tier's
    worst case, which warm runs would hide.

**Expected on the laptop** (estimates, to be measured; Xe-LPG, 75 GB/s, ~11 TOPS DP4a):
- **Generation:** ~3.6 GB read a token (attention 1.4 GB over the 48 passes, experts ~1.7 GB, head 0.5 GB): a ceiling
  of ~20 tok/s, likely 10-15 once the cache is warm, about the 35B's 21.6.
- **Prompt:** ~10 int8 GOP a token. The int8 GEMMs set a ceiling of ~1,000 tok/s, likely 300-500. Until then, I1.5's
  f32 attention kernel is the limit at long prompts: the prompt's attention work is I3's.
- **A cold start** reads the experts from the SSD as they are first chosen, like the 35B's first prompt.

**Steps:**
- **a.** The builder (Q8_0 for float experts).
- **b.** `ROUTE_SIG`, `MOE_SITU` and the gate-up scale table in the kernels (ocloc on mtl / dg2 / bmg).
- **c.** The engine: `hp.moe_std` and streaming for Inkling, `layer_moe` called from `ink_layer` with its input in
  `aq_` and its output in `ink_mo_`, the plan.
- **d.** The run list:
  - the tiny file streamed vs resident (KL, top token), with a cache smaller than the experts so that misses happen;
  - the expert file's build on the real model (14.7 GB written to C:);
  - the real model on the laptop through gpu_guard: KL against ref_forward on a short prompt, then speed.

**I2. The fastest preset for every card** (the user, 2026-11-23: "continue inkling 2 with fastest techniques and
configuration for every card"; targets: pp 6,000 on the B70, pp 4,000 on the A770, tg 130 on both, #83). Plan only: each item opt-in
until a card measures it, then that family's default; bit for bit, or KL <= 0.015 token by token over hundreds of
positions with the same top token at 98 % or more. Built on I1.7 (the streamed experts), whose path every card takes:
with all experts cached it is the resident path, with smaller Q8_0 records for the BF16 experts.

**What a token costs** (Inkling 2 Mini, the expert file's records; estimates to check against `--prof-gen`):
- **Bytes read:** experts ~1.7 GB (8 of 128 a pass, 48 passes, NVFP4 at 4.5 bits, Q8_0 in layers 20-23); attention,
  controllers and convs ~1.4 GB (the BF16 matrices as Q8_0, read in each of the 48 passes); the head 0.49 GB (200,058 x
  2,304, Q8_0). **~3.6 GB a token.**
- **The ceilings** at 100 % of the measured bandwidth: B70 (603 GB/s) ~167 tok/s, so 130 is 78 % of it; B65 the same
  memory; B60 / B580 (456 GB/s) ~127; A770 (~405 GB/s) ~112, so **130 on the A770 needs drafting**; the iGPUs
  (B390, 140T, the laptop's 125H: 75-120 GB/s) 20-33.
- **Launches:** the I1.5 graph runs ~25 small kernels a layer pass, ~1,200 a token. At ~5 us each that alone is ~6 ms,
  a ceiling of ~165 tok/s on any card. **Fewer launches is a lever on every card**, before any kernel gets faster.
  **So fusion gates 130 on the B70 too**, not only on the slower cards: its launch ceiling is below its bandwidth
  ceiling. The launch cost is measured per family with `--prof-gen` first (Alchemist's command lists make it cheaper).
- **The prompt:** ~10 int8 GOP a token, ~75 % of it in the experts (8 routed experts against ~20 M dense weights a
  pass). 6,000 tok/s is ~60 int8 TOPS of useful work: ~16 % of a B70's XMX peak (~367 TOPS), ~26 % of an A770's
  measured 232. Attention is cheap at 4K: 40 of the 48 slots are sliding windows of 1,024.
  - **The B70: reachable, but demanding.** Its 35B prompt reached 5,177-5,647 tok/s (4K / 8K), ~34 TOPS at ~6 GOP a
    token, ~9 % of its peak; Inkling needs ~1.8x that. In its favour: 8 of 128 experts (the 35B: 8 of 256), so each
    expert gets ~512 rows at chunk 8,192 and the XMX tiles fill better, and NVFP4 decodes more cheaply than IQ4_XS.
    6,000 needs the expert GEMM near the dense GEMMs' efficiency at chunk 8,192; the 35B-like baseline is measured first.
  - **The A770: the target is 4,000** (the user, 2026-11-23: "Use 4,000"; the 6,000 is the B70's). ~40 TOPS of useful
    work, ~17 % of its measured 232. Its 35B runs
    2,538 tok/s at 4K, ~15 TOPS (6.6 % of its peak) with DP4a experts, and its whole dense 27B prompt is ~41 % of peak.
    What gets it there:
    - **the 8-lane XMX expert GEMM near the big tile's efficiency**, with dpasw;
    - **no separate decode_w8 pass for the experts** (12-15 % of the 35B's prompt today): the NVFP4 rows decoded in the
      GEMM's own loads, or read raw into the XMX operands as `MOE_Q` reads them;
    - **routing, scatter and combine fused and small**;
    - **the chunk as large as the expert cache allows.**
    Each of these is measured against 4,000 in its before-after row, with the A770's `--prof` share beside it.

**The steps, for every card unless a card is named** (each its own commit and before-after row):
1. **Fused layer kernels (generation first).** The controller (`ink_rms2` + `ink_ctrl_apply`), the K / V rows and
   their short convs and the store (`ink_rows` + `ink_conv` + `ink_kv_store`), the depth mixes (`ink_mix`), the
   router and situ: each group one launch. Bit for bit against I1.5 where the sums keep their order, else KL. Target:
   under 10 launches a layer pass. With Alchemist's command lists (`ARC_BATCH=64`, its default) the launches left cost
   less there; B-series has no such default (it measured neutral), so fusion is what counts there.
   **First part done (2026-11-24), bit for bit:** the depth mix writes its rows where the layer reads them (no
   `ink_take`), each attention input quantized once for all its projections, the controller's gain / forget / apply in
   one kernel (`ink_ctrl_fused`), one token's conv and its state's move in one (`ink_conv1`), the q / r / k / v rows
   and the cache store in one (`ink_rows_kv`). The tiny file: ~905 -> ~634 launches a token (~57 -> ~40 a layer pass),
   80.9 -> 95.0 tok/s on the laptop. Left: `ink_rms2` with the controller's quantization, the router and the MoE glue,
   the norms, the block copies; `ARC_INK_FUSE=0` gives the fused groups' launches of before.
2. **The dense matrices on the engine's fast paths.** Prompt: gemm.cl on XMX (16 lanes) or gemm_x8 (8 lanes), as the
   Qwen models' non-H8 GEMMs; H8 is Qwen-only and stays so until its rotation is checked for Inkling's shapes (a later
   item). Generation: the GEMV defaults per family (B-series `ARC_GEMV_DB` / `PF`, Alchemist `ARC_GVROWS=1`), the
   per-pass matrices of a loop layer being the same weights three times, so passes 2 and 3 may hit L2 on the B70 (18 MB).
   The rank-64 controllers and the 24-row sinks stay f32 (`ink_mm_f32`) until a GEMV form takes their shapes.
3. **The experts, prompt:**
   - B-series and the B390 (16 lanes): decode_w8 + gemm.cl's XMX tiles (the 35B's default), against `moe_rg` on XMX
     straight from the NVFP4 rows (`ARC_MOE_RG_XMX`: slower for the 35B on a B70, but Inkling's experts are smaller and
     NVFP4's decode is cheaper than IQ4_XS's): measure both, keep the faster.
   - **A770 and the 140T (8 lanes): the 8-lane XMX expert GEMM**, the same kernel as the 35B's (queued after this
     plan: the A770's 35B spends 52-57 % of its prompt in DP4a experts and their decode, #45), with the big tile's split
     DPAS (dpasw) on DG2. Without it Inkling's experts run DP4a there, and 6,000 is out of reach.
   - The chunk: 8,192 on the B70 / B65 (room for it at 32 GB), 4,096 on the B60, 2,048 on the A770 / B580 (the expert
     cache's room first), 512-1,024 on the iGPUs; the plan's rule, `--chunk` overrides.
4. **The experts, generation:** `MOE_Q` reads NVFP4's raw rows (4.5 bits) and Q8_0, so no decode pass. The GEMV rows
   per sub-group (`ARC_MOE_GR_GU` / `_DN`) swept per family with `bench\decode_try.ps1`, as for the 35B.
5. **The head:** 0.49 GB of 3.6 a token (14 %). Q8_0 by default; a Q6_K / Q4_K head is an opt-in with its own KL run.
6. **Drafting with DSpark** (the drafter in the HF cache, GGUF arch `dflash`): the engine's DFlash2 path, taught the
   drafter's own architecture. Each verified token costs one main pass for several tokens, so it is the lever where
   bandwidth caps: the A770 (130 needs ~1.2 tokens a pass at 100 % of its bandwidth, ~1.6 at a realistic 75 %), the
   B60 / B580, the iGPUs. The drafts a step per family as `xe2_mtp_drafts` does for MTP (Alchemist 4).
7. **The memory tiers per card** (I1.7's design, sized by the plan):

| Card | Memory, bandwidth | Experts (14.7 GiB of records) | Prompt experts | Drafting | Notes |
|---|---|---|---|---|---|
| Pro B70 / B65 | 32 GB, ~603 GB/s | all in VRAM | 16-lane XMX tiles | DSpark for > 130 | chunk 8,192; the B65 has the B60's EU count on the B70's die: its own GEMV rules (`zrt::bmg_g21`) |
| Pro B60 | 24 GB, 456 GB/s | all in VRAM (~1.2 GiB + context left: ~6 GiB at 32K) | 16-lane XMX tiles | DSpark | chunk 4,096 |
| B580 | 12 GB, 456 GB/s | ~9.5 GiB cached, the rest in the RAM tier (a 32 GB PC) or the SSD | 16-lane XMX tiles | DSpark | a 16 GB PC: GPU cache + SSD; `ARC_XCPU` for the loop's misses |
| A770 16 GB | 16 GB, ~405 GB/s | ~13 GiB cached, ~2 GiB in the RAM tier | the 8-lane XMX expert GEMM (dpasw) | DSpark, needed for 130 | `ARC_BATCH=64`; a ~3-bit expert file only if the cache misses cost too much |
| B390 (Panther Lake) | shared RAM, ~120 GB/s | the cache is the RAM share | 16-lane XMX (where its checks pass) | DSpark | 2D loads off (as under Windows) |
| Arc 140T (Arrow Lake H) | shared RAM, ~120 GB/s | as the B390 | 8-lane XMX expert GEMM (no dpasw: an iGPU, unmeasured) | DSpark | the 140T preset's other rules |
| The laptop's 125H (Xe-LPG) | shared RAM, 75 GB/s | ~14 GiB cache, ~95 % of the experts | DP4a (no XMX) | DSpark | the correctness host: every step's KL runs here first |
| 2 x A770 (layer split) | 2 x 16 GB | every expert on a card, its layers' share: no RAM tier | 8-lane XMX expert GEMM on each | DSpark | the likeliest way the A770 family nears 130: hansong80's dual-A770 35B on a layer split ran 2.9-3.4x the prompt and 1.7x the generation of one card (#45) |
| 2-4 x Pro B60 (layer split) | 2-4 x 24 GB | every expert in VRAM with room for long contexts | 16-lane XMX tiles | DSpark | experts per card halve or quarter |

- **A ~3-bit expert file for the A770 (opt-in, its own KL run first):** at ~88 % cached a few percent of the experts
  miss; at ~1.7 GB of experts a token, a 3-5 % miss rate is 50-85 MB a token over PCIe (~20-25 GB/s), 2-4 ms against a
  ~7.7 ms token at 130 tok/s, and drafting touches more experts. A ~3-bit file (~11 GiB) fits whole and reads ~1.1 GB a
  token, raising the ceiling from ~112 to ~140. Requantizing NVFP4 lower may well break KL 0.015, so the KL decides;
  the RAM tier stays the default.
- **Multi-GPU:** Inkling's layer split is refused today (I1.7). The 2 x A770 and B60 rows need it: whole cache slots
  move with their physical layer's experts, and the loop's three passes over layers 6-17 stay on the card that holds
  them.

**Order:** I1.7 steps 2-6 first (the streamed path, the base of every row above), then 1 (launches, all cards), then 3
(the 8-lane XMX expert GEMM, shared with the 35B) and 6 (DSpark), then 2, 4 and 5 by what the cards' profiles show.
**Checks for every step:** its KL on the laptop first, and the tiny file's 3-pass loop against ref_forward (I1.5's
check: `make_tiny_inkling2 --scale 0.3 --all-experts`, 200 positions token by token, KL 0.00017 mean / 0.00089 max) as a
regression test, since the loop is where errors grow.
**What each card is asked for:** `runz --bench-pp 4096 --prof` and `--n 128 --prof-gen` on the real file,
then the A / B of each step against the default; the B70 and A770 owners first (#83).

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
| I1.5 | `runz --dump-logits` on the 4B and the 35B, before and after these commits | laptop iGPU | the other models bit for bit (the warm-up marker hashes the ops source, so each model warms up once after this update: time only, no output change) |
| I1.5 | the engine on the real file vs ref_forward / llama.cpp | a 32 GB card (B70, B65) | KL <= 0.015 token by token |

## Done for this phase

- I1.1-I1.6 written, compiling (the touched files only), and documented.
- A list of the runs each step waits on, ready for the user's go.
- No run of any kind.
