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
- **The accuracy reference:** llama.cpp with **PR #32704** (Unsloth's perplexity 7.653 on Q8_K_XL). Read its code for
  every operation's exact order and constants; the engine must match it, not this summary.

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
5. **The engine's graph:**
   - `zero/engine_inkling.h` (hparams, the loader's tensor map, the layer loop with per-pass KV, the forward for one
     token and for a prompt chunk);
   - its kernels in `zero/kernels/ops_inkling.cl`: the relative-bias attention with sinks, qk clip and the log scale;
     the K / V short convs; the residual convs; the attention residuals; the controllers; situ; the softcap;
   - the MoE through the existing standard-GGUF expert path where it fits, with NVFP4 added there.
   - BF16 matrices take the engine's float requant to Q8_0, as other models do.
   - Explicit local sizes where a kernel's local memory or shuffles assume a group shape (AGENTS.md).
   - Waits on: the tiny model's KL on the laptop through gpu_guard, then the real file.
6. **Memory and fit:** ctx_bytes for the loop's per-pass KV (12 loop layers x 3 passes), the plan's estimate through
   the shared helpers (`ring_pos_bytes`, `swa_ring_cap`), and a catalog entry with the drafter.

**I2-I4. Speed, later, each needs cards** (from the analysis): tg on a B70 at ~83 % of its bandwidth, then DSpark
drafting; prompt with NVFP4 read straight in the expert GEMM and chunk 8,192; the A770 needs DSpark and a ~3-bit file or
the RAM tier, since the Q8 file doesn't fit 16 GB.

## Done for this phase

- I1.1-I1.6 written, compiling (the touched files only), and documented.
- A list of the runs each step waits on, ready for the user's go.
- No run of any kind.
