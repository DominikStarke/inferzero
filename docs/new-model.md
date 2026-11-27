# Adding a model to inferzero

A model in inferzero is more than its forward pass. It runs on every Intel GPU kind the engine supports, from every
weight type the engine reads, on one GPU or split over several, behind the server's features. A new model has to do
all of that before it is listed. This page says what that means, where the code goes, and how it is checked.
[CONTRIBUTING.md](../CONTRIBUTING.md) section 2 holds the rules every change follows; [docs/pull-requests.md](pull-requests.md)
says how a pull request is reviewed and merged.

## 1. Before any code: an issue

Open an issue, not a pull request. In it:
- **the model and why:** how it does against the models inferzero already runs, in real use (coding, reasoning, tool
  calls) rather than only benchmark tables. We decide on quality first: LFM2 (#94) was fast but weaker in use and
  wasn't taken;
- **its graph's source:** where the forward pass is defined (for example llama.cpp's `src/models/<name>.cpp`), and
  the GGUF files people use (the quantizations, the uploaders);
- **what is new in it** against the models here: a layer type, an attention form, an activation, routing.

Wait for a yes on the issue. A model that differs only in sizes from one inferzero runs (a new Qwen3.5 dense size)
may need no engine code at all: say so in the issue.

## 2. What a model must support before it is listed

Each item is checked by the reviewer before the merge. A model may be developed in stages on its branch (section 5),
but it merges only when every row holds, or the issue records an agreed exception with its reason.

### Weight types

Every type the engine reads today (`src/quants.h`, `is_w8_type`), on every matrix of the model:
- **K-quants:** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K;
- **i-quants:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL;
- **block types:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_0;
- **4-bit floats:** MXFP4, NVFP4;
- **packed formats:** AutoRound's AR4_128, the GSQ 2-bit experts;
- **plain floats:** BF16, F16 and F32 matrices, which go through Q8_0 or stay f32 where the engine keeps them.

Through the engine's matrix calls, these come for free: a model that calls `matmul` / `matmul1` gets every type and
every GPU path below. What has to be checked is the model's own steps around them, on the files people actually use.
Test at least the uploaders' common mixes (unsloth's UD-Q4_K_XL / UD-Q8_K_XL, bartowski's or ggml-org's Q4_K_M /
IQ4_XS / Q8_0), and one file per type family the model's own kernels touch.

### GPU kinds

The model runs on each, with the engine's default path there. Each kind is tested on a contributor's card when we
don't have it (section 4):

| Kind | Examples | What differs |
|---|---|---|
| Xe-LP iGPU | Iris Xe, UHD (Tiger / Alder / Raptor Lake) | DP4a, no 256-register mode (watch for spills) |
| Xe-LPG iGPU | Core Ultra 100 (Meteor Lake), 200S (Arrow Lake-S) | DP4a; H8 on for Qwen3.5-type dense models |
| Xe-LPG Plus | Arc 140T (Arrow Lake-H) | 8-lane XMX, its own preset |
| Xe2 / Xe3 iGPU | Arc 140V (Lunar Lake), B390 (Panther Lake) | 16-lane XMX, H8 from 256-token chunks |
| Alchemist | A770, A750, A580, A380 | 8-lane XMX, `dpasw`, `ARC_BATCH` lists |
| Battlemage | B580, B570, Pro B50 / B60 / B70 | 16-lane XMX, H8, the wide-tile GEMM |
| Data Center | GPU Max, Flex | a few defaults of their own (`xehpc_on`); must load and run correctly |

### One model over several GPUs

- **Tensor parallel** (`--gpus N`, same cards): N = 2, 3 and 4. Every new tensor of the model is split or replicated
  in `zero/mgpu/` (`engine_tp_shard.h`), and its sums go through the existing ones (`tp.cl`).
- **Layer split** (`--gpus N --split layer`): whole layers per GPU, including **unlike cards** (a B60 next to a B70, an
  A770 next to a B580), where tensor parallel is refused.
- **Without several GPUs:** `ARC_TP_SIM=1` runs every shard on one GPU, for correctness. It's how a contributor with
  one card checks both splits. A split's logits are compared with one GPU's: bit for bit where the sums keep their
  order, else by KL.

### Memory plans

- **Decoded weights resident, decoded just in time per chunk, and part of the weights in host memory** (the plans the
  engine picks by the GPU's memory; `--budget`, `ARC_HOST_WEIGHTS_GB` to force them).
- **`--ctx auto`:** the model's per-token cache size (`ctx_bytes`, `zero/engine_slots.h`) is exact, so the automatic
  context fits on 6 GB to 32 GB cards and on iGPUs.
- **Mixture-of-experts models:** the expert file on the SSD, the RAM tier, the GPU's expert cache, CPU experts
  (`ARC_XCPU`) and the OS file cache rule (`ARC_XBUF`). Never steer routing toward cached experts or skip experts.

### Server features

- **Several conversations at once** (`serverz --slots N`, batched decoding): `runz --bench-batch N` says the batched
  logits equal each conversation run alone.
- **Prompt cache and saved sessions:** a snapshot restored gives the same next tokens (`runz --snap-test`).
- **Drafting** where the model has a drafter or MTP head: MTP, DFlash2, DSpark, copy drafting. The greedy text with
  drafting equals the text without it.
- **Vision and audio** where the model has encoders: the next section.
- **The chat template, tool calls and thinking** (`--think-budget`), through the OpenAI API and the web UI.
- **The launchers and the model catalog** (`tools/server_catalog.h`, `scripts/model-catalog.ps1`, `zero-launcher.sh`):
  the model is listed, downloads, and starts with the right defaults.

### Encoders: vision, audio, speech

A model that reads images, audio or speech brings an encoder: a projector file next to the model (`mmproj-*.gguf`),
or a speech model of its own. The ones here are:
- **vision:** Qwen3-VL's ViT (`zero/vision_qwen3vl.h`) and Gemma 4's (`zero/vision_gemma4v.h`);
- **audio:** Gemma 4's conformer (`zero/audio_gemma4a.h`) and Qwen3's (`zero/audio_qwen3a.h`);
- **speech:** Qwen3-ASR (`zero/asr_qwen3.h`) and Whisper (`zero/whisper_engine.h`).

Their kernels are `ops_vit.cl`, `asr.cl` and `whisper.cl`. A new encoder, or a new model using an existing one, must
support:
- **Every GPU kind** of the table above, and every type the projector files come in (F16, BF16, Q8_0, F32).
- **Several GPUs:** the projector loads and runs on a split model, `--gpus N` and `--split layer` (a gap today, #103,
  that a new model doesn't add to).
- **The input side,** on Windows and Linux alike:
  - images: decoding (Windows' imaging component, `ffmpeg` on Linux: `src/image.h`), the resize to the encoder's
    grid (`resize_bicubic`, which matches Pillow), the token budget per image (`fit_size`'s minimum / maximum
    pixels), several images per message, and PDFs as pages;
  - audio: decoding (Media Foundation on Windows, `ffmpeg` on Linux: `src/audio.h`), resampling to the encoder's rate,
    the log-mel features, clips longer than the encoder's window.
- **The context:** an image's or a clip's tokens are counted in `--ctx auto`'s plan and in the per-request limits, so
  a message with images never runs out of context halfway.
- **The server:** OpenAI's `image_url` (data URLs and files) and `input_audio`, the web UI's attachments, and the
  prompt cache and sessions with images in them.
- **Accuracy against an f32 reference of the encoder** (`tools/gemma4v_ref.cpp`, `gemma4a_ref.cpp`, `ref_qwen3a.cpp`,
  `ref_whisper.cpp`, from the model's own modeling code):
  - the encoder's output per token: the relative error, its worst token, the lowest cosine;
  - then the text model's KL over the answer with the image or clip in the prompt, within the limits of
    CONTRIBUTING.md section 2;
  - **speech models below KL 0.01** (AGENTS.md), and silence and noise transcribed as empty text.
- **Speed:** the encoder's time per image or per second of audio, against llama.cpp's `mtmd` on the same file, in
  `docs/<model>.md`.

### Switches

There are over 300 `ARC_*` switches. A model must work with every switch that applies to the layer types it uses,
or the engine must refuse that switch for it at load, with a message naming the switch. "Works" means correct output:
the same as without the switch where the switch promises bit-identical results, within the KL limits otherwise.
`bench/gpu_sweep.ps1` / `.sh` runs the switches with a model; its runs on the model are part of the review. A
registry of the switches with the layers and GPUs each applies to is planned
([docs/plan-2026-11-30.md](plan-2026-11-30.md) section 6). When it exists, the model's row in it replaces this
paragraph.

## 3. Where the code goes

Inkling 2 Mini is the most recent model added, and the example to follow (`zero/engine_inkling.h`).

- **`zero/engine_<model>.h`:** the model's part of the Engine class, included inside the class body. Everything that
  is only this model's lives there: its hyperparameters (`load_hparams_<model>`), its weights, and its layer's forward
  pass.
- **One hook per place outside it,** each a short call under the model's flag (`if (hp.<model>) ...`), all of them
  listed at the top of the model's file:
  - the architecture name in `load_hparams` (`zero/engine_setup.h`);
  - `build_kernels` / `load_weights` (`zero/engine_load.h`);
  - `alloc_buffers` (`zero/engine_alloc.h`);
  - `forward` / `run_layer` (`zero/engine.h`);
  - `ctx_bytes` (`zero/engine_slots.h`).

  Nothing else in those files changes.
- **Several GPUs:** the shard's tensors in `zero/mgpu/engine_tp_shard.h`, its server calls in `tp_mirror.h`, the
  layer split's stage boundaries where the model needs them.
- **Kernels: reuse first.**
  - **Matrix products** go through `matmul` / `matmul1`. They choose the GEMM / GEMV for each type and GPU, XMX and
    H8 among them.
  - **Norms, quantization, attention, DeltaNet, and the MoE routing and expert kernels** exist in `gemm.cl`,
    `decode.cl`, `ops*.cl` and `moe.cl`. Read them before writing one.
  - **Only steps no existing kernel covers** (a new activation, a norm of the model's own) go in a new part,
    `zero/kernels/ops_<model>.cl`, added last to `KernelParts::kOps` (`zero/clrt.h`): a part builds together with
    the parts before it.
  - **Discrete-card or multi-GPU variants** of the model's own kernels go in `zero/xe2/` and `zero/mgpu/` with their
    own files, as every such kernel does.
- **The f32 reference:** `tools/ref_forward.cpp` runs the model (a `tools/ref_<model>.h`, as `ref_inkling2.h`).
  Every accuracy check is measured against it, so a model without one doesn't merge.
- **An encoder:**
  - a new one gets its own file next to the existing ones (`zero/vision_<name>.h`, `zero/audio_<name>.h`), built on
    the Engine's matrix calls as they are;
  - its kernels go in `ops_vit.cl` (vision) or a part of its own;
  - its f32 reference goes in `tools/<name>_ref.cpp`;
  - a model that reuses an existing encoder only adds its projector's type to that encoder's loader.
- **The server:** the chat template and special tokens if they are new; the catalog entry and its model words.
- **`docs/<model>.md`:** the graph's source, what runs, the numbers per GPU kind, and any agreed exception.

## 4. What not to do

- **No new GEMM or GEMV kernel, and no copy of an existing kernel with small changes.** If an existing kernel needs a
  variant, add a `-D` build option to it: off by default, and the kernel bit for bit unchanged without it.
- **No change to another model's results.** A Qwen3.5 dense model, a mixture-of-experts model and Gemma 4 give the
  same logits before and after the branch (`runz --dump-logits`, hashes equal, on one GPU and under `ARC_TP_SIM` with
  `--gpus 2`). The hashes go in the pull request.
- **No new `ARC_*` switch** unless a measurement needs one. Fewer is the goal.
- **No silent gaps.** Whatever doesn't work yet on the branch is refused at load with a clear message, never left to
  give wrong output.
- **No register spills** in the model's kernels on the compiler users have (ocloc 26.40: `tools/kernel_isa.sh`), on
  tgllp, mtl, arl-h, lnl, dg2-g10, dg2-g11, bmg-g21 and bmg-g31.
- **No Python,** conversion scripts included, and no new library dependencies.

## 5. Order of work and testing

1. **Correctness on one GPU:** one conversation, the KL against the f32 reference within the limits of
   CONTRIBUTING.md section 2 (mean KL at or below 0.015, the max reported, the same top token at 98 % or more of
   positions outside the reference's near-ties), over a prompt and a generated answer of a few hundred tokens each.
2. **Every weight type** of section 2, file by file: load, KL, `--dump-logits` stable from run to run.
3. **The server features:** slots (`--bench-batch`), snapshots (`--snap-test`), drafting (greedy text equal), the
   template and tool calls.
4. **The encoders,** where the model has them:
   - the encoder against its f32 reference;
   - then the answer's KL with an image or a clip in the prompt;
   - several images per message, a long clip, a PDF;
   - on Windows and Linux, on one GPU and split.
5. **Several GPUs:** `ARC_TP_SIM=1` with `--gpus 2 / 3 / 4` and `--split layer`, against one GPU. Then on real cards
   (a contributor's, if needed).
6. **Every GPU kind** of section 2: the KL and a `--bench-pp 4096 --n 128` line per kind, from contributors' cards
   for the kinds we don't have. Ask on the issue; GPU runs go through `bench/gpu_guard.ps1`.
7. **Speed:** after all of the above, in later pull requests: fusing the model's steps, its own tuning per GPU kind.

Large work lands as a series of pull requests on the model's branch, each reviewed. The model is listed in the
launchers and the README only when section 2 holds.

**A redesign is planned:** one shared GEMM layer, the steps around it fused per GPU, and fewer switches. It will
change how `engine.h` calls the matrix products and how the kernel files are laid out. To keep the model's rebase
small, keep everything model-specific in its own files behind the hooks, call `matmul` / `matmul1` and the engine's
helpers rather than kernels directly, and rebase on `main` often (docs/pull-requests.md).
