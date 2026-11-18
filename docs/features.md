# The best of inferzero

A tour of what inferzero does well, what each feature is for, how to use it, and what it measured. The numbers
come from [before-after.md](before-after.md) (how each was measured) and the [CHANGELOG](../CHANGELOG.md); the
test laptop is a Core Ultra 5 125H (Arc iGPU, no XMX) with 32 GB of RAM. The step-by-step guide is
[USAGE.md](../USAGE.md).

## 1. Fast prompts on an iGPU

Every matrix product runs on the iGPU's int8 DP4a units, the weights decoded once per chunk. Prompts are processed
2.7-7.4x faster than llama.cpp (Vulkan / SYCL) on the same GGUF files and laptop.

| Same file, same laptop | inferzero | llama.cpp Vulkan | llama.cpp SYCL |
|---|---|---|---|
| Qwen3.5 4B, 2K prompt | **911 tok/s** | 242 | 337 |
| Qwen3.8 27B, 2K prompt | **113 tok/s** | 23 | 25 |
| Qwen3.8 27B, 34K code prompt to the first token | **327 s** | 2,418 s | 1,624 s |

Why it matters: an agent's prompt (system text, tools, files) is thousands of tokens; on a laptop the prompt, not
the generation, decides how long each answer takes.

## 2. Models larger than the GPU memory

Mixture-of-experts models stream their experts from the SSD into a GPU cache while the GPU computes; only the
experts in use stay in memory. Qwen3.6 35B-A3B runs with a 131K-262K context; Qwen3.8 Flash Next (52 GB of experts)
runs on a 32 GB laptop; Gemma 4 26B-A4B likewise.

## 3. Speculative decoding that never changes the answer

The model's MTP block, the Gemma 4 assistant drafters and the DFlash2 block drafter guess several tokens; the model
checks them in one pass and keeps only those it would have chosen. Python code on the 35B runs at ~50 tok/s with
DFlash2, and the text is identical to plain decoding.

## 4. Accuracy measured, and the same output every time

- Every model is compared with an independent f32 CPU implementation: KL of the next-token distributions under
  0.015 (speech models under 0.01); perplexity at or below llama.cpp's on the same files (Qwen3.5 4B 6.056 against
  6.141).
- The same prompt gives the same logits bit for bit, run after run, whatever the expert cache size; every speed
  change is held to that (56 of 56 regression runs bit-identical).

## 5. Conversations that do not start over

A new message processes only itself:

- **Prompt cache**: the conversation's cached prefix is reused. Qwen's recurrent (DeltaNet) state is kept at the end
  of the prompt, the start of the model's turn and the turn start after the user's last message.
- **Gemma 4** (since 2026-10-31): a tool call is rendered without its reasoning every turn, so the prompt does not
  change before the tool results, and the snapshots keep a copy of the sliding-window rings, so the cache goes back
  further than the rings reach. After a 20K-token tool result, a one-line follow-up on Gemma 4 E4B took 0.8 s
  (38.4 s before).
- **Session cache**: other conversations are saved to disk and loaded when they come back (8 GB for all models
  together by default, set in the launcher or with `-SessionCacheGb`; other models' oldest go first); **system
  prompts** are kept on disk per project (Pi).
- **Two conversations at once** (`--slots 2`): their next tokens run in one engine pass (27B prose 7.3 -> 10.5 tok/s
  together, Gemma 4 26B 19.8 -> 26.2).

## 6. Long documents, read properly

With **Files** on in the web UI, the model reads the chat's files itself (DOCX, PDF, XLSX, PPTX, text, code,
Markdown, CSV), and every answer says which section or page it comes from:

- A document the model's context takes in one read comes **whole**, in one call (Qwen3.5 4B's 262K context: ~40K
  tokens; a 98 KB, 56-page report in one call).
- A longer one gives its **outline**; the model reads the sections or a range of pages it needs (`pages: "1-20"`),
  or finds passages with `search_file` (BM25).
- For **all** of a long document (a summary, a review, its weak points) **`digest_file`** has the server read it
  part by part with the loaded model, each section (or function, in source files) kept whole, and returns notes of
  every part with their pages. The notes are saved as `<name> (notes).md` in the chat's files, so the next
  question reads them at once. With a 16K context: README.md (57 KB) 3 parts in 3 minutes on Qwen3.5 4B; a 159 KB
  source file 4 parts on Gemma 4 E4B.
- **Scanned PDFs**: the loaded vision model transcribes the pages when they are read, and the text is kept with
  the file.
- The model also **writes** files: DOCX, PDF, RTF and XLSX from Markdown (tables become sheets, Rp amounts and
  dates become values), and any text format as written.

## 7. OCR: PDFs and images to text

An OCR model (Xiaomi-OCR-0, a Qwen3.5 0.8B with an image encoder) is asked the way it was trained, whatever the
client sends. Name the task in Indonesian or English (nothing: the text; `dokumen`: Markdown with OTSL tables shown
as tables; `tabel`, `rumus`, `json: tanggal, total`).

- A **PDF** is rendered by Windows page by page and read 50 pages a turn; the web UI goes on by itself to the last
  page (a 450-page manual in 9 turns).
- **Saved to a .md** (asked with each PDF): each page is appended as it is read, the chat shows a progress bar, and
  a chat model switched to later reads the file. A folder of your own works too (a copy stays in the chat's files).
- **Many files at once:** several images and PDFs sent in one message are read one after another, each under its
  own heading, into one merged .md or one .md per file. The 50-page limit counts over all of them, and the web UI
  goes on by itself to the last file.
- **From a project's folder:** a project's chat picks PDFs and images from the project's folder (several, or all)
  and reads them where they are; the .md files go into the project's `ocr` folder, which every chat of the project
  sees.
- ~1.4 s to encode a 2,000-token page image, 69-73 tok/s for its text.

## 8. Tool calls that stay well formed

While a model writes a tool call, only tokens that keep it valid for the tool's JSON schema are drawn (Qwen and
Gemma 4 formats): no lost required argument, no argument in the wrong place, enum values only. The rest of the call
is the model's own choice. The web UI's tools: web search and page reading, files, a PowerShell terminal in the
chat's folder (each command allowed by the user), chats search.

## 9. A web UI that searches, talks and transcribes

- **Web search** and page reading through the server (public addresses only). A search asks Brave Search, Bing,
  DuckDuckGo and Wikipedia at once and merges their results as SearXNG does (2026-11-16): a page several engines
  rank high comes first, and one engine's captcha or rate limit leaves the others' results.
- **Talk**: the microphone stays open, the answer is spoken sentence by sentence, can be interrupted, and looks
  things up while it talks (Gemma 4 E4B / 12B hear the audio themselves; Qwen3-ASR writes it down for any model).
- **Transcripts and Meetings**: audio and video files, recordings from the browser (microphone or a tab, mono or
  stereo, WAV, MP3 or M4A at a chosen bitrate) and meeting recordings, with speakers and times.
- **Queue and Steer** (2026-11-09): messages typed while the model answers wait and go in order when it ends;
  Steer puts one into the answer under way (before its next tool step, or the answer stops and it goes at once).
- **Speech to text only**: `serverz -m none` runs Qwen3-ASR / Whisper without a chat model (a 2:43 voice message in
  25 s); a chat model loads later on request.
- **From a phone, anywhere** (2026-11-05): through a Cloudflare Tunnel with a password, as an app on the home screen
  ([docs/remote-access.md](remote-access.md)). The terminal and folders stay off remotely unless you allow them.

## 9b. Memory and projects (2026-11-02)

- **Memory:** "remember that …" saves a dated note in `MEMORY.md` in the data folder. Every new chat starts with
  the notes, whatever the model. You can see and edit them in Settings > Memory.
- **Projects**: a project is a folder of yours (a Google Drive folder through Drive for desktop
  too). Its chats read the folder's files, follow its `AGENTS.md`, keep a memory of their own and open the terminal
  there. Paths outside the folder are refused, links and junctions included. ([USAGE.md](../USAGE.md), section 4)

## 10. More hardware

- **NPU next to the GPU** for prompts (`--npu on`): Qwen3.8 27B +15 %, Gemma 4 12B +10 %
  ([npu-experiment.md](npu-experiment.md)).
- **Several GPUs** for one model: Qwen dense, the 35B-A3B, Flash Next and DeepSeek-V4.1-Flash (`runz` / `serverz
  --gpus N`, tensor parallel):
  - each GPU reads 1/N of the weights per token, and the KL on 2-4 shards matches one GPU's;
  - batched slots work over the split;
  - failsafes cover a GPU that hangs.

  On 4 Arc Pro B60s the 27B generates 53 tok/s, 2.44x one card in the same run of 2026-10-31
  ([multi-gpu.md](multi-gpu.md)).
- **Discrete Arc cards** (experimental): the B-series (B580 / Pro B60 / Pro B70) and the A-series (Alchemist: A770 /
  A750 / A580, on the Arc 140T's preset since 2026-11-16). Their own switches in `zero/xe2/`, XMX for the prompt GEMMs after a start-up
  check, and scripts that measure each setting ([xe2-tuning.md](xe2-tuning.md)). Users measured the 27B's 4K prompt
  at 2,972 tok/s on a Pro B70, 1,726 on a Pro B60, 1,223 on an A770 (issues #27, #4, #36; README, "Discrete GPUs").
- **A dense model a little larger than the card** keeps part of its weights in RAM, read over PCIe: an Arc B580
  12 GB runs the 27B with 0.49 GiB of them there (`ARC_HOST_KEEP_GB=0.75`), a 4K prompt at 1,877 tok/s and 7.1 tok/s
  generation (issue #37).
- Xe2 / Xe3 iGPUs (Lunar Lake, Panther Lake) build the 8-lane kernels with 16 lanes; XMX engines are used after a
  start-up check where the GPU has them. Panther Lake's Arc B390 takes the discrete cards' H8 prompt path by default:
  the 27B's 4K prompt at 583-592 tok/s (issue #1).

## 11. Easy to start

A ready-to-use package (no compiler, no Python): unpack, double-click `zero-launcher.cmd`, pick a model that suits the
laptop's RAM, and the web UI opens. The launcher shows each model's memory plan, whether the options set fit, saved
presets and the running server's state and log, in Windows' light or dark mode with a tray icon (minimized, it waits
in the tray); `zero-server` does the same from a terminal (`list`, `start -Preset`, `-DryRun`, `logs`, `doctor`,
`pull`); one command connects the [Pi](https://pi.dev) coding agent. On Linux, `./zero-launcher.sh` is a terminal
menu that downloads a model and starts the headless server, with nothing but bash and curl, and
[docker/](../docker) holds Docker and Kubernetes files (not tested yet): [install.md](install.md#linux).
