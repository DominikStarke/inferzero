# Contributing to inferzero

inferzero is far from what Intel's GPUs can do. On every card we have measured, prompt processing and generation
leave a large share of the hardware unused, and the cards and setups we have never seen are the biggest unknowns.
Help of two kinds moves it most: **measurements from your hardware** and **code**.

## Where the speed is still missing

Measured, not guessed (details in [docs/xe2-tuning.md](docs/xe2-tuning.md) and [docs/before-after.md](docs/before-after.md)):

| Hardware | What is measured | Where it could be |
|---|---|---|
| Arc 140T (Arrow Lake-H iGPU) | the prompt GEMM at ~33 % of the XMX int8 peak | ~85 % with its operands in L1 (`bin\xw_peak --probe`) |
| Arc Pro B70 | the prompt GEMMs at ~62 % of the int8 peak; generation at ~62 % of the memory bandwidth | int8 GEMMs at 80 %+ are known on this card; int4 GEMVs reach 89-97 % of bandwidth |
| Arc B580 (12 GB) | generation at 61-70 % of bandwidth on the 4B / 9B; Flash Next limited by PCIe copies of its experts | more of the bandwidth; a smaller expert cache waste; fewer copies |
| Several GPUs | on Windows the sums go through host memory; a slow link (PCIe x4) halves a split model's prompt speed | GPU-to-GPU sums where the driver allows them (done on Linux); a layer split for slow links |
| Arc A770 / A750 / A580 (Alchemist) | first runs only (issue #35) | almost everything is untuned |
| Core Ultra iGPUs (Meteor Lake, Lunar Lake, Panther Lake) | tuned on one 125H laptop; the B390 measured by one user | every other laptop |

The open work, in order, is in [QUEUE.md](QUEUE.md) ("Open now" at the top) and
[docs/arc-plan-2026-11-17.md](docs/arc-plan-2026-11-17.md).

## 1. Measure on your hardware (no code needed)

This is the most useful thing most people can do: we can only tune for a card someone has measured.

1. Download the latest release ([Windows zip or Linux tar.gz](https://github.com/rizagp/inferzero/releases/latest))
   and a model with the launcher (or `setup.cmd`).
2. Run the benchmark script. Windows: `run-tests.cmd quick` (about an hour; `run-tests.cmd` alone runs overnight). It
   writes one zip to the Desktop with the paths' user names taken out. Linux: `bench/website_bench.sh`.
   A single model by hand: `bin\runz.exe <model.gguf> --bench-pp 4096 --n 128 --prof`.
3. Open an issue with the **Hardware results** template and attach the zip or paste the output: the GPU, the driver
   version, the OS, the PCIe slot, the release.

Results for discrete cards go in [issue #4](https://github.com/rizagp/inferzero/issues/4) or a new issue of their
own. When a switch is asked for on an issue, a run with it and one without, alternated, is worth more than a single
number.

## 2. Code

The engine is C++20 on Level Zero, with kernels in OpenCL C. Start with [AGENTS.md](AGENTS.md): the rules every
change follows, and a map of the source.

**Build.** Windows: Visual Studio Build Tools (C++), then `.\build.ps1 zero\runz.cpp zero\serverz.cpp`. Linux: g++ 13
or newer, then `./build.sh` ([docs/install.md](docs/install.md)). Rebuild both runz and serverz after any change to
the engine or the kernels.

**The rules a change is held to:**
- **Accuracy.** Every model's next-token KL against the f32 reference (`tools/ref_forward.cpp`, `runz --kl-ref`) stays
  at or below 0.015. A change meant to give the same results is checked bit for bit: `runz --dump-logits` before and
  after, the files identical.
- **Discrete-card code** goes in `zero/xe2/` and kernel files of its own; multi-GPU code in `zero/mgpu/`. The iGPU
  path must stay bit-identical.
- **Every speed or accuracy change** gets a row in [docs/before-after.md](docs/before-after.md) (measured the same way
  before and after) and a CHANGELOG entry. A new switch starts opt-in until a card measures it faster.
- **Memory:** extra VRAM or RAM only opt-in or from the room left free at load, and the load log says what it took.
- **Nothing a user runs may need Python.**

**Good places to start:**
- the issues labelled [good first issue](https://github.com/rizagp/inferzero/labels/good%20first%20issue) and
  [help wanted](https://github.com/rizagp/inferzero/labels/help%20wanted);
- a kernel on a card you own: `bench/` has micro-benchmarks for the GEMMs (`xw_peak`; `gpu_sweep.ps1` / `.sh` sweeps every setting and tile with a model), the GEMVs
  (`gemv_bench`), attention (`attn_bench`) and memory (`bw_bench`);
- Linux: the port works headless (`serverz`); packaging for more distributions, and the Docker / Kubernetes files in
  the repository, are untested by us;
- the web UI (`webui/index.html`, one file) and the launchers (`scripts/`).

**Audit your change before each commit.** We review every pull request against this list. Running it yourself first
saves a round trip; most items take a minute on a small change.
- **Correctness:**
  - real bugs: wrong conditions, off-by-one, out-of-bounds reads or writes, integer or size overflow (`int` against
    `size_t`, 32-bit counters on a server that runs for weeks), races between threads or between GPU command lists;
  - context and memory math: buffer sizes, the memory plan, `--ctx` and chunk sizes, and what a new buffer adds to
    VRAM or RAM;
  - **the logits:** with your switch off, the dump is bit for bit the same as before (`runz --dump-logits`, the files'
    hashes equal), and a switch meant to give the same results is checked the same way with it on;
  - **where your switch changes the numbers**, the accuracy against the f32 reference (`tools/ref_forward.cpp`,
    `runz --kl-ref` and `--kl-decode`), token by token over a few hundred positions of a prompt and a generated answer:
    - the mean KL at or below 0.015, with the max reported;
    - the same top token at 98 % or more of the positions, leaving out the reference's own near-ties (its top two
      within 0.08 nats; the tools print both shares);
    - the greedy text the same as before over the first tokens, or the first position where it differs and why.
- **Failsafes and robustness:**
  - **failsafe:** what happens when a GPU hangs, a wait times out, one GPU of a split fails, or the driver lacks a call
    or extension. The engine must stop cleanly with a message, never hang or give wrong answers;
  - **try / catch** around allocations, driver calls and file reads that can fail, and the state they leave behind;
  - **guards** on every input and switch value (ranges, null pointers), and a clear refusal for GPUs, models or
    options the change doesn't support;
  - **memory leaks:** every buffer, event, kernel and command list freed on every path, the error and early-return
    paths included;
  - **no state left behind:** a start-up check or a test switch leaves the engine exactly as it found it.
- **Switches and defaults:**
  - a new switch is off by default until a card measures it faster, and on or off where the docs say;
  - its start line shows when it is in use, and the log says when it is set but left out (for example when a load-time
    check takes another path);
  - every caller of a changed default is checked, and the other GPUs and models stay unchanged;
  - your measured runs show the switch's start line, so each number is known to come from the path you meant.
- **Kernels:**
  - a kernel launched without a local size (`run(..., nullptr)`) must not depend on `get_local_size()` beyond
    size-agnostic loops;
  - no barriers in divergent control flow;
  - local memory sized for an explicit work-group size;
  - no register spills on the compiler users have (ocloc 26.40, `tools/kernel_isa.sh`) for DG2 and BMG.
- **Platforms and text:**
  - Linux and Windows build and behave alike;
  - every PowerShell script works in Windows PowerShell 5.1 and PowerShell 7 (`tests/test_ps_compat.ps1`), and every
    bash script works under `set -euo pipefail`;
  - comments that the change makes wrong or obsolete are updated;
  - messages go through the log (`LOG_ERR` / `LOG_WRN` / `LOG_INF`, [docs/logging.md](docs/logging.md)), not
    `fprintf(stderr)`.

**Pull requests.** Small and focused, with the numbers before and after from your hardware in the description. We
merge after checking the results on the hardware we have, and rewrite the CHANGELOG and docs text in the project's own
format, so don't spend time polishing those; the credit stays yours in the CHANGELOG and the commit.

## Talk

Open an issue for anything: a question, a result, an idea. Measurements that disagree with the docs are especially
welcome; they are how most of the tuning here was found.
