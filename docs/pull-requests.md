# Pull requests

How a change gets into inferzero: what to do before opening a pull request, what it must contain, how it is
reviewed, and how it is merged. [CONTRIBUTING.md](../CONTRIBUTING.md) section 2 holds the rules every change is held
to and the audit list. A new model follows CONTRIBUTING.md section 3 as well.

## 1. Before you start

- **An issue first** for anything beyond a small fix: a new model, a new kernel, a new switch, a change to a default,
  a new dependency. Describe the problem and your plan, and wait for a reply. It saves you work that wouldn't merge.
- **Small fixes** (a typo, a crash with an obvious cause, a script that fails in one PowerShell version) can go
  straight to a pull request.
- **Read the code you change and its callers.** Most of the engine's files start with a comment on what they hold and
  why; the comments next to a default say which measurement set it.

## 2. One pull request, one change

- **One topic.** A fix and a speed change are two pull requests, even in the same file.
- **Small.** Under a few hundred lines where possible. A large change lands as a series, each one building and
  passing its checks on its own.
- **From an up-to-date `main`.** Rebase, don't merge `main` into your branch. Rebase again before the review and
  before the merge.
- **The redesign:** one shared GEMM layer, the steps around it fused per GPU, and fewer switches are planned. They will
  move how `zero/engine.h` calls the matrix products and how the kernel files are laid out. Keep new code behind the
  engine's own calls (`matmul` / `matmul1` and the helpers) and in files of its own, so a rebase stays small. A pull
  request that conflicts with the redesign is rebased before merging; ask in it if a conflict isn't clear.

## 3. What the pull request contains

The template GitHub shows when you open one asks for all of this:
- **What and why,** and the issue it closes.
- **Where it applies:** the GPU kinds, models, weight types and modes (one GPU, `--gpus N`, `--split layer`, slots,
  drafting) it changes, and that everything else is unchanged.
- **Accuracy:**
  - a change meant to give the same results: `runz --dump-logits` before and after, the hashes;
  - a change that moves the numbers: the KL against the f32 reference (mean, max, top-token share), as
    CONTRIBUTING.md section 2 says.
- **Speed:** before and after, measured the same way on your hardware. The command, the GPU, the driver, the release
  or commit, and several alternated runs (A B B A), not one each.
- **The start lines** of your measured runs, so each number is known to come from the path you meant.
- **Spills:** for a kernel change, the compiler's spill check on the families it builds for (`tools/kernel_isa.sh`
  with ocloc 26.40).
- **What you didn't test,** said plainly.

## 4. Review

- **Every pull request is read against CONTRIBUTING.md's audit list.** Running it yourself first saves a round trip.
- **The reviewer re-runs** the accuracy checks on the hardware we have (an iGPU) and, where the change touches them,
  under `ARC_TP_SIM=1` for several GPUs.
- **Findings come as review comments,** each with what to change. A pull request may take several rounds; that's
  normal, not a judgment of the work.
- **A change for a GPU kind we don't have** waits for a run on such a card: yours, or another contributor's from the
  issue thread.

## 5. Merging

- **Merged only after it's verified:** our checks (bit-identity or KL, the other models unchanged), and a run on the
  hardware the change is for.
- **CHANGELOG, before-after and docs text** are rewritten by us in the project's format, so don't spend time on
  them. The credit stays yours, in the CHANGELOG and the commit.
- **A new switch merges off by default.** It becomes a default only when cards measure it faster, in a later change.
- **Releases** are cut from `main` by the maintainer; a merged change goes out with the next one.

## 6. Things that don't merge

- a new GEMM or GEMV kernel, or a copy of an existing kernel with small changes (add a `-D` option to the existing one);
- a change to another model's or GPU kind's results that isn't the point of the pull request;
- register spills on the compiler users have;
- Python in anything users run or the build needs, or a new library dependency;
- routing changes that favour cached experts or skip experts in a mixture-of-experts model;
- a default turned on without measurements from the cards it applies to.
