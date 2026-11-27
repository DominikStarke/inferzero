<!-- Please read docs/pull-requests.md first. A new model: docs/new-model.md. Delete what doesn't apply. -->

## What and why

<!-- The change, the reason, and the issue it closes (Closes #...). -->

## Where it applies

<!-- GPU kinds, models, weight types and modes (one GPU, --gpus N, --split layer, slots, drafting) this changes.
     And that everything else is unchanged. -->

## Accuracy

<!-- Same results expected: `runz --dump-logits` before / after and the hashes.
     Numbers move: KL against the f32 reference (mean, max, top-token share), CONTRIBUTING.md section 2. -->

## Speed

<!-- Before / after, the same command on the same machine, alternated runs (A B B A).
     GPU, driver, OS, release or commit, the command, and the start lines of the runs. -->

## Spills

<!-- Kernel changes: tools/kernel_isa.sh with ocloc 26.40 on the families it builds for. -->

## Not tested

<!-- What you couldn't check. -->
