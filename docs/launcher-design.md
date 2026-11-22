# The launchers' redesign: one entry per model, its options as switches

**Built 2026-11-22** (QUEUE row 15, step 3), with the user's three decisions below taken as recommended: one row
per file with Drafting and Images as switches, the full-screen arrow-key menu on Linux, Advanced collapsed. Where the
build differs from this proposal:

- **Options go in as `-Opt` / `--opt`,** not as variant ids. zero-server matches `-Model` against the list's main
  entries only, and `-Opt` also keeps the choice in `options.json`. The server reads the same setting either way, and
  the Windows fit bar uses the variant's own context when one is on. Variant ids stay for Pi and API clients
  (step 4).
- **The switches that don't apply are folded** into one line, "Show the N that don't apply to this model or PC".
  Clicking it shows each with its reason. With all of them shown, the panel scrolled on a laptop screen.
- **The tip under the switches** has a fixed height of three lines; the tooltip holds all of it. The cost on its
  last line is cut at the first bracket or semicolon, since the catalog's cost column goes on with measurements.
- **The PC's words** are shortened in the reasons ("needs Arc A-series or Arc 140T"). On Linux they come from
  `lspci`; there a PC reason only warns on the command line and never refuses, since the names are a guess.
- **Linux has no fit bar:** the bash launcher has no memory estimate without `serverz --list-models`, which takes
  minutes the first time.
- **The rules' code:** `scripts/launcher-switches.ps1` (Windows) and `zero-launcher.sh`'s `sw_*` functions, both
  checked against `tests/launcher_switch_cases.tsv`.

A proposal for QUEUE row 15, step 3, to look at before the launchers are rewritten. It is built on the data that is
already in place: `scripts/options.tsv` (each option's tip, what it needs or can't be combined with, which models and
PCs it fits) and `serverz --list-models` (each model's variants, each with its own context).

The user's asks this follows: model options shown as part of a model, a tooltip on every option, which options need
or exclude each other, options that can or can't be chosen, ease of use, Windows and Linux.

## What changes for the user

1. **One row per model file.** Today the Windows list has up to six rows for one file (MTP + images, MTP, images,
   standard, MTP block only, DFlash2 only). These become switches of that one model.
2. **The model's options sit next to it,** as switches with a short tip each. An option that doesn't apply is shown
   greyed with the reason, never hidden.
3. **Options that exclude each other are one choice,** shown as a row of buttons where only one can be on.
4. **A fit bar** shows whether the model, its context and its options fit the GPU, and updates as switches change.
5. **"Changed from the defaults"** lists what you turned on or off, with a Reset button.

## Windows (WinForms)

```
+- inferzero ------------------------------------------------------------------------------------+
| Models                                  | Qwen3.8 27B GSQ IQ3_S                    13.2 GB file   |
|-----------------------------------------|-----------------------------------------------------|
| > Qwen3.8 27B GSQ IQ3_S    13.2 GB  fits |  Drafting       [ Off | MTP | MTP + DFlash2 ]  (i)  |
|   Qwen3.8 27B QSA Q4_K_M   16.8 GB  fits |  Images         [x] on                           (i)  |
|   Qwen3.8 Flash Next IQ3_S 52.1 GB  RAM  |                                                       |
|   Qwen3.5 4B Q4_K_XL        2.9 GB  fits |  Speed and accuracy                                   |
|                                         |   [x] Fast drafts (common tokens)                (i)  |
|                                         |   [ ] Copy drafting                              (i)  |
|                                         |   [ ] More precise prompt reading                (i)  |
|                                         |   [-] Attention: sparse   for QSA models only    (i)  |
|                                         |                                                       |
|                                         |  Context  (*) auto  ( ) [ 32768 ]                     |
|                                         |  GPU  [#########################.......] 13.9 / 16.0 GiB |
|                                         |        weights 10.9 | context 1.7 | buffers 1.3        |
|                                         |                                                       |
|                                         |  Changed from the defaults: Fast drafts     [Reset]   |
|                                         |  > Advanced (port, network, slots, presets, ...)      |
|-----------------------------------------+-----------------------------------------------------|
| Server: off                                                      [ Start ]  [ Open web UI ]  |
+-----------------------------------------------------------------------------------------------+
```

- **Left:** the models, one row each, with size and whether it fits ("fits", "RAM" for a MoE model whose experts
  stream from RAM, "too big").
- **Right:** the selected model's switches.
  - A choose-one group, such as Drafting, is a segmented control.
  - A plain option is a check box.
  - `[-]` is greyed, with its reason written next to it.
- **The fit bar** is split into weights, context and buffers, against this GPU's memory. It turns amber above 90 %
  and red when it won't fit, with the reason under it ("a smaller context, or Images off").
- **Advanced** is collapsed by default. It holds today's fields: port, network access, slots, presets, budget.

### The tooltip of an option

Each switch has a tooltip, and the same text shows under the switch while it has the keyboard focus:

```
Fast drafts (common tokens)
With MTP drafting: each draft looks only at the 32,768 most common tokens, which makes drafting faster for English
and code. Every draft is still checked against the full vocabulary, so the text stays the same. Chinese text drafts
less well.
Needs: Drafting on (MTP).          Same output.          Costs: 69 MB of GPU memory.
```

- The first lines are the catalog's `tip`.
- The last line comes from its other columns:
  - `relates` in words: "Needs ...", "Goes with ...", "Not with ...". It reads "Works on its own" when empty.
  - "Same output" or "Output checked against full precision".
  - The cost.

### Greyed, with the reason

The reason is always one short line, and the most specific one wins:

| Why it doesn't apply | The line shown |
|---|---|
| its parent is off (`needs:`) | `[-] Copy drafting  turn Drafting on first` |
| it excludes a chosen one (`not:`) | `[-] Multi-GPU sums  not with Multi-GPU by layers` |
| the model doesn't take it (`applies` model words) | `[-] Attention: sparse  for QSA models only` |
| this PC doesn't take it (`applies` PC words) | `[-] More precise prompt reading  needs an Arc GPU with XMX or the 140T` |
| one GPU only (`needs:gpus2`) | `[-] Multi-GPU by layers  needs 2 or more GPUs` |

A greyed switch can't be clicked, but its tooltip still explains it. Turning the parent on enables it at once.

### Choose-one groups

| Group | Choices | Shown for |
|---|---|---|
| Drafting | Off, MTP, MTP + DFlash2, DFlash2 | models with an MTP block or a drafter |
| Attention | the model's default, Sparse, Full | QSA models (the QSA 27B, Flash Next) |
| Multi-GPU | One GPU, Split by tensors, Split by layers | PCs with 2 or more GPUs |
| Images | on / off (a check box) | models with an image projector |

The groups come from the catalog's `or:` relations and from the model list's flags.

## Linux (bash, 80 columns)

```
 inferzero  Qwen3.8 27B GSQ IQ3_S  13.2 GB                       GPU 13.9 / 16.0 GiB
 -------------------------------------------------------------------------------
   Models                          Options for the selected model
 > Qwen3.8 27B GSQ IQ3_S    fits     Drafting   (MTP + DFlash2)   [Off|MTP|MTP+DF]
   Qwen3.8 27B QSA Q4_K_M   fits     Images     [x]
   Qwen3.8 Flash Next       RAM      [x] Fast drafts (common tokens)
   Qwen3.5 4B Q4_K_XL       fits     [ ] Copy drafting
                                     [ ] More precise prompt reading
                                     [-] Attention: sparse   for QSA models only
                                     Context    auto
 -------------------------------------------------------------------------------
 Fast drafts: each draft looks only at the 32,768 most common tokens; the text
 stays the same. Needs: Drafting on (MTP). Same output. 69 MB.
 Changed from the defaults: Fast drafts
 -------------------------------------------------------------------------------
 up/down move  tab models/options  space switch  enter start  r reset  ? help  q quit
```

- **Keys:** arrows move, Tab switches between the model list and its options, Space toggles a switch or steps a
  choose-one group, Enter starts the server. Number keys still work for the model list, as today.
- **The line under the lists** is the tip of the focused switch, wrapped to 80 columns.
- **Without colour** (`NO_COLOR`, a dumb terminal, or output not to a terminal), the same screen in plain text:
  `[x]` / `[ ]` / `[-]` and `>` carry the state, never colour alone.
- **Non-interactive use is unchanged:** `--model`, `--opt`, `--no-opt`, `--options`, `--install-service`. An option
  that doesn't apply is refused there with the same reason line, for example "--opt copy_draft: turn --mtp on first".

## Rules both launchers share

- **One source:** the switches, tips, greying and groups all come from `scripts/options.tsv` and
  `serverz --list-models`. Neither launcher keeps its own list of options.
- **What a switch does when you Start:** a variant option starts the model under its variant id
  (`qwen3.8-27b-mtp-fastdraft`, which the server loads with that option, step 2). Other options go in as today, as
  an environment variable or a flag. Choices are kept in `~/.inferzero/options.json` as now, so earlier choices still
  apply.
- **Defaults stay the engine's:** a switch left alone stays "default", so a later release's better default reaches
  you. "Changed from the defaults" lists only what you changed.
- **Accessibility:** every control reachable by keyboard, each with a name a screen reader reads (its title plus its
  state, for example "Fast drafts, on"), and layouts that hold at 150 % and 200 % display scaling.
- **Tests:** `tests/test_launcher_options.ps1` already checks the catalog. Step 3 adds a check that every switch the
  launchers draw has a tip, and that each greying rule gives the expected line for a set of models and PCs.

## Not in this step

- **Pi's `/model` list:** step 4, once the launchers show variants as switches. Pi then gets one entry per model with
  its variants.
- **New options:** any added later appear in both launchers by adding a catalog row, with no launcher code.

## Decisions for the user

1. **The current six rows per file:** fold them into one model row with the Drafting and Images switches, as above?
2. **Linux:** a full-screen menu with arrow keys, or keep today's numbered prompts and add the options screen's
   switches and tips there?
3. **Advanced collapsed by default?** Port, network access, slots, presets and budget would move under it.

## The automatic preset: "Automatic (best measured)" (built 2026-11-22)

The launchers' default preset applies `scripts/best-setups.tsv`: settings measured faster than the engine's
defaults on some kind of PC and model, each with its evidence. Adding a row needs no code change.

- **A row's columns:**
  - `pc`: the PC words it needs, all of them (`scripts/pc-kinds.ps1`; `igpu-ptl` is Panther Lake's B370 / B390);
  - `models`: the catalog shortcuts it is for, or `*`;
  - `os`: `windows`, `linux` or `any`;
  - `needs`: `gpus2same` (two or more discrete cards of one kind) or `notfit` (the model's weights partly in host
    memory);
  - `settings`: in zero-server's names (`Opt=<options.tsv id>`, `Npu=on`, `Gpus=all`, `NoMtp`);
  - `why`, `evidence`, `confidence`.
- **What goes in:** only settings that beat the defaults in measured runs, with an accuracy cost within the KL
  limits and no extra memory unless there is free room. Settings that depend on the text or on the number of users
  (DFlash2 by language, `--copy-draft`, `--slots`) stay out.
- **Windows launcher:**
  - the preset list starts with "Automatic (best measured)", then "(none: the engine's defaults)", then the saved
    presets;
  - on each model, Automatic sets what its rows say on every setting not changed by hand, and shows each reason
    ("best measured: ...") beside the switch and in "Changed from the defaults";
  - choosing another model, Reset or another preset undoes what it set.
- **Linux launcher:** the same rows when a catalog model starts, each named in a "best measured:" line; `--no-auto`
  leaves them out. A row whose needs it cannot tell (`notfit`) is left out there.
