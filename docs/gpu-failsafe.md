# GPU failsafe: hangs, lost devices and error handling

Status (2026-11-23): steps 1-4 of the order below built, with two changes from the design, and the rest still to do.

- **Built:**
  - `gpu_fatal` and the exit codes 3 / 4 / 5 (`src/gpu_fail.h`, `ZE_CHECK`, the launch and wait paths in `zero/clrt.h`);
  - the wait limit with the last kernel's name;
  - the hang marker, `runz --clear-hang`, `ARC_IGNORE_HANG`, `ARC_GPU_FAULT_TEST`;
  - `bench/gpu_guard.ps1` / `.sh`;
  - the launchers: no restart on 3 / 4, and the marker printed. That covers `zero-server.ps1` after a stop while
    loading, `zero-launcher.sh` and its systemd unit (`RestartPreventExitStatus=3 4`), and the docker entrypoint,
    which stays up idle instead of being restarted.
- **Changed from the design:**
  - The limit is **off by default on every GPU** until the longest real waits are measured (`ARC_WAIT_STATS=1`,
    section 2). A wait with a limit may also poll a CPU core in the driver, where the unlimited wait sleeps, and
    that needs its own check on the iGPU. `gpu_guard` sets the limit for every test run instead.
  - The OpenCL path (`src/clrt.h`, which `attn_bench` uses) checks the marker but has no wait limit. `gpu_guard`
    covers it.
- **Not built yet:**
  - the `catch (...)` audit and one top-level `catch` in every `main` (section 1);
  - a split model's shards ending through `gpu_fatal` after their own limit (they still throw `GpuWaitError` as before);
  - the server's `503` to open requests;
  - `--edge` in `attn_bench` / `xw_peak` (section 6).
- **Not run:** the tests of section 7 wait for the user's go-ahead.

Written after two forced restarts of the development laptop on 2026-11-23. A new attention kernel (X8_SLM R3) hung at a 20-token test. Windows did not reset the GPU, and more GPU work was then sent to it. The Core Ultra 5 125H's iGPU also drives the screen, so the screen froze. The second restart followed Modern Standby while the machine was busy.

## Goals

1. A hang is noticed, named (the kernel and the program), and the process ends in seconds, not never.
2. One hang stays one hang: nothing in inferzero starts GPU work again until the machine restarts.
3. Errors are handled the same way everywhere: what can be recovered is caught where the work started; what can't ends the process with a clear message and a distinct exit code.
4. No cost on the normal path: the same results bit for bit, and no measurable time.

What it cannot do: if the driver can't stop a kernel that never finishes, the first hang may still freeze the screen until the process exits or the machine restarts. The failsafe keeps that to one, and says which kernel it was.

## 1. Error classes

Every failure is one of three kinds, each with one way of handling it.

| Kind | Examples | Handled by |
|---|---|---|
| **Request error** | bad JSON, prompt longer than the context, unknown model, a file that won't open | `throw` a `std::runtime_error`, caught where the work began: the server fails that one request and goes on, and a tool's `main` prints `error: ...` and returns 1 |
| **Resource error** | out of GPU memory while loading or growing a buffer | `throw GpuError{OutOfMemory}` from the allocation calls only. The loader may catch it and retry with a smaller plan (as it does now); everywhere else it is a request error |
| **GPU fault** | a wait past its limit (a hang), device lost, any other Level Zero failure mid-work | **never caught to carry on**, since the device's state is unknown. One function, `gpu_fatal(kind, detail)`, ends the process (section 3) |

The one exception is a split model (`zero/mgpu/`): a shard's wait past its limit still throws `GpuWaitError`, which `TpEngine` catches to stop the other shards, as now. After that it calls `gpu_fatal` too: the split model is not trusted again.

### try / catch rules

- Catch by `const std::exception&` (or a subclass), never by value.
- `catch (...) {}` that swallows silently is not allowed. Each existing one (for example in `tools/server.cpp`) either logs `LOG_WRN` with what was being done, or is narrowed to the exception it expects.
- No GPU call runs in a `catch` block or a destructor after a GPU fault (the driver's cleanup can hang). `gpu_fatal` uses `_Exit`, not `exit`.
- A `catch` that recovers says in a comment what state it leaves behind (as `run_steps` does with `bind(cur_slot_)`).
- Every program's `main` has one top-level `catch (const std::exception&)`: `LOG_ERR("error: %s")`, return 1. `runz` has it; `serverz` and the tools get the same.

## 2. One wait, with a limit (`zero/clrt.h`)

Today `sync()` and `wait_guarded()` have two paths: an unlimited wait, and 100 ms pieces with a limit used only by split models. Tidy them into one function, `wait_for(what, fn)`, so every list, event and queue wait goes through it.

- **No limit:** one call with `UINT64_MAX`, as now (zero cost).
- **A limit:** 100 ms pieces. Each piece re-reads the limit and the split model's abort flag. Past the limit: `on_timeout(what)`, which is `throw GpuWaitError` on a split model's shard and `gpu_fatal(Hang, what)` otherwise.

The limit (`ARC_GPU_WAIT_S`):

| GPU | Default | Why |
|---|---|---|
| integrated (it drives the screen) | **120 s** | the longest real single wait must stay well below it: measured first (below) |
| discrete | off | a user's long prompt with weights paging over PCIe can legitimately take minutes; set it to turn it on |
| split model's shards | as now (`TpEngine`'s limit) | unchanged |

- **Before 120 s becomes the default:** `--prof` and `ARC_FIT_PARTS`-style diagnostics print the longest single wait of the run. It is checked on the 125H at its longest cases (the 27B at its longest context, a speech file, the 35B MoE prompt). If any comes within 4x of 120 s, the default goes up.
- **The last kernel launched:** each launch stores its kernel's name pointer in `Dev` (one store, no string copy). The hang message names it: `GPU work did not finish within 120 s (the last kernel launched: attn_x8, in attn_bench)`.

## 3. `gpu_fatal`: the one way out after a GPU fault

`gpu_fatal(kind, detail)`, in `zero/clrt.h`, used by `ZE_CHECK` (through `ze_fail_hook`, which exists), by `wait_for`'s timeout, and by device-lost results:

1. `LOG_ERR` one line: the kind, the detail, the last kernel, the program.
2. Writes the hang marker (section 4) for hangs and lost devices. An out-of-memory error at a Level Zero call mid-work doesn't write it.
3. Flushes the log, then `_Exit` with a code per kind:

| Exit code | Meaning |
|---|---|
| 1 | an error as today (request or other) |
| 3 | the GPU's work did not finish within the limit (a hang) |
| 4 | device lost (the driver reset the GPU) |
| 5 | another Level Zero failure mid-work |

The launchers (`scripts/zero-server.ps1`, `zero-launcher.sh`, and the docker entrypoint) don't restart the server on 3 or 4. They print the marker's text and "restart the computer before using the GPU again". A restart loop on a hung GPU is what made today worse.

The server answers the requests still open with `503 GPU failure: <kind>` before exiting, if it can without touching the GPU: a flag the HTTP threads check, and a 1-second grace. No GPU call is made after the fault.

## 4. The hang marker: one hang stays one

The file is `%USERPROFILE%\.inferzero\gpu_hang.txt` (on Linux `~/.inferzero/gpu_hang.txt`), written atomically (a temporary file, then a rename), as `key=value` lines:

```
time=2026-11-23T03:51:40Z
program=attn_bench
kernel=attn_x8
kind=hang
gpu=Intel(R) Arc(TM) Graphics
boot=<this boot's id>
```

- **The boot id:** on Linux `/proc/sys/kernel/random/boot_id`. On Windows the boot time (now minus `GetTickCount64`), counted as the same boot within ±2 minutes.
- **The check:** every program built on `clrt.h` checks the marker when it opens the device, before creating a queue. A marker from this boot means it refuses, exit code 3:

  `inferzero: a GPU hang at 03:51 in attn_x8 (attn_bench): restart the computer before using the GPU again (or delete <path>, or set ARC_IGNORE_HANG=1)`
- **Clearing it:** a restart clears it by itself (a marker from an older boot is deleted with an info line). `runz --clear-hang` deletes it.
- **A marker that can't be read** is ignored with a warning, never treated as a hang: a corrupt file must not lock anyone out.

## 5. `bench/gpu_guard.ps1` / `gpu_guard.sh`: the watchdog for test runs

For our own GPU tests, mostly the laptop's:

```powershell
.\bench\gpu_guard.ps1 -TimeoutS 60 -- .\bin\attn_bench.exe --edge
```

1. **Refuses to start** when the marker is from this boot, or when another GPU job is running. That means one of the GPU programs (`runz`, `serverz`, `attn_bench`, `xw_peak`, `gpu_bench`, `ref_forward`), or a lock file `~/.inferzero/gpu_guard.lock` held by a live PID. A lock whose PID is gone is stale and is removed.
2. **Keeps the machine and the screen awake** for the run: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)`, a request of this process only, not a settings change. Released at the end, also on Ctrl+C (`try / finally`).
3. **Runs the command** with its output to the console and a log file, and returns its exit code.
4. **On timeout:** kills the process tree. If it is still alive 10 s later (stuck in the driver), it writes the marker with `kind=hang, killed=no` and says so; no further runs start.
5. Exit code 3 or 4 from the command means the marker is there: it prints it and stops.

It has to work in PowerShell 5.1 and 7 (`tests/test_ps_compat.ps1`), and the bash form the same on Linux. On the laptop, every GPU test by the Developer or the Researcher runs through it.

## 6. New kernels with barriers or roles: the edge cases first

The R3 hang was a case no sweep reached first: 20 tokens, shorter than one span.

- `attn_bench --edge` and `xw_peak --edge`: N = 1, 15, 16, 17, 20, 31, 32, 33 and 65; a tail where `M_valid` isn't a multiple of the token group; one span; a partial last block. Each runs as its **own process** under `gpu_guard -TimeoutS 20`, smallest first. One that fails or times out stops the list.
- A kernel loop whose end depends on data (`while`, spin waits) gets an iteration cap, and the result is checked at start, as the bit-for-bit checks are now.
- A static check before any run: every barrier is reached by every work-item on every path. That means counting barriers per path, and no `return` or `continue` that skips one in some sub-groups but not others. It goes into the review checklist.

## 7. Testing the failsafe without a real hang

- `ARC_GPU_FAULT_TEST=hang`: `wait_for` acts as if the limit passed on the first wait of the first prompt. Expected: the message, the marker, exit code 3, and the next start refused.
- `ARC_GPU_FAULT_TEST=lost`: the same path as a device lost (code 4).
- `gpu_guard` with a command that sleeps past `-TimeoutS` (no GPU at all): the kill, the marker, the refusal.
- The results unchanged with the default limit on: `runz --dump-logits` the same before and after on the 4B, and the 27B's 4K prompt time the same within noise.

These run only with the user's go-ahead, with the user at the machine.

## Order

1. The error classes and `gpu_fatal` (sections 1 and 3), the `catch (...)` audit, and the exit codes in the launchers.
2. `wait_for` with the limit and the last kernel's name (section 2); the longest-wait measurement before the 120 s default.
3. The hang marker (section 4).
4. `gpu_guard.ps1` / `.sh` (section 5), and from then on every laptop GPU test through it.
5. `--edge` in attn_bench and xw_peak (section 6). Then R3 again.

The steps come in one or two commits, with a CHANGELOG entry, a note in `docs/lessons-intel-arc-igpu.md` (what happened, what the failsafe does), and `docs/logging.md` for the new messages.
