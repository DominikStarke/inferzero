# "Device lost" while recording a meeting (Qwen3-ASR), and what `ARC_ZE_SYNC_EACH=1` really says

**How it ended (2026-11-10, QUEUE row 1):** the access violations in `ze_intel_gpu64.dll` were the argument cache
keeping a freed allocation, fixed in `zrt::mem_free`. The diagnostic of section 5 became `ARC_ZE_WATCH=ms`
(CHANGELOG). The `DEVICE_LOST` crashes were TDR resets: none came back after `TdrDelay=10`. The analysis below is
the reading of the code that led there.

Static analysis only: no server and no GPU job was started (a `serverz` of the user was running). Every claim below
names the file and line it comes from, and says whether it is **proven from the code** or a **candidate** that needs
one measurement.

Where it stands (QUEUE 31.13): `serverz` dies with `DEVICE_LOST`, five times on 2026-11-04, every time while
Qwen3-ASR decoded a part of about 160 s; no thread race was found in the speech path; `ARC_ARG_CACHE=0` has been
stable since; the next step was the crash sequence again with `ARC_ZE_SYNC_EACH=1` to name the kernel. This document
is that reading of the code.

---

## 1. What the error actually is

`zeCommandListHostSynchronize` returning `0x7000000e` (`ZE_RESULT_ERROR_DEVICE_LOST`) means **the GPU context is
gone**, not that one kernel returned an error. On this stack there are only two ways to get there, and both are worth
telling apart because the fixes differ:

| What happened on the GPU | What the driver does | How to tell |
|---|---|---|
| A kernel touched a device virtual address with no mapping (a freed buffer, a pointer that is not a device VA, a read past an allocation that hits an unmapped page) | the context is marked lost; the next host wait returns `0x7000000e` | the failure is immediate and reproducible with the same input |
| A single command ran longer than Windows' TDR budget (2 s by default) | WDDM resets the adapter (`Display driver igfx stopped responding and has successfully recovered` in the Windows event log), then the context is lost | the screen blinks, the whole desktop stalls for ~2 s, and the event log has the TDR line |

Either way the engine ends the process on purpose (`_Exit(1)` in `zero/clrt.h:514-579`): a lost device cannot be
recovered inside the process. So the goal is not "recover", it is "do not fault".

The device-lost that happens while Qwen3-ASR decodes a long part is **not** in the meeting/recording plumbing: the
audio path (`tools/zero_recorder.cpp`, `tools/server_whisper.h` `live()`) only touches the disk and the CPU, and the
GPU part of a transcription is `q3asr::transcribe_ranges` (`zero/asr_qwen3.h:246`) on the ASR engine. The fault is
in or around a kernel launch of that path.

## 2. What `ARC_ZE_SYNC_EACH=1` does — and why it cannot be trusted to name the kernel

`zero/clrt.h:563-573`:

```cpp
static const bool sync_each = getenv("ARC_ZE_SYNC_EACH") != nullptr;
if (sync_each) {
  const ze_result_t s = zeCommandListHostSynchronize(zrt::D().cl, UINT64_MAX);
  if (s != ZE_RESULT_SUCCESS) { fprintf(stderr, "Level Zero error 0x%x running %s (...)\n", s, k->name.c_str(), ...); _Exit(1); }
}
```

Three things follow, and they change the *result* of the experiment, not only its noise:

1. **The switch is `!= nullptr`, so `ARC_ZE_SYNC_EACH=0` also turns it on** (`proven`). Anyone who "turned it off"
   with `0` still ran with a host wait after every launch. This must be `getenv() && atoi(getenv()) != 0`, as
   `ARC_ARG_CACHE` and the others are.
2. **It waits for the whole command list, which is shared.** All three engines in the server (the chat model, Whisper,
   Qwen3-ASR) and every server thread append to **one in-order immediate command list per GPU**
   (`zrt::Dev::cl`, created in `ClRt::init`, `zero/clrt.h:666`). `zeCommandListHostSynchronize` waits for everything
   previously appended to that list, by any engine and any thread. So the name printed is "the launch this thread had
   just appended", not necessarily the launch that faulted: on an in-order list the fault can be in an *earlier*
   kernel, and the wait is also waiting for other engines' work.
3. **It changes the submission pattern**: one submission per kernel instead of one long run of appends and a wait at
   `clFinish`. That is exactly why Windows' TDR budget (see §1) sees a per-kernel time with the switch on and a whole
   step with it off, and why a fault that used to be reported at the next `clEnqueueReadBuffer` is now reported at
   the kernel. The device-lost is **not caused by `ARC_ZE_SYNC_EACH`**; the switch moves the report to the launch and
   makes the crash deterministic (the host stops before the next 300 launches get appended).

Consequence for the experiment: "with `ARC_ZE_SYNC_EACH=1` it loses the device" is *expected* if the fault is real,
and it still does not name the kernel reliably. §5 gives a diagnostic that does.

Side effect worth knowing before the next recording test: with a host wait after every launch, one decode step of the
speech models goes from ~miliseconds to hundreds of host round trips. The live draft falls behind, and
`kLiveMaxPending` (`tools/server_whisper.h`, the "behind" flag of the audio route) starts dropping utterances, so a
recording made with the switch on loses its live text even when nothing crashes.

## 3. The one command list, and who is allowed to use it

**Proven from the code:**

- `zrt::sync()` (`zero/clrt.h:133`) and `clReleaseMemObject` (`zero/clrt.h:463`) wait on `D().cl` only. The **second
  command list** (`zrt::copy_list()`, `zero/clrt.h:157`, on a copy engine or a second compute queue; used by the
  expert prefetch's deep path, `zero/xcache.h:583-600`) is only ever waited on by the prefetch thread itself
  (`xcache.h:604`). Every "this memory can be freed now" argument in the code is therefore half an argument: nothing
  in `zrt::sync()`, `clFinish`, `clReleaseMemObject` or `Engine::release_all` waits for the copy list.
- The only thing that keeps two threads off the one immediate list is `Server::with_engine` (`tools/server.cpp:299`,
  the engine's mutex), reached through `wh->gpu_turn` (`tools/server.cpp:1312-1316`). That closure is
  ```cpp
  wh->gpu_turn = [&](const std::function<void()>& f) {
    std::shared_lock<std::shared_mutex> lk(swap_mu);
    if (cur && cur->srv) cur->srv->with_engine(f);
    else f();                                    // <- no lock at all
  };
  ```
  With **no chat model** (`-m none`, "Speech only"), or in the window where `cur` is a model that is being switched,
  `f()` runs with no engine lock: only `Service::turn_` (`EngineTurn`, `tools/server_whisper.h:1822`) is left, and
  that covers Whisper/ASR work but not the expert-prefetch thread of a chat model and not the model load/free path.
  Level Zero does not allow one immediate command list to be used from two threads; the list is also mutated by the
  model switch (`Engine::release_all`) and by `XCache::shrink()`.
- `XCache::shrink()` (`zero/xcache.h:222`) is reached through the **process-wide** OOM hook
  (`zrt::oom_hook`, set in `zero/engine_stream.h:73` to the *currently loaded chat model's* cache) from **any** thread
  whose allocation or launch returns out-of-device-memory — including the Qwen3-ASR thread while it allocates its
  4 slots / encoder (~3.7 GiB, `tools/server_whisper_asr.h:24-80`). It frees a 1 GiB piece of the chat model's expert
  pool (`zeMemFree`) and rewrites the expert address table from that foreign thread. `XCache::active()` itself is a
  plain, non-atomic static, written by the chat engine and read by that hook.
- `zeKernelSetIndirectAccess` is set only for programs built with `-DXCACHE` (`zero/clrt.h:824`), i.e. the MoE
  expert kernels. Any other kernel that reaches memory which is not one of its arguments gets no such promise from
  the driver. The comment there records that this exact omission was a "device lost now and then, in the server".

## 4. Where in the recording path this becomes a fault

Ranked by how well each one fits "always while Qwen3-ASR decoded a ~160-s part" and "stable with `ARC_ARG_CACHE=0`":

### 4.1 A device buffer that is freed (or handed away) while a command still references it — **primary candidate**

- The ASR path allocates at group boundaries: `AudioQwen3A::reserve` (`zero/audio_qwen3a.h:266`) frees ten buffers and
  makes bigger ones the moment a group has more token rows than before (`cap_`). A ~160-s part is 20 encoder windows,
  i.e. two groups (16 + 4 windows) with `Mp` of 1,664 then 416 rows, where the 25-s parts of the same meeting are
  416 rows: the first long part is exactly the moment `reserve` frees and reallocates.
- `Engine::free_buf` (`zero/engine_load.h:48`) drops the `from()` sub-buffers *before* the parent, and
  `clReleaseMemObject` waits on `d.cl` — but **not** on `D().cl_copy`, and `free_buf` is also reachable while a deep
  prefetch of a chat model is in flight on the copy list. That copy list survives across models (it lives in
  `zrt::Dev`, `zero/clrt.h:157`).
- `zeMemFree` on a VA that a queued or in-flight command still uses is the textbook way to produce
  `DEVICE_LOST` (not a ZE error code you can catch).
- Why `ARC_ARG_CACHE=0` helped: with the cache, an argument that is *the same value* as at the last launch is not set
  again (`zero/clrt.h:485-511`), so a buffer whose device address is unchanged keeps the old value — and a *freed*
  buffer's address can be handed back by the allocator to a *new* buffer, which then looks "unchanged" to the cache
  only if the value is identical. That is not a proof of the cache being wrong, but it is a real timing difference on
  exactly this code path, and it explains why a stale-argument bug would show up as a fault rather than as wrong text.

**Fix (both, they are cheap and independent):**
1. Make the "free is safe" rule true for every list: in `zrt::sync()` also wait on `D().cl_copy` when it exists
   (`zeCommandListHostSynchronize(D().cl_copy, ...)`, guarded by `if (d.cl_copy)`), and use `zrt::sync()` instead of
   the bare `zeCommandListHostSynchronize(d.cl, ...)` in `clReleaseMemObject`.
2. In `Engine::free_buf`, keep the `views_` cleanup, and make `clReleaseMemObject` of a *parent* invalidate (release)
   every sub-buffer `ZMem` that points into it, so a stale view can never be passed to a kernel again. Today
   `zrt::D().mems` keeps sub-buffers whose parent is gone; `from()`'s own guard only protects against views of slot
   buffers.

### 4.2 Two threads on the one immediate command list — **secondary, but it is the only explanation that ties the crash to the server state**

- The lock is missing exactly when nobody thinks about it: `-m none` (speech-only, the mode used for meetings) and
  during a model switch (§3). Add the prefetch thread and the OOM hook and there are paths where a chat-model command
  stream and a speech command stream can be interleaved on one list. Level Zero's immediate command lists are
  documented for one user thread at a time; a driver that is used that way is a good source of both faults and hangs,
  and it would explain "5 times, always while a long part decoded" without any wrong result in the text (the crash
  comes before a wrong answer is noticed).

**Fix:** a device-wide lock in `zrt`, not one per engine:
```cpp
// zero/clrt.h, namespace zrt
inline std::mutex& gpu_mu() { static std::mutex m; return m; }
struct GpuLock { std::lock_guard<std::mutex> g{gpu_mu()}; };
```
taken by `Server::with_engine` (instead of / in addition to the engine mutex), by `Service::turn_`'s users when there
is no chat model, by `XCache::prefetch_deep`'s thread for its appends, by `Engine::release_all`, and by
`XCache::shrink()`. A debug counter (`if (inside.fetch_add(1)) abort();`) makes the next such bug loud instead of
silent. Keep the per-engine mutex: it still serializes the *host* side (slots, caches).

### 4.3 A kernel whose loops are bounded by an argument that can be wrong

- `attn_fa_g` (`zero/kernels/ops_attn_g.cl:227`) loops `for (int b = bs0; b < bs1; ++b)` and `attn_fa_gh`
  (`ops_attn_g.cl:452`) goes from `lo/16` to `hi/16`, where `hi = pos0 + m` (or `nc_hi` inside a sliding window).
  A wrong `pos0` (a stale argument, an out-of-range position) does not fault — it makes the kernel do millions of
  extra blocks, i.e. **a TDR hang that reads as device lost**. For Qwen3-ASR `pos` is bounded by
  `u.room = min(max_new, ctx - pos - 1)` (`zero/asr_qwen3.h:343`), so an engine-side bound is not the likely source;
  a *stale argument* is. This is why the argument cache is still worth a hard look, and why §5's diagnostic must print
  the arguments of the launches around the failure, not only the kernel name.

### 4.4 The argument cache itself: sound in one thread, unproven across engines

`clSetKernelArg` (`zero/clrt.h:485`) caches the last value per (kernel, index) and assumes Level Zero copies the
arguments at append time (the code comment says so). Two things to fix regardless of the crash:

- `if (sz == sizeof(void*) && v)` treats **every 8-byte argument** as a possible `cl_mem`: a `uint64_t` offset, a
  `double`, a device VA passed as an integer is read as a `ZMem*` and looked up in `D().mems`. It works today only
  because a coincidence is improbable; a pointer-valued scalar that happens to equal a live `ZMem*` would be replaced
  by that buffer's device address. Pass the ABI explicitly (a `cl_mem` overload of `set_args`) instead of guessing
  from the size.
- the cache is per `ZKernel`, and `ZKernel`s are per `Engine`, so the ASR engine, Whisper's engine and the chat
  engine never share one. Keep it that way explicitly: assert that `k->args.size()` is only touched by the thread
  that owns the engine (the device lock of §4.2 covers this once it exists).

If the crash disappears with `ARC_ARG_CACHE=0` and nothing in §4.1-4.3 explains it, the next thing to test is the
Level Zero assumption itself: run one decode step with a forced `zeKernelSetArgumentValue` of *every* argument before
*every* append (`ARC_ARG_CACHE=0`) and compare the last-launch arguments recorded by the diagnostic in §5 with what the
kernel should have seen.

## 5. A diagnostic that can name the kernel (and the launch) safely

The current switch cannot (see §2). Replace it with a per-launch event and a launch ring, all inside
`clEnqueueNDRangeKernel`:

1. Give every launch a sequence number and keep the last 64 launches in a ring: kernel name, global/local size, and
   a copy of the *translated* argument values (the same 16 bytes the argument cache keeps — one `memcpy` per argument,
   off by default behind a switch, e.g. `ARC_ZE_ARGLOG=1`).
2. Get an event per launch (`zrt::ev_get()`), pass it as the launch's signal event, and wait on **the event**, not on
   the list, with a bounded timeout:
   `zeEventHostSynchronize(ev, 5 * 1000 * 1000 * 1000ull)` (5 s). A `ZE_RESULT_NOT_READY` timeout is a hang, an error
   code is a fault, and either way the ring names the exact launch.
3. On failure print the ring and `zrt::print_memory()`, then `_Exit(1)` as today. Do not synchronize the whole list
   for this: a wait on one event is local to this thread's launch and does not blame another engine's work.
4. Fix the `!= nullptr` trap of every debug switch in `clrt.h` (`ARC_ZE_SYNC_EACH`, `ARC_LAUNCH_STATS`,
   `ARC_KLIST`, `ARC_ZE_KPROPS`, `ARC_FAKE_OOM` uses `atol` and is fine) so that `=0` means off.

With that in place the recording sequence is reproducible on the command line, without the server:

```
asr_testz --model <Qwen3-ASR-1.7B-Q8_0.gguf> --wav <a 160 s part> --lang id --max-part 240
  (ARC_ZE_ARGLOG=1 ARC_ZE_SYNC_EACH=1 ARC_ARG_CACHE=1)
```

and the last ring entry names the kernel and its arguments. `ARC_ARG_CACHE=0` A/B, `ARC_ASR_Q2=0` (f16 encoder
instead of int8) and `--slots 1` vs 4 split the search space in four runs.

## 6. Order of work

| # | What | Where | Check |
|---|---|---|---|
| 1 | `zrt::sync()` and `clReleaseMemObject` wait for `cl_copy` too; sub-buffers die with their parent | `zero/clrt.h:133,463` | the existing tests (`runz` logits bit-identical: `--dump-logits` before/after) |
| 2 | One device-wide lock (`zrt::gpu_mu()`) around every command-list user, incl. the OOM hook, the prefetch thread, model load/free, and the no-chat-model speech path; a debug single-user assert | `zero/clrt.h`, `tools/server.cpp:299,1312`, `zero/xcache.h:222,583` | a meeting recording + a chat request at the same time; `ARC_TP_SIM` unaffected |
| 3 | Argument ABI made explicit (no `sz == 8` guessing); argument-cache ownership assert | `zero/clrt.h:485` | bit-identical logits on every model |
| 4 | The launch-ring diagnostic of §5, and `!= nullptr` → `atoi` on every debug switch | `zero/clrt.h:485-575` | the crash sequence names one kernel |
| 5 | Only then: the kernel fix the ring points at, and the Qwen3-ASR timeline cut back on (QUEUE 31.13) | `zero/kernels/*.cl` | KL against `tools/ref_forward.cpp` ≤ 0.01 for Qwen3-ASR (its own limit), and the 160-s part in `asr_testz` |

Steps 1-4 do not change any logit; step 5 does and needs the KL check. Each of them that touches speed or accuracy
gets a row in [before-after.md](before-after.md) and a CHANGELOG entry, measured the same way before and after
(AGENTS.md). Steps 1 and 2 are the ones I would run before the next long recording: they are the two ways in this
code that a *freed or busy* device buffer reaches a kernel, and a fault like that is exactly a `DEVICE_LOST` while a
long part decodes.

## 7. What this analysis does not claim

- It does not prove which kernel faulted. The code has no way to say that today (§5) and the log of the five crashes
  was not available. Everything in §4 is reachable by reading the code; which one is *the* one needs step 4.
- It does not rule out `ARC_ARG_CACHE=0` having been a coincidence: with the cache off the host is ~0.2 ms/token
  slower, which shifts every timing in §4.1 and §4.2. The A/B in §5 separates "the cache is wrong" from "the cache
  hides a race".
