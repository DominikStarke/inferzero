// ARC_REPLAY=1 (opt-in, 2026-11-24, the command-list replay plan's (b); UNTESTED HERE: no build or run on the laptop,
// the user's rule of 2026-11-24): a generation step recorded once into mutable command lists (Level Zero
// ZE_experimental_mutable_command_list), and later steps replayed from them with only what changed updated, instead of
// every kernel appended again. Included at the end of zero/clrt.h; the engine marks a step with zrt::RpScope.
//
// How a step maps onto lists: its kernels in order, cut into segments wherever the immediate list gets anything else
// or the host waits (zrt::flush: a copy, a fill; zrt::sync, wait_guarded: the same cuts as ARC_BATCH's). The first
// step of a key records each segment into a mutable list (each kernel's command id kept, with the arguments, group
// count and group size it was appended with) and appends it. A later step of that key runs the engine's code as
// always (every set_args and launch), but a launch is compared with the recorded command at the same place instead of
// appended: the same kernel, and for each argument its value now against the recorded one. At the segment's end the
// differences go to the list in one zeCommandListUpdateMutableCommandsExp (arguments, group counts, group sizes), the
// list is closed and appended again. Nothing about a model is assumed: whatever changes between tokens (the position,
// a ring's slot, a buffer) is found by the comparison. The same kernels with the same arguments in the same order: the
// same results, bit for bit (checked by runz --dump-logits against ARC_REPLAY=0).
//
// Where a step does not match its recording (another kernel, more or fewer launches or segments, a kernel whose
// arguments are not all known: local memory sizes, ARC_ARG_CACHE=0), or any GPU memory was freed since the recording
// (zrt::mem_gen: Level Zero keeps a buffer argument's allocation, so a buffer freed and another made at the same
// address must be set again, the 2026-11-04 crash), or an update fails, the step falls back: the commands it had taken
// so far are appended one by one with their own arguments, the rest of the step runs as without ARC_REPLAY, and the
// next step records again. After 8 steps in a row that fall back it is off for the run, with a line. A kernel released
// (clReleaseKernel: a model unloaded, a kernel rebuilt) drops every recording of its GPU first. A segment is updated
// only after its previous run has finished (its event, reset only once the update has gone through).
// Supported only where the driver has the extension and its devices can update arguments, group counts and sizes
// (ze_mutable_command_list_exp_properties_t); else one line and nothing per step. The counts at exit: steps replayed,
// recorded, fallen back by reason, arguments updated.
// (the Researcher's review of ede9772: the failed update's event, sync()'s cut, the support check, the fallback limit,
//  unload, the OOM retry, threads, the exception path, the reasons)
#pragma once
#include <cstddef>
#include <exception>
#include <map>
#include <thread>

namespace zrt {
// (the EXP form's structures and stype values, as oneapi-src/level-zero's ze_api.h; the EXT form needs Level Zero
//  1.20, the EXP form reaches more drivers)
struct RpFns {
  struct ListDesc { ze_structure_type_t stype; const void* pNext; uint32_t flags; };    // 0x00020016
  struct IdDesc { ze_structure_type_t stype; const void* pNext; uint32_t flags; };      // 0x00020017
  struct CmdsDesc { ze_structure_type_t stype; const void* pNext; uint32_t flags; };    // 0x00020018
  struct ArgDesc {                                                                       // 0x00020019
    ze_structure_type_t stype;
    const void* pNext;
    uint64_t commandId;
    uint32_t argIndex;
    size_t argSize;
    const void* pArgValue;
  };
  struct CountDesc {  // 0x0002001A
    ze_structure_type_t stype;
    const void* pNext;
    uint64_t commandId;
    const ze_group_count_t* pGroupCount;
  };
  struct SizeDesc {  // 0x0002001B
    ze_structure_type_t stype;
    const void* pNext;
    uint64_t commandId;
    uint32_t x, y, z;
  };
  struct DevProps {  // 0x00020015, in zeDeviceGetProperties' pNext chain
    ze_structure_type_t stype;
    void* pNext;
    uint32_t mutableCommandListFlags, mutableCommandFlags;
  };
  struct ExtProps { char name[256]; uint32_t version; };  // (ze_driver_extension_properties_t)
  ze_result_t(__stdcall* next_id)(ze_command_list_handle_t, const IdDesc*, uint64_t*) = nullptr;
  ze_result_t(__stdcall* update)(ze_command_list_handle_t, const CmdsDesc*) = nullptr;
  bool ok = false;
};
inline bool rp_env() {
  static const bool on = getenv("ARC_REPLAY") && atoi(getenv("ARC_REPLAY")) != 0;
  return on;
}
// the driver's calls and extension, looked up once (one line: on, or off and why)
inline RpFns& rp_fns() {
  static RpFns f = [] {
    RpFns r;
    void* zl = plat::dl_loaded("ze_loader.dll", "libze_loader.so.1");
    const char* why = nullptr;
    if (!zl || !driver) why = "no Level Zero loader";
    bool ext_ok = false;
    if (!why) {
      typedef ze_result_t(__stdcall * PFN_exts)(ze_driver_handle_t, uint32_t*, RpFns::ExtProps*);
      auto exts = (PFN_exts)plat::dl_sym(zl, "zeDriverGetExtensionProperties");
      uint32_t n = 0;
      if (exts && exts(driver, &n, nullptr) == ZE_RESULT_SUCCESS && n) {
        std::vector<RpFns::ExtProps> ep(n);
        if (exts(driver, &n, ep.data()) == ZE_RESULT_SUCCESS)
          for (uint32_t i = 0; i < n; ++i)
            ext_ok = ext_ok || strncmp(ep[i].name, "ZE_experimental_mutable_command_list", 256) == 0;
      }
      if (!ext_ok) why = "the driver does not list ZE_experimental_mutable_command_list";
    }
    if (!why) {
      typedef ze_result_t(__stdcall * PFN_ext)(ze_driver_handle_t, const char*, void**);
      auto ext = (PFN_ext)plat::dl_sym(zl, "zeDriverGetExtensionFunctionAddress");
      auto find = [&](const char* name) -> void* {
        void* p = plat::dl_sym(zl, name);
        if (!p && ext) ext(driver, name, &p);
        return p;
      };
      r.next_id = (decltype(r.next_id))find("zeCommandListGetNextCommandIdExp");
      r.update = (decltype(r.update))find("zeCommandListUpdateMutableCommandsExp");
      if (!r.next_id || !r.update) why = "no zeCommandListGetNextCommandIdExp / zeCommandListUpdateMutableCommandsExp";
      else if (!batch_fns(false).ok) why = "the driver cannot append lists to an immediate list";
    }
    r.ok = !why;
    if (r.ok) zlog::eprintf("inferzero: ARC_REPLAY=1: generation steps recorded once in mutable command lists and replayed\n");
    else zlog::eprintf("inferzero: ARC_REPLAY=1 set but not used: %s\n", why);
    return r;
  }();
  return f;
}
// whether this GPU's device can update a mutable list's arguments, group counts and sizes (bits 0-2)
inline bool rp_dev_ok(const Dev& d) {
  RpFns::DevProps mp{(ze_structure_type_t)0x00020015, nullptr, 0, 0};
  ze_device_properties_t p{};
  p.stype = ZE_STYPE_DEVICE_PROPERTIES;
  p.pNext = &mp;
  return zeDeviceGetProperties(d.dev, &p) == ZE_RESULT_SUCCESS && (mp.mutableCommandFlags & 7u) == 7u;
}
// a launch's or step's leaving the replay, by reason (the exit line's counts)
enum RpWhy { RpKernel, RpUntracked, RpMemGen, RpUpdate, RpOther, RpWhyN };
struct RpCmd {
  ZKernel* k = nullptr;
  uint64_t id = 0;
  ze_group_count_t gc{};
  uint32_t ls[3]{};
  std::vector<ZKernel::Arg> args;  // (as appended, then as last updated: the update descriptors point into it)
};
struct RpSeg {
  ze_command_list_handle_t l = nullptr;
  ze_event_handle_t done = nullptr;
  bool used = false;  // (appended at least once and its event not reset since: wait for done before the next update)
  std::vector<RpCmd> cmds;
};
struct RpStep {
  std::vector<RpSeg> segs;
  uint32_t gen = 0;     // (zrt::mem_gen when recorded)
  bool complete = false;
};
struct Replay {
  std::map<int, RpStep> steps;
  // (the step being run (-1 none) and the thread that began it: another thread's launches, flushes and waits on this
  //  GPU (the server's speech models) go as without ARC_REPLAY, never into the step's lists; atomics, as they read them)
  std::atomic<int> key{-1};
  std::atomic<std::thread::id> owner{};
  // (a kernel was released while this GPU's recordings could not be dropped (another thread's step, or a thread of
  //  another GPU): the owner drops them all at its next rp_begin or rp_end, and its step breaks at its next launch)
  std::atomic<bool> stale{false};
  bool rec = false, broken = false, off = false, dev_ok = false, dev_known = false;
  // (steps since the last kept replay: a recording counts until a replay matches it, so record / fail / record / fail
  //  reaches the limit; the Researcher's review of 9184768, 1)
  int falls_in_row = 0;
  size_t seg = 0, cmd = 0;
  std::vector<RpFns::ArgDesc> argd;
  std::vector<RpFns::CountDesc> cntd;
  std::vector<RpFns::SizeDesc> sized;
  ze_event_pool_handle_t pool = nullptr;
  std::vector<ze_event_handle_t> ev_free;
  uint32_t ord = 0;
  // (this thread is the step's: no other thread touches the rest of this struct while it is)
  bool mine() const { return key.load(std::memory_order_acquire) >= 0 && owner.load(std::memory_order_acquire) == std::this_thread::get_id(); }
};
inline std::atomic<uint64_t> rp_n_rep{0}, rp_n_rec{0}, rp_n_fall{0}, rp_n_upd{0}, rp_n_why[RpWhyN]{};
inline void rp_stats_at_exit() {
  static std::once_flag f;
  std::call_once(f, [] {
    std::atexit([] {
      zlog::eprintf("inferzero: ARC_REPLAY: %llu steps replayed, %llu recorded, %llu fell back (another kernel or count "
                    "%llu, an untracked argument %llu, memory freed %llu, an update refused %llu, other %llu); %llu "
                    "arguments updated\n",
                    (unsigned long long)rp_n_rep.load(), (unsigned long long)rp_n_rec.load(),
                    (unsigned long long)rp_n_fall.load(), (unsigned long long)rp_n_why[RpKernel].load(),
                    (unsigned long long)rp_n_why[RpUntracked].load(), (unsigned long long)rp_n_why[RpMemGen].load(),
                    (unsigned long long)rp_n_why[RpUpdate].load(), (unsigned long long)rp_n_why[RpOther].load(),
                    (unsigned long long)rp_n_upd.load());
    });
  });
}
// whether a kernel's every argument value is known (the cache keeps it): one Level Zero call a kernel for its count
inline bool rp_trackable(ZKernel* k) {
  if (k->rp_untracked) return false;
  if (k->rp_nargs < 0) {
    ze_kernel_properties_t kp{};
    kp.stype = (ze_structure_type_t)0x1e;  // (ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, as set_last_arg)
    k->rp_nargs = zeKernelGetProperties(k->k, &kp) == ZE_RESULT_SUCCESS ? (int)kp.numKernelArgs : 0;
  }
  if (k->rp_nargs <= 0 || (int)k->args.size() < k->rp_nargs) return false;
  for (int i = 0; i < k->rp_nargs; ++i)
    if (!k->args[(size_t)i].set) return false;
  return true;
}
// d's immediate list gets a recorded list or a kernel, with flush()'s retry after the out-of-memory hook
inline void rp_append_list(Dev& d, ze_command_list_handle_t l, ze_event_handle_t done) {
  const BatchAppend app = batch_append_for(d);
  ze_result_t e = app(d.cl, 1, &l, done, 0, nullptr);
  for (int i = 0; e == (ze_result_t)0x70000003 && i < 8 && &d == &D() && oom_hook && oom_hook(); ++i)
    e = app(d.cl, 1, &l, done, 0, nullptr);
  ZE_CHECK(e);
}
inline void rp_append_kernel(Dev& d, ZKernel* k, const ze_group_count_t* gc) {
  ze_result_t e = zeCommandListAppendLaunchKernel(d.cl, k->k, gc, nullptr, 0, nullptr);
  for (int i = 0; e == (ze_result_t)0x70000003 && i < 8 && &d == &D() && oom_hook && oom_hook(); ++i)
    e = zeCommandListAppendLaunchKernel(d.cl, k->k, gc, nullptr, 0, nullptr);
  ZE_CHECK(e);
}
// a step's lists destroyed, their events back to the pool (each list's last run waited for first)
inline void rp_drop(Dev& d, RpStep& s) {
  Replay& r = *d.rp.load();
  for (RpSeg& g : s.segs) {
    if (g.used && g.done) wait_plain(d, [&g](uint64_t t) { return zeEventHostSynchronize(g.done, t); }, "a replayed list");
    if (g.l) zeCommandListDestroy(g.l);
    g.l = nullptr;
    if (g.done) {
      zeEventHostReset(g.done);
      r.ev_free.push_back(g.done);
    }
    g.done = nullptr;
    g.used = false;
  }
  s.segs.clear();
  s.complete = false;
}
inline void rp_break(Dev& d, RpWhy why);
// every recording of d dropped (a kernel about to be released: clReleaseKernel); not during a step of another thread
inline void rp_drop_all(Dev& d) {
  Replay* r = d.rp.load();
  if (!r) return;
  if (r->mine()) {  // (a step of this thread under way: what it took goes to the GPU first, the rest as without it)
    if (!r->broken) rp_break(d, RpOther);
    for (auto& [key, s] : r->steps) rp_drop(d, s);
    r->steps.clear();
    return;
  }
  // (d not this thread's GPU, or a step claimed or under way on another thread: marked, the owner drops them; else
  //  claimed here (-1 -> -2) for the drop, so no step begins meanwhile)
  int free_key = -1;
  if (&d != &D() || !r->key.compare_exchange_strong(free_key, -2, std::memory_order_acq_rel)) {
    r->stale.store(true, std::memory_order_release);
    return;
  }
  for (auto& [key, s] : r->steps) rp_drop(d, s);
  r->steps.clear();
  r->key.store(-1, std::memory_order_release);
}
// a new segment for the recording: a mutable in-order list and an event; false when there is none to have
inline bool rp_new_seg(Dev& d, RpStep& s) {
  Replay& r = *d.rp.load();
  if (!r.pool) {
    const uint32_t kEv = 512;
    ze_event_pool_desc_t pd{ZE_STYPE_EVENT_POOL_DESC, nullptr, 0x1 /* host visible */, kEv};
    if (zeEventPoolCreate(ctx_of(d), &pd, 1, &d.dev, &r.pool) != ZE_RESULT_SUCCESS) return false;
    for (uint32_t i = 0; i < kEv; ++i) {
      ze_event_desc_t ed{ZE_STYPE_EVENT_DESC, nullptr, i, 0x4 /* signal: host scope */, 0};
      ze_event_handle_t e = nullptr;
      if (zeEventCreate(r.pool, &ed, &e) == ZE_RESULT_SUCCESS) r.ev_free.push_back(e);
    }
    uint32_t ng = 0;
    zeDeviceGetCommandQueueGroupProperties(d.dev, &ng, nullptr);
    std::vector<ze_command_queue_group_properties_t> gp(ng);
    for (auto& g : gp) g.stype = ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(d.dev, &ng, gp.data());
    for (uint32_t i = 0; i < ng; ++i)
      if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { r.ord = i; break; }  // (the immediate list's group)
  }
  if (r.ev_free.empty()) return false;
  const RpFns::ListDesc md{(ze_structure_type_t)0x00020016, nullptr, 0};
  BatchFns::ListDesc ld{0xfu /* ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC */, &md, r.ord, 8 /* in order */};
  RpSeg g;
  if (batch_fns(false).create(ctx_of(d), d.dev, &ld, &g.l) != ZE_RESULT_SUCCESS) return false;
  g.done = r.ev_free.back();
  r.ev_free.pop_back();
  s.segs.push_back(std::move(g));
  return true;
}
// the step leaves the replay: what it had taken of the current segment goes to the immediate list now (recording:
// the list as recorded so far; replaying: each command with its own arguments, then every kernel's current arguments
// and group size set back), the rest of the step as without ARC_REPLAY, the recording dropped at the step's end.
// (broken first, and the taken count cleared: an out-of-memory hook that waits comes back through flush, which then
//  finds nothing of the step's to append)
inline void rp_break(Dev& d, RpWhy why) {
  Replay& r = *d.rp.load();
  if (r.broken) return;
  r.broken = true;
  ++rp_n_why[why];
  const size_t taken = r.cmd;
  r.cmd = 0;
  r.argd.clear();
  r.cntd.clear();
  r.sized.clear();
  auto it = r.steps.find(r.key.load());
  if (it == r.steps.end() || r.seg >= it->second.segs.size() || taken == 0) return;
  RpSeg& g = it->second.segs[r.seg];
  if (r.rec) {
    ZE_CHECK(batch_fns(false).close(g.l));
    rp_append_list(d, g.l, g.done);
    g.used = true;
    return;
  }
  for (size_t j = 0; j < taken && j < g.cmds.size(); ++j) {
    RpCmd& c = g.cmds[j];
    for (size_t i = 0; i < c.args.size(); ++i)
      if (c.args[i].set) ZE_CHECK(zeKernelSetArgumentValue(c.k->k, (uint32_t)i, c.args[i].sz, c.args[i].v));
    ZE_CHECK(zeKernelSetGroupSize(c.k->k, c.ls[0], c.ls[1], c.ls[2]));
    rp_append_kernel(d, c.k, &c.gc);
  }
  for (size_t j = 0; j < taken && j < g.cmds.size(); ++j) {  // (the kernels' own state back, as the cache has it)
    ZKernel* k = g.cmds[j].k;
    for (size_t i = 0; i < k->args.size(); ++i)
      if (k->args[i].set) ZE_CHECK(zeKernelSetArgumentValue(k->k, (uint32_t)i, k->args[i].sz, k->args[i].v));
    ZE_CHECK(zeKernelSetGroupSize(k->k, k->gx, k->gy, k->gz));
  }
}
// a launch during a step: taken (recorded, or compared for the replay) or not (false: launched as without ARC_REPLAY)
inline bool rp_launch(Dev& d, ZKernel* k, const ze_group_count_t& gc, const uint32_t* ls) {
  Replay* rp = d.rp.load(std::memory_order_acquire);
  if (!rp || !rp->mine()) return false;
  Replay& r = *rp;
  if (r.broken) return false;
  if (r.stale.load(std::memory_order_acquire)) { rp_break(d, RpOther); return false; }  // (a released kernel)
  RpStep& s = r.steps[r.key.load()];
  if (!rp_trackable(k)) { rp_break(d, RpUntracked); return false; }
  if (r.rec) {
    if (r.seg == s.segs.size() && !rp_new_seg(d, s)) { rp_break(d, RpOther); return false; }
    RpSeg& g = s.segs[r.seg];
    RpCmd c;
    c.k = k;
    c.gc = gc;
    for (int i = 0; i < 3; ++i) c.ls[i] = ls[i];
    c.args.assign(k->args.begin(), k->args.begin() + k->rp_nargs);
    const RpFns::IdDesc idd{(ze_structure_type_t)0x00020017, nullptr, 1u | 2u | 4u /* arguments, group count, size */};
    if (rp_fns().next_id(g.l, &idd, &c.id) != ZE_RESULT_SUCCESS ||
        zeCommandListAppendLaunchKernel(g.l, k->k, &gc, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS) {
      rp_break(d, RpOther);  // (the commands recorded before it go as recorded)
      return false;
    }
    g.cmds.push_back(std::move(c));
    ++r.cmd;
    return true;
  }
  if (mem_gen.load(std::memory_order_relaxed) != s.gen) { rp_break(d, RpMemGen); return false; }
  if (r.seg >= s.segs.size() || r.cmd >= s.segs[r.seg].cmds.size()) { rp_break(d, RpKernel); return false; }
  RpCmd& c = s.segs[r.seg].cmds[r.cmd];
  if (c.k != k || (int)c.args.size() != k->rp_nargs) { rp_break(d, RpKernel); return false; }
  for (size_t i = 0; i < c.args.size(); ++i)
    if (c.args[i].sz != k->args[i].sz) { rp_break(d, RpKernel); return false; }
  for (size_t i = 0; i < c.args.size(); ++i) {
    const ZKernel::Arg& now = k->args[i];
    if (memcmp(c.args[i].v, now.v, now.sz) != 0) {
      c.args[i] = now;
      r.argd.push_back(RpFns::ArgDesc{(ze_structure_type_t)0x00020019, nullptr, c.id, (uint32_t)i, now.sz, c.args[i].v});
    }
  }
  if (c.gc.groupCountX != gc.groupCountX || c.gc.groupCountY != gc.groupCountY || c.gc.groupCountZ != gc.groupCountZ) {
    c.gc = gc;
    r.cntd.push_back(RpFns::CountDesc{(ze_structure_type_t)0x0002001A, nullptr, c.id, &c.gc});
  }
  if (c.ls[0] != ls[0] || c.ls[1] != ls[1] || c.ls[2] != ls[2]) {
    for (int i = 0; i < 3; ++i) c.ls[i] = ls[i];
    r.sized.push_back(RpFns::SizeDesc{(ze_structure_type_t)0x0002001B, nullptr, c.id, ls[0], ls[1], ls[2]});
  }
  ++r.cmd;
  return true;
}
// a segment's end (zrt::flush, sync, wait_guarded: anything else is about to go to the immediate list, the host is
// about to wait, or the step ends). The step's state moves on before the append, so an out-of-memory hook that waits
// and comes back here finds nothing to append twice
inline void rp_flush(Dev& d) {
  Replay* rp = d.rp.load(std::memory_order_acquire);
  if (!rp || !rp->mine()) return;
  Replay& r = *rp;
  if (r.broken || r.cmd == 0) return;
  RpStep& s = r.steps[r.key.load()];
  RpSeg& g = s.segs[r.seg];
  if (r.rec) {
    ZE_CHECK(batch_fns(false).close(g.l));
  } else {
    if (r.cmd != g.cmds.size()) { rp_break(d, RpKernel); return; }  // (fewer launches than recorded: those taken one by one)
    // (its previous run first; the event reset only after the update went through, so a refused update leaves it
    //  signalled and rp_drop's wait returns: the Researcher's review of ede9772, 1)
    if (g.used) wait_plain(d, [&g](uint64_t t) { return zeEventHostSynchronize(g.done, t); }, "a replayed list");
    const size_t n = r.argd.size() + r.cntd.size() + r.sized.size();
    if (n) {  // (one chain of every kind of change, built now: the vectors no longer move)
      std::vector<const void*> chain;
      chain.reserve(n);
      for (auto& a : r.argd) chain.push_back(&a);
      for (auto& a : r.cntd) chain.push_back(&a);
      for (auto& a : r.sized) chain.push_back(&a);
      for (size_t i = 0; i + 1 < n; ++i) *(const void**)((char*)chain[i] + offsetof(RpFns::ArgDesc, pNext)) = chain[i + 1];
      const RpFns::CmdsDesc cd{(ze_structure_type_t)0x00020018, chain[0], 0};
      if (rp_fns().update(g.l, &cd) != ZE_RESULT_SUCCESS || batch_fns(false).close(g.l) != ZE_RESULT_SUCCESS) {
        rp_break(d, RpUpdate);  // (every command of the segment one by one; the recording dropped at the step's end)
        return;
      }
      rp_n_upd += r.argd.size();
    }
    if (g.used) ZE_CHECK(zeEventHostReset(g.done));
  }
  r.argd.clear();
  r.cntd.clear();
  r.sized.clear();
  ++r.seg;
  r.cmd = 0;
  g.used = true;
  rp_append_list(d, g.l, g.done);
}
// the step of `key` starts on this thread's GPU; false (no step) without ARC_REPLAY, its support, or while another
// thread holds a step on this GPU
inline bool rp_begin(int key) {
  if (!rp_env() || !rp_fns().ok) return false;
  Dev& d = D();
  if (!batch_append_for(d)) {
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
      zlog::eprintf("inferzero: ARC_REPLAY=1 set but not used on GPU %d: its driver cannot append lists to an immediate "
                    "list\n", d.index);
    return false;
  }
  Replay* rp = d.rp.load(std::memory_order_acquire);
  if (!rp) {  // (two threads at once: one Replay wins, the other's goes)
    Replay* mine_new = new Replay;
    if (d.rp.compare_exchange_strong(rp, mine_new, std::memory_order_acq_rel)) rp = mine_new;
    else delete mine_new;
  }
  Replay& r = *rp;
  // (the step claimed first: key -1 -> -2 (claimed, not yet begun), so of two threads only one goes on and nothing
  //  else of the struct is touched by the other; -2 is no step for mine(), and is one for rp_drop_all. The
  //  Researcher's review of 9184768, 2)
  int free_key = -1;
  if (!r.key.compare_exchange_strong(free_key, -2, std::memory_order_acq_rel)) {
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
      zlog::eprintf("inferzero: ARC_REPLAY: a step while another is under way on GPU %d (two engines on one GPU): run "
                    "without a replay\n", d.index);
    return false;
  }
  struct Unclaim {  // (every way out below without a step gives the claim back)
    std::atomic<int>& k;
    bool keep = false;
    ~Unclaim() { if (!keep) k.store(-1, std::memory_order_release); }
  } unclaim{r.key};
  if (r.off) return false;
  if (!r.dev_known) {
    r.dev_known = true;
    r.dev_ok = rp_dev_ok(d);
    if (!r.dev_ok)
      zlog::eprintf("inferzero: ARC_REPLAY=1 set but not used on GPU %d: its device cannot update a mutable list's "
                    "arguments, group counts and sizes\n", d.index);
  }
  if (!r.dev_ok) return false;
  rp_stats_at_exit();
  flush(d);  // (ARC_BATCH's list first: the step's own lists come after it)
  // (a kernel released by another thread while it could not drop this GPU's recordings: every one goes now, Researcher's
  //  review of 9184768, 3)
  if (r.stale.exchange(false, std::memory_order_acq_rel)) {
    for (auto& [k2, s2] : r.steps) rp_drop(d, s2);
    r.steps.clear();
  }
  auto it = r.steps.find(key);
  if (it != r.steps.end() && (!it->second.complete || it->second.gen != mem_gen.load(std::memory_order_relaxed))) {
    if (it->second.complete) ++rp_n_why[RpMemGen];
    rp_drop(d, it->second);
    r.steps.erase(it);
    it = r.steps.end();
  }
  r.rec = it == r.steps.end();
  if (r.rec) r.steps[key].gen = mem_gen.load(std::memory_order_relaxed);
  r.seg = r.cmd = 0;
  r.broken = false;
  r.argd.clear();
  r.cntd.clear();
  r.sized.clear();
  r.owner.store(std::this_thread::get_id(), std::memory_order_release);
  unclaim.keep = true;
  r.key.store(key, std::memory_order_release);
  return true;
}
// the step ends: its last segment appended; a recording kept only when whole and no memory was freed meanwhile, a
// replay that took fewer segments than recorded or fell back records again next time; 8 such steps in a row: off
inline void rp_end() {
  Dev& d = D();
  Replay* rp = d.rp.load(std::memory_order_acquire);
  if (!rp || !rp->mine()) return;
  Replay& r = *rp;
  flush(d);
  const int key = r.key.load();
  // (-2 before anything that could throw: no step, so no launch is taken (mine() wants >= 0), but still claimed, so no
  //  other thread begins one while this drops lists; -1 on every way out)
  r.key.store(-2, std::memory_order_release);
  struct Release { std::atomic<int>& k; ~Release() { k.store(-1, std::memory_order_release); } } release{r.key};
  auto it = r.steps.find(key);
  if (it == r.steps.end()) return;
  RpStep& s = it->second;
  bool keep = !r.broken;
  if (r.rec) {
    keep = keep && s.gen == mem_gen.load(std::memory_order_relaxed);
    if (keep) ++rp_n_rec;
  } else {
    keep = keep && r.seg == s.segs.size();
    if (keep) ++rp_n_rep;
  }
  if (!keep) {
    ++rp_n_fall;
    if (!r.broken) ++rp_n_why[r.rec ? RpMemGen : RpKernel];  // (a recording with a free in it; fewer segments)
  }
  s.complete = keep;
  r.falls_in_row = !r.rec && keep ? 0 : r.falls_in_row + 1;
  if (!keep) {
    rp_drop(d, s);
    r.steps.erase(it);
  }
  if (r.stale.exchange(false, std::memory_order_acq_rel)) {  // (a kernel released meanwhile: every recording goes)
    for (auto& [k2, s2] : r.steps) rp_drop(d, s2);
    r.steps.clear();
  }
  if (r.falls_in_row >= 8) {
    r.off = true;
    for (auto& [k2, s2] : r.steps) rp_drop(d, s2);
    r.steps.clear();
    zlog::eprintf("inferzero: ARC_REPLAY: off for this run on GPU %d: 8 steps in a row did not match their recording "
                  "(the exit line says why)\n", d.index);
  }
}
// a step's scope (zero/engine.h forward): begun when on, ended on every way out (an exception too: what it took goes
// to the GPU and the recording is dropped)
struct RpScope {
  bool on;
  explicit RpScope(bool want, int key = 1) : on(want && rp_begin(key)) {}
  ~RpScope() {
    if (!on) return;
    try {
      Replay* rp = D().rp.load();
      if (std::uncaught_exceptions() && rp && rp->mine()) rp_break(D(), RpOther);
      rp_end();
    } catch (...) {
    }
  }
};
}  // namespace zrt
