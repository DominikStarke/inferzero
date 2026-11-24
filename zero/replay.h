// ARC_REPLAY=1 (opt-in, 2026-11-24, the command-list replay plan's (b); UNTESTED HERE: no build or run on the laptop,
// the user's rule of 2026-11-24): a generation step recorded once into mutable command lists (Level Zero
// ZE_experimental_mutable_command_list), and later steps replayed from them with only what changed updated, instead of
// every kernel appended again. Included at the end of zero/clrt.h; the engine marks a step with zrt::RpScope.
//
// How a step maps onto lists: its kernels in order, cut into segments wherever the immediate list gets anything else
// (a copy, a fill, a host wait: zrt::flush, the same cuts as ARC_BATCH's). The first step of a key records each segment
// into a mutable list (each kernel's command id kept, with the arguments, group count and group size it was appended
// with) and appends it. A later step of that key runs the engine's code as always (every set_args and launch), but a
// launch is compared with the recorded command at the same place instead of appended: the same kernel, and for each
// argument its value now against the recorded one. At the segment's end the differences go to the list in one
// zeCommandListUpdateMutableCommandsExp (arguments, group counts, group sizes), the list is closed and appended again.
// Nothing about a model is assumed: whatever changes between tokens (the position, a ring's slot, a buffer) is found
// by the comparison. The same kernels with the same arguments in the same order: the same results, bit for bit
// (checked by runz --dump-logits against ARC_REPLAY=0).
//
// Where a step does not match its recording (another kernel, more or fewer launches or segments, a kernel whose
// arguments are not all known: local memory sizes, ARC_ARG_CACHE=0), or any GPU memory was freed since the recording
// (zrt::mem_gen: Level Zero keeps a buffer argument's allocation, so a buffer freed and another made at the same
// address must be set again, the 2026-11-04 crash), the step falls back: the commands it had taken so far are
// appended one by one with their own arguments, the rest of the step runs as without ARC_REPLAY, and the next step
// records again. A segment is updated only after its previous run has finished (its event).
// The counts at exit: steps replayed, recorded, fallen back, arguments updated.
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
  ze_result_t(__stdcall* next_id)(ze_command_list_handle_t, const IdDesc*, uint64_t*) = nullptr;
  ze_result_t(__stdcall* update)(ze_command_list_handle_t, const CmdsDesc*) = nullptr;
  bool ok = false;
};
inline bool rp_env() {
  static const bool on = getenv("ARC_REPLAY") && atoi(getenv("ARC_REPLAY")) != 0;
  return on;
}
inline RpFns& rp_fns() {
  static RpFns f = [] {
    RpFns r;
    void* zl = plat::dl_loaded("ze_loader.dll", "libze_loader.so.1");
    if (!zl || !driver) return r;
    typedef ze_result_t(__stdcall * PFN_ext)(ze_driver_handle_t, const char*, void**);
    auto ext = (PFN_ext)plat::dl_sym(zl, "zeDriverGetExtensionFunctionAddress");
    auto find = [&](const char* name) -> void* {
      void* p = plat::dl_sym(zl, name);
      if (!p && ext) ext(driver, name, &p);
      return p;
    };
    r.next_id = (decltype(r.next_id))find("zeCommandListGetNextCommandIdExp");
    r.update = (decltype(r.update))find("zeCommandListUpdateMutableCommandsExp");
    r.ok = r.next_id && r.update && batch_fns().ok;
    zlog::eprintf(r.ok ? "inferzero: ARC_REPLAY=1: generation steps recorded once in mutable command lists and replayed\n"
                       : "inferzero: ARC_REPLAY=1: this driver has no mutable command lists (or cannot append lists to "
                         "an immediate list): off\n");
    return r;
  }();
  return f;
}
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
  bool used = false;  // (appended at least once: wait for done before the next update)
  std::vector<RpCmd> cmds;
};
struct RpStep {
  std::vector<RpSeg> segs;
  uint32_t gen = 0;     // (zrt::mem_gen when recorded)
  bool complete = false;
};
struct Replay {
  std::map<int, RpStep> steps;
  int key = -1;  // (the step being run, -1 none)
  std::thread::id owner;  // (the thread that began it: another thread's launches and flushes on this GPU (the server's
                          //  speech models) go as without ARC_REPLAY, never into the step's lists)
  bool rec = false, broken = false;
  size_t seg = 0, cmd = 0;
  std::vector<RpFns::ArgDesc> argd;
  std::vector<RpFns::CountDesc> cntd;
  std::vector<RpFns::SizeDesc> sized;
  ze_event_pool_handle_t pool = nullptr;
  std::vector<ze_event_handle_t> ev_free;
  uint32_t ord = 0;
};
inline std::atomic<uint64_t> rp_n_rep{0}, rp_n_rec{0}, rp_n_fall{0}, rp_n_upd{0};
inline void rp_stats_at_exit() {
  static std::once_flag f;
  std::call_once(f, [] {
    std::atexit([] {
      zlog::eprintf("inferzero: ARC_REPLAY: %llu steps replayed, %llu recorded, %llu fell back; %llu arguments updated\n",
                    (unsigned long long)rp_n_rep.load(), (unsigned long long)rp_n_rec.load(),
                    (unsigned long long)rp_n_fall.load(), (unsigned long long)rp_n_upd.load());
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
// a step's lists destroyed, their events back to the pool (each list's last run waited for first)
inline void rp_drop(Dev& d, RpStep& s) {
  Replay& r = *d.rp;
  for (RpSeg& g : s.segs) {
    if (g.used) wait_plain(d, [&g](uint64_t t) { return zeEventHostSynchronize(g.done, t); }, "a replayed list");
    if (g.l) zeCommandListDestroy(g.l);
    if (g.done) {
      zeEventHostReset(g.done);
      r.ev_free.push_back(g.done);
    }
  }
  s.segs.clear();
  s.complete = false;
}
// a new segment for the recording: a mutable in-order list and an event; false when there is none to have
inline bool rp_new_seg(Dev& d, RpStep& s) {
  Replay& r = *d.rp;
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
  if (batch_fns().create(ctx_of(d), d.dev, &ld, &g.l) != ZE_RESULT_SUCCESS) return false;
  g.done = r.ev_free.back();
  r.ev_free.pop_back();
  s.segs.push_back(std::move(g));
  return true;
}
// the step leaves the replay: what it had taken of the current segment goes to the immediate list now (recording:
// the list as recorded so far; replaying: each command with its own arguments, then every kernel's current arguments
// and group size set back), the rest of the step as without ARC_REPLAY, the recording dropped at the step's end
inline void rp_break(Dev& d) {
  Replay& r = *d.rp;
  RpStep& s = r.steps[r.key];
  if (r.seg < s.segs.size() && r.cmd > 0) {
    RpSeg& g = s.segs[r.seg];
    if (r.rec) {
      ZE_CHECK(batch_fns().close(g.l));
      ZE_CHECK(batch_append_for(d)(d.cl, 1, &g.l, g.done, 0, nullptr));
      g.used = true;
    } else {
      for (size_t j = 0; j < r.cmd && j < g.cmds.size(); ++j) {
        RpCmd& c = g.cmds[j];
        for (size_t i = 0; i < c.args.size(); ++i)
          if (c.args[i].set) ZE_CHECK(zeKernelSetArgumentValue(c.k->k, (uint32_t)i, c.args[i].sz, c.args[i].v));
        ZE_CHECK(zeKernelSetGroupSize(c.k->k, c.ls[0], c.ls[1], c.ls[2]));
        ZE_CHECK(zeCommandListAppendLaunchKernel(d.cl, c.k->k, &c.gc, nullptr, 0, nullptr));
      }
      for (size_t j = 0; j < r.cmd && j < g.cmds.size(); ++j) {  // (the kernels' own state back, as the cache has it)
        ZKernel* k = g.cmds[j].k;
        for (size_t i = 0; i < k->args.size(); ++i)
          if (k->args[i].set) ZE_CHECK(zeKernelSetArgumentValue(k->k, (uint32_t)i, k->args[i].sz, k->args[i].v));
        ZE_CHECK(zeKernelSetGroupSize(k->k, k->gx, k->gy, k->gz));
      }
    }
  }
  r.argd.clear();
  r.cntd.clear();
  r.sized.clear();
  r.broken = true;
}
// a launch during a step: taken (recorded, or compared for the replay) or not (false: launched as without ARC_REPLAY)
inline bool rp_launch(Dev& d, ZKernel* k, const ze_group_count_t& gc, const uint32_t* ls) {
  Replay& r = *d.rp;
  if (r.key < 0 || r.broken || r.owner != std::this_thread::get_id()) return false;
  RpStep& s = r.steps[r.key];
  if (!rp_trackable(k)) { rp_break(d); return false; }
  if (r.rec) {
    if (r.seg == s.segs.size() && !rp_new_seg(d, s)) { rp_break(d); return false; }
    RpSeg& g = s.segs[r.seg];
    RpCmd c;
    c.k = k;
    c.gc = gc;
    for (int i = 0; i < 3; ++i) c.ls[i] = ls[i];
    c.args.assign(k->args.begin(), k->args.begin() + k->rp_nargs);
    const RpFns::IdDesc idd{(ze_structure_type_t)0x00020017, nullptr, 1u | 2u | 4u /* arguments, group count, size */};
    if (rp_fns().next_id(g.l, &idd, &c.id) != ZE_RESULT_SUCCESS ||
        zeCommandListAppendLaunchKernel(g.l, k->k, &gc, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS) {
      rp_break(d);  // (the commands recorded before it go as recorded)
      return false;
    }
    g.cmds.push_back(std::move(c));
    ++r.cmd;
    return true;
  }
  if (r.seg >= s.segs.size() || r.cmd >= s.segs[r.seg].cmds.size() || mem_gen.load(std::memory_order_relaxed) != s.gen) {
    rp_break(d);
    return false;
  }
  RpCmd& c = s.segs[r.seg].cmds[r.cmd];
  if (c.k != k || (int)c.args.size() != k->rp_nargs) { rp_break(d); return false; }
  for (size_t i = 0; i < c.args.size(); ++i) {
    const ZKernel::Arg& now = k->args[i];
    if (c.args[i].sz != now.sz) { rp_break(d); return false; }
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
// a segment's end (zrt::flush: anything else is about to go to the immediate list, or the step ends)
inline void rp_flush(Dev& d) {
  if (!d.rp) return;
  Replay& r = *d.rp;
  if (r.key < 0 || r.broken || r.cmd == 0 || r.owner != std::this_thread::get_id()) return;
  RpStep& s = r.steps[r.key];
  RpSeg& g = s.segs[r.seg];
  if (r.rec) {
    ZE_CHECK(batch_fns().close(g.l));
  } else {
    if (r.cmd != g.cmds.size()) { rp_break(d); return; }  // (fewer launches than recorded: those taken go one by one)
    if (g.used) {
      wait_plain(d, [&g](uint64_t t) { return zeEventHostSynchronize(g.done, t); }, "a replayed list");
      ZE_CHECK(zeEventHostReset(g.done));
    }
    const size_t n = r.argd.size() + r.cntd.size() + r.sized.size();
    if (n) {  // (one chain of every kind of change, built now: the vectors no longer move)
      std::vector<const void*> chain;
      chain.reserve(n);
      for (auto& a : r.argd) chain.push_back(&a);
      for (auto& a : r.cntd) chain.push_back(&a);
      for (auto& a : r.sized) chain.push_back(&a);
      for (size_t i = 0; i + 1 < n; ++i) *(const void**)((char*)chain[i] + offsetof(RpFns::ArgDesc, pNext)) = chain[i + 1];
      const RpFns::CmdsDesc cd{(ze_structure_type_t)0x00020018, chain[0], 0};
      if (rp_fns().update(g.l, &cd) != ZE_RESULT_SUCCESS || batch_fns().close(g.l) != ZE_RESULT_SUCCESS) {
        rp_break(d);  // (every command of the segment one by one; the recording dropped at the step's end)
        return;
      }
      rp_n_upd += r.argd.size();
    }
  }
  ZE_CHECK(batch_append_for(d)(d.cl, 1, &g.l, g.done, 0, nullptr));
  g.used = true;
  r.argd.clear();
  r.cntd.clear();
  r.sized.clear();
  ++r.seg;
  r.cmd = 0;
}
// the step of `key` starts on this thread's GPU (no-op without ARC_REPLAY or its calls)
inline void rp_begin(int key) {
  if (!rp_env() || !rp_fns().ok) return;
  Dev& d = D();
  if (!batch_append_for(d)) return;
  if (!d.rp) d.rp = new Replay;
  rp_stats_at_exit();
  flush(d);  // (ARC_BATCH's list first: the step's own lists come after it)
  Replay& r = *d.rp;
  auto it = r.steps.find(key);
  if (it != r.steps.end() && (!it->second.complete || it->second.gen != mem_gen.load(std::memory_order_relaxed))) {
    rp_drop(d, it->second);
    r.steps.erase(it);
    it = r.steps.end();
  }
  r.rec = it == r.steps.end();
  if (r.rec) r.steps[key].gen = mem_gen.load(std::memory_order_relaxed);
  r.key = key;
  r.owner = std::this_thread::get_id();
  r.seg = r.cmd = 0;
  r.broken = false;
}
// the step ends: its last segment appended; a recording kept only when whole and no memory was freed meanwhile, a
// replay that took fewer segments than recorded or fell back records again next time
inline void rp_end() {
  Dev& d = D();
  if (!d.rp || d.rp->key < 0) return;
  Replay& r = *d.rp;
  flush(d);
  RpStep& s = r.steps[r.key];
  bool keep = !r.broken;
  if (r.rec) {
    keep = keep && s.gen == mem_gen.load(std::memory_order_relaxed);
    if (keep) ++rp_n_rec;
  } else {
    keep = keep && r.seg == s.segs.size();
    if (keep) ++rp_n_rep;
  }
  if (r.broken) ++rp_n_fall;
  s.complete = keep;
  if (!keep) {
    rp_drop(d, s);
    r.steps.erase(r.key);
  }
  r.key = -1;
}
// a step's scope (zero/engine.h forward): begun when on, ended on every way out (an exception too: what it took goes
// to the GPU and the recording is dropped)
struct RpScope {
  bool on;
  explicit RpScope(bool want, int key = 1) : on(want) {
    if (on) rp_begin(key);
  }
  ~RpScope() {
    if (!on) return;
    try {
      if (std::uncaught_exceptions() && D().rp && D().rp->key >= 0 && !D().rp->broken) rp_break(D());
      rp_end();
    } catch (...) {
    }
  }
};
}  // namespace zrt
