// What recording a generation step and replaying it could save (the idea behind vLLM's CUDA graphs): the same
// sequence of small dependent kernels, as the engine launches them now (each appended to the in-order immediate
// command list) against a regular command list recorded once and executed again (in order: a barrier between
// kernels, or the in-order list flag). The difference is the GPU's idle time between kernels that a replay removes.
//   bin\replay_probe.exe [--kernels 700] [--floats 65536] [--reps 30] [--updates 20]
// (2026-11-24) Also a mutable command list: recorded once, the changing arguments updated each step and replayed, its
// results checked bit for bit against the appended sequence: does the driver support it, and what does it save.
// 700 kernels ~ one token of the 35B-A3B, 930 of the 27B (ARC_LAUNCH_STATS=1 with runz counts them).
#include "../zero/clrt.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

static const char* kSrc = R"CLC(
__kernel void step(__global float* y, __global const float* x, int n) {
  const int i = get_global_id(0);
  if (i < n) y[i] = y[i] * 0.999f + x[i];
}
)CLC";

struct ze_command_list_desc_t {
  ze_structure_type_t stype;
  const void* pNext;
  uint32_t commandQueueGroupOrdinal, flags;  // flags 8: in order (ZE_COMMAND_LIST_FLAG_IN_ORDER)
};
typedef struct _ze_command_queue_handle_t* ze_command_queue_handle_t;
typedef struct _ze_fence_handle_t* ze_fence_handle_t;
typedef ze_result_t(__stdcall* PFN_listCreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_list_desc_t*,
                                              ze_command_list_handle_t*);
typedef ze_result_t(__stdcall* PFN_listClose)(ze_command_list_handle_t);
typedef ze_result_t(__stdcall* PFN_queueCreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*,
                                               ze_command_queue_handle_t*);
typedef ze_result_t(__stdcall* PFN_queueExec)(ze_command_queue_handle_t, uint32_t, ze_command_list_handle_t*, ze_fence_handle_t);
typedef ze_result_t(__stdcall* PFN_queueSync)(ze_command_queue_handle_t, uint64_t);
typedef ze_result_t(__stdcall* PFN_listReset)(ze_command_list_handle_t);
// (Level Zero 1.9, ZE_experimental_immediate_command_list_append: regular lists executed through the immediate list,
//  in its order with everything else appended there)
typedef ze_result_t(__stdcall* PFN_immAppend)(ze_command_list_handle_t, uint32_t, ze_command_list_handle_t*, ze_event_handle_t,
                                             uint32_t, ze_event_handle_t*);
typedef ze_result_t(__stdcall* PFN_extAddr)(ze_driver_handle_t, const char*, void**);

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ---- (2026-11-24, the command-list replay plan's first question) a step recorded ONCE into a mutable command list
// (Level Zero ZE_experimental_mutable_command_list, 1.9+: zeCommandListGetNextCommandIdExp before a kernel is appended,
// zeCommandListUpdateMutableCommandsExp to change its arguments later), then per step only the arguments that change
// (as a token's position does in ~20 kernels) updated, the list closed again and replayed. Its results are compared
// bit for bit with the same sequence appended kernel by kernel to the immediate list, so a driver that accepts the
// calls but does not apply the update is caught. Structures as ze_api.h (oneapi-src/level-zero) defines them.
struct MutListDesc { ze_structure_type_t stype; const void* pNext; uint32_t flags; };   // (stype 0x00020016)
struct MutCmdIdDesc { ze_structure_type_t stype; const void* pNext; uint32_t flags; };  // (0x00020017)
struct MutCmdsDesc { ze_structure_type_t stype; const void* pNext; uint32_t flags; };   // (0x00020018)
struct MutArgDesc {                                                                      // (0x00020019)
  ze_structure_type_t stype;
  const void* pNext;
  uint64_t commandId;
  uint32_t argIndex;
  size_t argSize;
  const void* pArgValue;
};
typedef ze_result_t(__stdcall* PFN_nextId)(ze_command_list_handle_t, const MutCmdIdDesc*, uint64_t*);
typedef ze_result_t(__stdcall* PFN_update)(ze_command_list_handle_t, const MutCmdsDesc*);
static const char* kSrcA = R"CLC(
__kernel void step_a(__global float* y, __global const float* x, int n, float a) {
  const int i = get_global_id(0);
  if (i < n) y[i] = y[i] * 0.999f + a * x[i];
}
)CLC";
static void mutable_replay(ClRt& cl, void* zl, PFN_listCreate listCreate, PFN_listClose listClose, PFN_queueExec queueExec,
                           PFN_queueSync queueSync, ze_command_queue_handle_t q, uint32_t ord, int nk, int nf, int reps,
                           int nu, ze_group_count_t gc, uint32_t lws) {
  const char* what = "mutable list, recorded once";
  auto extAddr = (PFN_extAddr)plat::dl_sym(zl, "zeDriverGetExtensionFunctionAddress");
  auto find = [&](const char* name) -> void* {
    void* f = plat::dl_sym(zl, name);
    if (!f && extAddr) extAddr(zrt::driver, name, &f);
    return f;
  };
  auto nextId = (PFN_nextId)find("zeCommandListGetNextCommandIdExp");
  auto update = (PFN_update)find("zeCommandListUpdateMutableCommandsExp");
  if (!nextId || !update) { printf("%-40s not supported by this driver (no mutable command lists)\n", what); return; }
  cl_program pa = cl.build(kSrcA, "-cl-std=CL3.0");
  cl_kernel ka = cl.kernel(pa, "step_a");
  cl_mem ym = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4), yr = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4),
         x1 = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4);
  // (the program, kernel and buffers released on every way out, an early "not supported" too: the Researcher's review)
  struct Rel {
    cl_program p; cl_kernel k; cl_mem a, b, c;
    ~Rel() { clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(c); clReleaseKernel(k); clReleaseProgram(p); }
  } rel{pa, ka, ym, yr, x1};
  const float zero = 0.0f, one = 1.0f;
  CL_CHECK(clEnqueueFillBuffer(cl.queue, ym, &zero, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  CL_CHECK(clEnqueueFillBuffer(cl.queue, yr, &zero, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  CL_CHECK(clEnqueueFillBuffer(cl.queue, x1, &one, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  clFinish(cl.queue);
  ZE_CHECK(zeKernelSetGroupSize(ka->k, lws, 1, 1));
  nu = std::clamp(nu, 0, nk);
  // (kernel i's a at step r: the nu updated kernels (every nk / nu-th) take a value that changes every step, the rest 1)
  const int stride = nu ? std::max(1, nk / nu) : nk + 1;
  auto a_of = [&](int i, int r) {
    return (nu && i % stride == 0 && i / stride < nu) ? 1.0f + 0.001f * (float)(r + 1) + 0.0001f * (float)i : 1.0f;
  };
  // the reference: the same reps x nk kernels appended to the immediate list, each with its own a
  for (int r = 0; r < reps; ++r)
    for (int i = 0; i < nk; ++i) {
      const float a = a_of(i, r);
      set_args(ka, 0, yr, x1, nf, a);
      ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, ka->k, &gc, nullptr, 0, nullptr));
    }
  zrt::sync();
  // the mutable list: in order (flag 8), recorded once with step 0's values, each kernel's command id kept
  const MutListDesc md{(ze_structure_type_t)0x00020016, nullptr, 0};
  ze_command_list_desc_t ld{0xfu, &md, ord, 8};
  ze_command_list_handle_t rl = nullptr;
  if (listCreate(zrt::ctx, zrt::D().dev, &ld, &rl) != 0) { printf("%-40s not supported (the list does not create)\n", what); return; }
  std::vector<uint64_t> ids((size_t)nk);
  const MutCmdIdDesc idd{(ze_structure_type_t)0x00020017, nullptr, 1u /* kernel arguments */};
  const auto tr = std::chrono::steady_clock::now();
  for (int i = 0; i < nk; ++i) {
    if (nextId(rl, &idd, &ids[(size_t)i]) != 0) {
      printf("%-40s not supported (no command id)\n", what);
      zeCommandListDestroy(rl);
      return;
    }
    const float a = a_of(i, 0);
    set_args(ka, 0, ym, x1, nf, a);
    ZE_CHECK(zeCommandListAppendLaunchKernel(rl, ka->k, &gc, nullptr, 0, nullptr));
  }
  ZE_CHECK(listClose(rl));
  const double rec = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
  std::vector<double> t, tu;
  std::vector<float> av((size_t)nk);
  std::vector<MutArgDesc> ad;
  bool upd_ok = true;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    if (r > 0 && nu > 0) {  // (step 0 ran with the recorded values)
      ad.clear();
      for (int i = 0; i < nk; ++i)
        if (a_of(i, r) != a_of(i, r - 1)) {
          av[(size_t)i] = a_of(i, r);
          ad.push_back(MutArgDesc{(ze_structure_type_t)0x00020019, nullptr, ids[(size_t)i], 3u, sizeof(float), &av[(size_t)i]});
        }
      for (size_t j = 0; j + 1 < ad.size(); ++j) ad[j].pNext = &ad[j + 1];
      const MutCmdsDesc cd{(ze_structure_type_t)0x00020018, ad.empty() ? nullptr : &ad[0], 0};
      if (update(rl, &cd) != 0 || listClose(rl) != 0) { upd_ok = false; break; }
    }
    tu.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ZE_CHECK(queueExec(q, 1, &rl, nullptr));
    ZE_CHECK(queueSync(q, UINT64_MAX));
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  zeCommandListDestroy(rl);
  if (!upd_ok) { printf("%-40s not supported (the update or the second close failed)\n", what); return; }
  std::vector<float> hm((size_t)nf), hr((size_t)nf);
  CL_CHECK(clEnqueueReadBuffer(cl.queue, ym, CL_TRUE, 0, (size_t)nf * 4, hm.data(), 0, nullptr, nullptr));
  CL_CHECK(clEnqueueReadBuffer(cl.queue, yr, CL_TRUE, 0, (size_t)nf * 4, hr.data(), 0, nullptr, nullptr));
  const bool same = memcmp(hm.data(), hr.data(), (size_t)nf * 4) == 0;
  printf("%-40s %8.3f ms per step (%.2f us per kernel; %d kernels' arguments updated a step in %.3f ms; recorded once in "
         "%.2f ms); results %s\n", what, median(t), median(t) * 1000 / nk, nu, median(tu), rec,
         same ? "the same as appended, bit for bit" : "DIFFER from appended (the update is not applied: please report)");
}

int main(int argc, char** argv) {
  int nk = 700, nf = 65536, reps = 30, nu = 20;
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      printf("usage: replay_probe [--kernels 700] [--floats 65536] [--reps 30] [--updates 20]\n"
             "Times a chain of small dependent kernels launched one by one, as the engine does, against the same chain\n"
             "recorded once in a command list and replayed: the GPU's idle time between kernels that a replay saves.\n"
             "  --kernels N   kernels per chain (default 700, about one token of the 35B-A3B; 930 for the 27B)\n"
             "  --floats N    floats each kernel reads and writes (default 65536)\n"
             "  --reps N      chains timed per way, the median counts (default 30)\n"
             "  --updates N   the mutable list's kernels whose argument changes every step (default 20, as a token's\n"
             "                position in ~20 kernels; 0: a pure replay)\n");
      return 0;
    }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--kernels") nk = atoi(argv[i + 1]);
    else if (a == "--floats") nf = atoi(argv[i + 1]);
    else if (a == "--reps") reps = atoi(argv[i + 1]);
    else if (a == "--updates") nu = atoi(argv[i + 1]);
  }
  ClRt cl;
  if (!cl.init()) return 1;
  void* zl = plat::dl_loaded("ze_loader.dll", "libze_loader.so.1");  // (Windows and Linux: src/platform.h)
  if (!zl) { fprintf(stderr, "Level Zero: the loader is not loaded
"); return 1; }
  auto listCreate = (PFN_listCreate)plat::dl_sym(zl, "zeCommandListCreate");
  auto listClose = (PFN_listClose)plat::dl_sym(zl, "zeCommandListClose");
  auto queueCreate = (PFN_queueCreate)plat::dl_sym(zl, "zeCommandQueueCreate");
  auto queueExec = (PFN_queueExec)plat::dl_sym(zl, "zeCommandQueueExecuteCommandLists");
  auto queueSync = (PFN_queueSync)plat::dl_sym(zl, "zeCommandQueueSynchronize");
  if (!listCreate || !listClose || !queueCreate || !queueExec || !queueSync) { fprintf(stderr, "Level Zero: no regular command lists\n"); return 1; }
  uint32_t ng = 0, ord = 0;
  zeDeviceGetCommandQueueGroupProperties(zrt::D().dev, &ng, nullptr);
  std::vector<ze_command_queue_group_properties_t> gp(ng, ze_command_queue_group_properties_t{ZE_STYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
  zeDeviceGetCommandQueueGroupProperties(zrt::D().dev, &ng, gp.data());
  for (uint32_t g = 0; g < ng; ++g)
    if (gp[g].flags & 1) { ord = g; break; }

  cl_program prog = cl.build(kSrc, "-cl-std=CL3.0");
  cl_kernel k = cl.kernel(prog, "step");
  cl_mem y = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4), x = cl.buffer(CL_MEM_READ_WRITE, (size_t)nf * 4);
  const float zero = 0.0f;
  CL_CHECK(clEnqueueFillBuffer(cl.queue, y, &zero, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  CL_CHECK(clEnqueueFillBuffer(cl.queue, x, &zero, 4, 0, (size_t)nf * 4, 0, nullptr, nullptr));
  set_args(k, 0, y, x, nf);
  const size_t gws = ((size_t)nf + 255) / 256 * 256, lws = 256;
  clFinish(cl.queue);
  ze_group_count_t gc{(uint32_t)(gws / lws), 1, 1};
  ZE_CHECK(zeKernelSetGroupSize(k->k, (uint32_t)lws, 1, 1));
  auto ms_since = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  };

  // 1. now: every kernel appended to the immediate in-order list, then a host wait
  std::vector<double> imm;
  for (int r = 0; r < reps; ++r) {
    const auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < nk; ++i) ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
    zrt::sync();
    imm.push_back(ms_since(t));
  }
  // 2. regular lists, recorded once and replayed: with a barrier after each kernel, and with the in-order flag
  ze_command_queue_desc_t qd{ZE_STYPE_COMMAND_QUEUE_DESC, nullptr, ord, 0, 0, 2 /* async */, 0};
  ze_command_queue_handle_t q = nullptr;
  ZE_CHECK(queueCreate(zrt::ctx, zrt::D().dev, &qd, &q));
  auto replay = [&](uint32_t flags, bool barriers, const char* what) {
    ze_command_list_desc_t ld{0xfu /* ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC */, nullptr, ord, flags};
    ze_command_list_handle_t rl = nullptr;
    if (listCreate(zrt::ctx, zrt::D().dev, &ld, &rl) != 0) { printf("%-40s not supported\n", what); return; }
    const auto tr = std::chrono::steady_clock::now();
    for (int i = 0; i < nk; ++i) {
      ZE_CHECK(zeCommandListAppendLaunchKernel(rl, k->k, &gc, nullptr, 0, nullptr));
      if (barriers) ZE_CHECK(zeCommandListAppendBarrier(rl, nullptr, 0, nullptr));
    }
    ZE_CHECK(listClose(rl));
    const double rec = ms_since(tr);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      ZE_CHECK(queueExec(q, 1, &rl, nullptr));
      ZE_CHECK(queueSync(q, UINT64_MAX));
      t.push_back(ms_since(t0));
    }
    printf("%-40s %8.3f ms per step (%.2f us per kernel; recorded once in %.2f ms)\n", what, median(t), median(t) * 1000 / nk, rec);
    zeCommandListDestroy(rl);
  };
  printf("%d dependent kernels of %d floats each, median of %d steps\n", nk, nf, reps);
  printf("%-40s %8.3f ms per step (%.2f us per kernel)\n", "immediate list (the engine now)", median(imm), median(imm) * 1000 / nk);
  replay(0, true, "replayed list, barrier after each kernel");
  replay(8, false, "replayed list, in-order flag");
  mutable_replay(cl, zl, listCreate, listClose, queueExec, queueSync, q, ord, nk, nf, reps, nu, gc, (uint32_t)lws);

  // 3. the engine's way to a replay without recording a step once for all (its arguments change every token):
  // each step recorded again into regular in-order lists of `chunk` kernels, each list appended to the immediate list
  // as soon as it is full (zeCommandListImmediateAppendCommandListsExp), so the GPU starts on the first chunk while
  // the host records the next. The recording's host time is counted. chunk = all: one list, recorded, then appended.
  auto listReset = (PFN_listReset)plat::dl_sym(zl, "zeCommandListReset");
  auto immAppend = (PFN_immAppend)plat::dl_sym(zl, "zeCommandListImmediateAppendCommandListsExp");
  if (!immAppend) {
    auto extAddr = (PFN_extAddr)plat::dl_sym(zl, "zeDriverGetExtensionFunctionAddress");
    void* f = nullptr;
    if (extAddr && extAddr(zrt::driver, "zeCommandListImmediateAppendCommandListsExp", &f) == 0) immAppend = (PFN_immAppend)f;
  }
  if (!immAppend || !listReset) {
    printf("%-40s not supported by this driver\n", "lists appended to the immediate list");
    return 0;
  }
  {  // (a pre-recorded list appended: the same replay as 2., through the immediate list)
    ze_command_list_desc_t ld{0xfu, nullptr, ord, 8};
    ze_command_list_handle_t rl = nullptr;
    ZE_CHECK(listCreate(zrt::ctx, zrt::D().dev, &ld, &rl));
    for (int i = 0; i < nk; ++i) ZE_CHECK(zeCommandListAppendLaunchKernel(rl, k->k, &gc, nullptr, 0, nullptr));
    ZE_CHECK(listClose(rl));
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      ZE_CHECK(immAppend(zrt::D().cl, 1, &rl, nullptr, 0, nullptr));
      zrt::sync();
      t.push_back(ms_since(t0));
    }
    printf("%-40s %8.3f ms per step (%.2f us per kernel)\n", "replayed list, appended to the immediate", median(t),
           median(t) * 1000 / nk);
    zeCommandListDestroy(rl);
  }
  for (int chunk : {8, 32, 128, nk}) {
    const int nl = (nk + chunk - 1) / chunk;
    std::vector<ze_command_list_handle_t> ls(nl);
    for (auto& l : ls) {
      ze_command_list_desc_t ld{0xfu, nullptr, ord, 8};
      ZE_CHECK(listCreate(zrt::ctx, zrt::D().dev, &ld, &l));
    }
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int c = 0; c < nl; ++c) {
        ZE_CHECK(listReset(ls[c]));  // (the previous step finished: sync below)
        const int n = std::min(chunk, nk - c * chunk);
        for (int i = 0; i < n; ++i) ZE_CHECK(zeCommandListAppendLaunchKernel(ls[c], k->k, &gc, nullptr, 0, nullptr));
        ZE_CHECK(listClose(ls[c]));
        ZE_CHECK(immAppend(zrt::D().cl, 1, &ls[c], nullptr, 0, nullptr));
      }
      zrt::sync();
      t.push_back(ms_since(t0));
    }
    char what[64];
    snprintf(what, sizeof what, "recorded each step, lists of %d", chunk);
    printf("%-40s %8.3f ms per step (%.2f us per kernel)\n", what, median(t), median(t) * 1000 / nk);
    for (auto l : ls) zeCommandListDestroy(l);
  }
  // (laptop, Core Ultra 5 125H iGPU: 700 kernels of 4,096 floats 1.36 ms immediate, 1.07 replayed; of 65,536 floats
  //  1.95 against 2.01: a replay saves ~0.4 us a kernel at most there)
  return 0;
}
