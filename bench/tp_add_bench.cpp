// tp_add's read: a split model's sum kernel (zero/kernels/tp.cl) reads every shard's part out of one pinned host
// buffer, `stride` floats apart. This measures that kernel alone, on the same layout, at the strides a model gives it,
// so the question "is it the distance between the parts or the number of them?" is answered without loading a model.
//   bin/tp_add_bench [--stride-mb 80] [--np 2,3,4] [--values 5120] [--reps 256] [--pinned 1]
// --values is n of the kernel (a generated token's n_embd; a prompt chunk's rows * n_embd). --pinned 0 puts the parts
// in a device buffer instead, the form the direct sums (`ARC_TP_P2P`) read, to tell pinned host memory from VRAM.
#include "../zero/clrt.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// (the kernel as zero/kernels/tp.cl has it, the vector form, and the scalar one it replaced: both measured)
static const char* kSrc = R"CLC(
__kernel void tp_add(__global float* x, __global const float* p, ulong off, int n, int np, ulong stride) {
  const int i0 = get_global_id(0) * 4;
  if (i0 + 3 < n) {
    float4 s = vload4(0, x + i0);
    for (int k = 0; k < np; ++k) s += vload4(0, p + off + k * stride + i0);
    vstore4(s, 0, x + i0);
    return;
  }
  for (int i = i0; i < n; ++i) {
    float s = x[i];
    for (int k = 0; k < np; ++k) s += p[off + k * stride + i];
    x[i] = s;
  }
}
__kernel void tp_add_scalar(__global float* x, __global const float* p, ulong off, int n, int np, ulong stride) {
  const int i = get_global_id(0);
  if (i >= n) return;
  float s = x[i];
  for (int k = 0; k < np; ++k) s += p[off + k * stride + i];
  x[i] = s;
}
)CLC";

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  double stride_mb = 80.0;
  std::string npl = "2,3,4";
  int values = 5120, reps = 256, pinned = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      printf("usage: tp_add_bench [--stride-mb 80] [--np 2,3,4] [--values 5120] [--reps 256] [--pinned 1]\n"
             "The time of zero/kernels/tp.cl's tp_add: np parts, `stride` floats apart, `values` of them summed.\n"
             "  --stride-mb D  distance between two parts (default 80: a 4096-token chunk of the 27B)\n"
             "  --np L         part counts to test, comma-separated (default 2,3,4)\n"
             "  --values N     values summed (default 5120: n_embd of the 27B; a prompt chunk: rows * n_embd)\n"
             "  --reps R       launches a test (default 256)\n"
             "  --pinned 0     the parts in a device buffer instead of pinned host memory\n");
      return 0;
    }
  }
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) break;
    if (a == "--stride-mb") stride_mb = atof(argv[i + 1]);
    else if (a == "--np") npl = argv[i + 1];
    else if (a == "--values") values = atoi(argv[i + 1]);
    else if (a == "--reps") reps = atoi(argv[i + 1]);
    else if (a == "--pinned") pinned = atoi(argv[i + 1]);
  }
  std::vector<int> nps;
  for (size_t p = 0; p <= npl.size();) {
    size_t q = npl.find(',', p);
    if (q == std::string::npos) q = npl.size();
    if (q > p) nps.push_back(std::max(1, atoi(npl.substr(p, q - p).c_str())));
    p = q + 1;
  }
  if (nps.empty()) nps.push_back(4);
  if (values % 4) { fprintf(stderr, "--values must be a multiple of 4\n"); return 1; }

  ClRt cl;
  if (!cl.init()) return 1;
  printf("GPU: %s | parts in %s\n", cl.device_id.c_str(), pinned ? "pinned host memory" : "a device buffer");
  cl_program prog = cl.build(kSrc, "-cl-std=CL3.0");
  cl_kernel kv = cl.kernel(prog, "tp_add");
  cl_kernel ks = cl.kernel(prog, "tp_add_scalar");

  const size_t stride = (size_t)(stride_mb * 1048576.0 / 4);  // (floats, as toe stride of the kernel)
  const int maxnp = *std::max_element(nps.begin(), nps.end());
  const size_t parts = (size_t)2 * maxnp;  // (two halves, as TpComm::host)
  const size_t bytes = parts * stride * 4 + (size_t)values * 4;
  void* hbuf = pinned ? zrt::host_alloc(bytes) : zrt::device_alloc(bytes);
  if (!hbuf) { fprintf(stderr, "could not allocate %zu MB\n", bytes >> 20); return 1; }
  {  // random words (not one value: the cards' memory compression, membw_probe's note)
    std::vector<uint32_t> w(1 << 16);
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (auto& v : w) { s ^= s << 13, s ^= s >> 7, s ^= s << 17; v = (uint32_t)(s >> 16); }
    for (size_t off = 0; off < bytes; off += w.size() * 4)
      memcpy((char*)hbuf + off, w.data(), std::min(w.size() * 4, bytes - off));
  }
  cl_mem pbuf = pinned ? zrt::wrap(hbuf, bytes) : (cl_mem)hbuf;
  cl_mem xbuf = cl.buffer(CL_MEM_READ_WRITE, (size_t)values * 4);
  printf("stride %.1f MB, %zu parts, %d values, %d launches a test\n", stride_mb, parts, values, reps);

  // one test: `np` parts summed `reps` times; the parts are k * stride apart, x at the front
  auto run = [&](cl_kernel k, int np, bool vec) {
    const size_t g = vec ? (size_t)((values / 4 + 255) / 256 * 256) : (size_t)((values + 255) / 256 * 256);
    set_args(k, 0, xbuf, pbuf, (cl_ulong)0, values, np, (cl_ulong)stride);
    ZE_CHECK(zeKernelSetGroupSize(k->k, 256, 1, 1));
    ze_group_count_t gc{(uint32_t)(g / 256), 1, 1};
    for (int r = 0; r < 8; ++r) {  // (build, first touch, caches warm)
      ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
      zrt::sync();
    }
    const auto t0 = clk::now();
    for (int r = 0; r < reps; ++r) ZE_CHECK(zeCommandListAppendLaunchKernel(zrt::D().cl, k->k, &gc, nullptr, 0, nullptr));
    zrt::sync();
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
  };

  printf("\n%-4s %12s %12s %12s %12s\n", "np", "vector us", "scalar us", "vec GB/s", "scalar GB/s");
  for (int np : nps) {
    const double tv = run(kv, np, true), ts = run(ks, np, false);
    const double gb = (double)values * np * 4 / 1e9;  // (x read + written counts with the parts' np-1 foreign reads)
    printf("%-4d %12.2f %12.2f %12.1f %12.1f\n", np, tv * 1000, ts * 1000, gb / (tv / 1000), gb / (ts / 1000));
  }
  return 0;
}
