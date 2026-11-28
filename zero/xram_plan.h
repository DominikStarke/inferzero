// The RAM tier's room on Windows and its runtime guard, as pure functions (no GPU, no Windows call), so they are tested
// on the host alone (tests/test_xram_plan.cpp) with the numbers of the machines that failed. engine_stream.h feeds them
// what it reads (Windows' non-local budget and usage, zrt::video_budget) and what it knows it maps.
// PR #100 (manoreken2, 4 Pro B60s, Windows, 1 TB RAM, DeepSeek-V4.1-Flash UD-Q8_K_XL): the pinned host memory of the
// one context the shards share counts against Windows' budget once per GPU, so a split's tiers together get about
// budget / N; the automatic size 49.1 GiB a GPU ran out of memory or hung at 204.5-205.5 GiB mapped, 47 ran, 48 failed.
#pragma once
#include <algorithm>
#include <cstdint>
#include <mutex>

namespace xram {

constexpr uint64_t kGiB = 1ull << 30;

// what the engine maps of RAM outside the tiers, known before the other GPUs have made it: every GPU's expert staging
// (xcache.h: 2 x the largest record x ARC_XSTAGE records) and 64 MiB of other pinned buffers a GPU (the read bounce
// buffers, the expert tables), and a split's sum buffers (engine_tp.h comm_.host: 2 x N x the chunk x n_embd floats)
inline uint64_t known_mapped(int n_gpus, bool split, uint64_t stage_bytes, uint64_t chunk, uint64_t n_embd) {
  const uint64_t N = split ? (uint64_t)std::max(1, n_gpus) : 1;
  const uint64_t sums = split ? 2 * N * chunk * n_embd * 4 : 0;
  return N * (stage_bytes + (64ull << 20)) + sums;
}

// the margin above it: a split max(8 GiB, 5 % of budget / N), one GPU max(4 GiB, 2.5 %), the 4 GiB of before its floor
inline uint64_t margin(uint64_t top, bool split) {
  return split ? std::max<uint64_t>(8 * kGiB, top / 20) : std::max<uint64_t>(4 * kGiB, top / 40);
}

// The start size: the RAM tier a GPU may take. budget / in_use: Windows' non-local budget and usage as this GPU reads
// them (in use read on GPU 0 before the others have made their staging, so the known mapped buffers count when they
// are more). A split: budget / N for all the GPUs, less what they map and the margin, / N a GPU; one GPU: the budget
// less those. 0 when nothing is left.
struct Room {
  uint64_t top = 0, used = 0, margin = 0, total = 0, per_gpu = 0;
};
inline Room room(uint64_t budget, uint64_t in_use, uint64_t known, int n_gpus, bool split) {
  Room r;
  const uint64_t N = split ? (uint64_t)std::max(1, n_gpus) : 1;
  r.top = budget / N;
  r.used = std::max(in_use, known);
  r.margin = margin(r.top, split);
  r.total = r.top > r.used + r.margin ? r.top - r.used - r.margin : 0;
  r.per_gpu = r.total / N;
  return r;
}

// The runtime guard: before the tier pins `piece` more bytes, with `in_use` of RAM mapped now (Windows' non-local
// usage, every GPU's tier and staging in it), whether it may: budget / N less min(4 GiB, a tenth of it). A budget of
// 0 (unknown: Linux, or the query failed) never stops it.
inline uint64_t guard_margin(uint64_t top) { return std::min<uint64_t>(4 * kGiB, top / 10); }
inline bool may_take(uint64_t budget, uint64_t in_use, int n_gpus, bool split, uint64_t piece) {
  if (budget == 0) return true;
  const uint64_t top = budget / (split ? (uint64_t)std::max(1, n_gpus) : 1);
  return in_use + piece + guard_margin(top) <= top;
}

// The guard's check and the allocation as one step for the whole process: a split's GPUs fill their tiers on threads
// of their own, and N pieces passing the same check together are N GiB, the guard's whole margin on 4 GPUs (the
// Researcher's review of c85d0d9). read(budget, in_use) -> false when the budget is unknown (no stop then);
// alloc(n) -> the memory or null. *stopped: set when the guard refused (not when the allocation failed).
inline std::mutex& take_mutex() {
  static std::mutex m;
  return m;
}
template <typename Read, typename Alloc>
void* guarded_alloc(Read read, Alloc alloc, int n_gpus, bool split, uint64_t piece, bool* stopped = nullptr,
                    uint64_t* budget_out = nullptr, uint64_t* in_use_out = nullptr) {
  std::lock_guard<std::mutex> lock(take_mutex());
  uint64_t budget = 0, in_use = 0;
  if (read(budget, in_use) && !may_take(budget, in_use, n_gpus, split, piece)) {
    if (stopped) *stopped = true;
    if (budget_out) *budget_out = budget;
    if (in_use_out) *in_use_out = in_use;
    return nullptr;
  }
  return alloc(piece);
}

}  // namespace xram
