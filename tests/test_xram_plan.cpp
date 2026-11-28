// The RAM tier's room and its runtime guard (zero/xram_plan.h), host only: the numbers of the machines whose tier failed.
// Build: .\build.ps1 tests\test_xram_plan.cpp   Run: .\bin\test_xram_plan.exe   (prints ALL OK)
#include "../zero/xram_plan.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
static const double GiB = 1073741824.0;
static uint64_t gib(double g) { return (uint64_t)(g * GiB); }

int main() {
  using namespace xram;
  // manoreken2 (PR #100): 4 Pro B60s, Windows' non-local budget 818.15 GiB, staging 2,295 MiB a GPU, DeepSeek-V4.1's
  // n_embd 4096; the chunk as engine_stream.h counts it, the largest a plan picks (8,192: its sums 1 GiB, erring high;
  // his own is 1,024); "in use" read on GPU 0 as 0.0 before GPUs 1-3 made their staging
  const uint64_t budget = gib(818.15), staging = 2295ull << 20;
  const uint64_t known = known_mapped(4, true, staging, 8192, 4096);
  CHECK(known > gib(10.1) && known < gib(10.3));  // (4 x (2.24 + 0.0625) + 1.0)
  CHECK(known_mapped(4, true, staging, 1024, 4096) < known);  // (his own chunk: smaller sums)
  const Room r = room(budget, 0, known, 4, true);
  printf("4 B60s: top %.2f, mapped %.2f, margin %.2f, %.2f in all, %.2f a GPU\n", r.top / GiB, r.used / GiB, r.margin / GiB,
         r.total / GiB, r.per_gpu / GiB);
  CHECK(r.per_gpu >= gib(45.5) && r.per_gpu <= gib(46.6));  // (46.2: under the 47 that ran, the 48 that failed)
  CHECK(r.margin >= gib(10.0) && r.margin <= gib(10.4));    // (5 % of 204.5)
  // the old estimate, "in use" alone and 8 GiB: 49.1 a GPU (the size that failed)
  {
    const uint64_t top = budget / 4, old = (top - 0 - gib(8)) / 4;
    CHECK(old > gib(49.0) && old < gib(49.3));
    CHECK(r.per_gpu < old);
  }
  // in use read later, above the known buffers: it counts instead (the larger of the two)
  CHECK(room(budget, gib(20), known, 4, true).used == gib(20));
  CHECK(room(budget, gib(20), known, 4, true).per_gpu < r.per_gpu);
  // a hand-set 48 a GPU is over the room (engine_stream.h caps it to per_gpu, with its line)
  CHECK(gib(48) > r.per_gpu);
  CHECK(gib(47) > r.per_gpu);  // (and 47, which ran: the estimate keeps a margin below it)

  // the guard: the tiers fill during the run. All 4 GPUs at the size: the mapped total (tiers + outside) under top
  {
    const uint64_t top = budget / 4, piece = gib(1);
    uint64_t in_use = known;
    int pieces = 0;
    while (may_take(budget, in_use, 4, true, piece) && pieces < 1000) in_use += piece, ++pieces;
    printf("guard: %d pieces of 1 GiB taken, mapped %.2f of %.2f (stops %.2f short)\n", pieces, in_use / GiB, top / GiB,
           (top - in_use) / GiB);
    CHECK(in_use <= top - guard_margin(top));
    CHECK(in_use + piece + guard_margin(top) > top);  // (stopped at the first piece that would pass it)
    CHECK(guard_margin(top) == gib(4));               // (min(4, 20.45))
    // the failures were at 204.45-205.45 mapped: the guard never lets the mapped RAM reach 204.5
    CHECK(in_use < gib(204.45));
  }
  // a budget that runs out mid-fill: something else maps 150 GiB more during the run; the next piece is refused
  CHECK(may_take(budget, gib(150), 4, true, gib(1)));
  CHECK(!may_take(budget, gib(200), 4, true, gib(1)));
  CHECK(!may_take(budget, gib(204.5), 4, true, gib(1)));
  // one GPU (LukaszBrzyszkiewicz's B70, #82: a budget of 47.9 GiB, a 52.1 GiB tier ran out of memory at the first copy)
  {
    const uint64_t b = gib(47.9), k = known_mapped(1, false, 244ull << 20, 4096, 2048);
    CHECK(k < gib(0.4));  // (no sums on one GPU)
    const Room r1 = room(b, gib(2), k, 1, false);
    printf("one B70: %.2f a GPU (margin %.2f)\n", r1.per_gpu / GiB, r1.margin / GiB);
    CHECK(r1.margin == gib(4));  // (max(4, 1.2))
    CHECK(r1.per_gpu < gib(52.1) && r1.per_gpu > gib(41) && r1.per_gpu < gib(42));  // (47.9 - 2 - 4)
    uint64_t in_use = gib(2);
    while (may_take(b, in_use, 1, false, gib(1))) in_use += gib(1);
    CHECK(in_use <= b - guard_margin(b) && in_use + gib(1) + guard_margin(b) > b);
  }
  // an unknown budget (Linux, or the query failed): no stop, and no room from it
  CHECK(may_take(0, gib(500), 4, true, gib(1)));
  CHECK(room(0, 0, known, 4, true).per_gpu == 0);
  // nothing left: room 0, never an underflow
  CHECK(room(gib(16), gib(30), 0, 4, true).per_gpu == 0);
  CHECK(room(gib(3), 0, 0, 1, false).per_gpu == 0);  // (under the margin's 4 GiB)
  CHECK(room(gib(5), 0, 0, 1, false).per_gpu == gib(1));
  // the chunk the sums are counted at: a B70's plan may pick 8,192 rows where the option and the old default said
  // 4,096; the larger count errs high (engine_stream.h passes the largest a plan picks)
  CHECK(known_mapped(2, true, staging, 8192, 5120) > known_mapped(2, true, staging, 4096, 5120));
  CHECK(known_mapped(2, true, staging, 8192, 5120) - known_mapped(2, true, staging, 4096, 5120) ==
        2ull * 2 * 4096 * 5120 * 4);  // (2 x N x the extra rows x n_embd floats)
  // many GPUs' threads taking at the edge at once (guarded_alloc): the mapped total never passes the guard's line.
  // The usage the read sees is what the allocations so far added (the tiers' pinned memory as Windows counts it), with
  // the room for as many pieces as one thread alone would take (computed below); 16 threads try 40 pieces each.
  // Without the one mutex, up to 16 could pass a check together (the review's case: 4 GPUs x 1 GiB, the margin's
  // whole 4 GiB)
  {
    const uint64_t b = gib(4 * 10.5), piece = gib(1);  // (top 10.5 a GPU, 1 in use, the guard's margin a tenth: 1.05)
    std::atomic<uint64_t> used{gib(1)};
    std::atomic<int> taken{0}, stops{0};
    auto read = [&](uint64_t& bb, uint64_t& u) { bb = b, u = used.load(); return true; };
    auto alloc = [&](uint64_t n) -> void* {
      // (a host allocation takes time: the window between the check and the count, where an unlocked guard lets other
      //  threads pass the same check; with the window, a build without the lock takes 16+ pieces here)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      used += n;
      return (void*)1;
    };
    std::atomic<int> ready{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 16; ++t)
      ts.emplace_back([&] {
        ++ready;
        while (ready.load() < 16) std::this_thread::yield();  // (all at the edge together)
        for (int i = 0; i < 40; ++i) {
          bool st = false;
          if (guarded_alloc(read, alloc, 4, true, piece, &st)) ++taken;
          else if (st) ++stops;
        }
      });
    for (auto& t : ts) t.join();
    const uint64_t top = b / 4;
    printf("16 threads at the edge: %d pieces taken, %d refused, mapped %.2f of %.2f (the line %.2f)\n", taken.load(),
           stops.load(), used.load() / GiB, top / GiB, (top - guard_margin(top)) / GiB);
    int alone = 0;  // (what one thread alone takes from the same start: the concurrent ones must take no more)
    for (uint64_t u = gib(1); may_take(b, u, 4, true, piece); u += piece) ++alone;
    CHECK(alone == 8);  // (1 + 8 = 9 <= 10.5 - 1.05)
    CHECK(taken.load() == alone);
    CHECK(used.load() + guard_margin(top) <= top);
    CHECK(stops.load() == 16 * 40 - alone);
  }
  // an unknown budget through guarded_alloc: every piece taken, none refused
  {
    bool st = false;
    int n = 0;
    for (int i = 0; i < 3; ++i)
      if (guarded_alloc([](uint64_t&, uint64_t&) { return false; }, [](uint64_t) -> void* { return (void*)1; }, 4, true,
                        gib(1), &st))
        ++n;
    CHECK(n == 3 && !st);
  }
  // a split's margin grows with the budget past 160 GiB a GPU set (5 %), its floor 8 GiB
  CHECK(margin(gib(100), true) == gib(8));
  CHECK(margin(gib(400), true) == gib(20));
  if (fails) { printf("%d FAILED\n", fails); return 1; }
  printf("ALL OK\n");
  return 0;
}
