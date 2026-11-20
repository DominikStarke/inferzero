// inferzero: the routed experts of a MoE GGUF in a file of their own on the SSD, laid out for streaming:
//   <gguf name>-<fingerprint>.experts next to the GGUF when it is inside %USERPROFILE%\.inferzero, else in
//   %USERPROFILE%\.inferzero\zero\models\<gguf name>\ (open_or_build; before 2026-11-09 all in ...\zero\)
// Every (layer, expert) is one record: its gate, up and down rows one after the other in the GPU raw form inferzero
// uploads (src/quants.h gpu_raw_sb: regrouped / padded superblocks), the record padded to a multiple of 4 KB, so one
// unbuffered read (FILE_FLAG_NO_BUFFERING: no Windows file cache in RAM) brings a whole expert, ready for the GPU.
// The file is made from the GGUF when missing or when it does not match the GGUF (size, time, name: the fingerprint),
// and reused otherwise; it can be deleted at any time.
#pragma once
#include <atomic>
#include <csignal>
#include "../src/app_dir.h"
#include "../src/gguf.h"
#include "../src/autoround_revision.h"
#include "../src/load_progress.h"
#include "../src/quants.h"
#include "io_beat.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ---- a stop (SIGTERM: docker compose stop, a pod deleted; SIGINT: Ctrl+C) while an expert file or a package is
// written: the work stops between batches, its .tmp files are removed, and the signal then does what it did before
// (by default it ends the process). Before (self review 2026-11-19, D1), SIGTERM ended serverz at once mid-build: a
// .tmp of tens of GB left behind, and the next start began again from zero. The handler only sets a flag.
inline std::atomic<int> g_xbuild_sig{0};
extern "C" inline void xbuild_on_signal(int sig) { g_xbuild_sig.store(sig); }
// A .tmp left by a build that was killed (power cut, SIGKILL, a crash): as large as the file it was to become. Removed
// with a line saying so before a new build writes it again (it was truncated silently before, its space taken until
// then; S7).
// (self audit 2026-11-20 A9: the caller holds `lock`, a plat::BuildLock on tmp + ".lock", for the whole build: a .tmp
//  another process is still writing has its lock held, and this one stops with a message instead of removing it; the
//  line says a removal only when one happened)
inline void drop_stale_tmp(const std::string& tmp, const plat::BuildLock& lock) {
  if (!lock.held())
    throw std::runtime_error("another inferzero is writing " + tmp + " now: wait for it to finish, or stop it");
  std::error_code e;
  const auto st = std::filesystem::status(std::filesystem::path(tmp), e);
  if (e || !std::filesystem::is_regular_file(st)) return;
  const uintmax_t n = std::filesystem::file_size(std::filesystem::path(tmp), e);
  const double gb = e ? 0.0 : n / 1e9;
  if (std::filesystem::remove(std::filesystem::path(tmp), e) && !e)
    zlog::eprintf("inferzero: removed %s (%.1f GB) left by an earlier build that did not finish\n", tmp.c_str(), gb);
  else
    zlog::eprintf("inferzero: could not remove %s (%.1f GB) left by an earlier build: %s\n", tmp.c_str(), gb,
                  e.message().c_str());
}
// Refuses to start writing `bytes` next to `path` when its drive has less free (plus 256 MB of room): a full disk
// stopped a build after tens of minutes, with the numbers nowhere (S3). A drive whose free space cannot be read: no
// check.
inline void need_free_space(const std::string& path, uint64_t bytes, const char* what) {
  std::error_code e;
  const auto dir = std::filesystem::path(path).parent_path();
  const auto sp = std::filesystem::space(dir.empty() ? std::filesystem::path(".") : dir, e);
  if (e) return;
  const uint64_t need = bytes + (256ull << 20);
  if (sp.available < need) {
    char msg[160];
    snprintf(msg, sizeof msg, " needs %.1f GB on its drive, %.1f GB are free: ", need / 1e9, sp.available / 1e9);
    throw std::runtime_error(std::string(what) + " " + path + msg + "free some space, or put the models on another drive");
  }
}
// (self audit 2026-11-20 A8: guards counted, not told apart by the handler they replaced, so two builds on two threads
//  (a pack beside a load) no longer restore each other's handlers out of order; the first installs, the last restores
//  and hands the signal on. A signal the process ignored (SIG_IGN, e.g. nohup's SIGINT) stays ignored: a build is not
//  stopped by it)
struct XBuildStopGuard {
  static std::mutex& mu() { static std::mutex m; return m; }
  static inline int depth = 0;
  static inline void (*old_int)(int) = SIG_DFL;
  static inline void (*old_term)(int) = SIG_DFL;
  static void (*take(int sig))(int) {
    void (*old)(int) = std::signal(sig, xbuild_on_signal);
    if (old == SIG_IGN) std::signal(sig, SIG_IGN);  // (kept ignored)
    return old;
  }
  XBuildStopGuard() {
    std::lock_guard<std::mutex> g(mu());
    if (depth++ == 0) {
      old_int = take(SIGINT);
      old_term = take(SIGTERM);
    }
  }
  ~XBuildStopGuard() {
    std::lock_guard<std::mutex> g(mu());
    if (--depth > 0) return;
    // (the last guard restores the handlers and hands the signal on, once the .tmp files are gone)
    std::signal(SIGINT, old_int == SIG_ERR ? SIG_DFL : old_int);
    std::signal(SIGTERM, old_term == SIG_ERR ? SIG_DFL : old_term);
    const int sig = g_xbuild_sig.exchange(0);
    if (sig) std::raise(sig);
  }
  static void check(const std::string& what) {
    if (g_xbuild_sig.load())
      throw std::runtime_error("stopped by " + std::string(g_xbuild_sig.load() == SIGTERM ? "SIGTERM" : "Ctrl+C / SIGINT") +
                               " while writing " + what + ": its .tmp removed, the next start begins it again");
  }
};

struct ExpertStore {
  struct LayerInfo {
    uint32_t type[3] = {};  // gate, up, down GGUF types
    uint64_t part[3] = {};  // bytes of each matrix of one expert (GPU raw form)
    uint64_t rec = 0;       // bytes of one expert record (multiple of 4096)
    uint64_t off = 0;       // file offset of the layer's expert 0
    bool operator==(const LayerInfo& other) const {
      // (field by field: the 4 padding bytes after type[] are no field and GCC leaves them unset, so a memcmp of the
      //  entry refused a matching AutoRound package, or wrote a model's expert file again; pull request #24)
      for (int i = 0; i < 3; ++i)
        if (type[i] != other.type[i] || part[i] != other.part[i]) return false;
      return rec == other.rec && off == other.off;
    }
  };
  std::vector<LayerInfo> layers;  // by layer index; rec == 0: not a MoE layer
  int n_expert = 0;
  std::string path;
  uint64_t file_bytes = 0;
  double build_ms = 0;
  bool built = false;

  static constexpr uint32_t kMagic = 0x58435241u;  // "ARCX"
  static constexpr uint32_t kVersion = 1;
  static constexpr uint64_t kAlign = 4096;

  // matrix i (0 gate, 1 up, 2 down) of a layer's experts: its tensor and the rows of one expert it takes (Gemma 4
  // keeps gate and up in one tensor, ffn_gate_up_exps: per expert the gate rows, then the up rows)
  struct Src {
    const GgufTensor* t;
    uint64_t row0, rows, per;  // first row and rows of the matrix within an expert's `per` rows of the tensor
  };
  static Src src(const GgufFile& f, const std::string& p, int i) {
    static const char* names[3] = {"gate", "up", "down"};
    if (i < 2 && !f.find(p + "ffn_gate_exps.weight") && f.find(p + "ffn_gate_up_exps.weight")) {
      const GgufTensor& t = f.get(p + "ffn_gate_up_exps.weight");
      if (t.dims.size() < 2) throw std::runtime_error("unexpected tensor shape: " + p + "ffn_gate_up_exps.weight");
      const uint64_t r = t.dims[1] / 2;
      return {&t, i ? r : 0, r, t.dims[1]};
    }
    const GgufTensor& t = f.get(p + "ffn_" + names[i] + "_exps.weight");
    if (t.dims.size() < 2) throw std::runtime_error("unexpected tensor shape: " + p + "ffn_" + names[i] + "_exps.weight");
    return {&t, 0, t.dims[1], t.dims[1]};
  }

  ~ExpertStore() {
    af_.close();
    if (ram_free) for (char* p : ram_pieces_) ram_free(p);
  }

  // Where the engine keeps a GGUF's expert file `file` (2026-11-09; before, every one went loose into zero\, and a
  // deleted GGUF's stayed there unnoticed: a Gemma 4 26B's 12.3 GB):
  //   a GGUF inside the data folder (%USERPROFILE%\.inferzero\models, zero\models): next to it, so deleting the
  //     model's file or folder is where its expert file is too;
  //   a GGUF elsewhere (the Hugging Face cache, another drive, which the engine does not write into):
  //     zero\models\<name>\, a folder of its own, with source.txt naming the GGUF (see note_orphans).
  // settle (a GGUF loaded, not a package or one being made): one written to zero\ before moves to its place (a rename,
  // the same drive) instead of being written again, source.txt is written, and note_orphans runs.
  static std::string place(const std::string& gguf_path, const std::string& name, const std::string& file, bool settle) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string old_path = app_dir() + PLAT_SEP "zero" PLAT_SEP + file;
    auto lower = [](std::string s) { for (char& c : s) c = (char)std::tolower((unsigned char)c); return s; };
    const std::string home = lower(fs::weakly_canonical(app_dir(), ec).string()) + PLAT_SEP;
    std::error_code eg;
    const fs::path gdir = fs::weakly_canonical(fs::path(gguf_path), eg).parent_path();
    const bool inside = !eg && lower(gdir.string()).rfind(home, 0) == 0;
    // (the folder named as zero/gguf_pack.h names a package of this GGUF: a split model's first part without its
    //  "-00001-of-N", so a package made from it later lands in the same folder and takes the file over there)
    std::string folder = name;
    if (const size_t sp = folder.find("-00001-of-"); sp != std::string::npos) folder.resize(sp);
    const fs::path own_dir = inside ? gdir : fs::path(app_dir()) / "zero" / "models" / folder;
    fs::create_directories(own_dir, ec);
    std::string path = (own_dir / file).string();
    if (!settle) return path;
    if (!fs::exists(path, ec) && fs::exists(old_path, ec)) {
      std::error_code e2;
      fs::rename(old_path, path, e2);
      if (!e2) {
        const std::string lru_old = old_path.substr(0, old_path.size() - 8) + ".lru";
        if (fs::exists(lru_old, ec)) fs::rename(lru_old, path.substr(0, path.size() - 8) + ".lru", ec);
        zlog::eprintf("inferzero: the expert file %s moved to %s\n", old_path.c_str(), path.c_str());
      } else {
        path = old_path;  // (could not move it: used where it is)
      }
    }
    if (!inside) {  // (which GGUF this folder's expert file belongs to: a full path, so note_orphans can check it)
      std::error_code e2;
      const fs::path full = fs::absolute(fs::path(gguf_path), e2);
      std::ofstream src_txt(own_dir / "source.txt", std::ios::binary | std::ios::trunc);
      src_txt << (e2 ? gguf_path : full.string()) << "\r\n";
    }
    note_orphans();
    return path;
  }

  // A line in the log for each expert folder under zero\models whose GGUF is gone (its source.txt names a file that no
  // longer exists, on a drive that is there: a GGUF on an unplugged drive is not "gone"), and for expert files still
  // loose in zero\ (made before 2026-11-09; one moves to its folder when its model is loaded): what can be deleted.
  // Nothing is deleted here. A folder that is a package (<folder>\<folder>.gguf) is never named: a package made from
  // a GGUF lands in that GGUF's folder, and the original is usually deleted afterwards.
  static void note_orphans() {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& d : fs::directory_iterator(fs::path(app_dir()) / "zero" / "models", ec)) {
      std::error_code e2;
      if (!d.is_directory(e2)) continue;
      if (fs::exists(d.path() / (d.path().filename().string() + ".gguf"), e2)) continue;  // (a package)
      std::ifstream in(d.path() / "source.txt", std::ios::binary);
      std::string src;
      if (!in || !std::getline(in, src)) continue;
      while (!src.empty() && (src.back() == '\r' || src.back() == '\n')) src.pop_back();
      const fs::path sp(src);
      if (src.empty() || fs::exists(sp, e2) || !fs::exists(sp.root_path(), e2)) continue;
      uint64_t bytes = 0;
      for (const auto& f : fs::directory_iterator(d.path(), e2)) {
        std::error_code e3;
        if (f.is_regular_file(e3)) bytes += f.file_size(e3);
      }
      zlog::eprintf("inferzero: %s (%.1f GB) holds the expert file of %s, which no longer exists: the folder can be deleted\n",
              d.path().string().c_str(), bytes / 1e9, src.c_str());
    }
    for (const auto& f : fs::directory_iterator(fs::path(app_dir()) / "zero", ec)) {
      std::error_code e2;
      if (f.is_regular_file(e2) && f.path().extension() == ".experts")
        zlog::eprintf("inferzero: %s (%.1f GB): an expert file from before 2026-11-09; it moves next to its model when "
                        "that is loaded, and can be deleted if that model is gone\n",
                f.path().string().c_str(), f.file_size(e2) / 1e9);
    }
  }

  // prefixes[i]: the GGUF tensor prefix of layer i ("blk.7.") when it has routed experts, else "".
  // fixed_path: write / use the expert file there (a package being made). A package GGUF ("inferzero.package")
  // has no expert data: its expert file next to it is used as it is (never rebuilt).
  void open_or_build(const GgufFile& f, const std::string& gguf_path, const std::vector<std::string>& prefixes, int n_exp,
                     const std::string& fixed_path = "") {
    n_expert = n_exp;
    const bool package = f.has("inferzero.package");
    namespace fs = std::filesystem;
    std::error_code ec;
    const uint64_t gsize = fs::file_size(gguf_path, ec);
    const auto gtime = fs::last_write_time(gguf_path, ec).time_since_epoch().count();
    const std::string name = fs::path(gguf_path).stem().string();
    uint64_t fp = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) { for (size_t i = 0; i < n; ++i) { fp ^= ((const uint8_t*)p)[i]; fp *= 1099511628211ull; } };
    mix(&gsize, 8);
    mix(&gtime, sizeof(gtime));
    mix(name.data(), name.size());
    mix(&kVersion, 4);
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fp);
    const std::string file = name + "-" + std::string(hex).substr(0, 8) + ".experts";
    const std::string old_path = app_dir() + PLAT_SEP "zero" PLAT_SEP + file;
    path = place(gguf_path, name, file, !package && fixed_path.empty());
    // (where the engine keeps this GGUF's expert file; for a package being made, one still loose in zero\ counts too)
    const std::string def_path = !fixed_path.empty() && !fs::exists(path, ec) && fs::exists(old_path, ec) ? old_path : path;
    if (!fixed_path.empty()) path = fixed_path;
    if (package) {  // (a file name next to the package, never a path the GGUF chooses)
      const std::string xf = f.get_str("inferzero.experts_file");
      if (xf.empty() || xf.find_first_of("/\\:") != std::string::npos || xf == "." || xf == "..")
        throw std::runtime_error("package: inferzero.experts_file must be a file name next to the GGUF");
      path = (fs::path(gguf_path).parent_path() / xf).string();
    }

    // layout
    layers.assign(prefixes.size(), LayerInfo{});
    uint64_t off = kAlign * ((64 + prefixes.size() * 80 + kAlign - 1) / kAlign);  // after the header
    for (size_t l = 0; l < prefixes.size(); ++l) {
      if (prefixes[l].empty()) continue;
      LayerInfo& L = layers[l];
      uint64_t sum = 0;
      for (int i = 0; i < 3; ++i) {
        const Src m = src(f, prefixes[l], i);
        const GgufTensor& t = *m.t;
        L.type[i] = t.type;
        // (a 32-value block type whose rows are not whole superblocks, Flash Next's K = 640 and Gemma 4 26B's 704
        // down rows, is padded with zero blocks to padded_k: quants.h pad_small_block_rows)
        const uint64_t rows = m.rows, nsb = (uint64_t)padded_k((int)t.dims[0]) / 256;
        L.part[i] = rows * nsb * (uint64_t)raw_padded_sb_bytes(t.type);
        sum += L.part[i];
      }
      L.rec = (sum + kAlign - 1) / kAlign * kAlign;
      L.off = off;
      off += L.rec * n_expert;
    }
    file_bytes = off;

    // a package being made from a GGUF whose expert file the engine already wrote: that file moves into the package
    // (a rename on the same drive) instead of being written a second time (Flash Next: 52 GiB), with its .lru state
    if (!fixed_path.empty() && !package && !fs::exists(fixed_path, ec) && fs::exists(def_path, ec)) {
      path = def_path;
      if (valid(fp)) {
        fs::rename(def_path, fixed_path, ec);
        if (!ec) {
          const std::string lru = def_path.substr(0, def_path.size() - 8) + ".lru";
          std::error_code ec2;
          if (fs::exists(lru, ec2)) fs::rename(lru, fixed_path.substr(0, fixed_path.size() - 8) + ".lru", ec2);
          zlog::eprintf("inferzero: the expert file %s moved into the package\n", def_path.c_str());
        }
      }
      path = fixed_path;
    }
    if (!valid(fp, package, package ? &f : nullptr)) {
      if (package)
        throw std::runtime_error("the package's expert file is missing or does not match: " + path +
                                 " (make the package again from the original GGUF: serverz --pack <gguf>)");
      build(f, prefixes, fp);
    }
    for (int i = 0; i < 2 * kQD; ++i) slots_[i].id = i;  // (before the open: no path reads with ids unset)
    const int op = af_.open(path.c_str(), 2 * kQD);  // (unbuffered, reads in flight: src/platform.h AsyncFile)
    if (op == 1) throw std::runtime_error("cannot open " + path);
    if (op == 2) throw std::runtime_error("cannot create the read events");
  }

  // ---- RAM tier (a GPU with memory of its own and a large system RAM: Arc B60 + hundreds of GB): the unbuffered
  // reads never fill the Windows file cache, so records read from the SSD go into RAM pieces the GPU can copy from
  // (pinned host memory from ram_alloc), up to `bytes`, and stay there: a later read of one is no read at all, its
  // Req's src points at it and the caller's GPU copy takes it from there. No CPU copies: a memcpy ran at 3-4 GB/s on
  // the laptop, slower than its SSD. Never on an iGPU, whose GPU memory is that RAM.
  struct Slot { int id = 0; bool busy = false; };  // (a read in flight on af_'s slot id)
  std::function<char*(size_t)> ram_alloc;  // (pinned host memory of n bytes, nullptr when none is left)
  std::function<void(char*)> ram_free;
  void set_ram(uint64_t bytes) {
    ram_cap_ = bytes;
    if (bytes && !ram_rec_) {
      ram_n_ = layers.size() * (size_t)n_expert;
      ram_rec_.reset(new std::atomic<char*>[ram_n_]);
      for (size_t i = 0; i < ram_n_; ++i) ram_rec_[i].store(nullptr, std::memory_order_relaxed);
    }
  }
  uint64_t ram_bytes() const {  // (the prefetch thread adds to it: read under its lock)
    std::lock_guard<std::mutex> g(ram_mu_);
    return ram_used_;
  }
  std::atomic<uint64_t> ram_hits{0};
  // (expert e of layer l's record in the RAM tier, or null: not there, or being read; ARC_XCPU's cache fills)
  const void* ram_src(int l, int e) const {
    if (!ram_rec_ || l < 0 || (size_t)l >= layers.size() || e < 0 || e >= n_expert) return nullptr;
    char* m = ram_rec_[(size_t)l * n_expert + e].load(std::memory_order_acquire);
    return m && m != kPending ? m : nullptr;
  }

  // ---- reads: up to kQD unbuffered reads in flight. dst must be 4 KB aligned (host USM is). set: which of the two
  // sets of read slots (0: the caller's thread, 1: the expert cache's prefetch thread; two threads may read at once).
  // src (out): where the record is now, for the GPU copy: dst, or its place in the RAM tier (to_dst: always dst)
  struct Req { int layer, expert; void* dst; const void* src = nullptr; };
  // keep: records read from the SSD may enter the tier (false: the expert cache's fill at load, ARC_XRAM_FILL=0, so
  // the tier holds what the GPU's cache does not; tier hits are still taken)
  void read(std::vector<Req>& reqs, int set = 0, bool to_dst = false, bool keep = true) {
    Slot* slots = slots_ + (set ? kQD : 0);
    if (!ram_rec_) {
      read_ssd(reqs, slots);
      for (Req& r : reqs) r.src = r.dst;
      return;
    }
    // the RAM tier's records are used where they are; the others read from the SSD, into a new record of the tier
    // while it has room (published once read), else into dst
    std::vector<Req>& todo = ram_todo_[set ? 1 : 0];
    std::vector<size_t>& at = ram_at_[set ? 1 : 0];
    todo.clear(), at.clear();
    for (size_t i = 0; i < reqs.size(); ++i) {
      Req& r = reqs[i];
      std::atomic<char*>& slot = ram_rec_[(size_t)r.layer * n_expert + r.expert];
      char* m = slot.load(std::memory_order_acquire);
      if (m && m != kPending) {
        r.src = m;
        ++ram_hits;
        if (to_dst) memcpy(r.dst, m, layers[r.layer].rec), r.src = r.dst;
        continue;
      }
      char* home = m || !keep ? nullptr : ram_take(slot, layers[r.layer].rec);
      todo.push_back({r.layer, r.expert, home ? home : r.dst});
      at.push_back(i);
    }
    try {
      read_ssd(todo, slots);
    } catch (...) {
      for (const Req& t : todo) {  // (records not read: their reserved slots free again, the memory stays unused)
        std::atomic<char*>& slot = ram_rec_[(size_t)t.layer * n_expert + t.expert];
        char* exp = kPending;
        if (t.dst != reqs[at[&t - todo.data()]].dst) slot.compare_exchange_strong(exp, nullptr);
      }
      throw;
    }
    for (size_t j = 0; j < todo.size(); ++j) {
      Req& r = reqs[at[j]];
      if (todo[j].dst != r.dst) {  // (read into the tier: published)
        ram_rec_[(size_t)r.layer * n_expert + r.expert].store((char*)todo[j].dst, std::memory_order_release);
        if (to_dst) memcpy(r.dst, todo[j].dst, layers[r.layer].rec);
        else r.src = todo[j].dst;
      }
      if (!r.src) r.src = r.dst;
    }
    for (Req& r : reqs) if (to_dst || !r.src) r.src = r.dst;
  }

 private:
  static inline char* const kPending = (char*)1;  // (a record being read into the tier by a read() call)
  // a new record of `rec` bytes in the tier for `slot` (marked pending), or nullptr: full, or another call has it
  char* ram_take(std::atomic<char*>& slot, uint64_t rec) {
    std::lock_guard<std::mutex> g(ram_mu_);
    if (slot.load(std::memory_order_relaxed) || ram_used_ + rec > ram_cap_) return nullptr;
    if (ram_left_ < rec) {  // (pieces of 1 GiB)
      const size_t piece = std::max<size_t>(rec, 1ull << 30);
      char* p = ram_alloc ? ram_alloc(piece) : nullptr;
      if (!p) { ram_cap_ = ram_used_; return nullptr; }  // (no more pinned memory: the tier stays as it is)
      ram_pieces_.push_back(p);
      ram_next_ = p, ram_left_ = piece;
    }
    char* r = ram_next_;
    ram_next_ += rec, ram_left_ -= rec, ram_used_ += rec;
    slot.store(kPending, std::memory_order_relaxed);
    return r;
  }
  std::unique_ptr<std::atomic<char*>[]> ram_rec_;  // (layer * n_expert + expert -> its record in RAM, kPending, null)
  size_t ram_n_ = 0;
  uint64_t ram_cap_ = 0, ram_used_ = 0, ram_left_ = 0;
  char* ram_next_ = nullptr;
  std::vector<char*> ram_pieces_;
  mutable std::mutex ram_mu_;
  std::vector<Req> ram_todo_[2];
  std::vector<size_t> ram_at_[2];

  void read_ssd(const std::vector<Req>& reqs, Slot* slots) {
    if (reqs.empty()) return;
    size_t next = 0, done = 0;
    int busy = 0;
    auto issue = [&](int s) {
      const Req& r = reqs[next++];
      const LayerInfo& L = layers[r.layer];
      const uint64_t o = L.off + (uint64_t)r.expert * L.rec;
      Slot& sl = slots[s];
      if (!af_.start(sl.id, r.dst, (uint32_t)L.rec, o))
        throw std::runtime_error("expert read failed (" + std::to_string(af_.error(sl.id)) + ")");
      sl.busy = true;
      ++busy;
      bytes_read += L.rec;
    };
    try {
      for (int s = 0; s < kQD && next < reqs.size(); ++s) issue(s);
      while (done < reqs.size()) {
        int ids[kQD], map[kQD], n = 0;
        for (int s = 0; s < kQD; ++s) if (slots[s].busy) ids[n] = slots[s].id, map[n++] = s;
        const int w = af_.wait_any(ids, n);
        if (w < 0) throw std::runtime_error("expert read: wait failed (" + std::to_string(af_.error()) + ")");
        const int s = map[w];
        if (!af_.finish(slots[s].id)) throw std::runtime_error("expert read failed");
        slots[s].busy = false;
        --busy;
        ++done;
        g_expert_reads_done.fetch_add(1, std::memory_order_relaxed);  // (a sign of life for a split model's waits)
        if (next < reqs.size()) issue(s);
      }
    } catch (...) {
      // (a failed read: the reads still in flight are cancelled and waited for, so no late read writes into the
      // staging and the next read() finds its slots free)
      for (int s = 0; s < kQD; ++s) {
        if (!slots[s].busy) continue;
        af_.cancel(slots[s].id);
        slots[s].busy = false;
      }
      throw;
    }
    reads += reqs.size();
  }

 public:
  std::atomic<uint64_t> reads{0}, bytes_read{0};

 private:
  static constexpr int kQD = 16;
  Slot slots_[2 * kQD] = {};
  plat::AsyncFile af_;  // (the expert file, unbuffered; slots_[i] reads on its slot i)

  struct Header {
    uint32_t magic, version;
    uint64_t fingerprint, layers, experts, bytes;
  };
  bool valid(uint64_t fp, bool any_fp = false, const GgufFile* package = nullptr) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) != file_bytes) return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    Header h{};
    const bool ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == kMagic && h.version == kVersion && (any_fp || h.fingerprint == fp) &&
                    h.layers == layers.size() && h.experts == (uint64_t)n_expert && h.bytes == file_bytes;
    bool same = ok;
    for (size_t l = 0; same && l < layers.size(); ++l) {
      LayerInfo li{};
      same = fread(&li, sizeof li, 1, f) == 1 && li == layers[l];
    }
    fclose(f);
    if (ok && package) check_autoround_expert_revision(*package, h.fingerprint);
    return same;
  }
  void build(const GgufFile& f, const std::vector<std::string>& prefixes, uint64_t fp) {
    XBuildStopGuard stop_guard;  // (before the .tmp's owner: it unwinds after the .tmp is removed; D1)
    const auto t0 = std::chrono::steady_clock::now();
    zlog::eprintf("inferzero: writing the expert file %s (%.1f GB, once per model)\n", path.c_str(), file_bytes / 1e9);
    const std::string tmp = path + ".tmp";
    plat::KeepAwake awake;  // (S11: a laptop left alone slept mid-build)
    const plat::BuildLock lock(tmp + ".lock");  // (A9: held until the .tmp is renamed or removed below)
    drop_stale_tmp(tmp, lock);                  // (S7)
    need_free_space(path, file_bytes, "the expert file");  // (S3: before anything is written)
    // unbuffered writes (4 KB multiples from page-aligned buffers): 15 GB through the Windows file cache filled the
    // RAM (3.9 GB left, paging) on the 32 GB laptop
    // (the .tmp file goes away if the build fails half way: it is as large as the expert file)
    // (src/platform.h DirectWriter: unbuffered, write-through; Linux O_DIRECT)
    struct Tmp {
      plat::DirectWriter w;
      std::string path;
      bool done = false;
      ~Tmp() {
        w.close();
        if (!done) plat::remove_file(path.c_str());
      }
    } out{{}, tmp};
    if (!out.w.open(tmp.c_str())) {
      out.done = true;
      throw std::runtime_error("cannot write " + tmp);
    }
    auto put = [&](const void* p, uint64_t n) {
      if (!out.w.write(p, n)) throw std::runtime_error("write failed: " + tmp);
    };
    // header (+ layer table), padded to the first layer's offset
    {
      uint64_t first = kAlign;
      for (auto& L : layers) if (L.rec) { first = L.off; break; }
      uint8_t* hb = (uint8_t*)plat::page_alloc(first);  // (zeroed: the header's padding)
      if (!hb) throw std::runtime_error("out of memory writing the expert file");
      Header h{kMagic, kVersion, fp, layers.size(), (uint64_t)n_expert, file_bytes};
      memcpy(hb, &h, sizeof h);
      memcpy(hb + sizeof h, layers.data(), layers.size() * sizeof(LayerInfo));
      struct Free { void* p; size_t n; ~Free() { plat::page_free(p, n); } } free_hb{hb, (size_t)first};
      put(hb, first);
    }
    const int G = 32;  // experts converted per batch (in parallel)
    size_t n_done = 0, n_all = 0;  // (layers with experts: the progress)
    for (const std::string& p : prefixes) n_all += !p.empty();
    for (size_t l = 0; l < prefixes.size(); ++l) {
      if (prefixes[l].empty()) continue;
      const LayerInfo& L = layers[l];
      Src m[3];
      const GgufTensor* t[3];
      for (int i = 0; i < 3; ++i) {
        m[i] = src(f, prefixes[l], i);
        t[i] = m[i].t;
      }
      struct Buf { uint8_t* p; size_t n; ~Buf() { plat::page_free(p, n); } } bb{(uint8_t*)plat::page_alloc((size_t)L.rec * G), (size_t)L.rec * G};
      uint8_t* buf = bb.p;
      if (!buf) throw std::runtime_error("out of memory writing the expert file");
      // ---- (2026-11-19) a batch's source rows read in order before the threads convert them. A batch's G experts
      //      are one contiguous range of rows in each expert tensor (gate, up, down; Gemma 4's gate_up holds both
      //      halves of each expert), so the source is read as up to three large ranges a batch: big sequential
      //      reads, where the threads reading the mapping themselves made ~3 x G small scattered ones (an external
      //      USB disk ran at ~17 MB/s, and hung under them). The next batch's ranges are prefetched while this one
      //      converts. The same bytes go through the same conversion: the expert file is the same, bit for bit.
      //      RAM: one batch of source rows (~85 MB for Flash Next IQ4_XS). ARC_XBUILD_POOL=0: as before.
      static const bool pool_on = !getenv("ARC_XBUILD_POOL") || atoi(getenv("ARC_XBUILD_POOL")) != 0;
      struct Span { const GgufTensor* t = nullptr; uint64_t lo_off = 0, rowb = 0; size_t at = 0; };
      Span sp[3];       // the distinct source tensors of this layer (one for gate and up when they share one)
      int sp_of[3] = {0, 0, 0}, n_sp = 0;
      for (int i = 0; i < 3; ++i) {
        int k = 0;
        while (k < n_sp && sp[k].t != t[i]) ++k;
        if (k == n_sp) {
          sp[k].t = t[i];
          const uint32_t ty = L.type[i];
          const uint64_t K = t[i]->dims[0];
          int blk = 0, bytes = 0;
          if (K % 256) ggml_block_info(ty, blk, bytes);
          sp[k].rowb = K % 256 ? (K / blk) * bytes : (K / 256) * gguf_sb_bytes(ty);  // (the offsets used below)
          ++n_sp;
        }
        sp_of[i] = k;
      }
      // the rows [lo, hi) of tensor k a batch of experts e0 .. e0 + ne - 1 reads (each i using it: row0 .. row0 + rows of
      // its experts' `per` rows)
      auto span_rows = [&](int k, int e0, int ne, uint64_t& lo, uint64_t& hi) {
        lo = UINT64_MAX, hi = 0;
        for (int i = 0; i < 3; ++i)
          if (sp_of[i] == k) {
            lo = std::min(lo, (uint64_t)e0 * m[i].per + m[i].row0);
            hi = std::max(hi, (uint64_t)(e0 + ne - 1) * m[i].per + m[i].row0 + m[i].rows);
          }
      };
      auto prefetch_batch = [&](int e0) {
        if (e0 >= n_expert) return;
        const int ne = std::min(G, n_expert - e0);
        for (int k = 0; k < n_sp; ++k) {
          uint64_t lo, hi;
          span_rows(k, e0, ne, lo, hi);
          plat::prefetch_mapped(sp[k].t->data + lo * sp[k].rowb, (size_t)((hi - lo) * sp[k].rowb));
        }
      };
      size_t pool_n = 0;
      if (pool_on)
        for (int k = 0; k < n_sp; ++k) {
          uint64_t lo, hi;
          span_rows(k, 0, std::min(G, n_expert), lo, hi);
          sp[k].at = pool_n;
          pool_n += (size_t)((hi - lo) * sp[k].rowb + 4095) / 4096 * 4096;
        }
      struct Pool { uint8_t* p; size_t n; ~Pool() { plat::page_free(p, n); } } pb{
          pool_n ? (uint8_t*)plat::page_alloc(pool_n) : nullptr, pool_n};
      // (no room for the batch copy: the threads read the mapping as before, slower but the same bytes)
      const bool pool = pool_on && pb.p;
      if (pool_on && !pb.p) zlog::eprintf("inferzero: no %.0f MB for the expert file's read buffer: reading in place\n", pool_n / 1e6);
      if (pool) prefetch_batch(0);
      for (int e0 = 0; e0 < n_expert; e0 += G) {
        XBuildStopGuard::check("the expert file " + path);
        const int ne = std::min(G, n_expert - e0);
        memset(buf, 0, (size_t)ne * L.rec);
        // where matrix i's rows are read from: the mapping from row 0, or this batch's copy from row lo_row[i]
        const uint8_t* base[3] = {t[0]->data, t[1]->data, t[2]->data};
        uint64_t lo_row[3] = {0, 0, 0};
        if (pool) {
          for (int k = 0; k < n_sp; ++k) {
            uint64_t lo, hi;
            span_rows(k, e0, ne, lo, hi);
            memcpy(pb.p + sp[k].at, sp[k].t->data + lo * sp[k].rowb, (size_t)((hi - lo) * sp[k].rowb));
            // (copied: on Linux those pages leave the process now, not at the end of the pass; B6)
            plat::release_mapped(sp[k].t->data + lo * sp[k].rowb, (size_t)((hi - lo) * sp[k].rowb));
            for (int i = 0; i < 3; ++i)
              if (sp_of[i] == k) base[i] = pb.p + sp[k].at, lo_row[i] = lo;
          }
          prefetch_batch(e0 + G);  // (the next batch's reads go on while this one converts)
        }
#pragma omp parallel for
        for (int j = 0; j < ne; ++j) {
          uint8_t* dst = buf + (size_t)j * L.rec;
          for (int i = 0; i < 3; ++i) {
            const uint32_t ty = L.type[i];
            const size_t sbb = gguf_sb_bytes(ty), bp = raw_padded_sb_bytes(ty);
            const size_t nsb = L.part[i] / bp;  // superblocks of one expert matrix
            const int K = (int)t[i]->dims[0], rows = (int)m[i].rows;
            // the expert's first row of this matrix, counted from the first row `base[i]` holds
            const size_t row1 = (size_t)(e0 + j) * m[i].per + m[i].row0 - lo_row[i];
            const uint8_t* src;
            std::vector<uint8_t> padded;
            if (K % 256) {  // rows padded to whole superblocks first
              int blk = 0, bytes = 0;
              ggml_block_info(ty, blk, bytes);
              padded.resize(nsb * sbb);
              pad_small_block_rows(ty, base[i] + row1 * (K / blk) * bytes, rows, K, padded.data());
              src = padded.data();
            } else {
              src = base[i] + row1 * (K / 256) * sbb;
            }
            for (size_t s = 0; s < nsb; ++s) gpu_raw_sb(ty, src + s * sbb, dst + s * bp);
            dst += L.part[i];
          }
        }
        put(buf, (uint64_t)ne * L.rec);
      }
      // the layer's GGUF pages leave the working set (they stay in the standby list, which Windows can reuse)
      plat::trim_working_set();
      ++n_done;
      if (loadp::st().on) loadp::step("writing the expert file (once per model)", (double)n_done, (double)n_all);
      else zlog::eprintf("\r  layer %zu / %zu", n_done, n_all);  // (the command-line tools: their own line)
    }
    if (!loadp::st().on) zlog::eprintf("\n");
    out.w.close();
    if (!plat::replace_file(tmp.c_str(), path.c_str()))  // (an old file of the same name: replaced)
      throw std::runtime_error("cannot rename " + tmp + " (" + std::to_string(plat::last_error()) + ")");
    out.done = true;
    build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    built = true;
  }
};
