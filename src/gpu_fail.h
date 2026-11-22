// The GPU failsafe (docs/gpu-failsafe.md, 2026-11-23, after a hung kernel froze the development laptop's screen
// twice): the one way out after a GPU fault, and the hang marker that keeps one hang to one.
//
// - gpu_fatal(kind, detail): one error line (the kind, the detail, the last kernel launched, the program), the marker
//   for a hang or a lost device, then _Exit with the kind's code: 3 a hang (a wait past ARC_GPU_WAIT_S), 4 a lost
//   device, 5 another Level Zero failure mid-work (1 stays an error as before). No GPU call after the fault: the
//   driver's cleanup can hang, so no exit(), no destructors, no diagnostics that ask the GPU.
// - The marker, %USERPROFILE%\.inferzero\gpu_hang.txt (~/.inferzero on Linux; the data folder, ARC_DATA_DIR when
//   set): key=value lines with this boot's id. Every program on clrt.h checks it before it starts the GPU and
//   refuses with code 3 while it is from this boot; a marker from an older boot is deleted. runz --clear-hang deletes
//   it, ARC_IGNORE_HANG=1 runs anyway. A marker that cannot be read is ignored with a warning, never taken for a hang.
#pragma once
#include "platform.h"
#include "app_dir.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace gpufail {
enum Kind { Hang = 3, Lost = 4, Other = 5 };
inline const char* kind_name(Kind k) { return k == Hang ? "hang" : k == Lost ? "lost" : "error"; }

// the name of the last kernel launched in this process (any thread, any GPU: one store a launch, no copy; cleared
// when that kernel is released, so it never points at a freed name)
inline std::atomic<const char*> last_kernel{nullptr};

inline std::string marker_path() { return app_dir() + PLAT_SEP "gpu_hang.txt"; }

// this program's file name (for the marker and the messages; no GPU)
inline std::string program() {
  std::string p;
#ifdef _WIN32
  char b[MAX_PATH];
  const DWORD n = GetModuleFileNameA(nullptr, b, MAX_PATH);
  if (n > 0 && n < MAX_PATH) p.assign(b, n);
#else
  char b[4096];
  const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
  if (n > 0) p.assign(b, (size_t)n);
#endif
  const size_t s = p.find_last_of("/\\");
  if (s != std::string::npos) p = p.substr(s + 1);
  if (p.size() > 4 && (p.compare(p.size() - 4, 4, ".exe") == 0 || p.compare(p.size() - 4, 4, ".EXE") == 0)) p.resize(p.size() - 4);
  return p.empty() ? "inferzero" : p;
}

// this boot's id: Linux's boot_id; on Windows the boot time in seconds since 1970 (now minus the time since boot),
// the same boot within 2 minutes
inline std::string boot_id() {
#ifdef _WIN32
  return std::to_string((long long)std::time(nullptr) - (long long)(GetTickCount64() / 1000));
#else
  std::ifstream f("/proc/sys/kernel/random/boot_id");
  std::string s;
  std::getline(f, s);
  return s;
#endif
}
inline bool same_boot(const std::string& a, const std::string& b) {
  if (a.empty() || b.empty()) return false;
#ifdef _WIN32
  char* e1 = nullptr;
  char* e2 = nullptr;
  const long long x = strtoll(a.c_str(), &e1, 10), y = strtoll(b.c_str(), &e2, 10);
  if (*e1 || *e2) return false;
  return (x > y ? x - y : y - x) <= 120;
#else
  return a == b;
#endif
}

// the marker's time= (YYYY-MM-DDTHH:MM:SSZ) in seconds since 1970; -1 if it is not one
inline long long parse_utc(const std::string& s) {
  std::tm t{};
  if (sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2dZ", &t.tm_year, &t.tm_mon, &t.tm_mday, &t.tm_hour, &t.tm_min, &t.tm_sec) != 6)
    return -1;
  t.tm_year -= 1900;
  t.tm_mon -= 1;
#ifdef _WIN32
  return (long long)_mkgmtime(&t);
#else
  return (long long)timegm(&t);
#endif
}
// (a marker older than this counts as stale: Windows' Fast Startup keeps the time since boot over a "Shut down",
//  so only a Restart would clear the marker otherwise)
constexpr long long kStaleS = 12 * 3600;

inline std::string utc_now() {
  std::tm t{};
  if (!plat::utc_time(std::time(nullptr), t)) return "";
  char b[32];
  strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &t);
  return b;
}

// the GPU's name, set when the device is opened (for the marker; never asked of the GPU after a fault)
inline char gpu_name[256] = "";

// the marker written atomically (a temporary file, then a rename); false when it could not be written
inline bool write_marker(Kind k, const char* kernel, const char* extra = nullptr) {
  const std::string p = marker_path(), tmp = p + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  fprintf(f, "time=%s\nprogram=%s\nkernel=%s\nkind=%s\ngpu=%s\nboot=%s\n", utc_now().c_str(), program().c_str(),
          kernel && *kernel ? kernel : "(none)", kind_name(k), gpu_name, boot_id().c_str());
  if (extra) fprintf(f, "%s\n", extra);
  const bool ok = fclose(f) == 0;
  if (!ok || !plat::replace_file(tmp.c_str(), p.c_str())) {
    plat::remove_file(tmp.c_str());
    return false;
  }
  return true;
}

// the marker's lines; false when there is none or it cannot be read
inline bool read_marker(std::map<std::string, std::string>& kv) {
  std::ifstream f(marker_path(), std::ios::binary);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t e = line.find('=');
    if (e != std::string::npos) kv[line.substr(0, e)] = line.substr(e + 1);
  }
  return true;
}

// deletes the marker; true when there was one
inline bool clear_marker() {
  std::ifstream f(marker_path());
  if (!f) return false;
  f.close();
  plat::remove_file(marker_path().c_str());
  return true;
}

// Before a program starts the GPU: a marker from this boot ends it with code 3 (ARC_IGNORE_HANG=1: a warning only);
// one from an older boot is deleted with an info line; one that cannot be read is ignored with a warning
inline void check_marker() {
  std::map<std::string, std::string> kv;
  if (!read_marker(kv)) return;
  const std::string path = marker_path();
  if (!kv.count("boot") || !kv.count("kind")) {
    LOG_WRN("inferzero: %s cannot be read as a GPU hang marker: ignored (delete it to drop this warning)\n", path.c_str());
    return;
  }
  if (!same_boot(kv["boot"], boot_id())) {
    plat::remove_file(path.c_str());
    LOG_INF("inferzero: a GPU %s marker from an earlier start of the computer (%s, %s in %s) deleted\n", kv["kind"].c_str(),
            kv["time"].c_str(), kv["kernel"].c_str(), kv["program"].c_str());
    return;
  }
  const long long at = parse_utc(kv["time"]);
  if (at > 0 && (long long)std::time(nullptr) - at > kStaleS) {
    plat::remove_file(path.c_str());
    LOG_INF("inferzero: a GPU %s marker more than 12 hours old (%s, %s in %s) deleted\n", kv["kind"].c_str(),
            kv["time"].c_str(), kv["kernel"].c_str(), kv["program"].c_str());
    return;
  }
  std::string when = kv["time"];
  if (when.size() >= 16) when = when.substr(11, 5) + " UTC";
  const char* ign = getenv("ARC_IGNORE_HANG");
  if (ign && atoi(ign) != 0) {
    LOG_WRN("inferzero: a GPU %s at %s in %s (%s) since this computer started; ARC_IGNORE_HANG is set: running anyway\n",
            kv["kind"].c_str(), when.c_str(), kv["kernel"].c_str(), kv["program"].c_str());
    return;
  }
  LOG_ERR("inferzero: a GPU %s at %s in %s (%s): restart the computer (Restart, not Shut down) before using the GPU again (or delete %s, run "
          "runz --clear-hang, or set ARC_IGNORE_HANG=1)\n",
          kv["kind"].c_str(), when.c_str(), kv["kernel"].c_str(), kv["program"].c_str(), path.c_str());
  fflush(stderr);
  _Exit(Hang);
}

// The one way out after a GPU fault (see the top). detail: what was being waited for or done
[[noreturn]] inline void gpu_fatal(Kind k, const std::string& detail) {
  const char* kn = last_kernel.load();
  const std::string prog = program();
  LOG_ERR("inferzero: GPU %s: %s (the last kernel launched: %s, in %s)\n", kind_name(k), detail.c_str(), kn ? kn : "none",
          prog.c_str());
  if (k == Hang || k == Lost) {
    if (write_marker(k, kn))
      LOG_ERR("inferzero: no GPU work starts again until the computer restarts (marker %s; runz --clear-hang deletes it)\n",
              marker_path().c_str());
    else
      LOG_ERR("inferzero: the hang marker %s could not be written\n", marker_path().c_str());
  }
  fflush(stdout);
  fflush(stderr);
  _Exit(k);
}

// A Level Zero result that ends the process: a lost device (code 4) or another failure (code 5)
[[noreturn]] inline void ze_exit(uint32_t r, const std::string& detail) {
  gpu_fatal(r == 0x70000001u ? Lost : Other, detail);
}

// ARC_GPU_FAULT_TEST=hang / lost (tests of this path without a real fault): the first GPU wait acts as if the limit
// passed, or as if the device was lost. 0: off, else the Kind
inline int fault_test() {
  static const int t = [] {
    const char* v = getenv("ARC_GPU_FAULT_TEST");
    if (!v) return 0;
    const std::string s = v;
    return s == "hang" ? (int)Hang : s == "lost" ? (int)Lost : 0;
  }();
  return t;
}
}  // namespace gpufail
