// The server's log file by default (2026-11-23, the user: "It should log by default, if not how we spot errors?"; a
// B580 under Linux lost its GPU and its owner found no log: there serverz wrote to stderr only). The same files on
// both systems, as zero-server.ps1 has kept them on Windows (docs/logging.md):
//   <the logs folder>/server.log                       this run's, written anew at each start
//   <the logs folder>/server_<YYYY-MM-DD_HH-MM-SS>.log  earlier runs', named by their last line's local time
//                                                      (" (2)" etc. for the same second); the newest 10
//   <the logs folder>/server_<...>.zip                  older ones, each in a zip of the same name with its time;
//                                                      zips over 90 days old or past 200 MB together removed,
//                                                      oldest first; never more than 30 .log where zipping fails
// The logs folder: --log-dir D, else ARC_LOG_DIR, else <the data folder>/logs (%USERPROFILE%\.inferzero\logs,
// ~/.inferzero/logs; ARC_DATA_DIR moves it with the rest). Created if missing; one that cannot be written: a W line
// and the default folder; the default one too: a W line and no file, the server runs on.
// One owner of the rotation per start: where serverz's stderr already is that server.log (zero-server.ps1's
// redirect, which rotated before it started serverz), serverz neither rotates nor opens it again.
#pragma once
#include "platform.h"
#include "app_dir.h"
#include "gpu_fail.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace slog {
namespace fs = std::filesystem;

inline fs::path default_dir() { return fs::path(app_dir()) / "logs"; }

// d made and a file creatable in it
inline bool usable_dir(const fs::path& d) {
  std::error_code ec;
  fs::create_directories(d, ec);
  if (!fs::is_directory(d, ec)) return false;
  const fs::path probe = d / ".write_test";
  FILE* f = plat::fopen_path(probe, "wb");
  if (!f) return false;
  fclose(f);
  fs::remove(probe, ec);
  return true;
}

// a file's last write as local time, yyyy-MM-dd_HH-mm-ss (zero-server.ps1's stamp); its time_t in *t
inline std::string stamp_of(const fs::path& f, std::time_t* t = nullptr) {
  std::error_code ec;
  const auto ft = fs::last_write_time(f, ec);
  std::time_t tt = std::time(nullptr);
  if (!ec)  // (the file clock shifted by the two clocks' difference now: MSVC has no file_clock::to_sys, g++ 13 no clock_cast)
    tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() +
                                              std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                                  ft - std::filesystem::file_time_type::clock::now()));
  if (t) *t = tt;
  std::tm tm{};
  if (!plat::local_time(tt, tm)) return "0000-00-00_00-00-00";
  char b[32];
  strftime(b, sizeof b, "%Y-%m-%d_%H-%M-%S", &tm);
  return b;
}

inline uint32_t crc32(const unsigned char* p, size_t n) {
  static uint32_t tab[256];
  static bool made = false;
  if (!made) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      tab[i] = c;
    }
    made = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) c = tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// src in a zip of its own (one entry, stored: no compression), written to a temporary file and then put in place,
// with src's time; false (and nothing left behind) on failure
inline bool zip_one(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  const uintmax_t sz = fs::file_size(src, ec);
  if (ec || sz >= 0xFFFFFFFFull) return false;  // (no zip64: such a log stays a .log)
  std::vector<unsigned char> data((size_t)sz);
  {
    FILE* in = plat::fopen_path(src, "rb");
    if (!in) return false;
    const size_t got = data.empty() ? 0 : fread(data.data(), 1, data.size(), in);
    fclose(in);
    if (got != data.size()) return false;
  }
  std::time_t t = 0;
  stamp_of(src, &t);
  std::tm tm{};
  plat::local_time(t, tm);
  const uint16_t dtime = (uint16_t)((tm.tm_hour << 11) | (tm.tm_min << 5) | (tm.tm_sec / 2));
  const uint16_t ddate = (uint16_t)((std::max(0, tm.tm_year + 1900 - 1980) << 9) | ((tm.tm_mon + 1) << 5) | tm.tm_mday);
  const uint32_t crc = crc32(data.data(), data.size());
  const std::string name = src.filename().string();
  std::vector<unsigned char> o;
  auto u16 = [&](uint32_t v) { o.push_back((unsigned char)(v & 0xFF)), o.push_back((unsigned char)((v >> 8) & 0xFF)); };
  auto u32 = [&](uint32_t v) { u16(v & 0xFFFF), u16(v >> 16); };
  auto header = [&](bool central, uint32_t off) {
    u32(central ? 0x02014b50u : 0x04034b50u);
    if (central) u16(20);   // (made by)
    u16(20), u16(0), u16(0);  // (version needed, flags, method: stored)
    u16(dtime), u16(ddate), u32(crc), u32((uint32_t)sz), u32((uint32_t)sz), u16((uint32_t)name.size()), u16(0);
    if (central) u16(0), u16(0), u16(0), u32(0), u32(off);  // (comment, disk, internal / external attributes, offset)
    o.insert(o.end(), name.begin(), name.end());
  };
  header(false, 0);
  const size_t data_at = o.size();
  const size_t cd_at = data_at + data.size();
  std::vector<unsigned char> head(o);
  o.clear();
  header(true, 0);
  const size_t cd_size = o.size();
  u32(0x06054b50u), u16(0), u16(0), u16(1), u16(1), u32((uint32_t)cd_size), u32((uint32_t)cd_at), u16(0);
  const fs::path tmp = dst.string() + ".tmp";
  FILE* out = plat::fopen_path(tmp, "wb");
  if (!out) return false;
  bool ok = fwrite(head.data(), 1, head.size(), out) == head.size() &&
            (data.empty() || fwrite(data.data(), 1, data.size(), out) == data.size()) &&
            fwrite(o.data(), 1, o.size(), out) == o.size();
  ok = fclose(out) == 0 && ok;
  if (ok) {
    fs::rename(tmp, dst, ec);
    ok = !ec;
  }
  if (!ok) {
    fs::remove(tmp, ec);
    return false;
  }
  const auto ft = fs::last_write_time(src, ec);
  if (!ec) fs::last_write_time(dst, ft, ec);
  return true;
}

// an earlier run's log moved into dir as server_<its time>.log; false if it could not be moved
inline bool keep_earlier(const fs::path& dir, const fs::path& log) {
  std::error_code ec;
  const std::string st = stamp_of(log);
  fs::path dest = dir / ("server_" + st + ".log");
  for (int k = 2; fs::exists(dest, ec); ++k) dest = dir / ("server_" + st + " (" + std::to_string(k) + ").log");
  fs::rename(log, dest, ec);
  if (ec) {  // (another drive: a copy, then the original removed)
    ec.clear();
    fs::copy_file(log, dest, ec);
    if (ec) return false;
    const auto ft = fs::last_write_time(log, ec);
    if (!ec) fs::last_write_time(dest, ft, ec);
    fs::remove(log, ec);
  }
  gpufail::marker_log_moved(log.string(), dest.string());  // (a GPU fault's marker names the log of its run)
  return true;
}

// the newest 10 earlier logs as they are, older ones zipped; at most 30 .log; zips past 90 days or 200 MB removed
inline void prune(const fs::path& dir) {
  std::error_code ec;
  struct F { fs::path p; fs::file_time_type t; uintmax_t n; };
  auto list = [&](const char* ext) {
    std::vector<F> v;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      const std::string n = e.path().filename().string();
      if (n.rfind("server_", 0) != 0 || e.path().extension() != ext) continue;
      std::error_code e2;
      v.push_back({e.path(), fs::last_write_time(e.path(), e2), fs::file_size(e.path(), e2)});
    }
    std::sort(v.begin(), v.end(), [](const F& a, const F& b) { return a.t > b.t; });  // (newest first)
    return v;
  };
  std::vector<F> logs = list(".log");
  for (size_t i = 10; i < logs.size(); ++i) {
    const fs::path z = logs[i].p.parent_path() / (logs[i].p.stem().string() + ".zip");
    if (zip_one(logs[i].p, z)) fs::remove(logs[i].p, ec);
  }
  logs = list(".log");
  for (size_t i = 30; i < logs.size(); ++i) fs::remove(logs[i].p, ec);
  const auto old = fs::file_time_type::clock::now() - std::chrono::hours(24 * 90);
  uintmax_t sum = 0;
  for (const F& z : list(".zip")) {
    sum += z.n;
    if (z.t < old || sum > 200ull * 1024 * 1024) fs::remove(z.p, ec);
  }
}

// The server's log at its start (see the top): cli_dir is --log-dir's value or "". Sets zlog's dfile and its path.
inline void open_default(const std::string& cli_dir) {
  zlog::Config& c = zlog::cfg();
  if (c.disabled) return;
  fs::path dir;
  const char* env = getenv("ARC_LOG_DIR");
  const std::string asked = !cli_dir.empty() ? cli_dir : env && *env ? std::string(env) : std::string();
  if (!asked.empty()) {
    if (usable_dir(asked)) dir = asked;
    else LOG_WRN("inferzero: the log folder %s cannot be written: the default one\n", asked.c_str());
  }
  if (dir.empty()) {
    dir = default_dir();
    if (!usable_dir(dir)) {
      LOG_WRN("inferzero: the log folder %s cannot be written: no log file (the log on stderr only)\n", dir.string().c_str());
      return;
    }
  }
  const fs::path log = dir / "server.log";
  snprintf(c.dfile_path, sizeof c.dfile_path, "%s", log.string().c_str());
  // (stderr already that file: zero-server.ps1 rotated it and redirects every line there)
  if (FILE* f = plat::fopen_path(log, "rb")) {
    const bool same = plat::same_file(stderr, f);
    fclose(f);
    if (same) return;
  }
  std::error_code ec;
  // (the place before 2026-11-23: the data folder's top, zero-server.ps1's server.log there; an earlier run now)
  const fs::path top = fs::path(app_dir()) / "server.log";
  if (fs::exists(top, ec) && !fs::equivalent(top, log, ec)) {
    bool same = false;
    if (FILE* f = plat::fopen_path(top, "rb")) {
      same = plat::same_file(stderr, f);
      fclose(f);
    }
    if (!same) keep_earlier(dir, top);
  }
  if (fs::exists(log, ec) && !keep_earlier(dir, log))
    LOG_WRN("inferzero: the last run's log %s could not be kept (in use?): written anew\n", log.string().c_str());
  prune(dir);
  c.dfile = plat::fopen_path(log, "wb");
  if (!c.dfile) {
    LOG_WRN("inferzero: cannot open the log file %s: the log on stderr only\n", log.string().c_str());
    c.dfile_path[0] = 0;
    return;
  }
  LOG_INF("inferzero: log file %s (earlier runs' next to it)\n", log.string().c_str());
}
}  // namespace slog
