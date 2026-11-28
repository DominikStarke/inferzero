// gemm_check: today's GEMM and GEMV forms against CPU references, at random sizes and at the shapes that break kernels
// (docs/design-shared-gemm.md 4: the shared GEMM layer's checks; Phase 0, the baseline of today's kernels). The checks
// are zero/gemm_check.h, a part of class Engine: synthetic weights of every GGUF type the engine takes, made through
// make_weight as a model's are, the engine's own quantization and launches, the reference from what each form read.
//   gemm_check --model M.gguf [--out results.tsv] [--quick] [--seed N] [--types Q4_K,IQ3_S] [--reps N] [--offsets]
// --offsets: the types with a min (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K) with a random min a block instead of one fitted to its
// scale, so the blocks' means are far from 0 (a stress for H8R's rotation; the end_to_end column shows it).
// end_to_end: the output against dequant_row's weights times the unquantized activations (the quantization's error in
// it: reported, not failed); e2e_vs_gemv: a prompt form's against its type and shape's one-token GEMV, noted above 8x.
// The model gives the engine its buffers and its start checks (its own weights are not checked): its rows bound the
// shapes (a case past them says so in the file, "skipped"). Run it through the GPU watchdog:
//   powershell -File bench\gpu_guard.ps1 -TimeoutS 1800 -- bin\gemm_check.exe --model M.gguf --out %TEMP%\gc.tsv
// Exit 1 when a case fails its bound (relative error 1e-3) or errors; DIFFERS in the det / pair columns is recorded,
// not failed (Phase 0 records today's behaviour).
#include "../src/full_speed.h"
#include "../zero/engine.h"
#include <cstdio>
#include <string>

// --weights M.gguf: no GPU. Each matrix's rows (up to 128 a tensor, spread over it; the experts' too) decoded by
// dequant_row, and the two statistics H8R's error depends on (Engine::gc_row_stats): a row's |mean| / deviation over K,
// and its largest rotation block's |mean| x sqrt(block) / deviation (the coefficient its one scale must cover after
// the rotation). Per tensor their median, 99th percentile and maximum; next to them the synthetic rows of the same types
// (K 4096, 128 rows), centred and with random mins, so a model's rows can be placed between the two.
static int weights_stats(const std::vector<std::string>& files, const std::string& out) {
  FILE* f = out.empty() ? stdout : plat::fopen_path(std::filesystem::path(out), "wb");
  if (!f) throw std::runtime_error("cannot write " + out);
  fprintf(f, "# gemm_check --weights: row = |mean| / deviation over K; dc = the largest rotation block's |mean| x sqrt(block) "
             "/ deviation (H8R's one scale a row covers it after the rotation)\n");
  fprintf(f, "source\ttensor\ttype\tN\tK\trows\trow_median\trow_p99\trow_max\tdc_median\tdc_p99\tdc_max\n");
  struct Agg { std::vector<double> row, dc; };
  std::map<std::string, Agg> by_type;  // (source | type)
  auto put = [&](const std::string& src, const std::string& name, uint32_t type, uint64_t N, int K, std::vector<double> row,
                 std::vector<double> dc) {
    auto q = [](std::vector<double>& v, double p) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
    };
    Agg& a = by_type[src + "\t" + ggml_type_name(type)];
    a.row.insert(a.row.end(), row.begin(), row.end());
    a.dc.insert(a.dc.end(), dc.begin(), dc.end());
    fprintf(f, "%s\t%s\t%s\t%llu\t%d\t%zu\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\t%.3g\n", src.c_str(), name.c_str(), ggml_type_name(type),
            (unsigned long long)N, K, row.size(), q(row, 0.5), q(row, 0.99), q(row, 1.0), q(dc, 0.5), q(dc, 0.99), q(dc, 1.0));
  };
  std::set<uint32_t> seen;
  for (const std::string& file : files) {
    GgufFile g(file);
    const std::string src = std::filesystem::path(file).filename().string();
    for (const GgufTensor& t : g.tensors) {
      if (t.dims.size() < 2 || !t.data) continue;
      int blk = 0, bytes = 0;
      ggml_block_info(t.type, blk, bytes);
      if (blk <= 0 || bytes <= 0 || t.type == GGML_F32) continue;  // (norms, biases: F32 vectors and small tensors)
      const int K = (int)t.dims[0];
      uint64_t N = 1;
      for (size_t d = 1; d < t.dims.size(); ++d) N *= t.dims[d];
      if (K < 256 || K % blk || N < 16) continue;
      const size_t rb = (size_t)K / blk * bytes;
      const int nr = (int)std::min<uint64_t>(N, 128);
      std::vector<float> w(K);
      std::vector<double> row, dc;
      try {
        for (int i = 0; i < nr; ++i) {
          const uint64_t n = N * (uint64_t)i / nr;
          dequant_row(t.type, t.data + n * rb, K, w.data());
          double r = 0, d = 0;
          Engine::gc_row_stats(w.data(), K, r, d);
          row.push_back(r), dc.push_back(d);
        }
      } catch (const std::exception&) {
        continue;  // (a type the CPU decoder does not take)
      }
      seen.insert(t.type);
      put(src, t.name, t.type, N, K, row, dc);
    }
  }
  // the synthetic rows of the same types, both ways
  for (uint32_t type : seen)
    for (int off = 0; off < 2; ++off) {
      if (type == GGML_BF16 || type == GGML_F16) continue;  // (gemm_check draws those as floats)
      Engine::GcGen gen;
      gen.gc_rng_ = 12345 + type;
      gen.gc_offsets_ = off != 0;
      const int K = 4096, N = 128;
      const std::vector<uint8_t> raw = gen.gc_rows(type, N, K);
      int blk = 0, bytes = 0;
      ggml_block_info(type, blk, bytes);
      const size_t rb = (size_t)K / blk * bytes;
      std::vector<float> w(K);
      std::vector<double> row, dc;
      for (int n = 0; n < N; ++n) {
        dequant_row(type, raw.data() + n * rb, K, w.data());
        double r = 0, d = 0;
        Engine::gc_row_stats(w.data(), K, r, d);
        row.push_back(r), dc.push_back(d);
      }
      put(off ? "synthetic offsets" : "synthetic centred", "-", type, N, K, row, dc);
    }
  if (f != stdout) fclose(f);
  // the summary: per source and type, the rows' statistics over every tensor
  printf("%-40s %-8s %8s %8s %8s %8s %8s\n", "source", "type", "rows", "row p50", "row p99", "dc p50", "dc p99");
  for (auto& [k, a] : by_type) {
    const size_t tab = k.find('\t');
    auto q = [](std::vector<double>& v, double p) {
      std::sort(v.begin(), v.end());
      return v.empty() ? 0.0 : v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
    };
    printf("%-40s %-8s %8zu %8.3g %8.3g %8.3g %8.3g\n", k.substr(0, tab).c_str(), k.substr(tab + 1).c_str(), a.row.size(),
           q(a.row, 0.5), q(a.row, 0.99), q(a.dc, 0.5), q(a.dc, 0.99));
  }
  return 0;
}

static int run(int argc, char** argv) {
  std::string model, out;
  std::vector<std::string> weights;
  Engine::GcOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--weights") weights.push_back(next());
    else if (a == "--out") out = next();
    else if (a == "--quick") o.quick = true;
    else if (a == "--offsets") o.offsets = true;
    else if (a == "--stage") {
      o.stage = next();
      if (o.stage != "all" && o.stage != "dense" && o.stage != "fused" && o.stage != "experts")
        throw std::runtime_error("--stage all | dense | fused | experts");
    }
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--types") o.types = next();
    else if (a == "--reps") o.reps = std::max(1, std::stoi(next()));
    else throw std::runtime_error("unknown option " + a + " (--model M --out F --quick --seed N --types A,B --reps N --offsets)");
  }
  if (!weights.empty()) return weights_stats(weights, out);
  if (model.empty()) {
    fprintf(stderr, "usage: gemm_check --model M.gguf [--out results.tsv] [--quick] [--seed N] [--types Q4_K,IQ3_S] [--reps N] [--offsets]\n"
                    "                  [--stage all|dense|fused|experts]   (experts: a MoE model's, the 35B-A3B's)\n"
                    "       gemm_check --weights M.gguf [--weights N.gguf ...] [--out stats.tsv]   (no GPU: the rows' means)\n");
    return 1;
  }
  EngineOptions eo;
  eo.ctx = 4096;  // (the checks need no context: the smallest a model takes leaves the most room for their weights)
  Engine eng(model, eo);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Engine::GcRow> rows = eng.gemm_check(o);
  {  // (a standard MoE model (the 35B-A3B): its experts' paths too, stage 3)
    const std::vector<Engine::GcRow> mr = eng.gemm_check_moe(o);
    rows.insert(rows.end(), mr.begin(), mr.end());
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  FILE* f = out.empty() ? stdout : plat::fopen_path(std::filesystem::path(out), "wb");
  if (!f) throw std::runtime_error("cannot write " + out);
  // (the notes first, as lines starting with "#": a reader skips them, Get-Content | Where-Object { $_ -notmatch '^#' })
  fprintf(f, "# gemm_check: err against what each form read (fails above 1e-3); end_to_end against the file's weights times the "
             "unquantized activations (reported). H8R's end-to-end error grows with a row's mean over K (one scale a row "
             "after the rotation): end_to_end_offsets, the types with a min drawn with random mins, shows it\n");
  fprintf(f, "type\tshape\tN\tK\tT\tform\treference\tstatus\terr\tend_to_end\te2e_vs_gemv\tend_to_end_offsets\t"
             "e2e_offsets_vs_gemv\tw8_vs_raw\tdeterministic\tpair\tprivate\tspill\tkernels\tnote\n");
  int ok = 0, bad = 0, skipped = 0, nondet = 0, differs = 0, spills = 0, n_far = 0, fdiff = 0, ntaken = 0;  // (not "far": a macro of the Windows headers)
  auto num = [](double v) {  // (-1: none)
    if (v < 0) return std::string("-");
    char b[32];
    snprintf(b, sizeof b, "%.3g", v);
    return std::string(b);
  };
  for (const auto& r : rows) {
    fprintf(f, "%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\t%.3g\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%u\t%u\t%s\t%s\n", r.type.c_str(),
            r.shape.c_str(), r.N, r.K, r.T, r.form.c_str(), r.ref.c_str(), r.status.c_str(), r.err, num(r.e2e).c_str(),
            num(r.e2e_ratio).c_str(), num(r.e2e_off).c_str(), num(r.e2e_off_ratio).c_str(), num(r.w8_vs_raw).c_str(),
            r.det.c_str(), r.pair.c_str(), r.priv, r.spill, r.kernels.c_str(), r.note.c_str());
    if (r.e2e_ratio > 8) ++n_far;
    if (r.status == "ok") ++ok;
    else if (r.status == "skipped") ++skipped;
    else if (r.status == "differs") ++fdiff;  // (a fused form unlike its unfused chain: a finding, recorded)
    else if (r.status == "not taken") ++ntaken;
    else if (r.status != "ran") ++bad;
    if (r.det == "DIFFERS") ++nondet;
    if (r.pair == "DIFFERS") ++differs;
    if (r.spill) ++spills;
  }
  if (f != stdout) fclose(f);
  printf("gemm_check: %zu cases in %.0f s: %d ok, %d not ok, %d skipped; %d not deterministic, %d pairs differ, %d with spills, "
         "%d prompt forms end to end over 8x their GEMV's; fused forms: %d unlike their unfused chain, %d not taken here%s%s\n",
         rows.size(), sec, ok, bad, skipped, nondet, differs, spills, n_far, fdiff, ntaken, out.empty() ? "" : "; ", out.c_str());
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  full_speed_process();
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    zlog::eprintf("error: %s\n", e.what());
    return 1;
  }
}
