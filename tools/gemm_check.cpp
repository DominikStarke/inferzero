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

static int run(int argc, char** argv) {
  std::string model, out;
  Engine::GcOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--quick") o.quick = true;
    else if (a == "--offsets") o.offsets = true;
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--types") o.types = next();
    else if (a == "--reps") o.reps = std::max(1, std::stoi(next()));
    else throw std::runtime_error("unknown option " + a + " (--model M --out F --quick --seed N --types A,B --reps N --offsets)");
  }
  if (model.empty()) {
    fprintf(stderr, "usage: gemm_check --model M.gguf [--out results.tsv] [--quick] [--seed N] [--types Q4_K,IQ3_S] [--reps N] [--offsets]\n");
    return 1;
  }
  EngineOptions eo;
  eo.ctx = 4096;  // (the checks need no context: the smallest a model takes leaves the most room for their weights)
  Engine eng(model, eo);
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<Engine::GcRow> rows = eng.gemm_check(o);
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  FILE* f = out.empty() ? stdout : plat::fopen_path(std::filesystem::path(out), "wb");
  if (!f) throw std::runtime_error("cannot write " + out);
  fprintf(f, "type\tshape\tN\tK\tT\tform\treference\tstatus\terr\tend_to_end\te2e_vs_gemv\tw8_vs_raw\tdeterministic\tpair\t"
             "private\tspill\tkernels\tnote\n");
  int ok = 0, bad = 0, skipped = 0, nondet = 0, differs = 0, spills = 0, n_far = 0;  // (not "far": a macro of the Windows headers)
  auto num = [](double v) {  // (-1: none)
    if (v < 0) return std::string("-");
    char b[32];
    snprintf(b, sizeof b, "%.3g", v);
    return std::string(b);
  };
  for (const auto& r : rows) {
    fprintf(f, "%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\t%.3g\t%s\t%s\t%s\t%s\t%s\t%u\t%u\t%s\t%s\n", r.type.c_str(), r.shape.c_str(), r.N,
            r.K, r.T, r.form.c_str(), r.ref.c_str(), r.status.c_str(), r.err, num(r.e2e).c_str(), num(r.e2e_ratio).c_str(),
            num(r.w8_vs_raw).c_str(), r.det.c_str(), r.pair.c_str(), r.priv, r.spill, r.kernels.c_str(), r.note.c_str());
    if (r.e2e_ratio > 8) ++n_far;
    if (r.status == "ok") ++ok;
    else if (r.status == "skipped") ++skipped;
    else if (r.status != "ran") ++bad;
    if (r.det == "DIFFERS") ++nondet;
    if (r.pair == "DIFFERS") ++differs;
    if (r.spill) ++spills;
  }
  if (f != stdout) fclose(f);
  printf("gemm_check: %zu cases in %.0f s: %d ok, %d not ok, %d skipped; %d not deterministic, %d pairs differ, %d with spills, "
         "%d prompt forms end to end over 8x their GEMV's%s%s\n",
         rows.size(), sec, ok, bad, skipped, nondet, differs, spills, n_far, out.empty() ? "" : "; ", out.c_str());
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
