// A part of class Engine (zero/engine.h), included inside the class body: ARC_GEMM_PLAN=1, the log of which kernel
// each matrix product and every other step takes (docs/gemm-inventory.md tables A and K; the user, 2026-11-27).
// Recorded at the launch (Engine::run), so it shows what ran, not what the code reads as: the start checks, the memory
// plan and the row counts have all decided by then. One line the first time each distinct case launches:
//   gemm-plan: <role> | <weight type> | M <load | 1 | 2-8 | chunk> | <memory> -> <kernel> [-D options]
//              grid G local L | fused: <what it fuses, or -> | at <file:line> from <model file:line> | gpu <device>
// (memory: a GEMV's "raw" / "host raw", the rows it reads; a GEMM's "w8 resident" / "w8 jit" / "host", where its W8
//  comes from; "-" for a step with no weight) and one summary line with the model, its routes and its fused forms
// when the engine is freed (runz's end, a server's model switch) or after serverz's warm-up. Logging only (LOG_INF):
// nothing it records changes a launch, so the results are bit-identical with it on and off. Off by default.
// The role is the name the launch is profiled under (--prof: gemm_gate, rmsnorm_quant, ...). The weight's type,
// host / resident / just in time come from matmul's scope (PlanScope); a kernel built -DQT=n says its type itself.
// The fused labels below are a table of today's fused kernels and options (table K); a kernel not in it says "-".
static bool plan_on() {
  static const bool on = getenv("ARC_GEMM_PLAN") && atoi(getenv("ARC_GEMM_PLAN")) != 0;
  return on;
}
// (the weight of the product launching now and matmul's caller, set by matmul / gemv_gu for their launches)
const GpuWeight* plan_w_ = nullptr;
const std::source_location* plan_from_ = nullptr;
bool plan_fwd_ = false;  // (the constructor has finished: its launches were the load's, "M load")
struct PlanScope {
  Engine* e;
  const GpuWeight* prev;
  const std::source_location* prev_from;
  PlanScope(Engine* en, const GpuWeight* w, const std::source_location& from)
      : e(en), prev(en->plan_w_), prev_from(en->plan_from_) {
    e->plan_w_ = w;
    if (!prev_from) e->plan_from_ = &from;  // (the outermost caller's line, a helper's PlanFrom's too: the model file's)
  }
  ~PlanScope() { e->plan_w_ = prev, e->plan_from_ = prev_from; }
  PlanScope(const PlanScope&) = delete;
  PlanScope& operator=(const PlanScope&) = delete;
};
// (a step with no weight, launched by a helper (rmsnorm_quant, add_norm_quant, add_rmsnorm, rmsnorm_h8): the model
//  file's line that called the helper, when no outer scope has set one; table K's chain per model)
struct PlanFrom {
  Engine* e;
  const std::source_location* prev;
  PlanFrom(Engine* en, const std::source_location& from) : e(en), prev(en->plan_from_) {
    if (!prev) e->plan_from_ = &from;
  }
  ~PlanFrom() { e->plan_from_ = prev; }
  PlanFrom(const PlanFrom&) = delete;
  PlanFrom& operator=(const PlanFrom&) = delete;
};
std::set<std::string> plan_seen_;     // (the cases logged, this engine's: a split model's shards each log their own)
std::set<std::string> plan_fused_;     // (the fused forms it took)
std::set<std::string> plan_kernels_;   // (the kernels it launched)
// the -D options and the large register file, the rest (-cl-std, -cl-mad-enable) being the same for every kernel
static std::string plan_opts(const std::string& o) {
  std::string r;
  for (size_t i = 0; i < o.size();) {
    while (i < o.size() && o[i] == ' ') ++i;
    size_t j = o.find(' ', i);
    if (j == std::string::npos) j = o.size();
    const std::string t = o.substr(i, j - i);
    if (t.rfind("-D", 0) == 0) r += (r.empty() ? "" : " ") + t;
    else if (t == "-cl-intel-256-GRF-per-thread") r += (r.empty() ? "" : " ") + std::string("grf256");
    i = j;
  }
  return r;
}
static bool plan_has(const std::string& o, const char* d) {  // (an option, as a whole word: -DGU2, not -DGU2X)
  const std::string s(d);
  for (size_t p = o.find(s); p != std::string::npos; p = o.find(s, p + 1)) {
    const size_t e = p + s.size();
    if ((p == 0 || o[p - 1] == ' ') && (e == o.size() || o[e] == ' ' || o[e] == '=')) return true;
  }
  return false;
}
// what a launch fuses, from its kernel, options and role; "" when it is a single step (table K's rows)
std::string plan_fused(const std::string& k, const std::string& o, const std::string& role) const {
  std::vector<std::string> f;
  if (plan_has(o, "-DGU2") || k == "gemv_gu") f.push_back("gate+up");
  if (plan_has(o, "-DXW_GU") || role == "gemm_gateup") f.push_back("gate+up+swiglu");
  if (role == "gemm_qkvz") f.push_back("qkv+z");
  if (role == "gemm_router_sh") f.push_back("router+shared gate/up");
  if (plan_has(o, "-DQ2F")) f.push_back("two int8 passes in one read");
  if (k == "rmsnorm_h8r") f.push_back("norm+rot");
  if (k == "attn_gate_h8r") f.push_back("attn gate+rot");
  if (k == "gated_norm_h8r") f.push_back("gated norm+rot");
  if (h8_fsrc_ && k.rfind("h8r_", 0) == 0) f.push_back("swiglu+rot");  // (ARC_H8_SWI: the rotation reads gate / up)
  if (plan_has(o, "-DH8F")) f.push_back("quant+f32 copy");
  if (k == "rmsnorm_quant") f.push_back("norm+quant");
  if (k == "add_norm_quant") f.push_back("add+norm+quant");
  if (k == "add_rmsnorm") f.push_back("add+norm");
  if (k == "swiglu_quant") f.push_back("swiglu+quant");
  if (k == "gated_norm_quant") f.push_back("gated norm+quant");
  if (k == "attn_gate_quant") f.push_back("attn gate+quant");
  if (k == "attn_prep" || k == "attn_prep_g") f.push_back("q/k norm+rope+kv store");
  if (plan_has(o, "-DCONV_UPD")) f.push_back("conv+state update");
  if (plan_has(o, "-DFUSE_GATES")) f.push_back("recurrence+gates");
  if (plan_has(o, "-DGDN_FUSE2")) f.push_back("two recurrence steps");
  if (k == "moe_rg") f.push_back("decode+expert gemm");
  if (k == "ink_ctrl_fused") f.push_back("controller");
  std::string r;
  for (const auto& s : f) r += (r.empty() ? "" : ", ") + s;
  return r;
}
// the file:line of the launch, from the repository's root (gcc gives the full path; MSVC's /d1trimfile trims it)
static std::string plan_at(const std::source_location& loc) {
  std::string f = loc.file_name();
  for (char& c : f) if (c == '\\') c = '/';
  for (const char* root : {"/zero/", "/tools/", "/src/"}) {
    const size_t p = f.rfind(root);
    if (p != std::string::npos) { f = f.substr(p + 1); break; }
  }
  return f + ":" + std::to_string(loc.line());
}
void plan_note(const char* name, cl_kernel k, int dims, const size_t* gws, const size_t* lws, const std::source_location& loc) {
  const std::string kn = k ? k->name : "?", ko = k ? plan_opts(k->opts) : "";
  std::string type = "-";
  if (const size_t p = ko.find("-DQT="); p != std::string::npos) {
    const int t = atoi(ko.c_str() + p + 5);
    type = ggml_type_name((uint32_t)t);
  } else if (plan_w_) {
    type = ggml_type_name(plan_w_->type);
  }
  const int T = cur_T_;
  const char* mcls = !plan_fwd_ ? "load" : T <= 1 ? "1" : T <= 8 ? "2-8" : "chunk";
  // (the weight's rows: in host memory (plan_tight), or on the GPU with a resident W8 copy for the prompt GEMM, or
  //  decoded to W8 just in time for each chunk; a GEMV reads the raw rows in every case)
  // (a GEMV reads the raw rows whatever W8 form the prompt GEMM has: "raw", "host raw")
  const bool gemv = kn.rfind("gemv", 0) == 0;
  const char* mem = !plan_w_ ? "-"
                    : gemv   ? (plan_w_->host ? "host raw" : "raw")
                    : plan_w_->host ? "host" : plan_w_->wq ? "w8 resident" : "w8 jit";
  const std::string fused = plan_fused(kn, ko, name);
  const std::string from = plan_from_ ? plan_at(*plan_from_) : "";
  const std::string key = std::string(name) + "|" + type + "|" + mcls + "|" + mem + "|" + kn + "|" + ko + "|" +
                          (lws ? "l" : "a") + "|" + from;
  if (!plan_seen_.insert(key).second) return;
  plan_kernels_.insert(kn);
  for (size_t i = 0; i < fused.size();) {  // (each fused form once in the summary, not each combination)
    size_t j = fused.find(", ", i);
    if (j == std::string::npos) j = fused.size();
    plan_fused_.insert(fused.substr(i, j - i));
    i = j + 2;
  }
  auto dim = [&](const size_t* v) {
    std::string s;
    for (int i = 0; i < dims; ++i) s += (i ? "x" : "") + std::to_string(v[i]);
    return s;
  };
  // (no local size given: the size clrt chose, shrunk from the driver's suggestion until it divides the grid)
  const std::string loc_s = lws ? dim(lws) : "auto " + std::to_string(k ? k->gx : 0) + "x" + std::to_string(k ? k->gy : 0);
  std::string gpu = cl_.device_id.substr(0, 40);
  if (tp_on()) gpu += " shard " + std::to_string(opt.tp_rank);
  if (opt.pp_size > 1) gpu += " stage " + std::to_string(opt.pp_begin) + "-" + std::to_string(opt.pp_end);
  LOG_INF("gemm-plan: %s | %s | M %s | %s -> %s [%s] grid %s local %s | fused: %s | at %s%s%s | gpu %s\n", name,
          type.c_str(), mcls, mem, kn.c_str(), ko.c_str(), dim(gws).c_str(), loc_s.c_str(),
          fused.empty() ? "-" : fused.c_str(), plan_at(loc).c_str(), from.empty() ? "" : " from ", from.c_str(),
          gpu.c_str());
}
// when: "freed" (the engine's destructor: runz's end, the server's model switch) or "after the warm-up" (serverz's
// start: the server is stopped by a kill, Ctrl+C or _exit, so no destructor runs; that summary holds the routes the
// warm-up took, and a model switch prints the full one)
 public:  // (the server calls it after its warm-up, tools/server_load.h; the rest of this part stays private)
void plan_summary(const char* when = "freed") const {
  if (!plan_on() || plan_seen_.empty()) return;
  std::string fused;
  for (const auto& s : plan_fused_) fused += (fused.empty() ? "" : "; ") + s;
  LOG_INF("gemm-plan summary (%s): %s (%s)%s: %zu routes, %zu kernels; fused: %s\n", when, hp.arch.c_str(),
          std::filesystem::path(f_.path).filename().string().c_str(),
          tp_on() ? (" shard " + std::to_string(opt.tp_rank)).c_str() : "", plan_seen_.size(), plan_kernels_.size(),
          fused.empty() ? "-" : fused.c_str());
}
 private:  // (as the class body around this part's include, zero/engine.h)
