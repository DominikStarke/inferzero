// A part of class Engine (zero/engine.h), included inside the class body: the checks of tools/gemm_check.cpp
// (docs/design-shared-gemm.md 4, the shared GEMM layer's Phase 0: today's kernels, the baseline). Each check makes
// synthetic weights of a GGUF type at a shape through make_weight, so they take the forms a model's weights of that type
// and shape take, quantizes random activations with the engine's own kernel (rmsnorm_quant) and runs the engine's own
// launch (matmul). The reference is computed on the CPU, in double, from what that form itself read:
//   - a one-token GEMV or the 2-8-token GEMV (T <= 8): the raw rows as dequant_row decodes them, times the int8
//     activations and their per-32 scales as the GPU wrote them (aq_ / asc_);
//   - the W8 GEMM (T > 8): the W8 form the GEMM read (the resident copy, or the stage buffer decode_w8 filled), decoded
//     here, times aq_ / asc_ (per-256 or per-32 scales): Q2_K / Q4_K / Q5_K are requantized in that form, so their
//     reference is the requantized weight, not the file's;
//   - the H8R GEMM (gemm.cl QTYPE 111): the rotated int8 weight (in the W8 order) and its column scales, times the
//     rotated int8 activations and their per-token scales (h8_aq_ / h8_asc_): an exact integer sum times two scales.
// So the error measures the kernel, not one quantization against another. Each case also runs twice on the same input
// (bit-equal, or not: atomics or a reduction order that varies), and records the kernels it launched with their build
// options, private and spill bytes. Weights that the engine keeps both ways are made resident and just in time, and the
// two results compared bit for bit (a pair recorded, not failed: Phase 0 is the baseline).
// Off unless tools/gemm_check.cpp calls it; the hook in run() is one null test a launch.
 public:
  struct GcOptions {
    uint64_t seed = 1;
    bool quick = false;
    std::string types;  // (a comma list of GGUF type names; empty: every type the engine takes)
    int reps = 2;       // (runs of each case on the same input: 2 or more checks it is deterministic)
    // (the types with a min or an offset (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K): false, blocks centred as a quantizer makes
    //  them (the min fitted to the scale); true, a random min a block, a mean far from 0: a stress for the forms that
    //  rotate (H8R's one scale a column then covers the mean))
    bool offsets = false;
  };
  struct GcRow {
    std::string type, shape, form, ref, kernels, status, note;
    int N = 0, K = 0, T = 0;
    double err = 0, w8_vs_raw = -1;
    // (end to end: the output against dequant_row's weights times the unquantized activations, the quantization's
    //  error included: reported, not failed; e2e_ratio: against the same type and shape's one-token GEMV, the plain
    //  int8 form, for the prompt forms (H8's rotation must not be far worse); -1 where there is none)
    double e2e = -1, e2e_ratio = -1;
    double e2e_off = -1, e2e_off_ratio = -1;  // (the types with a min, drawn again with random mins: the offsets columns)
    std::string det = "-", pair = "-";
    uint32_t priv = 0, spill = 0;
  };
  std::vector<GcRow> gemm_check(const GcOptions& o) {
    std::vector<GcRow> rows;
    gc_gen_.gc_rng_ = o.seed * 0x9E3779B97F4A7C15ull + 1;
    gc_gen_.gc_offsets_ = o.offsets;
    const int tm = gemm_tm(), kmax = aq_max_k(), mmax = max_chunk;
    // the types: every one make_weight takes, and the float types load_w turns into Q8_0
    std::vector<uint32_t> types = {GGML_Q8_0, GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_1, GGML_Q2_K, GGML_Q3_K,
                                   GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS,
                                   GGML_IQ3_S, GGML_IQ1_S, GGML_IQ1_M, GGML_IQ4_NL, GGML_IQ4_XS, GGML_MXFP4, GGML_NVFP4,
                                   GGML_Q2_0, GGML_AR4_128, GGML_BF16, GGML_F16};
    if (!o.types.empty()) {
      std::vector<uint32_t> keep;
      for (uint32_t t : types)
        if (("," + o.types + ",").find("," + std::string(ggml_type_name(t)) + ",") != std::string::npos) keep.push_back(t);
      types = keep;
    }
    // the shapes: the model's own (its hidden size and FFN width), the tails that break kernels, random ones
    struct Shape { std::string name; int N, K; };
    std::vector<Shape> shapes;
    const int E = hp.n_embd, F = hp.n_ff > 0 ? hp.n_ff : hp.n_ff_exp;
    if (E > 0 && F > 0) {
      shapes.push_back({"model up", F, E});
      shapes.push_back({"model down", E, F});
    }
    shapes.push_back({"N 1040 (65 x 16)", 1040, 1024});
    shapes.push_back({"K 640 (padded)", 512, 640});
    shapes.push_back({"F 1152", 1152, 2048});
    shapes.push_back({"F 512", 512, 2048});
    shapes.push_back({"N 200 (not x16)", 200, 512});
    if (!o.quick) {
      shapes.push_back({"head 200,058", 200058, E > 0 ? E : 2048});
      shapes.push_back({"head 248,320", 248320, E > 0 ? E : 2048});
    }
    for (int i = 0; i < (o.quick ? 2 : 4); ++i) {
      const int N = 16 * (int)(1 + gc_below(512)), K = 256 * (int)(1 + gc_below((uint64_t)std::max(1, std::min(kmax, 8192) / 256)));
      shapes.push_back({"random", N, K});
    }
    // the token rows: each class edge and the chunk (1, 2, 8: the GEMVs; 9 and up: the GEMM)
    std::vector<int> Ts = {1, 2, 8, 9, 127, 128, 129};
    if (mmax > 129) Ts.push_back(mmax);
    if (o.quick) Ts = {1, 2, 9, 128};
    for (uint32_t type : types)
      for (const Shape& s : shapes) {
        // (a head-sized matrix: the GEMV of one token only, as the head's one-token decode; a prompt's head is 16 rows)
        const bool head = s.N > 100000;
        for (int T : Ts) {
          if (head && T != 1) continue;
          const uint64_t st = gc_gen_.gc_rng_;  // (the pair: the same random weights, activations and columns for both forms)
          for (int res = 0; res < 2; ++res) {
            gc_gen_.gc_rng_ = st;
            GcRow r;
            r.type = ggml_type_name(type);
            r.shape = s.name;
            r.N = s.N, r.K = s.K, r.T = T;
            r.form = res ? "resident" : "just in time";
            gc_case(type, s.N, s.K, T, res != 0, o.reps, r, tm, kmax, mmax);
            // (one-token and 2-8-token GEMVs read the raw rows either way: the resident row only for the GEMM)
            const bool both = T > 8 && r.status != "skipped";
            rows.push_back(r);
            if (!both) break;
            if (res == 1 && rows.size() >= 2) {
              GcRow& a = rows[rows.size() - 2];
              GcRow& b = rows.back();
              if ((a.status == "ok" || a.status == "FAIL") && (b.status == "ok" || b.status == "FAIL")) {
                const bool eq = gc_last_[0] == gc_last_[1];
                a.pair = b.pair = eq ? "equal" : "DIFFERS";
              }
            }
          }
          // (the types with a min: the same case again with random mins, the blocks' means far from 0, its end-to-end
          //  error in the rows' offsets columns, every run: H8R's error grows with a row's mean over K, one scale a row
          //  after the rotation, and the baseline keeps that visible. Not with --offsets, whose main run is that already)
          const bool has_min = type == GGML_Q4_1 || type == GGML_Q5_1 || type == GGML_Q2_K || type == GGML_Q4_K ||
                               type == GGML_Q5_K || type == GGML_Q2_0;
          if (has_min && !o.offsets && !rows.empty() && rows.back().status == "ok") {
            GcRow ro;
            gc_gen_.gc_offsets_ = true;
            gc_case(type, s.N, s.K, T, false, 1, ro, tm, kmax, mmax);
            gc_gen_.gc_offsets_ = false;
            if (ro.status == "ok")
              for (size_t i = rows.size(); i-- > 0 && rows[i].T == T && rows[i].shape == s.name && rows[i].type == ggml_type_name(type);)
                rows[i].e2e_off = ro.e2e;
          }
        }
      }
    gc_fused(o, types, rows, tm, kmax, mmax);
    // (each prompt form's end-to-end error against the plain int8 form's, the one-token GEMV of the same type and shape:
    //  a rotation that does not match its weights', or a decode wrong on both sides, shows here and not in err)
    for (GcRow& r : rows) {
      if (r.T <= 8 || r.e2e < 0) continue;
      for (const GcRow& g : rows)
        if (g.T == 1 && g.type == r.type && g.N == r.N && g.K == r.K && g.shape == r.shape && g.e2e > 0) {
          r.e2e_ratio = r.e2e / g.e2e;
          if (r.e2e_ratio > 8)
            r.note += std::string(r.note.empty() ? "" : "; ") + "END TO END " + std::to_string(r.e2e_ratio).substr(0, 5) +
                      "x the one-token GEMV's";
          if (r.e2e_off > 0 && g.e2e_off > 0) r.e2e_off_ratio = r.e2e_off / g.e2e_off;  // (reported, not noted)
          break;
        }
    }
    return rows;
  }
 private:
  std::vector<std::string>* gc_log_ = nullptr;   // (set while a check runs: run() notes each kernel launched)
  std::vector<std::pair<uint32_t, uint32_t>> gc_props_;  // (each noted kernel's private and spill bytes)
  std::vector<float> gc_last_[2];  // (the last two cases' results, resident and just in time: the pair)
  void gc_note(cl_kernel k) {
    const std::string s = k->name + " [" + plan_opts(k->opts) + "]";
    if (std::find(gc_log_->begin(), gc_log_->end(), s) != gc_log_->end()) return;
    gc_log_->push_back(s);
    ze_kernel_properties_t kp{};
    kp.stype = (ze_structure_type_t)0x1e;
    if (zeKernelGetProperties(k->k, &kp) == ZE_RESULT_SUCCESS) gc_props_.push_back({kp.privateMemSize, kp.spillMemSize});
    else gc_props_.push_back({0, 0});
  }
 public:
  // the random rows' generator: no GPU, so tools/gemm_check.cpp --weights uses it too (the synthetic rows' statistics
  // next to a model file's)
  struct GcGen {
  uint64_t gc_rng_ = 1;
  bool gc_offsets_ = false;  // (GcOptions::offsets: the types with a min drawn with random mins, block means far from 0)
  uint64_t gc_next() {  // (splitmix64)
    uint64_t z = (gc_rng_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint64_t gc_below(uint64_t n) { return n ? gc_next() % n : 0; }
  float gc_uniform() { return (float)((gc_next() >> 40) * (1.0 / 16777216.0)); }  // [0, 1)
  // random rows of a type: random code bytes, the scales set to sane values (an f16 scale from random bytes may be
  // huge, Inf or NaN), every value checked finite by dequant_row
  std::vector<uint8_t> gc_rows(uint32_t type, int N, int K) {
    int blk = 0, bytes = 0;
    ggml_block_info(type, blk, bytes);
    const size_t nb = (size_t)N * K / blk;
    std::vector<uint8_t> v(nb * bytes);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
      const uint64_t r = gc_next();
      memcpy(&v[i], &r, 8);
    }
    for (size_t i = v.size() / 8 * 8; i < v.size(); ++i) v[i] = (uint8_t)gc_next();
    auto f16 = [&](uint8_t* p, bool sign) {  // (a scale of 2^-9 .. 2^-6, random sign where the type allows one)
      const float d = std::ldexp(1.0f + gc_uniform(), -9 + (int)gc_below(3)) * (sign && (gc_next() & 1) ? -1.0f : 1.0f);
      const uint16_t h = f32_to_fp16(d);
      memcpy(p, &h, 2);
    };
    for (size_t b = 0; b < nb; ++b) {
      uint8_t* p = &v[b * bytes];
      switch (type) {
        case GGML_Q2_0:  // (values (q - 1) d, q 0..3: centred, q in 0..2, the mean 0; --offsets: q uniform, mean d / 2)
          f16(p, true);
          if (!gc_offsets_)
            for (int j = 2; j < bytes; ++j) {
              uint8_t c = 0;
              for (int f = 0; f < 4; ++f) c |= (uint8_t)(gc_below(3) << (2 * f));
              p[j] = c;
            }
          break;
        case GGML_Q4_0: case GGML_Q5_0: case GGML_Q8_0: case GGML_IQ4_NL: case GGML_AR4_128:
        case GGML_IQ2_XXS: case GGML_IQ2_XS: case GGML_IQ2_S: case GGML_IQ3_XXS: case GGML_IQ3_S: case GGML_IQ1_S:
        case GGML_IQ4_XS:
          f16(p, true);
          break;
        case GGML_Q8_1:
          f16(p, false), f16(p + 2, false);
          break;
        case GGML_Q4_1: case GGML_Q5_1:  // (w = d q + m: centred, m about -d x the codes' middle, as a quantizer's min)
          f16(p, false);
          if (gc_offsets_) {
            f16(p + 2, false);
          } else {
            uint16_t hd;
            memcpy(&hd, p, 2);
            const float mid = type == GGML_Q4_1 ? 7.5f : 15.5f;
            const uint16_t hm = f32_to_fp16(-fp16_to_f32(hd) * mid * (0.9f + 0.2f * gc_uniform()));
            memcpy(p + 2, &hm, 2);
          }
          break;
        case GGML_Q4_K: case GGML_Q5_K: {  // (w = d sc q - dmin m: centred, dmin = d, m = sc x the codes' middle +- 1)
          f16(p, false), f16(p + 2, false);
          if (gc_offsets_) break;
          memcpy(p + 2, p, 2);
          const bool q5 = type == GGML_Q5_K;
          const int scmax = q5 ? 4 : 8;
          const float mid = q5 ? 15.5f : 7.5f;
          int sc[8], mn[8];
          for (int j = 0; j < 8; ++j) {
            sc[j] = 1 + (int)gc_below((uint64_t)scmax);
            mn[j] = std::clamp((int)std::lrint(mid * sc[j]) + (int)gc_below(3) - 1, 0, 63);
          }
          uint8_t* q = p + 4;  // (the 12 bytes of get_scale_min_k4, packed)
          for (int j = 0; j < 4; ++j) {
            q[j] = (uint8_t)(sc[j] | ((sc[j + 4] >> 4) << 6));
            q[j + 4] = (uint8_t)(mn[j] | ((mn[j + 4] >> 4) << 6));
            q[j + 8] = (uint8_t)((sc[j + 4] & 15) | ((mn[j + 4] & 15) << 4));
          }
          break;
        }
        case GGML_Q2_K:  // (scales: low nibble sc, high nibble m; centred: dmin = d, m = 1.5 sc +- 1)
          f16(p + 80, false), f16(p + 82, false);
          if (gc_offsets_) break;
          memcpy(p + 82, p + 80, 2);
          for (int j = 0; j < 16; ++j) {
            const int sc = 1 + (int)gc_below(10), mn = std::clamp((int)std::lrint(1.5f * sc) + (int)gc_below(3) - 1, 0, 15);
            p[j] = (uint8_t)(sc | (mn << 4));
          }
          break;
        case GGML_Q3_K: f16(p + 108, true); break;
        case GGML_Q6_K: f16(p + 208, true); break;
        case GGML_IQ1_M: {  // (its f16 scale is the top nibbles of its four scale words)
          uint16_t h = f32_to_fp16(std::ldexp(1.0f + gc_uniform(), -6)), sc[4];
          memcpy(sc, p + 48, 8);
          for (int i = 0; i < 4; ++i) sc[i] = (uint16_t)((sc[i] & 0x0FFF) | (((h >> (4 * i)) & 15) << 12));
          memcpy(p + 48, sc, 8);
          break;
        }
        case GGML_MXFP4: p[0] = (uint8_t)(118 + gc_below(6)); break;  // (2^(e - 127): 2^-9 .. 2^-4)
        default: break;  // (NVFP4: its UE4M3 scales are finite and small whatever the byte)
      }
    }
    return v;
  }
  };  // (GcGen)
  // the statistics H8R's error depends on, of one row of K values (a weight row is an H8R column): the row's mean over
  // its standard deviation, and its largest rotation block's mean over that block's deviation x sqrt(the block): after
  // the rotation the block's mean is one coefficient of that size, which the row's one scale must cover
  static void gc_row_stats(const float* w, int K, double& row_ratio, double& dc_ratio) {
    int hb = 256;
    while (hb * 2 <= 1024 && K % (hb * 2) == 0) hb *= 2;  // (as h8_block)
    if (K % 256) hb = K;
    double s = 0, s2 = 0;
    for (int k = 0; k < K; ++k) s += w[k], s2 += (double)w[k] * w[k];
    const double mean = s / K, sd = std::sqrt(std::max(0.0, s2 / K - mean * mean));
    row_ratio = sd > 0 ? std::fabs(mean) / sd : 0;
    dc_ratio = 0;
    for (int b0 = 0; b0 + hb <= K; b0 += hb) {
      double bs = 0, bs2 = 0;
      for (int k = b0; k < b0 + hb; ++k) bs += w[k], bs2 += (double)w[k] * w[k];
      const double bm = bs / hb, bsd = std::sqrt(std::max(0.0, bs2 / hb - bm * bm));
      if (bsd > 0) dc_ratio = std::max(dc_ratio, std::fabs(bm) * std::sqrt((double)hb) / bsd);
    }
  }
 private:
  GcGen gc_gen_;
  uint64_t gc_next() { return gc_gen_.gc_next(); }
  uint64_t gc_below(uint64_t n) { return gc_gen_.gc_below(n); }
  float gc_uniform() { return gc_gen_.gc_uniform(); }
  std::vector<uint8_t> gc_rows(uint32_t type, int N, int K) { return gc_gen_.gc_rows(type, N, K); }
  // a W8-form weight (zero/kernels/decode.cl decode_w8_*: [g][s][j][w][lane] words, [g][s][HSTR][lane] headers, groups
  // of 16 rows) decoded to floats, rows `cols` only. per32: the Q8_0 form, one f32 scale a 32-block; h8r: the rotated
  // int8 values alone (their scale is the column's, applied by the caller)
  static void gc_w8_rows(const std::vector<uint32_t>& q, const std::vector<uint32_t>& h, int K, bool per32, bool h8r,
                         const std::vector<int>& cols, std::vector<double>& out) {
    const int nsb = K / 256, GL = 16, HSTR = per32 ? 8 : 10;
    out.assign(cols.size() * (size_t)K, 0.0);
    for (size_t ci = 0; ci < cols.size(); ++ci) {
      const int n = cols[ci], g = n / GL, lane = n % GL;
      for (int k = 0; k < K; ++k) {
        const int s = k / 256, ib = (k % 256) / 32, kk = (k % 32) / 4, by = k % 4;
        const size_t wi = (((size_t)g * nsb + s) * 4 + ib / 2) * 16 * GL + (size_t)(ib % 2) * 8 * GL + (size_t)kk * GL + lane;
        const int8_t qv = (int8_t)((q[wi] >> (8 * by)) & 0xFF);
        if (h8r) { out[ci * K + k] = qv; continue; }
        const size_t hb = ((size_t)g * nsb + s) * HSTR * GL;
        if (per32) {
          float d;
          memcpy(&d, &h[hb + (size_t)ib * GL + lane], 4);
          out[ci * K + k] = (double)d * qv;
          continue;
        }
        float d, dm;
        memcpy(&d, &h[hb + 0 * GL + lane], 4);
        memcpy(&dm, &h[hb + 1 * GL + lane], 4);
        const int hh = (k % 256) / 16;
        const int8_t sc = (int8_t)((h[hb + (size_t)(2 + hh / 4) * GL + lane] >> (8 * (hh % 4))) & 0xFF);
        const uint8_t mn = (uint8_t)((h[hb + (size_t)(6 + hh / 4) * GL + lane] >> (8 * (hh % 4))) & 0xFF);
        out[ci * K + k] = (double)d * sc * qv - (double)dm * mn;
      }
    }
  }
  // the per-32 int8 form decode32 (zero/kernels/decode.cl) gives the types it requantizes as it decodes, for every path:
  // Q5_1 / Q4_1 (s = max(|m|, |m + qmax d|) / 127, q = rint((d q + m) / s)) and NVFP4 (its per-16 scales folded per 32,
  // src/quants.h nvfp4_block32_w8). false for any other type: dequant_row is then the form
  static bool gc_per32_form(uint32_t type, const uint8_t* row, int K, double* out) {
    if (type != GGML_Q5_1 && type != GGML_Q4_1 && type != GGML_NVFP4) return false;
    for (int b = 0; b < K / 32; ++b) {
      if (type == GGML_NVFP4) {
        const uint8_t* p = row + (size_t)(b / 2) * 36;
        const uint8_t sc[2] = {p[2 * (b % 2)], p[2 * (b % 2) + 1]};
        int8_t q[32];
        const float d = nvfp4_block32_w8(sc, p + 4 + 16 * (b % 2), q);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = (double)d * q[i];
        continue;
      }
      const bool q5 = type == GGML_Q5_1;
      const uint8_t* p = row + (size_t)b * (q5 ? 24 : 20);
      uint16_t hd, hm;
      memcpy(&hd, p, 2);
      memcpy(&hm, p + 2, 2);
      const float bd = fp16_to_f32(hd), bm = fp16_to_f32(hm);
      uint32_t qh = 0;
      if (q5) memcpy(&qh, p + 4, 4);
      const uint8_t* qs = p + (q5 ? 8 : 4);
      const float vmax = std::fmax(std::fabs(bm), std::fabs(bm + (q5 ? 31.0f : 15.0f) * bd));
      const float inv = vmax > 0.0f ? 127.0f / vmax : 0.0f, d = vmax / 127.0f;
      for (int e = 0; e < 32; ++e) {
        int q = (qs[e % 16] >> (e < 16 ? 0 : 4)) & 15;
        if (q5) q |= (int)((qh >> e) & 1u) << 4;
        const int i8 = std::clamp((int)std::rint((bd * (float)q + bm) * inv), -127, 127);
        out[b * 32 + e] = (double)d * i8;
      }
    }
    return true;
  }
  template <typename T> std::vector<T> gc_read(cl_mem m, size_t n) {
    std::vector<T> v(n);
    if (n) CL_CHECK(clEnqueueReadBuffer(cl_.queue, m, CL_TRUE, 0, n * sizeof(T), v.data(), 0, nullptr, nullptr));
    return v;
  }
  // one case: type, N x K, T token rows, the weight resident or just in time
  void gc_case(uint32_t type, int N, int K, int T, bool resident, int reps, GcRow& r, int tm, int kmax, int mmax) {
    const int Mp = (T + tm - 1) / tm * tm;
    auto skip = [&](const std::string& why) { r.status = "skipped"; r.note = why; };
    int blk = 0, bytes = 0;
    ggml_block_info(type, blk, bytes);
    if (K % blk) return skip("K not a multiple of the type's block");
    const int Kp = padded_k(K);  // (the GPU's row: small-block rows padded to whole 256-superblocks)
    if (Kp > kmax) return skip("K over this model's activation buffers (" + std::to_string(kmax) + "): --model with wider rows");
    if (Mp > mmax) return skip("rows over this model's chunk (" + std::to_string(mmax) + ")");
    if (T > 8 && (N % 16 || Kp % 256)) return skip("the GEMM takes N x 16, K x 256 (a model's matrix of this shape: GEMV only)");
    if (type == GGML_AR4_128 && K % 128) return skip("AR4_128 rows of 128");
    // the weight: float types as load_w makes them (Q8_0 rows)
    std::vector<uint8_t> raw, fl;  // (fl: a float type's own rows, the end-to-end reference's weights)
    uint32_t wt = type;
    const int K0 = K;  // (the row as the file has it, before padding)
    if (type == GGML_BF16 || type == GGML_F16) {
      fl.resize((size_t)N * K * 2);
      for (size_t i = 0; i < (size_t)N * K; ++i) {
        const float f = (gc_uniform() - 0.5f) * 0.1f;
        uint16_t h;
        if (type == GGML_F16) {
          h = f32_to_fp16(f);
        } else {
          uint32_t u;
          memcpy(&u, &f, 4);
          h = (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
        }
        memcpy(&fl[i * 2], &h, 2);
      }
      raw = requant_rows_q8_0(type, fl.data(), (size_t)K * 2, N, K);
      wt = GGML_Q8_0;
    } else {
      raw = gc_rows(type, N, K);
    }
    if (Kp != K) {  // (a small-block row not whole superblocks: padded with zero blocks, as load_w_pad pads it)
      int b2 = 0, by2 = 0;
      ggml_block_info(wt, b2, by2);
      if (b2 >= 256) return skip("K not whole superblocks for a 256-block type");
      std::vector<uint8_t> p((size_t)N * (Kp / b2) * by2, 0);
      pad_small_block_rows(wt, raw.data(), N, K, p.data());
      raw.swap(p);
    }
    K = Kp;  // (from here the weight's own K, as the engine has it)
    // (the columns the CPU checks: up to 48 of N, the first, the last, random ones; every valid token row up to 64)
    std::vector<int> cols = {0, N - 1};
    while ((int)cols.size() < std::min(N, 48)) cols.push_back((int)gc_below((uint64_t)N));
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    std::vector<double> deq(cols.size() * (size_t)K);
    {
      int b2 = 0, by2 = 0;
      ggml_block_info(wt, b2, by2);
      const size_t rb = (size_t)K / b2 * by2;
      std::vector<float> row(K);
      for (size_t ci = 0; ci < cols.size(); ++ci) {
        dequant_row(wt, raw.data() + (size_t)cols[ci] * rb, K, row.data());
        for (int k = 0; k < K; ++k) {
          if (!std::isfinite(row[k])) return skip("a random row decoded to a value that is not finite");
          deq[ci * K + k] = row[k];
        }
      }
    }
    // (the end-to-end reference's weights: the file's values, the float types' own before their Q8_0 rows; zeros
    //  past a padded row's end)
    std::vector<double> wfile = deq;
    if (!fl.empty()) {
      std::vector<float> row(K0);
      for (size_t ci = 0; ci < cols.size(); ++ci) {
        dequant_row(type, fl.data() + (size_t)cols[ci] * K0 * 2, K0, row.data());
        for (int k = 0; k < K; ++k) wfile[ci * K + k] = k < K0 ? row[k] : 0.0;
      }
    }
    // the engine's state this case changes, restored after it
    const bool res0 = w8_resident;
    const int T0 = cur_T_;
    std::vector<size_t> nq0(stage_need_q_, stage_need_q_ + kRoles), nh0(stage_need_h_, stage_need_h_ + kRoles);
    struct NeedBack {
      Engine* e;
      std::vector<size_t>& q;
      std::vector<size_t>& h;
      ~NeedBack() { std::copy(q.begin(), q.end(), e->stage_need_q_), std::copy(h.begin(), h.end(), e->stage_need_h_); }
    } need_back{this, nq0, nh0};  // (make_weight raises a role's stage need: the engine's own stays as it was)
    w8_resident = resident;
    GpuWeight w;
    cl_mem x = nullptr, nw = nullptr, C = nullptr, sq = nullptr, sh = nullptr;
    const int role = 7;  // (ffn_gate's role: a named matrix of the text model, so H8 may take it as it takes a model's)
    cl_mem sq0 = stage_q_[role], sh0 = stage_h_[role];
    std::vector<std::string> log;
    struct Restore {
      Engine* e;
      GpuWeight& w;
      bool res0;
      int T0, role;
      cl_mem sq0, sh0;
      cl_mem *x, *nw, *C, *sq, *sh;
      ~Restore() {
        e->gc_log_ = nullptr;
        e->w8_resident = res0;
        e->cur_T_ = T0;
        e->stage_q_[role] = sq0, e->stage_h_[role] = sh0;
        for (int i = 0; i < kRoles; ++i) e->stage_owner_[i] = nullptr, e->stage_h8_[i] = false, e->stage_packed_[i] = 0;
        auto it = e->h8_col_.find(&w);
        if (it != e->h8_col_.end()) { e->free_buf(it->second); e->h8_col_.erase(it); }
        auto ir = e->h8_res_.find(&w);
        if (ir != e->h8_res_.end()) { e->free_buf(ir->second); e->h8_res_.erase(ir); }
        e->h8_res_rot_.erase(&w);
        e->h8_pre_live_ = false;
        e->h8_last_K_ = -1;
        e->free_buf(w.raw), e->free_buf(w.wq), e->free_buf(w.wh);
        e->free_buf(*x), e->free_buf(*nw), e->free_buf(*C), e->free_buf(*sq), e->free_buf(*sh);
      }
    } restore{this, w, res0, T0, role, sq0, sh0, &x, &nw, &C, &sq, &sh};
    try {
      w = make_weight(wt, raw.data(), raw.size(), N, K, role);
      if (!w.wq && T > 8) {  // (just in time: stage buffers of this weight's size, the engine's own left as they are)
        sq = buf(w8_bytes_q(N, Kp));
        sh = buf(w8_bytes_h(N, Kp) + 64);
        stage_q_[role] = sq, stage_h_[role] = sh;
      }
    } catch (const std::exception& e) {
      return skip(std::string("make_weight refused: ") + e.what());
    }
    if (T <= 8) r.form = "raw rows";  // (the GEMVs read the raw rows, resident copy or not)
    // the activations: random rows, the engine's norm and quantization (a norm weight of ones)
    std::vector<float> xs((size_t)Mp * Kp, 0.0f), ones((size_t)Kp, 1.0f);
    for (int m = 0; m < T; ++m)
      for (int k = 0; k < K; ++k) xs[(size_t)m * Kp + k] = (gc_uniform() - 0.5f) * 4.0f;
    x = buf(xs.size() * 4, xs.data());
    nw = buf(ones.size() * 4, ones.data());
    C = buf((size_t)Mp * N * 4);
    cur_T_ = T;
    const int a32 = act32_for(false);
    gc_log_ = &log;
    gc_props_.clear();
    std::vector<float> got;
    std::string det = "equal";
    for (int rep = 0; rep < std::max(1, reps); ++rep) {
      try {
        rmsnorm_quant(x, 0, nw, Kp, Mp, T, a32);
        matmul("gemm_check", w, Mp, C, N, 0, 0, a32);
      } catch (const std::exception& e) {
        gc_log_ = nullptr;
        r.status = "ERROR";
        r.note = e.what();
        return;
      }
      std::vector<float> c = gc_read<float>(C, (size_t)(T > 8 ? Mp : T == 1 ? 1 : Mp) * N);
      if (rep == 0) got = c;
      else if (memcmp(c.data(), got.data(), got.size() * 4)) det = "DIFFERS";
    }
    clFinish(cl_.queue);
    gc_log_ = nullptr;
    r.det = reps > 1 ? det : "-";
    for (size_t i = 0; i < log.size(); ++i) {
      r.kernels += (i ? "; " : "") + log[i];
      r.priv = std::max(r.priv, gc_props_[i].first);
      r.spill = std::max(r.spill, gc_props_[i].second);
    }
    // the reference of the form that ran
    const bool h8 = T > 8 && h8_takes(w, Mp);
    std::vector<double> wref;  // (cols x K: the weight as the form read it)
    std::vector<int8_t> aq;
    std::vector<float> asc;
    int B = 32;  // (the activations' scale block)
    std::vector<float> tok;  // (H8R: the per-token scales)
    std::vector<float> colsc;
    if (h8) {
      if (!h8_row()) { r.status = "ran"; r.ref = "none (H8 per-256, ARC_H8_SB)"; gc_keep(resident, got); return; }
      const cl_mem wq = w.wq ? w.wq : stage_q_[role];
      const std::vector<uint32_t> q = gc_read<uint32_t>(wq, w8_bytes_q(N, Kp) / 4);
      gc_w8_rows(q, {}, Kp, false, true, cols, wref);
      const std::vector<float> cs = gc_read<float>(h8_col_.at(&w), (size_t)N);
      for (int c : cols) colsc.push_back(cs[c]);
      aq = gc_read<int8_t>(h8_aq_, (size_t)Mp * Kp);
      tok = gc_read<float>(h8_asc_, (size_t)Mp);
      r.ref = "H8R int8 x int8";
    } else {
      aq = gc_read<int8_t>(aq_, (size_t)Mp * Kp);
      B = a32 ? 32 : 256;
      asc = gc_read<float>(asc_, (size_t)(Kp / B) * Mp);
      if (T <= 8) {
        wref = deq;
        r.ref = "raw rows x int8";
        int b2 = 0, by2 = 0;
        ggml_block_info(wt, b2, by2);
        const size_t rb = (size_t)K / b2 * by2;
        for (size_t ci = 0; ci < cols.size(); ++ci)
          if (gc_per32_form(wt, raw.data() + (size_t)cols[ci] * rb, K, &wref[ci * K])) r.ref = "decode32's per-32 int8 (CPU) x int8";
        // (those types' int8 codes from the GPU's own decode32, decode_w8's launch, where it takes the rows: the CPU copy
        //  above may round a code the other way at a tie, the GPU's division and fused multiply-add being its own)
        //  (N not a multiple of 16: a copy of the rows padded with zero rows to whole groups, decoded the same way)
        if (r.ref != "raw rows x int8") {
          const int Np = (N + 15) / 16 * 16;
          cl_mem rp = nullptr;
          if (Np != N) {
            std::vector<uint8_t> pr(rb * Np, 0);
            memcpy(pr.data(), raw.data(), rb * N);
            rp = upload_padded(wt, pr.data(), pr.size());
          }
          cl_mem dq = buf(w8_bytes_q(Np, K)), dh = buf(w8_bytes_h(Np, K) + 64);
          decode_w8(rp ? rp : w.raw, wt, Np, K, dq, dh, rp ? false : w.iq3t, w.w8_mins);  // (the copy: never tiled)
          const std::vector<uint32_t> q = gc_read<uint32_t>(dq, w8_bytes_q(Np, K) / 4);
          const std::vector<uint32_t> hh = gc_read<uint32_t>(dh, w8_bytes_h(Np, K) / 10 * 8 / 4);
          free_buf(dq), free_buf(dh), free_buf(rp);
          gc_w8_rows(q, hh, K, true, false, cols, wref);
          r.ref = "decode32's per-32 int8 (GPU) x int8";
        }
      } else {
        const cl_mem wq = w.wq ? w.wq : stage_q_[role], wh = w.wq ? w.wh : stage_h_[role];
        const bool per32 = w.gemm_qtype == (int)GGML_Q8_0;
        const std::vector<uint32_t> q = gc_read<uint32_t>(wq, w8_bytes_q(N, Kp) / 4);
        const std::vector<uint32_t> hh = gc_read<uint32_t>(wh, (per32 ? w8_bytes_h(N, Kp) / 10 * 8 : w8_bytes_h(N, Kp)) / 4);
        gc_w8_rows(q, hh, Kp, per32, false, cols, wref);
        r.ref = per32 ? "W8 (per-32) x int8" : "W8 x int8";
        // (the W8 form against the raw rows: 0 for the types stored exactly, the requantization's error for the rest)
        double dm = 0, wm = 0;
        for (size_t ci = 0; ci < cols.size(); ++ci)
          for (int k = 0; k < K; ++k) {
            dm = std::max(dm, std::fabs(wref[ci * Kp + k] - deq[ci * K + k]));
            wm = std::max(wm, std::fabs(deq[ci * K + k]));
          }
        r.w8_vs_raw = wm > 0 ? dm / wm : 0;
      }
    }
    double em = 0, rm = 0, ee = 0, er = 0;
    const int rows_chk = std::min(T, 64);
    for (int mi = 0; mi < rows_chk; ++mi) {
      const int m = T <= 64 ? mi : (int)((int64_t)mi * (T - 1) / (rows_chk - 1));
      // (the end-to-end reference: the f32 rows as the norm makes them, x / sqrt(mean(x^2) + eps) over the row the
      //  kernel normed (Kp, the padding's zeros in it), unquantized, times the file's weights; in double)
      double ss = 0;
      for (int k = 0; k < Kp; ++k) ss += (double)xs[(size_t)m * Kp + k] * xs[(size_t)m * Kp + k];
      const double inv_rms = 1.0 / std::sqrt(ss / Kp + (double)hp.eps);
      for (size_t ci = 0; ci < cols.size(); ++ci) {
        double s2 = 0;
        for (int k = 0; k < Kp; ++k) s2 += wfile[ci * Kp + k] * ((double)xs[(size_t)m * Kp + k] * inv_rms);
        const float g2 = got[(size_t)m * N + cols[ci]];
        ee = std::max(ee, std::isfinite(g2) ? std::fabs((double)g2 - s2) : INFINITY);
        er = std::max(er, std::fabs(s2));
      }
      for (size_t ci = 0; ci < cols.size(); ++ci) {
        double s = 0;
        if (h8) {
          int64_t is = 0;
          for (int k = 0; k < Kp; ++k) is += (int64_t)wref[ci * Kp + k] * aq[(size_t)m * Kp + k];
          s = (double)is * colsc[ci] * tok[m];
        } else {
          for (int k = 0; k < Kp; ++k)
            s += wref[ci * Kp + k] * ((double)aq[(size_t)m * Kp + k] * asc[(size_t)(k / B) * Mp + m]);
        }
        const float g = got[(size_t)m * N + cols[ci]];
        em = std::max(em, std::isfinite(g) ? std::fabs((double)g - s) : INFINITY);
        rm = std::max(rm, std::fabs(s));
      }
    }
    r.err = rm > 0 ? em / rm : em;
    r.e2e = er > 0 ? ee / er : ee;
    r.status = r.err <= 1e-3 ? "ok" : "FAIL";
    gc_keep(resident, got);
  }
  void gc_keep(bool resident, const std::vector<float>& got) { gc_last_[resident ? 1 : 0] = got; }

  // ---- stage 2: today's fused forms against their unfused chains, bit for bit, at random sizes (the design's rule:
  // "every fused form must equal its unfused chain bit for bit"). Each runs the engine's own calls in the order the
  // model files make them (engine_qwen.h layer_ffn for the FFN's); a form whose switch is off on this GPU, or that the
  // engine did not take, is a row "not taken" (proves nothing), and a difference is a row "differs" (recorded: a
  // finding, the exit code unchanged). Forms that need a layer's own state (the DeltaNet conv / gates, attn_prep, the
  // H8 output rotations of ARC_H8_OUT) are not here: the golden set's fusion rows prove them on the models
  // a weight made for a fused check (resident: no stage buffers), freed with its H8 records when the check ends
  struct GcW {
    Engine* e;
    GpuWeight w;
    std::vector<uint8_t> raw;
    GcW(Engine* en, uint32_t type, int N, int K, int role, const std::vector<uint8_t>* rows = nullptr) : e(en) {
      raw = rows ? *rows : e->gc_rows(type, N, K);
      const bool r0 = e->w8_resident;
      e->w8_resident = true;
      struct Back { Engine* e; bool r; ~Back() { e->w8_resident = r; } } back{e, r0};
      w = e->make_weight(type, raw.data(), raw.size(), N, K, role);
    }
    ~GcW() {
      for (int i = 0; i < kRoles; ++i) e->stage_owner_[i] = nullptr, e->stage_h8_[i] = false, e->stage_packed_[i] = 0;
      auto it = e->h8_col_.find(&w);
      if (it != e->h8_col_.end()) { e->free_buf(it->second); e->h8_col_.erase(it); }
      auto ir = e->h8_res_.find(&w);
      if (ir != e->h8_res_.end()) { e->free_buf(ir->second); e->h8_res_.erase(ir); }
      e->h8_res_rot_.erase(&w);
      e->free_buf(w.raw), e->free_buf(w.wq), e->free_buf(w.wh);
    }
    GcW(const GcW&) = delete;
    GcW& operator=(const GcW&) = delete;
  };
  struct GcBuf {  // (a buffer freed with the check)
    Engine* e;
    cl_mem m = nullptr;
    GcBuf(Engine* en, size_t bytes, const void* host = nullptr) : e(en) { m = e->buf(bytes, host); }
    ~GcBuf() { e->free_buf(m); }
    GcBuf(const GcBuf&) = delete;
    GcBuf& operator=(const GcBuf&) = delete;
  };
  std::vector<float> gc_randf(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& x : v) x = lo + (hi - lo) * gc_uniform();
    return v;
  }
  // the activations' quantized form as the next GEMV / GEMM reads it: aq_ rows [0, T) of K, asc_ and absum_ of those
  // rows, and the f32 copy at aq_ + H8F when the engine keeps one
  std::vector<uint8_t> gc_act_bytes(int K, int Mp, int T, int a32) {
    const int B = a32 ? 32 : 256;
    const std::vector<int8_t> q = gc_read<int8_t>(aq_, (size_t)Mp * K);
    const std::vector<float> s = gc_read<float>(asc_, (size_t)(K / B) * Mp);
    const std::vector<uint16_t> b = gc_read<uint16_t>(absum_, (size_t)(K / 16) * Mp);
    std::vector<uint8_t> o;
    auto add = [&](const void* p, size_t n) { o.insert(o.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
    for (int m = 0; m < T; ++m) {
      add(&q[(size_t)m * K], K);
      for (int j = 0; j < K / B; ++j) add(&s[(size_t)j * Mp + m], 4);
      for (int j = 0; j < K / 16; ++j) add(&b[(size_t)j * Mp + m], 2);
    }
    if (h8f_off_) {  // (the f32 copy H8R rotates from: rows of K floats after the int8 rows)
      std::vector<float> f((size_t)T * K);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, aq_, CL_TRUE, h8f_off_, f.size() * 4, f.data(), 0, nullptr, nullptr));
      add(f.data(), f.size() * 4);
    }
    return o;
  }
  // H8R's rotated input of the next GEMM: h8_aq_ rows [0, T) and their token scales
  std::vector<uint8_t> gc_h8_bytes(int K, int Mp, int T) {
    const std::vector<int8_t> q = gc_read<int8_t>(h8_aq_, (size_t)Mp * K);
    const std::vector<float> s = gc_read<float>(h8_asc_, (size_t)Mp);
    std::vector<uint8_t> o((const uint8_t*)q.data(), (const uint8_t*)q.data() + (size_t)T * K);
    o.insert(o.end(), (const uint8_t*)s.data(), (const uint8_t*)(s.data() + T));
    return o;
  }
  static void gc_cmp(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, GcRow& r) {
    if (a.size() != b.size()) { r.status = "ERROR"; r.note = "the two outputs differ in size"; return; }
    size_t nd = 0;
    for (size_t i = 0; i < a.size(); ++i) nd += a[i] != b[i];
    r.status = nd ? "differs" : "ok";
    r.err = a.empty() ? 0 : (double)nd / a.size();  // (the share of bytes that differ)
    if (nd) r.note = std::to_string(nd) + " of " + std::to_string(a.size()) + " bytes differ";
  }
  static void gc_cmpf(const std::vector<float>& a, const std::vector<float>& b, GcRow& r) {
    std::vector<uint8_t> x((const uint8_t*)a.data(), (const uint8_t*)(a.data() + a.size())),
        y((const uint8_t*)b.data(), (const uint8_t*)(b.data() + b.size()));
    gc_cmp(x, y, r);
    if (r.status != "differs") return;
    double dm = 0, vm = 0;
    for (size_t i = 0; i < a.size(); ++i) dm = std::max(dm, (double)std::fabs(a[i] - b[i])), vm = std::max(vm, (double)std::fabs(b[i]));
    r.err = vm > 0 ? dm / vm : dm;  // (floats: the largest difference over the range)
    r.note = "largest difference " + std::to_string(r.err).substr(0, 8) + " of the range; " + r.note;
  }
  void gc_fused(const GcOptions& o, const std::vector<uint32_t>& types, std::vector<GcRow>& rows, int tm, int kmax, int mmax) {
    const int T0 = cur_T_;
    struct Back { Engine* e; int t; ~Back() { e->cur_T_ = t; e->h8_pre_live_ = false; e->h8_last_K_ = -1; } } back{this, T0};
    const int reps = o.quick ? 2 : 6;  // (random sizes a form)
    const float eps = hp.eps;
    (void)eps;
    auto row = [&](const std::string& form, uint32_t type, int N, int K, int T) {
      GcRow r;
      r.type = type ? ggml_type_name(type) : "-";
      r.shape = "random";
      r.form = form;
      r.ref = "unfused chain";
      r.N = N, r.K = K, r.T = T;
      return r;
    };
    auto mp_of = [&](int T) { return (T + tm - 1) / tm * tm; };
    // the quantized types the fused checks take (a few with --quick: one of each kind)
    std::vector<uint32_t> ft;
    for (uint32_t t : types)
      if (t != GGML_BF16 && t != GGML_F16 &&
          (!o.quick || t == GGML_Q8_0 || t == GGML_Q4_K || t == GGML_Q6_K || t == GGML_IQ3_S || t == GGML_IQ4_XS || t == GGML_Q5_1))
        ft.push_back(t);
    auto rand_T = [&](bool prompt) {
      if (!prompt) return 1 + (int)gc_below(8);
      const int hi = std::min(mmax, 512);
      return 9 + (int)gc_below((uint64_t)std::max(1, hi - 8));
    };
    std::vector<int> e_sizes;  // (row lengths: multiples of 512 up to the activations' buffers, 8192 at most)
    for (int k = 512; k <= std::min(kmax, 8192); k += 512) e_sizes.push_back(k);
    if (e_sizes.empty()) return;
    // 1. the norm and H8R's rotation in one kernel (rmsnorm_h8r) against rmsnorm_quant + h8_act
    for (int i = 0; i < reps; ++i) {
      // (half the rows of an odd number of 512s: rotation blocks of 512, the form Xe-LPG takes by default)
      int K = e_sizes[gc_below(e_sizes.size())];
      if (i % 2 == 0) K = std::min(512 * (int)(1 + 2 * gc_below(8)), std::min(kmax, 8192) / 512 * 512 - (std::min(kmax, 8192) / 512 % 2 ? 0 : 512));
      if (K < 512) K = 512;
      const int T = rand_T(true), Mp = mp_of(T);
      GcRow r = row("norm+rot (rmsnorm_h8r)", GGML_Q8_0, 16, K, T);
      if (Mp > mmax) continue;
      try {
        GcW next(this, GGML_Q8_0, 16, K, 7);
        const std::vector<float> xs = gc_randf((size_t)Mp * K, -2, 2), ws = gc_randf(K, 0.5f, 1.5f);
        GcBuf x(this, xs.size() * 4, xs.data()), nw(this, ws.size() * 4, ws.data());
        cur_T_ = T;
        const int a32 = act32_for(false);
        std::vector<uint8_t> fused, plain;
        bool taken = false;
        {
          H8Share sh(this);
          rmsnorm_h8(x.m, 0, nw.m, K, Mp, T, a32, next.w);
          taken = h8_pre_live_;
          if (taken) fused = gc_h8_bytes(K, Mp, T);
        }
        if (!taken) {
          r.status = "not taken";
          r.note = "rmsnorm_h8r not taken here (H8 off, ARC_H8_NORM off, or its row limits)";
        } else {
          rmsnorm_quant(x.m, 0, nw.m, K, Mp, T, a32);
          h8_act(Mp, K, a32 != 0);
          plain = gc_h8_bytes(K, Mp, T);
          gc_cmp(fused, plain, r);
        }
      } catch (const std::exception& e) {
        r.status = "ERROR", r.note = e.what();
      }
      rows.push_back(r);
    }
    for (uint32_t type : ft) {
      // 2. the FFN's SwiGLU in the up GEMM's epilogue (a = 2), then swiglu_quant's copy (fused = 1), against both GEMMs
      //    into [gate | up] and swiglu_quant's SwiGLU (fused = 0): the next GEMM's input bit for bit; and 3. with H8R,
      //    the down GEMM rotating big_ itself (ARC_H8_SWI, H8Src) against swiglu_quant + its own rotation
      for (int i = 0; i < reps; ++i) {
        const int E = e_sizes[gc_below(e_sizes.size())], F = 256 * (int)(1 + gc_below(32)), T = rand_T(true), Mp = mp_of(T);
        if (Mp > mmax || F > kmax) continue;
        GcRow r = row("SwiGLU in the up GEMM", type, F, E, T);
        GcRow r3 = row("down GEMM rotating the SwiGLU (H8_SWI)", type, E, F, T);
        try {
          GcW g(this, type, F, E, 7), u(this, type, F, E, 8), d(this, type, E, F, 9);
          const std::vector<float> xs = gc_randf((size_t)Mp * E, -2, 2), ws = gc_randf(E, 0.5f, 1.5f);
          GcBuf x(this, xs.size() * 4, xs.data()), nw(this, ws.size() * 4, ws.data());
          GcBuf cf(this, (size_t)Mp * F * 4), cu(this, (size_t)Mp * 2 * F * 4), d1(this, (size_t)Mp * E * 4),
              d2(this, (size_t)Mp * E * 4);
          cur_T_ = T;
          const int a32 = act32_for(false);
          auto swiglu = [&](cl_mem src, int fused) {
            int FF = F, MM = Mp, TT = T, a = a32, fu = fused, gelu = 0;
            set_args(k_swiglu_, 0, src, FF, MM, TT, aq_, asc_, absum_, a, fu, gelu);
            size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
            run("swiglu_quant", k_swiglu_, 2, gws, lws);
          };
          {  // (fused, as layer_ffn's prompt path)
            H8Share sh(this);
            rmsnorm_h8(x.m, 0, nw.m, E, Mp, T, a32, g.w);
            matmul("gemm_gate", g.w, Mp, cf.m, F, 0, 0, a32);
            matmul("gemm_up", u.w, Mp, cf.m, F, 0, 2, a32);
          }
          std::vector<float> cfv = gc_read<float>(cf.m, (size_t)Mp * F);
          swiglu(cf.m, 1);
          const std::vector<uint8_t> fused = gc_act_bytes(F, Mp, T, a32);
          {  // (unfused: [gate | up], swiglu_quant computes the SwiGLU)
            rmsnorm_quant(x.m, 0, nw.m, E, Mp, T, a32);
            H8Share sh(this);
            matmul("gemm_gate", g.w, Mp, cu.m, 2 * F, 0, 0, a32);
            matmul("gemm_up", u.w, Mp, cu.m, 2 * F, F, 0, a32);
          }
          swiglu(cu.m, 0);
          gc_cmp(fused, gc_act_bytes(F, Mp, T, a32), r);
          // 3: from the fused epilogue's f32 rows (cf), the down GEMM's input both ways
          if (!h8_swi_ok(d.w, Mp)) {
            r3.status = "not taken";
            r3.note = "H8_SWI not taken here (needs H8R with the f32 copy)";
          } else {
            {
              H8Src src(this, cf.m);
              matmul("gemm_down", d.w, Mp, d1.m, E, 0, 0, a32);
            }
            swiglu(cf.m, 1);
            matmul("gemm_down", d.w, Mp, d2.m, E, 0, 0, a32);
            const std::vector<float> a = gc_read<float>(d1.m, (size_t)T * E), b = gc_read<float>(d2.m, (size_t)T * E);
            gc_cmpf(a, b, r3);
          }
          (void)cfv;
        } catch (const std::exception& e) {
          r.status = "ERROR", r.note = e.what();
        }
        rows.push_back(r);
        if (!r3.status.empty()) rows.push_back(r3);
      }
      // 4. stacked weights (qkv + z, the router + the shared expert's gate / up, gate + up stacked) against each matrix
      //    alone: a row's sum must not depend on the rows around it, at one token, 2-8 and a prompt chunk
      for (int i = 0; i < reps; ++i) {
        const int K = e_sizes[gc_below(e_sizes.size())], N1 = 16 * (int)(1 + gc_below(256)), N2 = 16 * (int)(1 + gc_below(256));
        for (int cls = 0; cls < 3; ++cls) {
          const int T = cls == 0 ? 1 : cls == 1 ? 2 + (int)gc_below(7) : rand_T(true), Mp = mp_of(T);
          if (Mp > mmax) continue;
          GcRow r = row(std::string("stacked rows (") + (cls == 0 ? "one token" : cls == 1 ? "2-8 tokens" : "prompt") + ")",
                        type, N1 + N2, K, T);
          try {
            const std::vector<uint8_t> ra = gc_rows(type, N1, K), rb = gc_rows(type, N2, K);
            std::vector<uint8_t> rs(ra);
            rs.insert(rs.end(), rb.begin(), rb.end());
            GcW a(this, type, N1, K, 0, &ra), b(this, type, N2, K, 1, &rb), s(this, type, N1 + N2, K, 0, &rs);
            const std::vector<float> xs = gc_randf((size_t)Mp * K, -2, 2), ws = gc_randf(K, 0.5f, 1.5f);
            GcBuf x(this, xs.size() * 4, xs.data()), nw(this, ws.size() * 4, ws.data());
            GcBuf ca(this, (size_t)Mp * (N1 + N2) * 4), cs(this, (size_t)Mp * (N1 + N2) * 4);
            cur_T_ = T;
            const int a32 = act32_for(false);
            const int ld = T == 1 ? 0 : N1 + N2;
            rmsnorm_quant(x.m, 0, nw.m, K, Mp, T, a32);
            {
              H8Share sh(this);
              matmul("gemm_a", a.w, Mp, ca.m, N1 + N2, 0, 0, a32);
              matmul("gemm_b", b.w, Mp, ca.m, N1 + N2, N1, 0, a32);
              matmul("gemm_stack", s.w, Mp, cs.m, N1 + N2, 0, 0, a32);
            }
            (void)ld;
            const size_t n = (size_t)(T == 1 ? 1 : T) * (N1 + N2);
            gc_cmpf(gc_read<float>(cs.m, n), gc_read<float>(ca.m, n), r);
          } catch (const std::exception& e) {
            r.status = "ERROR", r.note = e.what();
          }
          rows.push_back(r);
        }
      }
      // 6. gate + up in one GEMV launch (gemv_gu, -DGU2; opt-in, ARC_GEMV_GU=1) against the two GEMVs
      for (int i = 0; i < reps; ++i) {
        const int K = e_sizes[gc_below(e_sizes.size())], N = 16 * (int)(1 + gc_below(512));
        GcRow r = row("gate + up in one GEMV (gemv_gu)", type, 2 * N, K, 1);
        try {
          GcW g(this, type, N, K, 7), u(this, type, N, K, 8);
          const std::vector<float> xs = gc_randf((size_t)tm * K, -2, 2), ws = gc_randf(K, 0.5f, 1.5f);
          GcBuf x(this, xs.size() * 4, xs.data()), nw(this, ws.size() * 4, ws.data());
          GcBuf c1(this, (size_t)2 * N * 4), c2(this, (size_t)2 * N * 4);
          cur_T_ = 1;
          const int a32 = act32_for(false);
          rmsnorm_quant(x.m, 0, nw.m, K, tm, 1, a32);
          if (!gemv_gu(g.w, u.w, tm, c1.m)) {
            r.status = "not taken";
            r.note = "gemv_gu off (ARC_GEMV_GU=1 turns it on)";
          } else {
            matmul("gemm_gate", g.w, tm, c2.m, 2 * N, 0, 0, a32);
            matmul("gemm_up", u.w, tm, c2.m, 2 * N, N, 0, a32);
            gc_cmpf(gc_read<float>(c1.m, (size_t)2 * N), gc_read<float>(c2.m, (size_t)2 * N), r);
          }
        } catch (const std::exception& e) {
          r.status = "ERROR", r.note = e.what();
        }
        rows.push_back(r);
        if (r.status == "not taken") break;  // (off for every type then: one row says so)
      }
    }
    // 5. the residual add, its norm and the next input's norm and quantization in one kernel (add_norm_quant, Gemma 4's
    //    sandwich) against add_rmsnorm + rmsnorm_quant: x_ and the next input, on the model's row length
    for (int i = 0; i < reps && hp.n_embd % 256 == 0 && hp.n_embd > 0; ++i) {
      const int E = hp.n_embd, T = 1 + (int)gc_below(8), Mp = mp_of(T);
      GcRow r = row("add + norm + norm + quant (add_norm_quant)", 0, E, E, T);
      try {
        const std::vector<float> xv = gc_randf((size_t)Mp * E, -2, 2), bv = gc_randf((size_t)Mp * E, -2, 2),
                                 w1 = gc_randf(E, 0.5f, 1.5f), w2 = gc_randf(E, 0.5f, 1.5f);
        GcBuf nw1(this, w1.size() * 4, w1.data()), nw2(this, w2.size() * 4, w2.data());
        const float s = 0.5f + gc_uniform();
        cur_T_ = T;
        const int a32 = act32_for(false);
        auto load = [&] {
          CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, xv.size() * 4, xv.data(), 0, nullptr, nullptr));
          CL_CHECK(clEnqueueWriteBuffer(cl_.queue, big_, CL_TRUE, 0, bv.size() * 4, bv.data(), 0, nullptr, nullptr));
        };
        load();
        add_norm_quant(nw1.m, T, Mp, s, nw2.m, true, a32);
        std::vector<uint8_t> fused = gc_act_bytes(E, Mp, T, a32);
        {
          const std::vector<float> xf = gc_read<float>(x_, (size_t)T * E);
          fused.insert(fused.end(), (const uint8_t*)xf.data(), (const uint8_t*)(xf.data() + xf.size()));
        }
        load();
        add_rmsnorm(nw1.m, T, s);
        rmsnorm_quant(x_, 0, nw2.m, E, Mp, T, a32);
        std::vector<uint8_t> plain = gc_act_bytes(E, Mp, T, a32);
        {
          const std::vector<float> xf = gc_read<float>(x_, (size_t)T * E);
          plain.insert(plain.end(), (const uint8_t*)xf.data(), (const uint8_t*)(xf.data() + xf.size()));
        }
        // (the f32 copy for H8R is not add_norm_quant's: the GEMV path reads only the int8 rows; compared without it)
        if (h8f_off_) {
          const size_t f32 = (size_t)T * E * 4;
          fused.erase(fused.end() - (long long)(f32 + (size_t)T * E * 4), fused.end() - (long long)((size_t)T * E * 4));
          plain.erase(plain.end() - (long long)(f32 + (size_t)T * E * 4), plain.end() - (long long)((size_t)T * E * 4));
        }
        gc_cmp(fused, plain, r);
        if (r.status == "differs") {  // (which part: the residual x_, or the next input made from it)
          const size_t xb = (size_t)T * E * 4;
          const std::vector<uint8_t> fx(fused.end() - (long long)xb, fused.end()), px(plain.end() - (long long)xb, plain.end()),
              fa(fused.begin(), fused.end() - (long long)xb), pa(plain.begin(), plain.end() - (long long)xb);
          GcRow rx, ra;
          gc_cmp(fx, px, rx);
          gc_cmp(fa, pa, ra);
          std::vector<float> xf(T * (size_t)E), xp(T * (size_t)E);
          memcpy(xf.data(), fx.data(), xb), memcpy(xp.data(), px.data(), xb);
          GcRow rxf;
          gc_cmpf(xf, xp, rxf);
          r.note = "x_: " + (rx.status == "ok" ? std::string("equal") : rxf.note) + "; the next input: " +
                   (ra.status == "ok" ? std::string("equal") : ra.note);
        }
      } catch (const std::exception& e) {
        r.status = "ERROR", r.note = e.what();
      }
      rows.push_back(r);
    }
  }
 public:
