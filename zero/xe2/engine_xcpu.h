// Part of the Engine class body (included from zero/engine.h): one token's MoE experts that are not in the GPU's
// expert cache, computed by the CPU (ARC_XCPU=1, opt-in).
//
// A miss otherwise copies the expert's record (Flash Next: ~2.3 MB) to the GPU and the layer waits for it: on a
// B60 in a PCIe 3.0 x8 slot ~0.36 ms a miss, 49 misses a token at 90 % hits, ~17 of 44 ms (expert cache size sweep,
// 2026-11-19). Here the routing read the split already does (engine_moe.h, ARC_XSPLIT) also brings the token's int8
// activations and the slots' weights; the cached experts and the shared expert are queued on the GPU, the CPU
// computes the others meanwhile (gate / up rows against the GPU's own int8 activations, SwiGLU, h quantized as
// moe_quant does, down rows), and their outputs (weighted as moe_down_q writes them) are copied into their slots of my_ ahead of
// moe_combine, in the immediate list's order: no host wait for them. The kernels are zero/xe2/cpu_experts.h's
// (ggml's AVX2 forms). Not bit-identical to the GPU's sums (other blocks, another order): KL-checked.
// The misses still enter the GPU's cache, copied in the background (xcache.h fill), for the next tokens.
// ARC_XCPU_THREADS=n: the CPU threads (default: the logical CPUs / 2, the physical cores of a 2-way SMT CPU).


struct XcpuLayer {
  const uint8_t* w[3] = {};  // gate, up, down: expert 0's first row in the GGUF's mapping
  size_t rb[3] = {};         // bytes a row there
  size_t eb[3] = {};         // bytes an expert there
  cpux::Kern k[3], kg[3];    // the kernels for the GGUF's layout and for the expert file's (GPU raw form)
  size_t prb[3] = {}, poff[3] = {};  // the expert file's record: bytes a row, offset of each matrix
  bool ok = false;
};
std::vector<XcpuLayer> xcpu_l_;
std::vector<int8_t> xcpu_aq_;
std::vector<float> xcpu_asc_, xcpu_swt_, xcpu_as_, xcpu_gu_, xcpu_h_, xcpu_hs_;
std::vector<int8_t> xcpu_hq_;
float* xcpu_y_ = nullptr;  // (host memory the GPU copies from: NU rows of n_embd)
uint8_t* xcpu_in_ = nullptr;  // (host memory the inputs are copied into)
uint64_t xcpu_n_ = 0, xcpu_ram_ = 0;  // (experts computed; of them read from the RAM tier)
double xcpu_ms_ = 0, xcpu_rd_ms_ = 0, xcpu_fill_ms_ = 0;  // (compute + copies; the input reads; fills and commits)

static bool xcpu_on() {
  static const bool on = getenv("ARC_XCPU") && atoi(getenv("ARC_XCPU")) != 0;
  return on;
}
static int xcpu_threads() {
  static const int t = getenv("ARC_XCPU_THREADS") ? std::max(1, atoi(getenv("ARC_XCPU_THREADS")))
                                                  : std::max(1, (int)std::thread::hardware_concurrency() / 2);
  return t;
}
// (this layer's experts all have CPU kernels, separate gate / up, a down width the activations' blocks divide)
bool xcpu_layer_ok(const Layer& L) {
  // (DeepSeek-V4.1 since 2026-11-19: its MXFP4 / Q5_1 / Q8_0 experts and its SwiGLU clamp; Gemma 4 not: gate_up fused)
  if (!xcpu_on() || !hp.moe_std || hp.gemma) return false;
  if (xcpu_l_.empty()) xcpu_l_.resize(hp.n_layer + 1);
  if (L.il < 0 || L.il >= (int)xcpu_l_.size()) return false;
  XcpuLayer& X = xcpu_l_[L.il];
  if (X.ok || X.w[0]) return X.ok;
  const std::string p = "blk." + std::to_string(L.il) + ".";
  const char* nm[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
  X.w[0] = (const uint8_t*)1;  // (looked at: not again)
  for (int i = 0; i < 3; ++i) {
    const GgufTensor* t = f_.find(p + nm[i]);
    if (!t || t->dims.size() != 3 || !t->data) return false;
    X.k[i] = cpux::kern(t->type, false);
    X.kg[i] = cpux::kern(t->type, true);
    const uint64_t K = t->dims[0], N = t->dims[1], NE = t->dims[2];
    if (!X.k[i].dot || !X.kg[i].dot || K % X.k[i].blk || (int)NE != hp.n_expert) return false;
    // (the record's matrix: N rows of padded_k(K) values in superblocks of the GPU raw form, expert_store.h build)
    const ExpertStore::LayerInfo& LI = xc_->st.layers[L.il];
    if (LI.type[i] != t->type || LI.part[i] != N * (padded_k((int)K) / 256) * X.kg[i].sb) return false;
    X.prb[i] = LI.part[i] / N;
    X.poff[i] = i == 0 ? 0 : X.poff[i - 1] + LI.part[i - 1];
    if ((i < 2 && (K != (uint64_t)hp.n_embd || N != (uint64_t)hp.n_ff_exp)) ||
        (i == 2 && (K != (uint64_t)hp.n_ff_exp || N != (uint64_t)hp.n_embd)))
      return false;
    X.rb[i] = t->nbytes / (N * NE);
    X.eb[i] = X.rb[i] * N;
    X.w[i] = t->data;
  }
  X.ok = true;
  return true;
}
// The routing read of a layer the CPU may help: the tiles, the activations and the slots' weights in one wait (three
// small copies more into host memory than the split's read alone)
void xcpu_read_tiles(int Mp, std::vector<int>& tl) {
  const auto t0 = std::chrono::steady_clock::now();
  const int E = hp.n_embd, NB = E / 32, NU = hp.n_expert_used;
  const size_t nt = (tl.size() * 4 + 63) / 64 * 64, na = (size_t)(E + 63) / 64 * 64, ns = (size_t)NB * Mp * 4, nw = (size_t)NU * 4;
  if (!xcpu_in_) xcpu_in_ = (uint8_t*)zrt::host_alloc(nt + na + ns + nw);
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), xcpu_in_, mtiles_->ptr, tl.size() * 4, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), xcpu_in_ + nt, aq_->ptr, E, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), xcpu_in_ + nt + na, asc_->ptr, ns, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), xcpu_in_ + nt + na + ns, swt_->ptr, nw, nullptr, 0, nullptr));
  zrt::sync();
  memcpy(tl.data(), xcpu_in_, tl.size() * 4);
  xcpu_aq_.assign((const int8_t*)(xcpu_in_ + nt), (const int8_t*)(xcpu_in_ + nt) + E);
  xcpu_asc_.assign((const float*)(xcpu_in_ + nt + na), (const float*)(xcpu_in_ + nt + na + ns));
  xcpu_swt_.assign((const float*)(xcpu_in_ + nt + na + ns), (const float*)(xcpu_in_ + nt + na + ns + nw));
  xcpu_rd_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
// The misses (expert, slot) of layer L: computed, their weighted outputs copied into my_'s slot rows (queued, in
// order before moe_combine). xcpu_read_tiles read this layer's inputs. Each expert's rows from its record in the RAM tier
// (pinned, the expert file's layout) when it is there, else from the GGUF's mapping (whose pages may still have to be
// read). One parallel region: gate / up rows, then SwiGLU and the down input's quantization per miss, then down rows.
void xcpu_run(const Layer& L, const std::vector<std::pair<int, int>>& miss, int Mp) {
  const auto t0 = std::chrono::steady_clock::now();
  const XcpuLayer& X = xcpu_l_[L.il];
  const int E = hp.n_embd, F = hp.n_ff_exp, Fp = padded_k(F), NB = E / 32, NM = (int)miss.size();
  struct Src { const uint8_t* w[3]; size_t rb[3]; const cpux::Kern* k[3]; int kd; };  // (kd: the down rows' width)
  std::vector<Src> src(NM);
  for (int m = 0; m < NM; ++m) {
    const int e = miss[m].first;
    Src& S = src[m];
    if (const uint8_t* rec = (const uint8_t*)xc_->st.ram_src(L.il, e)) {
      for (int i = 0; i < 3; ++i) S.w[i] = rec + X.poff[i], S.rb[i] = X.prb[i], S.k[i] = &X.kg[i];
      // (a per-32 / per-64 type's padded down rows read to F only: the blocks past it are zero; superblock types: all)
      S.kd = X.kg[2].blk < 256 ? F : Fp;
      ++xcpu_ram_;
    } else {
      for (int i = 0; i < 3; ++i) S.w[i] = X.w[i] + (size_t)e * X.eb[i], S.rb[i] = X.rb[i], S.k[i] = &X.k[i];
      S.kd = F;
    }
  }
  // the activations as the GPU's experts read them: aq_'s int8 and asc_'s scale per 32 values (column 0 of Mp)
  xcpu_as_.resize(NB);
  for (int b = 0; b < NB; ++b) xcpu_as_[b] = xcpu_asc_[(size_t)b * Mp];
  // (h quantized as moe_quant does, at Fp values, zero past F: the GGUF's rows read the first F, the record's all Fp)
  xcpu_gu_.resize((size_t)NM * 2 * F);
  xcpu_h_.assign((size_t)NM * Fp, 0.0f);
  xcpu_hq_.resize((size_t)NM * Fp);
  xcpu_hs_.resize((size_t)NM * Fp / 32);
  if (!xcpu_y_) xcpu_y_ = (float*)zrt::host_alloc((size_t)hp.n_expert_used * E * 4);
  const int8_t* xq = xcpu_aq_.data();
  const float L_clamp = hp.ds41 ? ds_.clamp : 0.0f;  // (DeepSeek-V4.1's SwiGLU clamp, ds_swi_defs; 0: none)
  const float* xs = xcpu_as_.data();
#pragma omp parallel num_threads(xcpu_threads())
  {
#pragma omp for schedule(static)
    for (int r = 0; r < NM * 2 * F; ++r) {  // gate and up rows of every miss
      const int m = r / (2 * F), j = r % (2 * F), g = j < F ? 0 : 1;
      const Src& S = src[m];
      xcpu_gu_[r] = S.k[g]->dot(E, S.w[g] + (size_t)(j % F) * S.rb[g], xq, xs);
    }
#pragma omp for schedule(static)
    for (int m = 0; m < NM; ++m) {
      const float* gu = &xcpu_gu_[(size_t)m * 2 * F];
      float* h = &xcpu_h_[(size_t)m * Fp];
      if (L_clamp > 0.0f) {  // (DeepSeek-V4.1: the gate clamped from above, the up both ways, as moe_gateup_q's SWI_CLAMP)
        for (int j = 0; j < F; ++j) {
          const float gc = std::min(gu[j], L_clamp), uc = std::clamp(gu[F + j], -L_clamp, L_clamp);
          h[j] = gc / (1.0f + std::exp(-gc)) * uc;
        }
      } else {
        for (int j = 0; j < F; ++j) h[j] = gu[j] / (1.0f + std::exp(-gu[j])) * gu[F + j];
      }
      cpux::quant_act32(h, xcpu_hq_.data() + (size_t)m * Fp, xcpu_hs_.data() + (size_t)m * Fp / 32, Fp);
    }
#pragma omp for schedule(static)
    for (int r = 0; r < NM * E; ++r) {  // down rows, weighted by the slot's routing weight (as moe_down_q writes y)
      const int m = r / E, n = r % E, slot = miss[m].second;
      const Src& S = src[m];
      xcpu_y_[(size_t)slot * E + n] = xcpu_swt_[slot] * S.k[2]->dot(S.kd, S.w[2] + (size_t)n * S.rb[2], xcpu_hq_.data() + (size_t)m * Fp,
                                                                  xcpu_hs_.data() + (size_t)m * Fp / 32);
    }
  }
  for (const auto& [e, slot] : miss)
    ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), my_->ptr + (size_t)slot * E * 4, xcpu_y_ + (size_t)slot * E,
                                           (size_t)E * 4, nullptr, 0, nullptr));
  xcpu_n_ += NM;
  xcpu_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
