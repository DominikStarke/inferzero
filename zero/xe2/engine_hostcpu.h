// Part of the Engine class body (included from zero/engine.h): the one-token GEMV of the weights kept in host memory,
// done by the CPU (QUEUE "cards too small for the whole model", 2026-11-19, opt-in: ARC_HOST_CPU=1).
//
// A card too small for the whole model (an Arc B580 12 GB and the 27B: 0.5-2.2 GiB of its weights in RAM,
// plan_tight) has its GPU read those matrices over PCIe every token: d-dorok's B580 (PCIe 3.0) 7.1 tok/s with 0.49 GiB
// in RAM, where llama.cpp, which runs the layers it cannot fit on the CPU, makes 11.4 (issue #37). Here the CPU
// computes those matrices from RAM, and only the small input and result vectors cross the bus: a host wait per
// matrix, no weight read over PCIe.
//
// The CPU's form of such a matrix, made at load: its rows decoded (src/quants.h dequant_row) and requantized to int8
// with one f32 scale per 32 values (as Q8_0), the blocks the activations have too (quant_store16 with act32: int8 and
// one scale per 32). The dot product of a block in int16 / int32 (AVX2: maddubs on |a| and w signed by a), its two
// scales once per block, the sum over blocks in f32. One kernel for every weight type, read at ~1.1 bytes a weight
// from RAM (a 3-bit matrix's 0.43 over PCIe). Not bit-identical to the GPU's sums (int8 weights here, the GPU's own
// decode there, another order): KL-checked. RAM: ~1.1 bytes a weight of those matrices more, while the switch is on.

struct HostMat {
  std::vector<int8_t> q;   // (N x K int8)
  std::vector<float> d;    // (N x K / 32 scales)
  int N = 0, K = 0;
};
std::unordered_map<cl_mem, HostMat> host_mat_;  // (zero/engine_load.h upload_padded: a matrix put in host memory)
std::vector<int8_t> hcpu_q_;
std::vector<float> hcpu_y_, hcpu_sc_;
uint64_t hcpu_n_ = 0;
double hcpu_ms_ = 0;

static bool host_cpu_on() {
  static const bool on = getenv("ARC_HOST_CPU") && atoi(getenv("ARC_HOST_CPU")) != 0;
  return on;
}
// (upload_padded: a matrix it put in host memory; its rows are `data` (nbytes of GGUF `type`), valid during the call)
void host_src_note(cl_mem m, uint32_t type, const uint8_t* data, size_t nbytes, size_t nsb) {
  if (!host_cpu_on() || !m || opt.dry) return;
  // (every value in order, N x K: the rows are cut by the matmul's N and K later; nsb superblocks of 256 values,
  //  nbytes / nsb bytes each, as upload_padded counts them (gguf_sb_bytes))
  const size_t vals = nsb * 256;
  if (!vals || nbytes % nsb) return;
  HostMat h;
  h.q.resize(vals);
  h.d.resize(vals / 32);
  const int64_t nch = (int64_t)(vals / 4096 + (vals % 4096 ? 1 : 0));
  const size_t bpv = nbytes / nsb;  // (bytes a superblock of 256)
#pragma omp parallel
  {
    std::vector<float> f(4096 + 256);
#pragma omp for schedule(static)
    for (int64_t c = 0; c < nch; ++c) {
      const size_t v0 = (size_t)c * 4096, n = std::min((size_t)4096, vals - v0);
      // (4096 is a whole number of superblocks of 256 and of blocks of 32)
      dequant_row(type, data + v0 / 256 * bpv, (int)n, f.data());
      for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(f[b * 32 + i]));
        const float d = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        h.d[(v0 >> 5) + b] = d;
        for (int i = 0; i < 32; ++i) h.q[v0 + b * 32 + i] = (int8_t)std::lrintf(f[b * 32 + i] * inv);
      }
    }
  }
  host_mat_[m] = std::move(h);
}
// (one row's dot with the int8 activations: sum over blocks of wd * ad * sum(w * a))
static float host_dot(const int8_t* w, const float* wd, const int8_t* a, const float* ad, int nb) {
#ifdef __AVX2__
  __m256 acc = _mm256_setzero_ps();
  const __m256i ones = _mm256_set1_epi16(1);
  for (int b = 0; b < nb; ++b) {
    const __m256i wv = _mm256_loadu_si256((const __m256i*)(w + 32 * b));
    const __m256i av = _mm256_loadu_si256((const __m256i*)(a + 32 * b));
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(av, av), _mm256_sign_epi8(wv, av));
    const __m256 p = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(wd[b] * ad[b]), p, acc);
  }
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
#elif defined(__SSSE3__)
  // (Ivy Bridge / Sandy Bridge, older AMD: the same arithmetic in 128-bit halves, build.sh -march=native, issue #50)
  __m128 acc = _mm_setzero_ps();
  const __m128i ones = _mm_set1_epi16(1);
  for (int b = 0; b < nb; ++b) {
    __m128i t = _mm_setzero_si128();
    for (int h = 0; h < 2; ++h) {
      const __m128i wv = _mm_loadu_si128((const __m128i*)(w + 32 * b + 16 * h));
      const __m128i av = _mm_loadu_si128((const __m128i*)(a + 32 * b + 16 * h));
      t = _mm_add_epi32(t, _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(av, av), _mm_sign_epi8(wv, av)), ones));
    }
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(wd[b] * ad[b]), _mm_cvtepi32_ps(t)));
  }
  const __m128 s2 = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
#else
  float s = 0.0f;
  for (int b = 0; b < nb; ++b) {
    int t = 0;
    for (int i = 0; i < 32; ++i) t += (int)w[32 * b + i] * (int)a[32 * b + i];
    s += wd[b] * ad[b] * (float)t;
  }
  return s;
#endif
}
// (matmul1, one token: true when the CPU did it; C[c_off + n] (+)= sum_k W[n][k] * a[k])
bool host_cpu_gemv(const GpuWeight& w, int M, cl_mem C, int c_off, int acc) {
  if (!host_cpu_on() || acc > 1 || q2mm_ || profile) return false;
  const auto it = host_mat_.find(w.raw);
  if (it == host_mat_.end()) return false;
  const HostMat& h = it->second;
  const int N = w.N, K = w.K, nb = K / 32;
  if (N <= 0 || K % 32 || h.q.size() != (size_t)N * K) return false;
  const auto t0 = std::chrono::steady_clock::now();
  hcpu_q_.resize(K);
  hcpu_sc_.resize((size_t)nb * M);
  hcpu_y_.resize(N);
  // (the activations: int8 values, then one scale per 32 values at stride M (quant_store16, act32); the blocking
  //  read last, so the queue has run up to here)
  CL_CHECK(clEnqueueReadBuffer(cl_.queue, aq_, CL_FALSE, 0, (size_t)K, hcpu_q_.data(), 0, nullptr, nullptr));
  CL_CHECK(clEnqueueReadBuffer(cl_.queue, asc_, CL_TRUE, 0, hcpu_sc_.size() * 4, hcpu_sc_.data(), 0, nullptr, nullptr));
  if (acc) CL_CHECK(clEnqueueReadBuffer(cl_.queue, C, CL_TRUE, (size_t)c_off * 4, (size_t)N * 4, hcpu_y_.data(), 0, nullptr, nullptr));
  std::vector<float> ad(nb);
  for (int b = 0; b < nb; ++b) ad[b] = hcpu_sc_[(size_t)b * M];
  float* y = hcpu_y_.data();
  const int8_t* a = hcpu_q_.data();
#pragma omp parallel for schedule(static)
  for (int n = 0; n < N; ++n) {
    const float s = host_dot(h.q.data() + (size_t)n * K, h.d.data() + (size_t)n * nb, a, ad.data(), nb);
    y[n] = acc ? y[n] + s : s;
  }
  CL_CHECK(clEnqueueWriteBuffer(cl_.queue, C, CL_TRUE, (size_t)c_off * 4, (size_t)N * 4, y, 0, nullptr, nullptr));
  ++hcpu_n_;
  hcpu_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return true;
}
