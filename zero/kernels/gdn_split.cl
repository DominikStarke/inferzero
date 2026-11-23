// The gated delta rule of gdn_recur (ops.cl) with GDN_P (2 or 4) work-items a state column instead of one, so P
// times the threads in flight for the same state (ARC_GDN_SPLIT=2 / 4, opt-in, 2026-11-23: the A770's gdn_recur, 343
// ms of a 27B 4K prompt, ~1.2x a B60's, is bound by the latency of one token's step with 6,144 columns over 4,096
// hardware threads; the A770 plan's S2, #45).
//
// Bit for bit gdn_recur: it sums a column in four chains, value i into chain i % 4 in order of i, and adds them as
// (a0 + a1) + (a2 + a3). Here work-item p of a column owns whole chains: P 4 the chain p, P 2 the chains p and p + 2,
// each over its values in gdn_recur's order, and the four chain sums are gathered from the column's work-items by
// shuffles and added the same way in each. The decay and the update are per value. So every sum and every state value
// is the default's (the user, 2026-11-23: the KL as before the change; a first form that split the column into halves
// summed in another order, KL 0.0008 mean, up to 0.024 at a position).
// x: [M][C] = [q (nk*128) | k (nk*128) | v (nv*128)]; ab: [M][2*nv] = [decay | beta] (-DFUSE_GATES: the raw
// projections, turned into them here as gdn_gates does). grid (nv * 128 * P), local 64 (explicit: a column's P
// work-items must be neighbours in one sub-group).
#ifndef GDN_P
#define GDN_P 2
#endif
#if GDN_P != 2 && GDN_P != 4
#error "GDN_P: 2 or 4"
#endif
#define GDN_NC (4 / GDN_P)  // (chains a work-item: 2 at P 2, 1 at P 4)
#define GDN_NI (128 / GDN_P)
#ifdef FUSE_GATES
inline float gdns_decay(float x, float b, float sa) {
  const float a = x + b;
  return exp((a > 20.0f ? a : log(1.0f + exp(a))) * sa);
}
#define GDNS_DECAY(x) gdns_decay((x), dtb, ssa)
#define GDNS_BETA(x) (1.0f / (1.0f + exp(-(x))))
#else
#define GDNS_DECAY(x) (x)
#define GDNS_BETA(x) (x)
#endif
// the column's four chain sums, (c0 + c1) + (c2 + c3) as gdn_recur adds them, the same in each of its work-items.
// x0 / x1: this work-item's chains (P 2: chains p and p + 2; P 4: chain p in x0); base: the column's first lane
inline float gdns_join(float x0, float x1, int base) {
#if GDN_P == 4
  (void)x1;
  const float c0 = intel_sub_group_shuffle(x0, base), c1 = intel_sub_group_shuffle(x0, base + 1);
  const float c2 = intel_sub_group_shuffle(x0, base + 2), c3 = intel_sub_group_shuffle(x0, base + 3);
#else
  const float c0 = intel_sub_group_shuffle(x0, base), c2 = intel_sub_group_shuffle(x1, base);
  const float c1 = intel_sub_group_shuffle(x0, base + 1), c3 = intel_sub_group_shuffle(x1, base + 1);
#endif
  return (c0 + c1) + (c2 + c3);
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gdn_recur_split(__global const float* x, int C, __global const float* ab, int nk, int nv,
                              __global const float* S, int M_valid, __global float* o, __global float* Sout
#ifdef FUSE_GATES
                              , __global const float* dt_bias, __global const float* ssm_a
#endif
                              ) {
  const int gid = get_global_id(0);
  const int col = gid / GDN_P, p = gid % GDN_P, base = (int)get_sub_group_local_id() - p;
  const int h = col / 128, j = col % 128, kh = h % nk;
#ifdef FUSE_GATES
  const float dtb = dt_bias[h], ssa = ssm_a[h];
#endif
  const int qo = kh * 128 + p, ko = nk * 128 + kh * 128 + p, vo = 2 * nk * 128 + h * 128 + j;
  // (this work-item's values: index i = 4 m + p + 2 c (P 2, chains p and p + 2) or 4 m + p (P 4), m = 0..31, in s[m *
  //  GDN_NC + c]; chain c's values in order of i, as gdn_recur's)
  __global const float* sp = S + ((size_t)h * 128 + j) * 128 + p;
  float s[GDN_NI];
#pragma unroll
  for (int m = 0; m < 32; ++m) {
    s[m * GDN_NC] = sp[4 * m];
#if GDN_P == 2
    s[m * GDN_NC + 1] = sp[4 * m + 2];
#endif
  }
  // (the next token's decay, beta and v requested a token ahead, as gdn_recur's)
  float dn = 0.0f, bn = 0.0f, vn = 0.0f;
  if (M_valid > 0) dn = ab[h], bn = ab[nv + h], vn = x[vo];
  for (int t = 0; t < M_valid; ++t) {
    __global const float* row = x + (size_t)t * C;
    const float decay = GDNS_DECAY(dn), beta = GDNS_BETA(bn), v = vn;
    if (t + 1 < M_valid) {
      dn = ab[(size_t)(t + 1) * 2 * nv + h];
      bn = ab[(size_t)(t + 1) * 2 * nv + nv + h];
      vn = x[(size_t)(t + 1) * C + vo];
    }
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
      s[m * GDN_NC] *= decay;
      a0 = fma(s[m * GDN_NC], row[ko + 4 * m], a0);
#if GDN_P == 2
      s[m * GDN_NC + 1] *= decay;
      a1 = fma(s[m * GDN_NC + 1], row[ko + 4 * m + 2], a1);
#endif
    }
    const float d = (v - gdns_join(a0, a1, base)) * beta;
    float b0 = 0.0f, b1 = 0.0f;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
      s[m * GDN_NC] = fma(row[ko + 4 * m], d, s[m * GDN_NC]);  // (k read again: L1; kept, it spilled at 128 registers)
      b0 = fma(s[m * GDN_NC], row[qo + 4 * m], b0);
#if GDN_P == 2
      s[m * GDN_NC + 1] = fma(row[ko + 4 * m + 2], d, s[m * GDN_NC + 1]);
      b1 = fma(s[m * GDN_NC + 1], row[qo + 4 * m + 2], b1);
#endif
    }
    const float ob = gdns_join(b0, b1, base);
    if (p == 0) o[(size_t)t * nv * 128 + h * 128 + j] = ob;
  }
  __global float* so = Sout + ((size_t)h * 128 + j) * 128 + p;
#pragma unroll
  for (int m = 0; m < 32; ++m) {
    so[4 * m] = s[m * GDN_NC];
#if GDN_P == 2
    so[4 * m + 2] = s[m * GDN_NC + 1];
#endif
  }
}
