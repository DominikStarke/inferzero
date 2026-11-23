// Inkling 2 (inkling2 / inkling2_loop): the operations no other model has, as llama.cpp's graph computes them
// (src/models/inkling2.cpp; the f32 reference: tools/ref_inkling2.h). f32 throughout; the K / V cache in f16. Written
// for correctness first (docs/inkling2.md, I1.5): the speed work (I2 / I3) comes after the KL checks.
//
// Work-group sizes: every kernel here with local memory takes an explicit local size, INK_LS work-items (a power of
// two, at most 256; ink_attn: exactly INK_D), and every barrier is reached by all work-items in the same control flow
// (loop counts uniform across the group; AGENTS.md, docs/lessons-intel-arc-igpu.md 2026-11-23). The element-wise
// kernels (ink_ctrl_apply, ink_conv, ink_conv_state, ink_kv_store) take any size.

#ifndef INK_LS
#define INK_LS 256
#endif
#ifndef INK_D
#define INK_D 128  // (the head size: Inkling 2 Mini's 128)
#endif
#ifndef INK_MAXS
#define INK_MAXS 16  // (states a mix may hold: the blocks + the stream)
#endif
#ifndef INK_EXT
#define INK_EXT 1024  // (the longest relative-bias table: Inkling 2 Mini's window and extent, both 1024)
#endif

// sum of v over the work-group (every work-item gets it); red: INK_LS floats of local memory
inline float ink_wg_sum(float v, __local float* red) {
  const int l = get_local_id(0), n = get_local_size(0);
  red[l] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] += red[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}
inline float ink_wg_max(float v, __local float* red) {
  const int l = get_local_id(0), n = get_local_size(0);
  red[l] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] = fmax(red[l], red[l + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}
inline float ink_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// llama's mix (attention over depth), one work-group per token t: the states (nb blocks, block s at blocks + s bstride,
// then cur), each RMS-normed (eps 1e-6, no weight); per query j (q: nq rows of E) a softmax over the states of
// (normed state . q_j) / sqrt(E); out[t][j] = the softmax-weighted sum of the normed states. Rows of E per token.
__kernel void ink_mix(__global const float* blocks, int nb, long bstride, __global const float* cur,
                      __global const float* q, int nq, __global float* out, int E) {
  __local float red[INK_LS];
  __local float rs[INK_MAXS];
  __local float sc[4 * INK_MAXS];
  const int t = get_group_id(0), l = get_local_id(0), n = get_local_size(0), S = nb + 1;
#define INK_STATE(s) ((s) < nb ? blocks + (s) * bstride + (size_t)t * E : cur + (size_t)t * E)
  for (int s = 0; s < S; ++s) {
    __global const float* x = INK_STATE(s);
    float ss = 0.0f;
    for (int i = l; i < E; i += n) ss += x[i] * x[i];
    ss = ink_wg_sum(ss, red);
    if (l == 0) rs[s] = rsqrt(ss / (float)E + 1e-6f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv = rsqrt((float)E);
  for (int j = 0; j < nq; ++j)
    for (int s = 0; s < S; ++s) {
      __global const float* x = INK_STATE(s);
      float d = 0.0f;
      for (int i = l; i < E; i += n) d += x[i] * q[(size_t)j * E + i];
      d = ink_wg_sum(d, red);
      if (l == 0) sc[j * INK_MAXS + s] = d * rs[s] * inv;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l < nq) {  // (the softmax per query, by one work-item each)
    float mx = -INFINITY, z = 0.0f;
    for (int s = 0; s < S; ++s) mx = fmax(mx, sc[l * INK_MAXS + s]);
    for (int s = 0; s < S; ++s) z += sc[l * INK_MAXS + s] = exp(sc[l * INK_MAXS + s] - mx);
    for (int s = 0; s < S; ++s) sc[l * INK_MAXS + s] /= z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int j = 0; j < nq; ++j)
    for (int i = l; i < E; i += n) {
      float acc = 0.0f;
      for (int s = 0; s < S; ++s) acc += sc[j * INK_MAXS + s] * rs[s] * INK_STATE(s)[i];
      out[((size_t)t * nq + j) * E + i] = acc;
    }
#undef INK_STATE
}

// the controller's input, one work-group per token: g[t] = [rms(x[t]), rms(m[t])] (eps 1e-6, no weight), 2E values;
// x and m rows at xstride / mstride floats, m from element moff (one message of a [T][4][E] mix output: moff c E)
__kernel void ink_rms2(__global const float* x, int xstride, __global const float* m, int moff, int mstride, __global float* g,
                       int E) {
  __local float red[INK_LS];
  const int t = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  __global const float* a = x + (size_t)t * xstride;
  __global const float* b = m + moff + (size_t)t * mstride;
  float sa = 0.0f, sb = 0.0f;
  for (int i = l; i < E; i += n) { sa += a[i] * a[i]; sb += b[i] * b[i]; }
  const float ra = rsqrt(ink_wg_sum(sa, red) / (float)E + 1e-6f), rb = rsqrt(ink_wg_sum(sb, red) / (float)E + 1e-6f);
  __global float* o = g + (size_t)t * 2 * E;
  for (int i = l; i < E; i += n) { o[i] = a[i] * ra; o[E + i] = b[i] * rb; }
}

// the controller's result, element-wise (any size): x = x - tanh(f) x + tanh(gn) m, for T x E (m at mstride floats)
__kernel void ink_ctrl_apply(__global float* x, __global const float* f, __global const float* gn, __global const float* m,
                             int moff, int mstride, int E, int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  const int t = i / E, e = i % E;
  const float v = x[i];
  x[i] = v - tanh(f[i]) * v + tanh(gn[i]) * m[moff + (size_t)t * mstride + e];
}

// llama's convolve on a chunk of T tokens after the slot's history (any size; one work-item per value):
// out[t][c] = (cur[t][c] + sum_i w[c][i] x(t - (NC - 1) + i) + [dyn] sum_lag x(t - lag) dyn[t][h(c)][lag]) * scale,
// x(tt) the chunk's input for tt >= 0, else the slot's state (the NC - 1 inputs before the chunk, oldest first, per
// channel). dyn: T x (nh NC) or none (has_dyn 0). Call ink_conv_state afterwards (this reads the old state).
__kernel void ink_conv(__global const float* cur, __global const float* w, __global const float* dyn, int has_dyn, int nh,
                       __global const float* state, __global float* out, int C, int T, int NC, float scale) {
  const int i = get_global_id(0);
  if (i >= C * T) return;
  const int t = i / C, c = i % C;
#define INK_X(tt) ((tt) >= 0 ? cur[(size_t)(tt) * C + c] : state[(size_t)c * (NC - 1) + (NC - 1) + (tt)])
  float acc = 0.0f;
  for (int k = 0; k < NC; ++k) acc += w[(size_t)c * NC + k] * INK_X(t - (NC - 1) + k);
  if (has_dyn) {
    __global const float* dt = dyn + (size_t)t * nh * NC + (size_t)(c / (C / nh)) * NC;
    for (int lag = 0; lag < NC; ++lag) acc += INK_X(t - lag) * dt[lag];
  }
  out[i] = (cur[i] + acc) * scale;
#undef INK_X
}

// the slot's conv state after a chunk: the last NC - 1 inputs, oldest first (one work-item per channel; in place,
// ascending, reads only entries not yet written)
__kernel void ink_conv_state(__global const float* cur, __global float* state, int C, int T, int NC) {
  const int c = get_global_id(0);
  if (c >= C) return;
  __global float* s = state + (size_t)c * (NC - 1);
  for (int k = 0; k < NC - 1; ++k) {
    const int tt = T - (NC - 1) + k;  // (the input this entry now holds)
    s[k] = tt >= 0 ? cur[(size_t)tt * C + c] : s[k + T];
  }
}

// llama's clip on rows of n values (one work-group per row; rows: T x nrow, n <= INK_LS): times the row's scale
// (scale: nrow rows of n, or has_scale 0), scaled down to an L2 norm of at most thr (factor clamp(thr / (norm + 1e-6),
// 0, 1)); then, when use_tau (q and r of the global layers, whatever log_alpha's sign), times
// tau = 1 + log_alpha log(max(1, (pos0 + t + 1) / log_floor)). mode 1: an RMS norm instead (V: eps, no weight)
__kernel void ink_rows(__global float* x, int nrow, int n, __global const float* scale, int has_scale, float thr,
                       int use_tau, float log_alpha, float log_floor, int pos0, int mode, float eps) {
  __local float red[INK_LS];
  const int g = get_group_id(0), l = get_local_id(0), t = g / nrow, r = g % nrow;
  __global float* v = x + (size_t)g * n;
  float a = 0.0f;
  if (l < n) {
    a = v[l];
    if (has_scale) a *= scale[(size_t)r * n + l];
  }
  const float ss = ink_wg_sum(a * a, red);
  float f;
  if (mode == 1) f = rsqrt(ss / (float)n + eps);
  else f = clamp(thr / (sqrt(ss) + 1e-6f), 0.0f, 1.0f);
  if (use_tau) f *= 1.0f + log_alpha * log(fmax(1.0f, (float)(pos0 + t + 1) / log_floor));
  if (l < n) v[l] = a * f;
}

// K and V of a chunk into the slot's f16 cache (any size): row of position pos0 + t at (pos0 + t) % cap, HKV x D each
__kernel void ink_kv_store(__global const float* k, __global const float* v, __global half* kc, __global half* vc, int KG,
                           int T, int pos0, int cap) {
  const int i = get_global_id(0);
  if (i >= KG * T) return;
  const int t = i / KG, e = i % KG;
  const size_t dst = (size_t)((pos0 + t) % cap) * KG + e;
  vstore_half(k[i], dst, kc);
  vstore_half(v[i], dst, vc);
}

// Attention of a chunk's queries over the slot's cache, one work-group of INK_D work-items per (token, head):
// score(u) = q . k_u / sqrt(D) + r . bias[pos - u] - sink, the bias 0 at distances past the table (ext rows of NREL),
// over keys u = lo .. pos (lo = pos - window + 1 for a sliding layer, else 0), at (u % cap) in the cache. The softmax
// has a zero sink logit as well (llama: the sink subtracted from the bias, zero sinks in the attention), so it starts
// from max 0, sum 1. Online over tiles of INK_D keys. out[t][h] = the weighted V sum times sigmoid(gate[t][h]).
// q, gate: T x H x D; r: T x H x NREL; sink: T x H; kc, vc: cap x HKV x D (f16).
__kernel void ink_attn(__global const float* q, __global const float* r, __global const float* sink,
                       __global const float* gate, __global const half* kc, __global const half* vc,
                       __global const float* bias, int ext, int NREL, __global float* out, int H, int HKV, int pos0,
                       int window, int cap) {
  __local float qs[INK_D];
  __local float table[INK_EXT + 1];
  __local float p[INK_D];
  __local float red[INK_D];
  const int g = get_group_id(0), t = g / H, h = g % H, kvh = h / (H / HKV), d = get_local_id(0);
  const int pos = pos0 + t, lo = window > 0 ? max(0, pos - window + 1) : 0;
  const float sk = sink[(size_t)t * H + h];
  qs[d] = q[((size_t)t * H + h) * INK_D + d];
  __global const float* rr = r + ((size_t)t * H + h) * NREL;
  for (int e = d; e <= ext; e += INK_D) {  // (the bias by distance, minus the sink; past the table: 0 - sink)
    float b = 0.0f;
    if (e < ext)
      for (int j = 0; j < NREL; ++j) b += bias[(size_t)e * NREL + j] * rr[j];
    table[e] = b - sk;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv = rsqrt((float)INK_D);
  float m = 0.0f, z = 1.0f, acc = 0.0f;  // (the zero sink logit)
  for (int base = lo; base <= pos; base += INK_D) {
    const int u = base + d;
    float s = -INFINITY;
    if (u <= pos) {
      __global const half* kr = kc + ((size_t)(u % cap) * HKV + kvh) * INK_D;
      float dot = 0.0f;
      for (int i = 0; i < INK_D; ++i) dot += qs[i] * vload_half(i, kr);
      const int dist = pos - u;
      s = dot * inv + table[min(dist, ext)];
    }
    const float mt = ink_wg_max(s, red);
    const float mn = fmax(m, mt), corr = exp(m - mn);
    const float e = u <= pos ? exp(s - mn) : 0.0f;
    p[d] = e;
    z = z * corr + ink_wg_sum(e, red);  // (the barriers in it also publish p)
    acc *= corr;
    const int nk = min(INK_D, pos - base + 1);
    for (int k = 0; k < nk; ++k) acc += p[k] * vload_half(((size_t)((base + k) % cap) * HKV + kvh) * INK_D + d, vc);
    m = mn;
    barrier(CLK_LOCAL_MEM_FENCE);  // (p is rewritten by the next tile)
  }
  const size_t o = ((size_t)t * H + h) * INK_D + d;
  out[o] = acc / z * ink_sigmoid(gate[o]);
}

// situ on a fused gate-up row pair (any size): out[i] = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), g and u the
// expert's outputs times its scale (llama: the scale on the matmul's output, before the activation). gu: rows of 2F
// (gate F, then up F) per (token, slot); sc: the experts' gate-up scales (indexed by ids[row]), or has_scale 0
__kernel void ink_situ(__global const float* gu, __global const int* ids, __global const float* sc, int has_scale,
                       __global float* out, int F, int rows, float beta, float lb) {
  const int i = get_global_id(0);
  if (i >= F * rows) return;
  const int row = i / F, f = i % F;
  const float s = has_scale ? sc[ids[row]] : 1.0f;
  const float g = gu[(size_t)row * 2 * F + f] * s, u = gu[(size_t)row * 2 * F + F + f] * s;
  const float a = beta * tanh(g / beta) * ink_sigmoid(g);
  out[i] = a * (lb > 0.0f ? lb * tanh(u / lb) : u);
}

// The experts on their GGUF rows, read exactly (the correctness path, I1.5; the fast one goes through the engine's MoE
// kernels later): out[row][n] = sum_k W_e[n][k] x[row][k], e = ids[row], one work-item per (row, n) (any size). W: the
// tensor's raw bytes, expert e's row n at (e N + n) rb; type 40 NVFP4 (per 16 exactly, as ggml's dequantize_row_nvfp4,
// no per-32 fold), 30 BF16, 8 Q8_0, 0 F32. kv: kvalues_mxfp4 (16 bytes)
inline float ink_ue4m3(uint x) {
  if (x == 0u || x == 0x7Fu) return 0.0f;
  const uint e = (x >> 3) & 15u, m = x & 7u;
  return e == 0u ? ldexp((float)m, -10) : ldexp((float)(8u + m), (int)e - 11);
}
__kernel void ink_expert_mv(__global const uchar* W, int type, ulong rb, int N, int K, __global const int* ids,
                            __global const float* x, int xdiv, __global float* out, int rows,
                            __global const char* kv) {
  const size_t i = get_global_id(0);
  if (i >= (size_t)rows * N) return;
  const int row = (int)(i / N), n = (int)(i % N);
  __global const uchar* w = W + ((size_t)ids[row] * N + n) * rb;
  __global const float* xr = x + (size_t)(row / xdiv) * K;  // (xdiv: rows sharing an input, a token's NU experts)
  float acc = 0.0f;
  if (type == 40) {
    for (int b = 0; b < K / 64; ++b) {
      __global const uchar* p = w + (size_t)b * 36;
      for (int s = 0; s < 4; ++s) {
        const float d = ink_ue4m3(p[s]);
        float part = 0.0f;
        for (int j = 0; j < 8; ++j) {
          const uint c = p[4 + 8 * s + j];
          part += (float)kv[c & 15u] * xr[64 * b + 16 * s + j] + (float)kv[c >> 4] * xr[64 * b + 16 * s + 8 + j];
        }
        acc += d * part;
      }
    }
  } else if (type == 30) {
    __global const ushort* h = (__global const ushort*)w;
    for (int k = 0; k < K; ++k) acc += as_float((uint)h[k] << 16) * xr[k];
  } else if (type == 8) {
    for (int b = 0; b < K / 32; ++b) {
      __global const uchar* p = w + (size_t)b * 34;
      const float d = vload_half(0, (__global const half*)p);
      float part = 0.0f;
      for (int j = 0; j < 32; ++j) part += (float)(char)p[2 + j] * xr[32 * b + j];
      acc += d * part;
    }
  } else {
    __global const float* f = (__global const float*)w;
    for (int k = 0; k < K; ++k) acc += f[k] * xr[k];
  }
  out[i] = acc;
}

// the weighted sum of a token's experts (any size): y[t][i] = sum_k w[t][k] scale[ids[t][k]] de[t][k][i] (llama: the
// down matmul's output times the expert's scale, then times the router weight); scale: per expert, or has_scale 0
__kernel void ink_expert_sum(__global const float* de, __global const int* ids, __global const float* w,
                             __global const float* scale, int has_scale, __global float* y, int E, int NU, int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  const int t = i / E, e = i % E;
  float acc = 0.0f;
  for (int k = 0; k < NU; ++k) {
    const int id = ids[(size_t)t * NU + k];
    acc += w[(size_t)t * NU + k] * (has_scale ? scale[id] : 1.0f) * de[((size_t)t * NU + k) * E + e];
  }
  y[i] = acc;
}

// the router for one token per work-group (explicit local size INK_LS >= NE): sigmoid of the logits, the selection
// adds the bias, the top NU by selection (ties: the lower index, as a stable descending sort), weights
// prob / (sum + 1e-20) x wscale (wscale 0 or 1: none). ids / w: T x NU
__kernel void ink_route(__global const float* logits, __global const float* bias, int has_bias, int NE, int NU,
                        float wscale, __global int* ids, __global float* w) {
  __local float prob[INK_LS];
  __local float sel[INK_LS];
  const int t = get_group_id(0), l = get_local_id(0);
  if (l < NE) {
    prob[l] = ink_sigmoid(logits[(size_t)t * NE + l]);
    sel[l] = prob[l] + (has_bias ? bias[l] : 0.0f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0) {
    float sum = 0.0f;
    int pick[64];
    for (int k = 0; k < NU && k < 64; ++k) {
      int best = -1;
      for (int e = 0; e < NE; ++e) {
        bool used = false;
        for (int j = 0; j < k; ++j) used = used || pick[j] == e;
        if (!used && (best < 0 || sel[e] > sel[best])) best = e;
      }
      pick[k] = best;
      sum += prob[best];
    }
    for (int k = 0; k < NU && k < 64; ++k) {
      float wk = prob[pick[k]] / (sum + 1e-20f);
      if (wscale != 0.0f && wscale != 1.0f) wk *= wscale;
      ids[(size_t)t * NU + k] = pick[k];
      w[(size_t)t * NU + k] = wk;
    }
  }
}

// ---- small helpers

// RMS norm with a weight, one work-group per row (explicit local size INK_LS): out[t] = x[t] / rms(x[t]) * w (x rows at
// xstride floats; out rows of E; in place when out == x)
__kernel void ink_rmsnorm(__global const float* x, int xstride, __global const float* w, __global float* out, int E,
                          float eps) {
  __local float red[INK_LS];
  const int t = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  __global const float* a = x + (size_t)t * xstride;
  float ss = 0.0f;
  for (int i = l; i < E; i += n) ss += a[i] * a[i];
  const float r = rsqrt(ink_wg_sum(ss, red) / (float)E + eps);
  for (int i = l; i < E; i += n) out[(size_t)t * E + i] = a[i] * r * w[i];
}

// out = a + b, element-wise (any size; in place allowed)
__kernel void ink_add(__global const float* a, __global const float* b, __global float* out, int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}

// out[t] = src[t][j] of a [T][nq][E] mix output (any size)
__kernel void ink_take(__global const float* src, int nq, int j, __global float* out, int E, int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  out[i] = src[((size_t)(i / E) * nq + j) * E + i % E];
}

// y[t][n] = sum_k W[n][k] x[t][k] with an f32 W (any size; one work-item per (t, n)): the matrices whose shapes the
// int8 GEMM does not take (N not a multiple of 16 or K of 256: the sinks' 24 rows, the controllers' rank-64 inputs)
// and the router, kept f32 as llama.cpp keeps it (expert_router_f32)
__kernel void ink_mm_f32(__global const float* W, __global const float* x, __global float* y, int N, int K, int T) {
  const int i = get_global_id(0);
  if (i >= N * T) return;
  const int t = i / N, n = i % N;
  __global const float* w = W + (size_t)n * K;
  __global const float* xr = x + (size_t)t * K;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += w[k] * xr[k];
  y[i] = acc;
}

// the logits past the vocabulary (the head's padding rows, 201,024 for 200,058 tokens) to -inf, so that nothing samples
// them (llama.cpp drops them); rows of nl logits (any size)
__kernel void ink_mask_logits(__global float* logits, int nl, int nv, int rows) {
  const int i = get_global_id(0), pad = nl - nv;
  if (i >= pad * rows) return;
  logits[(size_t)(i / pad) * nl + nv + i % pad] = -INFINITY;
}
