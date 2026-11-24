// Multi-GPU (tensor parallel, zero/mgpu/): the split model's own kernels, apart from the one-GPU kernels.

// x[i] += part 0 + part 1 + ... (each GPU's partial sum of an output projection, in pinned host memory), added in
// rank order so every GPU's residual stream gets the same float sum
__kernel void tp_add(__global float* x, __global const float* p, ulong off, int n, int np, ulong stride) {
  // The parts lie `stride` floats apart (a 4096-token chunk of the 27B: 80 MB), so every k is its own page range and
  // a generated token's 128 sums read 4 far-apart regions of n_embd values each. A thread takes four values: the
  // same sums in the same rank order, four loads in flight where one was (the scalar form read 5 GB/s a sum, 15.5 us
  // of a 35 ms step on 4 B60s). n is rows * n_embd, a multiple of 4 for every model.
  const int i0 = get_global_id(0) * 4;
  if (i0 + 3 < n) {
    float4 s = vload4(0, x + i0);
    for (int k = 0; k < np; ++k) s += vload4(0, p + off + k * stride + i0);
    vstore4(s, 0, x + i0);
    return;
  }
  for (int i = i0; i < n; ++i) {  // (a tail of 1-3 values, never reached: n_embd is a multiple of 4)
    float s = x[i];
    for (int k = 0; k < np; ++k) s += p[off + k * stride + i];
    x[i] = s;
  }
}

// Prompt chunks (zero/mgpu/engine_tp.h, tp_reduce_rs): a reduce-scatter in f16. Each GPU sends the other GPUs' rows of
// its part, sums its own rows over every part, sends that sum, and adds the others' sums. Half the bytes of f32 and
// each part's row read once in all, not once per GPU. (half pointers: vload_half / vstore_half need no cl_khr_fp16)

// f32 -> f16, rounded to nearest even. A value beyond lim (65504 / GPUs: the sum of every GPU's part must fit f16 too)
// or not a number sets *flag: then every GPU sums this projection in f32 instead (tp_reduce_rs)
// (this GPU's own rows [i0, i1) are only checked: tp_rs_sum reads them in f32 and writes its sums over them)
// f32 -> f16, eight values a thread: one value a thread read 115 GB/s of the card's 448 on a 2048-token chunk's sums
// (three B60s), and what is left is transaction count, not arithmetic.
__kernel void tp_half(__global const float* src, __global half* dst, int n, __global int* flag, float lim, int i0, int i1) {
  const int i0_ = get_global_id(0) * 8;
  if (i0_ + 7 < n) {
    const float8 v = vload8(0, src + i0_);
    const float a[8] = {v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7};
    if (!(fabs(a[0]) <= lim) || !(fabs(a[1]) <= lim) || !(fabs(a[2]) <= lim) || !(fabs(a[3]) <= lim) ||
        !(fabs(a[4]) <= lim) || !(fabs(a[5]) <= lim) || !(fabs(a[6]) <= lim) || !(fabs(a[7]) <= lim))
      *flag = 1;
    if (i0_ < i0 || i0_ >= i1) vstore_half8(v, 0, dst + i0_);
    return;
  }
  for (int i = i0_; i < n; ++i) {  // (a tail of fewer than 8: n is rows * n_embd, a multiple of 8 for every model)
    const float v = src[i];
    if (!(fabs(v) <= lim)) *flag = 1;
    if (i < i0 || i >= i1) vstore_half(v, i, dst);
  }
}

// this GPU's values [i0, i1): the parts summed in rank order (its own from its memory in f32, the others' in f16 from
// host memory), the sum rounded to f16 once into res (sent to the others) and that same value added to x, as the
// other GPUs add it: every GPU's residual stream stays identical
__kernel void tp_rs_sum(__global float* x, __global const float* part, __global const half* put, ulong put_off, ulong stride,
                        int np, int me, int i0, int i1, __global half* res) {
  const int i = i0 + get_global_id(0);
  if (i >= i1) return;
  float s = 0.0f;
  for (int k = 0; k < np; ++k) s += k == me ? part[i] : vload_half(put_off + k * stride + i, put);
  vstore_half(s, i, res);
  x[i] += vload_half(i, res);
}

// the other GPUs' sums (every value outside [i0, i1)) added to x
__kernel void tp_rs_add(__global float* x, __global const half* res, ulong res_off, int n, int i0, int i1) {
  const int i = get_global_id(0);
  if (i >= n || (i >= i0 && i < i1)) return;
  x[i] += vload_half(res_off + i, res);
}

// the pipelined reduce-scatter (ARC_TP_PIPE): another GPU's sums of values [j0, j1) added to x, as tp_rs_add does
__kernel void tp_rs_add_r(__global float* x, __global const half* res, ulong res_off, int j0, int j1) {
  const int i = j0 + get_global_id(0);
  if (i >= j1) return;
  x[i] += vload_half(res_off + i, res);
}

// Mixture of experts split by expert (each shard holds experts [e0, e1), global ids everywhere): the routing as in
// one piece, then only this shard's (token, expert) pairs get slots and tiles; the combine sums the shard's experts
// into a partial output that tp_reduce adds over the shards.

// prompt chunks: after moe_route, the counts of other shards' experts zeroed (moe_scan then makes no tiles for them).
// grid (NE)
__kernel void tp_moe_mask(__global int* counts, int ne, int e0, int e1) {
  const int e = get_global_id(0);
  if (e < ne && (e < e0 || e >= e1)) counts[e] = 0;
}

// moe_scatter for this shard's pairs only; the others' inv = -1 (no slot). grid (T * nu)
__kernel void tp_moe_scatter(__global const int* ids, __global const float* wts, int n, __global int* cursor,
                             __global int* stok, __global float* swt, __global int* inv, int nu, int e0, int e1) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const int e = ids[i];
  if (e < e0 || e >= e1) { inv[i] = -1; return; }
  const int s = atomic_inc(&cursor[e]);
  stok[s] = i / nu;
  swt[s] = wts[i];
  inv[i] = s;
}

// one token (moe_route1's tiles: tiles[0] = count, then (expert, slot, rows), slot k = the k-th choice): the tiles
// of this shard's experts moved to the front, in order; the others' inv = -1. One work-item.
__kernel void tp_moe_tiles1(__global int* tiles, __global int* inv, int e0, int e1) {
  if (get_global_id(0)) return;
  const int n = tiles[0];
  int m = 0;
  for (int k = 0; k < n; ++k) {
    const int e = tiles[1 + 3 * k], s = tiles[2 + 3 * k], r = tiles[3 + 3 * k];
    if (e >= e0 && e < e1) {
      tiles[1 + 3 * m] = e;
      tiles[2 + 3 * m] = s;
      tiles[3 + 3 * m] = r;
      ++m;
    } else {
      inv[s] = -1;
    }
  }
  tiles[0] = m;
}

// moe_combine into a partial output: out = the shard's experts (y at the slots, in choice order) + the gated shared
// expert on the shard that adds it (shared != 0). grid (N, T)
__kernel void tp_moe_combine(__global float* out, __global const float* sh, __global const float* sgate,
                             __global const float* y, __global const int* inv, int N, int nu, int shared) {
  const int c = get_global_id(0), t = get_global_id(1);
  if (c >= N) return;
  float v = 0.0f;
  for (int k = 0; k < nu; ++k) {
    const int s = inv[t * nu + k];
    if (s >= 0) v += y[(size_t)s * N + c];
  }
  out[(size_t)t * N + c] = shared ? v + sgate[t] * sh[(size_t)t * N + c] : v;
}

// rows of `ld` values (a GEMM's output padded to its tiles of 16) packed to rows of n
__kernel void tp_pack(__global const float* src, int ld, __global float* dst, int n, int rows) {
  const int c = get_global_id(0), r = get_global_id(1);
  if (c < n && r < rows) dst[(size_t)r * n + c] = src[(size_t)r * ld + c];
}
