// attn_x8: the prompt attention as flash attention on 8-lane XMX (Alchemist A770 / A750, the Arc 140T; QUEUE A6 / T4,
// 2026-11-20; zero/xe2/engine_xe2.h attnfx_init). The 8-lane form of attn_fx.cl (16 lanes, Battlemage): the same int8
// K / V cache, scales, int8 q and f32 output as attn_q8 (kernels/ops_attn.cl), the same online softmax and the same P
// quantization per 32 keys as attn_fx, so the integer sums are attn_fx's and only the order of the softmax's sum of P
// differs (not bit-identical to attn_q8: checked at start within a tolerance, KL in docs/before-after.md).
//
// 8-lane int8 DPAS: 8 rows x 32 k x 8 columns; A: component r = row r, lane l = k 4l..4l+3; B: lane n = column n, its
// 8 words = k 0..31; result: component r = row r, lane n = column n. A sub-group owns 8 query tokens of one q head.
//
//   scores, transposed  S^T[8 keys][8 tokens] += K[8 keys][32 dims] . Q^T[32 dims][8 tokens]
//     K is the A operand: lane l loads word (32-dim slice kb, word l) of 16 keys at once (the cache's [word][16 keys]
//     layout gives 64 contiguous bytes a lane, full cache lines; a B operand of keys would need 8 gathers a DPAS), so
//     4 DPAS per slice cover the span's 32 keys (component r of DPAS t = key 8t + r). Q^T is the B operand: lane n =
//     token n's 32 dims, held in registers.
//     After them lane n holds all 32 scores of its token: the softmax's max and sums are in-lane, no reductions.
//   P . V               O[8 tokens][8 dims] += P[8 tokens][32 keys] . V[32 keys][8 dims]
//     P^T packed per lane (word j = keys 4j..4j+3, 4 bytes from one score vector, no shuffles), then one 8 x 8 word
//     transpose (12 xor shuffles) gives P's A layout; V is the B operand straight from the cache (lane n = dim n:
//     uint4 of block 0, uint4 of block 1, the bytes attn_fx reads). Per-token values (the rescale, P's scale, 1 / sum)
//     are lane values used as row values by broadcasts (operand regions, no instructions).
//
// Registers (256 a thread, 32 bytes each): 8 tokens x 256 dims of f32 output is 256 registers, so a head of more than
// 128 dims is split over DS = 2 sub-groups (128 dims each: 128 registers of output, 32 of q): each adds the two halves'
// integer partial scores through SLM (exact; two barriers a span), then both run the same softmax on the same sums
// (the same P bit for bit) and P . V for their own dims. With DS = 2 the work-group runs in step: every sub-group runs
// the spans of the work-group's last token (later spans of an earlier token group are masked: nothing added).
// -D: HD (head dims, multiple of 32, <= 256), HPG (q heads per KV head), NT (token groups of 8 per work-group), NH (q
// heads per work-group, divides HPG), X8_PF (the L1 prefetch of the next span), X8_EMU (the DPAS as DP4a dot products,
// for GPUs without XMX: tests). grid (ceil(M / (8 NT)) * 8 NT NH DS, nkv * HPG / NH), local (8 NT NH DS, 1).
// Opt-in forms (2026-11-22, engine_xe2.h ARC_ATTN_X8_PSH=1 sets PSH and LEAN), all bit for bit the default: X8_PSH (DS 2:
// the pair's first sub-group sums the scores over all HD dims and makes the softmax once, P and the per-token factors
// to the second through local memory, one barrier a span; ocloc dg2-g10 had each of the two make the same softmax),
// X8_LEAN (the span's blocks as ring counters, 32-bit offsets), X8_FULL (each sub-group sums all dims itself, no
// exchange; spills on dg2 unrolled), X8_SUNROLL (the score loop's unroll, 1 with PSH: 8 spilled).
// Plain causal prompts only: no window, no non-causal span, no sparse mask (attn_q8 keeps those).

#define X8_RT 8          // query tokens per sub-group (DPAS columns of the scores, rows of P . V)
#if HD > 128
#define DS 2             // sub-groups sharing a token group's dims
#else
#define DS 1
#endif
#define DH (HD / DS)     // dims per sub-group
#define KBH (DH / 32)    // 32-dim slices of q per sub-group
#define DGH (DH / 8)     // 8-dim groups of the output per sub-group
#define X8_BU (4 * HD)   // uints in one 16-key block of K (and of V)
#define NSG (NT * NH * DS)
#if DS == 1
#undef X8_PSH  // (one sub-group a head: nothing to share)
#endif
#if HD % 32 || HD > 256 || DH % 32 || HPG % NH
#error "attn_x8: HD a multiple of 32 up to 256, NH dividing HPG"
#endif

#ifdef X8_EMU
// the DPAS as the extension's text gives it: result row r, column n = sum over lanes l of A row r's word at lane l
// dotted with B's word l at lane n
inline int x8e_ss(uint a, int8 b, int acc) {
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 0), (uint)b.s0);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 1), (uint)b.s1);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 2), (uint)b.s2);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 3), (uint)b.s3);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 4), (uint)b.s4);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 5), (uint)b.s5);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 6), (uint)b.s6);
  acc += dot_4x8packed_ss_int(intel_sub_group_shuffle(a, 7), (uint)b.s7);
  return acc;
}
inline int x8e_us(uint a, int8 b, int acc) {
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 0), (uint)b.s0);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 1), (uint)b.s1);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 2), (uint)b.s2);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 3), (uint)b.s3);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 4), (uint)b.s4);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 5), (uint)b.s5);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 6), (uint)b.s6);
  acc += dot_4x8packed_us_int(intel_sub_group_shuffle(a, 7), (uint)b.s7);
  return acc;
}
inline int8 x8_ss(int8 a, int8 b, int8 c) {
  return (int8)(x8e_ss((uint)a.s0, b, c.s0), x8e_ss((uint)a.s1, b, c.s1), x8e_ss((uint)a.s2, b, c.s2), x8e_ss((uint)a.s3, b, c.s3),
                x8e_ss((uint)a.s4, b, c.s4), x8e_ss((uint)a.s5, b, c.s5), x8e_ss((uint)a.s6, b, c.s6), x8e_ss((uint)a.s7, b, c.s7));
}
inline int8 x8_us(uint8 a, int8 b, int8 c) {
  return (int8)(x8e_us(a.s0, b, c.s0), x8e_us(a.s1, b, c.s1), x8e_us(a.s2, b, c.s2), x8e_us(a.s3, b, c.s3),
                x8e_us(a.s4, b, c.s4), x8e_us(a.s5, b, c.s5), x8e_us(a.s6, b, c.s6), x8e_us(a.s7, b, c.s7));
}
#else
#define x8_ss(a, b, c) intel_sub_group_i8_i8_matrix_mad_k32((a), (b), (c))
#define x8_us(a, b, c) intel_sub_group_u8_i8_matrix_mad_k32((a), (b), (c))
#endif

// a lane value as a row value: component r = lane r's (operand regions, no instructions)
inline float8 x8_rows(float v) {
  return (float8)(sub_group_broadcast(v, 0), sub_group_broadcast(v, 1), sub_group_broadcast(v, 2), sub_group_broadcast(v, 3),
                  sub_group_broadcast(v, 4), sub_group_broadcast(v, 5), sub_group_broadcast(v, 6), sub_group_broadcast(v, 7));
}

// 8 x 8 words transposed over the 8 lanes: in, lane n component j = M[n][j]; out, lane n component j = M[j][n]. Three
// butterfly stages (bit k of the lane and of the component swapped), 4 xor shuffles and 12 selects each
// (named components, no array: an array here went to private memory on dg2)
#define X8_SW(a, b, k, hi)                                         \
  {                                                                \
    const uint rcv = intel_sub_group_shuffle_xor(hi ? a : b, k);   \
    a = hi ? rcv : a;                                              \
    b = hi ? b : rcv;                                              \
  }
inline uint8 x8_tr(uint8 x) {
  const uint l = get_sub_group_local_id();
  uint r0 = x.s0, r1 = x.s1, r2 = x.s2, r3 = x.s3, r4 = x.s4, r5 = x.s5, r6 = x.s6, r7 = x.s7;
  const bool h4 = (l & 4) != 0, h2 = (l & 2) != 0, h1 = (l & 1) != 0;
  X8_SW(r0, r4, 4, h4) X8_SW(r1, r5, 4, h4) X8_SW(r2, r6, 4, h4) X8_SW(r3, r7, 4, h4)
  X8_SW(r0, r2, 2, h2) X8_SW(r1, r3, 2, h2) X8_SW(r4, r6, 2, h2) X8_SW(r5, r7, 2, h2)
  X8_SW(r0, r1, 1, h1) X8_SW(r2, r3, 1, h1) X8_SW(r4, r5, 1, h1) X8_SW(r6, r7, 1, h1)
  return (uint8)(r0, r1, r2, r3, r4, r5, r6, r7);
}
#undef X8_SW

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void attn_x8(__global const char* qi, __global const float* qsc, __global const uint* kq,
                      __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                      int nh, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi) {
  (void)win; (void)nc_lo; (void)nc_hi;  // (attn_q8's arguments; the engine sends only plain causal prompts here)
  const int lid = get_local_id(0), sg = lid / 8, lane = lid % 8;
  const int dh = sg % DS, tg = (sg / DS) % NT, hh = sg / (DS * NT);
  const int wt = (int)get_num_groups(0) - 1 - (int)get_group_id(0);  // (the latest tokens, the most keys, first)
  const int g = (int)get_group_id(1) / (HPG / NH), h = g * HPG + ((int)get_group_id(1) % (HPG / NH)) * NH + hh;
  const int mw = wt * NT * X8_RT;  // the work-group's first token
  if (mw >= M_valid) return;       // (the whole work-group)
  const int m0 = mw + tg * X8_RT;
  const int nrow = clamp(M_valid - m0, 0, X8_RT);
#if DS > 1 && !defined(X8_FULL)
  // (in step: every sub-group runs the spans of the work-group's last token, a token group past M_valid included)
  const int plw = pos0 + min(M_valid, mw + NT * X8_RT) - 1;
#else
  if (nrow == 0) return;
  const int plw = pos0 + m0 + nrow - 1;
#endif
  const int pfirst = pos0 + m0, plast = pos0 + m0 + nrow - 1;
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;

  // this lane's token (q as the scores' B operand: its DH dims of this sub-group's half, 4 per word). A lane past
  // M_valid reads the last token's q with a zero scale and sees no key (a token's scores, P and output are its own
  // lane's and row's: nothing reaches the other tokens)
  const bool act = lane < nrow;
  const int tok = min(m0 + lane, M_valid - 1);
  __global const uint* qp = (__global const uint*)(qi + ((size_t)tok * nh + h) * HD + dh * DH);
#if defined(X8_FULL) || defined(X8_PSH)
  // (-DX8_FULL: every sub-group sums the scores over all HD dims itself, so the two halves need no exchange through
  //  local memory and no barriers; the integer sums are exact, so the same scores bit for bit)
  // (-DX8_PSH: the pair's first sub-group alone sums them over all HD dims, makes the softmax and P, and hands P and
  //  the per-token factors to the second through local memory, below)
  __global const uint* qpf = (__global const uint*)(qi + ((size_t)tok * nh + h) * HD);
#define X8_KBS (HD / 32)
#define X8_KOFF 0
#else
#define X8_KBS KBH
#define X8_KOFF (dh * DH * 4)
#endif
#ifdef X8_QH
  // (q held in registers, 32 of them: dg2 spills with it; ARC_ATTN_FX_DEFS=-DX8_QH)
  uint8 qb[KBH];
#pragma unroll
  for (int kb = 0; kb < KBH; ++kb) qb[kb] = vload8(kb, qp);
#endif
  const float qs = act ? qsc[(size_t)tok * nh + h] * M_LOG2E_F : 0.0f;  // (log2 domain, as attn_q8)
  const int pos = act ? pfirst + lane : -1;

  float8 o[DGH];
#pragma unroll
  for (int i = 0; i < DGH; ++i) o[i] = (float8)0.0f;
  float mx = -INFINITY;
  float lp = 0.0f;  // (the softmax's sum)
  const int bl = plw / 16;    // the last block any row of the work-group sees
  const int nsp = plw / 32 + 1;
  const int8 kr = (int8)(0, 1, 2, 3, 4, 5, 6, 7);
#if defined(X8_PSH)
  // (a pair's P: 8 words a lane, then its tokens' rescale and P scale; two slots, a span's and the next one's, so one
  //  barrier a span: the second has read slot s % 2 before it reaches the barrier after which the first writes it
  //  again, at span s + 2)
  __local uint psh[NSG / 2][2][80];
  __local float plp[NSG / 2][8];
  const bool prod = dh == 0;
#elif DS > 1 && !defined(X8_FULL)
  __local int xs[NSG * 4 * 64];  // the integer partial scores, a sub-group's 4 x 8 words a lane
  __local int* xmine = xs + sg * 256;
  __local const int* xpart = xs + (sg ^ 1) * 256;
#endif

#ifdef X8_LEAN
  // (-DX8_LEAN: the span's blocks kept as ring indices stepped by 2 instead of b % nbc each span, and 32-bit offsets
  //  from the KV head's first block: ocloc dg2-g10 counted most of the span loop's integer work in those divisions
  //  and 64-bit products. The same blocks, the same addresses: bit for bit)
  __global const uint* kh = kq + blk0 * X8_BU;
  __global const uint4* vh = vq + blk0 * (X8_BU / 4);
  __global const float* ksh = ksc + blk0 * 16;
  __global const float* vsh = vsc + blk0 * 16;
  int r0 = 0;  // (2 s % nbc)
#endif
  for (int s = 0; s < nsp; ++s) {
    const int k0 = 32 * s, b0 = 2 * s, b1 = min(2 * s + 1, bl);  // (b1 past the last block: its keys are all masked)
    const bool b1ok = b1 == 2 * s + 1;
#ifdef X8_LEAN
    (void)b0;
    const int r1 = b1ok ? (r0 + 1 == nbc ? 0 : r0 + 1) : r0;  // (b1 % nbc: b1 is 2 s + 1 or, past the last block, 2 s)
    const uint q0 = (uint)r0, q1 = (uint)r1;
#define X8_KP(q) (kh + (q) * (uint)X8_BU + X8_KOFF)
#define X8_VP(q) (vh + (q) * (uint)(X8_BU / 4) + dh * DH)
#define X8_KS(q, i) ksh[(q) * 16u + (i)]
#define X8_VS(q, i) vsh[(q) * 16u + (i)]
#else
    const size_t o0 = blk0 + (size_t)(b0 % nbc), o1 = blk0 + (size_t)(b1 % nbc);
#define q0 o0
#define q1 o1
#define X8_KP(q) (kq + (q) * X8_BU + X8_KOFF)
#define X8_VP(q) (vq + (q) * (X8_BU / 4) + dh * DH)
#define X8_KS(q, i) ksc[(q) * 16 + (i)]
#define X8_VS(q, i) vsc[(q) * 16 + (i)]
#endif
    // (this half's words: word w of a key at w * 16 + key, the half's first word DH / 4 x half; X8_FULL: all words)
    __global const uint* kp0 = X8_KP(q0);
    __global const uint* kp1 = X8_KP(q1);
#if defined(X8_PF) && defined(cl_intel_subgroup_buffer_prefetch)
    // the next span's K and V blocks (this half's: 4 x DH / 16 pieces of 256 B) into L1, shared by the half's sub-groups
    if (s + 1 < nsp) {
      const int nb0 = 2 * s + 2, nb1 = min(2 * s + 3, bl), ppb = DH / 16;
      for (int p = sg / DS; p < 4 * ppb; p += NSG / DS) {
        const int which = p / ppb;
        const size_t ob = blk0 + (size_t)(((which & 1) ? nb1 : nb0) % nbc);
        __global const uint* base = (which & 2) ? (__global const uint*)vq : kq;
        intel_sub_group_block_prefetch_ui8(base + ob * X8_BU + dh * DH * 4 + (p % ppb) * 64);
      }
    }
#endif
#ifdef X8_PSH
    uint8 pa;
    float corr, ps;
    if (prod) {
#endif
    // scores: int8 DPAS sums, DPAS t = keys 8t .. 8t + 7 of the span (t 0, 1 block b0; 2, 3 block b1)
    int8 a0 = (int8)0, a1 = (int8)0, a2 = (int8)0, a3 = (int8)0;
#ifndef X8_SUNROLL
#ifdef X8_PSH
#define X8_SUNROLL 1  // (all HD dims: unrolled, dg2-g10 spills)
#else
#define X8_SUNROLL 8
#endif
#endif
    __attribute__((opencl_unroll_hint(X8_SUNROLL)))
    for (int kb = 0; kb < X8_KBS; ++kb) {
#if defined(X8_FULL) || defined(X8_PSH)
      const int8 qv = as_int8(vload8(kb, qpf));
#elif defined(X8_QH)
      const int8 qv = as_int8(qb[kb]);
#else
      const int8 qv = as_int8(vload8(kb, qp));  // (read again every span, from L1)
#endif
      const uint16 kv0 = vload16(0, kp0 + (kb * 8 + lane) * 16), kv1 = vload16(0, kp1 + (kb * 8 + lane) * 16);
      a0 = x8_ss(as_int8(kv0.lo), qv, a0);
      a1 = x8_ss(as_int8(kv0.hi), qv, a1);
      a2 = x8_ss(as_int8(kv1.lo), qv, a2);
      a3 = x8_ss(as_int8(kv1.hi), qv, a3);
    }
#if DS > 1 && !defined(X8_FULL) && !defined(X8_PSH)
    // the two halves' sums added (exact; the same sums in both, so the same softmax and P)
    if (s) barrier(CLK_LOCAL_MEM_FENCE);  // (the partner has read the last span's)
    vstore8(a0, lane, xmine);
    vstore8(a1, lane + 8, xmine);
    vstore8(a2, lane + 16, xmine);
    vstore8(a3, lane + 24, xmine);
    barrier(CLK_LOCAL_MEM_FENCE);
    a0 += vload8(lane, xpart);
    a1 += vload8(lane + 8, xpart);
    a2 += vload8(lane + 16, xpart);
    a3 += vload8(lane + 24, xpart);
#endif
    // q scale x k scale (log2 domain); component r of DPAS t = key k0 + 8t + r, the same for every lane
    const float ksl0 = X8_KS(q0, lane), ksl1 = X8_KS(q0, 8 + lane), ksl2 = X8_KS(q1, lane), ksl3 = X8_KS(q1, 8 + lane);
    float8 s0 = convert_float8(a0) * qs * x8_rows(ksl0), s1 = convert_float8(a1) * qs * x8_rows(ksl1),
           s2 = convert_float8(a2) * qs * x8_rows(ksl2), s3 = convert_float8(a3) * qs * x8_rows(ksl3);
    if (k0 + 31 > pfirst || !b1ok) {  // (the causal edge, or a missing second block: mask per token)
      // (key k0 + 8t + r <= pos: r + 8t against pos - k0, the left side constants)
      const int dk = pos - k0, dk1 = b1ok ? dk : -1;
      s0 = select((float8)(-INFINITY), s0, kr <= (int8)(dk));
      s1 = select((float8)(-INFINITY), s1, kr + 8 <= (int8)(dk));
      s2 = select((float8)(-INFINITY), s2, kr + 16 <= (int8)(dk1));
      s3 = select((float8)(-INFINITY), s3, kr + 24 <= (int8)(dk1));
    }
    // online softmax step per token (in-lane); the rescale folded into the P . V update (attn_fx's form)
    const float8 m8 = fmax(fmax(s0, s1), fmax(s2, s3));
    const float tm = fmax(fmax(fmax(m8.s0, m8.s1), fmax(m8.s2, m8.s3)), fmax(fmax(m8.s4, m8.s5), fmax(m8.s6, m8.s7)));
    const float mn = fmax(mx, tm);
#ifndef X8_PSH
    const float
#endif
    corr = isgreater(tm, mx) ? native_exp2(mx - mn) : 1.0f;  // (mx = -inf: 0)
    lp *= corr;
    mx = mn;
    const float ms = isinf(mx) ? 0.0f : mx;  // (a token that saw no key yet: its scores are -inf too)
    // P x V scale quantized to 0..255 against the token's bound over the span (attn_fx's: span max weight x the
    // largest V scale of the keys up to the sub-group's last token); the bound known first, so each score vector
    // turns into its bytes at once (scores and P never in registers together: dg2 spilled)
    const float pb = native_exp2(tm - ms);
    const float vsl0 = k0 + lane <= plast ? X8_VS(q0, lane) : 0.0f;
    const float vsl1 = k0 + 8 + lane <= plast ? X8_VS(q0, 8 + lane) : 0.0f;
    const float vsl2 = b1ok && k0 + 16 + lane <= plast ? X8_VS(q1, lane) : 0.0f;
    const float vsl3 = b1ok && k0 + 24 + lane <= plast ? X8_VS(q1, 8 + lane) : 0.0f;
    const float vmt = sub_group_reduce_max(fmax(fmax(vsl0, vsl1), fmax(vsl2, vsl3)));
    const float pmv = pb * vmt;
    const float inv = isgreater(pmv, 0.0f) ? 255.0f * native_recip(pmv) : 0.0f;
    // (min(rte(x), 255) of attn_fx for x >= 0)
#define X8_P(t)                                                \
  const float8 p##t = native_exp2(s##t - ms);                  \
  lp += ((p##t.s0 + p##t.s1) + (p##t.s2 + p##t.s3)) + ((p##t.s4 + p##t.s5) + (p##t.s6 + p##t.s7)); \
  const uchar8 u##t = convert_uchar8_sat_rte(p##t * x8_rows(vsl##t) * inv);
    X8_P(0) X8_P(1) X8_P(2) X8_P(3)
#undef X8_P
#ifndef X8_PSH
    const float
#endif
    ps = pmv * (1.0f / 255.0f);
    // P^T (lane = token, word j = keys 4j .. 4j + 3) -> P as the A operand (component = token, lane l = word l)
#ifndef X8_PSH
    const uint8
#endif
    pa = x8_tr((uint8)(as_uint(u0.lo), as_uint(u0.hi), as_uint(u1.lo), as_uint(u1.hi), as_uint(u2.lo),
                                   as_uint(u2.hi), as_uint(u3.lo), as_uint(u3.hi)));
#ifdef X8_PSH
      __local uint* sl = psh[sg / 2][s & 1];
      vstore8(pa, lane, sl);
      sl[64 + lane] = as_uint(corr);
      sl[72 + lane] = as_uint(ps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!prod) {
      __local const uint* sl = psh[sg / 2][s & 1];
      pa = vload8(lane, sl);
      corr = as_float(sl[64 + lane]);
      ps = as_float(sl[72 + lane]);
    }
#endif
    // P . V: lane = dim 8 i + lane of this half, the 32 keys' bytes of that dim from the two blocks
    __global const uint4* vp0 = X8_VP(q0);
    __global const uint4* vp1 = X8_VP(q1);
    const float8 cr = x8_rows(corr), pr = x8_rows(ps);
#ifdef X8_LAZY
    // (-DX8_LAZY, a prototype, not bit for bit: where no token's maximum moved this span (every rescale 1), the update
    //  as one fused multiply-add, o + acc x pr rounded once instead of twice; a KL question)
    if (sub_group_all(corr == 1.0f)) {
#pragma unroll
      for (int i = 0; i < DGH; ++i) {
        const uint4 v0 = vp0[8 * i + lane], v1 = vp1[8 * i + lane];
        const int8 acc = x8_us(pa, as_int8((uint8)(v0, v1)), (int8)0);
        o[i] = fma(convert_float8(acc), pr, o[i]);
      }
    } else
#endif
#pragma unroll
    for (int i = 0; i < DGH; ++i) {
      const uint4 v0 = vp0[8 * i + lane], v1 = vp1[8 * i + lane];
      const int8 acc = x8_us(pa, as_int8((uint8)(v0, v1)), (int8)0);
      o[i] = fma(o[i], cr, convert_float8(acc) * pr);
    }
#ifdef X8_LEAN
    r0 += 2;
    while (r0 >= nbc) r0 -= nbc;
#else
#undef q0
#undef q1
#endif
#undef X8_KP
#undef X8_VP
#undef X8_KS
#undef X8_VS
  }

#ifdef X8_PSH
  // (the softmax's sum is the first sub-group's)
  if (prod) plp[sg / 2][lane] = lp;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!prod) lp = plp[sg / 2][lane];
#endif
  const float8 ir = x8_rows(1.0f / lp);
#define X8_WR(r, c)                                                                         \
  if (r < nrow) {                                                                           \
    __global float* dst = out + ((size_t)(m0 + r) * nh + h) * HD + dh * DH + lane;        \
    _Pragma("unroll") for (int i = 0; i < DGH; ++i) dst[8 * i] = o[i].c * ir.c;             \
  }
  X8_WR(0, s0) X8_WR(1, s1) X8_WR(2, s2) X8_WR(3, s3) X8_WR(4, s4) X8_WR(5, s5) X8_WR(6, s6) X8_WR(7, s7)
#undef X8_WR
}
