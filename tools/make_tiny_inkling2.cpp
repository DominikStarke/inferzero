// A small Inkling 2 (`inkling2_loop`) GGUF with the real layout and random weights, for building and checking the
// engine's path against tools/ref_forward without the 23 GB model (docs/inkling2.md, step I1.4). After
// tools/make_tiny_deepseek41.cpp: its random numbers, encoders and writer.
//
// The keys and tensor names of unsloth/Inkling-2-Mini-GGUF's mainline Q8_K_XL, smaller sizes, chosen so that a
// prompt of ~40 tokens reaches every case:
//  - 8 layers, attention-residual blocks of 2, the loop core [2, 6) run 3 times (a fixed loop, as the real file's);
//    sliding-window layers and global ones (1, 5 and 7: 5 inside the loop, so a global layer gets messages too);
//  - head size 128 as the real model's (kernels may be written for it), 6 query heads over 2 KV heads (GQA 3, as 24 /
//    8), 16 relative buckets;
//  - a sliding window of 16 and a relative extent of 32, so that distances pass both tables' ends; a log-scale floor of
//    8, so that the global layers' factor moves from position 8;
//  - 8 experts with 2 active, FFN 128; NVFP4 experts with their f32 .scale tensors in layers 0-5 and BF16 ones in 6-7
//    (Unsloth's file: NVFP4 in 0-19, BF16 in 20-23); every other matrix BF16, norms and small tensors F32, the router F32;
//  - a vocabulary of 500 padded to 512 rows (the real 200,058 / 201,024), tokenizer pre "inkling2".
// --no-loop: architecture inkling2 (the same layers once, no loop keys or controllers).
// --nvfp4-equal: every NVFP4 block's four scales equal (the per-32 fold then exact: zero/kernels/decode.cl QT 40).
// --all-experts: every expert active (expert_used_count = expert_count), so no top-k choice: the random router's near
//   ties (its sigmoids near 1, the bias at 0.1) flip under int8 rounding and part a check from the reference for that
//   reason alone (2026-11-23, the first engine runs); with all experts the comparison measures the arithmetic.
//
// usage: make_tiny_inkling2 OUT.gguf [--seed N] [--no-loop] [--nvfp4-equal] [--all-experts]
#include "../src/log.h"  // (zlog: first, as gguf_pack.h's headers use it)
#include "../zero/gguf_pack.h"  // (zpack::Writer / put_kv)
#include "../src/quants.h"  // (f32_to_fp16, nvfp4_scale)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int E = 256, NL = 8, H = 6, HKV = 2, D = 128, NREL = 16, SWA = 16, EXT = 32, NC = 4, BLOCK = 2;
int NE = 8, NU = 2, FF = 128, RANK = 16, LSTART = 2, LEND = 6, PASSES = 3, NV = 500, NVP = 512;
const std::vector<int> GLOBAL = {1, 5, 7};
const int NVFP4_LAYERS = 6;  // (experts of layers 0..5 NVFP4, the rest BF16)

// ---- random numbers: xoshiro256** seeded through splitmix64; normals by Box-Muller (as make_tiny_deepseek41)
struct Rng {
  uint64_t s[4];
  bool have = false;
  double spare = 0;
  explicit Rng(uint64_t seed) {
    for (auto& x : s) {
      seed += 0x9E3779B97F4A7C15ull;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      x = z ^ (z >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    if (have) { have = false; return spare; }
    double u1;
    do u1 = uniform(); while (u1 <= 0);
    const double u2 = uniform(), r = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
    spare = r * std::sin(a);
    have = true;
    return r * std::cos(a);
  }
};

typedef std::vector<float> F;

uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
template <typename T> void put(std::vector<uint8_t>& o, T v) {
  const size_t n = o.size();
  o.resize(n + sizeof v);
  memcpy(&o[n], &v, sizeof v);
}

bool g_nvfp4_equal = false;

// NVFP4 (ggml type 40): per 16 values a UE4M3 scale s (value = kvalues_mxfp4[code] x s, s = ggml_ue4m3_to_fp32),
// chosen as the representable s nearest to amax / 12 (the largest code 12 then reaches about amax), the codes the
// nearest values; --nvfp4-equal: one scale for the whole 64-block, from its largest magnitude
void encode_nvfp4(const float* v, std::vector<uint8_t>& o) {
  static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  auto pick_scale = [](double want) {  // the byte (0x00..0x7E) whose scale is nearest to want
    int best = 0;
    double bd = INFINITY;
    for (int c = 0; c < 0x7F; ++c) {
      const double d = std::fabs(nvfp4_scale((uint8_t)c) - want);
      if (d < bd) bd = d, best = c;
    }
    return (uint8_t)best;
  };
  uint8_t sc[4];
  double amax_all = 0;
  for (int i = 0; i < 64; ++i) amax_all = std::max(amax_all, (double)std::fabs(v[i]));
  for (int s = 0; s < 4; ++s) {
    double amax = 0;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, (double)std::fabs(v[16 * s + i]));
    sc[s] = pick_scale((g_nvfp4_equal ? amax_all : amax) / 12.0);
  }
  for (int s = 0; s < 4; ++s) put<uint8_t>(o, sc[s]);
  for (int s = 0; s < 4; ++s) {
    const double d = nvfp4_scale(sc[s]);
    uint8_t q[16];
    for (int i = 0; i < 16; ++i) {
      const double y = d > 0 ? v[16 * s + i] / d : 0.0;
      int best = 0;
      double bd = INFINITY;
      for (int c = 0; c < 16; ++c) {
        const double dd = std::fabs(y - kv[c]);
        if (dd < bd) bd = dd, best = c;
      }
      q[i] = (uint8_t)best;
    }
    for (int j = 0; j < 8; ++j) put<uint8_t>(o, (uint8_t)(q[j] | (q[j + 8] << 4)));
  }
}

std::vector<uint8_t> encode(const F& x, uint32_t type) {
  std::vector<uint8_t> o;
  if (type == GGML_F32) {
    o.resize(x.size() * 4);
    memcpy(o.data(), x.data(), o.size());
  } else if (type == GGML_BF16) {
    for (float v : x) put<uint16_t>(o, f2bf(v));
  } else if (type == GGML_NVFP4) {
    if (x.size() % 64) throw std::runtime_error("NVFP4 needs a multiple of 64 values");
    for (size_t b = 0; b < x.size() / 64; ++b) encode_nvfp4(&x[b * 64], o);
  } else {
    throw std::runtime_error("encode: type " + std::to_string(type));
  }
  return o;
}

struct Kv { std::string key; uint32_t t, et; GgufValue v; };
struct Tensor {
  std::string name;
  std::vector<uint64_t> dims;  // (ne0 first)
  uint32_t type;
  std::vector<uint8_t> data;
};

struct Model {
  std::vector<Kv> kv;
  std::vector<Tensor> ts;
  void u32(const std::string& k, int64_t v) { kv.push_back({k, 4, 0, v}); }
  void f32(const std::string& k, double v) { kv.push_back({k, 6, 0, v}); }
  void b(const std::string& k, bool v) { kv.push_back({k, 7, 0, v}); }
  void str(const std::string& k, const std::string& v) { kv.push_back({k, 8, 0, v}); }
  void arr(const std::string& k, uint32_t et, const GgufValue& v) { kv.push_back({k, 9, et, v}); }
  // data in row-major [.., ne1, ne0]; shape outermost first
  void tensor(const std::string& name, const F& data, uint32_t type, std::vector<uint64_t> shape) {
    Tensor t;
    t.name = name;
    t.dims.assign(shape.rbegin(), shape.rend());
    t.type = type;
    t.data = encode(data, type);
    ts.push_back(std::move(t));
  }
  void write(const std::string& path) {
    zpack::Writer w(path);
    w.put(std::string("GGUF").data(), 4);
    w.put<uint32_t>(3);
    w.put<uint64_t>(ts.size());
    w.put<uint64_t>(kv.size());
    for (auto& k : kv) zpack::put_kv(w, k.key, k.t, k.et, k.v);
    uint64_t off = 0;
    for (auto& t : ts) {
      w.str(t.name);
      w.put<uint32_t>((uint32_t)t.dims.size());
      for (uint64_t d : t.dims) w.put<uint64_t>(d);
      w.put<uint32_t>(t.type);
      w.put<uint64_t>(off);
      off = (off + t.data.size() + 31) / 32 * 32;
    }
    w.pad(32);
    for (auto& t : ts) {
      w.put(t.data.data(), t.data.size());
      w.pad(32);
    }
    w.close();
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    printf("usage: make_tiny_inkling2 OUT.gguf [--seed N] [--no-loop] [--nvfp4-equal] [--all-experts]\n");
    return argc < 2 ? 1 : 0;
  }
  const std::string out = argv[1];
  uint64_t seed = 2;
  bool loop = true;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
    else if (a == "--no-loop") loop = false;
    else if (a == "--nvfp4-equal") g_nvfp4_equal = true;
    else if (a == "--all-experts") NU = NE;
    else { fprintf(stderr, "make_tiny_inkling2: unknown option %s\n", a.c_str()); return 2; }
  }
  try {
    Rng rng(seed);
    Model w;
    const std::string ARCH = loop ? "inkling2_loop" : "inkling2", p = ARCH + ".";
    auto mat = [&](int n_out, int n_in, double scale = 1.0) {  // [n_out][n_in], outputs of about unit size
      F v((size_t)n_out * n_in);
      const double s = scale / std::sqrt((double)n_in);
      for (auto& x : v) x = (float)(rng.normal() * s);
      return v;
    };
    auto normw = [&](size_t n, double c = 1.0, double sd = 0.1) {
      F v(n);
      for (auto& x : v) x = (float)(c + sd * rng.normal());
      return v;
    };
    auto randn = [&](size_t n, double scale = 1.0) {
      F v(n);
      for (auto& x : v) x = (float)(scale * rng.normal());
      return v;
    };
    typedef std::vector<uint64_t> S;
    auto is_global = [](int il) { return std::find(GLOBAL.begin(), GLOBAL.end(), il) != GLOBAL.end(); };

    w.str("general.architecture", ARCH);
    w.str("general.name", "Inkling 2 tiny (random weights, inferzero test)");
    w.u32("general.alignment", 32);
    w.u32(p + "block_count", NL);
    w.u32(p + "context_length", 4096);
    w.u32(p + "embedding_length", E);
    w.u32(p + "feed_forward_length", FF);
    w.arr(p + "attention.head_count", 5, std::vector<int64_t>(NL, H));
    w.arr(p + "attention.head_count_kv", 5, std::vector<int64_t>(NL, HKV));
    w.f32(p + "attention.layer_norm_rms_epsilon", 1e-6);
    w.u32(p + "expert_count", NE);
    w.u32(p + "expert_used_count", NU);
    w.u32(p + "attention.key_length", D);
    w.u32(p + "attention.value_length", D);
    if (loop) {
      w.u32(p + "loop.count", PASSES);
      w.u32(p + "loop.start", LSTART);
      w.u32(p + "loop.end", LEND);
      w.u32(p + "loop.gate_rank", RANK);
      w.b(p + "loop.varying", false);
    }
    w.u32(p + "vocab_size", NVP);
    w.u32(p + "vocab_size_unpadded", NV);
    w.u32(p + "expert_feed_forward_length", FF);
    w.u32(p + "attention.key_length_swa", D);
    w.u32(p + "attention.value_length_swa", D);
    w.u32(p + "rope.dimension_count", 0);
    w.u32(p + "attention.sliding_window", SWA);
    std::vector<int64_t> pattern(NL);
    for (int il = 0; il < NL; ++il) pattern[il] = is_global(il) ? 0 : 1;
    w.arr(p + "attention.sliding_window_pattern", 7, pattern);
    w.u32(p + "shortconv.l_cache", NC);
    w.f32(p + "expert_weights_scale", 8.0);
    w.f32(p + "activation.situ_beta", 4.0);
    w.f32(p + "activation.situ_linear_beta", 25.0);
    w.u32(p + "attn_res.block_size", BLOCK);
    w.f32(p + "final_logit_softcapping", 30.0);
    w.u32(p + "attention.relative_length", NREL);
    w.u32(p + "attention.relative_extent", EXT);
    w.f32(p + "attention.qk_clip", 22.62742);
    w.f32(p + "attention.relative_clip", 8.0);
    w.u32(p + "attention.log_scale_floor", 8);
    w.f32(p + "attention.log_scale_alpha", 0.1);

    // a token list so that the tokenizer loads (prompts are fed as ids): the real file's kind of specials at the end
    std::vector<std::string> toks;
    std::vector<int64_t> ttype;
    const char* specials[] = {"<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>",
                              "<|message_tool|>", "<|content_text|>", "<|content_model_end_sampling|>",
                              "<|content_thinking|>", "<|end_message|>", "<|content_invoke_tool_json|>"};
    const int nsp = (int)(sizeof(specials) / sizeof(specials[0]));
    for (int i = 0; i < NV - nsp; ++i) { toks.push_back("t" + std::to_string(i)); ttype.push_back(1); }
    for (int i = 0; i < nsp; ++i) { toks.push_back(specials[i]); ttype.push_back(3); }
    w.str("tokenizer.ggml.model", "gpt2");
    w.str("tokenizer.ggml.pre", "inkling2");
    w.arr("tokenizer.ggml.tokens", 8, toks);
    w.arr("tokenizer.ggml.token_type", 5, ttype);
    w.arr("tokenizer.ggml.merges", 8, std::vector<std::string>{});
    w.u32("tokenizer.ggml.eos_token_id", NV - nsp + 6);  // (<|content_model_end_sampling|>, as 200006)
    w.u32("tokenizer.ggml.padding_token_id", NV - nsp);
    w.b("tokenizer.ggml.add_bos_token", false);

    const uint32_t F32 = GGML_F32, BF16 = GGML_BF16, NVFP4 = GGML_NVFP4;
    const uint64_t uE = E;
    w.tensor("output_res_query.weight", randn(E, 0.5), F32, S{uE});
    w.tensor("token_embd.weight", randn((size_t)NVP * E), BF16, S{(uint64_t)NVP, uE});
    w.tensor("token_embd_norm.weight", normw(E), F32, S{uE});
    w.tensor("output_norm.weight", normw(E), F32, S{uE});
    w.tensor("output.weight", mat(NVP, E, 1.5), BF16, S{(uint64_t)NVP, uE});
    for (int il = 0; il < NL; ++il) {
      const std::string b = "blk." + std::to_string(il) + ".";
      const bool glob = is_global(il), swa = !glob;
      const int ext = swa ? SWA : EXT, KG = HKV * D;
      if (il > 0) {
        const int nq = glob ? 3 : 1;
        w.tensor(b + "attn_res_query.weight", randn((size_t)nq * E, 0.5), F32, nq == 1 ? S{uE} : S{(uint64_t)nq, uE});
      }
      w.tensor(b + "ffn_res_query.weight", randn(E, 0.5), F32, S{uE});
      w.tensor(b + "attn_sink_proj.weight", mat(H, E, 0.5), BF16, S{(uint64_t)H, uE});
      w.tensor(b + "attn_gate.weight", mat(H * D, E), BF16, S{(uint64_t)H * D, uE});
      w.tensor(b + "attn_k_conv.weight", randn((size_t)KG * NC, 0.3), F32, S{(uint64_t)KG, (uint64_t)NC});
      w.tensor(b + "attn_q_clip.weight", normw((size_t)H * D), F32, S{(uint64_t)H, (uint64_t)D});
      w.tensor(b + "attn_rel_clip.weight", normw((size_t)H * NREL), F32, S{(uint64_t)H, (uint64_t)NREL});
      w.tensor(b + "attn_rel_bias.weight", randn((size_t)ext * NREL, 0.5), F32, S{(uint64_t)ext, (uint64_t)NREL});
      w.tensor(b + "attn_v_conv.weight", randn((size_t)KG * NC, 0.3), F32, S{(uint64_t)KG, (uint64_t)NC});
      if (glob) {
        w.tensor(b + "attn_k_conv_proj.weight", mat(NC * HKV, E, 0.3), BF16, S{(uint64_t)NC * HKV, uE});
        w.tensor(b + "attn_v_conv_proj.weight", mat(NC * HKV, E, 0.3), BF16, S{(uint64_t)NC * HKV, uE});
      }
      // (q scaled up so that the L2 clip of 22.6 bites on some rows)
      w.tensor(b + "attn_k.weight", mat(KG, E, 2.0), BF16, S{(uint64_t)KG, uE});
      w.tensor(b + "attn_output.weight", mat(E, H * D), BF16, S{uE, (uint64_t)H * D});
      w.tensor(b + "attn_q.weight", mat(H * D, E, 2.5), BF16, S{(uint64_t)H * D, uE});
      w.tensor(b + "attn_rel.weight", mat(NREL * H, E, 3.0), BF16, S{(uint64_t)NREL * H, uE});
      w.tensor(b + "attn_v.weight", mat(KG, E), BF16, S{(uint64_t)KG, uE});
      if (swa) {
        w.tensor(b + "attn_norm.weight", normw(E), F32, S{uE});
      } else {
        w.tensor(b + "attn_q_inp_norm.weight", normw(E), F32, S{uE});
        w.tensor(b + "attn_k_inp_norm.weight", normw(E), F32, S{uE});
        w.tensor(b + "attn_v_inp_norm.weight", normw(E), F32, S{uE});
      }
      w.tensor(b + "attn_post_conv.weight", randn((size_t)E * NC, 0.3), F32, S{uE, (uint64_t)NC});
      const uint32_t XT = il < NVFP4_LAYERS ? NVFP4 : BF16;
      const uint64_t nE = NE;
      w.tensor(b + "ffn_gate_up_exps.weight", randn((size_t)NE * 2 * FF * E, 3 / std::sqrt((double)E)), XT,
               S{nE, (uint64_t)2 * FF, uE});
      if (XT == NVFP4) w.tensor(b + "ffn_gate_up_exps.scale", normw(NE, 1.0, 0.2), F32, S{nE});
      w.tensor(b + "ffn_down_exps.weight", randn((size_t)NE * E * FF, 1 / std::sqrt((double)FF)), XT,
               S{nE, uE, (uint64_t)FF});
      if (XT == NVFP4) w.tensor(b + "ffn_down_exps.scale", normw(NE, 1.0, 0.2), F32, S{nE});
      w.tensor(b + "exp_probs_b.bias", randn(NE, 0.1), F32, S{nE});
      w.tensor(b + "ffn_gate_inp.weight", mat(NE, E, 4.0), F32, S{nE, uE});
      w.tensor(b + "ffn_norm.weight", normw(E), F32, S{uE});
      w.tensor(b + "ffn_post_conv.weight", randn((size_t)E * NC, 0.3), F32, S{uE, (uint64_t)NC});
      if (loop && il >= LSTART && il < LEND) {
        w.tensor(b + "loop_msg_query.weight", randn((size_t)4 * E, 0.5), F32, S{4, uE});
        for (const char* c : {"q", "k", "v", "ffn"}) {
          const std::string lp = b + "loop_" + c + "_ctrl_";
          w.tensor(lp + "down.weight", mat(RANK, 2 * E), BF16, S{(uint64_t)RANK, (uint64_t)2 * E});
          w.tensor(lp + "forget.weight", mat(E, RANK, 0.5), BF16, S{uE, (uint64_t)RANK});
          w.tensor(lp + "gain.weight", mat(E, RANK, 0.5), BF16, S{uE, (uint64_t)RANK});
        }
      }
    }
    w.write(out);
    printf("%s: %s, %zu tensors, %zu keys\n", out.c_str(), ARCH.c_str(), w.ts.size(), w.kv.size());
  } catch (const std::exception& e) {
    fprintf(stderr, "make_tiny_inkling2: %s\n", e.what());
    return 1;
  }
  return 0;
}
