// A part of class Engine (zero/engine.h), included inside the class body, not a header of its own: Inkling 2
// (GGUF inkling2 / inkling2_loop, docs/inkling2.md step I1.5). The graph is llama.cpp master's (src/models/inkling2.cpp,
// inkling2-loop.cpp), the f32 reference tools/ref_inkling2.h; this follows it step by step. Correctness first:
//  - the dense matrices through the engine's int8 matmuls (BF16 requantized to Q8_0, as for every model); those whose
//    shapes the GEMM does not take (sinks: 24 rows; the controllers' rank-64 inputs) and the F32 router in f32;
//  - the experts on their raw GGUF rows by ink_expert_mv (NVFP4 per 16 exactly; BF16), not yet the MoE kernels;
//  - attention in f32 over an f16 K / V cache per cache slot (llama's slots: the 24 physical layers, then the loop's
//    later passes, 48 for Inkling 2 Mini), with each slot's short-convolution state;
//  - one sequence: no slots, drafters, snapshots or multi-GPU yet (refused at load).
// Hooks outside this file, each under hp.inkling: load_hparams (engine_setup.h), build_kernels / load_weights
// (engine_load.h), alloc_buffers (engine_alloc.h), forward / run_layer / head_rows / reset / snapshot / restore /
// max_chunk (engine.h), ctx_bytes (engine_slots.h), and ops_inkling.cl in KernelParts::kOps (clrt.h).

  struct InkHp {
    bool loop = false;
    int start = 0, end = 0, passes = 1, rank = 0, nrel = 0, ext = 0, swa = 0, nc = 0, block = 0, nv = 0, nslot = 0;
    float qk_clip = 0, rel_clip = 0, log_floor = 1, log_alpha = 0, wscale = 1, situ_b = 1, situ_lb = 0;
  } ink_;
  // a matrix: the engine's int8 matmul (w) or, for shapes the GEMM does not take, f32 rows (f)
  struct InkMat {
    GpuWeight w;
    cl_mem f = nullptr;
    int N = 0, K = 0;
  };
  struct InkLayer {  // per physical layer
    bool glob = false;
    int nq = 0, ext = 0, nu = 0;
    InkMat q, k, v, o, gate, rel, sink, kcp, vcp, router;
    cl_mem res_q = nullptr, ffn_res_q = nullptr, an = nullptr, inorm[3] = {nullptr, nullptr, nullptr}, fn = nullptr;
    cl_mem q_clip = nullptr, rel_clip = nullptr, rel_bias = nullptr, k_conv = nullptr, v_conv = nullptr;
    cl_mem post_conv = nullptr, ffn_conv = nullptr, probs_b = nullptr, gu_scale = nullptr, dn_scale = nullptr;
    cl_mem gu = nullptr, dn = nullptr;  // (the experts' raw rows)
    uint32_t gu_type = 0, dn_type = 0;
    uint64_t gu_rb = 0, dn_rb = 0;
    // a loop core layer's controllers (q, k, v, ffn) and its message queries
    cl_mem msg_q = nullptr;
    InkMat ctrl_down[4], ctrl_gain[4], ctrl_forget[4];
  };
  struct InkSlot {  // per cache slot
    int ip = 0, cap = 0;
    cl_mem kc = nullptr, vc = nullptr, conv[4] = {nullptr, nullptr, nullptr, nullptr};  // (conv: K, V, attn out, ffn out)
  };
  std::vector<InkLayer> inkl_;
  std::vector<InkSlot> inks_;
  // the chunk's buffers (rows of max_chunk)
  cl_mem ink_cur_ = nullptr, ink_in_[3] = {nullptr, nullptr, nullptr}, ink_q_ = nullptr, ink_k0_ = nullptr, ink_v0_ = nullptr;
  cl_mem ink_kc_ = nullptr, ink_vc_ = nullptr, ink_r_ = nullptr, ink_s_ = nullptr, ink_g_ = nullptr, ink_kd_ = nullptr;
  cl_mem ink_vd_ = nullptr, ink_att_ = nullptr, ink_ao_ = nullptr, ink_tmp_ = nullptr, ink_xm_ = nullptr, ink_mix_ = nullptr;
  cl_mem ink_msgs_ = nullptr, ink_cg_ = nullptr, ink_lo_ = nullptr, ink_gain_ = nullptr, ink_fgt_ = nullptr;
  cl_mem ink_rl_ = nullptr, ink_ids_ = nullptr, ink_w_ = nullptr, ink_gu_ = nullptr, ink_act_ = nullptr, ink_de_ = nullptr;
  cl_mem ink_mo_ = nullptr, ink_blocks_ = nullptr, ink_bank_ = nullptr, ink_bank_out_ = nullptr, ink_prelude_ = nullptr;
  cl_mem ink_kv_ = nullptr, ink_dummy_ = nullptr;
  int ink_maxb_ = 0;  // (blocks the chunk buffer holds)

  int ink_layer_phys(int il) const {
    return il < hp.n_layer ? il : ink_.start + (il - hp.n_layer) % (ink_.end - ink_.start);
  }

  void load_hparams_inkling() {
    const std::string a = hp.arch;
    auto I = [&](const std::string& k, int64_t def = -1) {
      const int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("inkling2: missing hparam " + k);
      return (int)v;
    };
    auto Fl = [&](const std::string& k, double def) { return (float)f_.get_float(a + "." + k, def); };
    // (a per-layer array, or one value for every layer; llama reads them per layer)
    auto per = [&](const std::string& k, int il) {
      if (const auto* v = f_.get_ints(a + "." + k)) {
        if (il >= (int)v->size()) throw std::runtime_error("inkling2: " + k + " has too few layers");
        return (int)(*v)[il];
      }
      return I(k);
    };
    hp.inkling = true;
    hp.full_attn_interval = 1;  // (no DeltaNet layers)
    hp.n_layer = I("block_count");
    hp.n_embd = I("embedding_length");
    hp.head_dim = I("attention.key_length");
    hp.eps = Fl("attention.layer_norm_rms_epsilon", 1e-6);
    hp.n_expert = I("expert_count");
    hp.n_ff_exp = I("expert_feed_forward_length");
    hp.n_ff = hp.n_ff_exp;
    hp.n_vocab = I("vocab_size");  // (the embedding and head rows; the tokens: vocab_size_unpadded)
    hp.softcap = Fl("final_logit_softcapping", 0.0);
    InkHp& k = ink_;
    k.nv = I("vocab_size_unpadded");
    k.nrel = I("attention.relative_length");
    k.ext = I("attention.relative_extent");
    k.swa = I("attention.sliding_window");
    k.nc = I("shortconv.l_cache");
    k.block = I("attn_res.block_size");
    k.qk_clip = Fl("attention.qk_clip", 0);
    k.rel_clip = Fl("attention.relative_clip", 0);
    k.log_floor = (float)I("attention.log_scale_floor");
    k.log_alpha = Fl("attention.log_scale_alpha", 0);
    k.wscale = Fl("expert_weights_scale", 0);
    k.situ_b = Fl("activation.situ_beta", 1);
    k.situ_lb = Fl("activation.situ_linear_beta", 0);
    k.loop = a == "inkling2_loop";
    k.start = k.loop ? I("loop.start") : hp.n_layer;
    k.end = k.loop ? I("loop.end") : hp.n_layer;
    k.passes = k.loop ? I("loop.count") : 1;
    k.rank = k.loop ? I("loop.gate_rank") : 0;
    k.nslot = hp.n_layer + (k.passes - 1) * (k.end - k.start);
    hp.n_swa = k.swa;
    const auto* pat = f_.get_ints(a + ".attention.sliding_window_pattern");
    if (!pat || (int)pat->size() != hp.n_layer) throw std::runtime_error("inkling2: no sliding_window_pattern per layer");
    hp.swa.assign(hp.n_layer, 0);
    hp.kv_heads.assign(hp.n_layer, 0);
    inkl_.assign(hp.n_layer, InkLayer());
    for (int il = 0; il < hp.n_layer; ++il) {
      hp.swa[il] = (*pat)[il] != 0;
      hp.kv_heads[il] = per("attention.head_count_kv", il);
      inkl_[il].glob = !hp.swa[il];
      inkl_[il].nu = per("expert_used_count", il);
      const int h = per("attention.head_count", il);
      if (il == 0) hp.n_head = h, hp.n_head_kv = hp.kv_heads[0];
      if (h != hp.n_head || hp.kv_heads[il] != hp.n_head_kv)
        throw std::runtime_error("inkling2: head counts that differ by layer are not supported yet");
      if (inkl_[il].nu < 1 || inkl_[il].nu > hp.n_expert || inkl_[il].nu > 64)
        throw std::runtime_error("inkling2: unsupported expert_used_count");
      hp.n_expert_used = std::max(hp.n_expert_used, inkl_[il].nu);
    }
    // (the checks llama.cpp makes, and this engine's limits: the kernels' head size, the bias table in local memory,
    //  the router in one work-group, the mix's states)
    if (hp.head_dim != 128 || hp.n_head % hp.n_head_kv || k.nrel < 1 || k.swa < 1 || k.ext < 1 || k.nc < 2 || k.nc > 64 ||
        k.block < 1 || k.log_floor < 1 || k.nv < 1 || k.nv > hp.n_vocab || hp.n_expert > 256 || hp.n_embd % 256)
      throw std::runtime_error("inkling2: unsupported dimensions (head size 128, the embedding a multiple of 256, at most "
                               "256 experts)");
    if (std::max(k.swa, k.ext) > 1024) throw std::runtime_error("inkling2: a relative-bias table over 1024 rows");
    if (k.nrel > 256) throw std::runtime_error("inkling2: more than 256 relative buckets (ink_rows' work-group)");
    if (hp.n_ff_exp % 64) throw std::runtime_error("inkling2: an expert FFN width not a multiple of 64 (NVFP4 blocks)");
    if (k.loop && (k.passes < 2 || k.rank < 1 || k.start < 1 || k.start >= k.end || k.end > hp.n_layer ||
                   k.start % k.block || k.end % k.block))
      throw std::runtime_error("inkling2: invalid loop core");
    if (opt.slots > 1 || opt.mtp || !opt.dflash.empty() || opt.tp_size > 1)
      throw std::runtime_error("Inkling 2: slots, drafters and several GPUs are not supported yet");
    for (const GgufTensor& t : f_.tensors)
      if (t.name.size() > 12 && t.name.compare(t.name.size() - 12, 12, ".input_scale") == 0)
        throw std::runtime_error("Inkling 2: activation scales (" + t.name + ") are not supported");
  }

  // a matrix [N][K]: the engine's matmul where the GEMM takes its shape and it is not kept f32, else f32 rows.
  // llama's build_lora_mm multiplies a 2D weight's output by its <name>.scale: not supported here yet (refused)
  InkMat ink_mat(const std::string& name, int N, int K, bool f32 = false) {
    InkMat m;
    m.N = N, m.K = K;
    const std::string sn = name.substr(0, name.size() - 7) + ".scale";
    if (f_.find(sn)) throw std::runtime_error("Inkling 2: weight scales on 2D weights (" + sn + ") are not supported yet");
    const GgufTensor& t = f_.get(name);
    if (t.dims.size() != 2 || (int)t.dims[0] != K || (int)t.dims[1] != N)
      throw std::runtime_error("unexpected tensor shape: " + name + " (" + std::to_string(N) + " x " + std::to_string(K) + ")");
    if (f32 || N % 16 || K % 256) {
      m.f = load_vec(name, N * K);
    } else {
      m.w = load_w(name);
      expect(m.w, N, K, name);
    }
    return m;
  }

  void load_weights_inkling() {
    const InkHp& k = ink_;
    const int E = hp.n_embd, D = hp.head_dim, H = hp.n_head, HKV = hp.n_head_kv, KG = HKV * D, F = hp.n_ff_exp, NE = hp.n_expert;
    layers_.resize(hp.n_layer);  // (the engine's own per-layer state stays unused but for il)
    for (int il = 0; il < hp.n_layer; ++il) {
      layers_[il].il = il;
      InkLayer& L = inkl_[il];
      const std::string p = "blk." + std::to_string(il) + ".";
      L.ext = L.glob ? k.ext : k.swa;
      L.q = ink_mat(p + "attn_q.weight", H * D, E);
      L.k = ink_mat(p + "attn_k.weight", KG, E);
      L.v = ink_mat(p + "attn_v.weight", KG, E);
      L.o = ink_mat(p + "attn_output.weight", E, H * D);
      L.gate = ink_mat(p + "attn_gate.weight", H * D, E);
      L.rel = ink_mat(p + "attn_rel.weight", k.nrel * H, E);
      L.sink = ink_mat(p + "attn_sink_proj.weight", H, E);
      if (L.glob) {
        L.kcp = ink_mat(p + "attn_k_conv_proj.weight", k.nc * HKV, E);
        L.vcp = ink_mat(p + "attn_v_conv_proj.weight", k.nc * HKV, E);
        for (int j = 0; j < 3; ++j) L.inorm[j] = load_vec(p + "attn_" + "qkv"[j] + "_inp_norm.weight", E);
      } else {
        L.an = load_vec(p + "attn_norm.weight", E);
      }
      if (il > 0) {  // (its queries from the tensor's size, as llama.cpp and the reference: 1 or 3)
        const uint64_t n = f_.get(p + "attn_res_query.weight").nelements();
        L.nq = (int)(n / E);
        if (n % E || (L.nq != 1 && L.nq != 3) || (L.nq == 3 && !L.glob))
          throw std::runtime_error("unexpected tensor shape: " + p + "attn_res_query.weight");
        L.res_q = load_vec(p + "attn_res_query.weight", L.nq * E);
      }
      L.ffn_res_q = load_vec(p + "ffn_res_query.weight", E);
      L.fn = load_vec(p + "ffn_norm.weight", E);
      L.q_clip = load_vec(p + "attn_q_clip.weight", H * D);
      L.rel_clip = load_vec(p + "attn_rel_clip.weight", H * k.nrel);
      L.rel_bias = load_vec(p + "attn_rel_bias.weight", L.ext * k.nrel);
      L.k_conv = load_vec(p + "attn_k_conv.weight", KG * k.nc);
      L.v_conv = load_vec(p + "attn_v_conv.weight", KG * k.nc);
      L.post_conv = load_vec(p + "attn_post_conv.weight", E * k.nc);
      L.ffn_conv = load_vec(p + "ffn_post_conv.weight", E * k.nc);
      L.router = ink_mat(p + "ffn_gate_inp.weight", NE, E, true);
      if (f_.find(p + "exp_probs_b.bias")) L.probs_b = load_vec(p + "exp_probs_b.bias", NE);
      if (f_.find(p + "ffn_gate_up_exps.scale")) L.gu_scale = load_vec(p + "ffn_gate_up_exps.scale", NE);
      if (f_.find(p + "ffn_down_exps.scale")) L.dn_scale = load_vec(p + "ffn_down_exps.scale", NE);
      // the experts' raw rows ([K][N][NE]: expert e's row n at (e N + n) row_bytes)
      auto experts = [&](const std::string& name, int N, int K, cl_mem& raw, uint32_t& type, uint64_t& rb) {
        const GgufTensor& t = f_.get(name);
        if (t.dims.size() != 3 || (int)t.dims[0] != K || (int)t.dims[1] != N || (int)t.dims[2] != NE)
          throw std::runtime_error("unexpected expert tensor " + name);
        type = t.type;
        if (type != GGML_NVFP4 && type != GGML_BF16 && type != GGML_Q8_0 && type != GGML_F32)
          throw std::runtime_error("Inkling 2: expert type " + std::string(ggml_type_name(type)) + " (" + name +
                                   ") not supported yet (NVFP4, BF16, Q8_0, F32)");
        if (!t.data) throw std::runtime_error("Inkling 2: " + name + " has no data in this file");
        rb = t.nbytes / ((uint64_t)NE * N);
        raw = buf(t.nbytes, t.data);
      };
      experts(p + "ffn_gate_up_exps.weight", 2 * F, E, L.gu, L.gu_type, L.gu_rb);
      experts(p + "ffn_down_exps.weight", E, F, L.dn, L.dn_type, L.dn_rb);
      if (k.loop && il >= k.start && il < k.end) {
        L.msg_q = load_vec(p + "loop_msg_query.weight", 4 * E);
        static const char* kc[4] = {"q", "k", "v", "ffn"};
        for (int c = 0; c < 4; ++c) {
          const std::string lp = p + "loop_" + kc[c] + "_ctrl_";
          L.ctrl_down[c] = ink_mat(lp + "down.weight", k.rank, 2 * E);
          L.ctrl_gain[c] = ink_mat(lp + "gain.weight", E, k.rank);
          L.ctrl_forget[c] = ink_mat(lp + "forget.weight", E, k.rank);
        }
      }
      clFinish(cl_.queue);
    }
    if (f_.find("output.scale")) throw std::runtime_error("Inkling 2: a scale on the output head is not supported yet");
    ink_tok_norm_ = load_vec("token_embd_norm.weight", E);
    ink_out_res_q_ = load_vec("output_res_query.weight", E);
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    ink_kv_ = buf(16, kv);
    if (lm_head_.N != hp.n_vocab) throw std::runtime_error("inkling2: the output head's rows differ from vocab_size");
    // the cache slots: K / V (f16) at the slot's capacity, the conv states (zeroed by ink_reset)
    inks_.assign(k.nslot, InkSlot());
    const int swa_cap = swa_ring_cap(ctx, max_chunk, 0);
    for (int s = 0; s < k.nslot; ++s) {
      InkSlot& S = inks_[s];
      S.ip = ink_layer_phys(s);
      S.cap = inkl_[S.ip].glob ? ctx : swa_cap;
      S.kc = buf((size_t)S.cap * KG * 2);
      S.vc = buf((size_t)S.cap * KG * 2);
      const int ch[4] = {KG, KG, E, E};
      for (int c = 0; c < 4; ++c) S.conv[c] = buf((size_t)ch[c] * (k.nc - 1) * 4);
    }
    ink_reset();
  }
  cl_mem ink_tok_norm_ = nullptr, ink_out_res_q_ = nullptr;

  // GPU bytes of a c-token context: the slots' K / V (the global layers' slots hold c positions, the sliding ones a
  // ring), by the formulas of load_weights_inkling
  size_t ink_ctx_cache_bytes(int c) const {
    size_t n = 0;
    const int swa_cap = swa_ring_cap(c, max_chunk, 0);
    for (int s = 0; s < ink_.nslot; ++s)
      n += (size_t)(inkl_[ink_layer_phys(s)].glob ? c : swa_cap) * hp.n_head_kv * hp.head_dim * 2 * 2;
    return n;
  }

  void alloc_inkling() {
    const size_t M = max_chunk;
    const InkHp& k = ink_;
    const int E = hp.n_embd, HD = hp.n_head * hp.head_dim, KG = hp.n_head_kv * hp.head_dim, F = hp.n_ff_exp;
    const int NU = hp.n_expert_used, NE = hp.n_expert;
    auto f = [&](size_t per_row) { return buf(M * per_row * 4); };
    ink_cur_ = f(E);
    for (int j = 0; j < 3; ++j) ink_in_[j] = f(E);
    ink_q_ = f(HD); ink_g_ = f(HD); ink_att_ = f(HD);
    ink_k0_ = f(KG); ink_v0_ = f(KG); ink_kc_ = f(KG); ink_vc_ = f(KG);
    ink_r_ = f((size_t)k.nrel * hp.n_head);
    ink_s_ = f(hp.n_head);
    ink_kd_ = f((size_t)k.nc * hp.n_head_kv); ink_vd_ = f((size_t)k.nc * hp.n_head_kv);
    ink_ao_ = f(E); ink_tmp_ = f(std::max(E, KG)); ink_xm_ = f(E); ink_mo_ = f(E);
    ink_mix_ = f(3 * (size_t)E);
    ink_msgs_ = f(4 * (size_t)E);
    ink_cg_ = f(2 * (size_t)E);
    ink_lo_ = f(std::max(k.rank, 1));
    ink_gain_ = f(E); ink_fgt_ = f(E);
    ink_rl_ = f(NE);
    ink_ids_ = f(NU); ink_w_ = f(NU);
    ink_gu_ = f((size_t)NU * 2 * F); ink_act_ = f((size_t)NU * F); ink_de_ = f((size_t)NU * E);
    // the blocks: one per attn_res block of the layers before the output (the prelude's, a pass's, the epilogue's)
    ink_maxb_ = (hp.n_layer + k.block - 1) / k.block + 1;
    if (ink_maxb_ + 1 > 16) throw std::runtime_error("inkling2: more residual blocks than the mix holds (16)");
    ink_blocks_ = f((size_t)ink_maxb_ * E);
    ink_bank_ = f((size_t)ink_maxb_ * E);
    ink_bank_out_ = f(E);
    ink_prelude_ = f(E);
    ink_dummy_ = buf(64);
  }

  // the conv states to zero (a new sequence; llama.cpp's recurrent state starts at zero)
  void ink_reset() {
    const int E = hp.n_embd, KG = hp.n_head_kv * hp.head_dim;
    const int ch[4] = {KG, KG, E, E};
    for (auto& S : inks_)
      for (int c = 0; c < 4; ++c)
        if (S.conv[c]) fill(S.conv[c], (size_t)ch[c] * (ink_.nc - 1) * 4);
  }

  // ---- launches
  void ink_run1(const char* name, cl_kernel kk, size_t n) {  // (element-wise kernels: any size, bounds-checked)
    size_t gws[1] = {(n + 63) / 64 * 64}, lws[1] = {64};
    run(name, kk, 1, gws, lws);
  }
  void ink_rung(const char* name, cl_kernel kk, size_t groups, size_t ls) {  // (one work-group per row / token)
    size_t gws[1] = {groups * ls}, lws[1] = {ls};
    run(name, kk, 1, gws, lws);
  }
  cl_kernel ink_k(const char* name) {
    return kernel("ops", "-DINK_LS=256 -DINK_D=" + std::to_string(hp.head_dim) + " -DINK_EXT=1024", name);
  }
  // y[T][N] = M x[T][K]
  void ink_mm(const char* name, const InkMat& m, cl_mem x, cl_mem y, int T, int Mp) {
    if (m.f) {
      cl_kernel kk = ink_k("ink_mm_f32");
      set_args(kk, 0, m.f, x, y, m.N, m.K, T);
      ink_run1(name, kk, (size_t)m.N * T);
      return;
    }
    const int a32 = 1;  // (per-32 scales: the activations of a reference check)
    quant_rows(x, m.K, Mp, T, a32);
    matmul(name, m.w, Mp, y, m.N, 0, 0, a32 != 0);
  }
  void ink_rmsnorm(cl_mem x, int xstride, cl_mem w, cl_mem out, int T) {
    cl_kernel kk = ink_k("ink_rmsnorm");
    int E = hp.n_embd;
    float eps = hp.eps;
    set_args(kk, 0, x, xstride, w, out, E, eps);
    ink_rung("ink_rmsnorm", kk, T, 256);
  }
  void ink_add(cl_mem a, cl_mem b, cl_mem out, int n) {
    cl_kernel kk = ink_k("ink_add");
    set_args(kk, 0, a, b, out, n);
    ink_run1("ink_add", kk, n);
  }
  void ink_copy(cl_mem src, size_t src_off, cl_mem dst, size_t dst_off, size_t bytes) {
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, src, dst, src_off, dst_off, bytes, 0, nullptr, nullptr));
  }
  // out[t][j] = the mix of the blocks (nb of them, each M x E apart in `blocks`) and cur with q's nq queries
  void ink_mix(cl_mem blocks, int nb, cl_mem cur, cl_mem q, int nq, cl_mem out, int T) {
    cl_kernel kk = ink_k("ink_mix");
    int64_t bs = (int64_t)max_chunk * hp.n_embd;
    int E = hp.n_embd;
    set_args(kk, 0, blocks ? blocks : ink_dummy_, nb, bs, cur, q, nq, out, E);
    ink_rung("ink_mix", kk, T, 256);
  }
  void ink_take(cl_mem src, int nq, int j, cl_mem out, int T) {
    cl_kernel kk = ink_k("ink_take");
    int E = hp.n_embd;
    set_args(kk, 0, src, nq, j, out, E, T);
    ink_run1("ink_take", kk, (size_t)E * T);
  }
  // llama's convolve over the chunk with the slot's state, then the state moved on (out != cur)
  void ink_conv(cl_mem cur, cl_mem w, cl_mem dyn, cl_mem state, cl_mem out, int C, int T) {
    cl_kernel kk = ink_k("ink_conv");
    int has = dyn ? 1 : 0, nh = hp.n_head_kv, NC = ink_.nc;
    float sc = 1.0f / std::sqrt((float)((dyn ? 2 : 1) * NC + 1));
    set_args(kk, 0, cur, w, dyn ? dyn : ink_dummy_, has, nh, state, out, C, T, NC, sc);
    ink_run1("ink_conv", kk, (size_t)C * T);
    cl_kernel ks = ink_k("ink_conv_state");
    set_args(ks, 0, cur, state, C, T, NC);
    ink_run1("ink_conv_state", ks, C);
  }
  // the controller of input c (0..3: q, k, v, ffn) on x: x - tanh(forget) x + tanh(gain) message
  void ink_control(const InkLayer& L, cl_mem x, int c, int T, int Mp) {
    const int E = hp.n_embd;
    int moff = c * E, mst = 4 * E, EE = E, TT = T, xs = E;
    cl_kernel k2 = ink_k("ink_rms2");
    set_args(k2, 0, x, xs, ink_msgs_, moff, mst, ink_cg_, EE);
    ink_rung("ink_rms2", k2, T, 256);
    ink_mm("ink_ctrl_down", L.ctrl_down[c], ink_cg_, ink_lo_, T, Mp);
    ink_mm("ink_ctrl_gain", L.ctrl_gain[c], ink_lo_, ink_gain_, T, Mp);
    ink_mm("ink_ctrl_forget", L.ctrl_forget[c], ink_lo_, ink_fgt_, T, Mp);
    cl_kernel ka = ink_k("ink_ctrl_apply");
    set_args(ka, 0, x, ink_fgt_, ink_gain_, ink_msgs_, moff, mst, EE, TT);
    ink_run1("ink_ctrl_apply", ka, (size_t)E * T);
  }

  // one layer in cache slot `slot` (llama's build_layer); msgs: a later loop pass (ink_msgs_ holds its messages);
  // ink_cur_ the stream in and out, nb the blocks (ink_blocks_), pushed to at a block's first layer
  void ink_layer(int slot, int T, int Mp, bool msgs, int& nb) {
    const InkSlot& S = inks_[slot];
    const int ip = S.ip;
    const InkLayer& L = inkl_[ip];
    const InkHp& k = ink_;
    const int E = hp.n_embd, D = hp.head_dim, H = hp.n_head, HKV = hp.n_head_kv, KG = HKV * D;
    const size_t rowE = (size_t)E * 4;
    // the attention inputs: a mix (one query, SWA; three, global) of the blocks and the stream, from layer 1 on
    cl_mem in[3] = {ink_cur_, ink_cur_, ink_cur_};
    if (L.res_q) {
      ink_mix(ink_blocks_, nb, ink_cur_, L.res_q, L.nq, ink_mix_, T);
      if (L.nq == 1 && !msgs) {
        ink_take(ink_mix_, 1, 0, ink_in_[0], T);
        in[0] = in[1] = in[2] = ink_in_[0];
      } else {
        for (int j = 0; j < 3; ++j) {
          ink_take(ink_mix_, L.nq, L.nq == 1 ? 0 : j, ink_in_[j], T);
          in[j] = ink_in_[j];
        }
      }
    } else if (msgs) {  // (layer 0 is never a loop layer: the loop core starts after a block)
      for (int j = 0; j < 3; ++j) {
        ink_copy(ink_cur_, 0, ink_in_[j], 0, rowE * T);
        in[j] = ink_in_[j];
      }
    }
    if (msgs)
      for (int j = 0; j < 3; ++j) ink_control(L, in[j], j, T, Mp);
    if (!L.glob) {
      if (msgs) {
        for (int j = 0; j < 3; ++j) ink_rmsnorm(in[j], E, L.an, ink_in_[j], T);
        for (int j = 0; j < 3; ++j) in[j] = ink_in_[j];
      } else {
        ink_rmsnorm(in[0], E, L.an, ink_in_[0], T);
        in[0] = in[1] = in[2] = ink_in_[0];
      }
    } else {
      for (int j = 0; j < 3; ++j) {
        ink_rmsnorm(in[j], E, L.inorm[j], ink_in_[j], T);
        in[j] = ink_in_[j];
      }
    }
    // the projections; K and V through their convolutions (global: the input-dependent taps)
    ink_mm("ink_q", L.q, in[0], ink_q_, T, Mp);
    ink_mm("ink_k", L.k, in[1], ink_k0_, T, Mp);
    ink_mm("ink_v", L.v, in[2], ink_v0_, T, Mp);
    ink_mm("ink_rel", L.rel, in[0], ink_r_, T, Mp);
    ink_mm("ink_sink", L.sink, in[0], ink_s_, T, Mp);
    ink_mm("ink_gate", L.gate, in[0], ink_g_, T, Mp);
    if (L.glob) {
      ink_mm("ink_kcp", L.kcp, in[1], ink_kd_, T, Mp);
      ink_mm("ink_vcp", L.vcp, in[2], ink_vd_, T, Mp);
    }
    ink_conv(ink_k0_, L.k_conv, L.glob ? ink_kd_ : nullptr, S.conv[0], ink_kc_, KG, T);
    ink_conv(ink_v0_, L.v_conv, L.glob ? ink_vd_ : nullptr, S.conv[1], ink_vc_, KG, T);
    // clips (q, r: per-head scales; global: the log-length factor), K's clip, V's RMS norm
    {
      cl_kernel kr = ink_k("ink_rows");
      const int gt = L.glob ? 1 : 0;  // (llama: tau on q and r of the global layers, whatever alpha's sign)
      auto rows = [&](cl_mem x, int nrow, int n, cl_mem sc, float thr, int tau, int mode) {
        int nr = nrow, nn = n, hs = sc ? 1 : 0, p0 = pos, md = mode, ut = tau;
        float th = thr, al = k.log_alpha, lf = k.log_floor, eps = hp.eps;
        set_args(kr, 0, x, nr, nn, sc ? sc : ink_dummy_, hs, th, ut, al, lf, p0, md, eps);
        ink_rung("ink_rows", kr, (size_t)T * nrow, 256);
      };
      rows(ink_q_, H, D, L.q_clip, k.qk_clip, gt, 0);
      rows(ink_r_, H, k.nrel, L.rel_clip, k.rel_clip, gt, 0);
      rows(ink_kc_, HKV, D, nullptr, k.qk_clip, 0, 0);
      rows(ink_vc_, HKV, D, nullptr, 0.0f, 0, 1);
    }
    {  // the slot's cache, then the attention over it (gated)
      cl_kernel ks = ink_k("ink_kv_store");
      int kg = KG, TT = T, p0 = pos, cap = S.cap;
      set_args(ks, 0, ink_kc_, ink_vc_, S.kc, S.vc, kg, TT, p0, cap);
      ink_run1("ink_kv_store", ks, (size_t)KG * T);
      cl_kernel ka = ink_k("ink_attn");
      int ext = L.ext, nrel = k.nrel, HH = H, HK = HKV, win = L.glob ? 0 : k.swa;
      set_args(ka, 0, ink_q_, ink_r_, ink_s_, ink_g_, S.kc, S.vc, L.rel_bias, ext, nrel, ink_att_, HH, HK, p0, win, cap);
      ink_rung("ink_attn", ka, (size_t)T * H, (size_t)D);
    }
    ink_mm("ink_o", L.o, ink_att_, ink_ao_, T, Mp);
    ink_conv(ink_ao_, L.post_conv, nullptr, S.conv[2], ink_tmp_, E, T);
    // a block's first layer: the stream before it joins the blocks, the attention output starts a new one
    if (ip % k.block == 0) {
      ink_copy(ink_cur_, 0, ink_blocks_, (size_t)nb * max_chunk * rowE, rowE * T);
      ++nb;
      ink_copy(ink_tmp_, 0, ink_cur_, 0, rowE * T);
    } else {
      ink_add(ink_cur_, ink_tmp_, ink_cur_, E * T);
    }
    // the MoE: its input a mix over the blocks (with the new one) and the stream, the controller, ffn_norm
    ink_mix(ink_blocks_, nb, ink_cur_, L.ffn_res_q, 1, ink_xm_, T);
    if (msgs) ink_control(L, ink_xm_, 3, T, Mp);
    ink_rmsnorm(ink_xm_, E, L.fn, ink_xm_, T);
    ink_mm("ink_router", L.router, ink_xm_, ink_rl_, T, Mp);
    {
      cl_kernel kr = ink_k("ink_route");
      int hb = L.probs_b ? 1 : 0, NE = hp.n_expert, NU = L.nu;
      float ws = k.wscale;
      set_args(kr, 0, ink_rl_, L.probs_b ? L.probs_b : ink_dummy_, hb, NE, NU, ws, ink_ids_, ink_w_);
      ink_rung("ink_route", kr, T, 256);
      const int F = hp.n_ff_exp, rows = T * NU;
      cl_kernel km = ink_k("ink_expert_mv");
      int ty = (int)L.gu_type, N2 = 2 * F, EE = E, xd = NU, R = rows;
      uint64_t rb = L.gu_rb;
      set_args(km, 0, L.gu, ty, rb, N2, EE, ink_ids_, ink_xm_, xd, ink_gu_, R, ink_kv_);
      ink_run1("ink_expert_gu", km, (size_t)rows * 2 * F);
      cl_kernel kt = ink_k("ink_situ");
      int hs = L.gu_scale ? 1 : 0, FF = F;
      float b = k.situ_b, lb = k.situ_lb;
      set_args(kt, 0, ink_gu_, ink_ids_, L.gu_scale ? L.gu_scale : ink_dummy_, hs, ink_act_, FF, R, b, lb);
      ink_run1("ink_situ", kt, (size_t)rows * F);
      int ty2 = (int)L.dn_type, one = 1;
      uint64_t rb2 = L.dn_rb;
      set_args(km, 0, L.dn, ty2, rb2, EE, FF, ink_ids_, ink_act_, one, ink_de_, R, ink_kv_);
      ink_run1("ink_expert_dn", km, (size_t)rows * E);
      cl_kernel kx = ink_k("ink_expert_sum");
      int hd = L.dn_scale ? 1 : 0, TT = T;
      set_args(kx, 0, ink_de_, ink_ids_, ink_w_, L.dn_scale ? L.dn_scale : ink_dummy_, hd, ink_mo_, EE, NU, TT);
      ink_run1("ink_expert_sum", kx, (size_t)E * T);
    }
    ink_conv(ink_mo_, L.ffn_conv, nullptr, S.conv[3], ink_tmp_, E, T);
    ink_add(ink_cur_, ink_tmp_, ink_cur_, E * T);
  }

  // The whole chunk (run_layer's first layer runs it; x_ holds the embeddings, and on return the output mix's rows,
  // which head() then normalizes with output_norm and multiplies by the head)
  void ink_forward(int T, int Mp) {
    const InkHp& k = ink_;
    const int E = hp.n_embd;
    const size_t rowE = (size_t)E * 4, blockB = (size_t)max_chunk * rowE;
    ink_rmsnorm(x_, E, ink_tok_norm_, ink_cur_, T);
    int nb = 0;
    for (int il = 0; il < k.start; ++il) ink_layer(il, T, Mp, false, nb);
    if (k.start < k.end) {
      const int np = nb;
      ink_copy(ink_cur_, 0, ink_prelude_, 0, rowE * T);
      for (int pass = 0; pass < k.passes; ++pass) {
        int nbank = 0;
        if (pass > 0) {  // the bank: the previous pass's blocks after the first core one, and its output
          nbank = nb - (np + 1);
          for (int b = 0; b < nbank; ++b) ink_copy(ink_blocks_, (size_t)(np + 1 + b) * blockB, ink_bank_, (size_t)b * blockB, rowE * T);
          ink_copy(ink_cur_, 0, ink_bank_out_, 0, rowE * T);
          ink_copy(ink_prelude_, 0, ink_cur_, 0, rowE * T);
          nb = np;
        }
        for (int ip = k.start; ip < k.end; ++ip) {
          const int slot = pass == 0 ? ip : hp.n_layer + (pass - 1) * (k.end - k.start) + (ip - k.start);
          if (pass > 0) ink_mix(nbank ? ink_bank_ : nullptr, nbank, ink_bank_out_, inkl_[ip].msg_q, 4, ink_msgs_, T);
          ink_layer(slot, T, Mp, pass > 0, nb);
        }
      }
    }
    for (int il = k.end; il < hp.n_layer; ++il) ink_layer(il, T, Mp, false, nb);
    // the output's mix, into x_ (head(): output_norm, the head, the softcap)
    ink_mix(ink_blocks_, nb, ink_cur_, ink_out_res_q_, 1, x_, T);
  }

  // the tokens' count (vocab_size_unpadded): the logits that are tokens, the head's first rows (runz --kl-ref)
 public:
  int ink_vocab() const { return ink_.nv; }
 private:

  // the logits past the vocabulary to -inf (rows of nl)
  void ink_mask_logits(int rows, int nl) {
    if (nl <= ink_.nv) return;
    cl_kernel kk = ink_k("ink_mask_logits");
    int n = nl, nv = ink_.nv, r = rows;
    set_args(kk, 0, logits_, n, nv, r);
    ink_run1("ink_mask_logits", kk, (size_t)(nl - ink_.nv) * rows);
  }
