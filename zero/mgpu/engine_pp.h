// One model over several GPUs by layers (--split layer; QUEUE "a layer split beside the tensor split", L1 + L2;
// docs/tiered-placement.md P2). Each GPU is a stage: an Engine over a range of the layers (EngineOptions pp_begin /
// pp_end), with those layers' weights, KV caches and DeltaNet states only; the first stage embeds, the last has the
// norm, the head and the MTP block. A token's rows go through the stages in turn: stage r's rows are copied to stage
// r + 1 (through host memory: the first form, L2's GPU-to-GPU copies later). The stages run the same kernels on the
// same rows in the same order as one GPU does, so the logits are the same bit for bit (ARC_TP_SIM=1: every stage on GPU
// 0, the check).
// What it is for: a model that a card cannot hold alone without paging, cards of different kinds, and links too slow
// for the tensor split's sums per layer. What it is not: faster for one request than one card that holds the model,
// since a token visits every stage in turn (the tensor split is the fast form on well-linked matching cards).
// Not yet (QUEUE): the server's slots and sessions, MTP over stages' verification beyond plain greedy drafting, the
// prompts pipelined over the stages (L3), two requests in flight (L4).
#pragma once
#include <memory>
#include <string>
#include <vector>

class PpEngine {
 public:
  // shares: each stage's share of the layers (empty: equal); a stage gets at least one layer
  PpEngine(const std::string& path, EngineOptions o, int n, const std::vector<double>& shares = {}) {
    if (n < 2 || n > 64) throw std::runtime_error("--split layer: 2 to 64 GPUs");
    const bool sim = getenv("ARC_TP_SIM") && atoi(getenv("ARC_TP_SIM")) != 0;
    {
      ClRt rt;  // (the runtime, before the stages)
      if (!rt.init()) throw std::runtime_error("cannot start Level Zero");
      std::string err;
      if (ClRt::open_devices(n, sim, err) != n) throw std::runtime_error("--gpus " + std::to_string(n) + ": " + err);
    }
    int NL = 0;
    {
      GgufFile g(path);
      const std::string a = g.get_str("general.architecture");
      NL = (int)g.get_int(a + ".block_count", 0) - (int)g.get_int(a + ".nextn_predict_layers", 0);  // (not the MTP layer)
    }
    if (NL < n) throw std::runtime_error("--split layer: " + std::to_string(NL) + " layers for " + std::to_string(n) + " GPUs");
    // (the cut: by the shares, each stage at least one layer)
    std::vector<double> s = shares;
    if ((int)s.size() != n) s.assign(n, 1.0);
    double sum = 0;
    for (double x : s) sum += std::max(0.0, x);
    std::vector<int> cut(n + 1, 0);
    double acc = 0;
    for (int r = 0; r < n; ++r) {
      acc += std::max(0.0, s[r]);
      cut[r + 1] = r + 1 == n ? NL : std::clamp((int)(NL * acc / sum + 0.5), cut[r] + 1, NL - (n - 1 - r));
    }
    zlog::eprintf("inferzero: %d GPUs by layers%s:", n, sim ? " (ARC_TP_SIM: all on GPU 0)" : "");
    for (int r = 0; r < n; ++r) zlog::eprintf(" GPU %d layers %d-%d", r, cut[r], cut[r + 1] - 1);
    zlog::eprintf("\n");
    zrt::Dev* const saved = zrt::cur;
    for (int r = 0; r < n; ++r) {
      zrt::cur = zrt::devs[r];
      EngineOptions so = o;
      so.pp_begin = cut[r];
      so.pp_end = cut[r + 1];
      if (r + 1 < n) so.mtp = false;  // (the MTP block is the last stage's)
      stages_.push_back(std::make_unique<Engine>(path, so));
    }
    zrt::cur = saved;
    rows_.resize(n);
  }
  int stages() const { return (int)stages_.size(); }
  Engine& stage(int r) { return *stages_[r]; }
  Engine& last() { return *stages_.back(); }
  // the tokens through every stage; the last stage's logits (as Engine::forward)
  const std::vector<float>& forward(const int* tokens, int n) {
    zrt::Dev* const saved = zrt::cur;
    struct Back { zrt::Dev* s; ~Back() { zrt::cur = s; } } back{saved};
    const int S = stages();
    for (int r = 0; r < S; ++r) {
      zrt::cur = zrt::devs[r];
      Engine& e = *stages_[r];
      e.pp_in_ = r ? rows_[r - 1].data() : nullptr;
      e.pp_out_ = r + 1 < S ? &rows_[r] : nullptr;
      const std::vector<float>& l = e.forward(tokens, n);
      if (r + 1 == S) return l;
    }
    static const std::vector<float> none;
    return none;
  }
  void reset() {
    zrt::Dev* const saved = zrt::cur;
    for (int r = 0; r < stages(); ++r) {
      zrt::cur = zrt::devs[r];
      stages_[r]->reset();
    }
    zrt::cur = saved;
  }
  size_t gpu_bytes(int r) const { return stages_[r]->gpu_bytes; }

 private:
  std::vector<std::unique_ptr<Engine>> stages_;
  std::vector<std::vector<float>> rows_;  // (stage r's output rows, the next stage's input)
};
