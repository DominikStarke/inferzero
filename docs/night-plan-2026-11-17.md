# The night plan of 2026-11-17: generation, prompts and long prompts

The user's requests (2026-11-17, evening): faster generation (TG) for the 27B, the 35B-A3B and Flash Next (QFN, "like a
snail"); the B70's 4K prompt past 3,000 tok/s for the 27B, and the same work on the B60, B580 and B390; long prompts
(64K-262K); a plan for all of it, executed tonight. Every change keeps AGENTS.md's rules (KL / bit checks, a
before-after row, a CHANGELOG entry, a deep self-audit before each commit). What only a card can measure is asked of
its owner with exact commands.

## Where each one stands (measured)

| | Card | Now | Where the time goes |
|---|---|---|---|
| 27B prompt 4K | B70 | 2,897 tok/s (2026.11.17.06 defaults) | GEMMs ~40 %, `gdn_recur` 236 ms of 1,489 (16 %), `h8_act` 88 + `rmsnorm_quant` 70, `gated_norm_quant` 40, attention ~60 |
| 27B generation | B70 | 35.2 plain, 70.6 MTP (4 drafts) | the GEMVs at ~71 % of 559 GB/s; the codebook types (IQ3_S, IQ3_XXS) bound by their decoding (Q4_K reaches ~530 GB/s, they 340-450) |
| 35B-A3B generation | B70 | 95-98 plain (10.0 ms a step) | ~1.6 GB a token is ~27 % of the bandwidth: launches and the gaps between ~600 dependent kernels |
| Flash Next prompt 4K | B70 | 1,414 | `moe_gateup` 506, `moe_down` 494, `decode_w8` 473 ms (every expert decoded again in each 2048-token chunk) |
| Flash Next generation | B70 | 38.6 warm (256 tokens), 6.6 in tg64; **5.44 with MTP** | MTP: 93 main passes in 47 s, ~506 ms a pass against ~26 ms a plain token |
| Flash Next generation | laptop | 4.75 (96 tokens) | `ensure` (missing experts) 1.6 ms a layer wait, 7.4 of 20.2 s |
| Long prompts | B60 / B70 | 27B 1,022 -> 420 tok/s from 8K to 131K (B60) | attention's share grows with the square of the length |

Outside work (research 2026-11-17): no public Arc decode beats these (llama.cpp SYCL on a B70: 27B 20-27, 30B-A3B ~93);
their documented best is ~66 % of the bandwidth. Ideas worth taking: verify GEMVs that read the weights once for all
k tokens (ours do: `gemv_mt`), 2 MTP drafts best for a dense 27B in llama.cpp's B70 runs, MoE drafts limited by their
experts' cost (EVICT), predicted-expert prefetch (pre-gating), fused decode ops, one kernel per DeltaNet layer.

## The items, in order

1. **N1, Flash Next MTP (generation, the largest single loss).** Find why a verification pass costs ~20 plain tokens
   (laptop: `--mtp --mtp-model` with `--prof-gen`; B70: 506 ms a pass), fix it bit-identically to the plain path's
   tokens, ask joshdev-de to re-run his README + 256 pair.
2. **N2, Flash Next prompt chunk.** 2048-token chunks decode every expert's gate / up twice per 4K prompt since
   `ARC_MOE_GEMM_PAD` became the default. Ask the B70 for `--chunk 2048 / 4096 / 8192` at 4K and 16K; make the winner the
   default on discrete cards (B60 / B580 the same chip) if the expert cache allows.
3. **N3, the 27B's generation on the B-series: the IQ3_S planar repack** (`ARC_IQ3S_TILE=1`, branch `iq3s-tile`,
   bit-identical). The laptop is bandwidth-bound and gained nothing, but the B70 is decode-bound, where the repack's
   block reads without the local window and barriers are meant to help. Merge it opt-in and ask for a generation A/B.
   Also ask for MTP drafts 2 / 3 / 4 (llama.cpp's best on a B70 was 2).
4. **N4, the B70's prompt past 3,000.** `ARC_H8_NORM=1` (the norm and the rotation in one kernel, bit-identical; a B60
   measured no gain on 11-14, before the 128-register default) on the B70, with the H8 register file at 128 now; and
   the DeltaNet's prompt kernel (`gdn_recur` 16 %): `ARC_GDN_L1PF` 8 (chumbleyjl's best) against 2.
5. **N5, the 35B's generation: fewer launches.** Count the kernels of one token (`ARC_LAUNCH_STATS=1`) and fuse the
   ones that can be fused bit for bit (the router / top-k / tiles, the shared expert's gate, the combine).
6. **N6, Flash Next generation without MTP:** the next layer's experts read while this layer runs (the router's
   input is known one layer late, so a prefetch by the previous token's routes or a cheap pre-gate); measured first
   on the laptop, where `ensure` is 36 % of a token.
7. **N7, long prompts:** a B70 profile of the 27B and 35B at 64K and 131K with today's defaults (flash attention on
   XMX, the 2D GEMM) to see attention's share now; then the next attention change from it.
8. **N8, the other cards:** every default above that is "discrete cards" covers the B60 and B580; the B390 (Panther
   Lake) gets each one opt-in with an A/B request (gee456, issue #1).

## Results

- **N1 (2026-11-17 evening, the laptop):** not reproduced. After README.md (5,798 tokens), 96 tokens: plain 4.75-5.02
  tok/s, MTP (the Q4_0 head) 5.78-6.18, 2.74 tokens a main pass. One first run gave 1.27 with 0.12 GiB of RAM free
  (paging); it did not come back in four runs. The B70's 5.44 is then on what only a discrete card runs (its RAM
  tier, the B-series GEMV rows for 2-8 tokens, the XMX paths, the Q8_0 head): a `--prof-gen` of its verify passes is
  asked (issue #27). The laptop's profile of 35 verify passes: `moe_gateup` 1,164, `moe_down` 934, `lm_head` 885 ms of
  6,590, nothing out of proportion.
