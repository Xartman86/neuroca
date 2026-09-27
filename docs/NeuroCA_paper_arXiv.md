# NeuroCA: A Hybrid Neural Network Based on Cellular Automata

## From Register Neurons to a Verifiable Code Generator — Series v0.1–v120

**NeuroCA Laboratory**

*Independent research / DeepSeek Harness lab*

**Comments:** 26 pages, 17 references. Submitted to arXiv category **cs.NE**
(Neural and Evolutionary Computing); related: cs.LG, cs.SE, cs.PL.

**Document version:** 26.09.2026 · **Registry etalon:** `v120big50_kit2b` (26 versions)

> **Reproducibility-first preprint.** All artifacts (checkpoints, corpora,
> reports, registry) are versioned and referenced in §8. The evaluation harness is
> exec-based (compile + real tests in a sandbox); the format guard and the Tier1
> audit are described in §4.3 and §6.4.

---

## Abstract

We present NeuroCA, a research neural-network architecture in which a neuron is
interpreted as a **W-bit register on an N-dimensional torus**, and the whole network
as a binary cellular automaton (CA) in N+1 dimensions. Unlike classical neural
networks that train millions of weights "blindly", NeuroCA keeps the main
computational dynamics fixed and local (given by cellular-automaton rules), while
only a compact readout layer over deterministic reservoir features is trained.
Across the version series v0.1–v120 the architecture evolved from a single 4D
machine (5832 bits) through agent networks, rule evolution, STaR self-training and
a dual-loop "generate + assemble" system, to a hybrid of a **register CA substrate
(~530 features) and a Transformer/MoE (53M parameters)** solving 33 algorithmic
tasks with a full exec-run of **658/660 = 99.7%** (etalon `v120big50_kit2b`).

Key measured results and lessons of the series: (1) model capacity matters — growing
parameters from 7.2M to 53M removes the "binarity" of training profiles (feature
overlap 0.764→0.285); (2) quality growth comes **from data, not parameters**
(544→648→658 as the corpus expands; epoch plateau); (3) in a controlled ablation at
the current scale the CA substrate adds `+5.0 ± 5.8` p.p. (within noise), while data
has a ~18× larger effect; (4) synthetic elisions/template steps in the corpus
**hurt** (base-only > synthetic); (5) the LLM-teacher loop (qwen2.5:14b, yield ~51%
EXEC-valid) works: a real defect in `second_max` was fixed 0→20/20; (6) the
distillation pilot with an external LLM achieved 9/10 on the first attempt; (6a) the
decisive edge-data test (kit2c, 1161 rows) **refuted** the hypothesis "Tier1 zeros =
uncovered inputs": edges gave no gain (660 = 656, Tier1 485 vs 487 for kit2b) — the
next lever is staged loss decomposition; (7) "the harness can lie": the eval-test
format bug understated the result (82.4% instead of the real 99.8%), and a Tier1
evaluation defect (overwrite of a repeated family) makes the 640 denominator
ambiguous — a metric is only as honest as the harness.

We discuss limitations (closed vocabulary V=1078, narrow base tests, no functional
tests for Stack/Queue, Tier1 failures), and directions: staged loss decomposition,
experts 8→16, Tier2 compositions via distillation, and a controlled substrate
ablation at a ×5 data scale.

**Keywords:** cellular automata, register machine, reservoir computing, code
generation, STaR self-training, LLM teacher, distillation, Mixture-of-Experts,
exec-verification, honest metrics.

---

## 1. Introduction

### 1.1 Motivation

A classical neural network trains millions of weights and remains a "black box":
its state is not observable and result verification is heuristic. NeuroCA proposes
an alternative: **train a thin readout layer, while the main computational load is
carried by the deterministic local dynamics of a cellular automaton**, where each
neuron is a register of bits, and the rules are local, interpretable, and not
trained.

The motivation is threefold:

1. **Interpretability** — the machine state is a bit array; it can be observed,
   edited, and verified.
2. **Robustness** — local rules provide self-repair: a damaged state is restored by
   the dynamics.
3. **Honest verification** — in the applied setting (code generation) the result is
   not assessed subjectively but **executed**: compile + real tests in a sandbox.

### 1.2 Problem statement and working hypothesis

**Task:** given a Russian-language formulation of an algorithmic task (e.g.,
"задача find_max" — "task find_max"), generate valid Python code that passes
compilation and functional exec on tests.

**Working hypothesis:** the dynamics of a cellular automaton over registers can
replace most of the trained capacity, leaving to learning only the "reading" of
reservoir features. The final metric is the share of generations that are actually
executable and correct on tests (EXEC).

### 1.3 Contributions

1. The architecture "neuron = register, network = CA in N+1 dimensions" with an
   open set of local bit rules (majority, Toom, HDC/VSA, Life, Kauffman, identity).
2. The hybrid scheme **HybridPrefix + Substrate3**: a trainable Transformer/MoE over
   a deterministic CA reservoir (~530 features), with CA-aware gating.
3. A complete honest-CI loop: version registry, format guard, 660 exec-protocol,
   regression gate, article auto-sync (plugin `neuroca-lab`).
4. An **LLM-teacher** loop (an external LLM as a "more knowledgeable other") and a
   distillation pilot of a verified corpus.
5. Systematic recording of **negative results** (synthetic data, LoRA, head blocks,
   frozen core, hybrid profiles) as scientifically valuable data.

---

## 2. Related Work

Cellular automata as computational models were systematized by Wolfram [1].
Growing Neural Cellular Automata demonstrated self-repair and morphogenesis [2].
Reservoir computing uses fixed nonlinear dynamics with a trainable readout [3];
NeuroCA inherits this idea, replacing the analog reservoir with a bit-level CA.
Self-taught reasoner (STaR) [4] is the basis of the self-correction loop; language
model scaling laws [5, 6] and emergent abilities [7] frame the role of capacity
(measured here as G1: feature overlap 0.285 at 53M). Mixtures of agents [8] and
MoE layers [9] are the architectural basis for expert expansion. Lindenmayer
L-systems [10] and cognitive maps (O'Keefe, Moser; Tolman) [11, 12] motivate the
"living text" and spatial-understanding directions. Interiorization (Vygotsky) [13],
dialogism (Bakhtin) [14], predictive coding (Friston) [15], and dual-process theory
(Kahneman) [16] provide the theoretical frame for an internal-dialogue analog. LoRA
[17] was used in negative experiments of modular growth.

## 3. Methods

### 3.1 RegisterCA core: neuron = register

The machine state is a `uint8` array of shape `spatial_shape + (W,)`. A cell of the
torus lattice holds a W-bit register; bit `b` of a cell interacts with spatial
neighbors (same bit) and with register neighbors `b±1` of the same cell — **the
register becomes the (N+1)-th axis** of the machine. The flagship configuration: a
9×9×9 torus × 8 bits = **5832 bits**.

Core API: `reset(zeros|random)`, `set_bitplane` (input injection),
`step(n, noise_p, async_frac)` (synchronous/asynchronous dynamics with noise),
`features(bits|grid|flat)`, `bitplane_densities`, `popcount`. Throughput is ~130
million bit-updates/s on CPU without bit-packing.

Measured fact "register as an axis": without bit-plane coupling the signal does not
spread across the register; coupling (Toom/HDC rules) genuinely transfers
information between bits of a cell (v0.1 experiments: 4D-machine dynamics gives
0.929 vs 0.844 for static injection, +8.5 p.p.).

### 3.2 Local dynamics rules

All rules are purely bit-level; `MAJ3(a,b,c) = (a&b)|(a&c)|(b&c)`.

| Rule | Mechanics |
|---|---|
| `majority_nd` | majority of 2N+1 votes (N — torus dimension) |
| `toom_reg` | MAJ(axis-0 neighbor, center, register bit b+1) — Toom's rule |
| `hdc` | R ← MAJ(R, rot_k(R(+e_x)), rot_m(R(+e_y))) — HDC/VSA binding |
| `life_reg` | Life in N+1 dimensions |
| `kauffman` | K=2 random inputs + own LUT (subject of evolution) |
| `identity` | identity (delay-line, baseline) |

### 3.3 Substrate3 features (CA reservoir)

`Substrate3` is a deterministic set of ~530 CA-state features computed at each
decoding step: a shift chain of the last 12 tokens, syntax detectors, ripple-carry
counters, phase clocks, block counters 8/16/32/64, conjunctions, and RBN subnets.
Features are computed in a GPU cache and concatenated with the transformer state at
each token. The substrate is **not trained** and **not partitioned** (shared memory
= the abstraction mechanism; probe qtree 0.826).

### 3.4 HybridPrefix architecture

```
task (Russian text)
   |  tokenization (reversible, round-trip 100%, V=1078)
   ▼
┌────────────────────────────────────────────────┐
│ HybridPrefix:                                  │
│  tok + pos → L×[Attention + MoE(8 experts,     │
│  top-3) + Substrate gating]                   │
│  d=256, L=6, H=8, FDIM=530, R=2               │
│  → head(torch.cat([h, last_feats]))           │
└────────────────────────────────────────────────┘
   |  cascade: <A> analysis → <S> strategy → <C> scheme → <K> code
   ▼
Python code → compile → EXEC on tests (sandbox)
```

Parameters: Branch A — 53M (d=256, L=6, H=8, MoE 8/3, R=2); early versions —
7.2M (V=1078). The **entire** transformer is trained (not a "thin layer"). The
cascade agent "task → analysis → strategy → scheme → code": early stages use an
internal "pidgin" (in-vocab), code is the target Python.

### 3.5 Training

- Optimization: AdamW (lr=3e-4 for scratch, **lr=1e-4 for fine-tuning from a
  checkpoint** — 3e-4 destabilizes: ep112 76→36), weight_decay 1e-4, cosine
  annealing, bf16 autocast + GradScaler, grad clip 1.0, batch 160–256.
- Training on the GPU feature cache; data are (prefix, next-token) pairs from
  `labels/*.jsonl` / `.npz` corpora (up to 153–155K pairs).
- STaR self-training: generate → filter (compile+exec) → fine-tune on the mixture;
  anti-saturation (skeleton cap, double weight for new examples) unlocks rounds 2+.
- Full training v0.59–v0.61: 0.98M parameters, WIN=150; records rate 69.27%,
  E 39.72% before corpus expansion.

### 3.6 LLM-teacher loop and distillation

**Teacher:** Ollama qwen2.5:14b (semantics), qwen2.5-coder:14b / deepseek-coder
(code), backup minimax-m3:cloud (pilot G6: 9/10 first-try, latency 0.7–2.5s).
Prompt: "task, analysis, strategy, scheme, code, tests"; double filter (vocabulary
parser + compile + exec) → only **verified rows** enter the corpus.

Measurements: 208 generations → 106 EXEC-valid (~51% yield) in ~30 min; a real
defect in `second_max` (0/20 in v114b) was fixed to 20/20; `average` 15→19/20.
Batch generation of the full cascade (B=4) is ×2.3 faster and better than
stage-by-stage (fixed bubble_sort).

**Distillation (path "b"):** NeuroCA 53M = planner (task → in-vocab
analysis/strategy) + LLM-generator (general code) + verifier (exec) + retry loop;
verified rows are distilled into the student corpus. This removes the closed
vocabulary blocker for Tier2 compositions (the teacher reads arbitrary names).

---

## 4. Experimental Setup

### 4.1 Corpora

The corpus grew: 91 → 419 → 519 examples (v112–v114c), then expansion to 33
families (v115+): 2094 rows (kit2: base 1054 / A-elision 522 / R-steps 483 /
weak_fix 35), kit2_edges +72 edge rows (24 families × 3 variants), Branch A —
153–155K pairs (v121: 1532 rows, weight 3–5, UNK=0). **Decision:** synthetic
(elisions/template steps) is excluded from the corpus — base-only > synthetic.

### 4.2 Benchmarks

| Bench | Composition | What it measures |
|---|---|---|
| **660 (base)** | 33 families × 20 samples | generation stability on base tests |
| **Tier1** | 32 bench_tier1 items (31 unique families) × 20 | behavior on **new inputs** (edges: `[]→None`, `2^-1`, `−123`, tabs, `−1234`/10^9) with the same prompts |
| **Tier2** | 12 v122 compositions × 5–20 | skill compositions (sort_and_search etc.) — out-of-corpus |

Measured: 660 = 658/660 (99.7%), Tier1 = 487 over saved entries (31×20 = 620; the
declared 640 denominator is ambiguous — audit §6.4), Tier2 = 0/60 (OOV-name
blocker, §6.5).

### 4.3 Eval protocol and format guard

1. **Format guard** (mandatory): tests of the form `((args...), expected)`, NOT
   `(a, b, expected)` — a historical eval bug since v112 "killed" all
   multi-argument functions (artifact: 82.4% instead of the real 99.8%).
2. 660 run through the 16-thread CPU pool (exec harness, ~1200–1500 s), exec cache
   by code hash.
3. Per-sample accounting: `{exec, compile, name}`; the main metric is **exec**.
4. Auto-detection of log encodings (UTF-8/UTF-16/binary).

### 4.4 Gates and promotion criteria

| Gate | Condition | Result |
|---|---|---|
| G0 | Tier2 EXEC 100% | PASSED (data ready) |
| G1 | 50M overlap < 0.3 | **PASSED** (0.285) → Branch A |
| G2 | 660 ≥ 651, Tier1 ≥ 529, Tier2 ≥ 30% | not strictly met (Tier2 not measurable) |
| G6 | distillation pilot 9/10 | **PASSED** |
| Regression gate | pct drop ≥ 2 p.p. OR any task ≥ 20% | FAIL for v114d, v116, v117e etc. |
| STOP-REGRESSION | −14 on 8 seeds | auto-rollback to old peak |

Promotion to etalon requires a PASS gate **and human confirmation** (auto-promotion
is forbidden; lesson v114d).

## 5. Results

### 5.1 Series timeline (key milestones)

| Version(s) | Method | Key result |
|---|---|---|
| v0.1 | 4D machine 9×9×9×8 | digits 0.929 (+8.5 p.p. by dynamics); 130M bit/s |
| v0.2–0.3 | agents + hierarchy | routing 1.000; hierarchy 0.564 vs 0.262 |
| v0.6–0.11 | ES rule evolution | honest negative; diversity = model property |
| v0.24–0.30 | sliding window + linear | ppl 2.00; reservoir saturation; P ~ ppl^−len |
| v0.34–0.36 | TinyGPT vocab500 + mono-detok | **42%** (was 0.13%) → **62.1%** with names+F1 |
| v0.39 | STaR | **73.7%** (+12.3 p.p.) |
| v0.42–0.48 | substrate + MLP → hybrid | 20.6% → **57.1%** (×440 vs v0.28) |
| v0.49–0.54 | STaR r2 + assembly + critic | **65.09%**; best-of-4 **97.62%**; E 99.75%, rate×E 0.995 |
| v0.59–0.61 | full training 0.98M | rate 69.27% — record; E 39.72% |
| v112–v114c | +teacher loop, honest eval | EXEC 459/460 = 99.8% (after format fix) |
| v115–v117 | expansion to 33 families | 583 → 628/660; second_max — problem family |
| v118_sub / A2 | substrate ablation | +5.0 ± 5.8 (noise); data ~18× more important |
| v119–v120 | canonization, warmup, replay | remove_spaces 0→20; Tier1 record 529 (replay2) |
| v120big50_v121 | Branch A 53M, 153K pairs | 648/660 (old 450, new 198), Tier1 474 |
| **v120big50_kit2b** | Branch A, base-only 1089 rows | **658/660 = 99.7%**, Tier1 487 — **etalon** |

### 5.2 Version registry (26 entries, etalon ★)

| # | Version | Verdict | EXEC/total | pct | Note |
|---|---|---|---|---|---|
| 1 | v114c | in_progress | 459/460 | 99.8% | historical etalon, 23 tasks |
| 2 | v114d | rejected | 435/460 | 94.6% | −5.2 p.p.; average 19→0 |
| 3 | v115 | in_progress | 583/660 | 88.3% | expansion to 33 families |
| 4 | v116 | rejected | 551/660 | 83.5% | −10.4; power 20→6 |
| 5 | v116b | rejected | 577/660 | 87.4% | −5.0; second_max 20→0 |
| 6 | v116c | under_review | 613/660 | 92.9% | −0.2; second_max 20→18 |
| 7 | v117 | under_review | 628/660 | 95.2% | −0.4; second_max 20→17 |
| 8 | v117e | rejected | 589/660 | 89.2% | −6.7; remove_spaces 20→3 |
| 9 | v118_sub | experiment_closed | 587/660 | 88.9% | ablation: +5.0 ± 5.8 (noise) |
| 10 | v118_a2control | control | 497/660 | 75.3% | no seeding: merge_sorted 0 |
| 11 | v119transfer | under_review | 618/660 | 93.6% | UNK canonization: remove_spaces 0→20 |
| 12 | v119scratch | control | 527/660 | 75.3% | canonization w/o transfer: +40 SWA |
| 13 | v119replay | control | 541/660 | 82.0% | Tier1 SWA 502 — series record |
| 14 | v120frozen_blocks | control | 621/660 | 94.1% | frozen core + head blocks: old 458 |
| 15 | v120warmup | control | 625/660 | 94.7% | warmup base-only: caesar 20→15 |
| 16 | v120warmup2 | rejected (replaced) | 651/660 | 98.6% | base-profile balance champion |
| 17 | v120replay2 | control | 596/660 | 90.3% | **Tier1 record 529/640** |
| 18 | v120replay3 | control | 596/660 | 90.3% | determinism confirmation |
| 19 | v120hybrid | control | 650/660 | 98.5% | base/edge compromise not confirmed |
| 20 | v120edgeblock | control | 264/264 (check) | 100% | edge head does not specialize |
| 21 | v120lora_e1 | control | 264→109 | — | LoRA edge-only destroys base |
| 22 | v120lora | control | 264→199 | — | LoRA 50/50 too |
| 23 | v120big50 | G1-PASSED | 114/264 (check) | 43% | overlap 0.285 → Branch A |
| 24 | v120big50_v121 | Branch A final | 648/660 | 98.2% | new 198 > etalon 192; Tier2 0/60 |
| 25 | **★ v120big50_kit2b** | **etalon** | **658/660** | **99.7%** | base-only 1089 rows; Tier1 487 |
| 26 | v120big50_kit2c | final (etalon unchanged) | 656/660 | 99.4% | base+weak_fix+edges (1161 rows); old 460, new 196, Tier1 485 — **edges gave no gain (noise ±2)** |

### 5.3 Per-task breakdown of the etalon (660)

All 33 families ≥ 18/20; the only base failure is `is_balanced_parentheses` (2 of
20 samples). Old 23 families: 460/460 (100%). New 10: 198/200 (99%).
**Tier1 weak spots** (0/20 while 660 = 20/20): `find_max`, `power`, `digit_sum`,
`remove_spaces`, `count_digits` — interpretation: uncovered edge inputs (negative
numbers, `[]→None`, tabs, `0^0`, `2^-1`), not rephrasings.

### 5.4 Negative results (scientifically valuable)

1. **Synthetic data (A-elision/R-steps) in the corpus** — v121warmupA1: retrieval
   19%→15%, 660 −8; kit2b A/B: base-only (1089) > synthetic (1532) at 53M.
2. **LoRA channels over a frozen core** — v120lora_e1 264→109, v120lora 264→199:
   corrections dominate the frozen core without a router.
3. **Head-only blocks over a frozen core** — v120frozen_blocks 621 (new 163 =
   v117), v120edgeblock (edge head overfits: Tier1 49→21): heads cannot carry
   specialization; depth is required.
4. **Hybrid profile (warmup + edge)** — v120hybrid 650/454: no compromise between
   the base champion (651/477) and the edge champion (596/529) exists; the space is
   binary until capacity scaling.
5. **Substrate at current scale** — v118_sub: +5.0 ± 5.8 p.p. (within noise).

### 5.5 Ablation: data vs substrate vs capacity

- Control A2 (scratch without seeding, 497/660) vs v118_sub (586.7): a 89.7
  difference — the effect of **data**, not architecture; factor ~18×.
- G1 probe: feature overlap 0.764 → 0.285 at 7.2M→53M (ridge_sep 0.842,
  MLP_sep 0.9): **capacity removes binarity**.
- 53M: 94K pairs → 544; 153K pairs → 648; base-only 1089 rows → 658: **growth from
  data, epoch plateau** (25 vs 35).

## 6. Discussion

### 6.1 CA as extended mind

Measurements v118–v120 show: the CA substrate in late NeuroCA is not a "thinking
part" but a **deterministic provider of features** (extended mind; probe qtree
0.826). Its contribution at the current scale is small (`+5.0 ± 5.8`), but this
does not prove the CA is useless: the narrow benchmark (33 iterative algorithms)
and small data do not let the reservoir show compositional compression.
**Open hypothesis:** the substrate's contribution grows with data scale and task
compositionality (Tier2). Verification — a controlled with/without-substrate
ablation at a ×5 data scale, 3 seeds.

### 6.1a Negative result of kit2c: edge data do not cure Tier1

The decisive experiment of the hypothesis "Tier1 zeros = uncovered inputs" is
complete: **kit2c** (base 1054 + weak_fix 35 + edges 72, 1161 rows, transfer from
kit2b-best) gave **660 = 656 (old 460, new 196), Tier1 = 485** — below kit2b
(658/487). Edge rows (empty lists `[]→None`, tabs, `0^0`, negative numbers) already
present in the kit2c corpus **gave no gain (noise ±2)**. Consequence: Tier1 failures
are **not simply uncovered inputs** but a deeper problem of mapping
"formulation/inputs → family → strategy"; the next lever is **staged loss
decomposition** (task→analysis→strategy→code) and/or **formulation distillation via
the teacher**, not more edge data.

### 6.2 Data versus parameters

Twice confirmed: quality grows from data (544→648→658), not from epochs/size
(plateau 25 vs 35 epochs at 53M). Hence the order "annotation → data → parameters";
the RTX 5080 parameter ceiling (~70–80M) is not the bottleneck before corpus
expansion.

### 6.3 Profile binarity and capacity

Before Branch A, profiles were "binary": either the base champion (651/477) or the
edge champion (596/529), no compromise possible. G1 (overlap 0.285 at 53M) showed
the cause is **insufficient capacity**, not a fatal incompatibility: with more
capacity the model masters both profiles (kit2b: 658/487).

### 6.4 Harness honesty (metrics)

Two confirmed evaluation defects:
1. **Test format** `(a,b,expected)` instead of `((args...),expected)` since v112
   understated the result: 82.4% instead of the real 99.8% (fixed by the format
   guard).
2. **Tier1 audit** (26.09): `res[fam]=got` overwrites the result of a repeated
   family (`decimal_to_binary` appears twice in the bench) — 31×20 = 620 saved, not
   640; `t1_Stack`/`t1_Queue` are absent from the report; 660-Stack/Queue are scored
   by compile+name without functional tests (`tests=None`).

Conclusion: a metric is only as honest as the harness; the P0 priority is fixing
eval (dedup, denominator by fact, per-sample, functional Stack/Queue tests, hashes
of bench/model/corpus) before final measurements.

### 6.5 Limitations

1. **Closed vocabulary** V=1078: Tier2 composition names (sort_and_search etc.) are
   OOV; `text_to_tokens` drops them → Tier2 = 0/60 is not measurable without V
   expansion or teacher-based distillation (path "b").
2. **Narrow base tests** 660: `is_sorted` (only sorted), `is_leap` (only 2000),
   `second_max` (no duplicates), `find_max` (one test) — 660 is not 660 independent
   tasks but 33×20 generations with narrow checks.
3. **53M cannot absorb a 14B LLM**: the student learns planning and family code;
   the general spectrum stays at runtime with the LLM (honest distillation limit).
4. **Generation window** of 150–300 tokens limits long programs.
5. **No controlled substrate ablation at scale** — open item.

---

## 7. Conclusion and Future Work

NeuroCA demonstrates the viability of the original thesis: the dynamics of a
cellular automaton over registers can carry deterministic features under full state
observability and honest exec verification. The hybrid "Transformer/MoE + CA
substrate" reached **99.7% (658/660)** on 33 algorithmic tasks; the series measured
the role of capacity (G1), data (factor ~18× over the substrate at the current
scale) and recorded valuable negative results (synthetic data, LoRA, head blocks,
profile binarity before 53M).

Future work (order of application):
1. **P0:** fix the eval protocol (dedup, denominator by fact, functional Stack/Queue
   tests) before final measurements.
2. **kit2c done** (656/660, Tier1 485): edge data gave no gain — the hypothesis
   "Tier1 = uncovered inputs" is refuted; move to **staged loss decomposition**
   (task→analysis→strategy→code, loss masks) — curing the weak first stage without
   new parameters.
3. **Distillation (path "b"):** formulation diversity 33×3–5, Tier2 via the teacher
   (independent of V), student 53M on the distilled corpus.
4. **Substrate ablation at ×5 data** — separate "CA is useless" from "CA is useful
   at scale"; expert-specialization diagnostics (G4); experts 8→16 (G5).
5. Theoretical directions: hieroglyphic internal language, meta-observer, L-systems
   ("living text"), spatial understanding.

## 8. Reproducibility

### 8.1 Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16 GB GDDR7 (Tier 0; CUDA) |
| CPU | AMD Ryzen 7 7840HS, 8C/16T Zen4 (Tier 1; exec harness, 16-thread pool) |
| iGPU | Radeon 780M (Tier 2; rendering/encoding only, never used for training) |
| Software | Python 3.12, PyTorch (bf16 autocast + GradScaler), NumPy, Ollama (HTTP 11434) |
| LLM teacher | qwen2.5:14b (semantics), qwen2.5-coder:14b / deepseek-coder (code), backup minimax-m3:cloud |
| Working root | `D:\TOOLS_KA` (paths below are absolute on the reference machine) |

### 8.2 Artifacts (versioned)

| Artifact | Path |
|---|---|
| Etalon checkpoint (53M) | `D:\TOOLS_KA\orch_artifacts\hybrid_v120big50_kit2b_agent.pt` |
| Etalon eval report | `D:\TOOLS_KA\orch_artifacts\report_v120big50_kit2b.json` |
| Feature cache (corpus) | `D:\TOOLS_KA\orch_artifacts\branchA_kit2b.npz` (1089 base-only rows), `branchA_kit2c.npz` (1161 rows) |
| Tier1 benchmark | `D:\TOOLS_KA\orch_artifacts\bench_tier1.json` (32 items) |
| Version registry | `D:\TOOLS_KA\.neuroca_lab\versions.json` (26 entries, incl. per-task results and gates) |
| Eval logs (v114c/v114d/v115) | `D:\TOOLS_KA\logs\v114c_eval_fixed_log.txt`, `v114d_eval_log.txt`, `v115_eval_fixed_log.txt` |
| Teacher status | `D:\TOOLS_KA\.neuroca_lab\teacher_status.json`, `sched_queue.json` |

### 8.3 Commands

```bash
# Training (Branch A, scratch; or INIT=... for transfer)
python D:/TOOLS_KA/train_branchA50.py branchA_kit2b.npz kit2b

# Final 660 + Tier1 evaluation of a saved checkpoint
python D:/TOOLS_KA/eval_big50.py            # 660 protocol
python D:/TOOLS_KA/eval_big50_t1.py         # Tier1 protocol

# Teacher pipeline (task generation → exec-verification → corpus)
python D:/TOOLS_KA/teacher_pipeline.py --tasks two_sum

# Paper/PDF build (markers NEUROCA-TABLE in NeuroCA_article.md)
python D:/TOOLS_KA/make_article_pdf.py
```

### 8.4 Determinism and honest metrics

- Training seeds: `torch.manual_seed(2024)`; per-sample eval seeds are explicit
  (300+k checks, 1000+k finals, 2000+ Tier1); per-run exec cache keyed by code hash.
- **Format guard is mandatory** for any eval: tests must be `((args...), expected)`;
  the historical `(a,b,expected)` bug (v112) understated results (82.4% vs real
  99.8%) — see §4.3.
- **Tier1 audit** (§6.4): `res[fam]=got` overwrites a repeated family
  (`decimal_to_binary` appears twice in `bench_tier1.json`); saved entries are
  31×20 = 620, while the declared denominator is 640. The Tier1 figure must be
  read with this caveat; the fix is in progress.
- Stack/Queue families are scored by compile + presence of the class/function name
  (`tests=None`); functional tests are absent — see Appendix B.

### 8.5 License and data

Corpora (`labels/*.jsonl`, `.npz` feature caches) are project-internal; the code
generated by the model is Python snippets derived from algorithmic task
formulations. No personal data is involved. License for the preprint: CC BY 4.0
(suggested); for code/artifacts — to be declared by the laboratory.

---

## Addendum (27.09.2026): root data defects and the new etalon

Two root defects found after this paper was written refine the results.

**Corpus defect.** `kit2_build.clean_text` silently dropped OOV words: 23,097 of
107,541 (21%) never reached training. The UNK fix (`NEUROCA_TOK_UNK=1` keeps the
word position as UNK) restored the text; model `v120big50_unkbase` (transfer from
v121b, seed 2024, 25 epochs) scored **660=479** (old 345 / new 134) and
**Tier1=355/620** — +45/+29 over the previous etalon `kit2b` (434/326). Etalon
updated; the 21% cut text explains part of the Tier1 edge deficits.

**P2 format defect.** None of the 2,094 training cues contained the "задача X:
описание" construction that the P2 benchmark tests with — the model never saw the
format, which explains the reversed bare/full-cue gap (373 vs 292). Treatment
without using P2: 264 corpus descriptions converted to P2 format, 165 teacher
paraphrases, textbook (29 tasks) in P2 format, vocab V=2451 (80% P2-word coverage),
corpus now 279,042 pairs (+21.7%).

**G7 substrate verdict.** At 53M/104K the substrate features give no significant
gain: scratch 332±20 vs 366±24 (−34, n.s.); transfer 434/326 vs 440/313 (noise). A
model trained with the substrate, however, fails without it (cross-ablation 4/660):
the substrate is a working support of the trained model. Branches L2/L3 are not built.

**Eval protocol.** The 658/660 (99.7%) figure was measured on the legacy bare-cue
protocol ("задача X"). The strict full-cue protocol scores the same model 434/660,
the new etalon 479/660: a stricter metric, not a regression.

## References

1. Wolfram S. *A New Kind of Science*. Wolfram Media, 2002.
2. Mordvintsev A., Randazzo E., Niklasson E., Levin M. Growing Neural Cellular
   Automata. *Distill*, 2020.
3. Jaeger H. The "echo state" approach to analysing and training recurrent neural
   networks. GMD Report 148, 2001. (Reservoir computing.)
4. Zelikman E., Wu Y., Mu J., Goodman N. STaR: Bootstrapping Reasoning With
   Reasoning. *NeurIPS*, 2022.
5. Kaplan J. et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.
6. Hoffmann J. et al. Training Compute-Optimal Large Language Models. *NeurIPS*, 2022.
7. Wei J. et al. Emergent Abilities of Large Language Models. *TMLR*, 2022.
8. Liu T. et al. Mixture-of-Agents Enhances Large Language Model Capabilities.
   arXiv:2406.04692, 2024.
9. Shazeer N. et al. Outrageously Large Neural Networks: The Sparsely-Gated
   Mixture-of-Experts Layer. *ICLR*, 2017.
10. Lindenmayer A. Mathematical models for cellular interaction in development.
    *Journal of Theoretical Biology*, 30(3), 1968.
11. O'Keefe J., Nadel L. *The Hippocampus as a Cognitive Map*. Clarendon, 1978.
12. Tolman E. C. Cognitive maps in rats and men. *Psychological Review*, 55(4), 1948.
13. Vygotsky L. S. *Thought and Language*. 1934.
14. Bakhtin M. M. *Problems of Dostoevsky's Poetics*. 1963.
15. Friston K. The free-energy principle: a unified brain theory? *Nature Reviews
    Neuroscience*, 11, 2010.
16. Kahneman D. *Thinking, Fast and Slow*. Farrar, Straus and Giroux, 2011.
17. Hu E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022.

---

## Appendix A. Etalon v120big50_kit2b details

| Parameter | Value |
|---|---|
| Architecture | HybridPrefix d=256, L=6, H=8, MoE 8/3, R=2, FDIM=530 |
| Parameters | 53M (Branch A) |
| Corpus | 1089 base-only rows (base 1054 + weak_fix 35), transfer from v121b |
| 660 | 658/660 = 99.7% (old 460/460, new 198/200) |
| Tier1 | 487 over saved entries (31 families; 640 denominator — audit) |
| Weak base | is_balanced_parentheses 18/20 |
| Tier1 = 0/20 | find_max, power, digit_sum, remove_spaces, count_digits |
| Tier1 drops | is_power_of_two 8, count_vowels 7, caesar_cipher 15 |
| Checkpoint | `orch_artifacts/hybrid_v120big50_kit2b_agent.pt` |
| Report | `orch_artifacts/report_v120big50_kit2b.json` |

## Appendix B. Per-task breakdown 660/Tier1

| Family | 660 | Tier1 | Family | 660 | Tier1 |
|---|---|---|---|---|---|
| factorial | 20 | 20 | average | 20 | 20 |
| fibonacci | 20 | 20 | second_max | 20 | 20 |
| is_palindrome | 20 | 20 | remove_spaces | 20 | **0** |
| find_max | 20 | **0** | is_leap | 20 | 20 |
| find_min | 20 | 20 | is_sorted | 20 | 20 |
| bubble_sort | 20 | 20 | count_digits | 20 | **0** |
| binary_search | 20 | 20 | is_power_of_two | 20 | 8 |
| is_prime | 20 | 20 | fizzbuzz | 20 | 20 |
| unique | 20 | 20 | is_anagram | 20 | 19 |
| count_vowels | 20 | 7 | caesar_cipher | 20 | 15 |
| count_words | 20 | 20 | is_balanced_parentheses | 18 | 18 |
| count_evens | 20 | 20 | decimal_to_binary | 20 | 20 |
| Stack | 20* | — | merge_sorted | 20 | 20 |
| Queue | 20* | — | longest_common_prefix | 20 | 20 |
| power | 20 | **0** | two_sum | 20 | 20 |
| gcd | 20 | 20 | | | |
| reverse_string | 20 | 20 | | | |
| digit_sum | 20 | **0** | | | |

\* Stack/Queue are scored by compile + presence of the class/function name
(`tests=None`); no functional tests — a protocol limitation.

---

*Document compiled from the registry `versions.json` (26 versions), the article
`NeuroCA_article.md`, the plan `PLAN.md` (§12 "Reflections: Transformer on CA"),
the eval-protocol audit (26.09.2026) and scripts `train_branchA50.py` /
`eval_big50_t1.py`. All numerical values are direct copies from artifacts.*




