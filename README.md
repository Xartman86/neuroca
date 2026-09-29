# NeuroCA — A Hybrid Neural Network Based on Cellular Automata

**Register-based cellular automaton substrate + Transformer/MoE for verifiable
Python code generation.**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-pending-red.svg)](https://arxiv.org/)
[![Code size](https://img.shields.io/github/languages/code-size/Xartman86/neuroca)](https://github.com/Xartman86/neuroca)

- 🇷🇺 Русская версия: [README.ru.md](README.ru.md) · [Полная статья на русском](docs/NeuroCA_paper_ru.md)

## Demo

| Register machine (9×9×9×8 = 5832 bits) | Crystal dynamics (N+1 dims) |
|---|---|
| ![neuroca_anim](assets/neuroca_anim.gif) | ![neuroca_anim_4d](assets/neuroca_anim_4d.gif) |

## Visuals: growth curve & 3D structure

| Growth (660 EXEC · Tier1 · P2 · effects) | 3D view ("brain": semantic cube + PCA) | Isometric 4D crystal |
|---|---|---|
| ![growth_curve](assets/growth_curve.png) | ![neuroca_brain3d](assets/neuroca_brain3d.png) | ![neuroca_iso_4d](assets/neuroca_iso_4d.png) |

Growth curve data (27.09.2026): scratch 53M (332–366) → §14 transfer (434–440) →
`kit2b` 434/326 → **`unkbase` 479/660 · Tier1 355/620** (UNK fix restored 21% of the
corpus); P2 full-cue gap is a vocabulary/format deficit, not abstraction; measured
effects: §14 +68…+108, UNK-fix +45/+29. 3D view: semantic concept cube (24³) and
PCA-3D of token embeddings with per-task expert-activation chains.

NeuroCA is a research architecture in which a neuron is a **W-bit register on an
N-dimensional torus**, and the whole network is a binary cellular automaton (CA) in
N+1 dimensions. The deterministic CA dynamics provides ~530 features (the
`Substrate3` reservoir); a compact Transformer/MoE reads them and generates Python
code that is verified by **actual execution** (compile + real tests in a sandbox).

## How NeuroCA works

**1. Neuron = W-bit register on an N-dimensional torus.** The state of a neuron
is a register; the whole network is a binary cellular automaton (CA) in N+1
dimensions, where the register axis is the (N+1)-th dimension. The flagship
machine is 9×9×9×8 = **5832 bits**; the core runs at ~130M bit-updates/s on CPU
(no bit-packing needed).

**2. Deterministic substrate as a reservoir.** The CA dynamics produces ~530
frozen features (`Substrate3`): shift chains (12 tokens), syntax detectors,
counters 8/16/32/64, phase clocks, conjunctions and random Boolean networks
(RBN). The substrate is **not trained** — it is a fixed, reproducible
bit-level reservoir; its feature cache is precomputed once per corpus.

**3. Learnable head.** `HybridPrefix`: d=256, L=6, H=8, MoE with 8 experts
(top-3), FDIM=530, closed vocabulary V=1078. It reads the substrate features and
generates Python code. Branch A is 53M parameters (from scratch); an early
7.2M variant is kept for capacity comparisons.

**4. Generation cascade.** Each task runs through the chain
`задача → анализ → стратегия → схема → код` (task → analysis → strategy →
scheme → code). The inner stages (analysis, strategy, scheme) are written in the
**internal hieroglyphic language** — compact radicals such as `<SORT> <LIST>
<RET>` (see [NeuroCA_internal_language.md](docs/NeuroCA_internal_language.md)) —
short formal plans that are verifiable at each step.

**5. Training: LLM teacher + exec verification.** A STaR-like loop: the teacher
(Ollama `qwen2.5:14b`, `qwen2.5-coder:14b`, `deepseek-coder`) generates
solutions; the **exec harness** (compile + real tests in a sandbox) filters
them; only EXEC-valid solutions are distilled into the 53M student. Measured:
208 generations → 106 EXEC-valid (~51% yield). Corpus: 33 algorithmic task
families (sorting, binary search, GCD, digit sum, brackets, base conversion,
max-finding, …), 1089 base rows; series also trained on 94K and 153K pairs
(current corpus: 279,042 pairs).

**6. Verification instead of string matching.** Code is judged by **real
execution**, not by text similarity. Two honest-metrics safeguards: the
*format guard* (every eval must pass a format check — a historical
eval-format bug inflated numbers from 82.4% to 99.8% was caught this way), and
the *edge-input set* (Tier1 — empty lists, negative numbers, tabs) kept separate
from reformulations.

## Key results (series v0.1–v120, 36 registry versions)

| Metric | Value |
|---|---|
| 660 exec-run (33 tasks × 20), **etalon `v120big50_unkbase`** (promoted 27.09) | **479/660** (old 345 / new 134) |
| Tier1 (edge inputs, 620) | **355/620** |
| Prior etalon `kit2b` (same strict protocol) | 434/660 · Tier1 326 |
| Historical 660 on legacy bare-cue protocol (`kit2b`) | 658/660 = 99.7% — see protocol note below |
| Core throughput | ~130M bit-updates/s (CPU, no bit-packing) |
| Teacher loop (qwen2.5:14b) | 208 gens → 106 EXEC-valid (~51% yield); `second_max` 0→20/20 |

> **Protocol note (honesty):** the 99.7% figure was measured on the legacy
> bare-cue protocol (`"задача X"` only). In September 2026 the eval was
> strengthened to the **full-cue protocol** (`"задача X: <description>"`) — a
> construction that turned out to be **absent from the training corpus** (see
> P2-format defect below). Under the strict protocol the same model scores
> 434/660; the new etalon `unkbase` reaches 479/660.

## Latest findings (27.09.2026) — root defects found

1. **Corpus defect (root cause of Tier1):** `kit2_build.clean_text` silently
   dropped OOV words — 23,097 of 107,541 words (21%) never reached training.
   UNK fix (`NEUROCA_TOK_UNK=1`) restored them: `unkbase` **660=479, Tier1=355**
   (+45/+29 vs `kit2b`) → new etalon.
2. **P2-format defect:** of 2,094 training cues, **0** used the
   `"задача X: <описание>"` construction that the P2 test harness evaluates
   with — the model never saw the format, which explains the reversed
   bare/full-cue gap (373 vs 292). Treatment (without using P2): 264 corpus
   descriptions converted to P2 format + 165 teacher paraphrases + textbook
   (29 tasks) + vocab V=2451 (80% P2 coverage). Corpus now 279,042 pairs (+21.7%).
3. **G7 verdict (substrate ablation closed):** 530 substrate features do **not**
   pay off at 53M/104K — scratch 332±20 vs 366±24 (−34, n.s.); transfer
   434/326 vs 440/313 (noise). However, a model trained **with** the substrate
   fails **without** it (cross-ablation 4/660): the substrate is a working
   support of the trained model, not a free lunch at this scale.

Key lessons measured in the series: capacity solves profile "binarity"
(7.2M→53M, feature overlap 0.764→0.285); growth comes **from data, not
parameters** (544→648→658 on the legacy protocol); synthetic elisions/template
steps in the corpus **hurt**; edge data alone did not cure Tier1; the P2
deficit has a **format/semantics nature**, not just a vocabulary one.

## Repository structure

```
neuroca-repo/
├── README.md                 ← this file (English)
├── README.ru.md              ← Russian version
├── LICENSE                   ← CC BY 4.0
├── docs/
│   ├── NeuroCA_paper_ru.md       ← full paper, Russian (Markdown)
│   ├── NeuroCA_internal_language.md ← internal hieroglyphic language (RU, with figures)
│   ├── NeuroCA_paper_arXiv.md    ← full preprint (Markdown)
│   └── NeuroCA_paper_arXiv.pdf   ← preprint (11 pages, A4)
└── neuroca/                  ← core: RegisterCA engine, rules, hierarchy, ES
    ├── __init__.py
    ├── engine.py             ← RegisterCA: reset/set_bitplane/step/features
    ├── rules.py              ← local bit rules (majority, Toom, HDC, Life, Kauffman)
    ├── readout.py            ← readout layers
    ├── hierarchy.py / hierarchical.py ← hierarchical CA
    ├── es.py                 ← evolution strategy for rule LUTs
    ├── net.py / cluster.py   ← agent/cluster layers
    ├── arx.py                ← registers as hidden state
    ├── orchestrator.py       ← orchestration
    ├── tasks.py / tasks_code.py ← benchmark tasks (23/33 families)
    ├── collect_data.py       ← corpus collection helpers
    ├── viz.py                ← visualization (isometric bit rendering)
    └── hrun.py               ← run helpers
```

## Paper

- **arXiv:** pending (arXiv:XXXX.XXXXX — update after announcement)
- Full text: [`docs/NeuroCA_paper_arXiv.md`](docs/NeuroCA_paper_arXiv.md) and
  [`docs/NeuroCA_paper_arXiv.pdf`](docs/NeuroCA_paper_arXiv.pdf)
- Reproducibility details (environment, artifacts, commands, determinism): §8 of
  the paper.

## Quick start (core)

```bash
pip install numpy            # matplotlib needed only for viz.py
python -c "
import sys; sys.path.insert(0, '.')
from neuroca.engine import RegisterCA
m = RegisterCA(shape=(9,9,9), W=8)   # 5832-bit machine
m.reset('random')
m.step(16)
print(m.bitplane_densities())
"
```

The full training pipeline (feature cache, `train_branchA50.py`, exec harness
`eval_big50.py`, teacher loop) is part of the laboratory environment and is
available on request; the honest-metrics protocol is described in the paper
(§4.3 format guard, §6.4 Tier1 audit).

## License

This work is licensed under the **Creative Commons Attribution 4.0 International**
(CC BY 4.0). See [LICENSE](LICENSE) and https://creativecommons.org/licenses/by/4.0/.

## Citation (preprint)

```bibtex
@misc{neuroca2026,
  title  = {NeuroCA: A Hybrid Neural Network Based on Cellular Automata},
  author = {NeuroCA Laboratory},
  year   = {2026},
  note   = {Series v0.1--v120, 36 registry versions, etalon v120big50\_unkbase (479/660 EXEC, Tier1 355/620)}
}
```
