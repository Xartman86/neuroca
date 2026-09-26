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

NeuroCA is a research architecture in which a neuron is a **W-bit register on an
N-dimensional torus**, and the whole network is a binary cellular automaton (CA) in
N+1 dimensions. The deterministic CA dynamics provides ~530 features (the
`Substrate3` reservoir); a compact Transformer/MoE reads them and generates Python
code that is verified by **actual execution** (compile + real tests in a sandbox).

## Key results (series v0.1–v120, 26 registry versions)

| Metric | Value |
|---|---|
| 660 exec-run (33 tasks × 20 samples), etalon `v120big50_kit2b` | **658/660 = 99.7%** |
| Old 23 families / new 10 families | 460/460 (100%) / 198/200 (99%) |
| Tier1 (new edge inputs) | 487 (31 saved families; see audit §6.4 of the paper) |
| Core throughput | ~130M bit-updates/s (CPU, no bit-packing) |
| Teacher loop (qwen2.5:14b) | 208 gens → 106 EXEC-valid (~51% yield); `second_max` 0→20/20 |

Key lessons measured in the series: capacity solves profile "binarity"
(7.2M→53M, feature overlap 0.764→0.285); growth comes **from data, not
parameters** (544→648→658); synthetic elisions/template steps in the corpus
**hurt**; the CA substrate adds +5.0 ± 5.8 p.p. at the current scale (within
noise); edge data did not cure Tier1 (kit2c 656/660, Tier1 485) — the next lever
is staged loss decomposition.

## Repository structure

```
neuroca-repo/
├── README.md                 ← this file (English)
├── README.ru.md              ← Russian version
├── LICENSE                   ← CC BY 4.0
├── docs/
│   ├── NeuroCA_paper_ru.md       ← full paper, Russian (Markdown)
│   ├── NeuroCA_paper_arXiv.md    ← full preprint (Markdown)
│   └── NeuroCA_paper_arXiv.pdf   ← preprint (10 pages, A4)
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
  note   = {Series v0.1--v120, 26 registry versions, etalon v120big50\_kit2b}
}
```
