# NeuroCA changelog (research log)

All numbers are EXEC-verified (compile + real tests in a sandbox). Honest
protocol notes included — see the protocol note in README.

## 27.09.2026 — root defects found (registry: 36 versions)

- **Etalon change:** `v120big50_kit2b` (434/660 · Tier1 326) → **`v120big50_unkbase`**
  (479/660 · Tier1 355). Reason: UNK fix restored 21% of the corpus text that
  `clean_text` had silently dropped (+45/+29, significant).
- **P2-format defect:** 0 of 2,094 training cues used the `"задача X: <описание>"`
  construction that the P2 harness tests with → the model never saw the format.
  Treatment: 264 converted descriptions + 165 teacher paraphrases + textbook
  (29 tasks) + vocab V=2451; corpus now 279,042 pairs (+21.7%). Queue:
  `unkfull → unkext → unkext_fmt`.
- **G7 closed (substrate ablation):** 530 substrate features do not pay off at
  53M/104K — scratch 332±20 vs 366±24 (n.s.); transfer 434/326 vs 440/313 (noise).
  Cross-ablation: a model trained with the substrate fails without it (4/660).
  §13 decision: L2/L3 branches (D/B/A/E/C) are **not** built.
- **§14 confirmed:** transfer (434–440) ≫ scratch (332–366) (+68..+108).

## 26.09.2026 (registry: 26 → 36)

- kit2c (edge data): 656/660, Tier1 485 — no gain vs kit2b; etalon unchanged.
- Branch A kit2b: 658/660 on the **legacy bare-cue protocol** (460/460 + 198/200).

## Earlier milestones (summary)

- v120 hybrid series; A2-control (no substrate): 497.
- Legacy protocol peak: 658/660 = 99.7% (bare cue, etalon kit2b).
- Teacher loop: 208 gens → 106 EXEC-valid (~51% yield).
- Capacity effect: 7.2M → 53M, feature overlap 0.764 → 0.285.
- Format-guard: historical eval-format bug fixed (82.4% → 99.8%).
