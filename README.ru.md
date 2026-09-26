# NeuroCA — Гибридная нейросеть на принципах клеточного автомата

**Регистровый КА-субстрат + трансформер/MoE для верифицируемой генерации Python-кода.**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-pending-red.svg)](https://arxiv.org/)
[![Code size](https://img.shields.io/github/languages/code-size/Xartman86/neuroca)](https://github.com/Xartman86/neuroca)

- 🇬🇧 English version: [README.md](README.md)
- 📄 Полная научная работа (рус.): [docs/NeuroCA_paper_ru.md](docs/NeuroCA_paper_ru.md)

## Демо

| Регистровая машина (9×9×9×8 = 5832 бита) | Динамика кристалла (N+1 измерений) |
|---|---|
| ![neuroca_anim](assets/neuroca_anim.gif) | ![neuroca_anim_4d](assets/neuroca_anim_4d.gif) |
- 📄 Preprint (eng., arXiv-версия): [docs/NeuroCA_paper_arXiv.md](docs/NeuroCA_paper_arXiv.md) · [PDF](docs/NeuroCA_paper_arXiv.pdf)

## Ключевые результаты (серия v0.1–v120, 26 версий реестра)

| Показатель | Значение |
|---|---|
| EXEC-прогон 660 (33 задачи × 20), эталон `v120big50_kit2b` | **658/660 = 99.7%** |
| Старые 23 / новые 10 семейств | 460/460 (100%) / 198/200 (99%) |
| Тир1 (новые краевые входы) | 487 (31 сохранённая семья; см. аудит §6.4 статьи) |
| Производительность ядра | ~130 млн обновлений бит/с (CPU, без бит-упаковки) |
| Контур LLM-учителя (qwen2.5:14b) | 208 генераций → 106 EXEC-валидных (~51%); `second_max` 0→20/20 |

Ключевые измеренные уроки серии: ёмкость решает «бинарность» профилей
(7.2M→53M, overlap признаков 0.764→0.285); рост качества — **от данных, а не от
параметров** (544→648→658); синтетические элизии/шаблонные шаги в корпусе
**вредят**; КА-субстрат на текущем масштабе даёт +5.0 ± 5.8 п.п. (в пределах
шума); краевые данные не вылечили Тир1 (kit2c 656/660, Тир1 485) — следующий
рычаг — стадиальная декомпозиция потерь.

## Структура репозитория

```
neuroca-repo/
├── README.md              ← английская версия
├── README.ru.md           ← русская версия (этот файл)
├── LICENSE                ← CC BY 4.0
├── docs/
│   ├── NeuroCA_paper_ru.md     ← научная работа на русском (Markdown)
│   ├── NeuroCA_paper_arXiv.md  ← препринт на английском (Markdown)
│   └── NeuroCA_paper_arXiv.pdf ← препринт (10 стр., A4)
└── neuroca/               ← ядро: RegisterCA, правила, иерархия, ES
    ├── __init__.py
    ├── engine.py          ← RegisterCA: reset/set_bitplane/step/features
    ├── rules.py           ← локальные битовые правила (majority, Тум, HDC, Life, Кауффман)
    ├── readout.py         ← слои считывания
    ├── hierarchy.py / hierarchical.py ← иерархический КА
    ├── es.py              ← эволюционная стратегия для LUT-правил
    ├── net.py / cluster.py← слои агентов/кластеров
    ├── arx.py             ← регистры как скрытое состояние
    ├── orchestrator.py    ← оркестрация
    ├── tasks.py / tasks_code.py ← бенчмарк-задачи (23/33 семейства)
    ├── collect_data.py    ← сбор корпуса
    ├── viz.py             ← визуализация (изометрический рендер битов)
    └── hrun.py            ← вспомогательные запуски
```

## Статья

- **arXiv:** в процессе (arXiv:XXXX.XXXXX — обновить после анонса)
- Русская версия: [`docs/NeuroCA_paper_ru.md`](docs/NeuroCA_paper_ru.md)
- Английская (arXiv): [`docs/NeuroCA_paper_arXiv.md`](docs/NeuroCA_paper_arXiv.md) · [`PDF`](docs/NeuroCA_paper_arXiv.pdf)
- Детали воспроизводимости (окружение, артефакты, команды, детерминизм): §8 статьи.

## Быстрый старт (ядро)

```bash
pip install numpy            # matplotlib нужен только для viz.py
python -c "
import sys; sys.path.insert(0, '.')
from neuroca.engine import RegisterCA
m = RegisterCA(shape=(9,9,9), W=8)   # машина 5832 бита
m.reset('random')
m.step(16)
print(m.bitplane_densities())
"
```

Полный пайплайн обучения (feature-кэш, `train_branchA50.py`, exec-харнесс
`eval_big50.py`, контур учителя) — часть лабораторной среды и доступен по
запросу; протокол честных метрик описан в статье (§4.3 формат-гвард, §6.4 аудит
Тир1).

## Лицензия

Работа распространяется по лицензии **Creative Commons Attribution 4.0
International** (CC BY 4.0). См. [LICENSE](LICENSE) и
https://creativecommons.org/licenses/by/4.0/.

## Цитирование (препринт)

```bibtex
@misc{neuroca2026,
  title  = {NeuroCA: A Hybrid Neural Network Based on Cellular Automata},
  author = {NeuroCA Laboratory},
  year   = {2026},
  note   = {Серия v0.1--v120, 26 версий реестра, эталон v120big50\_kit2b}
}
```
