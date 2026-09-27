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

## Ключевые результаты (серия v0.1–v120, 36 версий реестра)

| Показатель | Значение |
|---|---|
| EXEC-прогон 660 (33 задачи × 20), **эталон `v120big50_unkbase`** (промоция 27.09) | **479/660** (старые 345 / новые 134) |
| Тир1 (краевые входы, 620) | **355/620** |
| Прежний эталон `kit2b` (тот же строгий протокол) | 434/660 · Тир1 326 |
| Исторические 660 на старом протоколе (голый cue, `kit2b`) | 658/660 = 99.7% — см. примечание о протоколе |
| Производительность ядра | ~130 млн обновлений бит/с (CPU, без бит-упаковки) |
| Контур LLM-учителя (qwen2.5:14b) | 208 генераций → 106 EXEC-валидных (~51%); `second_max` 0→20/20 |

> **Примечание о протоколе (честность):** 99.7% измерено на старом протоколе
> (голый cue — только `«задача X»`). В сентябре 2026 eval усилен до **полного
> cue** (`«задача X: <описание>»`) — конструкции, которой, как выяснилось,
> **не было в обучающем корпусе** (см. P2-формат-дефект ниже). На строгом
> протоколе та же модель даёт 434/660; новый эталон `unkbase` — 479/660.

## Новые находки (27.09.2026) — корневые дефекты найдены

1. **Дефект корпуса (корневая причина Тир1):** `kit2_build.clean_text` молча
   выбрасывал OOV-слова — 23 097 из 107 541 слов (21%) не дошли до обучения.
   UNK-фикс (`NEUROCA_TOK_UNK=1`) восстановил их: `unkbase` **660=479, Тир1=355**
   (+45/+29 к `kit2b`) → новый эталон.
2. **P2-формат-дефект:** из 2094 обучающих cue **ровно 0** содержали
   конструкцию `«задача X: <описание>»`, которой тестирует P2 — модель никогда
   не видела этот формат; этим объясняется перевёрнутый разрыв «голый/полный cue»
   (373 vs 292). Лечение (без использования P2): 264 корпусных описания
   переведены в формат P2 + 165 перефразировок учителя + учебник (29 задач) +
   словарь V=2451 (покрытие P2-слов 80%). Корпус теперь 279 042 пары (+21.7%).
3. **Вердикт G7 (абляция субстрата закрыта):** 530 субстратных признаков **не
   окупаются** на 53M/104K — scratch 332±20 vs 366±24 (−34, незначимо); transfer
   434/326 vs 440/313 (шум). Однако модель, обученная **с** субстратом, без него
   отказывает (cross-ablation 4/660): субстрат — рабочая опора обученной модели,
   а не бесплатный выигрыш на этом масштабе.

Ключевые измеренные уроки серии: ёмкость решает «бинарность» профилей
(7.2M→53M, overlap признаков 0.764→0.285); рост качества — **от данных, а не от
параметров** (544→648→658 на старом протоколе); синтетические элизии/шаблонные
шаги в корпусе **вредят**; краевые данные сами по себе не вылечили Тир1;
дефицит P2 имеет **форматно-семантическую природу**, а не только словарную.

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
