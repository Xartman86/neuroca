# -*- coding: utf-8 -*-
"""
Сбор данных из .py файлов для обучения CA-LM.

Источники:
  1. ext_data/local_py/ — стандартная библиотека Python + neuroca
  2. tasks_code — оригинальный dataset (качественный, но маленький)

Обработка:
  - Читаем каждый .py файл
  - Разбиваем на функции/классы/методы (AST-based decomposition)
  - Каждый фрагмент = отдельная обучающая пара
  - Фильтруем: только компилируемые фрагменты > 3 строк
"""
from __future__ import annotations

import ast
import sys
from io import StringIO
from pathlib import Path
from tokenize import generate_tokens, NEWLINE, NL, COMMENT, \
    ENDMARKER, INDENT, DEDENT, ENCODING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_functions(code: str, min_lines: int = 3) -> list[str]:
    """Извлекает функции, классы и top-level блоки из Python кода."""
    fragments = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Если AST не парсится — попробуем разбить по пустым строкам
        return _split_by_blanks(code, min_lines)

    # Top-level функции и классы
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 20
            lines = code.splitlines()[start:end]
            frag = "\n".join(lines)
            if len(lines) >= min_lines:
                fragments.append(frag)

    # Если нет функций — весь файл как один фрагмент (обрезанный до 60 строк)
    if not fragments:
        lines = code.splitlines()[:60]
        if len(lines) >= min_lines:
            fragments.append("\n".join(lines))

    return fragments


def _split_by_blanks(code: str, min_lines: int) -> list[str]:
    """Разбить код по двойным пустым строкам."""
    fragments = []
    current = []
    blank_count = 0
    for line in code.splitlines():
        if line.strip() == "":
            blank_count += 1
            if blank_count >= 2 and len(current) >= min_lines:
                fragments.append("\n".join(current))
                current = []
                blank_count = 0
        else:
            blank_count = 0
            current.append(line)
    if len(current) >= min_lines:
        fragments.append("\n".join(current))
    return fragments


def load_directory(path: str, min_lines: int = 3) -> list[str]:
    """Загрузить все .py файлы из директории и извлечь фрагменты."""
    all_frags = []
    d = Path(path)
    if not d.exists():
        return all_frags
    for f in sorted(d.glob("*.py")):
        try:
            code = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        frags = extract_functions(code, min_lines)
        # Валидация: каждый фрагмент должен компилироваться
        for frag in frags:
            try:
                compile(frag, "<frag>", "exec")
                all_frags.append(frag)
            except Exception:
                pass
    return all_frags


def collect_training_fragments(extra_dirs: list[str] = None,
                                min_lines: int = 3) -> list[str]:
    """Собрать все обучающие фрагменты из всех источников."""
    all_frags = []

    # Источник 1: оригинальный dataset
    try:
        from neuroca import tasks_code as tc
        data = tc.make_code_dataset(seed=7)
        for split in ("train", "test"):
            for code, kind, _ in data[split]:
                if kind == "valid":
                    frags = extract_functions(code, min_lines)
                    for f in frags:
                        try:
                            compile(f, "<frag>", "exec")
                            all_frags.append(f)
                        except Exception:
                            pass
    except Exception as e:
        print(f"  tasks_code: {e}")

    # Источник 2: внешние директории
    if extra_dirs:
        for d in extra_dirs:
            frags = load_directory(d, min_lines)
            print(f"  {d}: {len(frags)} фрагментов")
            all_frags += frags

    return all_frags


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="*", default=["ext_data/local_py"])
    parser.add_argument("--min-lines", type=int, default=3)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    frags = collect_training_fragments(args.dirs, args.min_lines)
    print(f"Всего фрагментов: {len(frags)}")

    if args.stats:
        # Статистика
        lines = [len(f.splitlines()) for f in frags]
        print(f"  строк: min={min(lines)}, max={max(lines)}, "
              f"mean={sum(lines)/len(lines):.1f}")

        # Топ токенов
        from collections import Counter
        SKIP = {NL, COMMENT, ENDMARKER, INDENT, DEDENT, ENCODING}
        freq = Counter()
        for code in frags:
            try:
                for t in generate_tokens(StringIO(code).readline):
                    if t.type in SKIP:
                        continue
                    if t.type == NEWLINE:
                        continue
                    freq[t.string] += 1
            except Exception:
                pass
        print(f"  уникальных токенов: {len(freq)}")
        print(f"  топ-30:")
        for w, c in freq.most_common(30):
            print(f"    {w!r}: {c}")

    # Сохранить
    out = Path("ext_data/training_frags.txt")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for frag in frags:
            f.write(frag.replace("\n", "\\n") + "\n---\n")
    print(f"Сохранено: {out} ({len(frags)} фрагментов)")


if __name__ == "__main__":
    main()
