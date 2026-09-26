"""Данные для кодинг-задач: фрагменты stdlib, мутации-порчи, оракул compile().

Идея: настоящий Python сам является лексером и судьёй. Берём фрагменты
стандартной библиотеки (валидные по построению, проверено compile()), портим
их мутациями токенов и получаем размеченные пары (код, тип ошибки, строка)
с нулевой стоимостью разметки.

Словарь: 0 = PAD, 1 = UNK, 2..size-1 — самые частые строки токенов.
Токен кодируется бинарно в W-битный регистр; строка фрагмента = слот по оси 0
(временная ось), токен в строке = слот по оси 1.
"""
from __future__ import annotations

import io
import random
import textwrap
import tokenize as tk
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SKIP_TYPES = {tk.NL, tk.NEWLINE, tk.INDENT, tk.DEDENT, tk.ENDMARKER,
              tk.COMMENT, tk.ENCODING}
BRACKETS = {"(", ")", "[", "]", "{", "}"}
KINDS = ["del", "dup", "swap", "bracket", "colon", "indent"]

# Пространство кода: 10 строк x 12 токенов x 1 x W бит
CODE_SHAPE = (10, 12, 1)


def _compiles(code):
    try:
        compile(code, "<f>", "exec")
        return True
    except SyntaxError:
        return False


def stdlib_files(limit=600, seed=7):
    """Небольшие .py файлы стандартной библиотеки."""
    import sysconfig
    root = Path(sysconfig.get_paths()["stdlib"])
    rng = random.Random(seed)
    excl = {"test", "tests", "site-packages", "idlelib", "__pycache__"}
    files = []
    for f in root.rglob("*.py"):
        if excl & {p.lower() for p in f.parts}:
            continue
        try:
            sz = f.stat().st_size
        except OSError:
            continue
        if 200 <= sz <= 20000:
            files.append(f)
    rng.shuffle(files)
    return files[:limit]


def harvest_fragments(n=1200, per_file=2, seed=7):
    """Валидные фрагменты кода длиной 5..9 строк (проверены compile())."""
    rng = random.Random(seed)
    out = []
    for f in stdlib_files(seed=seed):
        if len(out) >= n:
            break
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        if len(lines) < 8:
            continue
        got = 0
        for _ in range(per_file * 5):
            if got >= per_file or len(out) >= n:
                break
            L = rng.randint(5, 9)
            if len(lines) <= L:
                continue
            i = rng.randint(0, len(lines) - L)
            frag = "\n".join(lines[i:i + L])
            for cand in (frag, textwrap.dedent(frag)):
                if _compiles(cand) and any(ln.strip() for ln in cand.splitlines()):
                    out.append(cand)
                    got += 1
                    break
    return out


def build_vocab(samples, size=64):
    """Самые частые строки токенов; 0=PAD, 1=UNK, 2..size-1 — слова."""
    cnt = Counter()
    for code in samples:
        try:
            for t in tk.generate_tokens(io.StringIO(code).readline):
                if t.type not in SKIP_TYPES:
                    cnt[t.string] += 1
        except (tk.TokenError, IndentationError, SyntaxError):
            continue
    words = [s for s, _ in cnt.most_common(size - 2)]
    return words, {s: i + 2 for i, s in enumerate(words)}


def _tokens5(code):
    """Значимые токены (5-кортежи) валидного кода."""
    toks = []
    try:
        for t in tk.generate_tokens(io.StringIO(code).readline):
            if t.type in SKIP_TYPES:
                continue
            toks.append((t.type, t.string, t.start, t.end, t.line))
    except (tk.TokenError, IndentationError, SyntaxError):
        return None
    return toks


def _abs_spans(code):
    """Смещения начал строк -> [(offset, length), ...] по строкам."""
    offs, pos = [], 0
    for ln in code.splitlines(keepends=True):
        offs.append(pos)
        pos += len(ln)
    return offs


def _char_span(code, offs, start, end):
    """(row, col)-пара координат -> абсолютный диапазон [a, b)."""
    a = offs[start[0] - 1] + start[1]
    b = offs[end[0] - 1] + end[1]
    return a, b


def corrupt(code, rng, kind):
    """Текстовая правка одного токена -> (код, строка) или None.

    Позиции берутся из tokenize, правится сам текст (untokenize по координатам
    восстанавливает исходный текст, поэтому dup/swap им не сделать). Результат
    годится, только если код перестал компилироваться (оракул compile()).
    """
    if kind == "indent":
        lines = code.splitlines()
        cand = [i for i, ln in enumerate(lines) if ln[:1] == " "]
        if not cand:
            return None
        i = rng.choice(cand)
        new = lines[i][1:] if rng.random() < 0.5 else " " + lines[i]
        out = "\n".join(lines[:i] + [new] + lines[i + 1:])
        return (out, i + 1) if not _compiles(out) else None

    toks = _tokens5(code)
    if not toks or len(toks) < 4:
        return None
    offs = _abs_spans(code)
    if kind == "del":
        i = rng.randrange(len(toks))
        a, b = _char_span(code, offs, toks[i][2], toks[i][3])
        out = code[:a] + " " + code[b:]
        line = toks[i][2][0]
    elif kind == "dup":
        i = rng.randrange(len(toks))
        if toks[i][2][0] != toks[i][3][0]:
            return None
        a, b = _char_span(code, offs, toks[i][2], toks[i][3])
        out = code[:b] + " " + code[a:b] + code[b:]
        line = toks[i][2][0]
    elif kind == "swap":
        i = rng.randrange(len(toks) - 1)
        if toks[i][2][0] != toks[i][3][0] or toks[i + 1][2][0] != toks[i + 1][3][0]:
            return None
        a1, b1 = _char_span(code, offs, toks[i][2], toks[i][3])
        a2, b2 = _char_span(code, offs, toks[i + 1][2], toks[i + 1][3])
        out = code[:a1] + code[a2:b2] + code[b1:a2] + code[a1:b1] + code[b2:]
        line = toks[i][2][0]
    elif kind == "bracket":
        idxs = [k for k, t in enumerate(toks) if t[1] in BRACKETS]
        if not idxs:
            return None
        i = rng.choice(idxs)
        a, b = _char_span(code, offs, toks[i][2], toks[i][3])
        out = code[:a] + code[b:]
        line = toks[i][2][0]
    elif kind == "colon":
        idxs = [k for k, t in enumerate(toks) if t[1] == ":"]
        if not idxs:
            return None
        i = rng.choice(idxs)
        a, b = _char_span(code, offs, toks[i][2], toks[i][3])
        out = code[:a] + code[b:]
        line = toks[i][2][0]
    else:
        return None
    if _compiles(out):
        return None
    return out, line


def make_code_dataset(n_train_valid=400, n_test_valid=200, per_kind_train=90,
                      per_kind_test=40, seed=7):
    """Разделение по ФРАГМЕНТАМ-источникам (нет утечки train->test).

    -> dict(train=[(code, kind, line), ...], test=[...], vocab, tid)
    kind = "valid" для нетронутых фрагментов.
    """
    frags = harvest_fragments(n_train_valid + n_test_valid + 60, seed=seed)
    rng = random.Random(seed + 1)
    rng.shuffle(frags)
    n_te = n_test_valid + 60                      # запас под брак мутаций
    test_frags, train_frags = frags[:n_te], frags[n_te:]

    def corrupt_pool(pool, kind, per):
        out, tries = [], 0
        while len(out) < per and tries < per * 40 and pool:
            tries += 1
            r = corrupt(rng.choice(pool), rng, kind)
            if r:
                out.append((r[0], kind, r[1]))
        return out

    train = [(f, "valid", None) for f in train_frags[:n_train_valid]]
    test = [(f, "valid", None) for f in test_frags[:n_test_valid]]
    for kind in KINDS:
        train += corrupt_pool(train_frags, kind, per_kind_train)
        test += corrupt_pool(test_frags, kind, per_kind_test)
    rng.shuffle(train)
    rng.shuffle(test)
    vocab, tid = build_vocab([c for c, *_ in train], size=64)
    return {"train": train, "test": test, "vocab": vocab, "tid": tid}


BLOCK_KW = {"def", "if", "elif", "else", "for", "while", "try", "except",
            "finally", "with", "class", "async", "match", "case"}


def embed_code(code, tid, w=16, shape=CODE_SHAPE):
    """Фрагмент -> регистровое состояние (строка = ось 0, ячейка = ось 1).

    Двухпотоковое кодирование (устойчивое к нелексируемому коду):
      axis1 0  — сырой поток: длина строки (8 бит) + откр/закр скобки (4+4);
      axis1 1  — сырой поток: отступ строки (8 бит) + ':' + блочное слово;
      axis1 2+ — лексер-поток: id токена (8 бит), колонка//8 (4), флаги
                 скобка/двоеточие/блочное-слово. Пуст при ошибке лексера.
    Сырой поток заполняется ВСЕГДА — сломанные скобки и отступы не ослепляют
    сенсоры: именно они и есть признаки bracket/indent-порчи.
    """
    st = np.zeros(shape + (w,), np.uint8)

    def put8(cell, lo, x):
        cell[lo:lo + 8] = [(int(x) >> b) & 1 for b in range(8)]

    for i, ln in enumerate(code.splitlines()[:shape[0]]):
        c0 = st[i, 0, 0]
        put8(c0, 0, min(len(ln), 255))
        c0[8:12] = [min(sum(ln.count(x) for x in "([{"), 15) >> b & 1
                    for b in range(4)]
        c0[12:16] = [min(sum(ln.count(x) for x in ")]}"), 15) >> b & 1
                     for b in range(4)]
        c1 = st[i, 1, 0]
        put8(c1, 0, min(len(ln) - len(ln.lstrip(" ")), 255))
        c1[8] = 1 if ":" in ln else 0
        c1[9] = 1 if any(kw in ln for kw in
                         ("def ", "if ", "for ", "while ", "class ",
                          "try:", "with ", "else:", "elif ")) else 0

    try:
        toks = list(tk.generate_tokens(io.StringIO(code).readline))
        lex_ok = True
    except (tk.TokenError, IndentationError, SyntaxError):
        lex_ok = False
        toks = []
    st[0, 1, 0][10] = 0 if lex_ok else 1          # флаг «лексер упал»
    rows = defaultdict(list)
    if lex_ok:
        for t in toks:
            if t.type in SKIP_TYPES:
                continue
            rows[t.start[0]].append((t.start[1], t.string))
    for li, y in enumerate(sorted(rows)[:shape[0]]):
        for ti, (col, s) in enumerate(rows[y][:shape[1] - 2]):
            tid_ = tid.get(s, 1)
            cell = st[li, ti + 2, 0]
            cell[:8] = [(tid_ >> b) & 1 for b in range(8)]
            cell[8:12] = [((min(col, 127) // 8) >> b) & 1 for b in range(4)]
            cell[12] = 1 if s in "()[]{}" else 0
            cell[13] = 1 if s == ":" else 0
            cell[14] = 1 if s in BLOCK_KW else 0
            cell[15] = 0
    return st
