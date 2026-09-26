"""Авторегрессионный жгут: регистры КА как скрытое состояние LM.

Префикс вносится токен за токеном: состояние сдвигается по оси истории
(delay-line: ось 0 = прошлые шаги), новый токен пишется 16-битным регистром
в строку 0 (биты 0..7 = id токена, бит 8 = маркер входа), затем T шагов
локального правила перемешивают контекст по соседям. Скрытое состояние LM =
всё регистровое пространство; readout (softmax) предсказывает следующий токен.

Гипотеза S2: CA-перемешивание (T>0) переносит контекст ЗА пределы окна
delay-line и потому превосходит статическую линию задержки (T=0) на задачах,
где нужен дальний контекст.
"""
from __future__ import annotations

import numpy as np

from .engine import RegisterCA


class RegisterLM:
    """Инкрементальная регистровая память для авторегрессии."""

    def __init__(self, shape=(6, 6), W=16, rule="hdc", rule_params=None,
                 T=1, seed=0):
        self.ca = RegisterCA(shape, W=W, rule=rule,
                             rule_params=rule_params or {}, rule_seed=seed)
        self.ca.reset("zeros")
        self.T = T
        self.W = W
        self.k = 0

    def reset(self):
        self.ca.reset("zeros")
        self.k = 0

    def ingest(self, tid):
        """Один токен: сдвиг истории, запись регистра, T шагов правила."""
        self.ca.s = np.roll(self.ca.s, 1, axis=0)   # строка 0 = новый шаг
        self.ca.s[0].fill(0)
        cell = self.ca.s[0, 0]
        cell[:8] = [(int(tid) >> b) & 1 for b in range(8)]
        cell[8] = 1                                 # маркер входной ячейки
        self.k += 1
        if self.T:
            self.ca.step(self.T)

    def features(self):
        """Скрытое состояние: всё регистровое пространство + суммы по строкам
        истории (маска активности — помогает линейному readout)."""
        sums = self.ca.s.sum(axis=(1, 2)).astype(np.float32)
        return np.concatenate([self.ca.s.ravel(), sums]).astype(np.float32)


def tokenize_fragment(code, tid, cap=40):
    """Валидный фрагмент -> последовательность id токенов (без SKIP-типов)."""
    import io
    import tokenize as tk
    from . import tasks_code as tcode
    out = []
    try:
        for t in tk.generate_tokens(io.StringIO(code).readline):
            if t.type in tcode.SKIP_TYPES:
                continue
            out.append(tid.get(t.string, 1))
    except (tk.TokenError, IndentationError, SyntaxError):
        return []
    return out[:cap]


def collect_pairs(fragments, tid, lm_factory, cap_pairs=12000, bos=0,
                  tokenizer=None):
    """Пары (состояние после префикса -> следующий токен) инкрементально.

    lm_factory() -> новый RegisterLM (сброс между фрагментами обязателен).
    tokenizer(code, tid, cap) -> последовательность id (по умолчанию
    tokenize_fragment без NEWLINE).
    """
    tok = tokenizer or tokenize_fragment
    X, y = [], []
    lm = lm_factory()
    for code in fragments:
        toks = tok(code, tid)
        if len(toks) < 4:
            continue
        lm.reset()
        seq = [bos] + toks
        for i in range(len(seq) - 1):
            lm.ingest(seq[i])
            X.append(lm.features())
            y.append(seq[i + 1])
            if len(y) >= cap_pairs:
                return np.array(X, np.float32), np.array(y)
    return np.array(X, np.float32), np.array(y)
