# -*- coding: utf-8 -*-
"""
Иерархический регистровый КА: 4 уровня, общающихся через регистровый канал.

Уровни (сверху вниз):
  L4 (Meta)       — block_id ∈ {0..4}, shape (3, 3), W=8
  L3 (Substruct)  — substruct_id ∈ {0..3}, shape (4, 4), W=8
  L2 (Stmt)       — stmt_id ∈ {0..10}, shape (4, 4), W=8
  L1 (Token)      — token_id ∈ {0..V-1}, shape (6, 6), W=16

Связь сверху вниз: выход уровня L (block_id/substruct_id/stmt_id) на
каждом шаге записывается в дополнительную бит-плоскость регистра уровня
L-1. Это реализует "контекст сверху" без backprop.

Связь снизу вверх: выход уровня 1 (token) используется для обновления
верхних уровней на следующем шаге (но без обратного распространения —
просто другой КА с собственным динамическим переходом).

Все уровни используют одни и те же правила Кауффмана (или identity —
через rule_params), но каждый со своим seed и формой.
"""
from __future__ import annotations

import numpy as np

from .engine import RegisterCA


class HierarchicalRegisterCA:
    """4-уровневая иерархия регистровых КА."""

    # Размеры по уровням
    SHAPES = {
        4: (3, 3),   # Meta (block)
        3: (4, 4),   # Substruct
        2: (4, 4),   # Stmt
        1: (6, 6),   # Token
    }
    # Все уровни W=16. Bias от верхних уровней: 3 маркер-бита в cell[9..11]
    W_BY_LEVEL = {4: 16, 3: 16, 2: 16, 1: 16}

    def __init__(self, rule="kauffman", rule_params=None, T=1, seed=0):
        self.levels = {}
        for L in (4, 3, 2, 1):
            self.levels[L] = RegisterCA(
                shape=self.SHAPES[L], W=self.W_BY_LEVEL[L],
                rule=rule, rule_params=rule_params or {},
                rule_seed=seed + L * 13)
            self.levels[L].reset("zeros")
        self.T = T
        self.k = 0

    def reset(self):
        for L in (4, 3, 2, 1):
            self.levels[L].reset("zeros")
        self.k = 0

    def _encode_id(self, value, n_bits):
        """Закодировать целое число в n_bits младших бит плоскости 0 строки 0
        ячейки (0, 0) уровня L."""
        bits = [(int(value) >> b) & 1 for b in range(n_bits)]
        return bits

    def ingest_token(self, tid, block_id=0, stmt_id=0, sub_id=0,
                     block_changed=False, stmt_changed=False):
        """Внести токен на уровень 1. CA-шаг ПЕРВЫМ, bias ЗАТЕМ."""
        L1 = self.levels[1]
        L1.s = np.roll(L1.s, 1, axis=0)
        L1.s[0].fill(0)
        cell = L1.s[0, 0]
        cell[:8] = [(int(tid) >> b) & 1 for b in range(8)]
        cell[8] = 1  # маркер входа
        if self.T:
            L1.step(self.T)
        # Bias ПОСЛЕ CA — 3 маркер-бита (строго 0 или 1!)
        cell = L1.s[0, 0]
        cell[9] = int(bool(block_changed))   # 0 или 1
        cell[10] = int(bool(stmt_changed))   # 0 или 1
        cell[11] = int(bool(sub_id & 1))     # младший бит sub_id
        self.k += 1

    def ingest_meta(self, block_id):
        """Шаг 2: обновить уровень 4 (block_id) после токена."""
        L4 = self.levels[4]
        L4.s = np.roll(L4.s, 1, axis=0)
        L4.s[0].fill(0)
        cell = L4.s[0, 0]
        bits = self._encode_id(block_id, 4)  # 4 бита хватит на 0..15
        for b, v in enumerate(bits):
            cell[b] = v
        cell[4] = 1
        if self.T:
            L4.step(self.T)

    def ingest_substruct(self, sub_id):
        L3 = self.levels[3]
        L3.s = np.roll(L3.s, 1, axis=0)
        L3.s[0].fill(0)
        cell = L3.s[0, 0]
        bits = self._encode_id(sub_id, 3)
        for b, v in enumerate(bits):
            cell[b] = v
        cell[3] = 1
        if self.T:
            L3.step(self.T)

    def ingest_stmt(self, stmt_id):
        L2 = self.levels[2]
        L2.s = np.roll(L2.s, 1, axis=0)
        L2.s[0].fill(0)
        cell = L2.s[0, 0]
        bits = self._encode_id(stmt_id, 4)
        for b, v in enumerate(bits):
            cell[b] = v
        cell[4] = 1
        if self.T:
            L2.step(self.T)

    def _readout_id(self, L):
        """Прочитать текущий id уровня L: первые 4 бита cell(0,0)."""
        cell = self.levels[L].s[0, 0]
        v = 0
        for b in range(4):
            if int(cell[b]) & 1:
                v |= (1 << b)
        return [(v >> b) & 1 for b in range(4)]

    def features(self, level=1):
        """Состояние уровня level: всё пространство + суммы по строкам."""
        ca = self.levels[level]
        sums = ca.s.sum(axis=(1, 2)).astype(np.float32)
        return np.concatenate([ca.s.ravel(), sums]).astype(np.float32)

    def features_all(self):
        """Все уровни: concat features(L4) + features(L3) + features(L2) + features(L1)."""
        return np.concatenate([self.features(L) for L in (4, 3, 2, 1)])

    def current_block(self):
        return int(self.levels[4].s[0, 0, 0]) | (int(self.levels[4].s[0, 0, 1]) << 1) \
             | (int(self.levels[4].s[0, 0, 2]) << 2) | (int(self.levels[4].s[0, 0, 3]) << 3)

    def current_stmt(self):
        return int(self.levels[2].s[0, 0, 0]) | (int(self.levels[2].s[0, 0, 1]) << 1) \
             | (int(self.levels[2].s[0, 0, 2]) << 2) | (int(self.levels[2].s[0, 0, 3]) << 3)
