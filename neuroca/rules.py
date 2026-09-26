"""Локальные правила регистрового клеточного автомата.

Все правила действуют на массив состояния s формы spatial_shape + (W,):
оси 0..N-1 — пространственные (тор, реализованы через np.roll),
ось N — регистр клетки (W бит). Значения — uint8 в {0, 1}.

Семейства:
  majority_nd — мажоритарное голосование по 2N пространственным соседям + центр;
  toom_reg    — обобщённое правило Тума: MAJ(сосед по оси 0, центр, регистр-сосед b+1);
  hdc         — бит-параллельное гипервекторное правило: связывание сдвиг+поворот регистра;
  life_reg    — Life-подобное правило в N+1 измерениях (пространство + регистр);
  kauffman    — случайные булевы LUT на K входах (сеть Кауффмана на решётке);
  identity    — тождество (для базовых линий).
"""
from __future__ import annotations

import numpy as np


def _maj3(a, b, c):
    """Побитовое большинство из трёх: (a&b)|(a&c)|(b&c)."""
    return ((a & b) | (a & c) | (b & c)).astype(np.uint8)


def _maj_sum(arrs, thr):
    """Побитовая пороговая функция: 1 там, где не меньше thr входов равны 1."""
    s = arrs[0].astype(np.uint8).copy()
    for a in arrs[1:]:
        s = s + a  # uint8; максимум 2N+1+2 < 256
    return (s >= thr).astype(np.uint8)


def make_majority_nd(N, W, params, rng):
    """Мажоритарное правило по (2N+1) голосам: 2N пространственных соседей + центр.

    Порог по умолчанию N+1 (строгое большинство; при 2N+1 голосах ничьих нет).
    Регистр-ось не сцеплена: бит-плоскости эволюционируют независимо.
    """
    axes = list(range(N))
    thr = int(params.get("threshold", N + 1))

    def rule(s):
        votes = [s]
        for ax in axes:
            votes.append(np.roll(s, 1, axis=ax))
            votes.append(np.roll(s, -1, axis=ax))
        return _maj_sum(votes, thr)

    return rule


def make_toom_reg(N, W, params, rng):
    """Обобщённое правило Тума: MAJ(сосед в +направлении оси a0, центр, регистр-сосед b+bdir).

    Классический Тум (N, C, E) — мажоритарная самокорректирующаяся память.
    Здесь третий вход взят с оси регистра: сцепка «пространство + регистр».
    """
    a0 = int(params.get("axis0", 0)) % N
    bdir = int(params.get("bdir", 1))

    def rule(s):
        n = np.roll(s, -1, axis=a0)          # значение соседа a0 из +направления
        r = np.roll(s, -bdir, axis=N)        # значение регистра-соседа b+bdir
        return _maj3(n, s, r)

    return rule


def make_hdc(N, W, params, rng):
    """Гипервекторное правило: R <- MAJ(R, rot_k(R(+e_a0)), rot_m(R(+e_a1))).

    Соседский регистр сдвигается по пространству и циклически поворачивается
    по оси регистра (связывание в духе HDC/VSA), затем мажоритарное
    «наложение» с собственным состоянием. Чисто бит-параллельно.
    """
    ax0 = int(params.get("axis0", 0)) % N
    ax1 = int(params.get("axis1", 1 % N)) % N
    k = int(params.get("rot0", 1))
    m = int(params.get("rot1", -1))

    def rule(s):
        x = np.roll(np.roll(s, -1, axis=ax0), k, axis=N)
        y = np.roll(np.roll(s, -1, axis=ax1), m, axis=N)
        return _maj3(s, x, y)

    return rule


def make_life_reg(N, W, params, rng):
    """Life-подобное правило в N+1 измерениях: соседи = 2N пространственных + 2 регистра.

    birth — набор счётчиков для рождения (центр 0), survive — для выживания (центр 1).
    """
    birth = list(params.get("birth", [3]))
    survive = list(params.get("survive", [2, 3, 4]))
    axes = list(range(N))

    def rule(s):
        cnt = np.zeros(s.shape, np.uint8)
        for ax in axes:
            cnt += np.roll(s, 1, axis=ax)
            cnt += np.roll(s, -1, axis=ax)
        cnt += np.roll(s, 1, axis=N)
        cnt += np.roll(s, -1, axis=N)
        new = np.where(s.astype(bool),
                       np.isin(cnt, survive),
                       np.isin(cnt, birth))
        return new.astype(np.uint8)

    return rule


def make_kauffman(N, W, params, rng):
    """Сеть Кауффмана на решётке: на каждый бит b — K случайных входов и своя LUT (2^K).

    Детерминирован при фиксированном params["seed"] (смещения и таблицы строятся
    один раз при создании правила). K=2 — критический режим Кауффмана.
    """
    K = int(params.get("K", 2))
    p1 = float(params.get("p1", 0.5))
    lseed = int(params.get("seed", 0))
    grat = np.random.default_rng(lseed)

    if "offsets" in params and "tables" in params:
        # Инжект готового генома (ES): смещения и LUT заданы явно.
        offsets = [[tuple(int(x) for x in d) for d in offs]
                   for offs in params["offsets"]]
        tables = np.asarray(params["tables"], np.uint8).reshape(W, 1 << K)
    else:
        offsets = []  # offsets[b] — список из K смещений по пространственным осям
        tables = np.zeros((W, 1 << K), np.uint8)
        for b in range(W):
            offs = []
            for _ in range(K):
                while True:
                    d = grat.integers(-1, 2, size=N)
                    if np.any(d != 0):
                        break
                offs.append(tuple(int(x) for x in d))
            offsets.append(offs)
            tables[b] = (grat.random(1 << K) < p1).astype(np.uint8)

    def rule(s):
        out = np.empty_like(s)
        for b in range(W):
            idx = np.zeros(s.shape[:-1], np.uint8)
            for j, d in enumerate(offsets[b]):
                a = s
                for ax in range(N):
                    if d[ax]:
                        a = np.roll(a, d[ax], axis=ax)
                idx |= (a[..., b].astype(np.uint8) << j)
            out[..., b] = tables[b][idx]
        return out

    return rule


def make_identity(N, W, params, rng):
    return lambda s: s.astype(np.uint8)


_RULES = {
    "majority_nd": make_majority_nd,
    "toom_reg": make_toom_reg,
    "hdc": make_hdc,
    "life_reg": make_life_reg,
    "kauffman": make_kauffman,
    "identity": make_identity,
}


def make_rule(name, N, W, seed=0, params=None, rng=None):
    """Собрать правило по имени. seed пробрасывается в params["seed"]."""
    if name not in _RULES:
        raise KeyError(f"Неизвестное правило: {name}. Доступны: {sorted(_RULES)}")
    p = dict(params or {})
    p.setdefault("seed", int(seed) if isinstance(seed, (int, np.integer)) else 0)
    return _RULES[name](N, W, p, rng)
