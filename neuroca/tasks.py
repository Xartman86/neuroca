"""Данные и кодирование входов для задач-проверок."""
from __future__ import annotations

import numpy as np


def make_density_dataset(n, shape, lo, hi, seed):
    """Случайные Бернулли-поля с плотностью rho ~ U(lo, hi).

    Метка: 1, если rho > 0.5 (задача классификации плотности).
    Возвращает (X[n, *shape] uint8, y[n] int, rho[n] float).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, *shape), np.uint8)
    rho = rng.uniform(lo, hi, size=n)
    for i in range(n):
        X[i] = (rng.random(shape) < rho[i]).astype(np.uint8)
    y = (rho > 0.5).astype(int)
    return X, y, rho


def thermocode(img, W, vmax=16.0):
    """Термометр-код пикселя в W бит: bit_b = 1 <=> p >= (b+1)*vmax/W.

    Монотонное позиционное кодирование: яркость 0 -> все нули, vmax -> все единицы.
    img любой формы -> форма img.shape + (W,).
    """
    img = np.asarray(img, np.float64)
    thr = (np.arange(W) + 1) * (vmax / W)
    return (img[..., None] >= thr).astype(np.uint8)


# Рукописные шаблоны цифр 8x8 ('#' = штрих).
_DIGIT_TEMPLATES = [
    [".#####..", "##...##.", "##...##.", "##...##.", "##...##.", "##...##.", ".#####..", "........"],  # 0
    ["...##...", "..###...", "...##...", "...##...", "...##...", "...##...", ".#####..", "........"],  # 1
    [".#####..", "##...##.", ".....##.", "....##..", "...##...", "..##....", ".######.", "........"],  # 2
    [".#####..", "##...##.", ".....##.", "..####..", ".....##.", "##...##.", ".#####..", "........"],  # 3
    ["....##..", "...###..", "..####..", ".##.##..", "########", "....##..", "....##..", "........"],  # 4
    [".######.", "##......", "######..", ".....##.", ".....##.", "##...##.", ".#####..", "........"],  # 5
    ["..####..", ".##..##.", "##......", "######..", "##...##.", "##...##.", ".#####..", "........"],  # 6
    [".######.", ".....##.", "....##..", "...##...", "...##...", "...##...", "...##...", "........"],  # 7
    [".#####..", "##...##.", "##...##.", ".#####..", "##...##.", "##...##.", ".#####..", "........"],  # 8
    [".#####..", "##...##.", "##...##.", ".######.", ".....##.", "....##..", ".###....", "........"],  # 9
]


def load_digits(seed=0):
    """Цифры 8x8: sklearn load_digits при наличии, иначе рукописные шаблоны.

    Фолбэк: шаблон класса + случайный сдвиг (тор 8x8) + гауссов шум, значения 0..16.
    Возвращает (X[n, 64] float, y[n] int, source str).
    """
    try:
        from sklearn.datasets import load_digits  # type: ignore
        d = load_digits()
        return np.asarray(d.data, np.float64), np.asarray(d.target), "sklearn-digits"
    except Exception:
        rng = np.random.default_rng(seed)
        n_per = 150
        X = np.zeros((n_per * 10, 64))
        y = np.zeros(n_per * 10, dtype=int)
        for c, rows in enumerate(_DIGIT_TEMPLATES):
            base = np.array([[14.0 if ch == "#" else 0.0 for ch in row] for row in rows])
            for k in range(n_per):
                dx = int(rng.integers(-1, 2))
                dy = int(rng.integers(-1, 2))
                img = np.roll(np.roll(base, dx, axis=0), dy, axis=1)
                img = np.clip(img + rng.normal(0, 2.0, size=(8, 8)), 0, 16)
                X[c * n_per + k] = img.ravel()
                y[c * n_per + k] = c
        return X, y, "hand-drawn-8x8-templates(fallback)"
