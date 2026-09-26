"""Ядро: регистровый клеточный автомат «регистр = (N+1)-е измерение».

Клетка решётки shape = (L0, ..., L_{N-1}) (тор) хранит W-битный регистр.
Состояние — массив uint8 формы spatial_shape + (W,), значения в {0, 1}.
Обновление — локальное правило из neuroca.rules; поддержка синхронного и
асинхронного режимов, а также бит-флип шума (проверка живучести).
"""
from __future__ import annotations

import numpy as np

from .rules import make_rule


def block_mean(a, grid):
    """Блочное среднее по пространственным осям.

    a — массив формы spatial_shape + (W,), grid — кортеж числа блоков по каждой
    пространственной оси (каждая ось должна делиться нацело).
    Возвращает массив формы grid + (W,).
    """
    a = np.asarray(a, np.float32)
    for i, g in enumerate(grid):
        L = a.shape[i]
        if L % g != 0:
            raise ValueError(f"Ось {i}: длина {L} не делится на блок {g}")
        a = a.reshape(a.shape[:i] + (g, L // g) + a.shape[i + 1:])
        a = a.mean(axis=i + 1)
    return np.asarray(a, np.float32)


class RegisterCA:
    """Регистровый клеточный автомат: N-мерный тор клеток с W-битными регистрами."""

    def __init__(self, shape, W=8, rule="hdc", rule_params=None, rule_seed=0, seed=0):
        self.shape = tuple(int(x) for x in shape)
        self.N = len(self.shape)
        self.W = int(W)
        self.rng = np.random.default_rng(seed)
        self.rule_name = rule
        self._rule = make_rule(rule, self.N, self.W,
                               seed=rule_seed, params=rule_params, rng=self.rng)
        self.s = None
        self.t = 0

    # ---------- состояние ----------

    @property
    def state(self):
        return self.s

    def reset(self, mode="zeros", p=0.5, state=None):
        """Инициализация состояния: zeros / random(p) / явный массив."""
        if state is not None:
            st = np.asarray(state)
            if st.shape != self.shape + (self.W,):
                raise ValueError(f"Ожидается форма {self.shape + (self.W,)}, получено {st.shape}")
            self.s = (st != 0).astype(np.uint8)
        elif mode == "zeros":
            self.s = np.zeros(self.shape + (self.W,), np.uint8)
        elif mode == "random":
            self.s = (self.rng.random(self.shape + (self.W,)) < p).astype(np.uint8)
        else:
            raise ValueError(f"Неизвестный режим reset: {mode}")
        self.t = 0
        return self

    def set_bitplane(self, b, values):
        """Записать значения в бит-плоскость b (values — broadcast к пространственной форме)."""
        vals = np.broadcast_to(np.asarray(values) != 0, self.shape).astype(np.uint8)
        self.s[..., int(b)] = vals
        return self

    # ---------- динамика ----------

    def step(self, n=1, noise_p=0.0, async_frac=1.0):
        """n шагов; noise_p — вероятность флипа бита; async_frac — доля обновляемых битов."""
        for _ in range(int(n)):
            new = self._rule(self.s)
            if async_frac < 1.0:
                upd = self.rng.random(self.s.shape) < async_frac
                self.s = np.where(upd, new, self.s).astype(np.uint8)
            else:
                self.s = new
            if noise_p > 0:
                flips = self.rng.random(self.s.shape) < noise_p
                self.s ^= flips.astype(np.uint8)
            self.t += 1
        return self

    # ---------- телеметрия и признаки ----------

    @property
    def _spatial(self):
        return tuple(range(self.N))

    def popcount(self):
        """Число активных бит во всей машине."""
        return int(self.s.sum())

    def bitplane_densities(self):
        """Плотность активных бит по каждой бит-плоскости, форма (W,)."""
        return self.s.mean(axis=self._spatial)

    def features(self, mode="bits", grid=None):
        """Признаки для обучаемого выходного слоя.

        bits — плотности бит-плоскостей (W признаков);
        grid — блочные средние по пространству (prod(grid)*W признаков);
        flat — все биты состояния (prod(shape)*W признаков).
        """
        s = self.s.astype(np.float32)
        if mode == "flat":
            return s.ravel()
        if mode == "bits":
            return s.mean(axis=self._spatial)
        if mode == "grid":
            return block_mean(s, grid).ravel()
        raise ValueError(f"Неизвестный режим признаков: {mode}")
