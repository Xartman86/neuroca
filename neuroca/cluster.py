"""Кластер-агент: локальный RegisterCA + локальный обучаемый выход.

Кластер — «нейрон» следующего уровня: решает свою частную задачу на своём
локальном субстрате (свой экземпляр RegisterCA со своим правилом) и возвращает
**регистровое сообщение** — вектор из msg_w бит, как клетка возвращает биты
соседям. Протокол самодостаточен: арбитр принимает решение, читая только биты
сообщения (без доступа к внутреннему состоянию кластера).

Формат регистрового сообщения (C = число классов readout):
  биты 0..C-1    — one-hot класса-победителя;
  бит C          — флаг fired (уверенно И не «чужое»);
  биты C+1..end  — unary-шкала уверенности: бит b = 1 <=> p >= 1 - (b-C)/k.

Класс «чужое» (foreign) кластер учит отдельно: входы чужой модальности должны
попадать в него и не зажигать fired — так реализована специализация агента.
"""
from __future__ import annotations

import numpy as np

from .engine import RegisterCA
from .readout import SoftmaxReadout


class Cluster:
    """Агент = кластер нейронов: свой КА, своя задача, регистровое сообщение наружу."""

    def __init__(self, name, ca, msg_w=16):
        self.name = name
        self.ca = ca
        self.msg_w = int(msg_w)
        self.readout = None
        self.n_classes = None
        self.foreign_idx = None      # позиция класса «чужое» в readout.classes_
        self.T = 4                   # шаги локальной динамики (настраивается демо)
        self.feature_mode = "grid"   # "grid" | "flat" (используется AgentNetwork)
        self.last_p = None           # уверенность последнего forward (диагностика)
        self.last_y = None
        self.last_fired = False
        self.last_foreign = False
        self.living = False          # живая среда: биты b>=1 случайны при инъекции

    # ---------------------- обучение (строго локальное) ----------------------

    def fit(self, X, y):
        """X — признаки локального КА (n, dim), y — метки частной задачи кластера."""
        self.n_classes = int(len(np.unique(y)))
        self.readout = SoftmaxReadout(l2=1e-3, lr=0.8, epochs=200)
        self.readout.fit(np.asarray(X), np.asarray(y))
        return self

    # ---------------------- инференс -----------------------------------------

    def forward(self, x):
        """Признаки локального КА (dim,) -> регистровое сообщение (msg_w,) uint8.

        Энергетический вентиль: нулевой вход (пустой ареал) -> молчание —
        эксперт не стреляет по чужой/пустой территории.
        """
        x = np.asarray(x, np.float64)
        P = self.readout.predict_proba(x[None, :])[0]
        k = int(np.argmax(P))
        self.last_p = float(P[k])
        self.last_y = k
        self.last_foreign = (self.foreign_idx is not None and k == self.foreign_idx)
        silent = (x.sum() == 0.0)                    # вентиль пустого входа
        self.last_fired = (self.last_p >= 0.5) and not self.last_foreign and not silent

        C = self.n_classes
        msg = np.zeros(self.msg_w, np.uint8)
        if silent:
            return msg                               # пустой вход: нулевой регистр
        msg[k] = 1                                   # решение (one-hot)
        msg[C] = 1 if self.last_fired else 0         # fired
        n_conf = self.msg_w - C - 1                  # хвост уверенности
        for i in range(n_conf):
            msg[C + 1 + i] = 1 if self.last_p >= 1.0 - (i + 1) / n_conf else 0
        return msg
