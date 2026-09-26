"""Агентная сеть из кластеров: сообщения — регистры, арбитр — RegisterCA.

Принципы:
  * связь между кластерами — те же W-битные регистры, что и между нейронами;
  * арбитр сети — сам RegisterCA: клетка-слот хранит регистр-сообщение агента,
    локальное правило (majority/OR/Toom-семейство) прогоняет «совет» регистров;
  * общее пространство делится на территории-бэнды (по самой длинной оси),
    каждый кластер читает признаки только из своего ареала;
  * обучение строго локальное: кластер учит только свой выход; масштабирование =
    добавление кластера без переобучения существующих (веса не трогаются).
"""
from __future__ import annotations

import numpy as np

from .engine import RegisterCA, block_mean
from .cluster import Cluster


def assign_territories(shape, n_agents, slot=None):
    """Бэнды вдоль самой длинной оси: label[shape] = индекс агента, -1 = ничья.

    slot — фиксированная ширина бэнда (для стабильного роста сети: территории
    уже существующих агентов не меняются при добавлении новых).
    """
    axis = int(np.argmax(shape))
    L = shape[axis]
    if slot is None:
        slot = max(1, L // n_agents)
    assert L % slot == 0, f"Ось {axis}: {L} не делится на слот {slot}"
    label = np.full(shape, -1, np.int64)
    for a in range(n_agents):
        sl = (slice(None),) * axis + (slice(a * slot, (a + 1) * slot),)
        label[sl] = a
    return label


def territory_slice(label, agent_idx):
    """Прямоугольный ареал агента в общем пространстве (None, если клеток нет)."""
    idx = np.argwhere(label == agent_idx)
    if len(idx) == 0:
        return None
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    return tuple(slice(int(lo[d]), int(hi[d])) for d in range(len(lo)))


class AgentNetwork:
    """Сеть кластеров-агентов с арбитром на RegisterCA-субстрате."""

    def __init__(self, shape, cell_W=8, msg_W=16, arbiter_rule="majority_nd",
                 arbiter_params=None, seed=0):
        self.shape = tuple(int(x) for x in shape)
        self.cell_W = int(cell_W)      # ширина регистра клетки-нейрона
        self.msg_W = int(msg_W)        # ширина регистра-сообщения агента
        self.arbiter_rule = arbiter_rule
        self.arbiter_params = arbiter_params
        self.feature_grid = None       # grid-признаки территории (если задан)
        self.territory_slot = None     # фикс. ширина бэнда (стабильный рост сети)
        # Единый протокол регистрового сообщения (msg_W=16):
        self.class_bits = 11           # биты 0..10 — класс (one-hot, до 11 классов)
        self.fired_bit = 11            # бит 11 — флаг fired
        self.addr_bits = (12, 14)      # биты 12..13 — адрес агента (двоичный)
        self.conf_bits = (14, 16)      # биты 14..15 — уверенность (>=0.75, >=0.5)
        self.seed = seed
        self.clusters = []             # список Cluster
        self.label = None              # разметка территорий
        self.territories = []          # срезы территорий
        self.arbiter = None            # RegisterCA: клетки-слоты = сообщения

    # ---------------------- рост сети ---------------------------------------

    def add_cluster(self, name, rule="hdc", rule_params=None):
        """Добавить кластер-агента. Существующие кластеры и их веса не трогаются."""
        ca = RegisterCA(self.shape, W=self.cell_W, rule=rule,
                        rule_params=rule_params,
                        rule_seed=self.seed + 1000 + len(self.clusters))
        cl = Cluster(name, ca, msg_w=self.msg_W)
        self.clusters.append(cl)
        self._rebuild_arbiter()
        return cl

    def _rebuild_arbiter(self):
        n = len(self.clusters)
        if n == 1:
            self.label = np.zeros(self.shape, np.int64)
            self.territories = [tuple(slice(None) for _ in self.shape)]
        else:
            self.label = assign_territories(self.shape, n, slot=self.territory_slot)
            self.territories = [territory_slice(self.label, a) for a in range(n)]
        self.arbiter = RegisterCA((n,), W=self.msg_W, rule=self.arbiter_rule,
                                  rule_params=self.arbiter_params,
                                  rule_seed=self.seed)

    # ---------------------- признаки и сообщения -----------------------------

    def cluster_features(self, agent_idx, state):
        """Признаки кластера: свой ареал общего состояния + свой режим.

        "grid"   — блочные средние территории (net.feature_grid);
        "flat"   — вся территория целиком;
        "b0mean" — среднее входной бит-плоскости ареала (глобальная статистика).
        """
        cl = self.clusters[agent_idx]
        sub = state[self.territories[agent_idx]]
        mode = getattr(cl, "feature_mode", "grid")
        if mode == "flat":
            return sub.ravel().astype(np.float32)
        if mode == "b0mean":
            return np.array([sub[..., 0].mean()], np.float32)
        grid = self.feature_grid or tuple(1 for _ in self.shape)
        return block_mean(sub, grid).ravel().astype(np.float32)

    def message_from(self, agent_idx, feat):
        """Признаки агента -> регистр-сообщение в едином протоколе сети.

        Протокол: fired -> полное сообщение (класс one-hot, fired, адрес,
        уверенность); молчание -> нулевой регистр. Так OR-объединение регистров
        в арбитре не загрязняется битами молчащих агентов.
        """
        cl = self.clusters[agent_idx]
        raw = cl.forward(feat)                       # 0..C-1 класс, C fired, хвост увер.
        C = cl.n_classes
        if C > self.class_bits:
            raise ValueError(f"Кластер '{cl.name}': {C} классов > "
                             f"{self.class_bits} бит протокола")
        msg = np.zeros(self.msg_W, np.uint8)
        if not cl.last_fired:
            return msg                               # молчание: нулевой регистр
        msg[:C] = raw[:C]
        msg[self.fired_bit] = 1
        a0, a1 = self.addr_bits
        for j in range(a1 - a0):
            msg[a0 + j] = (agent_idx >> j) & 1
        c0, c1 = self.conf_bits
        msg[c1 - 1] = 1 if cl.last_p >= 0.5 else 0
        msg[c0] = 1 if cl.last_p >= 0.75 else 0
        return msg

    def collect_messages(self, state):
        """Все кластеры: признаки своих ареалов одного состояния -> регистры.

        Возвращает msgs (n_agents, msg_W) uint8 в едином протоколе.
        """
        msgs = np.zeros((len(self.clusters), self.msg_W), np.uint8)
        for a, cl in enumerate(self.clusters):
            if cl.readout is None:
                continue
            msgs[a] = self.message_from(a, self.cluster_features(a, state))
        return msgs

    # ---------------------- арбитраж ----------------------------------------

    def arbitrate(self, msgs):
        """Прямой арбитраж по регистрам: среди fired — максимальная уверенность.

        Читает только биты сообщений (самодостаточный протокол). Возвращает
        (индекс кластера, класс) или None (все молчат).
        """
        fb, (c0, c1) = self.fired_bit, self.conf_bits
        best, best_score = None, -1
        for i in range(len(self.clusters)):
            if not msgs[i, fb]:
                continue
            score = int(msgs[i, c0:c1].sum())
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return None
        return (best, int(np.argmax(msgs[best, :self.class_bits])))

    def arbitrate_merged(self, reg):
        """Декодер состояния арбитра: OR слотов -> адрес -> агент, класс из one-hot.

        reg — состояние арбитра (n_slots, msg_W) или уже объединённый регистр
        (msg_W,). Адресные биты восстанавливают стрелявшего агента, классовые —
        решение. Коллизия адресов (два агента сразу) даёт невалидный адрес -> None.
        """
        reg = np.asarray(reg)
        if reg.ndim == 2:
            reg = reg.max(axis=0)                    # OR-объединение слотов арбитра
        a0, a1 = self.addr_bits
        addr = int(sum(int(reg[a0 + j]) << j for j in range(a1 - a0)))
        if not reg[self.fired_bit] or addr >= len(self.clusters):
            return None
        return (addr, int(np.argmax(reg[:self.class_bits])))

    def arbitrate_register(self, msgs, steps=1):
        """Арбитраж в субстрате: регистры -> клетки арбитра -> локальное правило.

        При threshold=1 правило арбитра — OR-объединение регистров (идемпотентно,
        поэтому 1 и 2 шага эквивалентны); при threshold=2 — побитовое
        большинство, стирающее one-hot коды одиночных агентов.
        """
        if self.arbiter.s is None:
            self.arbiter.reset("zeros")
        self.arbiter.s[:] = msgs.copy()
        if steps:
            self.arbiter.step(steps)
        return self.arbitrate_merged(self.arbiter.s)

    def decide(self, state, register_steps=0):
        """Полный цикл: сообщения из состояния -> арбитраж."""
        msgs = self.collect_messages(state)
        if register_steps:
            return self.arbitrate_register(msgs, steps=register_steps)
        return self.arbitrate(msgs)
