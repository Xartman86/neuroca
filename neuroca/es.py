"""Ступень 3 стратегии: эволюция LUT-правил (ES) с фитнесом next-token.

Геном = строение правила Кауффмана: на каждый бит-плоскость b — K смещений
по пространственным осям и LUT из 2^K битов. Фитнес — top-1 accuracy
линейного readout, предсказывающего следующий токен из регистрового состояния
(тот же жгут, что в v0.5, но батчевый: B префиксов evolving-ся параллельно
в одном массиве состояния — ускорение ~B раз против поштучного прогона).

Эрозия (главный враг v0.5) наказывается автоматически: правило, стирающее
вход, даёт нулевую информативность -> низкий фитнес.
"""
from __future__ import annotations

import numpy as np

from .arx import tokenize_fragment
from .readout import SoftmaxReadout

BOS = 0
BITS_ID = slice(0, 8)
MARK_BIT = 8


class KauffmanGenome:
    """Правило как геном: offsets (W, K, N) в {-1,0,1} без нулей, tables (W, 2^K)."""

    def __init__(self, offsets, tables):
        self.offsets = np.asarray(offsets, np.int8)
        self.tables = np.asarray(tables, np.uint8)

    @classmethod
    def random(cls, W, N, rng, K=2, p1=0.5):
        offs = np.zeros((W, K, N), np.int8)
        for b in range(W):
            for j in range(K):
                while True:
                    d = rng.integers(-1, 2, size=N)
                    if np.any(d != 0):
                        offs[b, j] = d
                        break
        tabs = (rng.random((W, 1 << K)) < p1).astype(np.uint8)
        return cls(offs, tabs)

    @classmethod
    def identity_like(cls, W, N, rng, K=2, mix_frac=0.2):
        """Тёплый старт: большинство бит-плоскостей копируют вход (delay-line),
        случайные mix_frac плоскостей — случайное перемешивание."""
        offs = np.zeros((W, K, N), np.int8)
        tabs = np.zeros((W, 1 << K), np.uint8)
        for b in range(W):
            if rng.random() < mix_frac:
                for j in range(K):
                    while True:
                        d = rng.integers(-1, 2, size=N)
                        if np.any(d != 0):
                            offs[b, j] = d
                            break
                tabs[b] = (rng.random(1 << K) < 0.5).astype(np.uint8)
            else:
                offs[b, 0] = 0                     # копия входа: out = a1
                while True:
                    d = rng.integers(-1, 2, size=N)
                    if np.any(d != 0):
                        offs[b, 1] = d             # второй вход не используется
                        break
                # LUT «копия первого входа»: idx = a1 | a2<<1
                tabs[b] = [0, 1, 0, 1]
        return cls(offs, tabs)

    def key(self):
        return (self.offsets.tobytes(), self.tables.tobytes())

    def params(self):
        """Параметры для make_kauffman (инжект в обычный движок)."""
        return {"K": self.offsets.shape[1],
                "offsets": [[[int(x) for x in d] for d in offs]
                            for offs in self.offsets],
                "tables": self.tables.tolist()}

    def copy(self):
        return KauffmanGenome(self.offsets.copy(), self.tables.copy())

    def mutate(self, rng, p_lut=0.08, p_off=0.2):
        """Инplace-мутации: перевороты LUT, правки смещений."""
        W, K, N = self.offsets.shape
        flips = rng.random(self.tables.shape) < p_lut
        self.tables[flips] ^= 1
        mask = rng.random((W, K, N)) < p_off
        for b, j, ax in zip(*np.where(mask)):
            new = rng.integers(-1, 2)
            row = self.offsets[b, j].copy()
            row[ax] = new
            if not np.any(row):            # нулевое смещение запрещено
                row[ax] = 1 if rng.random() < 0.5 else -1
            self.offsets[b, j] = row
        return self

    @staticmethod
    def crossover(a, b, rng):
        m = rng.random(a.tables.shape) < 0.5
        tables = np.where(m, a.tables, b.tables).astype(np.uint8)
        mo = rng.random(a.offsets.shape) < 0.5
        offs = np.where(mo, a.offsets, b.offsets).astype(np.int8)
        return KauffmanGenome(offs, tables)


def batched_rule(genome):
    """Правило генома на батче состояния (B, H, X, W): оси 1,2 — пространство."""
    tabs = genome.tables
    offs = genome.offsets

    def rule(st):
        out = np.empty_like(st)
        for b in range(st.shape[-1]):
            (d11, d12), (d21, d22) = offs[b][0], offs[b][1]
            a1 = np.roll(st, int(d11), axis=1) if d11 else st
            a1 = np.roll(a1, int(d12), axis=2) if d12 else a1
            a2 = np.roll(st, int(d21), axis=1) if d21 else st
            a2 = np.roll(a2, int(d22), axis=2) if d22 else a2
            idx = (a1[..., b] & 1) | ((a2[..., b] & 1) << 1)
            out[..., b] = tabs[b][idx]
        return out

    return rule


def build_prefix_pairs(fragments, tid, cap=24):
    """(префикс с BOS, следующий токен) по валидным фрагментам."""
    pairs = []
    for code in fragments:
        toks = tokenize_fragment(code, tid, cap=40)
        if len(toks) < 3:
            continue
        seq = [BOS] + toks[:cap]
        pairs += [(seq[:i + 1], seq[i + 1]) for i in range(len(seq) - 1)]
    return pairs


def run_batch(genome, pairs, H=6, Xn=4, W=16, T=2):
    """Батчевый прогон: все префиксы параллельно в одном массиве.

    Возвращает (X, y): признаки состояний в моменты завершения префиксов.
    """
    B = len(pairs)
    st = np.zeros((B, H, Xn, W), np.uint8)
    rule = batched_rule(genome)
    maxt = max(len(p) for p, _ in pairs)
    Xs, ys = [], []
    lens = np.array([len(p) for p, _ in pairs])
    for t in range(maxt):
        st = np.roll(st, 1, axis=1)
        st[:, 0].fill(0)
        act = np.where(lens > t)[0]
        for i in act:
            tid_ = pairs[i][0][t]
            st[i, 0, 0, BITS_ID] = [(int(tid_) >> b) & 1 for b in range(8)]
            st[i, 0, 0, MARK_BIT] = 1
        if T:
            st[:] = rule(st)
        done = np.where(lens == t + 1)[0]
        if len(done):
            sub = st[done]
            # признаки = как в RegisterLM.features: плоско + суммы по строкам
            # истории (оси X и W свёрнуты, остаётся (D, H))
            feats = np.hstack([sub.reshape(len(done), -1),
                               sub.sum(axis=(2, 3))]).astype(np.float32)
            Xs.append(feats)
            ys.extend(pairs[i][1] for i in done)
    return np.vstack(Xs), np.array(ys)


def make_fitness(fit_pairs, val_pairs, W=16, T=2, epochs=6, l2=1e-2):
    """Фитнес = ВАЛ-acc (новые фрагменты) линейного readout.

    Readout сильно регуляризован (l2=1e-2, мало эпох): запомнить пары нельзя,
    поэтому выживает только правило, чьё состояние ОБЩЕПОЛЕЗНО для предсказания.
    Возвращает (val_acc, fit_acc, плотность)."""

    def fitness(genome):
        Xf, yf = run_batch(genome, fit_pairs, W=W, T=T)
        if len(set(yf.tolist())) < 2:
            return 0.0, 0.0, 0.0
        clf = SoftmaxReadout(l2=l2, lr=0.8, epochs=epochs)
        clf.fit(Xf, yf)
        acc_f = clf.score(Xf, yf)
        Xv, yv = run_batch(genome, val_pairs, W=W, T=T)
        idx = {c: i for i, c in enumerate(clf.classes_)}
        m = np.array([b in idx for b in yv])
        # таргеты в ИНДЕКСЫ классов readout'а (иначе сравнение id с индексами)
        ye = np.array([idx[b] for b in yv[m]])
        acc_v = float(np.mean(np.argmax(clf.predict_proba(Xv[m]), axis=1)
                              == ye)) if m.any() else 0.0
        return float(acc_v), float(acc_f), float(Xf.mean())

    return fitness


def evolve(fitness, W=16, N=2, pop=16, gens=12, seed=0, elite=2,
           warm=True, verbose=True):
    """ES: турнирный отбор, элитизм, кроссовер + мутации. -> (чемпион, лог).

    warm=True: половина стартовой популяции — identity-like геномы (delay-line
    с примесью перемешивания), иначе ES долго ищет переносчика с нуля.
    """
    rng = np.random.default_rng(seed)
    population = []
    n_warm = pop // 2 if warm else 0
    for i in range(pop):
        if i < n_warm:
            population.append(KauffmanGenome.identity_like(W, N, rng,
                                                           mix_frac=rng.uniform(0.05, 0.3)))
        else:
            population.append(KauffmanGenome.random(W, N, rng,
                                                    p1=(0.35, 0.5, 0.65)[i % 3]))
    cache = {}
    log = []
    best, best_key = None, None

    def evaluate(g):
        k = g.key()
        if k not in cache:
            cache[k] = fitness(g)
        return cache[k]

    for gen in range(gens + 1):
        scored = sorted(((evaluate(g)[0], i, g) for i, g in enumerate(population)),
                        key=lambda t: -t[0])
        accs = [evaluate(g) for _, _, g in scored[:3]]
        best_gen = scored[0]
        if best is None or best_gen[0] > best[0]:
            best = (best_gen[0], best_gen[2].copy())
        if verbose:
            v_acc, f_acc, dens = accs[0]
            print(f"  gen {gen:2d}: best_val {best_gen[0]:.3f} "
                  f"(fit {f_acc:.3f}, плотн {dens:.3f})  "
                  f"mean {np.mean([s[0] for s in scored]):.3f}")
        log.append({"gen": gen, "best_val": float(best_gen[0]),
                    "best_fit": float(accs[0][1])})
        if gen == gens:
            break
        parents = [g for _, _, g in scored]
        nxt = [g.copy() for _, _, g in scored[:elite]]
        while len(nxt) < pop:
            i1 = min(rng.integers(0, len(parents), 3))
            i2 = min(rng.integers(0, len(parents), 3))
            child = KauffmanGenome.crossover(parents[i1], parents[i2], rng)
            nxt.append(child.mutate(rng))
        population = nxt

    return best[1], log
