"""Обучаемый выход резервуара: многоклассовая логистическая регрессия на numpy.

Единственный обучаемый компонент системы: правила КА фиксированы (субстрат),
градиент идёт только в выходной слой поверх признаков резервуара.
"""
from __future__ import annotations

import numpy as np


def _softmax(A):
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)


class SoftmaxReadout:
    """Softmax-регрессия, full-batch градиентный спуск со стандартизацией признаков."""

    def __init__(self, l2=1e-3, lr=0.8, epochs=200, verbose=False, seed=None):
        self.l2 = float(l2)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.verbose = verbose
        self.seed = seed  # зарезервировано: метод детерминирован и без него
        self.mu = None
        self.sd = None

    def _design(self, X):
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        return np.hstack([Z, np.ones((Z.shape[0], 1))])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, np.float64)
        y = np.asarray(y)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd = np.where(self.sd == 0, 1.0, self.sd)
        Z = self._design(X)

        self.classes_ = np.unique(y)
        C = len(self.classes_)
        Y = np.zeros((y.shape[0], C))
        for ci, c in enumerate(self.classes_):
            Y[y == c, ci] = 1.0

        # Веса samples: единицы по умолчанию, иначе переданные
        if sample_weight is not None:
            w = np.asarray(sample_weight, np.float64)
        else:
            w = np.ones(Z.shape[0], np.float64)
        w_sum = w.sum() or 1.0

        W = np.zeros((Z.shape[1], C))
        for ep in range(self.epochs):
            P = _softmax(Z @ W)
            # Weighted gradient: G = Z.T @ (w * (P - Y)) / sum(w) + l2 * W
            residual = (P - Y) * w[:, None]
            G = (Z.T @ residual) / w_sum + self.l2 * W
            W -= self.lr * G
            if self.verbose and ep % 50 == 0:
                nll = -np.log(P[np.arange(len(y)), np.argmax(Y, axis=1)] + 1e-12).mean()
                print(f"    epoch {ep:4d}  loss {nll:.4f}")
        self.W_ = W
        return self

    def predict_proba(self, X):
        Z = self._design(np.asarray(X, np.float64))
        return _softmax(Z @ self.W_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class MLPReadout:
    """Однослойный MLP (D → H ReLU → C softmax) с backprop, full-batch GD."""

    def __init__(self, hidden=64, l2=1e-4, lr=0.01, epochs=100,
                 verbose=False, seed=0):
        self.hidden = int(hidden)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.mu = None
        self.sd = None

    def _design(self, X):
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        return np.hstack([Z, np.ones((Z.shape[0], 1))])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, np.float64)
        y = np.asarray(y)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd = np.where(self.sd == 0, 1.0, self.sd)
        Z = self._design(X)
        N, D = Z.shape

        self.classes_ = np.unique(y)
        C = len(self.classes_)
        Y = np.zeros((N, C))
        for ci, c in enumerate(self.classes_):
            Y[y == c, ci] = 1.0

        if sample_weight is not None:
            sw = np.asarray(sample_weight, np.float64)
        else:
            sw = np.ones(N, np.float64)
        sw_sum = sw.sum() or 1.0

        H = self.hidden
        # Xavier init
        scale1 = np.sqrt(2.0 / D)
        scale2 = np.sqrt(2.0 / (H + 1))
        W1 = self.rng.standard_normal((D, H)) * scale1
        W2 = self.rng.standard_normal((H + 1, C)) * scale2

        for ep in range(self.epochs):
            # Forward
            H_raw = Z @ W1                    # (N, H)
            H_act = np.maximum(H_raw, 0)      # ReLU
            H_bias = np.hstack([H_act, np.ones((N, 1))])  # (N, H+1)
            logits = H_bias @ W2              # (N, C)
            P = _softmax(logits)

            # Backward
            d_logits = (P - Y) * sw[:, None]  # weighted
            dW2 = H_bias.T @ d_logits / sw_sum + self.l2 * W2

            d_H_bias = d_logits @ W2.T        # (N, H+1)
            d_H_act = d_H_bias[:, :H]         # (N, H)
            d_H_raw = d_H_act * (H_raw > 0)   # ReLU grad
            dW1 = Z.T @ d_H_raw / sw_sum + self.l2 * W1

            W1 -= self.lr * dW1
            W2 -= self.lr * dW2

            if self.verbose and ep % 25 == 0:
                nll = -np.log(P[np.arange(N), np.argmax(Y, axis=1)] + 1e-12).mean()
                print(f"    epoch {ep:4d}  loss {nll:.4f}")

        self.W1_ = W1
        self.W2_ = W2
        return self

    def predict_proba(self, X):
        Z = self._design(np.asarray(X, np.float64))
        H_raw = Z @ self.W1_
        H_act = np.maximum(H_raw, 0)
        H_bias = np.hstack([H_act, np.ones((H_act.shape[0], 1))])
        return _softmax(H_bias @ self.W2_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
