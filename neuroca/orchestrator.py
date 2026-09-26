# -*- coding: utf-8 -*-
"""
MoE-оркестратор: один общий CA + N readouts + routing.

Ключевая идея: один CA (shared state) для всех, но N разных readouts
(ensemble). Оркестратор (routing readout) выбирает лучший readout для
текущего контекста. Каждый readout обучается на ВСЕХ данных.

Архитектура:
  Shared RegisterLM (6×6, W=16, T=2) — общий CA
  Expert Readouts (N штук) — каждый со своим readout на features
  Orchestrator Readout — routing: какой эксперт лучше для этого контекста

Генерация:
  1. CA обрабатывает токен → features
  2. Routing readout → expert_id (какой readout использовать)
  3. Выбранный readout → next token
  4. CA получает токен → повтор
"""
from __future__ import annotations

import numpy as np


class MoEOrchestrator:
    """MoE на одном CA: shared state + ensemble readouts + routing."""

    def __init__(self, n_experts=3):
        self.n_experts = n_experts
        self.routing_clf = None  # routing readout
        self.expert_clfs = {}    # expert_id → readout
        self.expert_params = {}  # expert_id → hyperparams dict

    def set_routing(self, clf):
        self.routing_clf = clf

    def set_expert(self, eid, clf, params=None):
        self.expert_clfs[eid] = clf
        if params:
            self.expert_params[eid] = params

    def predict_routing(self, features):
        """Предсказать expert_id для данных features."""
        if self.routing_clf is None:
            return 0
        probs = self.routing_clf.predict_proba(features[None])[0]
        return int(self.routing_clf.classes_[np.argmax(probs)])

    def predict_expert(self, eid, features):
        """Предсказать токен через выбранный эксперт."""
        if eid not in self.expert_clfs:
            eid = min(self.expert_clfs.keys())
        clf = self.expert_clfs[eid]
        probs = clf.predict_proba(features[None])[0]
        return clf.classes_, probs


def collect_moe_pairs(frags, tid, lm_factory, cap_pairs=18000,
                       aug_features=True):
    """Собрать обучающие пары для MoE.

    Каждая пара: (features, (next_token, block_id, stmt_id))
    где block_id → routing target (expert_id)

    features = CA state (+ one-hot(block_id, stmt_id) если aug_features)
    """
    from .hierarchy import annotate_tokens, BLOCK_MODULE, BLOCK_IMPORT, \
        BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP

    BLOCK_TO_EXPERT = {
        BLOCK_MODULE: 0,
        BLOCK_IMPORT: 1,
        BLOCK_DEF: 2,
        BLOCK_CLASS: 3,
        BLOCK_EXPR_TOP: 4,
    }

    X_all = []       # features
    y_tok = []        # next token id
    y_route = []      # expert_id (routing target)

    NL_TOKEN = 2
    PAD_TOKEN = 0

    for code in frags:
        ann = annotate_tokens(code, tid, max_tokens=60)
        if len(ann) < 4:
            continue
        lm = lm_factory()
        lm.reset()
        cur_block, cur_stmt = 0, 0

        for i, (tok, block, sub, stmt) in enumerate(ann):
            lm.ingest(tok)
            if block != cur_block:
                cur_block = block
            if stmt != cur_stmt:
                cur_stmt = stmt
            if i + 1 < len(ann):
                ntok, nblock, _, nstmt = ann[i + 1]
                nexp = BLOCK_TO_EXPERT.get(nblock, 0)

                # Features: CA state
                base = lm.features()
                if aug_features:
                    block_oh = np.zeros(5, np.float32)
                    if 0 <= cur_block < 5:
                        block_oh[cur_block] = 1.0
                    stmt_oh = np.zeros(11, np.float32)
                    if 0 <= cur_stmt < 11:
                        stmt_oh[cur_stmt] = 1.0
                    X_all.append(np.concatenate([base, block_oh, stmt_oh]))
                else:
                    X_all.append(base)

                y_tok.append(ntok)
                y_route.append(nexp)

            if len(y_tok) >= cap_pairs:
                break
        if len(y_tok) >= cap_pairs:
            break

    return np.array(X_all, np.float32), np.array(y_tok), np.array(y_route)


def generate_moe(lm_factory, moe, i2w, seed_seq,
                 max_len=30, n=1, rng=None, topk=8, temperature=0.9):
    """MoE генерация:
    1. CA обрабатывает seed → features
    2. Routing → expert_id
    3. Expert readout → token
    4. CA получает token → повтор
    """
    from .hierarchy import BLOCK_MODULE

    rng = rng or np.random.default_rng(0)
    out = []

    for _ in range(n):
        lm = lm_factory()
        lm.reset()
        cur_block, cur_stmt = 0, 0

        # Внести seed
        for t in seed_seq:
            lm.ingest(t)

        seq = list(seed_seq)
        for _ in range(max_len - len(seed_seq)):
            # Features
            base = lm.features()
            block_oh = np.zeros(5, np.float32)
            if 0 <= cur_block < 5:
                block_oh[cur_block] = 1.0
            stmt_oh = np.zeros(11, np.float32)
            if 0 <= cur_stmt < 11:
                stmt_oh[cur_stmt] = 1.0
            feat = np.concatenate([base, block_oh, stmt_oh])

            # Routing
            eid = moe.predict_routing(feat)
            classes, probs = moe.predict_expert(eid, feat)

            # Sample
            logits = np.log(np.maximum(probs, 1e-12)) / temperature
            k = min(topk, len(logits))
            idx = np.argpartition(-logits, k - 1)[:k]
            e = np.exp(logits[idx] - logits[idx].max())
            p = e / e.sum()
            chosen = int(rng.choice(idx, p=p))
            tok_id = int(classes[chosen])

            seq.append(tok_id)
            lm.ingest(tok_id)

            # Обновить контекст (эвристика)
            tok_str = i2w.get(tok_id, " ")
            if tok_str in ("def",):
                cur_block = 2  # BLOCK_DEF
                cur_stmt = 9   # STMT_OTHER
            elif tok_str in ("class",):
                cur_block = 3  # BLOCK_CLASS
            elif tok_str in ("from", "import"):
                cur_block = 1  # BLOCK_IMPORT
                cur_stmt = 6   # STMT_IMPORT
            elif tok_str == ":":
                pass  # stmt body follows

        out.append(seq)
    return out
