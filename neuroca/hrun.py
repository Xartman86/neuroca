# -*- coding: utf-8 -*-
"""
Итерация 2: curriculum training, усиленный top-down bias, reward за непустоту.
"""
from __future__ import annotations

import numpy as np

from .hierarchical import HierarchicalRegisterCA
from .hierarchy import (
    BLOCK_MODULE, BLOCK_IMPORT, BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP,
    SUB_HEADER, SUB_BODY, SUB_END, SUB_OTHER,
    STMT_ASSIGN, STMT_CALL, STMT_RETURN, STMT_IF, STMT_FOR, STMT_PASS,
    STMT_IMPORT, STMT_DEF, STMT_CLASS, STMT_EXPR, STMT_OTHER,
    is_stmt_allowed, BRACKET_PAIRS, OPENERS, CLOSERS,
)


def is_token_allowed_in_stmt(token_string, current_block, current_stmt,
                              bracket_depth, bracket_stack):
    """Hard mask: разрешён ли token_string как следующий токен?"""
    if current_stmt == STMT_RETURN:
        if token_string in {"def", "class", "import", "from", "if", "for",
                            "while", "return", "pass", "yield", "raise",
                            "break", "continue"}:
            return False
    if current_stmt == STMT_IMPORT:
        if token_string in {"def", "class", "if", "for", "while",
                            "return", "pass"}:
            return False
    if current_stmt == STMT_CLASS:
        if token_string == "return":
            return False
    if current_block == BLOCK_EXPR_TOP:
        if token_string in {"def", "class", "import", "from"}:
            return False
    return True


def collect_curriculum_pairs(frags, tid, rule="kauffman", rule_params=None,
                              T=1, cap_pairs=18000, seed=0):
    """Извлечь обучающие пары для ВСЕХ 4 уровней с усиленным bias.
    Возвращает (X1, y1, w1, X2, y2, w2, X3, y3, w3, X4, y4, w4)
    где w = веса samples (reward за непустоту токенов)."""
    from .hierarchy import annotate_tokens

    def make():
        return HierarchicalRegisterCA(rule=rule, rule_params=rule_params,
                                      T=T, seed=seed)

    X1, y1, w1 = [], [], []  # L1: token
    X2, y2, w2 = [], [], []  # L2: stmt
    X3, y3, w3 = [], [], []  # L3: substruct
    X4, y4, w4 = [], [], []  # L4: block

    NL_TOKEN = 2
    PAD_TOKEN = 0
    WS_TOKEN = 0  # " " = id 0

    for code in frags:
        ann = annotate_tokens(code, tid, max_tokens=60)
        if len(ann) < 4:
            continue
        h = make()
        cur_block, cur_sub, cur_stmt = BLOCK_MODULE, SUB_OTHER, STMT_OTHER
        block_changed = False
        stmt_changed = False

        for i, (tok, block, sub, stmt) in enumerate(ann):
            # Верхние уровни: обновить если изменились
            if block != cur_block:
                h.ingest_meta(block)
                cur_block = block
                block_changed = True
            else:
                block_changed = False
            if stmt != cur_stmt:
                h.ingest_stmt(stmt)
                cur_stmt = stmt
                stmt_changed = True
            else:
                stmt_changed = False
            if sub != cur_sub:
                h.ingest_substruct(sub)
                cur_sub = sub

            # Токен с полным bias
            h.ingest_token(tok, block_id=cur_block, stmt_id=cur_stmt,
                          sub_id=cur_sub, block_changed=block_changed,
                          stmt_changed=stmt_changed)

            if i + 1 < len(ann):
                ntok, nblock, nsub, nstmt = ann[i + 1]
                # Reward за непустоту: non-trivial токены весят 2x
                # (не NEWLINE, не PAD, не WS)
                weight = 2.0 if ntok not in (NL_TOKEN, PAD_TOKEN, WS_TOKEN) \
                               and ntok > 2 else 1.0

                X1.append(h.features(1))
                y1.append(ntok)
                w1.append(weight)

                X2.append(h.features(2))
                y2.append(nstmt)
                w2.append(weight)

                X3.append(h.features(3))
                y3.append(nsub)
                w3.append(weight)

                X4.append(h.features(4))
                y4.append(nblock)
                w4.append(weight)

            if len(y1) >= cap_pairs:
                break
        if len(y1) >= cap_pairs:
            break

    return (np.array(X1, np.float32), np.array(y1, np.int64), np.array(w1, np.float32),
            np.array(X2, np.float32), np.array(y2, np.int64), np.array(w2, np.float32),
            np.array(X3, np.float32), np.array(y3, np.int64), np.array(w3, np.float32),
            np.array(X4, np.float32), np.array(y4, np.int64), np.array(w4, np.float32))


def topdown_generate(tid, i2w, clfs, h_factory, max_len=30, n=1, seed=0,
                     topk=8, temperature=0.9, forbid_empty=True):
    """Top-down генерация с усиленным bias и запретом пустых программ.
    forbid_empty: если True, штрафуем последовательности без непустых stmt."""
    rng = np.random.default_rng(seed)
    NL_TOKEN = 2

    out = []
    attempts = 0
    max_attempts = n * 3 if forbid_empty else n

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        h = h_factory()
        # Стартуем с BOS
        h.ingest_token(0, block_id=BLOCK_MODULE, stmt_id=STMT_OTHER,
                       sub_id=SUB_OTHER)
        seq = [0]
        cur_block = BLOCK_MODULE
        cur_sub = SUB_OTHER
        cur_stmt = STMT_OTHER
        bracket_stack = []
        has_nontrivial = False  # есть ли непустые stmt?

        for _ in range(max_len - 1):
            # 1. Уровень 1 предсказывает токен (с полным bias)
            f1 = h.features(1)
            probs1 = clfs[1].predict_proba(f1[None])[0]

            sampled_id = None
            for _ in range(8):
                idx, p = _topk(probs1, topk, temperature)
                chosen = int(rng.choice(idx, p=p))
                tk_id = int(clfs[1].classes_[chosen])
                tk_str = i2w.get(tk_id, " ")
                if is_token_allowed_in_stmt(tk_str, cur_block, cur_stmt,
                                            len(bracket_stack), bracket_stack):
                    sampled_id = tk_id
                    break
            if sampled_id is None:
                sampled_id = int(clfs[1].classes_[idx[0]])

            seq.append(sampled_id)
            tk_str = i2w.get(sampled_id, " ")

            # Отслеживаем непустые stmt
            if tk_str not in (" ", "\n", "") and tk_str not in OPENERS \
               and tk_str not in CLOSERS and sampled_id != NL_TOKEN \
               and sampled_id > 2:
                has_nontrivial = True

            # 2. Уровень 2 может обновить stmt
            f2 = h.features(2)
            probs2 = clfs[2].predict_proba(f2[None])[0]
            new_stmt = _sample_class(clfs[2], probs2, rng, topk, temperature)
            stmt_changed = new_stmt != cur_stmt and new_stmt < 11
            if stmt_changed:
                cur_stmt = new_stmt

            # 3. Уровень 3 может обновить substruct
            f3 = h.features(3)
            probs3 = clfs[3].predict_proba(f3[None])[0]
            new_sub = _sample_class(clfs[3], probs3, rng, topk, temperature)
            if new_sub != cur_sub and new_sub < 4:
                cur_sub = new_sub

            # 4. Уровень 4 может обновить block
            f4 = h.features(4)
            probs4 = clfs[4].predict_proba(f4[None])[0]
            new_block = _sample_class(clfs[4], probs4, rng, topk, temperature)
            block_changed = new_block != cur_block and new_block < 5
            if block_changed:
                cur_block = new_block

            # Обновить bias в L1
            h.ingest_token(sampled_id, block_id=cur_block, stmt_id=cur_stmt,
                          sub_id=cur_sub, block_changed=block_changed,
                          stmt_changed=stmt_changed)
            if stmt_changed:
                h.ingest_stmt(cur_stmt)
            if block_changed:
                h.ingest_meta(cur_block)

            # Bracket stack
            if tk_str in OPENERS:
                bracket_stack.append(tk_str)
            elif tk_str in CLOSERS:
                if bracket_stack and BRACKET_PAIRS[bracket_stack[-1]] == tk_str:
                    bracket_stack.pop()

        # Фильтр: если forbid_empty и нет непустых stmt — пропускаем
        if forbid_empty and not has_nontrivial:
            continue

        out.append(seq)

    return out


def _topk(probs, k, temperature):
    logits = np.log(np.maximum(probs, 1e-12)) / temperature
    k = min(k, len(logits))
    idx = np.argpartition(-logits, k - 1)[:k]
    p = _softmax(logits[idx])
    return idx, p


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _sample_class(clf, probs, rng, topk, temperature):
    idx, p = _topk(probs, topk, temperature)
    chosen = int(rng.choice(idx, p=p))
    return int(clf.classes_[chosen])


def twophase_generate(tid, i2w, clfs, h_factory, max_len=30, n=1, seed=0,
                      topk=8, temperature=0.9, skeleton_len=3,
                      tokens_per_stmt=5):
    """Two-phase generation:
    Phase 1 (top-down): L4 генерирует block_id, L2 — stmt_id.
      Результат: скелет = [(block, stmt), ...] длиной skeleton_len.
    Phase 2 (bottom-up): для каждого (block, stmt) из скелета L1
      генерирует tokens_per_stmt токенов с явным bias-контекстом.
    Итого: skeleton_len × tokens_per_stmt токенов + NEWLINE между stmt.
    """
    NL_TOKEN = 2
    BOS = 0
    rng = np.random.default_rng(seed)
    out = []

    for _ in range(n):
        # --- Phase 1: skeleton ---
        skeleton = []
        h4 = h_factory()
        h2 = h_factory()
        cur_block = BLOCK_MODULE
        cur_stmt = STMT_OTHER
        for i in range(skeleton_len):
            # L4 предсказывает block
            f4 = h4.features(4)
            probs4 = clfs[4].predict_proba(f4[None])[0]
            new_block = _sample_class(clfs[4], probs4, rng, topk, temperature)
            if new_block < 5:
                cur_block = new_block
            h4.ingest_meta(cur_block)
            # L2 предсказывает stmt
            f2 = h2.features(2)
            probs2 = clfs[2].predict_proba(f2[None])[0]
            new_stmt = _sample_class(clfs[2], probs2, rng, topk, temperature)
            if new_stmt < 11:
                cur_stmt = new_stmt
            h2.ingest_stmt(cur_stmt)
            skeleton.append((cur_block, cur_stmt))

        # --- Phase 2: fill tokens ---
        h1 = h_factory()
        h1.ingest_token(BOS, block_id=skeleton[0][0], stmt_id=skeleton[0][1],
                        sub_id=0, block_changed=True, stmt_changed=True)
        seq = [BOS]
        prev_block, prev_stmt = skeleton[0]

        for si, (block, stmt) in enumerate(skeleton):
            bc = block != prev_block
            sc = stmt != prev_stmt
            if bc or sc:
                # Сигнал смены контекста
                h1.ingest_token(NL_TOKEN, block_id=block, stmt_id=stmt,
                                sub_id=0, block_changed=bc, stmt_changed=sc)
                seq.append(NL_TOKEN)
                prev_block, prev_stmt = block, stmt

            for _ in range(tokens_per_stmt):
                f1 = h1.features(1)
                probs1 = clfs[1].predict_proba(f1[None])[0]
                tk_str = None
                sampled_id = None
                for _ in range(8):
                    idx, p = _topk(probs1, topk, temperature)
                    chosen = int(rng.choice(idx, p=p))
                    tk_id = int(clfs[1].classes_[chosen])
                    tk_str = i2w.get(tk_id, " ")
                    if is_token_allowed_in_stmt(tk_str, block, stmt,
                                                0, []):
                        sampled_id = tk_id
                        break
                if sampled_id is None:
                    sampled_id = int(clfs[1].classes_[idx[0]])
                seq.append(sampled_id)
                h1.ingest_token(sampled_id, block_id=block, stmt_id=stmt,
                                sub_id=0, block_changed=False,
                                stmt_changed=False)

            # NEWLINE между stmt
            if si < len(skeleton) - 1:
                seq.append(NL_TOKEN)
                h1.ingest_token(NL_TOKEN, block_id=block, stmt_id=stmt,
                                sub_id=0, block_changed=False,
                                stmt_changed=False)

        out.append(seq)
    return out
