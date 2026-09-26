# -*- coding: utf-8 -*-
"""
Иерархия Python-фрагментов через AST.

Каждый токен получает 4 тега:
  block      ∈ {0=module, 1=import, 2=def, 3=class, 4=expr_top}
  substruct  ∈ {0=header, 1=body, 2=end, 3=other}
  stmt       ∈ {0=assign, 1=call, 2=return, 3=if, 4=for, 5=pass,
                 6=import, 7=def, 8=class, 9=expr, 10=other}
  token_id   (из существующего словаря S5)

block, substruct, stmt — новые уровни иерархии (уровни 4, 3, 2).
token_id — уровень 1 (существующий).
"""
import ast
from io import StringIO
from tokenize import generate_tokens, NEWLINE
from tokenize import NL, COMMENT, ENDMARKER, INDENT, DEDENT, ENCODING


# Категории верхнего уровня (block)
BLOCK_MODULE = 0
BLOCK_IMPORT = 1
BLOCK_DEF = 2
BLOCK_CLASS = 3
BLOCK_EXPR_TOP = 4  # выражение на верхнем уровне (type=Expression в ast)

# Подструктуры внутри блока
SUB_HEADER = 0  # "def name(...):" / "class Foo:" / "from x import y"
SUB_BODY = 1
SUB_END = 2
SUB_OTHER = 3

# Типы утверждений
STMT_ASSIGN = 0
STMT_CALL = 1
STMT_RETURN = 2
STMT_IF = 3
STMT_FOR = 4
STMT_PASS = 5
STMT_IMPORT = 6
STMT_DEF = 7
STMT_CLASS = 8
STMT_EXPR = 9
STMT_OTHER = 10


def _stmt_kind(node):
    """Тип stmt-узла в число."""
    if isinstance(node, ast.Assign):
        return STMT_ASSIGN
    if isinstance(node, ast.AugAssign):
        return STMT_ASSIGN
    if isinstance(node, ast.Expr):
        return STMT_EXPR  # выражение-утверждение (например, вызов)
    if isinstance(node, ast.Return):
        return STMT_RETURN
    if isinstance(node, ast.If):
        return STMT_IF
    if isinstance(node, ast.For):
        return STMT_FOR
    if isinstance(node, ast.While):
        return STMT_FOR  # while трактуем как for-like
    if isinstance(node, ast.Pass):
        return STMT_PASS
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return STMT_IMPORT
    if isinstance(node, ast.FunctionDef):
        return STMT_DEF
    if isinstance(node, ast.AsyncFunctionDef):
        return STMT_DEF
    if isinstance(node, ast.ClassDef):
        return STMT_CLASS
    return STMT_OTHER


def _block_for(node):
    """Блок верхнего уровня для узла Module.body или ClassDef.body."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return BLOCK_IMPORT
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return BLOCK_DEF
    if isinstance(node, ast.ClassDef):
        return BLOCK_CLASS
    if isinstance(node, ast.Expr):
        return BLOCK_EXPR_TOP
    return BLOCK_MODULE


def extract_spans(code):
    """Возвращает список (start_line, end_line, block, substruct, stmt)
    — границы span'ов в исходном коде (по номерам строк 1-based)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    spans = []
    lines = code.splitlines()
    _walk(tree, None, BLOCK_MODULE, SUB_OTHER, STMT_OTHER, spans, lines)
    return spans


def _walk(node, parent_block, block, substruct, stmt, spans, lines):
    if isinstance(node, ast.Module):
        for child in node.body:
            b = _block_for(child)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk_function(child, b, spans, lines)
            elif isinstance(child, ast.ClassDef):
                _walk_class(child, b, spans, lines)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                s = _stmt_kind(child)
                spans.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                              b, SUB_OTHER, s))
            elif isinstance(child, ast.Expr):
                s = _stmt_kind(child)
                spans.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                              b, SUB_OTHER, s))
            else:
                s = _stmt_kind(child)
                spans.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                              b, SUB_OTHER, s))


def _walk_function(node, parent_block, spans, lines):
    """def: header (всё до ':') + body."""
    # Header: до строки, где ":"
    header_end = node.body[0].lineno - 1 if node.body else node.lineno
    spans.append((node.lineno, header_end, parent_block, SUB_HEADER, STMT_DEF))
    # Тело: каждое утверждение
    for child in node.body:
        s = _stmt_kind(child)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_function(child, BLOCK_DEF, spans, lines)
        elif isinstance(child, ast.ClassDef):
            _walk_class(child, BLOCK_CLASS, spans, lines)
        else:
            sub = SUB_OTHER if not isinstance(child, ast.If) else SUB_OTHER
            spans.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                          BLOCK_DEF, sub, s))


def _walk_class(node, parent_block, spans, lines):
    header_end = node.body[0].lineno - 1 if node.body else node.lineno
    spans.append((node.lineno, header_end, parent_block, SUB_HEADER, STMT_CLASS))
    for child in node.body:
        s = _stmt_kind(child)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_function(child, BLOCK_DEF, spans, lines)
        elif isinstance(child, ast.ClassDef):
            _walk_class(child, BLOCK_CLASS, spans, lines)
        else:
            spans.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                          BLOCK_CLASS, SUB_OTHER, s))


# Связь stmt -> допустимые parent_block
STMT_PARENT_CONSTRAINTS = {
    STMT_ASSIGN: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP},
    STMT_CALL: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP},
    STMT_RETURN: {BLOCK_DEF},                # return только внутри def
    STMT_IF: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS},
    STMT_FOR: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS},
    STMT_PASS: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP},
    STMT_IMPORT: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS},  # import внутри def нелогичен, но не невалиден
    STMT_DEF: {BLOCK_MODULE, BLOCK_CLASS},
    STMT_CLASS: {BLOCK_MODULE, BLOCK_CLASS},
    STMT_EXPR: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP},
    STMT_OTHER: {BLOCK_MODULE, BLOCK_DEF, BLOCK_CLASS, BLOCK_EXPR_TOP},
}


def is_stmt_allowed(stmt, parent_block):
    """Жёсткая маска: stmt разрешён в parent_block?"""
    return parent_block in STMT_PARENT_CONSTRAINTS.get(stmt, set())


# Пары открывающий/закрывающий (token_string)
BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
}
OPENERS = set(BRACKET_PAIRS.keys())
CLOSERS = set(BRACKET_PAIRS.values())


def token_kind(t_string):
    """Классифицировать токен по строке: opener / closer / newblock / endblock / other."""
    if t_string in OPENERS:
        return "opener"
    if t_string in CLOSERS:
        return "closer"
    if t_string == ":":
        return "newblock"   # def f(x):, if x:, for x in y:
    if t_string == "return":
        return "may_end_block"  # часто завершает блок
    if t_string in {"elif", "else", "except", "finally"}:
        return "newblock"
    return "other"


def annotate_tokens(code, tid, max_tokens=60):
    """Возвращает список (token_id, block, substruct, stmt) для каждого токена
    (включая NEWLINE как id 2). Использует extract_spans для разметки
    верхних уровней по строкам.

    Точное соответствие строк: tokenize даёт start_line, span даёт lineno.
    Берём последний span, чей start_line <= token.start_line.
    """
    SKIP = {NL, COMMENT, ENDMARKER, INDENT, DEDENT, ENCODING}
    spans = extract_spans(code)
    out = []
    try:
        for t in generate_tokens(StringIO(code).readline):
            if t.type in SKIP:
                continue
            line = t.start[0]
            block, substruct, stmt = _tag_at(line, spans)
            if t.type == NEWLINE:
                out.append((2, block, substruct, stmt))
            else:
                out.append((tid.get(t.string, 1), block, substruct, stmt))
            if len(out) >= max_tokens:
                break
    except Exception:
        return []
    return out


def _tag_at(line, spans):
    """Найти последний span с lineno <= line. Вернуть (block, sub, stmt)."""
    block, sub, stmt = BLOCK_MODULE, SUB_OTHER, STMT_OTHER
    for sl, el, b, s, st in spans:
        if sl <= line:
            block, sub, stmt = b, s, st
        else:
            break
    return block, sub, stmt
