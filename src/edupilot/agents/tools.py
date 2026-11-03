"""
Tutor tools
===========
The `calculate` tool the answering agent can call for arithmetic and
statistics on numbers a student supplies.

Why a restricted evaluator rather than a Python sandbox: this server is shared
by every student using the app, and executing model-written code in-process
cannot be made safe — `exec` with a trimmed `__builtins__` is escapable in a
line. Real isolation means a separate container or microVM per call, which is
infrastructure this deployment does not have. What students actually need
here is a calculator: means, variances, z-scores, normal and binomial
probabilities, regression on a handful of points. So the expression is parsed
into an AST and only a whitelist is evaluated — numbers, list literals,
arithmetic, and named math/statistics functions. No names, attributes,
subscripts, imports, comprehensions, or calls outside the whitelist can
evaluate, and sizes are bounded so an expression cannot pin the CPU.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

#: Longest expression accepted, in characters.
MAX_EXPRESSION_CHARS = 2000
#: Most elements a list literal may hold.
MAX_LIST_LENGTH = 5000
#: Largest exponent for `**`, and largest integer argument to factorial/comb.
MAX_EXPONENT = 1000
MAX_FACTORIAL = 1000
#: Results beyond this magnitude are rejected rather than printed.
MAX_MAGNITUDE = 1e300


class CalculatorError(ValueError):
    """The expression is not allowed, or cannot be evaluated."""


# ---------------------------------------------------------------------------
# Whitelisted functions
# ---------------------------------------------------------------------------


def _bounded_int(n: Any, name: str) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise CalculatorError(f"{name} needs whole numbers")
    if abs(n) > MAX_FACTORIAL:
        raise CalculatorError(f"{name} arguments are limited to {MAX_FACTORIAL}")
    return n


def _factorial(n: int) -> int:
    return math.factorial(_bounded_int(n, "factorial"))


def _comb(n: int, k: int) -> int:
    return math.comb(_bounded_int(n, "comb"), _bounded_int(k, "comb"))


def _perm(n: int, k: int) -> int:
    return math.perm(_bounded_int(n, "perm"), _bounded_int(k, "perm"))


def _zscore(x: float, mean: float, sd: float) -> float:
    if sd == 0:
        raise CalculatorError("standard deviation must be non-zero")
    return (x - mean) / sd


def _normal_cdf(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    return statistics.NormalDist(mean, sd).cdf(x)


def _normal_ppf(p: float, mean: float = 0.0, sd: float = 1.0) -> float:
    if not 0 < p < 1:
        raise CalculatorError("normal_ppf needs a probability strictly between 0 and 1")
    return statistics.NormalDist(mean, sd).inv_cdf(p)


def _binom_pmf(k: int, n: int, p: float) -> float:
    return _comb(n, k) * p**k * (1 - p) ** (n - k)


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(_binom_pmf(i, n, p) for i in range(0, _bounded_int(k, "binom_cdf") + 1))


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / _factorial(k)


def _slope(x: list, y: list) -> float:
    return statistics.linear_regression(x, y).slope


def _intercept(x: list, y: list) -> float:
    return statistics.linear_regression(x, y).intercept


FUNCTIONS: dict[str, Callable[..., Any]] = {
    # arithmetic
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "log2": math.log2, "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "factorial": _factorial, "comb": _comb, "perm": _perm,
    "erf": math.erf, "erfc": math.erfc,
    # descriptive statistics
    "mean": statistics.fmean, "median": statistics.median, "mode": statistics.mode,
    "stdev": statistics.stdev, "pstdev": statistics.pstdev,
    "variance": statistics.variance, "pvariance": statistics.pvariance,
    "correlation": statistics.correlation, "covariance": statistics.covariance,
    "geometric_mean": statistics.geometric_mean, "harmonic_mean": statistics.harmonic_mean,
    # distributions and inference
    "zscore": _zscore, "normal_cdf": _normal_cdf, "normal_ppf": _normal_ppf,
    "binom_pmf": _binom_pmf, "binom_cdf": _binom_cdf, "poisson_pmf": _poisson_pmf,
    "slope": _slope, "intercept": _intercept,
}  # fmt: skip

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError("only numbers are allowed")
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_LIST_LENGTH:
            raise CalculatorError(f"lists are limited to {MAX_LIST_LENGTH} values")
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalculatorError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(left, list) or isinstance(right, list):
            # `[0] * 10**9` would allocate without limit; lists only go into functions.
            raise CalculatorError("arithmetic on lists is not allowed; pass them to a function")
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponents are limited to {MAX_EXPONENT}")
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int):
            # Integer powers grow without bound; float keeps them finite.
            left = float(left)
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else "that"
            raise CalculatorError(f"function {name!r} is not available")
        args = [_eval(a) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise CalculatorError("argument unpacking is not allowed")
            kwargs[kw.arg] = _eval(kw.value)
        return FUNCTIONS[node.func.id](*args, **kwargs)
    raise CalculatorError(f"{type(node).__name__} is not allowed in an expression")


def _format(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value) or abs(value) > MAX_MAGNITUDE:
            raise CalculatorError("result is not a finite number")
        return f"{value:.6g}"
    if isinstance(value, int):
        if abs(value) > MAX_MAGNITUDE:
            raise CalculatorError("result is too large")
        return str(value)
    raise CalculatorError("result is not a number")


def evaluate(expression: str) -> str:
    """
    Evaluate a whitelisted math/statistics expression and format the result.

    Raises:
        CalculatorError: the expression uses anything outside the whitelist,
            exceeds a size limit, or fails mathematically.
    """
    text = (expression or "").strip()
    if not text:
        raise CalculatorError("empty expression")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(f"expressions are limited to {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"not a valid expression: {exc.msg}") from exc
    try:
        return _format(_eval(tree))
    except CalculatorError:
        raise
    except (ArithmeticError, ValueError, TypeError, statistics.StatisticsError) as exc:
        raise CalculatorError(str(exc) or type(exc).__name__) from exc


@tool("calculate")
def calculate(expression: str) -> str:
    """Evaluate an arithmetic or statistics expression on numbers the student gave.

    Write one Python-style expression using numbers, list literals, + - * / // % **,
    and these functions: mean, median, mode, stdev, pstdev, variance, pvariance,
    correlation, covariance, zscore(x, mean, sd), normal_cdf(x, mean=0, sd=1),
    normal_ppf(p, mean=0, sd=1), binom_pmf(k, n, p), binom_cdf(k, n, p),
    poisson_pmf(k, lam), slope(x, y), intercept(x, y), sqrt, exp, log, comb, perm,
    factorial, abs, round, min, max, sum, len. Constants: pi, e.
    Example: variance([2, 4, 4, 4, 5, 5, 7, 9])
    """
    return evaluate(expression)


TUTOR_TOOLS = [calculate]
