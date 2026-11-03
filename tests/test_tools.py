"""
The calculator tool.

It evaluates model-written expressions on a shared server, so the tests that
matter most are the ones proving what it refuses.
"""

from __future__ import annotations

import pytest

from edupilot.agents.tools import (
    MAX_EXPRESSION_CHARS,
    TUTOR_TOOLS,
    CalculatorError,
    calculate,
    evaluate,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("variance([2, 4, 4, 4, 5, 5, 7, 9])", "4.57143"),
        ("pstdev([2, 4, 4, 4, 5, 5, 7, 9])", "2"),
        ("mean([1, 2, 3, 4])", "2.5"),
        ("median([3, 1, 2])", "2"),
        ("zscore(85, 70, 10)", "1.5"),
        ("normal_cdf(1.96)", "0.975002"),
        ("normal_ppf(0.975)", "1.95996"),
        ("binom_pmf(3, 10, 0.5)", "0.117188"),
        ("comb(5, 2)", "10"),
        ("slope([1, 2, 3], [2, 4, 6])", "2"),
        ("correlation([1, 2, 3], [2, 4, 7])", "0.993399"),
        ("2 ** 10", "1024"),
        ("sqrt(16) + pi - pi", "4"),
        ("normal_cdf(110, mean=100, sd=15)", "0.747507"),
    ],
)
def test_statistics_a_student_needs(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "().__class__.__bases__",
        "open('/etc/passwd')",
        "[x for x in range(10)]",
        "lambda: 1",
        "'a' * 3",
        "exec('1')",
        "x",
        "mean.__globals__",
        "f(*[1, 2])",
        "max(**{'a': 1})",
        "9 ** 9 ** 9",
        "[0] * 100000000",
        "factorial(100000)",
        "True + 1",
    ],
)
def test_anything_outside_the_whitelist_is_refused(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_expression_length_is_bounded():
    with pytest.raises(CalculatorError):
        evaluate("1+" * MAX_EXPRESSION_CHARS + "1")


@pytest.mark.parametrize("expression", ["1 / 0", "log(0)", "mean([])", "sqrt(-1)", "normal_ppf(1.5)"])
def test_maths_errors_are_reported_not_raised_raw(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_the_tool_is_exposed_to_the_model_with_its_documentation():
    assert TUTOR_TOOLS == [calculate]
    assert calculate.name == "calculate"
    assert set(calculate.args) == {"expression"}
    assert "normal_cdf" in calculate.description
    assert calculate.invoke({"expression": "mean([1, 3])"}) == "2"
