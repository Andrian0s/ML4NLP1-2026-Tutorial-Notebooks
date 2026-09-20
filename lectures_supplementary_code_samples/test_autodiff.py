"""
Comprehensive Unit Test & Investigation Suite for Autodiff Visualizer
======================================================================
Covers core calculus, security parser, edge cases (negative powers, zero powers,
variable powers), name collisions, propagated gradient accuracy, linkStyle
synchronization, forward-only mode, and layout bands.

Run via:
    python test_autodiff.py
or:
    pytest -v test_autodiff.py
"""

import math
import re
import sympy as sp
from autodiff_visualizer import (
    ComputationGraph,
    safe_parse_expr,
    format_num,
    format_expr,
    MAX_EXPR_LEN,
    MAX_AST_NODES,
    MAX_AST_DEPTH,
    MAX_NUMERIC_MAGNITUDE,
    MAX_EXPONENT_MAGNITUDE,
)


class raises:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected exception {self.expected_exception.__name__} was not raised.")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match and not re.search(self.match, str(exc_val)):
            raise AssertionError(f"Exception message '{exc_val}' does not match pattern '{self.match}'")
        return True

def _run_and_investigate(name, expr_str, feed_dict, root_name="out", forward_only=False):
    print("=" * 70)
    print(f"TEST CASE: {name}")
    print(f"Expression: {expr_str}")
    print(f"Inputs: {feed_dict}")
    print(f"Mode: {'Forward Only' if forward_only else 'Forward & Backward'}")
    print("-" * 70)

    expr = safe_parse_expr(expr_str)
    cg = ComputationGraph()
    cg.build(expr, root_name=root_name)
    cg.evaluate(feed_dict, forward_only=forward_only)

    # 1. Forward pass check
    expected_val = float(expr.subs(feed_dict).evalf())
    print(f"Forward Evaluation: v({cg.root.name}) = {cg.root.val} (Expected: {expected_val})")
    assert abs(cg.root.val - expected_val) < 1e-5, f"Value mismatch: {cg.root.val} vs {expected_val}"

    if not forward_only:
        # 2. Local derivatives check
        print("\nLocal Derivative Rules:")
        for node in cg.nodes:
            if not node.is_input:
                for loc in node.local_derivatives:
                    arg_name = loc["arg_node"].name
                    print(f"  ∂({node.name})/∂({arg_name}) = {loc['sym_expr']}  ==>  evaluated = {loc['local_val']} | contrib = {loc['grad_contrib']}")

        # 3. Reverse-mode gradients check against analytical SymPy
        print("\nReverse-Mode Gradient Verification:")
        for sym, val in feed_dict.items():
            computed_grad = cg.expr_to_node[sym].grad
            true_grad = float(sp.diff(expr, sym).subs(feed_dict).evalf())
            print(f"  ∂({root_name})/∂({sym}) : computed = {computed_grad} | analytical = {true_grad}")
            assert abs(computed_grad - true_grad) < 1e-5, (
                f"Gradient mismatch for {sym}: computed {computed_grad} vs expected {true_grad}"
            )

    # 4. Generate Mermaid code and check structural integrity
    mermaid_code = cg.to_mermaid(output_name=root_name, forward_only=forward_only)

    assert "graph BT" in mermaid_code
    assert "subgraph output" in mermaid_code
    assert "subgraph inputs" in mermaid_code
    assert "color:#000" in mermaid_code

    if forward_only:
        assert "classDef backward" not in mermaid_code
        assert "der_" not in mermaid_code
        assert "Backward computation" not in mermaid_code
    else:
        assert "subgraph operation" in mermaid_code
        assert "classDef backward" in mermaid_code
        assert "g = 1" in mermaid_code

    # Verify linkStyle indices strictly match emitted edges
    fwd_count = len(re.findall(r"-->", mermaid_code))
    bwd_count = len(re.findall(r"<-.->", mermaid_code))
    total_edges = fwd_count + bwd_count

    link_styles = re.findall(r"linkStyle\s+([0-9,]+)", mermaid_code)
    all_styled_indices = []
    for ls in link_styles:
        all_styled_indices.extend(map(int, ls.split(",")))

    assert len(all_styled_indices) == total_edges, (
        f"linkStyle count ({len(all_styled_indices)}) != total edges ({total_edges})"
    )
    if all_styled_indices:
        assert all_styled_indices == list(range(total_edges)), "linkStyle indices must be consecutive 0..N-1"

    print("\nMermaid Snippet (first 10 lines):")
    for line in mermaid_code.splitlines()[:10]:
        print("  " + line)
    print("=" * 70 + "\n")
    return cg, mermaid_code


# ---------------------------------------------------------------------------
# Core Course Examples
# ---------------------------------------------------------------------------

def test_01_simple_multiplication():
    """Introductory slide example: o = x * y"""
    x, y = sp.symbols("x y")
    _run_and_investigate(
        name="01. Simple Multiplication (o = x * y)",
        expr_str="x * y",
        feed_dict={x: 2.0, y: 3.0},
        root_name="o",
    )


def test_02_task_addition_and_square():
    """Slide task: o = (x * x) + y"""
    x, y = sp.symbols("x y")
    _run_and_investigate(
        name="02. Square plus Variable (o = (x * x) + y)",
        expr_str="(x * x) + y",
        feed_dict={x: 3.0, y: 4.0},
        root_name="o",
    )


def test_03_lecture_shared_variable_dag():
    """Main lecture DAG with shared y: z = (x * x) * y + (y + 2)"""
    x, y = sp.symbols("x y")
    _run_and_investigate(
        name="03. Multi-path DAG with Shared y: z = (x*x)*y + (y + 2)",
        expr_str="(x * x) * y + (y + 2)",
        feed_dict={x: 3.0, y: 4.0},
        root_name="z",
    )


def test_04_nested_expression_with_power():
    """Nested specification: L = ((x_1 + x_2) * x_3)**2"""
    x1, x2, x3 = sp.symbols("x_1 x_2 x_3")
    _run_and_investigate(
        name="04. Nested Product & Power: L = ((x_1 + x_2) * x_3)**2",
        expr_str="((x_1 + x_2) * x_3)**2",
        feed_dict={x1: 1.0, x2: 2.0, x3: 3.0},
        root_name="L",
    )


def test_05_subtraction_and_division():
    """Verify subtraction and division: f = (a - b) / (c + 1)"""
    a, b, c = sp.symbols("a b c")
    _run_and_investigate(
        name="05. Subtraction and Division: f = (a - b) / (c + 1)",
        expr_str="(a - b) / (c + 1)",
        feed_dict={a: 10.0, b: 2.0, c: 3.0},
        root_name="f",
    )


def test_06_transcendental_functions():
    """Trigonometric and exponential: g = sin(x) * exp(y)"""
    x, y = sp.symbols("x y")
    _run_and_investigate(
        name="06. Trigonometric and Exponential: g = sin(x) * exp(y)",
        expr_str="sin(x) * exp(y)",
        feed_dict={x: 1.5, y: 0.5},
        root_name="g",
    )


# ---------------------------------------------------------------------------
# Code Review Regressions & Edge Cases
# ---------------------------------------------------------------------------

def test_07_negative_base_constant_power():
    """Review Finding 3: x**2 at x = -3.0 must not introduce ln(x) or complex numbers."""
    x = sp.Symbol("x")
    cg, mermaid = _run_and_investigate(
        name="07. Negative Base Constant Power (x**2 at x = -3.0)",
        expr_str="x**2",
        feed_dict={x: -3.0},
        root_name="out",
    )
    # Check that root value is 9 and gradient of x is -6.0
    assert abs(cg.root.val - 9.0) < 1e-6
    assert abs(cg.expr_to_node[x].grad - (-6.0)) < 1e-6

    # Verify no log or ln appears in the derivative
    for node in cg.nodes:
        for loc in node.local_derivatives:
            sym_str = str(loc["sym_expr"])
            assert "log" not in sym_str and "ln" not in sym_str, f"Unexpected log in power rule: {sym_str}"

    # Verify Mermaid contains exact propagated gradient -6
    assert "g = -6" in mermaid


def test_08_zero_base_constant_power():
    """Zero base with constant power: x**2 at x = 0.0 must evaluate cleanly."""
    x = sp.Symbol("x")
    cg, mermaid = _run_and_investigate(
        name="08. Zero Base Constant Power (x**2 at x = 0.0)",
        expr_str="x**2",
        feed_dict={x: 0.0},
        root_name="out",
    )
    assert abs(cg.root.val - 0.0) < 1e-6
    assert abs(cg.expr_to_node[x].grad - 0.0) < 1e-6
    assert "v(x) = 0" in mermaid
    assert "g = 0" in mermaid


def test_09_variable_exponent():
    """Power with variable exponent: x**y at x=2.0, y=3.0 produces 8.0, with ln(x) for y."""
    x, y = sp.symbols("x y")
    cg, mermaid = _run_and_investigate(
        name="09. Variable Exponent (x**y at x=2.0, y=3.0)",
        expr_str="x**y",
        feed_dict={x: 2.0, y: 3.0},
        root_name="out",
    )
    assert abs(cg.root.val - 8.0) < 1e-6
    # d(x^y)/dx = y * x^(y-1) = 3 * 4 = 12
    assert abs(cg.expr_to_node[x].grad - 12.0) < 1e-6
    # d(x^y)/dy = x^y * ln(x) = 8 * ln(2) = 5.545177
    expected_dy = 8.0 * math.log(2.0)
    assert abs(cg.expr_to_node[y].grad - expected_dy) < 1e-4


def test_10_identifier_collisions():
    """Review Finding 4: Variable names like 'b', 'c', 'out' must not collide with node IDs."""
    b, c, out = sp.symbols("b c out")
    cg, mermaid = _run_and_investigate(
        name="10. Identifier Collisions: out = b * c + b",
        expr_str="b * c + b",
        feed_dict={b: 2.0, c: 5.0},
        root_name="target",
    )
    assert abs(cg.root.val - 12.0) < 1e-6
    assert abs(cg.expr_to_node[b].grad - 6.0) < 1e-6
    assert abs(cg.expr_to_node[c].grad - 2.0) < 1e-6

    # Node IDs must use opaque prefixes
    assert "inp_" in mermaid
    assert "op_" in mermaid
    assert "der_" in mermaid


def test_11_propagated_gradient_values():
    """Review Finding 2: Edges must show propagated adjoints (g = z_bar * dz/dx), not raw local derivatives."""
    x, y = sp.symbols("x y")
    # For (x + y)**2 at x=1, y=2:
    # forward: (1+2)^2 = 9
    # root grad = 1
    # op1 = x + y (val=3)
    # op2 = op1**2 (val=9)
    # op2 local diff wrt op1 = 2 * op1 = 6. op2.grad = 1 -> grad_contrib to op1 = 6.
    # op1.grad accumulated = 6.
    # op1 local diff wrt x = 1, contrib = 6 * 1 = 6.
    # op1 local diff wrt y = 1, contrib = 6 * 1 = 6.
    cg, mermaid = _run_and_investigate(
        name="11. Propagated Adjoints ((x + y)**2 at x=1, y=2)",
        expr_str="(x + y)**2",
        feed_dict={x: 1.0, y: 2.0},
        root_name="out",
    )
    # Ensure edge labels contain "g = 6"
    assert "g = 6" in mermaid
    assert cg.expr_to_node[x].grad == 6.0
    assert cg.expr_to_node[y].grad == 6.0


def test_12_safe_parser_security():
    """Review Finding 1: Unsafe expressions and arbitrary code execution attempts must be rejected."""
    malicious_inputs = [
        "__import__('os').system('ls')",
        "eval('1 + 1')",
        "open('/etc/passwd').read()",
        "lambda x: x",
        "[x for x in [1, 2]]",
        "{'a': 1}",
        "x.foo()",
        "x = 5",
        "while True: pass",
        "x; y",
        "",
        "   ",
        "x" * (MAX_EXPR_LEN + 1),
    ]

    for expr in malicious_inputs:
        with raises(ValueError):
            safe_parse_expr(expr)

    # Large constants exceeding bounds
    with raises(ValueError, match="exceeds maximum allowed magnitude"):
        safe_parse_expr("x + 1000000000000")

    # Exponent exceeding bounds
    with raises(ValueError, match="exceeds limit"):
        safe_parse_expr("x ** 100")

    # Deeply nested AST
    deep_expr = "x"
    for _ in range(MAX_AST_DEPTH + 2):
        deep_expr = f"sin({deep_expr})"
    with raises(ValueError, match="Expression nesting too deep"):
        safe_parse_expr(deep_expr)

    # Valid syntax parses correctly
    parsed = safe_parse_expr("x * y + 2.5")
    assert isinstance(parsed, sp.Expr)


def test_13_formatting_significant_digits():
    """Review Finding 7: format_num and format_expr decoupling and accuracy."""
    assert format_num(0.0) == "0"
    assert format_num(0) == "0"
    assert format_num(3.0) == "3"
    assert format_num(-5.0) == "-5"
    assert format_num(3.14159, sig_figs=4) == "3.142"
    assert format_num(0.001234, sig_figs=4) == "0.001234"
    assert format_num(1e-5, sig_figs=4) == "1.000e-5"

    # Expression formatting preserves ** and spaces
    assert format_expr(sp.Symbol("x") ** 2) == "x**2"
    assert format_expr(sp.Symbol("x") * sp.Symbol("y")) == "x * y"
    assert format_expr(sp.Symbol("x") + sp.Symbol("y")) == "x + y"


def test_14_forward_only_mode():
    """Forward-only generation omits backward passes entirely."""
    x, y = sp.symbols("x y")
    cg, mermaid = _run_and_investigate(
        name="14. Forward Only Mode",
        expr_str="x * y + 3",
        feed_dict={x: 2.0, y: 4.0},
        root_name="out",
        forward_only=True,
    )
    assert "classDef backward" not in mermaid
    assert "der_" not in mermaid
    assert "Backward computation" not in mermaid
    assert "grad(" not in mermaid
    assert "v(out) = 11" in mermaid


def test_15_single_variable_and_constant():
    """Single variable and constant graphs render without error."""
    x = sp.Symbol("x")
    cg, mermaid = _run_and_investigate(
        name="15. Single Variable (out = x)",
        expr_str="x",
        feed_dict={x: 5.0},
        root_name="out",
    )
    assert abs(cg.root.val - 5.0) < 1e-6
    assert abs(cg.expr_to_node[x].grad - 1.0) < 1e-6


if __name__ == "__main__":
    print("\nRUNNING COMPLETE UNIT TEST & INVESTIGATION SUITE...\n")
    test_01_simple_multiplication()
    test_02_task_addition_and_square()
    test_03_lecture_shared_variable_dag()
    test_04_nested_expression_with_power()
    test_05_subtraction_and_division()
    test_06_transcendental_functions()
    test_07_negative_base_constant_power()
    test_08_zero_base_constant_power()
    test_09_variable_exponent()
    test_10_identifier_collisions()
    test_11_propagated_gradient_values()
    test_12_safe_parser_security()
    test_13_formatting_significant_digits()
    test_14_forward_only_mode()
    test_15_single_variable_and_constant()
    print("\nALL 15 TESTS PASSED SUCCESSFULLY! 🎉\n")
