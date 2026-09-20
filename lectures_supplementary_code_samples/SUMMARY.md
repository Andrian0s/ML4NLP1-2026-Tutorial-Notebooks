# Automatic Differentiation Computational Graph Visualizer

This project provides an educational tool that decomposes arbitrary mathematical expressions into computational graphs and visualizes both **forward evaluation** and **reverse-mode automatic differentiation (backpropagation)** using `SymPy` and `Mermaid.js`.

---

## 1. Core Pedagogical Concept

Modern deep learning frameworks compute gradients automatically via computational graphs. Rather than presenting reverse-mode autodiff as a black-box algorithm, this tool visualizes each step of the chain rule as a structured computational unit:

1. **Top-Down Computational Hierarchy**:
   - **Top (`Output`)**: An `Output: out = <formula>` subgraph containing a clean target node displaying the evaluated scalar value ($v(\text{out})$ and $\text{out}$) without repeating operation syntax.
   - **Middle (`Operations`)**: The sequence of decomposed elementary operations ($+, -, \times, \div, \text{pow}, \dots$).
   - **Bottom (`Inputs`)**: An `Inputs: x = ..., y = ...` subgraph containing leaf input variables and constants, laid out in an aligned horizontal row.

2. **Explicit Local Calculus & Adjoint Propagation**:
   - Every elementary operation node exposes its **local partial derivatives** as dedicated trapezoid blocks (e.g., $\frac{\partial (x \cdot y)}{\partial x} = y$).
   - Downstream adjoints (incoming gradients $g$) are multiplied by the local derivative ($g = \bar{z} \cdot \frac{\partial z}{\partial x}$) on the backward edges to propagate backward to child nodes.
   - Shared input variables accumulate gradients across branching paths ($g(y) = \sum g_i$).

3. **Dual Data-Flow Semantics**:
   - **Forward Pass (Solid Green $\rightarrow$)**: Propagates numerical values upward: $f = v(\text{child})$.
   - **Backward Pass (Dashed Magenta $\dashrightarrow$)**: Propagates adjoints / gradients downward: $g = \bar{z} \cdot \frac{\partial z}{\partial x}$.
   - **Forward-Only Mode**: Bypasses derivative construction and backpropagation entirely, providing a clean forward graph for introductory pedagogy.

---

## 2. Visual & Diagram Conventions

| Element | Mermaid Shape | Color Scheme | Meaning |
| :--- | :--- | :--- | :--- |
| **Output Subgraph** | Rounded Box | Pale Green (`#e6ffee`) | Header summarizes target equation `Output: out = <formula>` |
| **Operation Subgraph** | Large Subgraph | Pale Yellow (`#fff2cc`) | Intermediate computation units (positioned in middle) |
| **Inputs Subgraph** | Rounded Box | Pale Blue (`#e6f3ff`) | Header summarizes inputs `Inputs: x = ..., y = ...` (aligned horizontally at bottom) |
| **Value Nodes** | Rounded Pill (`(["..."])`) | Pale Green Fill, Dark Border | Variable / operation value $v(\cdot)$ |
| **Derivative Nodes** | Trapezoid (`[\"...\"/]`) | Pale Pink Fill (`#fcf`) | Symbolic & numerical local derivative rule |
| **Forward Edges** | Solid Line (`-->`) | Bright Green (`#0a0`, 3.5px) | Forward flow $f = \text{val}$ (pointing upward toward Output) |
| **Backward Edges** | Dashed Line (`<-.-`) | Magenta (`#a0a`, 3px dashed) | Backward adjoint flow $g = \text{grad}$ (pointing downward toward Inputs) |

*The visual graph is declared as `graph BT` with downward backward edge definitions (`target <-.-|"g"| source`), completely eliminating rank cycles and conflicting invisible links. All input nodes align cleanly on the same horizontal plane directly beneath the Operations subgraph.*

---

## 3. Security & Engineering Architecture

```
autodiff-visualizer/
├── autodiff_visualizer.py   # Safe AST parser, DAG builder, forward/backward passes, Mermaid generator
├── app.py                   # Gradio Web UI: Interactive formula entry, mode toggles, bounded cache, downloads
├── test_autodiff.py         # Test suite: 15 comprehensive unit & regression verification cases
├── requirements.txt         # Minimal production dependencies (gradio, sympy)
└── README.md                # Hugging Face Space configuration and documentation
```

### Key Engineering Features

- **Safe AST-Based Mathematical Parser (`safe_parse_expr`)**:
  - Replaces unsafe `sympify` / Python `eval` with an explicit AST allowlist validator (`ast.parse(mode="eval")`).
  - Limits expression length ($\le 200$ chars), AST nodes ($\le 80$), AST depth ($\le 15$), numeric constants ($\le 10^9$), and exponents ($\le 50$).
  - Restricts functions strictly to approved mathematical operations (`sin`, `cos`, `tan`, `exp`, `log`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `abs`).
- **Power Derivative Robustness**:
  - Distinguishes constant exponents ($x^c \implies c x^{c-1}$) from variable exponents ($c^x \implies c^x \ln c$, $x^y$).
  - Supports negative bases ($(-3)^2 = 9, \frac{\partial}{\partial x} = -6$) and zero bases without introducing complex logarithm branches or NaN.
- **Collision-Free Opaque Node IDs**:
  - Nodes use prefixed opaque identifiers (`out_target`, `op_1`, `inp_1_x`, `der_op_1_inp_1_0`) to prevent graph merging when variable names match intermediate names (e.g. `b`, `c`, `out`).
- **1:1 Discrete Edge Statements & LinkStyle Counting**:
  - Multi-hop edges are split into distinct lines, ensuring Mermaid's link index matches `linkStyle` statements with zero index drift.
- **Bounded Request Caching (`prune_cache`)**:
  - Generated SVG/PNG assets are stored with unique request UUIDs in a pooled cache directory, with automated pruning by age ($> 1$ hour) and file count ($\le 60$).

---

## 4. Verification & Testing

The test suite (`test_autodiff.py`) contains 15 comprehensive automated test cases:
1. **Simple Multiplication**: $o = x \cdot y$
2. **Unary / Binary Operations**: $o = x^2 + y$
3. **Multi-Path Shared Variables (DAGs)**: $z = x^2 y + (y + 2)$ (verifying gradient addition along branching paths)
4. **Nested Powers & Products**: $L = ((x_1 + x_2) x_3)^2$
5. **Fractions & Subtractions**: $f = \frac{a - b}{c + 1}$
6. **Transcendental Functions**: $g = \sin(x) \cdot e^y$
7. **Negative Base Constant Power**: $x^2$ at $x = -3.0$ ($\text{grad} = -6.0$, no complex log)
8. **Zero Base Constant Power**: $x^2$ at $x = 0.0$ ($\text{grad} = 0.0$)
9. **Variable Exponent**: $x^y$ at $x=2.0, y=3.0$ ($\text{grad}_x = 12.0$, $\text{grad}_y = 8 \ln 2$)
10. **Identifier Collisions**: $b \cdot c + b$ (verifying `b` and `c` as user symbols without collision)
11. **Propagated Adjoints**: $(x+y)^2$ (verifying propagated gradient $g = 6.0$ on edges)
12. **Safe Parser Security**: Verifies rejection of `__import__`, `eval`, `open`, list comprehensions, excessive length, huge constants, and deep nesting.
13. **Formatting & Significant Digits**: Verifies `format_num` and `format_expr` decoupling.
14. **Forward-Only Mode**: Verifies complete omission of backward elements and styling.
15. **Single Variable & Constant Graphs**: $o = x$.
