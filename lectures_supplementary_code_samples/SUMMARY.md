# Automatic Differentiation Computational Graph Visualizer

This project provides an educational tool that decomposes arbitrary mathematical expressions into computational graphs and visualizes both **forward evaluation** and **reverse-mode automatic differentiation (backpropagation)** using `SymPy` and `Mermaid.js`.

---

## 1. Understanding the Computation Graph

The graph shows how an expression is evaluated in the forward pass and how its gradients are calculated in the backward pass. Read the forward computation from bottom to top and the backward computation from top to bottom.

* **Blue area — Inputs**: The input variables and their values. Each green node shows the value $v(\dots)$ used in the computation and, after backpropagation, the resulting $\text{grad}(\dots)$.
* **Yellow area — Operations**: The individual operations into which the expression has been decomposed. For an expression such as `(x**x) * y`, one operation computes $x^x$, and another multiplies that intermediate result by $y$. Each green operation node shows the expression and its value $v(\dots)$.
* **Pink trapezoids — Local derivatives**: These show how an operation changes with respect to one of its arguments. For example, for the multiplication $b \cdot y$, the local derivatives are $\frac{\partial (b \cdot y)}{\partial b} = y$ and $\frac{\partial (b \cdot y)}{\partial y} = b$. For $b = x^x$, the derivative with respect to $x$ is $x^x \cdot (\ln(x) + 1)$.
* **Green solid arrows — Forward computation ($f$)**: Values flow from the inputs through the operations toward the output. An edge labeled $f = \dots$ shows the numerical value being passed to the next operation.
* **Purple dotted arrows — Backward computation ($g$)**: Gradients flow in the opposite direction, from the output back toward the inputs. An edge labeled $g = \dots$ shows the gradient propagated along that path.
* **Green area — Output**: The final value of the complete expression. The backward computation starts at the output with $g = 1$.

During the forward pass, each operation receives values from below, computes its result, and passes that result upward. During the backward pass, each operation receives a gradient from above and uses its local derivatives to determine the gradients that are propagated toward its arguments.

When an input influences the output through more than one path—as $x$ does in $x^x$—the relevant derivative contributions are combined to obtain the final $\text{grad}(x)$.

*In short: green nodes and solid arrows represent values and forward computation; pink nodes and purple dotted arrows represent local derivatives and gradient propagation.*

---

## 2. Visual & Diagram Conventions

| Element | Mermaid Shape | Color Scheme | Meaning |
| :--- | :--- | :--- | :--- |
| **Output Subgraph** | Rounded Box | Pale Green fill (`#e6ffee`), Dark Green border (`#1e7e34`) | Header summarizes target equation `Output: out = <formula>` |
| **Operation Subgraph** | Large Subgraph | Pale Yellow fill (`#fff3cd`), Dark Amber border (`#d39e00`) | Intermediate computation units (positioned in middle) |
| **Inputs Subgraph** | Rounded Box | Pale Blue fill (`#e6f3ff`), Dark Blue border (`#0d6efd`) | Header summarizes inputs `Inputs: x = ..., y = ...` (aligned horizontally at bottom) |
| **Value Nodes** | Rounded Pill (`(["..."])`) | Pale Green fill (`#d4edda`), Dark Green border (`#1e7e34`) | Variable / operation value $v(\cdot)$ |
| **Derivative Nodes** | Trapezoid (`[\"...\"/]`) | Pale Pink fill (`#fce4ec`), Dark Berry border (`#c2185b`) | Symbolic & numerical local derivative rule |
| **Forward Edges** | Solid Line (`-->`) | Bright Green (`#0a0`, 3.5px) | Forward flow $f = \text{val}$ (pointing upward toward Output) |
| **Backward Edges** | Dashed Line (`<-.->`) | Magenta (`#a0a`, 3px dashed, `marker-end:none`) | Backward adjoint flow $g = \text{grad}$ (pointing downward toward Inputs) |

*The visual graph is declared as `graph BT` with downward backward edges (`child <-.->|"g"| parent` with `marker-end:none`), completely eliminating rank cycles and conflicting invisible links. Every element's border is styled as a dark, saturated variant of its fill color. All input nodes align cleanly on the same horizontal plane directly beneath the Operations subgraph.*

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
