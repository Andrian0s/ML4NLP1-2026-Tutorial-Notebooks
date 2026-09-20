import ast
import math
import re
import sympy as sp

# ---------------------------------------------------------------------------
# 1. Safe AST-Based Mathematical Parser (No eval, No sympify)
# ---------------------------------------------------------------------------

ALLOWED_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "Abs": sp.Abs,
    "abs": sp.Abs,
}

MAX_EXPR_LEN = 200
MAX_AST_NODES = 80
MAX_AST_DEPTH = 15
MAX_NUMERIC_MAGNITUDE = 1e9
MAX_EXPONENT_MAGNITUDE = 50


class ASTValidator(ast.NodeVisitor):
    def __init__(self):
        self.node_count = 0
        self.max_depth = 0
        self.current_depth = 0

    def visit(self, node):
        self.node_count += 1
        if self.node_count > MAX_AST_NODES:
            raise ValueError(f"Expression exceeds complexity limit ({MAX_AST_NODES} nodes).")
        
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            self.max_depth = self.current_depth
        if self.max_depth > MAX_AST_DEPTH:
            raise ValueError(f"Expression nesting too deep (max depth {MAX_AST_DEPTH}).")
            
        res = super().visit(node)
        self.current_depth -= 1
        return res

    def generic_visit(self, node):
        # Disallow any unsupported AST node types
        allowed_types = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.UAdd,
            ast.USub,
            ast.Name,
            ast.Constant,
            ast.Call,
            ast.Load,
        )
        # Compatibility with Python < 3.8
        if hasattr(ast, "Num"):
            allowed_types = allowed_types + (ast.Num,)

        if not isinstance(node, allowed_types):
            raise ValueError(f"Disallowed construct: {type(node).__name__}")
        super().generic_visit(node)


def ast_to_sympy(node):
    if isinstance(node, ast.Expression):
        return ast_to_sympy(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants are allowed, got: {type(node.value).__name__}")
        if abs(node.value) > MAX_NUMERIC_MAGNITUDE:
            raise ValueError(f"Numeric constant {node.value} exceeds maximum allowed magnitude ({MAX_NUMERIC_MAGNITUDE}).")
        return sp.Number(node.value)

    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        if abs(node.n) > MAX_NUMERIC_MAGNITUDE:
            raise ValueError(f"Numeric constant {node.n} exceeds maximum allowed magnitude.")
        return sp.Number(node.n)

    if isinstance(node, ast.Name):
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", node.id):
            raise ValueError(f"Invalid variable identifier: {node.id}")
        return sp.Symbol(node.id)

    if isinstance(node, ast.UnaryOp):
        operand = ast_to_sympy(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = ast_to_sympy(node.left)
        right = ast_to_sympy(node.right)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Pow):
            # Bound constant exponent
            if isinstance(right, sp.Number) and abs(float(right)) > MAX_EXPONENT_MAGNITUDE:
                raise ValueError(f"Exponent magnitude {right} exceeds limit of {MAX_EXPONENT_MAGNITUDE}.")
            return sp.Pow(left, right)
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls (e.g. sin(x)) are allowed.")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCS:
            raise ValueError(f"Function '{func_name}' is not in the allowlist: {sorted(ALLOWED_FUNCS.keys())}")
        if len(node.args) != 1:
            raise ValueError(f"Function '{func_name}' expects exactly 1 argument, got {len(node.args)}.")
        if node.keywords:
            raise ValueError(f"Keyword arguments are not allowed in function calls.")
        arg_sp = ast_to_sympy(node.args[0])
        return ALLOWED_FUNCS[func_name](arg_sp)

    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def safe_parse_expr(expr_str):
    if not isinstance(expr_str, str):
        raise ValueError("Expression must be a string.")
    cleaned = expr_str.strip()
    if not cleaned:
        raise ValueError("Expression cannot be empty.")
    if len(cleaned) > MAX_EXPR_LEN:
        raise ValueError(f"Expression too long ({len(cleaned)} chars, max {MAX_EXPR_LEN}).")

    try:
        parsed_ast = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e.msg}")

    validator = ASTValidator()
    validator.visit(parsed_ast)

    return ast_to_sympy(parsed_ast)


# ---------------------------------------------------------------------------
# 2. Formatting Helpers (Separated Typography & Numbers)
# ---------------------------------------------------------------------------

def format_num(val, sig_figs=4):
    if val is None:
        return ""
    val = float(val)
    if val == 0.0 or abs(val) < 1e-12:
        return "0"
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e9:
        return str(int(round(val)))
    if abs(val) < 1e-3 or abs(val) >= 1e5:
        s = f"{val:.{sig_figs-1}e}"
        return s.replace("e+0", "e+").replace("e-0", "e-")
    
    s = f"{val:.{sig_figs}g}"
    if 'e' in s:
        s = f"{val:.4f}".rstrip('0').rstrip('.')
    return s


def format_expr(expr):
    """Cleanly stringifies expressions without turning ** into * *."""
    s = str(expr)
    # Add spacing around standalone binary operators + and -, but not in scientific notation 1e-5
    s = re.sub(r'(?<![eE])\s*\+\s*', ' + ', s)
    s = re.sub(r'(?<![eE])\s*\-\s*', ' - ', s)
    # Space multiplication * that is not part of **
    s = re.sub(r'(?<!\*)\*(?!\*)', ' * ', s)
    # Clean up any duplicate spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---------------------------------------------------------------------------
# 3. Graph Nodes & Computation Graph Engine
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, uid, name, expr, is_input=False, is_constant=False):
        self.uid = uid            # Opaque unique graph identifier (e.g. inp_0, op_1)
        self.name = name          # Mathematical display name (e.g. x, y, b, out)
        self.expr = expr          # SymPy expression
        self.is_input = is_input
        self.is_constant = is_constant
        self.args = []
        self.val = None
        self.grad = 0.0
        # List of dicts: {'arg_node': Node, 'sym_expr': Expr, 'local_val': float, 'grad_contrib': float}
        self.local_derivatives = []


class ComputationGraph:
    def __init__(self):
        self.nodes = []
        self.expr_to_node = {}
        self.op_counter = 0
        self.inp_counter = 0
        self.cst_counter = 0
        self.orig_expr = None
        self.root = None

    def _get_op_display_name(self):
        # Letters a, b, c for first few operations, then n1, n2...
        if self.op_counter <= 3:
            return chr(ord('a') + self.op_counter - 1)
        return f"n{self.op_counter - 3}"

    def build(self, expr, root_name="out", forward_only=False):
        self.nodes = []
        self.expr_to_node = {}
        self.op_counter = 0
        self.inp_counter = 0
        self.cst_counter = 0
        self.orig_expr = expr

        self.root = self._build_recursive(expr, forward_only=forward_only)
        if root_name and not self.root.is_input:
            self.root.name = root_name
        return self

    def _build_recursive(self, expr, forward_only=False):
        if expr in self.expr_to_node:
            return self.expr_to_node[expr]

        if isinstance(expr, sp.Symbol):
            self.inp_counter += 1
            node = Node(
                uid=f"inp_{self.inp_counter}_{str(expr)}",
                name=str(expr),
                expr=expr,
                is_input=True,
                is_constant=False,
            )
            self.nodes.append(node)
            self.expr_to_node[expr] = node
            return node

        if isinstance(expr, sp.Number):
            self.cst_counter += 1
            node = Node(
                uid=f"cst_{self.cst_counter}_{str(expr).replace('-', 'neg_').replace('.', '_')}",
                name=str(expr),
                expr=expr,
                is_input=True,
                is_constant=True,
            )
            self.nodes.append(node)
            self.expr_to_node[expr] = node
            return node

        # Operation node
        self.op_counter += 1
        node_uid = f"op_{self.op_counter}"
        node_name = self._get_op_display_name()
        node = Node(uid=node_uid, name=node_name, expr=expr, is_input=False, is_constant=False)
        self.expr_to_node[expr] = node

        args = expr.args
        for arg in args:
            child_node = self._build_recursive(arg, forward_only=forward_only)
            node.args.append(child_node)

        # In Forward Only mode, completely bypass symbolic derivative construction
        if forward_only:
            self.nodes.append(node)
            return node

        dummy_symbols = [sp.Symbol(f"_u{i}") for i in range(len(node.args))]
        if isinstance(expr, sp.Add):
            dummy_op = sp.Add(*dummy_symbols)
        elif isinstance(expr, sp.Mul):
            dummy_op = sp.Mul(*dummy_symbols)
        elif isinstance(expr, sp.Pow):
            dummy_op = sp.Pow(dummy_symbols[0], dummy_symbols[1])
        else:
            dummy_op = expr.func(*dummy_symbols)

        # Mapping from dummy symbols to display symbols using child names
        display_map = {dummy_symbols[i]: sp.Symbol(node.args[i].name) for i in range(len(node.args))}

        if isinstance(expr, sp.Pow):
            base_node, exp_node = node.args[0], node.args[1]
            # Case A: Constant exponent (e.g. x**2, (x+y)**3)
            if exp_node.is_constant:
                c = float(exp_node.expr)
                c_disp = int(c) if c == int(c) else c
                base_sym = sp.Symbol(base_node.name)
                sym_d = c_disp * (base_sym ** (c_disp - 1)) if c_disp != 1 else sp.Number(1)
                node.local_derivatives.append({
                    'arg_node': base_node,
                    'sym_expr': sym_d,
                    'is_pow_constant_exp': True,
                    'exp_val': c,
                    'local_val': 0.0,
                    'grad_contrib': 0.0,
                })
            # Case B: Constant base, variable exponent (e.g. 2**x)
            elif base_node.is_constant:
                c = float(base_node.expr)
                exp_sym = sp.Symbol(exp_node.name)
                sym_d = (c ** exp_sym) * sp.log(c)
                node.local_derivatives.append({
                    'arg_node': exp_node,
                    'sym_expr': sym_d,
                    'is_pow_constant_base': True,
                    'base_val': c,
                    'local_val': 0.0,
                    'grad_contrib': 0.0,
                })
            # Case C: Both base and exponent are variables (x**y)
            else:
                d_base = sp.diff(dummy_op, dummy_symbols[0])
                d_exp = sp.diff(dummy_op, dummy_symbols[1])
                node.local_derivatives.append({
                    'arg_node': base_node,
                    'sym_expr': d_base.subs(display_map),
                    'dummy_deriv': d_base,
                    'dummy_symbols': dummy_symbols,
                    'local_val': 0.0,
                    'grad_contrib': 0.0,
                })
                node.local_derivatives.append({
                    'arg_node': exp_node,
                    'sym_expr': d_exp.subs(display_map),
                    'dummy_deriv': d_exp,
                    'dummy_symbols': dummy_symbols,
                    'local_val': 0.0,
                    'grad_contrib': 0.0,
                })
        else:
            for i, child in enumerate(node.args):
                if child.is_constant:
                    continue
                d_dummy = sp.diff(dummy_op, dummy_symbols[i])
                sym_d = d_dummy.subs(display_map)
                node.local_derivatives.append({
                    'arg_node': child,
                    'sym_expr': sym_d,
                    'dummy_deriv': d_dummy,
                    'dummy_symbols': dummy_symbols,
                    'local_val': 0.0,
                    'grad_contrib': 0.0,
                })

        self.nodes.append(node)
        return node

    def evaluate(self, feed_dict, forward_only=False):
        # 1. Forward pass
        for node in self.nodes:
            if node.is_input:
                if node.expr in feed_dict:
                    node.val = float(feed_dict[node.expr])
                elif node.is_constant:
                    node.val = float(node.expr)
                else:
                    raise ValueError(f"Missing numerical value for variable '{node.name}'")
            else:
                if isinstance(node.expr, sp.Add):
                    node.val = float(sum(arg.val for arg in node.args))
                elif isinstance(node.expr, sp.Mul):
                    prod = 1.0
                    for arg in node.args:
                        prod *= arg.val
                    node.val = float(prod)
                elif isinstance(node.expr, sp.Pow):
                    base_val = node.args[0].val
                    exp_val = node.args[1].val
                    if base_val < 0 and abs(exp_val - round(exp_val)) < 1e-9:
                        node.val = float(base_val ** int(round(exp_val)))
                    else:
                        node.val = float(base_val ** exp_val)
                else:
                    dummy_syms = [sp.Symbol(f"_u{j}") for j in range(len(node.args))]
                    dummy_op = node.expr.func(*dummy_syms)
                    val_dict = {dummy_syms[j]: node.args[j].val for j in range(len(node.args))}
                    node.val = float(dummy_op.subs(val_dict).evalf())

        # If forward only mode, bypass local derivative evaluation and backpropagation
        if forward_only:
            return

        # 2. Local derivative values
        for node in self.nodes:
            if not node.is_input:
                for loc in node.local_derivatives:
                    if loc.get('is_pow_constant_exp', False):
                        base_val = node.args[0].val
                        c = loc['exp_val']
                        if base_val == 0.0 and c < 1:
                            loc['local_val'] = 0.0
                        elif base_val < 0 and abs((c - 1) - round(c - 1)) < 1e-9:
                            loc['local_val'] = float(c * (base_val ** int(round(c - 1))))
                        else:
                            loc['local_val'] = float(c * (base_val ** (c - 1)))
                    elif loc.get('is_pow_constant_base', False):
                        c = loc['base_val']
                        exp_val = node.args[1].val
                        loc['local_val'] = float((c ** exp_val) * math.log(c))
                    else:
                        d_dummy = loc['dummy_deriv']
                        dummy_syms = loc['dummy_symbols']
                        val_dict = {dummy_syms[j]: node.args[j].val for j in range(len(node.args))}
                        loc['local_val'] = float(d_dummy.subs(val_dict).evalf())

        # 3. Backward pass (Accumulate adjoints & record edge contributions)
        for node in self.nodes:
            node.grad = 0.0

        self.root.grad = 1.0

        for node in reversed(self.nodes):
            if not node.is_input:
                for loc in node.local_derivatives:
                    child = loc['arg_node']
                    contrib = node.grad * loc['local_val']
                    loc['grad_contrib'] = contrib
                    child.grad += contrib

    def to_mermaid(self, output_name="out", forward_only=False, grad_symbol=None):
        lines = ["graph BT", ""]

        # Expression string for the decorated output subgraph title
        formula_display = format_expr(self.orig_expr if self.orig_expr is not None else self.root.expr)
        out_uid = "out_target"
        out_label = f"v({output_name}) = {format_num(self.root.val)}<hr>{output_name}"

        # 1. Output Subgraph (Top)
        lines.append(f'    subgraph output ["Output: {output_name} = {formula_display}"]')
        lines.append(f'        {out_uid}(["{out_label}"])')
        lines.append('    end')
        lines.append('')

        # 2. Operations Subgraph (Middle)
        op_nodes = [n for n in self.nodes if not n.is_input]
        
        lines.append('    subgraph operation ["Operations"]')
        for node in reversed(op_nodes):
            # Pretty-print operation assignment
            func_name = node.expr.func.__name__
            if func_name == "Mul":
                op_str = " * ".join([arg.name for arg in node.args])
            elif func_name == "Add":
                op_str = " + ".join([arg.name for arg in node.args])
            elif func_name == "Pow":
                op_str = f"{node.args[0].name} ** {node.args[1].name}"
            else:
                op_str = format_expr(node.expr)

            node_label = f"v({node.name}) = {format_num(node.val)}<hr>{node.name} = {op_str}"
            lines.append(f'        {node.uid}(["{node_label}"])')

            if not forward_only:
                for i, loc in enumerate(node.local_derivatives):
                    arg = loc['arg_node']
                    der_uid = f"der_{node.uid}_{arg.uid}_{i}"
                    sym_str = format_expr(loc['sym_expr'])
                    diff_str = f"diff({op_str}, {arg.name}) = {sym_str}"
                    val_str = f"v({arg.name}) = {format_num(arg.val)}"
                    lines.append(f'        {der_uid}[\\"{diff_str}<hr>{val_str}"/]')
        lines.append('    end')
        lines.append('')

        # 3. Inputs Subgraph (Bottom)
        input_nodes = [n for n in self.nodes if n.is_input]
        var_inputs = [n for n in input_nodes if not n.is_constant]
        if var_inputs:
            inputs_summary = ", ".join([f"{n.name} = {format_num(n.val)}" for n in var_inputs])
            lines.append(f'    subgraph inputs ["Inputs: {inputs_summary}"]')
        else:
            lines.append('    subgraph inputs ["Inputs"]')
        for node in input_nodes:
            if forward_only or node.is_constant:
                inp_label = f"v({node.name}) = {format_num(node.val)}<hr>{node.name}"
            else:
                inp_label = f"v({node.name}) = {format_num(node.val)}<hr>grad({node.name}) = {format_num(node.grad)}"
            lines.append(f'        {node.uid}(["{inp_label}"])')
        lines.append('    end')
        lines.append('')

        # 4. Forward Computation Edges (Solid Green)
        lines.append('    %% Forward computation')
        forward_edges = []
        for node in op_nodes:
            for arg in node.args:
                forward_edges.append(
                    f'    {arg.uid} -- "&nbsp;&nbsp;f = {format_num(arg.val)}&nbsp;&nbsp;" --> {node.uid}'
                )
        # Connect root operation to designated output target
        forward_edges.append(
            f'    {self.root.uid} -- "&nbsp;&nbsp;f = {format_num(self.root.val)}&nbsp;&nbsp;" --> {out_uid}'
        )

        for e in forward_edges:
            lines.append(e)
        lines.append('')

        # 5. Backward Computation Edges (Dashed Magenta)
        backward_edges = []
        if not forward_only:
            lines.append('    %% Backward computation: written in child-to-parent form so Dagre preserves')
            lines.append('    %% bottom-to-top ranking without 2-cycles, while directing arrowheads downward')
            # Seed gradient: out_target down to root
            seed_lbl = f"g = {grad_symbol}" if grad_symbol else "g = 1"
            backward_edges.append(f'    {self.root.uid} <-.->|"&nbsp;&nbsp;{seed_lbl}&nbsp;&nbsp;"| {out_uid}')

            for node in reversed(op_nodes):
                for i, loc in enumerate(node.local_derivatives):
                    arg = loc['arg_node']
                    der_uid = f"der_{node.uid}_{arg.uid}_{i}"
                    g_contrib = format_num(loc['grad_contrib'])
                    # 1:1 discrete downward edge from operation to derivative block
                    backward_edges.append(f'    {der_uid} <-.-> {node.uid}')
                    # 1:1 discrete downward edge from derivative block to child argument with propagated gradient
                    backward_edges.append(
                        f'    {arg.uid} <-.->|"&nbsp;&nbsp;g = {g_contrib}&nbsp;&nbsp;"| {der_uid}'
                    )

            for e in backward_edges:
                lines.append(e)
            lines.append('')

        # 6. Styling & Precise LinkStyle Application
        lines.append('    %% Styling')
        lines.append('    classDef default fill:#f9f,stroke:#333,stroke-width:1px,color:#000;')
        lines.append('    classDef forward fill:#cfc,stroke:#333,stroke-width:1px,color:#000;')
        if not forward_only:
            lines.append('    classDef backward fill:#fcf,stroke:#333,stroke-width:1px,color:#000;')
        lines.append('    classDef inputBg fill:#e6f3ff,stroke:#333,stroke-width:1px,color:#000;')
        lines.append('    classDef operationBg fill:#fff2cc,stroke:#333,stroke-width:1px,color:#000;')
        lines.append('    classDef outputBg fill:#e6ffee,stroke:#333,stroke-width:1px,color:#000;')
        lines.append('')

        forward_nodes = [out_uid] + [n.uid for n in self.nodes]
        lines.append(f'    class {",".join(forward_nodes)} forward;')

        if not forward_only:
            der_nodes = []
            for node in op_nodes:
                for i, loc in enumerate(node.local_derivatives):
                    der_nodes.append(f"der_{node.uid}_{loc['arg_node'].uid}_{i}")
            if der_nodes:
                lines.append(f'    class {",".join(der_nodes)} backward;')

        lines.append('    class inputs inputBg;')
        lines.append('    class operation operationBg;')
        lines.append('    class output outputBg;')
        lines.append('')

        # Accurate 1:1 linkStyle indices
        lines.append('    %% Apply Styles to links')
        current_idx = 0

        # Forward links
        if forward_edges:
            fwd_indices = list(range(current_idx, current_idx + len(forward_edges)))
            lines.append(f'    linkStyle {",".join(map(str, fwd_indices))} stroke:#0a0,stroke-width:3.5px;')
            current_idx += len(forward_edges)

        # Backward links
        if backward_edges:
            bwd_indices = list(range(current_idx, current_idx + len(backward_edges)))
            lines.append(f'    linkStyle {",".join(map(str, bwd_indices))} stroke:#a0a,stroke-width:3px,stroke-dasharray: 3 3,marker-end:none;')
            current_idx += len(backward_edges)

        lines.append('    style operation fill:solid,stroke:1px,stroke-width:1px,font-size:15px,font-weight:bold;')

        return "\n".join(lines)
