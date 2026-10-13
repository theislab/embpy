"""Cross-cell name checker for generated notebook parts.

Module-level bindings only: a name bound inside a function body is not
available to a later cell. Reports names read in a cell that no earlier cell
bound at module level, and names bound in the same cell are credited.
"""
import ast, json, sys

BUILTINS = set(dir(__builtins__)) | {"__name__", "__file__", "__doc__"}


def bindings(node):
    """Names this statement binds at the scope it appears in."""
    out = set()
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            out.add((a.asname or a.name).split(".")[0])
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        out |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                out |= {n.id for n in ast.walk(item.optional_vars)
                        if isinstance(n, ast.Name)}
    elif isinstance(node, ast.ExceptHandler) and node.name:
        out.add(node.name)
    return out


def walk_scope(stmts):
    """All bindings in a statement list, without descending into nested scopes."""
    out = set()
    for node in stmts:
        out |= bindings(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue                                  # its body is another scope
        for field in ("body", "orelse", "finalbody"):
            out |= walk_scope(getattr(node, field, []) or [])
        for h in getattr(node, "handlers", []):
            out |= bindings(h) | walk_scope(h.body)
    return out


def free_vars(fn):
    """Names a function reads that it does not itself bind."""
    local = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    local |= {a.arg for a in getattr(fn.args, "posonlyargs", [])}
    if fn.args.kwarg:
        local.add(fn.args.kwarg.arg)
    if fn.args.vararg:
        local.add(fn.args.vararg.arg)
    local |= walk_scope(fn.body)
    reads, nested = set(), []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reads.add(node.id)
        elif isinstance(node, ast.comprehension):
            local |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            nested.append(node)
    for inner in nested:
        local.add(inner.name)
    return reads - local


def cell_reads(tree):
    """Top-level reads plus free variables of any function defined here."""
    reads = set()
    top_bound = walk_scope(tree.body)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            reads |= free_vars(node)
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                reads.add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                pass
            elif isinstance(sub, ast.comprehension):
                top_bound |= {n.id for n in ast.walk(sub.target)
                              if isinstance(n, ast.Name)}
    return reads - top_bound


defined = set(BUILTINS)
issues = 0
for path in sys.argv[1:]:
    for i, (kind, src) in enumerate(json.load(open(path))):
        if kind != "code":
            continue
        tree = ast.parse(src)
        here = walk_scope(tree.body)
        missing = {n for n in cell_reads(tree) - defined - here
                   if not n.startswith("_")}
        if missing:
            issues += 1
            print(f"  [{path.split('/')[-1]} cell {i}] not bound at module level: "
                  f"{sorted(missing)}")
        defined |= here
print("scope-aware contract:", "clean" if not issues else f"{issues} issue(s)")
