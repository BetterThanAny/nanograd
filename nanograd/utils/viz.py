"""Computation graph visualization (Graphviz DOT format).

Walk the graph from a Tensor backward through its _ctx and emit a DOT string.
No external deps — renders via `dot -Tpng foo.dot -o foo.png` if graphviz is installed.
"""
from __future__ import annotations

import html
from typing import Optional

from nanograd.tensor import Tensor


def _shape_str(t: Tensor) -> str:
    return "×".join(str(s) for s in t.shape) if t.ndim else "scalar"


def build_graph(output: Tensor) -> tuple[set, set]:
    """Return (tensor_ids, edges). Edges are (src_id, dst_id, label)."""
    tensors: set = set()
    edges: set = set()
    visited: set = set()

    def visit(t: Tensor) -> None:
        if id(t) in visited:
            return
        visited.add(id(t))
        tensors.add(t)
        if t._ctx is not None:
            ctx_id = f"op_{id(t._ctx)}"
            edges.add((ctx_id, f"t_{id(t)}", "out"))
            for i, p in enumerate(t._ctx.parents):
                edges.add((f"t_{id(p)}", ctx_id, f"in{i}"))
                visit(p)

    visit(output)
    return tensors, edges


def to_dot(output: Tensor, title: Optional[str] = None) -> str:
    tensors, edges = build_graph(output)
    lines = ["digraph G {", '  rankdir=LR;', '  node [fontname="monospace", fontsize=10];']
    if title:
        lines.append(f'  label=<{html.escape(title)}>; labelloc="t"; fontname="monospace";')

    # tensor nodes
    ops_seen: set = set()
    for t in tensors:
        tid = f"t_{id(t)}"
        shape = _shape_str(t)
        color = "lightblue"
        if t.requires_grad:
            color = "lightgreen" if t._ctx is None else "lightyellow"
        label = f"shape: {shape}\\lrequires_grad: {t.requires_grad}"
        lines.append(f'  {tid} [shape=box, style=filled, fillcolor={color}, label="{label}"];')
        if t._ctx is not None:
            ctx = t._ctx
            cid = f"op_{id(ctx)}"
            if cid not in ops_seen:
                ops_seen.add(cid)
                lines.append(
                    f'  {cid} [shape=ellipse, style=filled, fillcolor=lightgray, '
                    f'label="{ctx.__class__.__name__}"];'
                )

    for src, dst, label in edges:
        lines.append(f'  {src} -> {dst} [label="{label}", fontsize=8];')

    lines.append("}")
    return "\n".join(lines)


def save_dot(output: Tensor, path: str, title: Optional[str] = None) -> None:
    with open(path, "w") as f:
        f.write(to_dot(output, title=title))
