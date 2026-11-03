"""Compile a pipeline's terms into a deduplicated, height-ordered graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from factorflow.terms import Constant, ConstFilter, Filter, Term, _Input


@dataclass(eq=False)
class Pipeline:
    columns: dict[str, Term]
    screen: Filter | None = None

    def __post_init__(self) -> None:
        for name, t in self.columns.items():
            if not isinstance(t, Term) or isinstance(t, (Constant, ConstFilter)):
                raise TypeError(f"pipeline column {name!r} must be a non-constant term")


@dataclass(eq=False)
class Graph:
    order: list[Term]  # topological, by height
    inputs: list[Term]  # input (leaf) terms
    outputs: dict[str, Term]
    screen: Term | None
    lookback: dict[Term, int] = field(default_factory=dict)  # rows of each node's output its consumers need
    heights: dict[Term, int] = field(default_factory=dict)
    consumers: dict[Term, list[Term]] = field(default_factory=dict)

    @property
    def input_names(self) -> list[str]:
        return [t.name for t in self.inputs]  # type: ignore[attr-defined]


def _canonical(t: Term, memo: dict[int, Term], table: dict[tuple, Term]) -> Term:
    """Common-subexpression elimination: structurally equal terms map to one node."""
    if id(t) in memo:
        return memo[id(t)]
    if isinstance(t, _Input):
        key: tuple = ("input", type(t).__name__, t.name)
    else:
        new_inputs = tuple(_canonical(i, memo, table) for i in t.inputs)
        if new_inputs != t.inputs and any(a is not b for a, b in zip(new_inputs, t.inputs, strict=True)):
            # rebuild with canonical inputs (terms are immutable, so copy and swap)
            clone = object.__new__(type(t))
            clone.__dict__.update(t.__dict__)
            clone.inputs = new_inputs
            if hasattr(t, "operands"):
                rep = dict(zip([id(x) for x in t.inputs], new_inputs, strict=True))
                clone.operands = tuple(rep.get(id(o), o) for o in t.operands)  # type: ignore[attr-defined]
            t = clone
        key = t._key()
    out = table.setdefault(key, t)
    memo[id(t)] = out
    return out


def compile_pipeline(p: Pipeline) -> Graph:
    memo: dict[int, Term] = {}
    table: dict[tuple, Term] = {}
    outputs = {k: _canonical(v, memo, table) for k, v in p.columns.items()}
    screen = _canonical(p.screen, memo, table) if p.screen is not None else None

    heights: dict[Term, int] = {}
    consumers: dict[Term, list[Term]] = {}
    order: list[Term] = []

    def visit(t: Term) -> int:
        if t in heights:
            return heights[t]
        h = 0
        for i in t.inputs:
            h = max(h, visit(i) + 1)
            consumers.setdefault(i, [])
            if t not in consumers[i]:
                consumers[i].append(t)
        heights[t] = h
        order.append(t)
        return h

    roots = list(outputs.values()) + ([screen] if screen is not None else [])
    for r in roots:
        visit(r)
    order.sort(key=lambda t: heights[t])
    lookback = {t: max([c.window for c in consumers.get(t, [])] + [1]) for t in order}
    inputs: list[Term] = [t for t in order if isinstance(t, _Input)]
    return Graph(order, inputs, outputs, screen, lookback, heights, consumers)
