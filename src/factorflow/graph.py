"""Compile a pipeline's terms into a deduplicated, height-ordered graph.

Compilation does three things the executors rely on:

* common-subexpression elimination, so ``close.rolling_mean(5)`` written twice
  is one node;
* a height (topological) order, so a node runs after its inputs;
* a lookback per node, the number of sealed rows a consumer needs before the
  open row. The live engine keeps that many rows in a ring buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from factorflow.terms import Constant, ConstFilter, Filter, Term, _Input


@dataclass(eq=False)
class Pipeline:
    """Named outputs, and an optional screen, over a term graph.

    Parameters
    ----------
    columns : dict of str to Term
        Output name to term. Names become columns of
        :meth:`~factorflow.batch.BatchResult.frame` and keys of
        :attr:`~factorflow.live.Snapshot.values`. A column must be a real
        term, not a Python number.
    screen : Filter or None, optional
        Rows that are ``False`` are dropped from frames. The arrays in
        ``values`` are still the full cross-section. Default is no screen.

    Raises
    ------
    TypeError
        If a column is not a :class:`~factorflow.terms.Term`, or is a constant.

    Examples
    --------
    >>> import factorflow as ff
    >>> close = ff.Column("close")
    >>> pipe = ff.Pipeline({"mom": close.rolling_mean(5)}, screen=close > 0)
    >>> pipe.columns["mom"].window
    5
    """

    columns: dict[str, Term]
    screen: Filter | None = None

    def __post_init__(self) -> None:
        """Reject constants and non-terms used as pipeline columns.

        Raises
        ------
        TypeError
            If any column is not a non-constant term.
        """
        for name, t in self.columns.items():
            if not isinstance(t, Term) or isinstance(t, (Constant, ConstFilter)):
                raise TypeError(f"pipeline column {name!r} must be a non-constant term")


@dataclass(eq=False)
class Graph:
    """A compiled pipeline.

    Parameters
    ----------
    order : list of Term
        Nodes in increasing height, inputs first.
    inputs : list of Term
        Leaf terms. Each has a ``name`` attribute, the panel column it reads.
    outputs : dict of str to Term
        The pipeline's named columns, after common-subexpression elimination.
    screen : Term or None
        The screen, or ``None``.
    lookback : dict of Term to int
        How many rows of this node's output its consumers need. At least 1.
        The live ring buffer keeps ``lookback - 1`` sealed rows plus the open slot.
    heights : dict of Term to int
        ``1 + max(input heights)``, and 0 for inputs.
    consumers : dict of Term to list of Term
        Nodes that read this node.
    """

    order: list[Term]  # topological, by height
    inputs: list[Term]  # input (leaf) terms
    outputs: dict[str, Term]
    screen: Term | None
    lookback: dict[Term, int] = field(default_factory=dict)  # rows of each node's output its consumers need
    heights: dict[Term, int] = field(default_factory=dict)
    consumers: dict[Term, list[Term]] = field(default_factory=dict)

    @property
    def input_names(self) -> list[str]:
        """Panel columns this graph reads, in execution order.

        Returns
        -------
        list of str
            Names of the input terms.
        """
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
    """Turn a pipeline into a :class:`Graph`.

    Parameters
    ----------
    p : Pipeline
        The expression to compile. Compiling twice is safe; nodes are immutable.

    Returns
    -------
    Graph
        Deduplicated nodes, topological order, and per-node lookback.

    Notes
    -----
    Two terms are the same node when their class, parameters, window, and
    input identities match. Inputs are keyed by type and column name, so
    ``Column("close")`` is one node no matter how many times it is written.
    """
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
