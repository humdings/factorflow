"""Batch executor: run a compiled pipeline over a whole :class:`~factorflow.panel.Panel`.

Batch and live share kernels. For a windowed term, this module prepends
``window - 1`` missing rows (the warmup) and calls ``compute`` once on the whole
history. The live engine instead calls the same ``compute`` on exactly ``window``
rows and keeps row 0 of the result. The arithmetic is the same; only the slice
of history differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from factorflow.graph import Graph, Pipeline, compile_pipeline
from factorflow.panel import Panel
from factorflow.terms import Term, _Input


def _pad(x: np.ndarray, rows: int, missing) -> np.ndarray:
    """Prepend ``rows`` copies of ``missing`` so a window of length ``w`` can start at row 0.

    The first ``w - 1`` output rows then see a window that contains the missing
    value and come out missing. That is the warmup. ``rows`` is 0 for a term
    whose window is 1.
    """
    if rows <= 0:
        return x
    pad = np.full((rows, *x.shape[1:]), missing, dtype=x.dtype)
    return np.concatenate([pad, x], axis=0)


def input_array(panel: Panel, t: Term) -> np.ndarray:
    """Read one input term out of a panel.

    Parameters
    ----------
    panel : Panel
        History the pipeline is running on.
    t : Term
        An input term (a :class:`~factorflow.terms.Column` node). ``t.name``
        is the column name.

    Returns
    -------
    ndarray of shape (T, N)
        Contiguous array in the term's dtype.

    Raises
    ------
    KeyError
        If the panel has no column of that name.
    """
    name = t.name  # type: ignore[attr-defined]
    if name not in panel.columns:
        raise KeyError(f"panel has no column {name!r} (needed by the pipeline)")
    return np.ascontiguousarray(panel.columns[name], dtype=t.dtype)


def compute_node(
    t: Term, blocks: list[np.ndarray], prev: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate one term on a whole history.

    Parameters
    ----------
    t : Term
        The node to run. Not an input term.
    blocks : list of ndarray
        One ``(T, N)`` array per input, already aligned to the panel.
    prev : ndarray of shape (N,) or None
        Previous output for a recursive term. Batch starts with ``None``,
        so an EWMA starts at the first finite observation.

    Returns
    -------
    values : ndarray of shape (T, N)
        One output row per panel row.
    last : ndarray of shape (N,) or None
        The last row, kept as recursive state. ``None`` if the term is not
        recursive or the history is empty.

    Notes
    -----
    Row-wise terms (cross-section, EWMA) loop one row at a time, which is the
    same loop the live engine runs for a single open row. Windowed terms are
    padded by ``window - 1`` missing rows and then handed to ``compute``.
    """
    n_rows = blocks[0].shape[0]
    if t.rowwise:
        out = np.empty(blocks[0].shape, dtype=t.dtype)
        p = prev
        for r in range(n_rows):
            row = t.compute_row([b[r : r + 1] for b in blocks], p)
            out[r] = row
            p = row if t.recursive else None
        return out, (out[-1].copy() if (t.recursive and n_rows) else prev)
    w = t.window
    padded = [_pad(b, w - 1, inp.missing) for b, inp in zip(blocks, t.inputs, strict=True)]
    return np.asarray(t.compute(padded), dtype=t.dtype), None


@dataclass
class BatchResult:
    """Everything :meth:`BatchEngine.run` computed.

    Parameters
    ----------
    times : ndarray of int64
        Bar ends, nanoseconds UTC. Same as the panel.
    assets : tuple of str
        Asset names, column order.
    values : dict of str to ndarray
        Named pipeline outputs, each ``(T, N)``. These are the full arrays,
        including rows a screen would drop and including warmup NaNs.
    screen : ndarray of bool, shape (T, N), or None
        The pipeline screen, or ``None`` if the pipeline has no screen.
    nodes : dict of Term to ndarray
        Every node's ``(T, N)`` values, including inputs. The live engine
        warms its ring buffers from this.
    recursive_state : dict of Term to ndarray or None
        Last sealed row of each recursive term.

    Examples
    --------
    >>> import numpy as np
    >>> import factorflow as ff
    >>> close = np.arange(6, dtype=float).reshape(3, 2)
    >>> panel = ff.Panel(np.arange(3), ("A", "B"), {"close": close})
    >>> out = ff.BatchEngine().run(ff.Pipeline({"lag": ff.Column("close").shift(1)}), panel)
    >>> out.values["lag"][2].tolist()
    [2.0, 3.0]
    """

    times: np.ndarray
    assets: tuple[str, ...]
    values: dict[str, np.ndarray]
    screen: np.ndarray | None
    nodes: dict[Term, np.ndarray]
    recursive_state: dict[Term, np.ndarray | None]

    def frame(self, open_last: bool = False) -> pl.DataFrame:
        """Stack the screened outputs into a long frame.

        Parameters
        ----------
        open_last : bool, optional
            If ``True``, the last bar is flagged ``is_open``. Use that when the
            last row was an open bar treated as final. Default is ``False``
            (every row sealed).

        Returns
        -------
        polars.DataFrame
            Columns ``ts`` (UTC datetime, ns), ``asset``, ``is_open``, then
            each output. Rows that fail the screen are omitted. With no screen,
            every asset and bar is present.
        """
        t, n = len(self.times), len(self.assets)
        keep = self.screen if self.screen is not None else np.ones((t, n), dtype=bool)
        ti, ai = np.nonzero(keep)
        is_open = np.zeros(ti.shape, dtype=bool)
        if open_last and t:
            is_open = ti == t - 1
        data = {
            "ts": pl.Series(self.times[ti]).cast(pl.Datetime("ns", "UTC")),
            "asset": [self.assets[i] for i in ai],
            "is_open": is_open,
        }
        for k, v in self.values.items():
            data[k] = v[ti, ai]
        return pl.DataFrame(data)


class BatchEngine:
    """Run a pipeline on a whole panel in one pass.

    There is no constructor state. :meth:`run` compiles the pipeline if needed
    and evaluates nodes in height order.
    """

    def run(self, pipeline: Pipeline | Graph, panel: Panel) -> BatchResult:
        """Evaluate ``pipeline`` on every row of ``panel``.

        Parameters
        ----------
        pipeline : Pipeline or Graph
            A :class:`~factorflow.graph.Pipeline`, or the graph from
            :func:`~factorflow.graph.compile_pipeline`.
        panel : Panel
            Must contain every input column the pipeline reads.

        Returns
        -------
        BatchResult
            Full ``(T, N)`` outputs. The first ``window - 1`` rows of a
            windowed factor are missing.

        Raises
        ------
        KeyError
            If a required column is absent.
        TypeError
            If a pipeline column is a bare constant. Raised while compiling.
        """
        g = pipeline if isinstance(pipeline, Graph) else compile_pipeline(pipeline)
        vals: dict[Term, np.ndarray] = {}
        rec: dict[Term, np.ndarray | None] = {}
        for t in g.order:
            if isinstance(t, _Input):
                vals[t] = input_array(panel, t)
                continue
            out, last = compute_node(t, [vals[i] for i in t.inputs], None)
            vals[t] = out
            if t.recursive:
                rec[t] = last
        outs = {k: vals[v] for k, v in g.outputs.items()}
        screen = vals[g.screen] if g.screen is not None else None
        return BatchResult(panel.times, panel.assets, outs, screen, vals, rec)
