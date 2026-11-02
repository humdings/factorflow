"""Batch executor: run a compiled pipeline over a whole :class:`Panel` at once."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from factorflow.graph import Graph, Pipeline, compile_pipeline
from factorflow.panel import Panel
from factorflow.terms import Term, _Input


def _pad(x: np.ndarray, rows: int, missing) -> np.ndarray:
    if rows <= 0:
        return x
    pad = np.full((rows, *x.shape[1:]), missing, dtype=x.dtype)
    return np.concatenate([pad, x], axis=0)


def input_array(panel: Panel, t: Term) -> np.ndarray:
    name = t.name  # type: ignore[attr-defined]
    if name not in panel.columns:
        raise KeyError(f"panel has no column {name!r} (needed by the pipeline)")
    return np.ascontiguousarray(panel.columns[name], dtype=t.dtype)


def compute_node(
    t: Term, blocks: list[np.ndarray], prev: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """All rows of one node. Returns (values (T, N), last row for recursive state)."""
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
    times: np.ndarray
    assets: tuple[str, ...]
    values: dict[str, np.ndarray]
    screen: np.ndarray | None
    nodes: dict[Term, np.ndarray]
    recursive_state: dict[Term, np.ndarray | None]

    def frame(self, open_last: bool = False) -> pl.DataFrame:
        """Long frame (ts, asset, is_open, outputs...), screened. ``open_last`` flags the last row."""
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
    def run(self, pipeline: Pipeline | Graph, panel: Panel) -> BatchResult:
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
