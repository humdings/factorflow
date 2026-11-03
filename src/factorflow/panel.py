"""Panel: dense ``T × N`` arrays on a bar grid (rows are sealed bars, labeled by end time)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Panel:
    times: np.ndarray  # int64 ns since epoch (UTC), bar end; non-decreasing (event bars may share a stamp)
    assets: tuple[str, ...]
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    starts: np.ndarray | None = None  # optional bar start times

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.int64)
        self.assets = tuple(self.assets)
        t, n = len(self.times), len(self.assets)
        if t > 1 and not np.all(np.diff(self.times) >= 0):
            raise ValueError("panel times must be non-decreasing")
        for k, v in self.columns.items():
            if v.shape != (t, n):
                raise ValueError(f"column {k!r} has shape {v.shape}, expected {(t, n)}")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.times), len(self.assets)

    def head(self, rows: int) -> Panel:
        return Panel(
            self.times[:rows],
            self.assets,
            {k: v[:rows] for k, v in self.columns.items()},
            None if self.starts is None else self.starts[:rows],
        )

    def with_row(self, row: Mapping[str, np.ndarray], time: int) -> Panel:
        """A copy with one more row appended (used to treat an open bar as final)."""
        cols = {
            k: np.vstack([v, np.asarray(row[k], dtype=v.dtype)[None, :]]) for k, v in self.columns.items()
        }
        return Panel(np.r_[self.times, np.int64(time)], self.assets, cols)

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        assets: Sequence[str] | None = None,
        time_col: str = "ts",
        asset_col: str = "asset",
    ) -> Panel:
        """Long frame (ts, asset, columns...) -> Panel. Missing (ts, asset) cells become NaN."""
        ts = df[time_col]
        if ts.dtype != pl.Int64:
            ts = ts.cast(pl.Datetime("ns", "UTC")).cast(pl.Int64)
        df = df.with_columns(ts.alias("__t"))
        times = np.unique(df["__t"].to_numpy())
        names = tuple(assets) if assets is not None else tuple(sorted(df[asset_col].unique().to_list()))
        ti = np.searchsorted(times, df["__t"].to_numpy())
        amap = {a: i for i, a in enumerate(names)}
        ai = np.array([amap.get(a, -1) for a in df[asset_col].to_list()])
        keep = ai >= 0
        cols = {}
        for c in df.columns:
            if c in (time_col, asset_col, "__t"):
                continue
            arr = np.full((len(times), len(names)), np.nan)
            arr[ti[keep], ai[keep]] = df[c].to_numpy()[keep].astype(np.float64)
            cols[c] = arr
        return cls(times, names, cols)
