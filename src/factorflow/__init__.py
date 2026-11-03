"""factorflow: point-in-time factor pipelines that run the same in batch and live.

Bars are rows; the newest row may be *open* (provisional) and is revised by ticks until it
is sealed. Batch and incremental execution give bit-identical results for sealed rows.
"""

from factorflow.bars import BarState, Trades, build_time_bars
from factorflow.batch import BatchEngine, BatchResult
from factorflow.events import (
    AdaptiveDaily,
    AsofSampler,
    EventBar,
    EventBarBuilder,
    EventBars,
    EventPipeline,
    Fixed,
    build_event_bars,
    sample_asof,
    sample_panel,
)
from factorflow.graph import Pipeline, compile_pipeline
from factorflow.grid import BarGrid, to_ns
from factorflow.live import LiveEngine, Snapshot
from factorflow.panel import Panel
from factorflow.terms import BarData, Classifier, Column, Factor, Filter, Term, where

__version__ = "0.0.1"

__all__ = [
    "AdaptiveDaily",
    "AsofSampler",
    "BarData",
    "BarGrid",
    "BarState",
    "BatchEngine",
    "BatchResult",
    "Classifier",
    "Column",
    "EventBar",
    "EventBarBuilder",
    "EventBars",
    "EventPipeline",
    "Factor",
    "Filter",
    "Fixed",
    "LiveEngine",
    "Panel",
    "Pipeline",
    "Snapshot",
    "Term",
    "Trades",
    "build_event_bars",
    "build_time_bars",
    "compile_pipeline",
    "sample_asof",
    "sample_panel",
    "to_ns",
    "where",
]
