"""Synthetic market data with known ground truth, a replay driver and validation tools.

    from factorflow.sim import SimConfig, simulate
    mkt = simulate(SimConfig(seed=7))          # equities + BTC/ETH/SOL, deterministic
    mkt.equities.daily                         # daily Panel (signal, industry, styles columns)
    mkt.equities.trades                        # ticks of the last ``intraday_days`` sessions

CLI: ``python -m factorflow.sim demo | validate | bench``.
"""

from factorflow.sim.market import (
    STYLES,
    AssetSim,
    CryptoSpec,
    EquitySpec,
    SimConfig,
    SimMarket,
    Truth,
    simulate,
)

__all__ = ["STYLES", "AssetSim", "CryptoSpec", "EquitySpec", "SimConfig", "SimMarket", "Truth", "simulate"]
