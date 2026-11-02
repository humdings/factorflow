import numpy as np

import factorflow as ff


def random_trades(
    rng: np.random.Generator, n_assets: int, t0: int, t1: int, k: int, sparse_assets: int = 1
) -> ff.Trades:
    """Random trades in [t0, t1). The last ``sparse_assets`` assets trade rarely (gaps / ffill)."""
    ts = np.sort(rng.integers(t0, t1, k))
    p = (
        np.r_[np.ones(n_assets - sparse_assets), 0.02 * np.ones(sparse_assets)]
        if n_assets > sparse_assets
        else np.ones(n_assets)
    )
    asset = rng.choice(n_assets, size=k, p=p / p.sum())
    price = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.002, k))), 2)
    size = rng.integers(1, 500, k).astype(float)
    return ff.Trades(asset, ts, price, size)


def sectors(n: int) -> np.ndarray:
    return (np.arange(n) % 3).astype(np.int64)
