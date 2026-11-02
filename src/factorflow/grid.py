"""Clock-time bar grids: a trading calendar × a bar frequency.

``BarGrid("XNYS", "1min")`` gives the regular-session minute bars of the NYSE calendar
(holidays and early closes from ``exchange_calendars``); ``BarGrid("24/7", "1min")`` is a
continuous UTC clock for crypto. Bars are half-open ``[start, end)`` and labeled by ``end``.
The last bar of a session is clipped to the session close.

``grace`` is how long after a bar's end the live engine waits before sealing it (late prints
for the bar are still accepted during the grace period; trades stamped ``>= end`` that
arrive during it are held for the next bar). Default: 2 s intraday, 0 for daily bars.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from functools import cache

import numpy as np

NS = 1_000_000_000
_UNITS = {"s": 1, "sec": 1, "min": 60, "m": 60, "h": 3600, "d": 86400}


def parse_freq(freq: str) -> int:
    m = re.fullmatch(r"(\d*)\s*([a-z]+)", freq.strip().lower())
    if not m or m.group(2) not in _UNITS:
        raise ValueError(f"bad bar frequency {freq!r} (use e.g. '1min', '5min', '1h', '1d')")
    return int(m.group(1) or 1) * _UNITS[m.group(2)] * NS


def to_ns(x: str | date | datetime | int | np.datetime64) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, np.datetime64):
        return int(x.astype("datetime64[ns]").astype(np.int64))
    if isinstance(x, str):
        x = datetime.fromisoformat(x)
    if isinstance(x, datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=UTC)
        return int(x.timestamp()) * NS + x.microsecond * 1000
    return int(datetime(x.year, x.month, x.day, tzinfo=UTC).timestamp()) * NS


@cache
def _calendar(name: str):
    import exchange_calendars as xcals

    return xcals.get_calendar(name)


class BarGrid:
    def __init__(self, calendar: str = "XNYS", freq: str = "1d", grace: float | None = None):
        self.calendar = calendar
        self.freq = freq
        self.step = parse_freq(freq)
        self.daily = self.step >= 86400 * NS
        self.continuous = calendar.upper() in ("24/7", "247", "ALWAYS_OPEN", "CRYPTO")
        g = grace if grace is not None else (0.0 if self.daily else 2.0)
        self.grace_ns = round(g * NS)
        if self.daily and self.step != 86400 * NS:
            raise ValueError("multi-day bars are not supported; use '1d'")

    def __repr__(self) -> str:
        return f"BarGrid({self.calendar!r}, {self.freq!r}, grace={self.grace_ns / NS}s)"

    # ------------------------------------------------------------------ sessions
    def _sessions(self, d0: date, d1: date) -> tuple[np.ndarray, np.ndarray]:
        """Session (open, close) ns for session dates in [d0, d1]."""
        if self.continuous:
            days = np.arange(np.datetime64(d0, "D"), np.datetime64(d1, "D") + np.timedelta64(1, "D"))
            opens = days.astype("datetime64[ns]").astype(np.int64)
            return opens, opens + 86400 * NS
        cal = _calendar(self.calendar)
        lo = max(d0, cal.first_session.date())
        hi = min(d1, cal.last_session.date())
        if lo > hi:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        sch = cal.schedule.loc[str(lo) : str(hi)]
        opens = _ns(sch["open"])
        closes = _ns(sch["close"])
        return opens, closes

    def bars(self, start, end) -> tuple[np.ndarray, np.ndarray]:
        """(starts, ends) of all bars in sessions dated ``start``..``end`` (inclusive)."""
        d0 = _as_date(start)
        d1 = _as_date(end)
        opens, closes = self._sessions(d0, d1)
        if self.daily:
            return opens.copy(), closes.copy()
        ss, es = [], []
        for o, c in zip(opens, closes, strict=True):
            e = np.arange(o + self.step, c + self.step, self.step, dtype=np.int64)
            e[-1] = min(e[-1], c)
            if e.size > 1 and e[-1] <= e[-2]:
                e = e[:-1]
            s = np.r_[o, e[:-1]]
            ss.append(s)
            es.append(e)
        if not es:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(ss), np.concatenate(es)

    def bars_after(self, after_end: int, count: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """The next ``count`` bars whose end is strictly after ``after_end`` (ns)."""
        d = _as_date(after_end) - timedelta(days=1)
        span = 7
        while True:
            s, e = self.bars(d, d + timedelta(days=span))
            k = np.searchsorted(e, after_end, side="right")
            if e.size - k >= count:
                return s[k : k + count], e[k : k + count]
            span *= 2
            if span > 4000:
                raise ValueError("calendar exhausted; extend the exchange_calendars range")

    def bar_containing(self, ts: int) -> tuple[int, int]:
        s, e = self.bars_after(ts, 1)
        return int(s[0]), int(e[0])


def _ns(col) -> np.ndarray:
    import pandas as pd

    return np.asarray(pd.DatetimeIndex(col.array).tz_convert("UTC").as_unit("ns").asi8, dtype=np.int64)


def _as_date(x) -> date:
    if isinstance(x, datetime):
        return x.astimezone(UTC).date() if x.tzinfo else x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return datetime.fromisoformat(x).date()
    return datetime.fromtimestamp(int(x) / NS, tz=UTC).date()
