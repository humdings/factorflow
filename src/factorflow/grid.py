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
    """Parse a bar size into nanoseconds.

    Parameters
    ----------
    freq : str
        A count and a unit, for example ``"1min"``, ``"5min"``, ``"30s"``,
        ``"1h"``, ``"1d"``. The count defaults to 1. Units are ``s``/``sec``,
        ``min``/``m``, ``h``, and ``d``.

    Returns
    -------
    int
        Length of one bar in nanoseconds.

    Raises
    ------
    ValueError
        If ``freq`` is not of that form. Multi-day bars are rejected later by
        :class:`BarGrid`, not here.
    """
    m = re.fullmatch(r"(\d*)\s*([a-z]+)", freq.strip().lower())
    if not m or m.group(2) not in _UNITS:
        raise ValueError(f"bad bar frequency {freq!r} (use e.g. '1min', '5min', '1h', '1d')")
    return int(m.group(1) or 1) * _UNITS[m.group(2)] * NS


def to_ns(x: str | date | datetime | int | np.datetime64) -> int:
    """Convert a timestamp to nanoseconds since the Unix epoch, UTC.

    Parameters
    ----------
    x : str, datetime.date, datetime.datetime, int, or numpy.datetime64
        An int is already nanoseconds and is returned unchanged. A naive
        datetime is treated as UTC. A date is midnight UTC. A string is
        parsed with :meth:`datetime.datetime.fromisoformat` (a trailing ``Z``
        is not accepted; use ``+00:00``).

    Returns
    -------
    int
        Nanoseconds UTC.

    Examples
    --------
    >>> to_ns("2026-11-30T14:30:00+00:00")
    1796049000000000000
    """
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
    """A calendar crossed with a bar frequency.

    Parameters
    ----------
    calendar : str, optional
        An :mod:`exchange_calendars` name, such as ``"XNYS"``, or ``"24/7"``
        (also ``"247"``, ``"ALWAYS_OPEN"``, ``"CRYPTO"``) for a clock that
        does not close. Default is ``"XNYS"``.
    freq : str, optional
        Bar size, parsed by :func:`parse_freq`. Default is ``"1d"``. Only
        ``"1d"`` is allowed for daily bars; ``"2d"`` and longer are rejected.
    grace : float or None, optional
        Seconds after a bar's end before the live engine seals it. ``None``
        means 0 for daily bars and 2 for anything finer. Late prints inside
        the grace are still applied to the bar. A trade stamped at or after
        ``end`` is held for the next bar.

    Raises
    ------
    ValueError
        If ``freq`` is a multi-day size.

    Examples
    --------
    A full NYSE session is 390 one-minute bars. The day after Thanksgiving
    2026 closes at 13:00 America/New_York and has 210::

        grid = BarGrid("XNYS", "1min")
        len(grid.bars("2026-11-30", "2026-11-30")[1])  # 390
        len(grid.bars("2026-11-27", "2026-11-27")[1])  # 210
    """

    def __init__(self, calendar: str = "XNYS", freq: str = "1d", grace: float | None = None):
        """Store the calendar, the bar length, and the seal grace.

        Parameters
        ----------
        calendar : str, optional
            Exchange calendar name, or a 24/7 alias. Default ``"XNYS"``.
        freq : str, optional
            Bar size. Default ``"1d"``.
        grace : float or None, optional
            Seal delay in seconds. ``None`` selects the default for ``freq``.

        Raises
        ------
        ValueError
            If ``freq`` spans more than one day.
        """
        self.calendar = calendar
        self.freq = freq
        self.step = parse_freq(freq)
        self.daily = self.step >= 86400 * NS
        self.continuous = calendar.upper() in ("24/7", "247", "ALWAYS_OPEN", "CRYPTO")
        g = grace if grace is not None else (0.0 if self.daily else 2.0)
        self.grace_ns = round(g * NS)
        if self.daily and self.step != 86400 * NS:
            raise ValueError("multi-day bars are not supported; use '1d'")
        self._cache_s: np.ndarray | None = None
        self._cache_e: np.ndarray | None = None

    def __repr__(self) -> str:
        """Return ``BarGrid(calendar, freq, grace=...s)``."""
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
        """List every bar whose session date is between ``start`` and ``end``.

        Parameters
        ----------
        start, end
            Inclusive session dates. Anything :func:`to_ns` accepts, or a
            ``datetime.date``. The calendar's own first and last sessions
            clip the range.

        Returns
        -------
        starts, ends : ndarray of int64
            Bar start and end, nanoseconds UTC, half-open ``[start, end)``.
            Daily bars use the session open and the official close, so an
            early close is a shorter bar rather than a missing one. Intraday
            bars step by :attr:`step`; the last bar of a session is clipped
            to the close and dropped if that would make it empty.

        Notes
        -----
        A holiday inside the range contributes no bars. ``"24/7"`` has no
        holidays: every UTC date from ``start`` through ``end`` is included.
        """
        d0 = _as_date(start)
        d1 = _as_date(end)
        opens, closes = self._sessions(d0, d1)
        if self.daily:
            return opens.copy(), closes.copy()
        ss, es = [], []
        for o, c in zip(opens, closes, strict=True):
            e = np.arange(o + self.step, c + self.step, self.step, dtype=np.int64)
            # Clip the last bar to the official close. An early close therefore
            # shortens the day. Drop the bar if the clip leaves it empty.
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
        """Return the next bars after a sealed bar.

        Parameters
        ----------
        after_end : int
            A bar end, nanoseconds UTC. The returned bars have ends strictly
            greater than this.
        count : int, optional
            How many bars. Default is 1. The live engine asks for one bar
            on every seal.

        Returns
        -------
        starts, ends : ndarray of int64
            The next ``count`` bars.

        Raises
        ------
        ValueError
            If the calendar runs out before ``count`` bars exist (more than
            about 4000 days ahead).

        Notes
        -----
        A contiguous block of about a month is cached. Repeated seals inside
        that block do not rebuild the exchange schedule.
        """
        ce = self._cache_e
        if ce is not None and ce.size and ce[0] <= after_end:
            k = int(np.searchsorted(ce, after_end, side="right"))
            if ce.size - k >= count:
                return self._cache_s[k : k + count].copy(), ce[k : k + count].copy()  # type: ignore[index]
        d = _as_date(after_end) - timedelta(days=1)
        span = 35
        while True:
            s, e = self.bars(d, d + timedelta(days=span))
            k = int(np.searchsorted(e, after_end, side="right"))
            if e.size - k >= count:
                self._cache_s, self._cache_e = s, e
                return s[k : k + count].copy(), e[k : k + count].copy()
            span *= 2
            if span > 4000:
                raise ValueError("calendar exhausted; extend the exchange_calendars range")

    def bar_containing(self, ts: int) -> tuple[int, int]:
        """Return the bar that contains ``ts``.

        Parameters
        ----------
        ts : int
            Nanoseconds UTC.

        Returns
        -------
        start, end : int
            The bar with ``start <= ts < end`` if ``ts`` falls in a session.
            Implemented as the first bar whose end is strictly after ``ts``,
            which is the ``[start, end)`` rule. A timestamp in a closed
            session still resolves to the next open bar.
        """
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
