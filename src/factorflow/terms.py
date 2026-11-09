"""Terms: the immutable, hashable nodes of a factor graph.

A term declares its ``inputs``, a ``window`` (how many rows of each input one output row
needs) and how to compute. Three kinds, as in Zipline's Pipeline:

* :class:`Factor`, float64 (NaN = missing)
* :class:`Filter`, bool (False = excluded)
* :class:`Classifier`, int64 group codes (-1 = missing)

Every term implements ``compute(blocks, prev)``: ``blocks[i]`` holds ``T_in`` rows of input
``i`` and the result has ``T_in - window + 1`` rows. Batch calls it on the whole history,
live calls it on exactly ``window`` rows. Row-wise terms (cross-sectional, recursive) are
looped one row at a time in both executors, so both run identical code on identical data.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from factorflow import kernels as K

Scalar = int | float | bool


class Term:
    """A node in a factor graph.

    Terms are immutable. Two terms are the same computation when their class,
    ``params``, ``window``, and input identities match; the compiler uses that
    as the common-subexpression key. ``==`` is identity, not structural
    equality, so ``close == close`` is not a filter. Value comparisons are
    methods (:meth:`Factor.eq`).

    Subclasses set the class attributes and implement :meth:`compute` or
    :meth:`compute_row`.

    Attributes
    ----------
    kind : str
        ``"factor"``, ``"filter"``, ``"classifier"``, or ``"term"``.
    dtype : type
        Numpy dtype of the output. float64, bool, or int64.
    missing
        Fill used where a value does not exist yet. NaN, ``False``, or ``-1``.
    window : int
        How many input rows one output row needs, including the current row.
        A 5-bar rolling mean has window 5. A 1-bar lag has window 2, because
        the kernel sees the lagged row and the current row.
    columnwise : bool
        ``True`` when output column ``j`` depends only on input column ``j``.
        The live engine may then recompute dirty columns alone.
    rowwise : bool
        ``True`` when both executors loop one row at a time (cross-section
        and EWMA). Those terms implement :meth:`compute_row`.
    recursive : bool
        ``True`` when the term needs its own previous output (EWMA).
    inputs : tuple of Term
        Child nodes. Empty for leaves.
    params : tuple
        Extra identity: window kind, half-life, comparison symbol, constants.
        Must be hashable.
    """

    kind: ClassVar[str] = "term"
    dtype: ClassVar[type] = np.float64
    missing: ClassVar[Any] = np.nan
    window: int = 1
    #: output column j depends only on input column j (so dirty columns can be recomputed alone)
    columnwise: bool = True
    #: computed one row at a time (cross-sectional or recursive) in both executors
    rowwise: bool = False
    #: needs its own previous sealed output (EWMA)
    recursive: bool = False

    inputs: tuple[Term, ...] = ()
    params: tuple = ()

    # -- identity -------------------------------------------------------------------
    def _key(self) -> tuple:
        return (type(self).__name__, self.params, self.window, tuple(id(i) for i in self.inputs))

    def __hash__(self) -> int:
        """Hash the structural key (class, params, window, input identities)."""
        return hash(self._key())

    def __eq__(self, other: object) -> bool:  # identity semantics; use .eq() for comparisons
        """Return whether ``other`` is this same node.

        Parameters
        ----------
        other : object
            Compared with ``is``, not by value.

        Returns
        -------
        bool
            ``True`` only when ``other`` is this object. Use :meth:`Factor.eq`
            to compare numeric values.
        """
        return self is other

    def __repr__(self) -> str:
        """Return ``ClassName(inputs, params)`` for debugging."""
        args = ", ".join([repr(i) for i in self.inputs] + [repr(p) for p in self.params])
        return f"{type(self).__name__}({args})"

    def compute(self, blocks: Sequence[np.ndarray], prev: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the term on a block of rows.

        Parameters
        ----------
        blocks : sequence of ndarray
            ``blocks[i]`` has shape ``(T_in, N)`` and is input ``i``. For a
            windowed term, ``T_in`` is the padded history in batch and exactly
            ``window`` in live.
        prev : ndarray of shape (N,) or None, optional
            Previous output of a recursive term. Ignored by everyone else.

        Returns
        -------
        ndarray
            Shape ``(T_in - window + 1, N)`` for windowed terms, or ``(T_in, N)``
            when ``window`` is 1. Live keeps row 0 of that array, which is the
            only row when ``T_in == window``.

        Raises
        ------
        NotImplementedError
            The base class does not compute.
        """
        raise NotImplementedError

    def compute_row(self, rows: Sequence[np.ndarray], prev: np.ndarray | None) -> np.ndarray:
        """Evaluate one row of a row-wise term.

        Parameters
        ----------
        rows : sequence of ndarray
            ``rows[i]`` has shape ``(1, N)`` or ``(window, N)``. Row-wise
            terms in this package read the last row.
        prev : ndarray of shape (N,) or None
            Previous sealed output for a recursive term, or ``None`` on the
            first row.

        Returns
        -------
        ndarray of shape (N,)
            The output row.

        Raises
        ------
        NotImplementedError
            The base class does not compute.
        """
        raise NotImplementedError

    def nonnull(self) -> np.ndarray:  # pragma: no cover - helper for subclasses
        """Return a mask of values that are present.

        Returns
        -------
        ndarray of bool

        Raises
        ------
        NotImplementedError
            Subclasses that need this override it. The engines do not call it.
        """
        raise NotImplementedError


def _lift(x: Term | Scalar) -> Term:
    if isinstance(x, Term):
        return x
    return Constant(x)


# ====================================================================== kinds
class Factor(Term):
    """A floating-point expression. Missing values are NaN.

    Notes
    -----
    Arithmetic and comparisons build new terms; they do not look at data.
    ``close + 1`` is a term you can put in a :class:`~factorflow.graph.Pipeline`.
    Comparisons return a :class:`Filter`. ``==`` on two terms is identity
    (``is``); use :meth:`eq` to compare values.

    Examples
    --------
    >>> import factorflow as ff
    >>> close = ff.Column('close')
    >>> (close.pct_change(1) > 0).kind
    'filter'
    """

    kind = "factor"
    dtype = np.float64
    missing = np.nan

    # arithmetic
    def __add__(self, o):
        """Add a factor or a scalar, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self + o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.add, "+", self, _lift(o))

    def __radd__(self, o):
        """Add a scalar on the left, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self + o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.add, "+", _lift(o), self)

    def __sub__(self, o):
        """Subtract a factor or a scalar, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self - o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.sub, "-", self, _lift(o))

    def __rsub__(self, o):
        """Subtract this factor from a scalar, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self - o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.sub, "-", _lift(o), self)

    def __mul__(self, o):
        """Multiply by a factor or a scalar, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self * o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.mul, "*", self, _lift(o))

    def __rmul__(self, o):
        """Multiply a scalar on the left, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self * o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.mul, "*", _lift(o), self)

    def __truediv__(self, o):
        """Divide by a factor or a scalar, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self / o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.truediv, "/", self, _lift(o))

    def __rtruediv__(self, o):
        """Divide a scalar by this factor, elementwise.

        Parameters
        ----------
        o : Factor or int or float or bool
            The other operand. A number is broadcast to every asset and bar.

        Returns
        -------
        Factor
            ``self / o`` when this factor is on the left, otherwise the reflected form. NaN propagates.
        """
        return BinOp(operator.truediv, "/", _lift(o), self)

    def __neg__(self):
        """Negate every value.

        Returns
        -------
        Factor
            ``-self``. NaN stays NaN.
        """
        return Unary(np.negative, "neg", self)

    def __abs__(self):
        """Take the absolute value.

        Returns
        -------
        Factor
            ``abs(self)``.
        """
        return Unary(np.abs, "abs", self)

    # comparisons -> Filter
    def __lt__(self, o):
        """Test strictly less than.

        Parameters
        ----------
        o : Factor or int or float or bool
            The value to compare against. A number is broadcast.

        Returns
        -------
        Filter
            ``self < o``. A comparison with NaN is False.
        """
        return Compare(operator.lt, "<", self, _lift(o))

    def __le__(self, o):
        """Test less than or equal.

        Parameters
        ----------
        o : Factor or int or float or bool
            The value to compare against. A number is broadcast.

        Returns
        -------
        Filter
            ``self <= o``. A comparison with NaN is False.
        """
        return Compare(operator.le, "<=", self, _lift(o))

    def __gt__(self, o):
        """Test strictly greater than.

        Parameters
        ----------
        o : Factor or int or float or bool
            The value to compare against. A number is broadcast.

        Returns
        -------
        Filter
            ``self > o``. A comparison with NaN is False.
        """
        return Compare(operator.gt, ">", self, _lift(o))

    def __ge__(self, o):
        """Test greater than or equal.

        Parameters
        ----------
        o : Factor or int or float or bool
            The value to compare against. A number is broadcast.

        Returns
        -------
        Filter
            ``self >= o``. A comparison with NaN is False.
        """
        return Compare(operator.ge, ">=", self, _lift(o))

    def eq(self, o):
        """Test equality of values.

        Parameters
        ----------
        o : Factor or int or float or bool
            The value to compare against. A number is broadcast.

        Returns
        -------
        Filter
            ``self == o`` as a filter. This is not Python ``==``, which tests node identity.
        """
        return Compare(operator.eq, "==", self, _lift(o))

    def ne(self, o):
        """Test inequality of values.

        Parameters
        ----------
        o : Factor or int or float or bool
            The value to compare against. A number is broadcast.

        Returns
        -------
        Filter
            ``self != o`` as a filter. A comparison with NaN is False.
        """
        return Compare(operator.ne, "!=", self, _lift(o))

    __hash__ = Term.__hash__

    def log(self) -> Factor:
        """Take the natural logarithm.

        Returns
        -------
        Factor
            ``log(self)``. Non-positive inputs are NaN. Computed on full rows so live and batch match.
        """
        return Unary(np.log, "log", self, columnwise=False)

    def sqrt(self) -> Factor:
        """Take the square root.

        Returns
        -------
        Factor
            ``sqrt(self)``. Negative inputs are NaN.
        """
        return Unary(np.sqrt, "sqrt", self)

    def isnull(self) -> Filter:
        """Test for missing values.

        Returns
        -------
        Filter
            True where this factor is NaN.
        """
        return IsNull(self)

    def notnull(self) -> Filter:
        """Test for present values.

        Returns
        -------
        Filter
            True where this factor is not NaN.
        """
        return ~IsNull(self)

    def fillna(self, value: float) -> Factor:
        """Replace missing values.

        Parameters
        ----------
        value : float
            Replacement, in the same units as the factor. Applied where the value is NaN.

        Returns
        -------
        Factor
            ``self`` with NaNs replaced by ``value``.
        """
        return FillNa(self, float(value))

    # time series
    def shift(self, n: int = 1) -> Factor:
        """Lag by ``n`` bars.

        Parameters
        ----------
        n : int, optional
            Bars to look back. Default is 1. The first ``n`` rows are NaN.

        Returns
        -------
        Factor
            ``self`` from ``n`` bars earlier.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        return Shift(self, n)

    def delta(self, n: int = 1) -> Factor:
        """Subtract the value from ``n`` bars ago.

        Parameters
        ----------
        n : int, optional
            Bars to look back. Default is 1.

        Returns
        -------
        Factor
            ``self - self.shift(n)``.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        return self - self.shift(n)

    def pct_change(self, n: int = 1) -> Factor:
        """Return the fractional change over ``n`` bars.

        Parameters
        ----------
        n : int, optional
            Bars to look back. Default is 1.

        Returns
        -------
        Factor
            ``self / self.shift(n) - 1``. Dimensionless. NaN when the lagged value is 0 or missing.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        return self / self.shift(n) - 1.0

    def rolling_sum(self, w: int) -> Factor:
        """Return the rolling sum.

        Parameters
        ----------
        w : int
            Window length in bars (rows), not calendar time. A window that contains a NaN is NaN.

        Returns
        -------
        Factor
            the rolling sum.

        Raises
        ------
        ValueError
            If ``w < 1``.
        """
        return Rolling(self, w, "sum")

    def rolling_mean(self, w: int) -> Factor:
        """Return the rolling mean.

        Parameters
        ----------
        w : int
            Window length in bars (rows), not calendar time. A window that contains a NaN is NaN.

        Returns
        -------
        Factor
            the rolling mean.

        Raises
        ------
        ValueError
            If ``w < 1``.
        """
        return Rolling(self, w, "mean")

    def rolling_var(self, w: int) -> Factor:
        """Return the rolling sample variance.

        Parameters
        ----------
        w : int
            Window length in bars. Must be at least 2. Divisor is ``w - 1``.

        Returns
        -------
        Factor
            NaN when the window contains a NaN.

        Raises
        ------
        ValueError
            If ``w < 2``.
        """
        return Rolling(self, w, "var")

    def rolling_std(self, w: int) -> Factor:
        """Return the rolling sample standard deviation.

        Parameters
        ----------
        w : int
            Window length in bars. Must be at least 2. Divisor is ``w - 1``.

        Returns
        -------
        Factor
            NaN when the window contains a NaN.

        Raises
        ------
        ValueError
            If ``w < 2``.
        """
        return Rolling(self, w, "std")

    def rolling_min(self, w: int) -> Factor:
        """Return the rolling minimum.

        Parameters
        ----------
        w : int
            Window length in bars (rows), not calendar time. A window that contains a NaN is NaN.

        Returns
        -------
        Factor
            the rolling minimum.

        Raises
        ------
        ValueError
            If ``w < 1``.
        """
        return Rolling(self, w, "min")

    def rolling_max(self, w: int) -> Factor:
        """Return the rolling maximum.

        Parameters
        ----------
        w : int
            Window length in bars (rows), not calendar time. A window that contains a NaN is NaN.

        Returns
        -------
        Factor
            the rolling maximum.

        Raises
        ------
        ValueError
            If ``w < 1``.
        """
        return Rolling(self, w, "max")

    def ts_rank(self, w: int) -> Factor:
        """Return the time-series rank of the current value inside the window.

        Parameters
        ----------
        w : int
            Window length in bars (rows), not calendar time. A window that contains a NaN is NaN.

        Returns
        -------
        Factor
            Fraction of the window less than or equal to the newest value, in ``(1/w, ..., 1)``.

        Raises
        ------
        ValueError
            If ``w < 1``.
        """
        return Rolling(self, w, "tsrank")

    def rolling_cov(self, other: Factor, w: int) -> Factor:
        """Return the rolling sample covariance with ``other``.

        Parameters
        ----------
        other : Factor
            The other series, paired asset by asset.
        w : int
            Window length in bars. Must be at least 2.

        Returns
        -------
        Factor
            Covariance. NaN if either window contains a NaN.

        Raises
        ------
        ValueError
            If ``w < 2``.
        """
        return Rolling2(self, other, w, "cov")

    def rolling_corr(self, other: Factor, w: int) -> Factor:
        """Return the rolling Pearson correlation with ``other``.

        Parameters
        ----------
        other : Factor
            The other series, paired asset by asset.
        w : int
            Window length in bars. Must be at least 2.

        Returns
        -------
        Factor
            Correlation in ``[-1, 1]``. NaN if a window is constant or contains a NaN.

        Raises
        ------
        ValueError
            If ``w < 2``.
        """
        return Rolling2(self, other, w, "corr")

    def rolling_beta(self, market: Factor, w: int) -> Factor:
        """Return the rolling regression slope on ``market``.

        Parameters
        ----------
        market : Factor
            The independent series. Usually a market return broadcast to every asset.
        w : int
            Window length in bars. Must be at least 2.

        Returns
        -------
        Factor
            ``cov(self, market) / var(market)``. NaN when the market window has no variance.

        Raises
        ------
        ValueError
            If ``w < 2``.
        """
        return Rolling2(self, market, w, "cov") / Rolling(market, w, "var")

    def ewm_mean(self, halflife: float) -> Factor:
        """Return the exponentially weighted mean.

        Parameters
        ----------
        halflife : float
            Half-life in bars. Must be positive. A half-life of 1 halves the previous mean each bar.

        Returns
        -------
        Factor
            Recursive mean. A NaN observation keeps the previous mean instead of wiping it.

        Raises
        ------
        ValueError
            If ``halflife <= 0``.
        """
        return EWMMean(self, halflife)

    # cross-sectional
    def rank(
        self, *, mask: Filter | None = None, groupby: Classifier | None = None, ascending: bool = True
    ) -> Factor:
        """Rank across assets on this bar.

        Parameters
        ----------
        mask : Filter or None, optional
            Who is eligible. ``None`` uses every finite value. Default is ``None``.
        groupby : Classifier or None, optional
            Group codes. ``None`` is one group. ``-1`` is missing and is dropped. Default is ``None``.
        ascending : bool, optional
            ``True`` gives rank 1 to the smallest value. Default is ``True``.

        Returns
        -------
        Factor
            Ranks ``1 .. k`` inside each group. Ties break by column order.
        """
        return CrossSection(self, "rank", mask, groupby, ascending)

    def zscore(self, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Factor:
        """Standardize across assets on this bar.

        Parameters
        ----------
        mask : Filter or None, optional
            Who is eligible. ``None`` uses every finite value. Default is ``None``.
        groupby : Classifier or None, optional
            Group codes. ``None`` is one group. ``-1`` is missing and is dropped. Default is ``None``.

        Returns
        -------
        Factor
            ``(x - mean) / sample_std`` inside the group. NaN if the group
            has fewer than two values or zero dispersion.
        """
        return CrossSection(self, "zscore", mask, groupby)

    def demean(self, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Factor:
        """Subtract the cross-sectional mean on this bar.

        Parameters
        ----------
        mask : Filter or None, optional
            Who is eligible. ``None`` uses every finite value. Default is ``None``.
        groupby : Classifier or None, optional
            Group codes. ``None`` is one group. ``-1`` is missing and is dropped. Default is ``None``.

        Returns
        -------
        Factor
            ``x - group mean``. A group of one becomes 0.
        """
        return CrossSection(self, "demean", mask, groupby)

    def top(self, n: int, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Filter:
        """Keep the ``n`` largest values.

        Parameters
        ----------
        n : int
            How many names to keep, counted inside each group when ``groupby`` is set.
        mask : Filter or None, optional
            Who is eligible. ``None`` uses every finite value. Default is ``None``.
        groupby : Classifier or None, optional
            Group codes. ``None`` is one group. ``-1`` is missing and is dropped. Default is ``None``.

        Returns
        -------
        Filter
            True for assets whose descending rank is at most ``n``.
        """
        return self.rank(mask=mask, groupby=groupby, ascending=False) <= n

    def bottom(self, n: int, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Filter:
        """Keep the ``n`` smallest values.

        Parameters
        ----------
        n : int
            How many names to keep, counted inside each group when ``groupby`` is set.
        mask : Filter or None, optional
            Who is eligible. ``None`` uses every finite value. Default is ``None``.
        groupby : Classifier or None, optional
            Group codes. ``None`` is one group. ``-1`` is missing and is dropped. Default is ``None``.

        Returns
        -------
        Filter
            True for assets whose ascending rank is at most ``n``.
        """
        return self.rank(mask=mask, groupby=groupby, ascending=True) <= n


class Filter(Term):
    """A boolean expression. Missing means False, so a missing value fails a screen.

    Notes
    -----
    ``&``, ``|``, and ``~`` combine filters. A bare ``True`` or ``False`` is accepted as the other operand.
    """

    kind = "filter"
    dtype = np.bool_
    missing = False

    def __and__(self, o):
        """Intersect with another filter.

        Parameters
        ----------
        o : Filter or bool
            The other mask. A bool is broadcast to every asset and bar.

        Returns
        -------
        Filter
            Elementwise boolean combination.

        Raises
        ------
        TypeError
            If ``o`` is not a filter or a bool.
        """
        return Logic(np.logical_and, "&", self, _as_filter(o))

    def __rand__(self, o):
        """Intersect, with this filter on the right.

        Parameters
        ----------
        o : Filter or bool
            The other mask. A bool is broadcast to every asset and bar.

        Returns
        -------
        Filter
            Elementwise boolean combination.

        Raises
        ------
        TypeError
            If ``o`` is not a filter or a bool.
        """
        return Logic(np.logical_and, "&", _as_filter(o), self)

    def __or__(self, o):
        """Union with another filter.

        Parameters
        ----------
        o : Filter or bool
            The other mask. A bool is broadcast to every asset and bar.

        Returns
        -------
        Filter
            Elementwise boolean combination.

        Raises
        ------
        TypeError
            If ``o`` is not a filter or a bool.
        """
        return Logic(np.logical_or, "|", self, _as_filter(o))

    def __ror__(self, o):
        """Union, with this filter on the right.

        Parameters
        ----------
        o : Filter or bool
            The other mask. A bool is broadcast to every asset and bar.

        Returns
        -------
        Filter
            Elementwise boolean combination.

        Raises
        ------
        TypeError
            If ``o`` is not a filter or a bool.
        """
        return Logic(np.logical_or, "|", _as_filter(o), self)

    def __invert__(self):
        """Invert the filter.

        Returns
        -------
        Filter
            True where this filter is False, including where it was missing.
        """
        return Not(self)

    def shift(self, n: int = 1) -> Filter:
        """Lag the filter by ``n`` bars.

        Parameters
        ----------
        n : int, optional
            Bars to look back. Default is 1. The first ``n`` rows are False.

        Returns
        -------
        Filter
            This filter from ``n`` bars earlier.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        return ShiftFilter(self, n)

    __hash__ = Term.__hash__


class Classifier(Term):
    """Integer group codes. ``-1`` is missing and is left out of grouped ranks and z-scores.

    Notes
    -----
    Build one with ``Column('sector', Classifier)``. Codes are whatever the caller stored in the panel.
    """

    kind = "classifier"
    dtype = np.int64
    missing = -1

    def eq(self, code: int) -> Filter:
        """Test membership in one code.

        Parameters
        ----------
        code : int
            Group code to match.

        Returns
        -------
        Filter
            True where the classifier equals ``code``.
        """
        return ClassifierEq(self, int(code))

    def isnull(self) -> Filter:
        """Test for a missing group code.

        Returns
        -------
        Filter
            True where the code is ``-1``.
        """
        return ClassifierEq(self, -1)

    __hash__ = Term.__hash__


def _as_filter(o) -> Filter:
    if isinstance(o, Filter):
        return o
    if isinstance(o, bool):
        return ConstFilter(o)
    raise TypeError(f"expected a Filter, got {type(o).__name__}")


# ====================================================================== leaves
class Column:
    """Factory for input columns: ``Column("close")``, ``Column("sector", Classifier)``."""

    _cache: ClassVar[dict[tuple[str, str], Term]] = {}

    def __new__(cls, name: str, kind: type[Term] = Factor) -> Any:  # type: ignore[misc]
        """Return the input term for a panel column.

        Parameters
        ----------
        cls : type
            Ignored. ``Column`` is a factory, not a container.
        name : str
            Column name in the :class:`~factorflow.panel.Panel`.
        kind : type, optional
            :class:`Factor` (default), :class:`Filter`, or :class:`Classifier`.

        Returns
        -------
        Term
            The same node every time for a given ``(name, kind)``.

        Raises
        ------
        KeyError
            If ``kind`` is not one of those three classes.
        """
        key = (name, kind.__name__)
        if key not in cls._cache:
            node_cls = {Factor: FactorInput, Filter: FilterInput, Classifier: ClassifierInput}[kind]
            cls._cache[key] = node_cls(name)
        return cls._cache[key]


class _Input:
    window = 1
    is_input = True

    def __init__(self, name: str):
        """Remember the column name.

        Parameters
        ----------
        name : str
            Panel column this leaf reads.
        """
        self.name = name
        self.params = (name,)

    def __repr__(self) -> str:
        """Return ``kind:name``, for example ``factor:close``.

        Returns
        -------
        str
            A short label.
        """
        return f"{self.kind}:{self.name}"  # type: ignore[attr-defined]

    def compute(self, blocks, prev=None):  # inputs are filled by the executor
        """Reject evaluation. The executors fill input terms from the panel.

        Raises
        ------
        RuntimeError
            Always. Inputs are not computed.
        """
        raise RuntimeError("input terms are not computed")


class FactorInput(_Input, Factor):
    """A float input column. Created by :class:`Column`, not called directly."""

    __hash__ = Term.__hash__


class FilterInput(_Input, Filter):
    """A boolean input column. Created by :class:`Column`, not called directly."""

    __hash__ = Term.__hash__


class ClassifierInput(_Input, Classifier):
    """An integer input column. Created by :class:`Column`, not called directly."""

    __hash__ = Term.__hash__


class BarData:
    """The standard bar fields produced by the bar builders."""

    open: Factor = Column("open")
    high: Factor = Column("high")
    low: Factor = Column("low")
    close: Factor = Column("close")
    volume: Factor = Column("volume")
    dollar: Factor = Column("dollar")
    n_ticks: Factor = Column("n_ticks")
    has_trade: Filter = Column("has_trade", Filter)
    FIELDS: ClassVar[tuple[str, ...]] = ("open", "high", "low", "close", "volume", "dollar", "n_ticks")


class Constant(Factor):
    """A scalar broadcast to every asset (it has no inputs and never changes)."""

    def __init__(self, value: Scalar):
        """Store a scalar that will be broadcast.

        Parameters
        ----------
        value : int or float or bool
            The constant. Stored as float.
        """
        self.value = float(value)
        self.params = (self.value,)

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Reject evaluation. Constants are folded into the parent operator.

        Raises
        ------
        RuntimeError
            Always.
        """
        raise RuntimeError("constants are folded into their parents")


class ConstFilter(Filter):
    """A broadcast boolean. Built when a filter is combined with ``True`` or ``False``."""

    def __init__(self, value: bool):
        """Store the boolean.

        Parameters
        ----------
        value : bool
            The constant mask.
        """
        self.value = bool(value)
        self.params = (self.value,)

    __hash__ = Term.__hash__


def _operand(t: Term, blocks: Sequence[np.ndarray], i: int):
    return t.value if isinstance(t, (Constant, ConstFilter)) else blocks[i]


def _real_inputs(*ts: Term) -> tuple[Term, ...]:
    return tuple(t for t in ts if not isinstance(t, (Constant, ConstFilter)))


# ====================================================================== elementwise
class _Elementwise:
    """Mixin: elementwise over inputs; constants are folded."""

    operands: tuple[Term, ...]

    def _args(self, blocks: Sequence[np.ndarray]):
        it = iter(blocks)
        return [t.value if isinstance(t, (Constant, ConstFilter)) else next(it) for t in self.operands]


class BinOp(_Elementwise, Factor):
    """Elementwise arithmetic node. Built by :class:`Factor` operators, not called directly."""

    def __init__(self, fn: Callable, sym: str, a: Term, b: Term):
        """Bind the operator and its two operands.

        Parameters
        ----------
        fn : callable
            The numpy or operator function.
        sym : str
            Symbol stored in the term identity, such as ``'+'``.
        a : Term
            Left operand. A :class:`Constant` is folded in and is not an input.
        b : Term
            Right operand.

        Raises
        ------
        TypeError
            If both operands are constants. Fold those before building a node.
        """
        if isinstance(a, Constant) and isinstance(b, Constant):
            raise TypeError("both operands are constants")
        self.fn, self.operands = fn, (a, b)
        self.inputs = _real_inputs(a, b)
        self.params = (sym, tuple(t.params if isinstance(t, Constant) else ("in",) for t in (a, b)))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Apply the operator.

        Parameters
        ----------
        blocks : sequence of ndarray
            Inputs that are not constants, in input order. Shape ``(T, N)``.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray
            Float64 result. Divide-by-zero and overflow become NaN, not exceptions.
        """
        a, b = self._args(blocks)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return np.asarray(self.fn(a, b), dtype=np.float64)


class Unary(Factor):
    """Elementwise unary node, such as ``log``, ``sqrt``, ``neg``, or ``abs``."""

    def __init__(self, fn: Callable, name: str, a: Factor, columnwise: bool = True):
        """Bind the function.

        Parameters
        ----------
        fn : callable
            A numpy ufunc.
        name : str
            Name stored in the term identity.
        a : Factor
            The input.
        columnwise : bool, optional
            ``False`` for ``log``, which is always computed on full rows so
            live and batch take the same path. Default is ``True``.
        """
        self.fn, self.inputs, self.params = fn, (a,), (name,)
        # transcendental ufuncs may take different SIMD paths for different array lengths,
        # so they are always computed on full rows (never on a column subset).
        self.columnwise = columnwise
        self.rowwise = not columnwise

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Apply the ufunc.

        Parameters
        ----------
        blocks : sequence of ndarray
            The single input.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray
            Float64 result. Domain errors become NaN.
        """
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return np.asarray(self.fn(blocks[0]), dtype=np.float64)

    def compute_row(self, rows, prev):
        """Apply the ufunc to the last row. Used when ``columnwise`` is False.

        Parameters
        ----------
        rows : sequence of ndarray
            Windows; the last row is evaluated.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of shape (N,)
            One row.
        """
        return self.compute([r[-1] for r in rows])


class FillNa(Factor):
    """Replace NaN with a constant. Built by :meth:`Factor.fillna`."""

    def __init__(self, a: Factor, value: float):
        """Store the input and the fill value.

        Parameters
        ----------
        a : Factor
            The input.
        value : float
            Replacement for NaN.
        """
        self.inputs, self.params, self.value = (a,), (value,), value

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Replace NaNs.

        Parameters
        ----------
        blocks : sequence of ndarray
            The input.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray
            ``value`` where the input is NaN, otherwise the input.
        """
        x = blocks[0]
        return np.where(np.isnan(x), self.value, x)


class Where(_Elementwise, Factor):
    """Conditional choice. Built by :func:`where`."""

    def __init__(self, cond: Filter, a: Factor | Scalar, b: Factor | Scalar):
        """Bind the condition and the two branches.

        Parameters
        ----------
        cond : Filter
            Which branch to take.
        a : Factor or scalar
            Value where ``cond`` is True.
        b : Factor or scalar
            Value where ``cond`` is False.
        """
        la, lb = _lift(a), _lift(b)
        self.operands = (cond, la, lb)
        self.inputs = _real_inputs(cond, la, lb)
        self.params = tuple(t.params if isinstance(t, Constant) else ("in",) for t in (la, lb))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Select elements.

        Parameters
        ----------
        blocks : sequence of ndarray
            Non-constant operands.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray
            Float64 ``np.where`` result.
        """
        c, a, b = self._args(blocks)
        return np.where(c, a, b).astype(np.float64)


def where(cond: Filter, a: Factor | Scalar, b: Factor | Scalar) -> Factor:
    """Choose ``a`` or ``b`` per element.

    Parameters
    ----------
    cond : Filter
        Where to take ``a``. False, including a missing filter, takes ``b``.
    a : Factor or int or float or bool
        Value where ``cond`` is True.
    b : Factor or int or float or bool
        Value where ``cond`` is False.

    Returns
    -------
    Factor
        ``a`` or ``b``, as float64.

    Examples
    --------
    >>> import factorflow as ff
    >>> ff.where(ff.Column('close') > 0, 1, 0).kind
    'factor'
    """
    return Where(cond, a, b)


class Compare(_Elementwise, Filter):
    """Elementwise comparison. Built by :class:`Factor` comparisons."""

    def __init__(self, fn: Callable, sym: str, a: Term, b: Term):
        """Bind the comparison.

        Parameters
        ----------
        fn : callable
            The comparison function.
        sym : str
            Symbol stored in the identity, such as ``'<'``.
        a : Term
            Left side.
        b : Term
            Right side.
        """
        self.fn, self.operands = fn, (a, b)
        self.inputs = _real_inputs(a, b)
        self.params = (sym, tuple(t.params if isinstance(t, Constant) else ("in",) for t in (a, b)))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Compare the operands.

        Parameters
        ----------
        blocks : sequence of ndarray
            Non-constant operands.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of bool
            False where a comparison involves NaN.
        """
        a, b = self._args(blocks)
        with np.errstate(invalid="ignore"):
            return np.asarray(self.fn(a, b), dtype=np.bool_)


class Logic(_Elementwise, Filter):
    """Elementwise ``and`` or ``or`` of two filters."""

    def __init__(self, fn: Callable, sym: str, a: Filter, b: Filter):
        """Bind the boolean operator.

        Parameters
        ----------
        fn : callable
            ``np.logical_and`` or ``np.logical_or``.
        sym : str
            ``'&'`` or ``'|'``.
        a : Filter
            Left filter.
        b : Filter
            Right filter.
        """
        self.fn, self.operands = fn, (a, b)
        self.inputs = _real_inputs(a, b)
        self.params = (sym, tuple(t.params if isinstance(t, ConstFilter) else ("in",) for t in (a, b)))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Combine the filters.

        Parameters
        ----------
        blocks : sequence of ndarray
            Non-constant operands.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of bool
            The combination.
        """
        a, b = self._args(blocks)
        return np.asarray(self.fn(a, b), dtype=np.bool_)


class Not(Filter):
    """Negation of a filter. Built by ``~``."""

    def __init__(self, a: Filter):
        """Store the filter to invert.

        Parameters
        ----------
        a : Filter
            The input.
        """
        self.inputs = (a,)

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Invert the mask.

        Parameters
        ----------
        blocks : sequence of ndarray
            The filter.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of bool
            The logical not.
        """
        return ~blocks[0]


class IsNull(Filter):
    """NaN test. Built by :meth:`Factor.isnull`."""

    def __init__(self, a: Factor):
        """Store the factor to test.

        Parameters
        ----------
        a : Factor
            The input.
        """
        self.inputs = (a,)

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Test for NaN.

        Parameters
        ----------
        blocks : sequence of ndarray
            The factor.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of bool
            True where the input is NaN.
        """
        return np.isnan(blocks[0])


class ClassifierEq(Filter):
    """Membership test. Built by :meth:`Classifier.eq` and :meth:`Classifier.isnull`."""

    def __init__(self, a: Classifier, code: int):
        """Store the classifier and the code.

        Parameters
        ----------
        a : Classifier
            The input.
        code : int
            Code to match. ``-1`` tests for missing.
        """
        self.inputs, self.params, self.code = (a,), (code,), code

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Compare codes.

        Parameters
        ----------
        blocks : sequence of ndarray
            The classifier, int64.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of bool
            True where the code equals ``self.code``.
        """
        return blocks[0] == self.code


# ====================================================================== time series
_ROLLING = {
    "sum": K.wsum,
    "mean": K.wmean,
    "var": K.wvar,
    "std": K.wstd,
    "min": K.wmin,
    "max": K.wmax,
    "tsrank": K.wtsrank,
}


class Rolling(Factor):
    """A one-input rolling window. Built by the ``rolling_*`` and :meth:`Factor.ts_rank` methods."""

    def __init__(self, a: Factor, w: int, how: str):
        """Store the input, the window, and which reduction to run.

        Parameters
        ----------
        a : Factor
            The input.
        w : int
            Window length in bars.
        how : str
            One of ``sum``, ``mean``, ``var``, ``std``, ``min``, ``max``, ``tsrank``.

        Raises
        ------
        ValueError
            If ``w < 1``, or if a variance or standard deviation has ``w < 2``.
        """
        if w < 1:
            raise ValueError("window must be >= 1")
        if how in ("var", "std") and w < 2:
            raise ValueError(f"rolling_{how} needs a window >= 2")
        self.inputs, self.window, self.params, self.how = (a,), int(w), (how,), how

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Run the windowed kernel.

        Parameters
        ----------
        blocks : sequence of ndarray
            The input, shape ``(T_in, N)``.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of shape (T_in - w + 1, N)
            The reduction. Same function batch and live call.
        """
        return _ROLLING[self.how](blocks[0], self.window)


class Rolling2(Factor):
    """A two-input rolling window: covariance or correlation."""

    def __init__(self, a: Factor, b: Factor, w: int, how: str):
        """Store the pair and the window.

        Parameters
        ----------
        a : Factor
            Left series.
        b : Factor
            Right series.
        w : int
            Window length in bars.
        how : str
            ``cov`` or ``corr``.

        Raises
        ------
        ValueError
            If ``w < 2``.
        """
        if w < 2:
            raise ValueError("a two-input rolling window must be >= 2")
        self.inputs, self.window, self.params, self.how = (a, b), int(w), (how,), how

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Run the two-series kernel.

        Parameters
        ----------
        blocks : sequence of ndarray
            The two inputs.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray
            Covariance or correlation, shape ``(T_in - w + 1, N)``.
        """
        fn = K.wcov if self.how == "cov" else K.wcorr
        return fn(blocks[0], blocks[1], self.window)


class Shift(Factor):
    """A lag of a factor. Built by :meth:`Factor.shift`."""

    def __init__(self, a: Factor, n: int):
        """Store the lag.

        Parameters
        ----------
        a : Factor
            The input.
        n : int
            Bars to look back. The kernel window is ``n + 1``.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError("shift must be >= 1")
        self.inputs, self.window, self.params = (a,), int(n) + 1, ()

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Drop the last ``n`` rows of the block so each output lines up with its lag.

        Parameters
        ----------
        blocks : sequence of ndarray
            The input.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray
            Shape ``(T_in - n, N)``.
        """
        return K.wshift(blocks[0], self.window)


class ShiftFilter(Filter):
    """A lag of a filter. Built by :meth:`Filter.shift`. The leading rows are False, not NaN."""

    def __init__(self, a: Filter, n: int):
        """Store the lag.

        Parameters
        ----------
        a : Filter
            The input.
        n : int
            Bars to look back.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError("shift must be >= 1")
        self.inputs, self.window, self.params = (a,), int(n) + 1, ()

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        """Lag the boolean block.

        Parameters
        ----------
        blocks : sequence of ndarray
            The filter.
        prev : ndarray or None
            Ignored.

        Returns
        -------
        ndarray of bool
            The lagged mask. Padding in batch is the filter's missing value, False.
        """
        return K.wshift(blocks[0], self.window).astype(np.bool_)


class EWMMean(Factor):
    """Exponentially weighted mean with a half-life in bars (recursive, O(1) per row)."""

    rowwise = True
    recursive = True

    def __init__(self, a: Factor, halflife: float):
        """Store the half-life and the matching smoothing weight.

        Parameters
        ----------
        a : Factor
            The input.
        halflife : float
            Half-life in bars. Must be positive.

        Raises
        ------
        ValueError
            If ``halflife <= 0``.
        """
        if halflife <= 0:
            raise ValueError("halflife must be > 0")
        self.inputs, self.params = (a,), (float(halflife),)
        self.alpha = K.ewm_alpha(float(halflife))

    __hash__ = Term.__hash__

    def compute_row(self, rows, prev):
        """Advance the mean by one row.

        Parameters
        ----------
        rows : sequence of ndarray
            The input window. The last row is the new observation.
        prev : ndarray of shape (N,) or None
            Previous mean. ``None`` starts at the observation.

        Returns
        -------
        ndarray of shape (N,)
            The updated mean. A NaN observation keeps ``prev``.
        """
        p = prev if prev is not None else np.full(rows[0].shape[-1], np.nan)
        return K.ewm_step(rows[0][-1], p, self.alpha)


# ====================================================================== cross-sectional
class CrossSection(Factor):
    """Rank, z-score, or demean across assets.

    Built by :meth:`Factor.rank`, :meth:`Factor.zscore`, and :meth:`Factor.demean`.

    Notes
    -----
    ``columnwise`` is False: one changed asset recomputes the whole row,
    because every output depends on the group.
    """

    rowwise = True
    columnwise = False

    def __init__(
        self, a: Factor, how: str, mask: Filter | None, groupby: Classifier | None, ascending: bool = True
    ):
        """Bind the reduction and the optional mask and groups.

        Parameters
        ----------
        a : Factor
            Values to reduce.
        how : str
            ``rank``, ``zscore``, or ``demean``.
        mask : Filter or None
            Who is eligible.
        groupby : Classifier or None
            Group codes.
        ascending : bool, optional
            Rank direction. Ignored by z-score and demean. Default is ``True``.
        """
        self.how, self.ascending = how, ascending
        self.has_mask, self.has_groups = mask is not None, groupby is not None
        ins: list[Term] = [a]
        if mask is not None:
            ins.append(mask)
        if groupby is not None:
            ins.append(groupby)
        self.inputs = tuple(ins)
        self.params = (how, ascending, self.has_mask, self.has_groups)

    __hash__ = Term.__hash__

    def compute_row(self, rows, prev):
        """Reduce one cross-section.

        Parameters
        ----------
        rows : sequence of ndarray
            Value, then the mask if any, then the groups if any. The last row of each is used.
        prev : ndarray or None
            Ignored. Cross-sections have no memory.

        Returns
        -------
        ndarray of shape (N,)
            The rank, z-score, or demeaned value. Ineligible assets are NaN.
        """
        it = iter(r[-1] for r in rows)
        v = next(it)
        mask = next(it) if self.has_mask else None
        groups = next(it) if self.has_groups else None
        if self.how == "rank":
            return K.cs_rank(v, mask, groups, self.ascending)
        if self.how == "zscore":
            return K.cs_zscore(v, mask, groups)
        return K.cs_demean(v, mask, groups)
