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
        return hash(self._key())

    def __eq__(self, other: object) -> bool:  # identity semantics; use .eq() for comparisons
        return self is other

    def __repr__(self) -> str:
        args = ", ".join([repr(i) for i in self.inputs] + [repr(p) for p in self.params])
        return f"{type(self).__name__}({args})"

    # -- compute ----------------------------------------------------------------------
    def compute(self, blocks: Sequence[np.ndarray], prev: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    def compute_row(self, rows: Sequence[np.ndarray], prev: np.ndarray | None) -> np.ndarray:
        """Row-wise terms: one output row from one row (or window) of each input."""
        raise NotImplementedError

    # -- graph helpers --------------------------------------------------------------------
    def nonnull(self) -> np.ndarray:  # pragma: no cover - helper for subclasses
        raise NotImplementedError


def _lift(x: Term | Scalar) -> Term:
    if isinstance(x, Term):
        return x
    return Constant(x)


# ====================================================================== kinds
class Factor(Term):
    kind = "factor"
    dtype = np.float64
    missing = np.nan

    # arithmetic
    def __add__(self, o):
        return BinOp(operator.add, "+", self, _lift(o))

    def __radd__(self, o):
        return BinOp(operator.add, "+", _lift(o), self)

    def __sub__(self, o):
        return BinOp(operator.sub, "-", self, _lift(o))

    def __rsub__(self, o):
        return BinOp(operator.sub, "-", _lift(o), self)

    def __mul__(self, o):
        return BinOp(operator.mul, "*", self, _lift(o))

    def __rmul__(self, o):
        return BinOp(operator.mul, "*", _lift(o), self)

    def __truediv__(self, o):
        return BinOp(operator.truediv, "/", self, _lift(o))

    def __rtruediv__(self, o):
        return BinOp(operator.truediv, "/", _lift(o), self)

    def __neg__(self):
        return Unary(np.negative, "neg", self)

    def __abs__(self):
        return Unary(np.abs, "abs", self)

    # comparisons -> Filter
    def __lt__(self, o):
        return Compare(operator.lt, "<", self, _lift(o))

    def __le__(self, o):
        return Compare(operator.le, "<=", self, _lift(o))

    def __gt__(self, o):
        return Compare(operator.gt, ">", self, _lift(o))

    def __ge__(self, o):
        return Compare(operator.ge, ">=", self, _lift(o))

    def eq(self, o):
        return Compare(operator.eq, "==", self, _lift(o))

    def ne(self, o):
        return Compare(operator.ne, "!=", self, _lift(o))

    __hash__ = Term.__hash__

    def log(self) -> Factor:
        return Unary(np.log, "log", self, columnwise=False)

    def sqrt(self) -> Factor:
        return Unary(np.sqrt, "sqrt", self)

    def isnull(self) -> Filter:
        return IsNull(self)

    def notnull(self) -> Filter:
        return ~IsNull(self)

    def fillna(self, value: float) -> Factor:
        return FillNa(self, float(value))

    # time series
    def shift(self, n: int = 1) -> Factor:
        return Shift(self, n)

    def delta(self, n: int = 1) -> Factor:
        return self - self.shift(n)

    def pct_change(self, n: int = 1) -> Factor:
        return self / self.shift(n) - 1.0

    def rolling_sum(self, w: int) -> Factor:
        return Rolling(self, w, "sum")

    def rolling_mean(self, w: int) -> Factor:
        return Rolling(self, w, "mean")

    def rolling_var(self, w: int) -> Factor:
        return Rolling(self, w, "var")

    def rolling_std(self, w: int) -> Factor:
        return Rolling(self, w, "std")

    def rolling_min(self, w: int) -> Factor:
        return Rolling(self, w, "min")

    def rolling_max(self, w: int) -> Factor:
        return Rolling(self, w, "max")

    def ts_rank(self, w: int) -> Factor:
        return Rolling(self, w, "tsrank")

    def rolling_cov(self, other: Factor, w: int) -> Factor:
        return Rolling2(self, other, w, "cov")

    def rolling_corr(self, other: Factor, w: int) -> Factor:
        return Rolling2(self, other, w, "corr")

    def rolling_beta(self, market: Factor, w: int) -> Factor:
        return Rolling2(self, market, w, "cov") / Rolling(market, w, "var")

    def ewm_mean(self, halflife: float) -> Factor:
        return EWMMean(self, halflife)

    # cross-sectional
    def rank(
        self, *, mask: Filter | None = None, groupby: Classifier | None = None, ascending: bool = True
    ) -> Factor:
        return CrossSection(self, "rank", mask, groupby, ascending)

    def zscore(self, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Factor:
        return CrossSection(self, "zscore", mask, groupby)

    def demean(self, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Factor:
        return CrossSection(self, "demean", mask, groupby)

    def top(self, n: int, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Filter:
        return self.rank(mask=mask, groupby=groupby, ascending=False) <= n

    def bottom(self, n: int, *, mask: Filter | None = None, groupby: Classifier | None = None) -> Filter:
        return self.rank(mask=mask, groupby=groupby, ascending=True) <= n


class Filter(Term):
    kind = "filter"
    dtype = np.bool_
    missing = False

    def __and__(self, o):
        return Logic(np.logical_and, "&", self, _as_filter(o))

    def __rand__(self, o):
        return Logic(np.logical_and, "&", _as_filter(o), self)

    def __or__(self, o):
        return Logic(np.logical_or, "|", self, _as_filter(o))

    def __ror__(self, o):
        return Logic(np.logical_or, "|", _as_filter(o), self)

    def __invert__(self):
        return Not(self)

    def shift(self, n: int = 1) -> Filter:
        return ShiftFilter(self, n)

    __hash__ = Term.__hash__


class Classifier(Term):
    kind = "classifier"
    dtype = np.int64
    missing = -1

    def eq(self, code: int) -> Filter:
        return ClassifierEq(self, int(code))

    def isnull(self) -> Filter:
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
        key = (name, kind.__name__)
        if key not in cls._cache:
            node_cls = {Factor: FactorInput, Filter: FilterInput, Classifier: ClassifierInput}[kind]
            cls._cache[key] = node_cls(name)
        return cls._cache[key]


class _Input:
    window = 1
    is_input = True

    def __init__(self, name: str):
        self.name = name
        self.params = (name,)

    def __repr__(self) -> str:
        return f"{self.kind}:{self.name}"  # type: ignore[attr-defined]

    def compute(self, blocks, prev=None):  # inputs are filled by the executor
        raise RuntimeError("input terms are not computed")


class FactorInput(_Input, Factor):
    __hash__ = Term.__hash__


class FilterInput(_Input, Filter):
    __hash__ = Term.__hash__


class ClassifierInput(_Input, Classifier):
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
        self.value = float(value)
        self.params = (self.value,)

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        raise RuntimeError("constants are folded into their parents")


class ConstFilter(Filter):
    def __init__(self, value: bool):
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
    def __init__(self, fn: Callable, sym: str, a: Term, b: Term):
        if isinstance(a, Constant) and isinstance(b, Constant):
            raise TypeError("both operands are constants")
        self.fn, self.operands = fn, (a, b)
        self.inputs = _real_inputs(a, b)
        self.params = (sym, tuple(t.params if isinstance(t, Constant) else ("in",) for t in (a, b)))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        a, b = self._args(blocks)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return np.asarray(self.fn(a, b), dtype=np.float64)


class Unary(Factor):
    def __init__(self, fn: Callable, name: str, a: Factor, columnwise: bool = True):
        self.fn, self.inputs, self.params = fn, (a,), (name,)
        # transcendental ufuncs may take different SIMD paths for different array lengths,
        # so they are always computed on full rows (never on a column subset).
        self.columnwise = columnwise
        self.rowwise = not columnwise

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return np.asarray(self.fn(blocks[0]), dtype=np.float64)

    def compute_row(self, rows, prev):
        return self.compute([r[-1] for r in rows])


class FillNa(Factor):
    def __init__(self, a: Factor, value: float):
        self.inputs, self.params, self.value = (a,), (value,), value

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        x = blocks[0]
        return np.where(np.isnan(x), self.value, x)


class Where(_Elementwise, Factor):
    def __init__(self, cond: Filter, a: Factor | Scalar, b: Factor | Scalar):
        la, lb = _lift(a), _lift(b)
        self.operands = (cond, la, lb)
        self.inputs = _real_inputs(cond, la, lb)
        self.params = tuple(t.params if isinstance(t, Constant) else ("in",) for t in (la, lb))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        c, a, b = self._args(blocks)
        return np.where(c, a, b).astype(np.float64)


def where(cond: Filter, a: Factor | Scalar, b: Factor | Scalar) -> Factor:
    return Where(cond, a, b)


class Compare(_Elementwise, Filter):
    def __init__(self, fn: Callable, sym: str, a: Term, b: Term):
        self.fn, self.operands = fn, (a, b)
        self.inputs = _real_inputs(a, b)
        self.params = (sym, tuple(t.params if isinstance(t, Constant) else ("in",) for t in (a, b)))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        a, b = self._args(blocks)
        with np.errstate(invalid="ignore"):
            return np.asarray(self.fn(a, b), dtype=np.bool_)


class Logic(_Elementwise, Filter):
    def __init__(self, fn: Callable, sym: str, a: Filter, b: Filter):
        self.fn, self.operands = fn, (a, b)
        self.inputs = _real_inputs(a, b)
        self.params = (sym, tuple(t.params if isinstance(t, ConstFilter) else ("in",) for t in (a, b)))

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        a, b = self._args(blocks)
        return np.asarray(self.fn(a, b), dtype=np.bool_)


class Not(Filter):
    def __init__(self, a: Filter):
        self.inputs = (a,)

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        return ~blocks[0]


class IsNull(Filter):
    def __init__(self, a: Factor):
        self.inputs = (a,)

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        return np.isnan(blocks[0])


class ClassifierEq(Filter):
    def __init__(self, a: Classifier, code: int):
        self.inputs, self.params, self.code = (a,), (code,), code

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
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
    def __init__(self, a: Factor, w: int, how: str):
        if w < 1:
            raise ValueError("window must be >= 1")
        if how in ("var", "std") and w < 2:
            raise ValueError(f"rolling_{how} needs a window >= 2")
        self.inputs, self.window, self.params, self.how = (a,), int(w), (how,), how

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        return _ROLLING[self.how](blocks[0], self.window)


class Rolling2(Factor):
    def __init__(self, a: Factor, b: Factor, w: int, how: str):
        if w < 2:
            raise ValueError("a two-input rolling window must be >= 2")
        self.inputs, self.window, self.params, self.how = (a, b), int(w), (how,), how

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        fn = K.wcov if self.how == "cov" else K.wcorr
        return fn(blocks[0], blocks[1], self.window)


class Shift(Factor):
    def __init__(self, a: Factor, n: int):
        if n < 1:
            raise ValueError("shift must be >= 1")
        self.inputs, self.window, self.params = (a,), int(n) + 1, ()

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        return K.wshift(blocks[0], self.window)


class ShiftFilter(Filter):
    def __init__(self, a: Filter, n: int):
        if n < 1:
            raise ValueError("shift must be >= 1")
        self.inputs, self.window, self.params = (a,), int(n) + 1, ()

    __hash__ = Term.__hash__

    def compute(self, blocks, prev=None):
        return K.wshift(blocks[0], self.window).astype(np.bool_)


class EWMMean(Factor):
    """Exponentially weighted mean with a half-life in bars (recursive, O(1) per row)."""

    rowwise = True
    recursive = True

    def __init__(self, a: Factor, halflife: float):
        if halflife <= 0:
            raise ValueError("halflife must be > 0")
        self.inputs, self.params = (a,), (float(halflife),)
        self.alpha = K.ewm_alpha(float(halflife))

    __hash__ = Term.__hash__

    def compute_row(self, rows, prev):
        p = prev if prev is not None else np.full(rows[0].shape[-1], np.nan)
        return K.ewm_step(rows[0][-1], p, self.alpha)


# ====================================================================== cross-sectional
class CrossSection(Factor):
    rowwise = True
    columnwise = False

    def __init__(
        self, a: Factor, how: str, mask: Filter | None, groupby: Classifier | None, ascending: bool = True
    ):
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
        it = iter(r[-1] for r in rows)
        v = next(it)
        mask = next(it) if self.has_mask else None
        groups = next(it) if self.has_groups else None
        if self.how == "rank":
            return K.cs_rank(v, mask, groups, self.ascending)
        if self.how == "zscore":
            return K.cs_zscore(v, mask, groups)
        return K.cs_demean(v, mask, groups)
