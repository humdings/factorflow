# Factor engine: design proposal

*Rocinante Research. Draft 2, 2026-11-02 (HST). This is the design that the factorflow M0 implementation follows.*

Every library status below (license, last release or push, maintenance) was checked on 2026-11-02 against GitHub, PyPI and the project sites.

**Decisions (David, 2026-11-02):**
- **Package and repo:** name `factorflow`, a separate private repo, Apache-2.0.
- **Data:** 10 years of history. A survivorship-biased PoC is OK. Crypto is BTC/ETH/SOL.
- **Bars:**
  - labeled by end time, with a 2 s grace before a minute bar is sealed
  - missing bars carry the close forward with zero volume
  - the dashboard shows open-bar values flagged as live, so every value carries an open/sealed flag
- **Live minute bars:** from additional IB lines (a v1 target; M0 has no IB integration).
- **Arb:** the `dirty_strikes` work proceeds separately in humdesk.

---

## 0. TL;DR

- **Build it, don't adopt it.** Nothing available does all of the following: a Pipeline-style factor/filter/classifier language, point-in-time fundamentals, a batch backtest mode, and an incremental live mode that gives the same answers. Zipline-reloaded's Pipeline comes closest, but it's tied to zipline's bundles, calendars and asset DB. Spectre is GPL-3.0 and dormant. qlib is a full ML platform with its own data format.
- **Don't plug in Jane Street Incremental itself.** It's OCaml (MIT, actively maintained) and has no Python bindings. Bridging it would be awkward and would gain nothing, because our graph nodes are coarse. Each node is a whole T×N array, so the time goes into the numeric kernels, not into graph bookkeeping. **Do copy its semantics**, which take about 300–500 lines of pure Python: variables, a height-ordered recompute heap, `stabilize()`, cutoffs, observers and necessity.
- **Architecture.** A Pipeline-style expression API builds a term DAG, and a compiler turns it into a plan. There are two executors over the same kernels:
  - **batch**, for backtests and research,
  - **incremental**, which stabilizes when new data arrives.
  
  A property test enforces that they agree bit for bit. Kernels are written against the Python **array API**, so the same code runs on numpy (the reference) and on torch (CPU or CUDA). Polars and DuckDB handle ingest, point-in-time as-of joins and storage.
- **Time model (revised 2026-11-02, see section 11).** One pipeline runs on **one bar grid** with a configurable frequency (1d, 1min, 5min, or 1s for arb). Ticks never create events. They only update the **open bar**, the last and still-provisional row. On a tick, the engine *revises* that row only. At bar close it **seals** the row, and from then on the row is immutable and identical to the batch result.
- **Data.** Everything is bitemporal, keyed by `(entity, period/event time, knowledge time)`. Filings are keyed by SEC **acceptance timestamp**, and macro data by ALFRED **vintage**. This is what rules out lookahead in backtests.
- **Storage.** Research history (fundamentals, vintages, factor panels) goes in Parquet + DuckDB, matching humdesk's existing `BarStore`. QuestDB keeps live ticks and bars and gains a narrow `factor_latest`/`factor_values` table for the dashboard.
- **PoC scope.** The S&P 100 (about 100 names, later 500) on daily bars, BTC/ETH/SOL plus up to 2 more majors, about 40 filing-based factors, about 60 FRED series and the Ken French factors for validation. That's roughly **1–3 GB** on disk.
- **Package.** **`factorflow`** lives in its own repo (decided 2026-11-02), with no humdesk imports, under Apache-2.0. humdesk consumes it as a Nix flake input or a uv git dependency.

---

## 1. Survey

### 1.1 The three inspirations

| Project | What it is | Status (checked 2026-11-02) | License | Verdict |
|---|---|---|---|---|
| **janestreet/incremental** | OCaml self-adjusting computation (after Acar). `Var`, `map`/`bind`, `observe`, `stabilize`, cutoffs | Active (pushed 2026-07, v0.17), 1.5k★ | MIT | **Copy the semantics, not the code** (see 1.3) |
| **Heerozh/spectre** | Factor engine and backtester on torch/GPU. Pipeline-like API with dense (date × asset) tensors | Last push 2025-04, 10 open issues, effectively unmaintained | **GPL-3.0** | Good ideas: dense tensors, GPU, `.shift()`/rolling on tensors. GPL would infect an Apache/MIT package, so **don't vendor any code** |
| **quantopian/zipline → zipline-reloaded** | Pipeline: `Factor`/`Filter`/`Classifier`, `CustomFactor(window_length, inputs)`, `rank/zscore/demean(groupby=, mask=)`, a term graph with an "extra rows" calculation | quantopian/zipline last pushed 2024-02 and is dead. **stefan-jansen/zipline-reloaded is active** (3.2.1, 2026-10-25) | Apache-2.0 | **Best API to imitate.** The engine is coupled to bundles, trading calendars and the asset DB. We reuse the *design*, which is fine under Apache-2.0 and allows code borrowing with attribution |

Pipeline's ideas worth keeping:
- Terms are immutable, hashable nodes, so the same expression is computed once (CSE for free).
- Each term declares `window_length` and `inputs`. The engine works out how many **extra lookback rows** each input needs, so a 252-day factor computed on day *t* sees exactly *t−251…t*.
- Filters and masks are first-class, and `mask=` restricts every downstream computation.
- Classifiers (sector, industry, size bucket) drive grouped normalization.
- Output is one cross-section per day, which gives a clean point-in-time unit.

What to fix compared with Pipeline:
- The calendar and asset DB are hardwired.
- Fundamentals have no notion of time.
- The engine is batch only.
- It's pandas-heavy and has awkward Cython.
- It has no GPU path.

### 1.2 Other modern options

| Option | Category | Status / license | Role for us |
|---|---|---|---|
| **Polars** 2.1 (2026-10) | Lazy expression DataFrame engine. The streaming engine became the default in 2.0 | Very active, MIT | **Ingest, PIT `join_asof`, reshaping, storage I/O.** It could also be a backend, since rolling and `over()` cover most factor ops. But its streaming means "out-of-core", not "incremental view maintenance" |
| **DuckDB** 2.0 (2026-10) | Embedded OLAP SQL over Parquet | Very active, MIT | **Query layer over Parquet.** It already backs humdesk `BarStore`. Strong ASOF JOIN |
| **ibis** 12 | Portable dataframe API that compiles to DuckDB, Polars and others | Active, Apache-2.0 | Not needed. It adds a layer without adding incrementality |
| **Apache Hamilton** (ex-DAGWorks, Apache incubating, 1.90) | DAG built from function names and parameters | Active, Apache-2.0 | Inspiration for "functions are nodes". Too coarse and too batch-oriented for per-factor incrementality |
| **Dagster** | Asset orchestration with partitions and freshness | Very active, Apache-2.0 | **Optional later** for *data-asset* scheduling (nightly EDGAR and FRED pulls). Overkill for the PoC, where systemd timers are enough |
| **Bytewax** | Python stream processing (Timely-based) | Company shut down in 2025. Last release 0.21.1 (2024-11), now community-maintained | Avoid |
| **Pathway** | Incremental, live data framework built on Rust differential dataflow, with a Python API | Very active, **BSL 1.1** (source-available, not OSI) | The closest Python-native "real incremental" engine. The BSL rules it out as a dependency of a package you want to open-source, and it's a heavyweight runtime |
| **Materialize** | Incremental SQL (differential dataflow) | Active, **BSL** | Server, SQL only, BSL. No |
| **Feldera** (DBSP) | Incremental SQL engine with a Python SDK | Very active, MIT (OSS edition) | Correct incremental view maintenance over SQL. Viable later for *relational* derived data, such as a live screen over the whole universe. A poor fit for the numeric/linear-algebra factor work (regressions, covariances) |
| **differential-dataflow** (Rust) | Foundation of Materialize and Pathway | Active, MIT | No maintained Python bindings |
| **reaktiv** | Signals/computed/effects (fine-grained, Angular/Solid style) | Active (0.24, 2026-07), MIT, 458★ | Nice API, but meant for UI and state. Value-level, no heights, no batch semantics |
| calyxos / pyinc / revlet | Small Salsa/Incremental-inspired memoization graphs | Tiny projects (pyinc 4★, revlet v0.1.0 released 2026-10) | Confirm the pattern is easy in Python. Too young to depend on |
| **qlib** (Microsoft) | Quant ML platform. Its expression engine uses strings such as `Ref($close, 5)/$close`, with an operator library and a cached feature store | Active, MIT. Last PyPI release 0.9.7 (2025-08) | Good **operator vocabulary** to copy (Alpha158/Alpha360 sets). The string DSL, the binary data format and the platform are too much |
| **vectorbt** | Fast vectorized backtesting (numba) | Active, **Apache-2.0 + Commons Clause** | Fine for personal research. The Commons Clause forbids *selling* it, so keep it out of the core dependencies |
| **alphalens-reloaded** | Factor tear sheets: IC, quantile returns, turnover | Active (0.5.1, 2026-10), Apache-2.0 | **Use for factor evaluation** on day one. It takes the (date, asset) long format we'll emit |
| array-api-compat | One array code path across numpy, torch, cupy and jax | Active, MIT | **The backend portability layer** |

### 1.3 Can we "plug in" Incremental? Should we?

**Technically yes, practically no.**
- OCaml can be embedded through `pyml` or a C-stub shared library, or it could run as a sidecar process.
- Either way, Python callables have to cross the boundary for every node, or our numeric kernels would need rewriting in OCaml. Two runtimes, two GCs and a serialization layer on every edge would all be paid for a scheduler that's about 400 lines.

**Why it buys nothing here.** Incremental is built for *fine-grained* graphs: millions of tiny nodes where per-node overhead in nanoseconds matters (UIs, spreadsheet-style P&L trees). Our graph has about 10²–10⁴ nodes, and each one is a vectorized kernel over a T×N array taking 10 µs to 10 ms. Python scheduling overhead (about 1 µs per node) is noise by comparison.

**What we copy.** These are the semantics that matter:

| Incremental concept | Our equivalent |
|---|---|
| `Var.set` | A data source appends a new row (a new bar date, a newly accepted filing, a new FRED vintage) or announces a correction for time *t₀* |
| Height and recompute heap | Each term's height is 1 + the maximum height of its inputs. `stabilize()` pops dirty nodes in increasing height and recomputes each only once |
| Cutoff | Per term: skip propagation if the new tail equals the old (`array_equal`, or `allclose(atol)` for floats). Examples: a quarterly fundamental that didn't change on a daily tick, or a filter whose membership didn't change. Cutoffs on filters and classifiers save the most work, because masks feed everything downstream |
| Observers and necessity | Only terms reachable from an observed output (pipeline columns, a dashboard panel, a screener) are kept live. Unobserved terms are dropped. This lets the dashboard register and unregister panels cheaply |
| `on_update` handlers | Observers push to QuestDB, Streamlit or alerts after a stabilization in which their value changed |
| `bind` (dynamic graph) | Restricted to *universe changes*. When the universe adds or removes assets, the asset axis is remapped. We don't support general `bind`. Pipeline graphs are static, which keeps things simple and testable |

Our version adds two things Incremental doesn't have:
1. **Time-axis incrementality.** On an append (a *seal*), a windowed term recomputes only the new tail row, using its last `window_length − 1` sealed rows of input. Cross-sectional ops (rank, zscore) recompute only the new cross-section. Between seals, **open-bar revisions** recompute the same last row from provisional inputs without committing any state (section 11.2).
2. **Retroactive corrections.** A restatement or data revision at *t₀* invalidates `[t₀, now]` for downstream terms. This is a suffix batch recompute that reuses the batch executor, so behavior stays the same.

---

## 2. Recommended architecture

### 2.1 Layers

```
                ┌─────────────── factorflow (standalone, no humdesk imports) ───────────────┐
 user code ───▶ │ expr API     Factor / Filter / Classifier, CustomFactor, Pipeline(columns, screen)│
                │ graph        immutable hashed Terms → TermGraph (CSE, heights, extra-rows calc)  │
                │ planner      lookback per input, mask propagation, dtype/missing rules           │
                │ kernels      array-API ops: rolling, ewm, rank, zscore, demean, winsorize, ols…   │
                │ executors    BatchExecutor (date range)  |  IncrementalExecutor (stabilize())     │
                │ loaders      Loader protocol: (columns, dates, assets, as_of) → dense arrays      │
                │ pit          bitemporal helpers: knowledge_time, as-of selection, TTM builders   │
                └───────────────────────────────────────────────────────────────────────────────┘
                         ▲ loaders implemented in humdesk (IB bars, EDGAR, FRED, crypto)
 humdesk ──────▶ humdesk.factors: data adapters, factor library, risk model, QuestDB sinks, Streamlit page, systemd timers
```

### 2.2 Expression API (Pipeline-style, modernized)

```python
from factorflow import Pipeline, CustomFactor, inputs as I
from factorflow.library import Returns, AverageDollarVolume, Fundamental as F

universe = AverageDollarVolume(window_length=63).top(100)  # Filter
sector = I.Equity.sic.to_classifier("ff12")  # Classifier
momentum = Returns(window_length=252).shift(21)  # 12-1 momentum
value = F.book_equity / I.Equity.market_cap  # PIT-aligned automatically
quality = F.gross_profit_ttm / F.total_assets_avg
composite = (
    momentum.zscore(groupby=sector, mask=universe)
    + value.winsorize(0.01, 0.99).zscore(groupby=sector, mask=universe)
    + quality.rank(mask=universe).zscore()
)

pipe = Pipeline(
    columns={"mom": momentum, "value": value, "score": composite, "sector": sector},
    screen=universe & composite.notnull(),
)

df = engine.run(pipe, start="2015-01-01", end="2026-10-30")  # batch: long (date, asset) Polars
live = engine.live(pipe)
live.on_update(sink_to_questdb)  # incremental: stabilize on new data
```

Rules:
- **Term types.** `Factor` (float), `Filter` (bool) and `Classifier` (int or categorical codes). All of them are immutable dataclasses hashed by `(type, params, inputs)`.
- **Operators.**
  - arithmetic and comparison
  - `rank/zscore/demean/winsorize/quantiles(groupby=, mask=)`
  - `shift`, `delta`, `pct_change`
  - `rolling_{mean,std,sum,min,max,corr,cov,beta,skew}`
  - `ewm`
  - `ts_rank`
  - `fillna/ffill(limit=)`
  - `top/bottom/percentile_between`
  - `isnull/notnull`
  - `where`
  
  The vocabulary follows qlib's Alpha158 operators and WorldQuant's 101 alphas.
- **`CustomFactor`.** A user `compute(out, *windows)` like Pipeline. It's optionally marked `@incremental` with an `update(state, new_rows)` method for O(1) rolling updates such as Welford variance or EWMA.
- **Data axis.** The canonical layout is **dense `T×N` arrays plus a validity mask**, where T is the bar grid of the pipeline (daily *or* minute) and the last row may be the open, provisional bar. It's the same model as Pipeline and Spectre, it's trivial to move to GPU, and append on T is natural. 500 assets × 5,000 days × float64 is 20 MB per term, which is fine for the PoC. Output is converted to long `(date, asset, column)` Polars for storage, alphalens and screens.
- **Calendars and grid.** A pluggable `Calendar` (XNYS sessions from `exchange_calendars`, including half days; 24/7 UTC for crypto) combined with a **frequency** gives the `BarGrid`. Bars are stored with both start and end times, and the engine labels each row by its **end** time. Slower inputs (daily factors or fundamentals inside a minute pipeline) are joined as of knowledge time, so the value used in a minute bar is the one known at that bar's end. Windows are counted in bars, and helpers translate durations (`days(20)`, `halflife('2h')`) per grid. Multi-calendar pipelines align with explicit `ffill(limit=)`. Crypto runs as its own pipeline by default.

### 2.3 Point-in-time correctness

- Every loader returns data **as known at the end of session *t***. The core question is always "what did the world know when this value was computed?"
- **Filings.** Knowledge time comes from the SEC **`acceptanceDateTime`** (submissions API, joined by accession number). It's more precise than the date-only `filed` field in companyfacts. The default availability rule is "usable from the next session open after acceptance". A filing accepted after 16:00 ET counts from t+1, and a stricter t+1 lag is an option.
- **Restatements.** companyfacts repeats prior-period values in later filings, each with its own `accn`. We store all of them. An as-of read picks, for each `(cik, tag, period)`, the value with the latest knowledge time ≤ *t*. The original 10-Q value is used before the restating 10-K, and the restated value after it.
- **The XBRL frames API is not point in time.** It returns the "last filed" value for each period. Use it only for live snapshots and quick looks, never in backtests.
- **Macro.** ALFRED vintages (`realtime_start/realtime_end`) are stored as `(series, obs_date, vintage_start, vintage_end, value)`. Backtests read the vintage valid at *t*. GDP, payrolls and CPI revisions are big enough to matter.
- **Universe.** Membership is stored as dated intervals. The S&P 100 membership history isn't free, so the PoC universe is **survivorship-biased** (see 4.3).
- **Equivalence test, the central invariant.** For random pipelines and random data, `BatchExecutor.run(0..T)` must equal folding `IncrementalExecutor` over the same data appended one bar at a time, including injected corrections. This runs in CI under hypothesis on both daily and minute grids.
  - **Sealed rows** must be bit-exact with batch.
  - **Open-bar revisions:** any revision of row *t* must equal a batch run over rows ≤ *t* in which row *t* holds the current provisional values. The test feeds random sequences of ticks inside each bar (section 11.5).

### 2.4 Incremental executor

- `live = engine.live(pipeline)` builds the necessary subgraph and warms it up with a batch run over the required lookback (the maximum extra rows).
- State per term:
  - a ring buffer of its last `extra_rows + 1` output rows,
  - an optional kernel state, such as Welford or EWMA accumulators,
  - its last output, used for cutoffs.
- `live.tick(asset, fields)` updates the open bar of one asset (close, high/low, volume, bid/ask) and marks that column dirty. `live.revise()` recomputes the last row only, for dirty columns where the term is time-series and for the whole row where it is cross-sectional. It commits no state. Revisions are coalesced and throttled (for example at most every 250 ms–1 s).
- `live.seal(t)` closes the bar. It commits each term's state with the final row values, appends the row, and opens the next bar. It is the only operation that advances time.
- `live.push(source, rows)` appends whole sealed rows (daily bars, a FRED vintage). `live.correct(source, t0)` marks a retroactive correction.
- `live.stabilize()`:
  - For appends, it pops dirty terms by height, computes only the new rows, applies cutoffs and enqueues the parents that changed.
  - For corrections, it finds the affected downstream set and calls the batch executor over `[t0 − lookback, now]`, then splices the result into the buffers.
- **Triggers in humdesk.** Each of these calls `stabilize()` on the live factor sets and writes the observers out:
  - a new daily bar after the close (IB historical / crypto),
  - a new EDGAR acceptance (a submissions poll every few minutes during filing season, otherwise hourly),
  - a FRED release (the `series/updates` endpoint, or the release calendar).
- Minute pipelines use the same executor on a minute grid: ticks or IB `keepUpToDate` bar updates revise the open bar, and the minute boundary plus a short grace period seals it. See section 11.

### 2.5 Backends

- **numpy reference backend.** Always on, and deterministic.
- **torch backend.** Optional extra `[gpu]`. The same kernels run through `array_api_compat`. Rolling windows use `unfold`, as in Spectre. It pays off for large universes, minute data and Monte Carlo (it's useless at 100 names daily).
- **Numba (optional).** For a few kernels with awkward vectorization, such as rolling rank and rolling OLS. It isn't needed for open-bar revisions, which run on one row at a time (section 11.4).
- **Polars.** Not the compute backend for factors. It's the I/O and PIT-join engine. A Polars-expression backend was considered: it offers free query optimization, but no GPU path shared with torch, harder dense cross-sectional linear algebra, and awkward ring-buffer incrementality.

### 2.6 Storage, and how it fits humdesk's QuestDB

| Data | Store | Why |
|---|---|---|
| Live ticks, quotes, intraday bars, positions, fills | **QuestDB** (existing) | Already there. Fast time-series append and ASOF |
| Daily bars (equities, crypto) | **Parquet** (existing `BarStore`: `<data_dir>/bars/<size>/<SYM>.parquet`) + DuckDB | Already built. Portable and compact |
| EDGAR facts, bitemporal | Parquet, partitioned by `cik` bucket: `fundamentals/facts/…` plus `filings.parquet` (accn → acceptance time, form, items) | Read-mostly, wide history, as-of joins in DuckDB or Polars |
| FRED/ALFRED vintages | Parquet `macro/alfred/<series>.parquet` | Small. Bitemporal |
| Factor panels (research/backtest) | Parquet partitioned by `factor_set/year` | Bulk columnar, read by notebooks |
| **Latest factor values and risk outputs (dashboard)** | **QuestDB** `factor_values(ts, set, factor, asset, value)` partitioned by DAY, plus `risk_exposures` and `factor_returns` | Live dashboard reads, retention managed with TTL |

The core package depends only on numpy and polars, plus optionally pyarrow, duckdb and torch. QuestDB sinks live in humdesk.

### 2.7 Package layout and humdesk integration

```
factorflow/ (separate repo)            # own pyproject, Apache-2.0, zero humdesk imports
  src/factorflow/
    terms.py  graph.py  planner.py  calendar.py
    kernels/{array_api.py, rolling.py, cross_section.py, linalg.py}
    executors/{batch.py, incremental.py}
    loaders/{protocol.py, frames.py, parquet.py}      # generic only
    pit/{bitemporal.py, ttm.py}
    library/{price.py, fundamentals.py, technical.py} # generic factor definitions on abstract columns
    eval/{ic.py, quantiles.py}                         # thin alphalens-compatible output
  tests/ (hypothesis equivalence suite, golden tests vs zipline-reloaded on toy data)
src/humdesk/factors/
    loaders/{ib_bars.py, crypto_bars.py, edgar_pit.py, fred.py, french.py}
    sets.py (named factor sets) · risk/ (model) · sinks/questdb.py · cli.py
app/pages/Factors.py, Risk.py, Screener.py
nix: services.humdesk.factors.{enable, timers, sets, fredKeyFile}
```

- humdesk depends on it as a flake input (`inputs.factorflow.url = "github:humdings/factorflow"`) or a uv git dependency. The factorflow flake exposes `packages.<system>.default` and an overlay.
- Splitting it out later is a `git subtree split` plus a PyPI release.
- **License for open-sourcing: Apache-2.0**, which is compatible with borrowing from zipline-reloaded. Avoid GPL code from Spectre and the OSAP package. We can still *use* their data.

---

## 3. Data

### 3.1 SEC EDGAR (free, no key)

The rules:
- Declare a `User-Agent` with company and contact email. humdesk's config already enforces this (`sec_user_agent`).
- Keep total traffic **≤ 10 requests/second** across all machines. Going over gets the IP temporarily blocked. humdesk already has a rate limiter.
- Use `Accept-Encoding: gzip`.
- There's no CORS, which doesn't matter for us.

| Source | Use |
|---|---|
| `data.sec.gov/submissions/CIK##########.json` | Filing index with **`acceptanceDateTime`**, form, items (8-K item codes), SIC, tickers and exchanges |
| `data.sec.gov/api/xbrl/companyfacts/CIK…json` | Every standard-taxonomy fact (us-gaap, dei, srt) with `start/end/fy/fp/form/filed/accn/frame`. **The main PoC source**, at 1 request per company |
| `…/companyconcept/…` | Single tag, for debugging |
| `…/xbrl/frames/{tag}/{unit}/CY2025Q3I.json` | Cross-sectional snapshot (**not PIT**). For live use and sanity checks only |
| Bulk `companyfacts.zip`, `submissions.zip` (rebuilt nightly about 3 am ET) | Full universe in one download. Use them when the universe grows past about 1,000 names |
| **Financial Statement Data Sets** (quarterly ZIPs: SUB/NUM/TAG/PRE) | `sub.accepted` acceptance timestamps and *all* facts, including dimensional ones and custom tags. A fallback and cross-check for companyfacts |
| **Insider Transactions Data Sets** (quarterly `YYYYqN_form345.zip`) plus recent Form 4 XML | Insider buys and sells |
| **Form 13F data sets** (quarterly) | Institutional ownership and breadth |
| 8-K item codes (submissions `items`) | Event filters: 2.02 (earnings), 5.02 (officer changes), 1.01 (material agreements), 4.02 (non-reliance, a red flag) |
| Fails-to-deliver (bi-monthly) | A short-squeeze and settlement-stress feature |
| 10-K/10-Q text (later) | "Lazy Prices" text-change factor (Cohen, Malloy and Nguyen 2020) |

Mature parsers exist if we want them: `edgartools` (MIT, very active) for Form 4 and 13F XML. humdesk's own `EdgarClient` already covers companyfacts and submissions.

### 3.2 XBRL → PIT fundamentals (TTM logic)

- **Tag fallbacks.** Use an ordered candidate list for each concept, extending humdesk's `METRIC_TAGS` (for example `RevenueFromContractWithCustomerExcludingAssessedTax → Revenues → SalesRevenueNet`). Record which tag won, for audit.
- **Flows** (income statement, cash flow) are duration facts.
  - **Discrete quarters** are facts with an 80–100 day duration.
  - **Q4** is derived as FY − 9M YTD, since many filers report no standalone Q4. humdesk already does this.
  - **Cash-flow statements are YTD** in 10-Qs: Q2 = 6M − 3M and Q3 = 9M − 6M.
  - **TTM** is the sum of the last 4 discrete quarters available as of *t*. If any are missing, fall back to the latest FY plus the YTD difference.
  - **Never sum** per-share or count items (EPS, weighted shares). Average them, or take the latest.
- **Stocks** (balance sheet) are instant facts. Use the latest instant ≤ as-of. "Average assets" is the mean of the current and the 4-quarters-ago instant.
- **Shares** come from `dei:EntityCommonStockSharesOutstanding` (cover page, the freshest) or `us-gaap:CommonStockSharesOutstanding`. **Market cap** is the PIT price × the latest shares known at *t*.
- **Fiscal calendars.** Index by period `end`, not `fy/fp`. The `frame` field must not be relied on.
- **Quality flags.** Mark missing tags, implausible jumps (|Δ| > 10× with no 8-K), unit mismatches and negative book equity. Each factor gets a coverage metric.

### 3.3 Filing-derived factor list (v1 ≈ 40)

| Family | Factor (definition; key us-gaap tags) | Literature |
|---|---|---|
| **Value** | B/M = Equity / MktCap (`StockholdersEquity`) · E/P = NI_TTM / MktCap · S/P = Revenue_TTM / MktCap · CF/P = CFO_TTM / MktCap · EV/EBIT, EV/EBITDA (EV = MktCap + Debt − Cash; `LongTermDebt`, `DebtCurrent`, `CashAndCashEquivalentsAtCarryingValue`; D&A `DepreciationDepletionAndAmortization`) · FCF yield = (CFO − Capex)/MktCap · Dividend yield (`PaymentsOfDividends`) · Shareholder yield (+ `PaymentsForRepurchaseOfCommonStock` − issuance) | Fama–French 1992; Loughran–Wellman 2011 (EBITDA/EV) |
| **Profitability** | Gross profitability = GP_TTM / Assets (`GrossProfit` or Revenue − `CostOfRevenue`) · Operating profitability = (Rev − COGS − SG&A − Interest)/BookEq · ROE, ROA, ROIC · Cash-based OP | Novy-Marx 2013; FF5 RMW; Ball et al. 2016 |
| **Quality** | Gross margin level and stability (8-quarter std) · Earnings variability · Cash conversion CFO/NI · **Piotroski F-score** (9 binary signals) · Mohanram G-score (later) · QMJ-style composite | Piotroski 2000; Asness–Frazzini–Pedersen QMJ 2019 |
| **Investment** | Asset growth Δ log Assets YoY · Capex/Assets · ΔPP&E + ΔInventory / Assets · NOA (net operating assets)/Assets | Cooper–Gulen–Schill 2008; FF5 CMA; Hirshleifer et al. 2004 |
| **Accruals** | Sloan accruals = (NI − CFO)/avg Assets · Percent accruals · ΔWorking capital / Assets | Sloan 1996; Hafzalla et al. 2011 |
| **Leverage / solvency** | Debt/Equity, Debt/Assets, Net debt/EBITDA · Interest coverage EBIT/`InterestExpense` · **Altman Z** (WC/TA, RE/TA via `RetainedEarningsAccumulatedDeficit`, EBIT/TA, MV/TL, Sales/TA) · Ohlson O (later) · Current and quick ratio | Altman 1968; Campbell–Hilscher–Szilagyi 2008 |
| **Growth** | Revenue, EPS and GP YoY growth (TTM and quarterly YoY) · SUE (standardized unexpected earnings, seasonal random walk) · R&D/Sales, R&D/MktCap (`ResearchAndDevelopmentExpense`) | Bernard–Thomas 1989 (PEAD); Chan–Lakonishok–Sougiannis 2001 |
| **Manipulation risk** | **Beneish M-score** (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA; `AccountsReceivableNetCurrent`, `SellingGeneralAndAdministrativeExpense`, `PropertyPlantAndEquipmentNet`) | Beneish 1999 |
| **Issuance / payout** | Net share issuance Δ log split-adjusted shares (12m and 36m) · Composite equity issuance (Daniel–Titman) · Buyback yield · Dividend initiation and omission (8-K or tags) | Pontiff–Woodgate 2008; Daniel–Titman 2006 |
| **Insider (Form 4)** | Net insider purchase ratio (90d, 180d) · # distinct buyers · **Opportunistic vs routine** trades · Cluster buys · Officer vs director weighting | Lakonishok–Lee 2001; Cohen–Malloy–Pomorski 2012 |
| **Institutional (13F)** | Ownership % · Breadth change (# holders Δ) · Concentration (HHI) | Chen–Hong–Stein 2002 |
| **Event / filters** | Days since last 10-Q or 10-K · Late-filer flag (NT 10-K/Q) · 8-K 4.02 non-reliance · Auditor change (8-K 4.01) · Earnings-window filter (8-K 2.02 ± k days) | — |
| **Classifiers** | SIC → Fama–French 12/49 industries (mapping from the Ken French library) · Exchange · Size and B/M buckets | GICS is proprietary, so use SIC/FF |

Price-based companions, computed from bars rather than filings:
- size (log MktCap)
- momentum 12-1 and 6-1
- short-term reversal 1m
- residual momentum
- realized vol and idiosyncratic vol (Ang et al. 2006)
- beta (rolling 252d vs SPY, Frazzini–Pedersen BAB-style shrinkage)
- Amihud illiquidity
- 52-week-high proximity (George–Hwang)
- max daily return (Bali et al.)
- dollar volume

### 3.4 FRED (and ALFRED for point in time)

**Access.** A free API key is required, and the limit is **120 requests/minute** per key, so throttle to about 1 request a second. Point-in-time history is read with `realtime_start/realtime_end` or `vintage_dates` on the same endpoints. The `fred/series/updates` and `releases/dates` endpoints drive refreshes. We'll call the REST API directly with httpx. `fredapi` (last release 2024) isn't needed.

| Theme | Series IDs (FRED) |
|---|---|
| Money and Fed balance sheet | `M2SL`, `M2REAL`, `M2V` (velocity), `BOGMBASE`, `WALCL` (Fed total assets), `WRESBAL` (reserves), `RRPONTSYD` (ON RRP), `WTREGEN` (TGA), `WSHOMCB` (MBS held) |
| **Net liquidity** (derived) | `WALCL − WTREGEN − RRPONTSYD`. **Units differ** (millions vs billions), so normalize. Weekly, with Wednesday-level vs average subtleties |
| Policy and short rates | `DFF`, `EFFR`, `SOFR`, `IORB`, `DTB3`, `DGS3MO` |
| Curve | `DGS1, DGS2, DGS5, DGS10, DGS30`, `T10Y2Y`, `T10Y3M`, term premium `THREEFYTP10` (Kim–Wright) |
| Real rates and inflation expectations | `DFII5`, `DFII10`, `T5YIE`, `T10YIE`, `T5YIFR`, `EXPINF1YR` (Cleveland Fed), `MICH` (UMich 1y expected) |
| Credit | `BAA10Y`, `AAA10Y`, `BAA`/`AAA` (Moody's, full history). ICE BofA OAS (`BAMLH0A0HYM2`, `BAMLC0A0CM`, `BAMLH0A3HYC`) **now has only a rolling ~3-year history on FRED**, so use the Moody's series for long backtests |
| Financial conditions and stress | `NFCI`, `ANFCI` (Chicago Fed), `STLFSI4` (St. Louis), **`VIXCLS`** (VIX close, which covers the VIX-index gap in his IB feeds), `VXVCLS` |
| Inflation | `CPIAUCSL`, `CPILFESL`, `PCEPI`, `PCEPILFE`, `PPIACO`, `CES0500000003` (avg hourly earnings) |
| Labor | `UNRATE`, `PAYEMS`, `ICSA`, `CCSA`, `JTSJOL`, `CIVPART`, `SAHMREALTIME` |
| Activity | `INDPRO`, `GDPC1`, `GDPNOW` (Atlanta Fed nowcast), `RSAFS`, `CFNAI`, `HOUST`, `PERMIT`, `UMCSENT`, `TOTALSA` (vehicle sales) |
| Recession | `USREC` (NBER, *ex post*, never a feature), `RECPROUSM156N` |
| Dollar, FX, commodities | `DTWEXBGS` (broad dollar), `DEXUSEU`, `DEXJPUS`, `DCOILWTICO`, `DHHNGSP` |
| Housing and credit growth | `CSUSHPINSA`, `MORTGAGE30US`, `TOTBKCR`, `BUSLOANS`, `DRTSCILM` (SLOOS tightening) |

Macro theses to model:
- the liquidity impulse (Δ net liquidity) vs BTC and Nasdaq
- curve un-inversion as a recession clock
- real rates vs gold and BTC
- credit spreads vs equity drawdowns
- the inflation surprise regime vs factor returns (value vs growth)
- M2 growth vs BTC with lags
- financial-conditions regime switching for risk sizing

### 3.5 Other free datasets common in the literature

Check each source's terms before redistributing anything. For our use, "commercial" means the LLC trades on it, which matters for **NC** licenses.

| Dataset | Content | Access / license |
|---|---|---|
| **Ken French Data Library** | FF3/FF5, momentum, ST/LT reversal, industry portfolios, breakpoints, **SIC→industry maps** | Free CSV/zip, no formal license, cite. **Primary validation target** |
| **AQR datasets** | QMJ, BAB, HML-Devil, Time-series momentum, Century of factor premia, Value and momentum everywhere | Free download (registration may be needed). AQR terms of use apply: research use, no redistribution |
| **Open Source Asset Pricing** (Chen–Zimmermann) | 200+ replicated anomaly signals and portfolios (Oct 2026 release), Parquet on Hugging Face | Free. `openassetpricing` package is GPL-2 (use the data, not the code). Firm-level signals need CRSP/Compustat (WRDS) |
| **Global Factor Data (JKP)** | 153 factors × 93 countries, factor-level returns | Factor portfolios are free. Stock-level data needs WRDS. Data CC BY-NC 4.0, code MIT |
| **Hou–Xue–Zhang q-factors** (global-q.org) | q⁵ factor returns | Free, cite |
| **Stambaugh–Yuan** mispricing factors, **Pastor–Stambaugh** liquidity | Factor returns | Free on the authors' sites |
| **Robert Shiller** | CAPE, S&P price, dividends, earnings since 1871 | Free (Yale) |
| **Welch–Goyal predictors** | The equity-premium predictor set (dp, ep, bm, tbl, tms, dfy, ntis, svar…), updated yearly | Free (Amit Goyal's site), cite |
| **Damodaran** | Implied ERP, industry multiples, cost of capital | Free (NYU) |
| **Baker–Wurgler** sentiment, **EPU** (Baker–Bloom–Davis), **GPR** (Caldara–Iacoviello) | Sentiment and uncertainty indices | Free, cite |
| **NY Fed** | ACM term premia, GSCPI (supply chain), recession probability, SOFR/EFFR, primary dealer stats | Free |
| **Philadelphia Fed** | Real-Time Data Set (vintages), Survey of Professional Forecasters, ADS business conditions | Free. Complements ALFRED |
| **Federal Reserve Board** DDP | H.4.1, H.8, H.15, Z.1 Financial Accounts, SLOOS | Public domain |
| **US Treasury FiscalData** | Daily Treasury Statement (TGA daily), yield curve, auctions, debt | Public domain, no key |
| **BLS** / **BEA** | CPI, employment / GDP, PCE, NIPA | Public domain. Free API keys (BLS v2, BEA) |
| **EIA** | Energy supply and inventories | Public domain, free key |
| **CFTC Commitments of Traders** | Positioning in futures, including CME BTC/ETH, ES, VX, rates | Public domain, Socrata API |
| **FINRA** | Margin statistics (leverage), short interest (bi-monthly) | Free |
| **Cboe** | VIX, VIX9D/VIX3M/VVIX/SKEW history CSVs, put/call ratios | Free download for personal use |
| **BIS** | Credit-to-GDP gaps, global liquidity, property prices, policy rates, FX | Free, SDMX API, attribution |
| **OECD** | CLIs (leading indicators), MEI | Free, **CC BY 4.0**, SDMX API |
| **IMF** | IFS, WEO, COFER (reserves) | Free, SDMX API, attribution |
| **World Bank** | WDI | Free, **CC BY 4.0** |
| **Crypto: Coin Metrics Community** | Network data (active addresses, fees, supply, MVRV inputs), reference rates | Free, no key, about 10 req/6 s. **CC BY-NC 4.0** |
| **mempool.space** | Bitcoin fees, mempool, blocks, hashrate, difficulty | Public API, rate-limited. Open source (AGPL), so self-hosting is possible |
| **Blockchain.com charts / Bitcoin Core node** | Hashrate, UTXO-derived stats | Free API. A pruned own node (about 10 GB) gives trustless live data but no deep history index |
| **DefiLlama** | Stablecoin supply (a crypto liquidity proxy), TVL | Free API, open data |
| **Exchange APIs** (Coinbase, Kraken, Hyperliquid, already in humdesk) | OHLCV, funding, OI, basis | Free public endpoints |

---

## 4. Proof-of-concept universe and storage

### 4.1 Universe
- **Equities v0:** the **S&P 100** (about 100 names; current constituents from public lists), plus SPY, QQQ, IWM and sector SPDRs (XLK…XLU) as benchmarks and hedges.
- **Equities v1:** the S&P 500 or a rule-based "top 500 US common stocks by 63-day dollar volume with XBRL filings".
- **Crypto:**
  - **BTC** (primary), **ETH**, **SOL**, plus optionally two of LINK, XRP, LTC and BCH. No memecoins.
  - Daily and hourly bars from Coinbase and Kraken public candles, plus Hyperliquid funding and OI (already collected).
  - ETF proxies: IBIT and ETHA.

### 4.2 Price sources (free or already paid)
- **IB historical bars** (`reqHistoricalData` through humdesk's existing `historical_bars`):
  - Daily `TRADES` or `ADJUSTED_LAST`, up to about 20 years per request.
  - Pacing is about **60 requests per 10 minutes**, with no identical request within 15 seconds. That makes ~120 symbols a 20-minute backfill and 500 symbols about 90 minutes, then 1 request per symbol per day.
  - Historical bars are not snapshots and are not charged. **We never call snapshot market data.**
  - *To verify:* whether his US subscriptions (non-consolidated Cboe One/IEX plus OPRA) return consolidated `TRADES` history, or only the venue's trades. Daily OHLC is usually fine either way.
- **Fallbacks:**
  - Stooq daily CSVs (free, no key)
  - Tiingo free tier (key, EOD adjusted, generous for 500 names)
  - Corporate actions from SEC (splits via `StockholdersEquityNoteStockSplitConversionRatio1`) and IB `ADJUSTED_LAST`
- **Crypto:** exchange public REST, with no key needed.

### 4.3 Known biases (accepted for the PoC, documented)
- **Survivorship bias.** We use today's S&P 100, and IB doesn't serve delisted symbols' history. A fix would be to reconstruct S&P 500 membership from the public changes history (approximate) and source delisted prices from EDGAR-era tickers plus Stooq. That's phase 6 or later.
- **Free data quality.** The Ken French and OSAP replications are the sanity check. Our HML/RMW/CMA/UMD on 100–500 large caps should be *positively correlated* with French's (expect ρ ≈ 0.5–0.8, given a large-cap-only universe).

### 4.4 Storage estimate

| Item | PoC (S&P 100 + 5 crypto, 20y daily) | v1 (500 names) |
|---|---|---|
| Daily bars (Parquet) | ~10 MB | ~50 MB |
| Crypto hourly bars | ~50 MB | same |
| companyfacts raw JSON cache | ~0.5 GB | ~2.5 GB (raw cache can be deleted after parsing) |
| Tidy bitemporal facts (Parquet, zstd) | ~50 MB | ~250 MB |
| Form 4 / 13F extracts | ~50 MB | ~300 MB |
| FRED + ALFRED vintages (~80 series) | ~50 MB | same |
| Ken French / AQR / OSAP portfolios | ~200 MB | same |
| Factor panels (100 factors × N × 5,000 days, float32) | ~200 MB | ~1 GB |
| QuestDB `factor_values` (latest 2y daily) | ~100 MB | ~500 MB |
| **Total** | **≈ 1–1.5 GB** | **≈ 5 GB** |

Minute bars would dominate. The revised estimate is in section 11.4: about 0.5–1 GB a year for 500 names in Parquet (zstd), and about 0.1–0.2 GB a year for the S&P 100. Live minute data for 100+ names is limited by IB line budgets more than by storage (section 11.6).

---

## 5. Risk model plan

### 5.1 Fundamental factor model (Barra-style, USE4-inspired)
- **Factors:**
  - Market (intercept)
  - **Industries:** FF12 to begin with (more would be too thin at 100 names), FF49 at 500
  - **Styles**, each a cap-weighted standardized z-score winsorized at ±3: Size, Beta, Momentum, Residual Vol, Value (B/P, E/P blend), Earnings Yield, Growth, Leverage, Liquidity, Profitability/Quality, Investment, Dividend Yield
  - **Crypto** is a separate block: BTC beta, ETH/BTC, funding/basis, liquidity
- **Estimation.** Each day, a cross-sectional WLS of returns on exposures (weights √MktCap). The industry dummy collinearity is handled with a cap-weighted sum-to-zero constraint, which gives daily factor returns **f** and residuals **ε**.
- **Factor covariance.**
  - EWMA, with a half-life of about 90 days for vol and 180–480 days for correlation
  - Newey–West adjustment for serial correlation (2–5 lags)
  - eigenfactor risk adjustment (Menchero), to fix the underestimation of risk in optimized portfolios
  - volatility regime adjustment (cross-sectional bias statistic)
- **Specific risk.** EWMA of residuals, with **Bayesian shrinkage** toward size-decile means and a structural model for thin histories.
- **Shrinkage.** Ledoit–Wolf on factor covariance as an option, and bias statistics as the acceptance test (rolling bias statistic ≈ 1 out of sample).
- **Macro overlay.**
  - Time-series betas of each asset and style factor to macro factors (Δ10y, Δcurve, Δcredit spread, Δbreakeven, Δdollar, Δnet liquidity, oil, VIX), estimated rolling with shrinkage.
  - These are reported as *exposures and stress sensitivities*, not as part of the core covariance, so the model stays well-conditioned.
  - A statistical (PCA) model runs alongside as a check.

### 5.2 Dashboard (new Streamlit pages)
- **Portfolio risk.** Reads live positions from QuestDB.
  - Total, factor and specific vol
  - Risk decomposition by factor and industry
  - Marginal and component contribution to risk
  - Ex-ante tracking error vs SPY
  - Exposures over time
  - Crypto shown separately and combined
- **Factor monitor.**
  - Factor returns (day, week, month, YTD), cumulative charts and drawdowns
  - Factor vol regime and correlation heatmap
  - Crowding proxies (valuation spread of long-short legs)
- **Macro panel.**
  - Net liquidity, curve, real rates, credit, NFCI, VIX
  - Current regime classification and surprise vs consensus where available
- **Screener.** Pipeline-driven screens (for example "quality value: F-score ≥ 7, EV/EBIT bottom 30% within industry, insider net buying"), with saved screens, change alerts (deduplicated as in the arb alerts) and drill-down to filings.
- **Factor lab.** IC/IR, quantile spreads, turnover and decay for any term, using alphalens-reloaded tear sheets in JupyterLab.
- **Data health.** Coverage per factor, stale sources, filing lag and last stabilize times.

### 5.3 Simulations
- **Factor Monte Carlo.** Draw from the factor covariance (Gaussian and Student-t, or a filtered historical bootstrap) plus specific risk, to get VaR/ES of the current book. This can use the torch backend.
- **Historical scenario replay.** 2008, Mar-2020, 2022 rates, Aug-2024 yen carry and crypto crashes, applied through today's exposures.
- **Macro shock propagation.** "+100 bp 10y", "+200 bp HY spread", "−20% net liquidity impulse" via macro betas.
- **Strategy backtests.** Feed pipeline outputs (weights or signals) into humdesk's backtest engine and vectorbt for sweeps. Costs come from humdesk's `fees.json`.

---

## 6. Phased plan

Each phase ends with a demo and tests. Nothing goes to main without David's approval.

| Phase | Build | Exit criteria |
|---|---|---|
| **P0 Decisions** | David answers the open questions below. Choose the name. Provide the FRED key | Design signed off |
| **P1 Engine core** | `terms/graph/planner`, numpy kernels (≈30 ops), `BatchExecutor`, `Calendar` + `BarGrid` (daily and minute), Pipeline API, long-format output. **Kernels are written as `step/peek/seal` from day one** (section 11.2), so P4 adds no new kernel code | Golden tests match zipline-reloaded Pipeline on toy data. 100-name × 20-year pipeline with 30 terms in < 5 s on belter |
| **P2 Data loaders (humdesk)** | IB daily-bar backfill for S&P 100 + ETFs, crypto bars, **EDGAR PIT facts** (companyfacts + acceptance times + TTM builder), FRED/ALFRED loader, Ken French loader | PIT audit: for 10 hand-picked filings, the as-of values match the filing documents on the acceptance date and the restated values after the restatement |
| **P3 Factor library v1 + evaluation** | About 40 filing factors + 12 price factors + classifiers, alphalens tear sheets, notebook | Our HML/RMW/CMA/UMD correlate with French's. F-score, Z and M scores spot-checked against published examples |
| **P4 Incremental live mode** | `IncrementalExecutor` (heights, cutoffs, observers, corrections, **open-bar revise/seal**), systemd timers (post-close bars, EDGAR poll, FRED updates), QuestDB sink, **Factors** dashboard page | Hypothesis equivalence suite passes (incremental == batch, including corrections). A daily stabilize after the close takes < 2 s |
| **P5 Risk model v1** | Style + FF12 industry model, EWMA/NW covariance, specific risk with shrinkage, macro betas, **Risk** page on live positions | Out-of-sample bias statistics within [0.8, 1.2] for test portfolios |
| **P6 Screeners, simulations, scale** | Screener page and alerts, Monte Carlo and scenarios, torch backend, S&P 500 universe, survivorship fixes, Form 4/13F factors | Screens with saved alerts. GPU path gives the same results as numpy (allclose) |
| **P7 Open-source split (optional)** | Docs, examples on public data only (French and EDGAR), PyPI release | Clean `pip install factorflow` with no humdesk dependency |

## 7. Open questions for David

1. ~~Name~~: decided, `factorflow`.
2. **Repo**: start as a uv workspace member inside humdesk (recommended), or as a separate repo from day one?
3. **Universe**: S&P 100 to begin with, then S&P 500 (recommended)? Or a rule-based liquid-500 from the start? Which 2 extra crypto majors besides BTC/ETH/SOL?
4. **History depth**: 20 years (XBRL starts around 2009–2011 for most large caps, so pre-2010 fundamentals would need the FSDS or text) or 10 years?
5. **Update cadence**: daily after the close is the PoC plan. The engine is now designed for minute grids from P1 (section 11), but *live* minute data needs a source decision (section 11.8).
6. **belter resources**: free disk for `data_dir`, RAM, and **whether belter has an NVIDIA GPU** (this decides whether the torch backend is worth building early).
7. **NC licenses**: Coin Metrics Community and the JKP data are CC BY-NC 4.0, and trading through the LLC is arguably commercial use. Should we skip them and use exchange, mempool and DefiLlama data, or use them only for research notebooks?
8. **Open-source intent**: is Apache-2.0 OK? Should anything (your factor definitions or the risk model) stay private in humdesk?
9. **Survivorship**: is a survivorship-biased PoC acceptable for now?

## 8. What David needs to provide

- **FRED API key** (free from fredaccount.stlouisfed.org). Put it on belter as a line `FRED_API_KEY=…` in the existing root-only env file `/etc/humdesk/humdesk.env` (root:root 0600), or in a separate `/etc/humdesk/fred.env` passed with `LoadCredential`. **Never commit it to git or put it in the Nix store, and don't paste it in chat.** The service reads it from the environment or credentials at runtime.
- Optional free keys, only if we use those sources: BLS, BEA, EIA, Tiingo. They're handled the same way.
- No SEC key is needed. The existing `sec_user_agent` (Rocinante Research contact email) is used.
- Confirmation that the IB historical-bar backfill (no snapshots) can run against the live read-only Gateway outside market hours. The pacing-limited backfill takes 20–90 minutes once.
- Answers to section 7.

## 9. Key risks and trade-offs

- **Building vs adopting.** We own about 3–5k lines of engine code. In return we get PIT correctness, the incremental/batch guarantee, and no dependency hell, all of which are the reasons Spectre and zipline disappointed.
- **Dense T×N arrays** are simple and GPU-friendly, but their memory grows with universe × history. That's fine up to about 3,000 names daily. Beyond that, chunk by date and go out-of-core through Parquet.
- **Static graphs (no general `bind`)** give up some of Incremental's flexibility in exchange for predictability and testability. Dynamic universes are handled through masks.
- **Free fundamentals (XBRL)** are noisier than Compustat because of tag drift, custom tags and missing Q4s. Mitigations are tag fallbacks, the FSDS cross-check, quality flags and coverage metrics.
- **Accepted PoC biases**: survivorship, a large-cap-only universe, and free price data quality.

---

## 10. Industry practice notes (Northfield reference docs, 2026-11-02)

David shared three Northfield Information Services manuals: Open Performance Attribution (about 2003), the Open Optimizer user manual (2015) and the Optimizer report formulae. My paraphrased notes are in `refs/northfield-notes.md`. They show what a mature vendor stack does. **We are not replicating it.** We borrow only these ideas.

**Borrow:**
1. **Separate the model from the analytics with a small, open file contract.** Their whole tool chain runs on four artifacts: factor covariance, exposures with specific risk, classifications, and per-period factor returns. Ours is a versioned set of Parquet files per model date: `exposures`, `factor_cov`, `specific_var`, `factor_returns`, plus `meta`. Attribution, risk, the optimizer, the dashboard and third-party models all read this contract, so a different model (statistical, macro, or a vendor's) can be plugged in.
2. **Factor types and groups as metadata.** Each factor is continuous or membership (discrete, fractional or hybrid), and factors belong to groups such as Market, Style, Industry, Macro and Crypto, which roll up into sectors. Every report aggregates by group. Fractional membership suits crypto (for example L1 vs DeFi exposure) and multi-segment companies.
3. **Standard math for the risk reports:**
   - the factor plus specific tracking variance split, with tracking error
   - per-factor contributed variance (own variance plus half the covariance terms, so the parts sum to the total)
   - R² vs the benchmark
   - marginal contribution per position, stating explicitly whether the offsetting asset is cash or the benchmark
   
   These are cheap and well understood.
4. **Attribution done both ways:** Brinson-style stratification (weight and selection impact by bucket or industry, which catches non-linear effects such as a U-shaped P/E payoff) **and** multivariate factor impact (active exposure × factor return). Over time, both get **impact t-stats** (mean / sd × √n) as the test of skill vs luck. This plugs straight into the arb and outcome-study style of analysis.
5. **Regression details:**
   - √cap weights
   - price ratios as reciprocals (E/P, B/P, S/P, CF/P) so nothing divides by zero
   - standardization over a reference universe with a size floor
   - **fundamental beta** (a Blume-shrunk time-series beta projected onto characteristics), so new listings get a beta from day one
6. **Optimizer and screen ideas for later:**
   - separate risk aversions for systematic and specific risk (λ_sys, λ_spec)
   - tracking-error targeting by rescaling λ
   - "reshaping" alphas: rank → normal score → α = IC × σ_cross × z (a Grinold–Kahn style scaling for screen scores)
   - transaction costs amortized by expected turnover
   - linear plus square-root impact with a days-to-trade knob
   - constraint bands that can be absolute, relative to initial or relative to benchmark, plus do-not-trade lists
   - covariance blending with simple targets (single-index, constant correlation)
   
   With **cvxpy** these are about 50 lines, not a product.
7. **Look-through composites.** ETFs (SPY, sector SPDRs, IBIT) are decomposed into their holdings for risk, so ETF hedges show up correctly against stock positions. Holdings come from issuer files or N-PORT filings at SEC.
8. **Exception and coverage reports** are a first-class output: unmatched symbols, missing exposures, stale data.

**Do differently, on purpose:**
- **Simultaneous constrained WLS, not sequential.** Their styles-first, then industries-on-residuals approach was a workaround for thin industries. We fit market, industries and styles jointly with a cap-weighted sum-to-zero industry constraint (Menchero/Barra style), and use coarse FF12 industries at small N so no industry is too thin.
- **EWMA, Newey–West and eigenfactor/vol-regime adjustments instead of a 60-month equal-weighted covariance.** We also add a short-horizon (daily) model alongside a monthly one. Their own optimizer needed a separate "near horizon" residual file to compensate.
- **Daily model, with monthly reporting views.** Their worry about daily attribution statistics is valid. We handle it with overlap-aware standard errors (Newey–West) and by reporting monthly aggregates, not by throwing daily data away.
- **Specific risk from EWMA residuals with Bayesian shrinkage toward size and industry groups**, instead of a 60-month standard deviation with an ad hoc serial-correlation bump.
- **Point in time and incremental throughout**, since our engine already provides it. Their monthly file drops can't do this.
- **A modern convex solver (cvxpy with OSQP or Clarabel), not greedy pairwise swaps.** Cardinality and round lots are handled as post-processing or with MIP only when needed.
- **Skip tax-lot optimization, multi-account aggregation and Excel I/O.** They don't apply to the LLC's current use.
- **Crypto and macro blocks** (BTC beta, funding/basis, net-liquidity and rate betas) are native factor groups, not a separate product.

---

## 11. Pre-coding research: one engine for daily, minute and live bars (2026-11-02)

**David's constraint:** keep it simple and avoid performance traps. The engine produces **one bar per time step on a fixed grid**. Ticks don't create events. They only update the most recent, still-open bar, and the engine recomputes that last row. When the bar closes it is sealed and the next bar starts. That rules out full event-time streaming: no watermarks, no per-asset irregular clocks, no out-of-order event handling. The rest of this section designs within that constraint.

### 11.1 Time model: a bar grid with one open bar

- **`BarGrid` = calendar × frequency.** Examples: `XNYS × 1d`, `XNYS × 1min` (390 bars per session, 210 on half days), `24/7 × 1min` for crypto, and `RTH × 1s` for arb.
  - A row is the bar `[start, end)`, labeled by `end`.
  - A pipeline has exactly one grid.
  - The same factor definition runs on any grid. Only window lengths, which are counted in bars, change meaning, and duration helpers (`days(20)`, `minutes(30)`, `halflife('3d')`) convert explicitly per grid.
- **Row states.** Every row except the last is **sealed** and immutable. The last row is **open**: its inputs are provisional and its outputs are provisional too, flagged `open=True` to observers and in QuestDB.
- **Tick → open bar.** A bar builder outside the engine core folds ticks into the open bar for each asset:
  - open is the first tick, high = max, low = min, close is the last tick
  - volume and VWAP are sums
  - bid, ask and sizes are last values, with `n_ticks` and `last_ts` alongside
  
  IB's `reqHistoricalData(keepUpToDate=True)` already delivers exactly this, repeated updates to the last unfinished bar, so for IB the builder can be a pass-through.
- **Seal.** At `end + grace` (a default of 2 s for minute bars; for daily bars, when the official daily bar arrives after the close), the row is finalized, every term commits its state, and a new open row is created.
  - **Assets with no tick** in a bar get `volume = 0` and the close carried forward (configurable per field), with the mask showing "no trade".
- **After a seal, nothing is revised.** A later difference from a vendor's official bar (for example IB's final daily bar vs the tick-built one) is a **correction**, handled by the existing suffix-recompute path (§2.4). It is not a revision.
- **Asynchrony without per-asset clocks.** Each asset's open bar simply holds its latest state. Staleness is exposed as data, `last_update_age` per asset, so filters and the arb engine can drop stale names. No watermark logic is needed.
- **Mixed frequencies.** Slower inputs are sampled onto the grid as of knowledge time: fundamentals as step functions, and daily factors inside a minute pipeline where yesterday's sealed value is used all day and today's daily bar is the open row of the daily pipeline. Faster-to-slower is aggregation in the bar builder only. There is no general multi-rate graph.

### 11.2 Kernels: `peek` for revisions, `seal` for commits

Each kernel has a small state over sealed rows and three operations:

| Op | Called on | Does |
|---|---|---|
| `warm(block)` | start-up | builds the state from the last W sealed rows |
| `peek(state, row) → out` | every revision | output for the open row. **Doesn't mutate the state** |
| `seal(state, row) → (state', out)` | bar close | commits the state |

The batch executor uses the same kernel definitions over the whole array, either vectorized or as a fold of `seal`. Typical `peek` costs, per revised column:

| Kernel | State over sealed rows | `peek` for the open row |
|---|---|---|
| rolling sum / mean | sum of the last W−1 rows plus a ring buffer | `S − x[t−W] + x_open`, O(1) |
| rolling var / std / cov / corr / beta | Welford moments of the last W−1 rows | Chan merge of the moments with the single provisional row, O(1). Subtract-on-evict is **re-anchored from the ring buffer at every seal**, so drift can't build up |
| EWMA (mean, var, cov) | `s[t−1]` | `α·x_open + (1−α)·s[t−1]`, O(1). Time-based α (`1−exp(−Δ/τ)`) for half-lives given as durations |
| rolling min / max | a monotonic deque over the sealed part | `max(deque.front, x_open)`, O(1) |
| ts_rank, rolling median / quantile | a sorted window (`sortedcontainers`, Apache-2.0) | bisect, O(log W) |
| rolling OLS / RLS | XᵀX and Xᵀy over sealed rows | add the provisional row and solve p×p, O(p³) with p small |
| cross-sectional rank / zscore / demean / winsorize (per group) | none | **recompute the full row**, O(N log N). That's under 1 ms in numpy for N ≤ 5,000, so order-statistic trees aren't needed |
| shift / delta / pct_change | ring buffer | O(1) |

Rules:
- **Exactness.** A sealed output must equal batch bit for bit. For kernels whose fast path isn't bit-stable (Welford vs a two-pass variance, for example), `seal` recomputes from the ring buffer with the *reference* kernel, which is the same function batch uses. `peek` may use the fast path, and its tests use `rtol=1e-12`.
- **Cutoff on floats.** By default a cutoff happens only on exact equality, comparing arrays with NaNs treated as equal. A revision whose open row is unchanged stops there. Tolerance cutoffs are opt-in per term, and terms that use them are excluded from bit-exact tests, because a held value would differ from batch.
- **Dirty columns.** A tick marks one asset's column dirty. Time-series terms `peek` only dirty columns. Cross-sectional terms recompute the full row. Everything downstream of a cross-sectional term is effectively full-row, which is fine at our N.
- **Fall back to recompute** for corrections older than the open row, universe changes, and a change in a term's parameters. These run the batch path over `[t0 − lookback, now]` and splice the result in. That's how it already works.
- **`CustomFactor`** gets the same protocol. If a user implements only `compute(window)`, the engine calls it on `window[:-1] + open_row` for `peek`, which is O(W) but always correct.

### 11.3 Prior art (checked 2026-11-02): what to reuse and what to learn from

| Project | Status / license | Take-away under the bar model |
|---|---|---|
| **talipp** | 2.7.0 (2025-09), MIT | **Closest match to the open-bar model.** Its indicators support "update the last input value" and "remove last" in O(1). **Learn the API, then decide: copy the pattern or add an optional adapter.** Its per-indicator, per-series object model doesn't fit dense T×N |
| **Point72 csp** | 0.19.1 (2026-10), Apache-2.0, C++ engine with Python nodes | Write once, run in simulation or in realtime. Engine cycles, `valid()` checks and baskets. It's tick-event based, which is more than we need. **Learn, don't adopt.** Revisit if we ever want event-time |
| **Nautilus Trader** | 2.0.0rc7 (2026-10), **LGPL-3.0**, Rust core | Bar aggregation from ticks, the indicator `handle_bar` / `handle_quote` split, and research-to-live parity. **Learn.** LGPL and a whole trading framework, so not a dependency |
| **Deephaven Community** | v42 (2026-07), **source-available DCLA**, JVM | Ticking tables that update incrementally and a real-time UI. **Learn** (its "last row modifies in place" behavior). License and JVM rule it out |
| **kdb+/KDB-X** | KDB-X Community is free with limits (16 GB RAM, 1 instance) | The bar-plus-tickerplant pattern: a realtime DB for today and a historical DB for sealed partitions. That's what our QuestDB + Parquet split already is. **Don't adopt** |
| **QuestDB materialized views** | `SAMPLE BY` views refresh incrementally, only the affected time slices | **Reuse** for server-side minute/OHLC bars from stored ticks, and for dashboard rollups. Not for factor logic |
| **Polars / DuckDB** | Polars 2.1, DuckDB 2.0 | **Reuse** for batch resampling (`group_by_dynamic`, `time_bucket`), as-of joins and the golden datasets |
| **river** | 0.26.1 (2026-08), BSD-3 | `river.stats` online and rolling statistics are a good reference for numerics. **Learn** |
| Feldera (MIT), RisingWave 3.0 (Apache-2.0), Arroyo 0.15 (Apache-2.0, now Cloudflare), Flink, Timely/Differential, Pathway (BSL), Bytewax (unmaintained) | active, except Bytewax | Event-time SQL streaming with watermarks, which **the bar model doesn't need**. Skip |
| streamz 0.6.6, reaktiv, Salsa, Adapton, Jane Street Incremental | — | Salsa/Adapton/Incremental confirm our design choices: heights, early cutoff, demand-driven necessity. Nothing to add as a dependency |
| LMAX Disruptor | pattern | **Adopt the principle:** one engine thread owns all state, IB callbacks only enqueue ticks, and the engine drains the queue per revision. No locks |
| Wingfoil (Rust, Python bindings) | young, little adoption | Watch only |

**Reuse as dependencies:** numpy, polars, duckdb, `exchange_calendars` (Apache-2.0, 4.13), `sortedcontainers` (Apache-2.0) and hypothesis. **Learn from:** talipp, csp, Nautilus and Deephaven. **Avoid:** any streaming runtime.

### 11.4 Performance and storage

- **Python overhead.** A numpy call on a small array costs about 1–5 µs, so a revision touching 200 terms costs about 0.5–2 ms plus the arithmetic. At 1–4 revisions per second that's negligible. With revisions coalesced (drain all pending ticks, then revise once), the cost per tick is about the cost per revision divided by the ticks per revision.
- **Targets:**

| Use | Grid | Revision cadence | Target | Notes |
|---|---|---|---|---|
| Daily factors and fundamentals | 1d | none, or hourly | seal of 500 names × 200 terms in < 2 s | already P4's exit criterion |
| Equity minute bars | 1min | ≤ 1/s | revise in < 50 ms and seal in < 200 ms for 500 × 100 terms | numpy only |
| Options arb | 1s | per drained queue | revise in < 20 ms p99 at about 400 ticks/s (90 lines × about 4 updates/s) | needs dirty-strike checks (11.5) |

- **Measured today** (box, scratch script, no repo change): `ArbEngine.evaluate` on 10 synthetic slices × 48 options takes **≈ 235 ms for the named checks alone (30k check results)** and about 270 ms including the graph and LP. One slice takes about 31 ms. A full re-evaluation per tick is therefore **too slow for per-tick revisions**, and recomputing only the dirty strikes is required.
- **Numba / Rust.** Not needed for M0–M2. Numba is optional for rolling rank and OLS in *batch*. Rust via PyO3 or maturin only if profiling shows a hot kernel we can't vectorize, and none is expected at these sizes. Free-threaded Python 3.14 isn't needed either, since the single engine thread is a deliberate choice.
- **Storage for minute data on belter (Parquet, zstd, OHLCV + VWAP + n):**

| Data | Rows per year | Size |
|---|---|---|
| S&P 100 minute bars | about 10M | ≈ 0.1–0.2 GB/yr |
| S&P 500 minute bars | about 49M | ≈ 0.5–1 GB/yr |
| 5 crypto majors, 1 min, 24/7 | about 2.6M | about 30 MB/yr |
| Minute factor panels (100 terms, float32) | — | about 4 GB/yr at 100 names, **so store only the observed outputs** |
| Arb 1 s bars | — | stored as today in QuestDB (quotes, violations) |

  Live open-bar state lives in memory: N × fields × 8 B, which is trivial.

### 11.5 Arb tie-in (assessed against humdesk `main` ba911dd)

**What exists:**
- `options/collector.py` streams IB quotes for 5 underlyings within a 90-line budget, rotating slices in batches (`dwell` 15 s), with follow lines sampled every `sample_s`.
- `arb/engine.py` evaluates pure functions over `Slice` objects:
  - named checks (`checks.py`: single, vertical, butterfly, parity, box, calendar)
  - the Bellman–Ford negative-cycle graph (`graph.py`)
  - the HiGHS LP (`lp.py`)
- `ViolationTracker` builds episodes, and `OutcomeRecorder` follows each violation at horizons and simulated fills, then labels it.

**Mapping to the bar model.** The quote snapshot is the open bar.
- Grid: `RTH × 1s` (or the dwell). Columns are **option contracts plus underlyings**. The open bar holds the latest bid, ask, sizes, `md_type` and `last_ts`. Sealing every second gives the tracker and the outcomes a clean time base.
- **Quotes** (open bar) → **executable legs** (per contract, elementwise) → **checks**. The checks are a *domain node*, not Pipeline expressions, because they're relational over the strike ladder:
  - a tick on contract (K, right, expiry) dirties only the checks touching K:
    - single and parity at K
    - verticals (K−1, K) and (K, K+1)
    - butterflies within the wing
    - box pairs (K, ·), which is O(K) instead of O(K²)
    - calendars at K across expiries
  - results are kept in a keyed cache per slice, so a revision replaces only those entries
  
  The measured cost suggests about 1–3 ms per tick, compared with about 31 ms per slice today.
- **Graph and LP** are whole-slice solves. They run **only at seal**, and only for dirty slices, or throttled (for example every 5 s per slice). They can be skipped when no pairwise slack near them is below a threshold.
- **`ViolationTracker` steps at seal.** Durations and episodes are then counted in sealed bars, so they're deterministic in replay. **Outcomes** become horizon marks in bars, using quotes observed in bar `anchor + h`. At 1 s bars this keeps today's tolerance semantics to within a second.
- **What this buys:**
  - per-tick detection on follow lines instead of once per batch
  - one consistent as-of quote state with explicit staleness
  - **a replay of stored QuestDB quotes through the same executor reproduces live results** (batch == live), which makes the outcome study reproducible
- **What stays separate:**
  - the IB collector: subscriptions, line budget, rotation and the no-snapshot guard
  - the alert I/O, as observers
  - the study's batch analytics
  - the solvers' internals
  
  Arb also doesn't depend on factorflow's factor library. It only uses the executor core.
- **Migration path (no rewrite now):**
  1. In humdesk, give the check functions a `dirty_strikes` argument that returns the affected subset. Test that full evaluation equals the union of the dirty-subset results. This is independent of factorflow and helps on its own.
  2. Add a keyed result cache per slice and feed the collector's follow-line samples into it per tick. Run the graph and LP at the end of the dwell as today.
  3. Once factorflow M2 exists, wrap this as a domain node on a 1 s grid and switch the tracker and outcomes to bar time.
  4. Replay QuestDB quotes through the same path for the study.

### 11.6 Other decisions to make before coding

- **IDs and security master.** An internal integer `asset_id` that never changes. A symbology table of `(asset_id, scheme, value, valid_from, valid_to)` for IB `conId`, ticker, CIK, CUSIP and optionally FIGI (OpenFIGI's API is free). Tickers are never keys.
- **Corporate actions.** Store raw prices plus an adjustment-factor table (splits, dividends) with knowledge time. Returns and windows use adjusted values computed as of *t*. A split sets a correction flag. Delisted assets stay in the master and are masked; history is never deleted.
- **Calendars and timezones.** Store UTC everywhere and use `exchange_calendars` for XNYS sessions and half days. Show New York time for sessions and HST for David's displays. Bars are labeled by **end** time; IB labels by start, so convert at the loader.
- **Point-in-time storage.** Every stored row has `knowledge_ts`. Sealed bars are immutable Parquet partitions; open bars exist only in memory and in QuestDB marked `open`.
- **Testing:**
  - hypothesis tests that incremental == batch at seal (bit-exact) and that a revision == batch over the provisional row, on daily and minute grids, with random tick sequences inside each bar
  - golden datasets as frozen Parquet fixtures: 10 names × 2 years daily, 5 days of minute data, and one recorded options chain day
  - tests against zipline-reloaded on toy data
- **API ergonomics and typing.** Use Python classes and operators, not a string DSL. `Factor`, `Filter` and `Classifier` are typed, and pyright runs in CI. A `str()` form can come later for saved screens.
- **Packaging.** A uv workspace member with numpy and polars as core dependencies, and optional extras `[duckdb]`, `[torch]` and `[numba]`.
- **Observability.** Per-term timings, revision and seal latency histograms, revisions per second, last seal time, dirty-column counts and staleness per source. These go to a QuestDB `engine_metrics` table and the Data health page.
- **Things we'd regret not deciding:**
  - Bar labeling convention (end vs start).
  - Missing-bar policy (forward-fill close, zero volume).
  - Seal grace and what happens to a tick after the seal: it goes into the next bar, by receive time.
  - Whether the dashboard ever shows open-row values for factors (recommended: yes, flagged).
  - The float dtype (float64 in the engine, float32 at rest).
  - The **live minute-data source** (see 11.8).

### 11.7 Ready to code?

**Yes, once David answers the decisions below.** The first milestone changes to:

- **M0: engine core on a bar grid** (numpy only, no humdesk integration and no IB):
  - `Term`, `Graph` and `Planner`
  - `BarGrid` for daily and minute grids
  - about 15 kernels with `warm/peek/seal`
  - `BatchExecutor` and `IncrementalExecutor` with `tick`, `revise` and `seal`
  - exact cutoffs and observers
  - the hypothesis equivalence suite on synthetic daily and minute data
  
  **Exit:** a sealed row matches batch bit for bit; revisions match batch over the provisional row; and a 500-name × 100-term minute pipeline revises in < 50 ms on belter-class hardware (benchmarked by David).
- **M1** is the old P1 remainder (the full operator set, the Pipeline API, golden tests against zipline). P2 onward are unchanged.
- **Arb step 1** (dirty-strike checks) can run in parallel inside humdesk at any time.

### 11.8 Remaining decisions for David

1. **Bar label:** end time (recommended) or start time?
2. **Grids for v1:** daily only live, with minute in research only, or minute live too?
3. **Live minute-data source.** IB streaming lines are shared with the arb collector (90 of 100 in use), and historical-bar polling is pacing-limited, so live minute bars for 100+ names don't fit the current IB setup. Options: buy more IB lines (quote booster packs), take a vendor feed (Alpaca IEX or Polygon/Massive, for example), or keep minute data to research and backfill.
4. **Seal grace** for minute bars (2 s default), and the daily seal rule (at the official IB daily bar, or at 16:00 + 5 min).
5. **Show provisional (open-row) factor values on the dashboard?**
6. **Arb:** approve migration step 1 (dirty-strike checks in humdesk) as a separate small PR, and decide whether the arb grid should be 1 s.
7. **Missing-bar policy** (forward-fill close and zero volume, or NaN).

## 12. M0 as built (branch `m0-core`)

* `src/factorflow/`: `kernels.py` (numerics), `terms.py` (Factor/Filter/Classifier
  expressions), `graph.py` (Pipeline, CSE, topological order, look-back), `batch.py`, `live.py`
  (ring buffers, revise/seal, exact cutoff, observers), `grid.py` (BarGrid), `bars.py`
  (time-bar accumulator), `events.py` (tick/volume/dollar bars, adaptive thresholds, as-of
  sampling, per-asset event pipelines), `panel.py`.
* Exactness comes from writing each kernel once, as a function of a block of rows. Batch calls
  it on the whole history and live calls it on the last `window` rows, so both run the same
  IEEE operations in the same order (fixed left-to-right accumulation over the window, no
  pairwise or BLAS reductions). Cross-sectional and recursive terms run row by row in both.
  Transcendental ufuncs (`log`) are always computed on full rows.
* The bar-close trigger is pluggable: the engine exposes `revise()`/`seal()`. Clock grids
  drive it from timestamps, and event-bar builders drive it per asset. See
  [`BARS.md`](BARS.md) for overshoot, adaptive thresholds and alignment.
* Column-subset recomputation (only dirty names) is implemented and tested but off by default
  (`partial_threshold=0.0`). At 500 names the per-lag Python overhead dominates, so a full
  row is as fast. Exact cutoff (NaN-aware equality) still stops propagation through
  unchanged nodes.

