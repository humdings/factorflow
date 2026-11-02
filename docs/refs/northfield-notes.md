# Northfield Information Services reference docs: paraphrased notes

These notes are my own summaries, not copies. The original PDFs are David's attachments: 06eb956f…pdf, 6539a405…pdf and 528c502e…pdf. They aren't reachable from Shell on the box.

## 1. Open Performance Attribution for Windows (36 pp, about 2003; sample run July–Sept 2001)
- **Purpose:** a performance attribution tool for a portfolio vs a benchmark. It works with any factor model.
- **Factor types:**
  - continuous (beta, P/E…)
  - membership factors, which come as discrete (one industry per stock), fractional (spread across groups, e.g. a REIT's regions) or hybrid
  
  Factors are organized into groups, and groups into user-defined sectors.
- **Reports:**
  - Brinson–Fachler univariate stratification (weight impact plus selection impact), by factor bucket, industry, sector and asset
  - multivariate factor impact, i.e. active exposure × factor return
  - risk decomposition: contributed variance per factor = own variance plus covariance terms, factor vs specific tracking variance, tracking error, total risk of portfolio and benchmark
  - return decomposition into factor and stock-specific parts, broken down by group
  - per-stock exposures, plus an exception report for unmatched IDs
  - over-time series of returns, risk forecasts and exposures
  - **impact t-stat** = mean / std × √n of the per-period impacts, used as a skill test
- They prefer monthly observations and argue that daily attribution statistics are often misused.
- **Appendix: the Northfield Fundamental US Equity Model (FFM), 67 factors**
  - 1 beta factor, 11 standardized style factors (E/P, B/P, dividend yield, trading activity, relative strength, market cap, earnings variability, EPS growth, S/P, debt/equity, price volatility) and 55 industries
  - Price ratios are computed as reciprocals (E/P rather than P/E) so they don't divide by zero.
  - Standardization uses a reference universe of US common stocks above $250M market cap.
  - Regressions are weighted by √cap (a compromise between equal and cap weighting).
  - Estimation is **sequential**: the styles are regressed first, then industries are fit to the residuals, because small industries would make the joint fit unstable.
  - **Beta:** a 60-month CAPM beta, Blume-adjusted toward 1, then projected onto fundamentals (a "fundamental beta" that works even for IPOs).
  - **Factor covariance:** an equal-weighted 60-month sample. They mention conditional variance as an ongoing research topic.
  - **Specific risk:** the 60-month std of residuals, adjusted upward for serial correlation (they cite Parkinson 1980).

## 2. Open Optimizer for Windows (72 pp, Feb 2015)
- **Inputs:** the risk model as files for factor variances, correlations, exposures with specific risk, and short-horizon residuals; the portfolio, benchmark and buy list; alphas with optional standard errors; composites (look-through ETFs and funds).
- **Objective:** utility = α − [systematic tracking variance / SysRAP + specific tracking variance / UnsysRAP] − amortized costs − quadratic penalties.
  - Their rule of thumb is RAP ≈ 6 × the acceptable tracking error.
  - A tracking-error target is met by rescaling both RAPs.
  - Separate risk aversions for systematic and specific risk: stock pickers set SysRAP low, factor timers set it high.
- **Alpha handling:**
  - "Reshape": rank → percentile → normal z → α = √(cross-sectional variance) × IC × z, with a default IC of 0.05.
  - A benchmark-relative Black–Litterman Bayes adjustment.
  - Jorion Bayes–Stein shrinkage of alphas.
- **Covariance blending:** a user-weighted mix of the model matrix with single-index, constant-correlation and constant-covariance targets (Ledoit–Wolf style).
- **Constraints:**
  - per-security min/max (absolute, relative to initial, or relative to benchmark), do-not-buy and do-not-sell
  - minimum trade size and position threshold
  - industry and sector bands or quadratic tilts, user attributes as soft or hard constraints, factor exposure bands
  - maximum turnover, trades and assets; min/max tracking error
  - round lots and ticket charges, long/short, tax lots (FIFO/LIFO/HIFO)
- **Transaction costs:** linear plus square-root market impact, a "days to trade" knob, and impact decay with temporary vs permanent impact (R ≈ 0.71). Costs are amortized by expected turnover, which converts one-off costs into annual α units, and there's a multi-period approximation.
- **Solver:** greedy pairwise swap iterations, with callbacks.

## 3. Open Optimizer: Report Descriptions with Formulae (24 pp)
- **Optimization summary:** factor return and risk, specific return and risk, total, tracking error, utility.
- **Analysis summary:** counts, long/short, turnover (two-way), amortized cost.
- **Risk decomposition (initial vs optimal):** exposures, factor variance, and variance contribution, where each factor gets its own variance plus half its covariances. Totals include R² vs the benchmark.
- **Return decomposition:** factor premia plus "economic changes" (macro model).
- **Holdings and trade list:** rounded vs unrounded trades, costs, taxes.
- **Penalties and constraints summary:** binding or violated constraints only.
- **Marginal utility, return and variance per security (buy and sell side):** ranks the best next trades. They note that marginal variance depends on the assumed offsetting asset (cash or benchmark).
