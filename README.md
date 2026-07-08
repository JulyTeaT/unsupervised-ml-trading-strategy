# Algorithmic Trading Strategy — Unsupervised Learning Approach

A quantitative trading strategy that uses **K-Means clustering** to group S&P 500 stocks by technical/risk features every month, selects a momentum-aligned cluster, and builds an **optimal portfolio (max Sharpe ratio)** from it — backtested against a SPY buy-and-hold benchmark.

##  Overview

Instead of a classic supervised "predict up/down" model, this project takes an **unsupervised learning** route:

1. Engineer technical, volatility, and risk-factor features for every S&P 500 stock.
2. Cluster stocks each month based on these features using K-Means.
3. Select stocks from a specific cluster hypothesized to show persistent momentum (RSI ≈ 70).
4. Optimize portfolio weights within the selected cluster using Mean-Variance / Efficient Frontier optimization.
5. Rebalance monthly in a walk-forward backtest and compare cumulative returns to the S&P 500.

##  Roadmap

```
Data Gathering → Feature Engineering → Monthly Resampling & Liquidity Filter
      → Momentum Features → Fama-French Factor Betas → Monthly K-Means Clustering
      → Cluster-Based Stock Selection → Portfolio Optimization (Max Sharpe)
      → Walk-Forward Backtest → Compare vs. SPY
```

##  Data Gathering

- **Universe**: Current S&P 500 constituents scraped from Wikipedia (`requests` + `pandas.read_html`).
- **Prices**: Daily OHLCV data for all tickers pulled via `yfinance`.
- **Risk factors**: Fama-French 5-Factor data pulled via `pandas_datareader` (Ken French's data library).
- **Backtest prices**: Fresh daily prices re-downloaded only for the shortlisted (post-filter) stocks, plus SPY for benchmarking.

##  Data Preprocessing & Cleaning

- Ticker symbols normalized (`.` → `-`) to match Yahoo Finance conventions.
- Daily data resampled to **month-end frequency** to reduce computation and noise.
- Filtered to the **top 150 most liquid stocks per month**, ranked by a 5-year rolling average dollar volume.
- Outlier clipping (0.5% / 99.5% quantiles) applied before computing returns.
- Stocks with fewer than 10 months of history dropped; remaining NaNs handled via group-wise fill/drop.

##  Feature Engineering

| Category | Features |
|---|---|
| Volatility | Garman-Klass Volatility |
| Momentum/Oscillators | RSI, MACD |
| Volatility Bands | Bollinger Bands (low, mid, high) |
| Risk | ATR (Average True Range) |
| Liquidity | Dollar Volume |
| Multi-horizon Returns | 1, 2, 3, 6, 9, 12-month returns (geometrically normalized) |
| Risk Factor Exposure | Rolling Fama-French betas (Mkt-RF, SMB, HML, RMW, CMA) via Rolling OLS (24-month window) |

##  Algorithms Used

- **K-Means Clustering** — groups stocks into 4 clusters each month based on the feature set above.
- **Predefined Centroid Initialization** — centroids anchored to target RSI values (42, 50, 59, 67) so clusters stay interpretable and consistent month over month, rather than relying purely on random/`k-means++` initialization.
- **Efficient Frontier Optimization** (`PyPortfolioOpt`) — Mean-Variance Optimization to maximize the Sharpe ratio of the selected cluster's stocks.

##  Optimization

Two layers of optimization are used:

1. **Clustering optimization** — centroids fixed on RSI values to encode the hypothesis that stocks clustering around RSI ~70 exhibit persistent momentum (cluster selected for the portfolio).
2. **Portfolio optimization** — `EfficientFrontier.max_sharpe()` (solved with `SCS` solver) using:
   - Expected returns: mean historical return (annualized, 252 trading days)
   - Covariance: sample covariance matrix
   - Constraints: per-stock weight bounds (min ≈ half of equal-weight, max 10%) for diversification
   - **Fallback**: if optimization fails for a given month, weights default to equal-weighted.

##  Training & Testing (Walk-Forward)

There is no traditional train/test split — this is a **walk-forward backtest**:

- **"Training"**: Each month, clusters are re-fit and portfolio weights are recomputed using only trailing data (12-month price window for optimization, expanding feature history for clustering) — no lookahead.
- **"Testing"**: The prior month's cluster selection and weights are applied to the following month's actual returns, simulating realistic monthly rebalancing.

##  Results

- Daily portfolio returns are computed from the log returns of the optimized holdings.
- Cumulative returns are plotted against SPY's buy-and-hold cumulative returns over the full backtest period.
- Output: **"Unsupervised Learning Trading Strategy Returns Over Time"** chart, visually comparing the strategy's growth to the S&P 500 benchmark.

## 🔧 Tech Stack

`pandas` · `numpy` · `matplotlib` · `statsmodels` · `pandas_datareader` · `yfinance` · `pandas_ta` · `scikit-learn` · `PyPortfolioOpt`

##  Limitations & Future Work

- No explicit risk metrics computed yet (Sharpe ratio, max drawdown, volatility) beyond the cumulative return plot.
- No systematic comparison across different numbers of clusters, lookback windows, or centroid choices.
- Strategy is built around a single hypothesis (RSI-momentum cluster) rather than testing multiple cluster-selection rules.
- Some notebook cells were left empty/incomplete during experimentation.

##  How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib statsmodels pandas_datareader yfinance pandas_ta scikit-learn PyPortfolioOpt
   ```
2. Open `capstone.ipynb` in Jupyter and run cells sequentially (top to bottom).
3. Note: live data downloads (Wikipedia, Yahoo Finance, Ken French Data Library) require an internet connection and may be subject to rate limits/schema chan
