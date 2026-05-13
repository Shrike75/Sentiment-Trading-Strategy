# Sentiment-Trading-Strategy

<p align="center">
  <b>News sentiment, market microstructure intuition, and walk-forward strategy evaluation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Research-Quantitative%20Finance-purple" alt="Research">
  <img src="https://img.shields.io/badge/NLP-FinBERT%20%7C%20SieBERT%20%7C%20SentenceTransformers-green" alt="NLP">
  <img src="https://img.shields.io/badge/Backtest-Walk--Forward-orange" alt="Walk-forward backtest">
  <img src="https://img.shields.io/badge/Status-Research%20Project-lightgrey" alt="Status">
</p>

---

## Overview

This repository explores whether news sentiment contains useful information for short-horizon equity return prediction and whether that information can be converted into a realistic trading rule.

The project develops in three stages:

1. **Initial NLP forecasting experiment**  
   A neural network model uses daily news headline embeddings and market features to predict next-day sector ETF direction.

2. **Tetlock-style replication**  
   A replication of the central ideas in Paul Tetlock's media pessimism framework, using the FRBSF Daily News Sentiment Index as a modern proxy for aggregate news tone.

3. **Full sentiment-driven trading strategy**  
   A multi-asset, asset-linked news sentiment strategy that compares FinBERT, SieBERT, and Loughran--McDonald lexicon sentiment signals under walk-forward out-of-sample testing and transaction-cost-aware portfolio construction.

The core conclusion: **sentiment contains a small but detectable forecasting signal, especially as a cross-sectional ranking device, but the trading edge is fragile once turnover and realistic frictions are included.**

This project is not a claim of a production-ready alpha model. It is a research notebook suite focused on signal construction, empirical discipline, robustness testing, and honest interpretation of financial signals.

---

## Repository Structure

```text
Sentiment-Trading-Strategy/
│
├── NN_headlines_Stock_movement.ipynb
│   └── Initial neural-network experiment using headline transformer embeddings
│
├── Tetlock_strategy_replication_DNSI_DJI.ipynb
│   └── Replication-style study of Tetlock media pessimism using DNSI and DJI returns
│
├── Sentiment_Trading_Strategy.ipynb
│   └── Full multi-asset sentiment signal and walk-forward trading strategy notebook
│
└── README.md
```

---

## Research Motivation

Financial markets are noisy, adaptive systems where most apparent signals disappear under proper out-of-sample testing. News sentiment is a particularly interesting test case because it has a plausible behavioral and informational mechanism:

- negative news may create temporary price pressure,
- high-salience news may increase investor attention,
- disagreement or emotional intensity may increase trading volume,
- asset-specific headlines may help rank securities cross-sectionally even when aggregate return predictability is weak.

The project asks a practical question:

> Can modern NLP sentiment measures improve short-horizon return forecasts, and does that improvement survive the transition from prediction to a cost-aware trading rule?

---

## Notebook 1: Neural Network Headline Embedding Experiment

**File:** `NN_headlines_Stock_movement.ipynb`

This notebook is the initial exploration into sentiment-driven market prediction. It uses global news headlines and sector ETF data to test whether sentence-transformer embeddings add value beyond standard numeric market features.

### Data and Features

- News data: `Combined_News_DJIA.csv`
- Market data: sector ETFs downloaded through `yfinance`
- Sector universe:
  - `XLE`, `XLF`, `XLV`, `XLK`, `XLY`, `XLP`, `XLI`, `XLB`, `XLU`
- Date range:
  - approximately 2008-08-08 to 2016-07-01
- Text representation:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - daily headline embeddings averaged across headlines
  - smoothed daily embedding features
- Numeric market features:
  - lagged log returns
  - rolling volatility
  - moving-average deviation features
  - volume z-scores

### Model Design

The notebook builds a supervised binary classification task:

```text
Input:  recent market features + optional daily news embedding
Target: next-day sector return direction
Model:  feed-forward neural network
Metric: validation accuracy and binary cross-entropy loss
```

Two model families are compared:

1. **Numeric-only model**
2. **Numeric + text embedding model**

### Key Findings

The early experiment found only a small improvement from adding headline embeddings. Several configurations placed the text-enhanced model slightly above the numeric-only baseline, but the absolute predictive edge remained close to noise.

Representative results from the notebook:

| Model | Validation Loss | Validation Accuracy |
|---|---:|---:|
| Numeric-only | 0.7693 | 0.4946 |
| Numeric + text | 1.1671 | 0.5013 |

In another configuration:

| Model | Validation Loss | Validation Accuracy |
|---|---:|---:|
| Numeric-only | 1.3256 | 0.4926 |
| Numeric + text | 1.6895 | 0.4984 |

### Interpretation

The main lesson from this notebook is not that the neural model is a strong trading strategy. Instead, it motivates the later notebooks:

- the signal is weak but potentially nonzero,
- text features may help more as a ranking signal than as a pure directional classifier,
- stronger empirical design is needed before making strategy claims,
- the next step should use more interpretable sentiment indicators and stricter out-of-sample testing.

---

## Notebook 2: Tetlock-Style Sentiment Replication

**File:** `Tetlock_strategy_replication_DNSI_DJI.ipynb`

This notebook replicates the spirit of Paul Tetlock's famous media pessimism study using a modern, accessible sentiment proxy.

Tetlock's original paper studies whether pessimistic financial media language predicts short-horizon market returns and trading activity. Since the original Wall Street Journal column data is difficult to reproduce directly, this notebook uses the **FRBSF Daily News Sentiment Index (DNSI)** as an aggregate news sentiment proxy.

### Research Design

The notebook constructs a standardized pessimism measure:

```text
pessimism = -standardized(DNSI)
```

It then tests whether pessimism predicts:

1. **next-day market returns**,
2. **multi-day return reversal**, and
3. **trading volume / activity.**

### Methods

The empirical design includes:

- daily Dow Jones market returns,
- lagged pessimism variables,
- lagged return controls,
- lagged volume controls,
- day-of-week and January controls,
- HAC / Newey--West robust inference,
- rolling-window stability checks,
- cross-asset robustness checks.

### Key Findings

The replication finds weaker return predictability than Tetlock's original paper, but more support for the attention and activity mechanism.

#### Return Predictability

In the paper-period return regression, the five pessimism lags are not individually or jointly significant in a robust way. The estimated return effect is noisy and economically modest.

This could reflect:

- a genuinely weak return effect,
- dilution from using a broad DNSI index instead of the original WSJ-specific measure,
- changes in market efficiency and news processing over time,
- the difficulty of identifying daily return signals in noisy aggregate data.

#### Stability

The early-period pessimism coefficient is much stronger than the late-period coefficient:

| Period | L1 Pessimism Coefficient |
|---|---:|
| Early subsample | ~20.8 bps |
| Late subsample | ~0.5 bps |

Rolling five-year estimates also decline over time, suggesting that the return effect is not fully stable across the sample.

#### Cross-Asset Extension

The notebook also tests sentiment effects across several assets:

| Asset | L1 Coefficient | L1 p-value | Sum L1--L5 | Sum p-value |
|---|---:|---:|---:|---:|
| S&P 500 | 8.05 bps | 0.645 | 1.45 bps | 0.373 |
| Nasdaq-100 / QQQ | 11.26 bps | 0.655 | 3.70 bps | 0.051 |
| Long Treasuries / TLT | -7.59 bps | 0.587 | 0.08 bps | 0.940 |
| Gold / GLD | 10.19 bps | 0.590 | 1.04 bps | 0.490 |
| Bitcoin / BTC | -26.10 bps | 0.721 | -3.46 bps | 0.547 |

### Interpretation

The Tetlock replication provides the economic foundation for the full strategy notebook. It suggests that sentiment may be more reliable as an **attention or activity signal** than as a direct aggregate return predictor.

That motivates shifting from an aggregate market timing framework to a **cross-sectional ranking framework**, where sentiment can be used to compare assets against one another rather than forecast the entire market direction.

---

## Notebook 3: Full Sentiment Trading Strategy

**File:** `Sentiment_Trading_Strategy.ipynb`

This is the main research notebook in the repository. It builds a complete asset-linked sentiment strategy and evaluates it through walk-forward testing.

### Objective

The notebook has two distinct goals:

1. **Statistical objective**  
   Test whether news sentiment improves next-day return forecasting beyond simple market controls.

2. **Economic objective**  
   Test whether the signal survives translation into a transparent, cost-aware long-short trading rule.

---

## Data Pipeline

The main strategy notebook uses:

### 1. Asset-Linked News

A panel of financial news headlines with:

- publication dates,
- headline text,
- associated stock tickers.

The strategy uses headline-level text rather than full article bodies to keep the pipeline computationally manageable and auditable.

### 2. Market Data

Daily market data is pulled using `yfinance` and used to construct:

- next-day target returns,
- lagged return controls,
- volume/activity controls,
- backtest returns,
- turnover and transaction-cost adjustments.

### 3. Sentiment Models

The notebook compares three sentiment sources:

| Sentiment Source | Role |
|---|---|
| FinBERT | Finance-specific transformer sentiment model |
| SieBERT | General-purpose sentiment transformer |
| Loughran--McDonald lexicon | Finance-domain dictionary baseline |

Different sentiment models are useful for different objectives. A model that best captures trading activity may not be the same model that best ranks next-day returns.

---

## Signal Construction

The strategy constructs several daily asset-level indicators:

| Indicator Family | Examples | Interpretation |
|---|---|---|
| Directional tone | `sentiment_mean`, `pessimism_mean` | Whether news is positive or negative |
| Intensity / disagreement | `abs_sentiment_mean`, `pessimism_std` | Strength or dispersion of tone |
| Coverage / attention | `news_count`, `log_news_count` | Amount of attention received |
| Normalized extremeness | `pessimism_x_news`, `pess_z_20` | Unusually negative or intense news |

The target is next-day return direction or next-day return ranking.

---

## Modeling Approach

The notebook deliberately favors simple, auditable models over highly flexible black-box models.

### Forecast Models

- Logistic regression
- XGBoost extensions
- Market-only baselines
- Sentiment-only models
- Sentiment-plus-controls models

### Evaluation Metrics

Forecasting quality is measured with:

- AUC,
- log loss,
- balanced accuracy,
- hit rate,
- calibration diagnostics,
- top-minus-bottom return spreads.

Trading quality is measured with:

- annualized return,
- annualized volatility,
- Sharpe ratio,
- turnover,
- transaction-cost-adjusted returns,
- fold-level and year-level stability.

---

## Trading Rule

The main trading rule is intentionally simple:

1. estimate out-of-sample next-day scores through a walk-forward process,
2. rank assets by predicted probability or sentiment score,
3. go long the highest-ranked assets,
4. go short the lowest-ranked assets,
5. equal-weight positions,
6. subtract transaction costs based on turnover.

This avoids hiding the economic result behind a complex optimizer.

---

## Walk-Forward Design

The full strategy notebook uses rolling-origin walk-forward testing:

```text
Train on past data
        ↓
Select/tune parameters using only available history
        ↓
Predict future out-of-sample period
        ↓
Construct long-short portfolio
        ↓
Evaluate gross and cost-adjusted performance
        ↓
Roll forward and repeat
```

This is the central guardrail against overfitting. The strategy is judged on future periods that were not available during model estimation.

---

## Main Results

### 1. Indicator Layer

Raw sentiment indicators are directionally plausible but weak as standalone predictors.

Key takeaways:

- pooled return correlations are close to zero,
- pessimism alone does not produce a clean unconditional return signal,
- absolute sentiment has a stronger relation to next-day trading activity,
- FinBERT is especially strong in the volume/activity mechanism test.

The strongest mechanism result is from FinBERT:

```text
absolute FinBERT sentiment → next-day log volume
β ≈ 0.044, t ≈ 5.78, p < 1e-8
```

This supports the idea that finance-specific sentiment is useful for identifying attention and trading-activity effects, even when direct return prediction is difficult.

### 2. Forecasting Layer

The best out-of-sample forecasting results come from simple sentiment-only logistic models.

Representative results:

| Model Family | Approx. AUC | Interpretation |
|---|---:|---|
| SieBERT sentiment-only logistic | ~0.5081 | Best pure forecast AUC |
| Lexicon sentiment-only logistic | ~0.5077 | Similar AUC, strong balanced accuracy / hit rate |
| FinBERT | Lower forecast metrics | Stronger mechanism interpretation than direct forecast performance |

The AUC values are only slightly above 0.50, so the signal is weak. However, in financial prediction, a stable weak ranking signal can still be meaningful if it survives proper out-of-sample testing and implementation costs.

### 3. Ranking Value

The ranking tests are more encouraging than the raw classification metrics. The best signal specifications produce positive top-minus-bottom return spreads, suggesting that sentiment is more useful for **cross-sectional ordering** than for confident binary prediction.

This is an important distinction:

```text
The model does not strongly know whether an asset will go up.
But it has some ability to rank which assets are more likely to outperform.
```

### 4. Gross Trading Performance

The strongest gross strategy is:

```text
lexicon sentiment-only logistic ranking strategy
```

Representative gross performance at zero transaction costs:

| Metric | Value |
|---|---:|
| Annual return | ~14.0% |
| Annual volatility | ~16.2% |
| Sharpe ratio | ~0.89 |

This outperforms the low-complexity Tetlock-style tercile comparator, whose best gross Sharpe is around 0.20 and is negative for most variants.

### 5. Transaction Cost Fragility

The strongest caveat is implementation cost.

The best gross strategy turns negative after modest transaction costs. The notebook finds that turnover is high, often around 3.0 on average, which causes Sharpe ratios to decay quickly as costs increase.

The main conclusion is therefore:

> The signal is real enough to show up in out-of-sample ranking tests, but not strong enough in this implementation to support robust net profitability after realistic frictions.

---

## Results Summary

| Research Layer | Finding | Interpretation |
|---|---|---|
| Neural headline embeddings | Small text improvement over numeric-only features | Motivated deeper sentiment research |
| Tetlock replication | Weak aggregate return predictability | DNSI may dilute WSJ-specific tone effects |
| Tetlock volume channel | More supportive than return channel | Sentiment may affect attention more than direction |
| Indicator tests | Raw sentiment weak but plausible | Not enough for standalone trading |
| Forecast models | Best AUC around 0.508 | Weak but detectable signal |
| Ranking tests | Positive top-minus-bottom spreads | Better as cross-sectional ranker than classifier |
| Gross backtest | Best Sharpe around 0.89 before costs | Signal can be monetized in idealized setting |
| Net backtest | Performance collapses under modest costs | Current rule is not production-ready |
| Mechanism tests | FinBERT strongly predicts volume | Finance-specific NLP captures attention channel |

---

## Reproducibility Notes

Some notebooks depend on external or local data files that are not necessarily included in this repository by default.

Expected data inputs include:

```text
Combined_News_DJIA.csv
analyst_ratings_processed.csv
Loughran-McDonald_MasterDictionary_1993-2024.csv
```

Market data is downloaded dynamically through `yfinance`, so exact results may vary slightly depending on data revisions, ticker availability, and package versions.

The main notebooks are research notebooks rather than packaged production code. They are intended to be read, audited, and extended.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/sentiment-trading-strategy.git
cd sentiment-trading-strategy
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels yfinance pandas_datareader transformers torch sentence-transformers xgboost tqdm
```

3. Add the required local datasets to the repository root or update the file paths inside the notebooks.

4. Run the notebooks in this order:

```text
1. NN_headlines_Stock_movement.ipynb
2. Tetlock_strategy_replication_DNSI_DJI.ipynb
3. Sentiment_Trading_Strategy.ipynb
```

The third notebook is the main strategy notebook and can also be read independently if the goal is to evaluate the final research design.

---

## Limitations

The project has several important limitations:

- headline-only sentiment loses information contained in full article bodies,
- external data sources may revise or change over time,
- transaction-cost assumptions are simplified,
- the asset universe is constrained for computational tractability,
- the signal is weak and should not be interpreted as a robust standalone alpha,
- daily close-to-close execution is a simplified approximation,
- the notebooks are research prototypes, not production trading infrastructure.

These limitations are part of the research conclusion rather than afterthoughts. The main finding is that weak sentiment signals require careful implementation discipline, especially around turnover control and cost-aware portfolio construction.

---

## Future Work

Natural extensions include:

- use full article bodies instead of headlines,
- add intraday timing and more realistic execution assumptions,
- use higher-quality survivorship-bias-free pricing data,
- incorporate richer OHLCV and liquidity features,
- explicitly optimize for turnover-aware ranking stability,
- test sector-neutral and beta-neutral portfolio constraints,
- evaluate larger transformer models and domain-specific fine-tuning,
- separate event types such as earnings, analyst ratings, litigation, M&A, and macro news,
- add uncertainty-aware forecasts and confidence-based position sizing.

---

## Project Takeaway

The project supports a realistic view of NLP-based trading research:

> Modern sentiment models can extract economically meaningful information from news, but turning that information into robust net trading performance is much harder than producing a slightly predictive model or an attractive gross backtest.

That gap between signal detection and implementable alpha is the central lesson of the repository.

---

## Disclaimer

This repository is for educational and research purposes only. It is not financial advice, investment advice, or a recommendation to trade any security or strategy.
