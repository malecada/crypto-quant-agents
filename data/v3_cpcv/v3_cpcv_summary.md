# V3 CPCV Evaluation Summary — Task 38

**Evaluation window**: 2024-05-01 → 2026-04-15 (~24 months)
**Method**: Combinatorial Purged Cross-Validation (López de Prado 2018)
**Config**: n_groups=8, test_groups=2, embargo=14 bars → C(8,2) = 28 splits
**Models**: Pre-trained bundles reused across folds (no per-fold retraining per Phase-7 simplification)

---

## Per-Coin CPCV Results

### Bitcoin (BTC)

| Metric             | Value       |
|--------------------|-------------|
| n_splits           | 28          |
| Sharpe mean        | **-2.40**   |
| Sharpe median      | -2.31       |
| Sharpe std         | 0.65        |
| Sharpe min         | -4.37       |
| Sharpe max         | **-0.92**   |
| Positive SR splits | 0 / 28      |
| DSR (n_trials=12)  | ~0.00       |
| 25th pctile SR     | -2.65       |
| 75th pctile SR     | -2.05       |

**Interpretation**: V3 is uniformly negative across all 28 test windows for BTC. Not a single split yields positive Sharpe. The maximum observed Sharpe (-0.92) is still strongly negative. DSR ≈ 0 means there is essentially zero probability that the observed performance reflects real skill rather than noise (after accounting for 12 trials).

### Ethereum (ETH)

| Metric             | Value       |
|--------------------|-------------|
| n_splits           | 28          |
| Sharpe mean        | **-2.92**   |
| Sharpe median      | -3.01       |
| Sharpe std         | 1.05        |
| Sharpe min         | -5.00       |
| Sharpe max         | **+0.59**   |
| Positive SR splits | 1 / 28      |
| DSR (n_trials=12)  | ~0.00       |
| 25th pctile SR     | -3.26       |
| 75th pctile SR     | -2.45       |

**Interpretation**: V3 is also nearly uniformly negative for ETH. One split (split 27: 2025-10-17 → 2026-04-14, the most recent 6-month window) yields a slightly positive Sharpe of +0.59, but this is the only window and barely above zero. All other 27 splits are negative, many severely so (the 2025-01–2025-10 sub-period is particularly bad, SR down to -5.0). DSR ≈ 0 confirms no detectable skill.

### Single Positive SR Window (ETH split 27)

Split 27: **2025-10-17 → 2026-04-14** (180 bars)
- Sharpe: +0.59
- Total return: +4.6%
- Max drawdown: 13.0%

This is the only sub-period where V3 shows marginal positive performance for either coin. It is far below the V2 baseline (ETH Sharpe 2.57) and not statistically significant given 27/28 negative splits.

---

## Comparison vs V2 Quant Baseline

V2 reference metrics over equivalent OOS window (363 days, from `data/multi_2coins_v2/report_v2/metrics.json`):

| Metric       | V3 CPCV Mean SR | V2 Single-Window SR | Difference  |
|--------------|-----------------|---------------------|-------------|
| BTC Sharpe   | -2.40           | +2.18               | **-4.58**   |
| ETH Sharpe   | -2.92           | +2.57               | **-5.49**   |

V3 underperforms V2 by ~4-5 Sharpe ratio points across both coins. This is not a marginal difference; it represents a complete reversal from profitable to loss-making.

### V3 88-bar A/B Reference (Task 37)

| Metric              | CPCV Mean SR | 88-bar A/B SR | Consistent? |
|---------------------|--------------|---------------|-------------|
| BTC Sharpe          | -2.40        | -2.71         | Yes         |
| ETH Sharpe          | -2.92        | +1.25         | No (ETH worse in CPCV) |
| Portfolio avg SR    | -2.66        | -0.73         | ETH worse in CPCV |

The 88-bar A/B showed ETH Sharpe +1.25 (mildly positive), but the CPCV across 24 months shows ETH is also deeply negative on average. The 88-bar result appears to have captured a favorable sub-period for ETH; CPCV unmasks that it was anomalous.

---

## Key Findings

1. **V3 is uniformly negative**: 0/28 BTC splits positive, 1/28 ETH splits positive. This is not a "mixed" result — it is consistent underperformance across all time windows.

2. **No sub-period shows life for BTC**: BTC V3 Sharpe ranges from -0.92 to -4.37, all negative. The 88-bar A/B BTC result (-2.71) is confirmed and generalizes across CPCV.

3. **ETH 88-bar outlier explained**: The ETH +1.25 single-window result from Task 37 was capturing the most recent portion of the data. CPCV across 24 months shows the true picture: mean -2.92, 27/28 splits negative.

4. **One marginal window (ETH, Oct 2025 – Apr 2026)**: SR +0.59 in split 27. This is the period closest to the current date; V3 may be slightly improving in recent periods, but the signal is extremely weak and not actionable.

5. **DSR ≈ 0 for both coins**: After adjusting for 12 trials (n_groups × test_groups variation), the Deflated Sharpe Ratio is effectively zero for both coins. No statistical evidence of positive skill.

6. **V3 feature regime detection is problematic**: The per-bar HMM re-run from full history (identified during debugging) causes regime detection to process ~1,844 bars per call. The LightGBM feature name warnings ("fitted with feature names, but X has no valid feature names") indicate the models were trained with named features but inference uses array inputs — a minor technical debt that may affect prediction quality slightly but does not explain the magnitude of underperformance.

---

## Bugs Fixed (Task 38)

### Bug 1: `_load_ohlcv_for_coin` parameter mismatch
- **Error**: `TypeError: _load_crypto_ohlcv() got an unexpected keyword argument 'coin'`
- **Root cause**: `evaluate_v3_cpcv.py` called `_load_crypto_ohlcv(coin=coin, days=days)`, but the function signature is `_load_crypto_ohlcv(coingecko_id: str, curr_date: str)`.
- **Fix**: Changed call to `_load_crypto_ohlcv(coingecko_id=coin, curr_date="2026-04-15")` (matching the pattern in `baseline_strategy_v3.py`).

---

## Conclusion

V3 shows **no evidence of positive skill** over the 24-month CPCV evaluation window (May 2024 – April 2026). The CPCV Sharpe distributions are uniformly negative for both BTC and ETH, confirming and extending the Task 37 88-bar findings. V3 does not outperform — or even match — the V2 quant baseline in any measurable sub-period for BTC, and in only 1/28 windows for ETH (weakly positive at +0.59 SR).

**Recommendation**: V3 as currently specified (NH-HMM regime + multi-horizon LGB/XGB/CatBoost ensemble + vol-target CDAP sizing) does not improve on V2. The primary issue appears to be the ensemble signal quality — the regime-weighted multi-horizon consensus is generating directional errors systematically, not just noise. Further investigation should focus on whether the signal is systematically inverted (which would suggest a sign flip bug) or genuinely uninformative.
