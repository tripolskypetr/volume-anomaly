# volume-anomaly — Project Memory

## What this project is
TypeScript npm library for **detecting volume anomalies in trading streams**.
Zero dependencies. Published as `volume-anomaly`.

## Purpose
Detects statistically unusual moments in market microstructure from Binance aggregated trade streams.
Answers: "is right now an abnormal moment?" — not price direction.
Designed to work alongside the `garch` library (garch → price corridors, this → entry timing).

## Architecture

### Public API (src/index.ts → src/detector.ts)
- `detect(historical, recent, confidence?)` — one-shot functional API
- `VolumeAnomalyDetector` class — stateful, train once, detect many times
- Config: windowSize, rateHorizonSec, slowHorizonSec, hazardLambda, cusumKSigmas, cusumHSigmas, scoreWeights

### Math modules (src/math/)
- `hawkes.ts` — Hawkes process (μ, α, β), MLE via Nelder-Mead, O(n) log-likelihood
- `cusum.ts` — CUSUM control chart, two-sided, applied to |imbalance|
- `bocpd.ts` — Bayesian Online Changepoint Detection (Adams & MacKay 2007), Normal-Gamma conjugate
- `optimizer.ts` — Nelder-Mead simplex (3 params for Hawkes only)

### Input type: IAggregatedTradeData
{ id, price, qty, timestamp (ms), isBuyerMaker }
isBuyerMaker=true → sell aggressor; false → buy aggressor

### Output: DetectionResult
{ anomaly, confidence[0,1], scores{hawkes,cusum,bocpd}, stats{zRate,zVol,zRateSlow,zVolSlow,lambdaRatio},
  signals[], imbalance[-1,+1], hawkesLambda, cusumStat, runLength }

## Score composition (recalibrated on real data, July 2026)
Primary statistic: self-calibrated robust z of rolling rate (trades/s) and
volume rate (qty/s) over TIME horizons 5 s and 30 s.
z = (peak_detect − median_train) / (1.4826 · max(MAD_train, 0.1·median)).
score_volume = max over 4 channels of sigmoid((z − 12)·0.4).
confidence = 1.0·score_volume + 0·cusum + 0·bocpd  (defaults [1,0,0])
anomaly = confidence >= threshold (default 0.75 ⇒ fires at z ≈ 14.7).

CUSUM/BOCPD (on rolling |imbalance|) are flow-shift detectors: computed,
self-calibrated (CUSUM h = max(5σ, 2× training excursion); BOCPD rescaled
above the training noise floor) and reported, but zero-weighted by default —
on the real-data benchmark any additive weight on them reduced recall and
added false alarms.

Training contract: historical must span 15–30 MINUTES of market time (not
"last N trades" — a count-based baseline masks the burst it precedes).
Internally: rate baselines use the full span (O(n)); Hawkes MLE capped to the
last 2000 trades; imbalance series to the last 1000.

## Real-data benchmark (test/eval.test.ts, EVAL=1)
Full day BTCUSDT 2025-03-01 (1.49M trades), 30s buckets, ground truth =
robust z ≥ 8 vs trailing hour; sliding-window operational protocol.
At confidence 0.75: bucket recall 90.3%, event recall 93.5%, FP 2.49%.
The eval test asserts recall ≥ 0.75 / event recall ≥ 0.8 / FP ≤ 0.05 as a
regression gate — if a change trips it, fix the change, not the thresholds.

## Tests (vitest, ~750 tests, 19 files) — ALL PASSING
Math units, detector integration, seeded scenarios, false-positive suite,
adversarial/extreme inputs, real-data fixtures (mock/*.json, 15-min baselines),
perf bounds, and the opt-in full-day eval sweep (EVAL=1).

## Build
- TypeScript + Rollup (rollup.config.js)
- vitest for testing (vitest.config.ts)
- exports: main + "volume-anomaly/math" subpath

## Key files
- src/detector.ts — main class VolumeAnomalyDetector
- src/math/hawkes.ts — volumeImbalance, hawkesFit, hawkesPeakLambda, hawkesAnomalyScore
- src/math/cusum.ts — cusumFit, cusumUpdate, cusumBatch, cusumAnomalyScore
- src/math/bocpd.ts — bocpdUpdate, bocpdBatch, bocpdAnomalyScore, defaultPrior
- src/math/optimizer.ts — nelderMead

## Critical bugs found and fixed (this session)

### hawkes.ts — LL origin bug
`hawkesLogLikelihood` computed T = timestamps[n-1] (absolute) instead of T = timestamps[n-1] - timestamps[0].
Fixed: t0 shift, all ti normalised to window origin. MLE now correct for real Unix-epoch timestamps.

### optimizer.ts — sortSimplex double-overwrite
Original code wrote simplex rows in first loop, then re-read corrupted data in second loop.
Fixed: snapshot idx/rows/vals before overwriting.

### optimizer.ts — penalty wall false convergence
`if (spread < tol)` fired when all vertices at 1e10 (penalty). Fixed: added `&& fvals[0]! < 1e9`.

### hawkes.ts — hawkesAnomalyScore + hawkesPeakLambda
- Added `hawkesPeakLambda(timestamps, params)`: O(n) recursive A(i) trick, returns max λ(tᵢ) over window.
- `hawkesAnomalyScore` now takes `empiricalRate` (events/s) as 3rd param; uses max of intensity score vs rate score.
  - empiricalRate/mu detects burst even when MLE assigns alpha≈0 (Poisson baseline).
- detector.ts uses hawkesPeakLambda + empiricalRate instead of hawkesLambda(lastT).

### bocpd.ts — bocpdAnomalyScore
- cpProbability is ALWAYS ≈ H = 1/hazardLambda (prior constant), it does NOT spike at changepoints.
- Real signal: drop in mapRunLength. After changepoint: mapRL resets from ~N to ~0.
- bocpdAnomalyScore(result, prevRunLength): sigmoid on (prevRL - currRL) / prevRL.
  - drop=0 (stable growth) → score≈0.12; drop=0.98 (reset 90→1) → score≈0.98.
- detector.ts: tracks prevRL per step, takes peak drop score over the window.

### detector.ts — BOCPD signed/unsigned mismatch
Prior trained on |imbalance| but BOCPD was fed signed values. Fixed: rollingAbsImbalance used for both.

### detector.ts — CUSUM peak score tracking
Alarm resets state to 0 mid-window, losing the spike. Fixed: track peakCusumScore before reset applies.

### detector.ts — cusumHSigmas default 4→5
ARL₀ raised from ≈55 to ≈148 (fewer false positives).
