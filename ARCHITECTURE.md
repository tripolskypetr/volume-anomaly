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
- Config: windowSize, hazardLambda, cusumKSigmas, cusumHSigmas, scoreWeights

### Math modules (src/math/)
- `hawkes.ts` — Hawkes process (μ, α, β), MLE via Nelder-Mead, O(n) log-likelihood
- `cusum.ts` — CUSUM control chart, two-sided, applied to |imbalance|
- `bocpd.ts` — Bayesian Online Changepoint Detection (Adams & MacKay 2007), Normal-Gamma conjugate
- `optimizer.ts` — Nelder-Mead simplex (3 params for Hawkes only)

### Input type: IAggregatedTradeData
{ id, price, qty, timestamp (ms), isBuyerMaker }
isBuyerMaker=true → sell aggressor; false → buy aggressor

### Output: DetectionResult
{ anomaly, confidence[0,1], signals[], imbalance[-1,+1], hawkesLambda, cusumStat, runLength }

## Score composition
confidence = 0.4·hawkes + 0.3·cusum + 0.3·bocpd  (defaults)
anomaly = confidence >= threshold (default 0.75)
No single detector can exceed threshold alone at defaults.

## Tests (vitest, 68 tests, 4 files) — ALL PASSING
- test/hawkes.test.ts — 20 tests
- test/cusum.test.ts — 15 tests
- test/bocpd.test.ts — 13 tests
- test/detector.test.ts — 20 tests

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
