/**
 * volume-anomaly — public API
 *
 * @example
 * ```typescript
 * import { detect, VolumeAnomalyDetector } from 'volume-anomaly';
 *
 * // One-shot (convenience wrapper, no state):
 * const result = detect(historicalTrades, recentTrades, 0.75);
 * if (result.anomaly) {
 *   console.log('Entry signal!', result.imbalance, result.confidence);
 * }
 *
 * // Stateful (recommended for production):
 * const detector = new VolumeAnomalyDetector({ windowSize: 50 });
 * detector.train(historicalTrades);
 * const r = detector.detect(recentTrades, 0.75);
 * ```
 */

export { VolumeAnomalyDetector } from './detector.js';
export type { DetectorConfig, DetectorSnapshot, TrainedModels } from './detector.js';

export type {
  IAggregatedTradeData,
  DetectionResult,
  AnomalySignal,
  AnomalyKind,
  Direction,
  PredictionResult,
} from './types.js';

// ─── Functional one-shot API ──────────────────────────────────────────────────

import { VolumeAnomalyDetector } from './detector.js';
import type { IAggregatedTradeData, DetectionResult, Direction, PredictionResult } from './types.js';

/**
 * Convenience function: train + detect in one call.
 *
 * @param historical  Long baseline window (≥ 50 trades) — used for model training.
 * @param recent      Short recent window — evaluated for anomalies.
 * @param confidence  Required confidence to flag anomaly [0,1]. Default 0.75.
 */
export function detect(
  historical:  IAggregatedTradeData[],
  recent:      IAggregatedTradeData[],
  confidence:  number = 0.75,
): DetectionResult {
  const detector = new VolumeAnomalyDetector();
  detector.train(historical);
  return detector.detect(recent, confidence);
}

/**
 * One-shot anomaly detection with directional signal.
 *
 * Wraps `detect()` and adds a `direction` field derived from `imbalance`:
 * - `'long'`    — anomaly detected + buy-side order flow dominates
 * - `'short'`   — anomaly detected + sell-side order flow dominates
 * - `'neutral'` — no anomaly, or anomaly is a pure rate spike with balanced flow
 *
 * The directional threshold is derived automatically from training data:
 * `imbalanceThreshold = p75 of the rolling signed imbalance series` (configurable
 * via `DetectorConfig.imbalancePercentile`), applied symmetrically: long above
 * +threshold, short below −threshold.  The threshold is clamped at zero — on a
 * sell-trended baseline the raw quantile goes negative, and an unclamped bound
 * would label balanced (or even sell-side) flow as `'long'`: after the clamp
 * `'long'` always implies imbalance > 0 and `'short'` implies imbalance < 0.
 * Pass an explicit number to override.
 *
 * @param historical          Baseline window (≥ 50 trades) for model training.
 * @param recent              Recent window to evaluate.
 * @param confidence          Anomaly threshold [0,1]. Default 0.75.
 * @param imbalanceThreshold  Override the trained threshold (applied as
 *                            symmetric ±max(0, thr)). Omit to use p75 from training.
 */
export function predict(
  historical:          IAggregatedTradeData[],
  recent:              IAggregatedTradeData[],
  confidence:          number = 0.75,
  imbalanceThreshold?: number,
): PredictionResult {
  const detector = new VolumeAnomalyDetector();
  detector.train(historical);
  const r   = detector.detect(recent, confidence);
  const thr = Math.max(
    0,
    imbalanceThreshold ?? detector.trainedModels!.imbalanceThreshold,
  );

  // Direction reads the BURST-local imbalance, not the full-window one: a
  // burst's onset direction gets diluted by surrounding two-way flow when
  // averaged over the whole window (measured on real data: −0.42 full-window
  // vs −0.9 at the burst).  burstImbalance is already shrunk by effective
  // sample size toward the training flow balance, so a few-trade window
  // cannot fake conviction.
  let direction: Direction = 'neutral';
  if (r.anomaly) {
    if (r.burstImbalance >  thr) direction = 'long';
    else if (r.burstImbalance < -thr) direction = 'short';
  }

  return {
    anomaly:        r.anomaly,
    confidence:     r.confidence,
    direction,
    imbalance:      r.imbalance,
    burstImbalance: r.burstImbalance,
    moveScore:      r.moveScore,
  };
}
