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

export { VolumeAnomalyDetector, severityOf } from './detector.js';
export type { DetectorConfig, DetectorSnapshot, TrainedModels, CalibrationReport } from './detector.js';

export type {
  IAggregatedTradeData,
  DetectionResult,
  AnomalySignal,
  AnomalyKind,
  Direction,
  Severity,
  PredictionResult,
} from './types.js';

// ─── Functional one-shot API ──────────────────────────────────────────────────

import { VolumeAnomalyDetector, severityOf } from './detector.js';
import type { DetectorConfig } from './detector.js';
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
    severity:       r.severity,
    direction,
    imbalance:      r.imbalance,
    burstImbalance: r.burstImbalance,
    moveScore:      r.moveScore,
  };
}

// ─── Convenience API for non-specialists ──────────────────────────────────────

/** Result of scan(): full detection output plus the directional signal. */
export type ScanResult = DetectionResult & { direction: Direction };

export interface ScanOptions extends DetectorConfig {
  /**
   * How much trailing market time (seconds) to evaluate as the "recent"
   * window; everything before it trains the baseline.  Default 30 s (the
   * alert timescale).
   */
  recentSec?:  number;
  /** Anomaly threshold [0,1]. Default 0.75. */
  confidence?: number;
}

/**
 * One-call scan of a single trade stream — no manual historical/recent
 * slicing (the #1 integration mistake: overlapping windows absorb the very
 * anomaly being detected into the baseline).
 *
 * The stream is split by TIME: the last `recentSec` seconds are evaluated,
 * everything before trains the baseline.  When the tail is nearly empty
 * (quiet market), the recent window extends to the last 20 trades so there
 * is always something to evaluate.
 *
 * @example
 * ```typescript
 * const r = scan(trades);          // trades = last 15–30+ min, oldest first
 * if (r.anomaly) console.log(explain(r));
 * ```
 */
export function scan(
  trades:  IAggregatedTradeData[],
  options: ScanOptions = {},
): ScanResult {
  const { recentSec = 30, confidence = 0.75, ...config } = options;
  if (!(recentSec > 0) || !Number.isFinite(recentSec)) {
    throw new Error(`recentSec must be a finite number > 0, got ${recentSec}`);
  }
  const sorted = [...trades].sort((a, b) => a.timestamp - b.timestamp);
  const cutoff = sorted.length > 0
    ? sorted[sorted.length - 1]!.timestamp - recentSec * 1000
    : 0;
  let split = sorted.findIndex((t) => t.timestamp > cutoff);
  if (split < 0) split = sorted.length;
  // Quiet tail: make sure the recent window has something to evaluate
  if (sorted.length - split < 20) split = Math.max(0, sorted.length - 20);
  const historical = sorted.slice(0, split);
  if (historical.length < 50) {
    throw new Error(
      `scan() needs >= 50 baseline trades before the recent window, got ${historical.length}; ` +
      `feed 15-30 minutes of trades (recent window = last ${recentSec}s)`,
    );
  }
  const detector = new VolumeAnomalyDetector(config);
  detector.train(historical);
  const r   = detector.detect(sorted.slice(split), confidence);
  const thr = Math.max(0, detector.trainedModels!.imbalanceThreshold);
  let direction: Direction = 'neutral';
  if (r.anomaly) {
    if (r.burstImbalance >  thr) direction = 'long';
    else if (r.burstImbalance < -thr) direction = 'short';
  }
  return { ...r, direction };
}

/**
 * Plain-language explanation of a detection result — what happened, how
 * unusual it is, who drove it, and how to read the numbers.  Accepts both
 * DetectionResult/ScanResult (full detail) and PredictionResult (summary).
 *
 * @param result     Output of detect(), scan() or predict().
 * @param threshold  The alert threshold the caller uses (for context). Default 0.75.
 */
export function explain(
  result:    DetectionResult | PredictionResult | ScanResult,
  threshold: number = 0.75,
): string {
  const lines: string[] = [];
  const sev = result.severity ?? severityOf(result.confidence);
  lines.push(
    (result.anomaly
      ? `Volume anomaly detected (severity: ${sev})`
      : `No anomaly (severity: ${sev})`) +
    ` — confidence ${result.confidence.toFixed(2)} vs alert threshold ${threshold.toFixed(2)}.`,
  );

  if ('stats' in result) {
    const { zRates, zVols, horizonsSec } = result.stats;
    let bestZ = 0, bestScale = 0, bestType: 'trade rate' | 'volume' = 'volume';
    for (let k = 0; k < horizonsSec.length; k++) {
      if ((zRates[k] ?? 0) > bestZ) { bestZ = zRates[k]!; bestScale = horizonsSec[k]!; bestType = 'trade rate'; }
      if ((zVols[k]  ?? 0) > bestZ) { bestZ = zVols[k]!;  bestScale = horizonsSec[k]!; bestType = 'volume'; }
    }
    if (bestZ > 0) {
      lines.push(
        `Strongest signal: ${bestType} ran ~${bestZ.toFixed(0)} robust sigma above the ` +
        `recent typical level at the ${bestScale.toFixed(0)}s scale.`,
      );
    }
    if (result.peakTs > 0) {
      lines.push(`Peak at ${new Date(result.peakTs).toISOString()}.`);
    }
    const buyShare = (1 + result.burstImbalance) / 2;
    if (Math.abs(result.burstImbalance) >= 0.2) {
      lines.push(
        `Order flow at the peak: ${(100 * Math.max(buyShare, 1 - buyShare)).toFixed(0)}% ` +
        `${result.burstImbalance > 0 ? 'buy' : 'sell'}-side.`,
      );
    } else {
      lines.push('Order flow at the peak: roughly balanced.');
    }
  }

  const move = result.moveScore;
  lines.push(
    `Follow-through ranking (moveScore): ${move.toFixed(2)} — ` +
    (move >= 0.75 ? 'top-tier; historically precedes real price movement more often than most alerts.'
      : move >= 0.5 ? 'moderate.'
      : 'low; likely routine even if flagged.'),
  );

  if ('direction' in result && result.direction !== 'neutral') {
    lines.push(
      `Dominant side: ${result.direction === 'long' ? 'buyers' : 'sellers'} drove the burst. ` +
      'Note: direction describes the event; measured on real data it does NOT predict the sign of the next move.',
    );
  }
  return lines.join('\n');
}
