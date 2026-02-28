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
export type { DetectorConfig }   from './detector.js';

export type {
  IAggregatedTradeData,
  DetectionResult,
  AnomalySignal,
  AnomalyKind,
} from './types.js';

// ─── Functional one-shot API ──────────────────────────────────────────────────

import { VolumeAnomalyDetector } from './detector.js';
import type { IAggregatedTradeData, DetectionResult } from './types.js';

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
