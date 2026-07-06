/**
 * regressions.test.ts — pins for a batch of fixes:
 *
 *  1. cusumFit with < 2 samples: wide std0 = 1 fallback instead of std0 = 1e-6
 *     (which made h microscopic → CUSUM alarmed on any deviation).
 *  2. Detector trained with exactly windowSize trades (1 rolling sample) is
 *     not hypersensitive on a calm detection window.
 *  3. detect() with fewer trades than windowSize: weights are renormalized so
 *     a Hawkes-only window can still reach confidence 1 (was capped at wH=0.4,
 *     silently making anomaly=true unreachable).
 *  4. Single-trade window: empiricalRate is meaningless (1 event / 1 ms floor
 *     = bogus 1000/s) and must be disabled, not amplified by renormalization.
 *  5. bocpdAnomalyScore: relative run-length drop is shrunk by run maturity —
 *     a reset of a barely-established run (2 → 0) scores far below the reset
 *     of a long run (50 → 0).
 *  6. cusum_alarm signal meta reports the peak pre-reset accumulators, not the
 *     post-reset zeros.
 */

import { describe, it, expect } from 'vitest';
import { VolumeAnomalyDetector } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';
import { cusumFit, bocpdAnomalyScore, bocpdInitState } from '../src/math/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

let _id = 0;

function makeTrade(
  timestamp: number,
  qty: number,
  isBuyerMaker: boolean,
  price = 100,
): IAggregatedTradeData {
  return { id: String(_id++), price, qty, timestamp, isBuyerMaker };
}

/** n alternating buy/sell trades, fixed interval — deterministic calm stream */
function makeCalm(n: number, startTs: number, intervalMs: number): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  for (let i = 0; i < n; i++) {
    trades.push(makeTrade(startTs + i * intervalMs, 1, i % 2 === 0));
  }
  return trades;
}

// ─── 1. cusumFit: degenerate sample counts ────────────────────────────────────

describe('cusumFit: fewer than 2 samples falls back to wide std0', () => {
  it('single sample: std0 = 1, not 1e-6', () => {
    const p = cusumFit([0.42]);
    expect(p.mu0).toBe(0.42);
    expect(p.std0).toBe(1);
  });

  it('single sample: h is not microscopic', () => {
    const p = cusumFit([0.42], 0.5, 5);
    expect(p.h).toBe(5);
    expect(p.k).toBe(0.5);
  });

  it('single non-finite sample behaves like empty input', () => {
    const p = cusumFit([NaN]);
    expect(p.mu0).toBe(0);
    expect(p.std0).toBe(1);
  });

  it('two samples: variance is estimated normally', () => {
    const p = cusumFit([0, 1]);
    expect(p.mu0).toBeCloseTo(0.5, 10);
    expect(p.std0).toBeCloseTo(Math.sqrt(0.5), 10); // sample variance, n-1
  });
});

// ─── 2. Training with exactly windowSize trades ───────────────────────────────

describe('train with exactly windowSize trades (1 rolling sample)', () => {
  it('calm detection window does not trigger CUSUM hypersensitivity', () => {
    // 50 trades = windowSize → rolling series has exactly 1 sample.
    // Before the fix: std0 = 1e-6 → h = 5e-6 → cusum_alarm on any deviation.
    const detector = new VolumeAnomalyDetector(); // windowSize 50
    detector.train(makeCalm(50, 0, 1000));

    expect(detector.trainedModels!.cusumParams.h).toBeGreaterThan(0.1);

    const result = detector.detect(makeCalm(200, 100_000, 1000), 0.75);
    expect(result.anomaly).toBe(false);
    expect(result.signals.every((s) => s.kind !== 'cusum_alarm')).toBe(true);
  });
});

// ─── 3. detect() shorter than windowSize: weight renormalization ─────────────

describe('detect() with trades < windowSize renormalizes weights', () => {
  it('1000× rate burst in a short window can exceed confidence 0.4', () => {
    const detector = new VolumeAnomalyDetector(); // windowSize 50, weights [0.4, 0.3, 0.3]
    detector.train(makeCalm(300, 0, 1000)); // baseline 1 trade/s

    // 30 trades 1 ms apart — 1000× the baseline rate, but < windowSize,
    // so the rolling series is empty and only Hawkes runs.
    const burst = makeCalm(30, 400_000, 1);
    const result = detector.detect(burst, 0.75);

    // Before the fix confidence was capped at 0.4 (wH) — anomaly unreachable.
    expect(result.confidence).toBeGreaterThan(0.75);
    expect(result.anomaly).toBe(true);
  });

  it('calm short window stays quiet', () => {
    const detector = new VolumeAnomalyDetector();
    detector.train(makeCalm(300, 0, 1000));

    const calm = makeCalm(30, 400_000, 1000); // same rate as baseline
    const result = detector.detect(calm, 0.75);

    expect(result.anomaly).toBe(false);
    expect(result.confidence).toBeLessThan(0.5);
  });

  it('wH = 0 with empty rolling series gives confidence 0 (no NaN)', () => {
    const detector = new VolumeAnomalyDetector({ scoreWeights: [0, 0.5, 0.5] });
    detector.train(makeCalm(300, 0, 1000));

    const burst = makeCalm(30, 400_000, 1);
    const result = detector.detect(burst, 0.75);

    expect(result.confidence).toBe(0);
    expect(result.anomaly).toBe(false);
  });
});

// ─── 4. Single-trade window: empirical rate disabled ─────────────────────────

describe('detect() with a single trade', () => {
  it('does not fabricate a 1000/s empirical rate', () => {
    const detector = new VolumeAnomalyDetector();
    detector.train(makeCalm(300, 0, 1000)); // baseline 1 trade/s

    // One trade: windowSec floors at 1 ms → naive rate = 1000/s, a pure
    // artifact.  With the rate signal off, only the (calm) intensity ratio
    // remains and confidence must stay low.
    const result = detector.detect([makeTrade(400_000, 1, false)], 0.75);

    expect(result.anomaly).toBe(false);
    expect(result.confidence).toBeLessThan(0.5);
  });
});

// ─── 5. bocpdAnomalyScore: run-maturity shrink ────────────────────────────────

describe('bocpdAnomalyScore: maturity-weighted drop', () => {
  const reset = { mapRunLength: 0, cpProbability: 0.5, state: bocpdInitState() };

  it('full reset from a barely-established run (2 → 0) scores low', () => {
    expect(bocpdAnomalyScore(reset, 2)).toBeLessThan(0.3);
  });

  it('full reset from an established run (50 → 0) scores high', () => {
    expect(bocpdAnomalyScore(reset, 50)).toBeGreaterThan(0.9);
  });

  it('score is monotone in previous run length for a full reset', () => {
    let prev = 0;
    for (const rl of [1, 2, 5, 10, 20, 50, 100]) {
      const s = bocpdAnomalyScore(reset, rl);
      expect(s).toBeGreaterThan(prev);
      prev = s;
    }
  });
});

// ─── 6. cusum_alarm meta: peak pre-reset accumulators ────────────────────────

describe('cusum_alarm meta reflects the pre-reset peak', () => {
  it('meta.sPos/sNeg are not both zero when the alarm fired mid-window', () => {
    const detector = new VolumeAnomalyDetector({
      windowSize:   20,
      cusumHSigmas: 2,                 // fire fast
      scoreWeights: [0.0, 1.0, 0.0],   // isolate CUSUM
    });
    detector.train(makeCalm(300, 0, 1000)); // balanced → |imbalance| ≈ 0

    // Sustained all-buy at baseline rate: |imbalance| = 1 in every window →
    // CUSUM alarms (and resets) repeatedly inside the window.
    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) {
      recent.push(makeTrade(400_000 + i * 1000, 1, false));
    }
    const result = detector.detect(recent, 0.0);
    const alarm  = result.signals.find((s) => s.kind === 'cusum_alarm');

    expect(alarm).toBeDefined();
    // Before the fix meta held the post-reset state — often exactly 0/0,
    // contradicting the alarm being signalled.
    const peak = Math.max(alarm!.meta['sPos']!, alarm!.meta['sNeg']!);
    expect(peak).toBeGreaterThanOrEqual(0.7 * alarm!.meta['h']!);
  });
});
