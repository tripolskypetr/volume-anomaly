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
 *  7. predict() direction on trended baselines: the signed-p75 threshold turns
 *     negative on sell-heavy training data, and the unclamped symmetric ±thr
 *     comparison then labelled balanced flow (imbalance ≈ 0) as 'long' while
 *     making 'neutral' unreachable.  The threshold is now clamped at zero.
 *  8. detect() on very large windows: Math.max(...rates) spread overflowed the
 *     call stack (RangeError) at ~10⁶ trades.
 *  9. Zero-variance CUSUM baseline: a constant training |imbalance| series
 *     (fully one-sided flow) collapsed std0 to the 1e-6 numerical floor, so
 *     with custom scoreWeights any deviation raised cusum_alarm at score 1.
 *     Now takes the same wide std0 = 1 fallback as the <2-samples case.
 */

import { describe, it, expect } from 'vitest';
import { VolumeAnomalyDetector, predict } from '../src/index.js';
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

// ─── 7. predict() direction on trended baselines ──────────────────────────────

describe('predict() direction: sell-biased training does not flip long/short', () => {
  /** n trades, `buyEveryN` of them buy-aggressor — e.g. 100/7 ≈ 14% buys */
  function makeBiased(n: number, startTs: number, buyEveryN: number): IAggregatedTradeData[] {
    const trades: IAggregatedTradeData[] = [];
    for (let i = 0; i < n; i++) {
      trades.push(makeTrade(startTs + i * 1000, 1, i % buyEveryN !== 0)); // isBuyerMaker=true → sell
    }
    return trades;
  }

  it('sell-heavy baseline → trained threshold quantile is negative', () => {
    const det = new VolumeAnomalyDetector();
    det.train(makeBiased(500, 0, 7)); // ~86% sell aggressors
    // Precondition of the bug: signed p75 goes below zero on sell-biased data.
    expect(det.trainedModels!.imbalanceThreshold).toBeLessThan(0);
  });

  it('balanced recent flow after sell-heavy training is not "long"', () => {
    const hist = makeBiased(500, 0, 7);
    const rec  = makeCalm(100, 600_000, 1000); // alternating buy/sell → imbalance = 0
    // confidence 0 → anomaly always true, isolating the direction logic.
    const r = predict(hist, rec, 0.0);
    expect(r.imbalance).toBeCloseTo(0, 6);
    // Before the fix: imbalance (0) > negative p75 → 'long' on balanced flow.
    expect(r.direction).toBe('neutral');
  });

  it('sell burst after sell-heavy training is still "short"', () => {
    const hist = makeBiased(500, 0, 7);
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) {
      rec.push(makeTrade(600_000 + i * 1000, 1, true)); // all sells → imbalance = −1
    }
    const r = predict(hist, rec, 0.0);
    expect(r.imbalance).toBe(-1);
    expect(r.direction).toBe('short');
  });

  it('buy burst after buy-heavy training is still "long"', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 500; i++) {
      hist.push(makeTrade(i * 1000, 1, i % 7 === 0)); // ~86% buy aggressors
    }
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) {
      rec.push(makeTrade(600_000 + i * 1000, 1, false)); // all buys → imbalance = +1
    }
    const r = predict(hist, rec, 0.0);
    expect(r.imbalance).toBe(1);
    expect(r.direction).toBe('long');
  });

  it('direction always matches the imbalance sign', () => {
    // Property pinned by the clamp: 'long' ⇒ imbalance > 0, 'short' ⇒ < 0.
    const histories = [
      makeBiased(500, 0, 7),          // sell-heavy
      makeBiased(500, 0, 2),          // balanced
      makeCalm(500, 0, 1000),         // alternating
    ];
    const recents = [
      makeCalm(100, 600_000, 1000),                                                // imb 0
      Array.from({ length: 100 }, (_, i) => makeTrade(600_000 + i * 1000, 1, true)),  // imb −1
      Array.from({ length: 100 }, (_, i) => makeTrade(600_000 + i * 1000, 1, false)), // imb +1
    ];
    for (const hist of histories) {
      for (const rec of recents) {
        const r = predict(hist, rec, 0.0);
        if (r.direction === 'long')  expect(r.imbalance).toBeGreaterThan(0);
        if (r.direction === 'short') expect(r.imbalance).toBeLessThan(0);
      }
    }
  });

  it('negative explicit override is clamped, not flipped', () => {
    const hist = makeCalm(500, 0, 1000);
    const rec  = makeCalm(100, 600_000, 1000); // imbalance = 0
    const r = predict(hist, rec, 0.0, -0.5);
    // Before the clamp: 0 > −0.5 → 'long' on perfectly balanced flow.
    expect(r.direction).toBe('neutral');
  });
});

// ─── 8. detect() on very large windows ────────────────────────────────────────

describe('detect() with ~10⁶ trades does not overflow the call stack', () => {
  it('1M-trade window: no RangeError from Math.max spread', () => {
    const detector = new VolumeAnomalyDetector();
    detector.train(makeCalm(300, 0, 500));

    const big: IAggregatedTradeData[] = new Array(1_000_000);
    for (let i = 0; i < big.length; i++) {
      big[i] = makeTrade(400_000 + i * 100, 1, i % 2 === 0);
    }
    // Before the fix this threw "RangeError: Maximum call stack size exceeded"
    // inside detect() before any scoring ran.
    const result = detector.detect(big, 0.75);
    expect(Number.isFinite(result.confidence)).toBe(true);
  }, 60_000);
});

// ─── 9. CUSUM on a zero-variance baseline ─────────────────────────────────────

describe('constant |imbalance| baseline does not make CUSUM hypersensitive', () => {
  /** All-sell stream: |imbalance| = 1 in every rolling window → zero variance */
  function makeOneSided(n: number, startTs: number): IAggregatedTradeData[] {
    return Array.from({ length: n }, (_, i) => makeTrade(startTs + i * 1000, 1, true));
  }

  it('trained h is not microscopic', () => {
    const det = new VolumeAnomalyDetector();
    det.train(makeOneSided(500, 0));
    // Wide fallback: std0 = 1 → h = 5σ = 5, not 5e-6.
    expect(det.trainedModels!.cusumParams.std0).toBe(1);
    expect(det.trainedModels!.cusumParams.h).toBeGreaterThan(1);
  });

  it('routine flow jitter after one-sided training → no cusum_alarm', () => {
    const det = new VolumeAnomalyDetector({ scoreWeights: [0, 1, 0] });
    det.train(makeOneSided(500, 0));

    // 96–97% sell with scattered single buys: |imbalance| wiggles in 0.92–1.
    // Before the fix each such window instantly alarmed at score 1.
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) {
      rec.push(makeTrade(600_000 + i * 1000, 1, i % 33 !== 0)); // every 33rd trade is a buy
    }
    const r = det.detect(rec, 0.75);

    expect(r.signals.every((s) => s.kind !== 'cusum_alarm')).toBe(true);
    expect(r.scores.cusum).toBeLessThan(0.5);
    expect(r.anomaly).toBe(false);
  });

  it('genuine flow collapse after one-sided training still alarms', () => {
    const det = new VolumeAnomalyDetector({ scoreWeights: [0, 1, 0] });
    det.train(makeOneSided(500, 0));

    // Alternating buy/sell → |imbalance| = 0 sustained: a full regime change
    // from the |imbalance| = 1 baseline accumulates 0.5σ/step and must alarm.
    const r = det.detect(makeCalm(200, 600_000, 1000), 0.75);

    expect(r.signals.some((s) => s.kind === 'cusum_alarm')).toBe(true);
    expect(r.anomaly).toBe(true);
  });
});
