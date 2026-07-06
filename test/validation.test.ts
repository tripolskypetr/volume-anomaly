/**
 * validation.test.ts — loud rejection of silently-broken configuration and
 * input, instead of a miscalibrated or dead detector:
 *
 *  1. Numeric config options (windowSize, hazardLambda, CUSUM sigmas,
 *     horizons, imbalancePercentile) — e.g. hazardLambda ≤ 1 made the BOCPD
 *     state silently collapse to empty on the first update.
 *  2. Timestamp units: seconds or microseconds instead of milliseconds never
 *     crash — they silently rescale every time horizon 1000×.
 *  3. detect() confidence: 75 instead of 0.75 silently never fires.
 */

import { describe, it, expect } from 'vitest';
import { VolumeAnomalyDetector } from '../src/index.js';
import type { IAggregatedTradeData, DetectorConfig } from '../src/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeStream(n: number, startTs: number, intervalMs: number): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  for (let i = 0; i < n; i++) {
    trades.push({ id: String(i), price: 100, qty: 1, timestamp: startTs + i * intervalMs, isBuyerMaker: i % 2 === 0 });
  }
  return trades;
}

// ─── 1. Config validation ─────────────────────────────────────────────────────

describe('constructor: numeric config validation', () => {
  const bad: Array<[string, DetectorConfig]> = [
    ['windowSize = 0',            { windowSize: 0 }],
    ['windowSize < 0',            { windowSize: -5 }],
    ['windowSize non-integer',    { windowSize: 2.5 }],
    ['windowSize NaN',            { windowSize: NaN }],
    ['hazardLambda = 1 (H = 1)',  { hazardLambda: 1 }],
    ['hazardLambda < 1 (H > 1)',  { hazardLambda: 0.5 }],
    ['hazardLambda NaN',          { hazardLambda: NaN }],
    ['hazardLambda Infinity',     { hazardLambda: Infinity }],
    ['cusumKSigmas = 0',          { cusumKSigmas: 0 }],
    ['cusumKSigmas < 0',          { cusumKSigmas: -1 }],
    ['cusumHSigmas = 0',          { cusumHSigmas: 0 }],
    ['rateHorizonSec = 0',        { rateHorizonSec: 0 }],
    ['rateHorizonSec < 0',        { rateHorizonSec: -3 }],
    ['slowHorizonSec = 0',        { slowHorizonSec: 0 }],
    ['imbalancePercentile < 0',   { imbalancePercentile: -1 }],
    ['imbalancePercentile > 100', { imbalancePercentile: 101 }],
    ['imbalancePercentile NaN',   { imbalancePercentile: NaN }],
  ];

  for (const [label, config] of bad) {
    it(`throws for ${label}`, () => {
      expect(() => new VolumeAnomalyDetector(config)).toThrow('must be');
    });
  }

  it('error message names the option and the received value', () => {
    expect(() => new VolumeAnomalyDetector({ hazardLambda: 0.5 }))
      .toThrow(/hazardLambda.*0\.5/);
  });

  const good: Array<[string, DetectorConfig]> = [
    ['defaults',                   {}],
    ['windowSize = 1 (boundary)',  { windowSize: 1 }],
    ['hazardLambda = 1.5',         { hazardLambda: 1.5 }],
    ['imbalancePercentile = 0',    { imbalancePercentile: 0 }],
    ['imbalancePercentile = 100',  { imbalancePercentile: 100 }],
    ['small explicit horizons',    { rateHorizonSec: 0.5, slowHorizonSec: 2 }],
  ];

  for (const [label, config] of good) {
    it(`accepts ${label}`, () => {
      expect(() => new VolumeAnomalyDetector(config)).not.toThrow();
    });
  }
});

// ─── 2. Timestamp units ───────────────────────────────────────────────────────

describe('train()/detect(): timestamp unit sanity check', () => {
  const EPOCH_S  = 1_740_812_760;             // 2025-03-01 in SECONDS
  const EPOCH_MS = 1_740_812_760_000;         // same instant in ms
  const EPOCH_US = 1_740_812_760_000_000;     // same instant in µs

  it('train() rejects epoch-seconds timestamps', () => {
    const det = new VolumeAnomalyDetector();
    expect(() => det.train(makeStream(100, EPOCH_S, 1))).toThrow(/SECONDS/);
  });

  it('train() rejects epoch-microseconds timestamps', () => {
    const det = new VolumeAnomalyDetector();
    expect(() => det.train(makeStream(100, EPOCH_US, 1_000_000))).toThrow(/MICRO/);
  });

  it('detect() applies the same check', () => {
    const det = new VolumeAnomalyDetector();
    det.train(makeStream(200, EPOCH_MS, 1000));
    expect(() => det.detect(makeStream(50, EPOCH_S, 1))).toThrow(/SECONDS/);
    expect(() => det.detect(makeStream(50, EPOCH_US, 1_000_000))).toThrow(/MICRO/);
  });

  it('epoch-milliseconds pass', () => {
    const det = new VolumeAnomalyDetector();
    expect(() => det.train(makeStream(200, EPOCH_MS, 1000))).not.toThrow();
    expect(() => det.detect(makeStream(50, EPOCH_MS + 300_000, 1000))).not.toThrow();
  });

  it('relative/synthetic timestamps (small values) pass', () => {
    const det = new VolumeAnomalyDetector();
    expect(() => det.train(makeStream(200, 0, 1000))).not.toThrow();
    expect(() => det.detect(makeStream(50, 300_000, 1000))).not.toThrow();
  });

  it('far-future milliseconds (Y2050) still pass', () => {
    const det = new VolumeAnomalyDetector();
    expect(() => det.train(makeStream(200, 2_500_000_000_000, 1000))).not.toThrow();
  });

  it('microsecond TAIL in otherwise-valid ms data is rejected (both ends checked)', () => {
    // First timestamp valid ms, last in µs: previously only the first element
    // was unit-checked, so the µs tail slipped through and inflated the span
    // (which downstream sizes loops and allocations).
    const trades = makeStream(200, EPOCH_MS, 1000);
    trades[trades.length - 1] = { ...trades[trades.length - 1]!, timestamp: EPOCH_US };
    const det = new VolumeAnomalyDetector();
    expect(() => det.train(trades)).toThrow(/MICRO/);
  });

  it('NaN / Infinity timestamps are rejected in train (anywhere in the stream)', () => {
    const det = new VolumeAnomalyDetector();
    const mid = makeStream(200, 0, 1000);
    mid[100] = { ...mid[100]!, timestamp: NaN };  // middle: survives sorting at either end
    expect(() => det.train(mid)).toThrow(/finite/);
    const inf = makeStream(200, 0, 1000);
    inf[199] = { ...inf[199]!, timestamp: Infinity };
    expect(() => det.train(inf)).toThrow(/finite/);
  });

  it('detect() rejects non-finite window ends', () => {
    const det = new VolumeAnomalyDetector();
    det.train(makeStream(200, 0, 1000));
    const rec = makeStream(50, 300_000, 1000);
    rec[49] = { ...rec[49]!, timestamp: NaN };
    // NaN sorts unpredictably but the ends check catches boundary corruption
    expect(() => det.detect([{ ...rec[0]!, timestamp: NaN }])).toThrow(/finite/);
  });
});

// ─── 4. Data-derived allocations are bounded (no hang on pathological spans) ──

describe('train(): pathological spans cannot hang the null calibration', () => {
  it('two clusters YEARS apart with pinned horizons train instantly via fallback', () => {
    // All timestamps sit in the valid ms band, but the span is ~9.5 years.
    // With explicitly pinned horizons (auto-horizons scale with the span and
    // self-protect; pinned ones do not) the null-calibration histogram would
    // be span/step ≈ 1.6e8 buckets — a multi-GB allocation and an effective
    // hang before the MAX_CALIB_BUCKETS bound.
    const trades: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) {
      trades.push({ id: String(i), price: 100, qty: 1, timestamp: 1_700_000_000_000 + i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    for (let i = 0; i < 100; i++) {
      trades.push({ id: `b${i}`, price: 100, qty: 1, timestamp: 2_000_000_000_000 + i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const det = new VolumeAnomalyDetector({ rateHorizonSec: 5, slowHorizonSec: 30 });
    const t0 = performance.now();
    det.train(trades);
    expect(performance.now() - t0).toBeLessThan(5000);
    // The calibration must have fallen back to the universal mapping
    // (empty null ladder) rather than allocating the histogram.
    expect(det.trainedModels!.channelCalib.rate[0]!.nullQ.length).toBe(0);
  });
});

// ─── 3. detect() confidence range ─────────────────────────────────────────────

describe('detect(): confidence must be a fraction in [0, 1]', () => {
  function trained(): VolumeAnomalyDetector {
    const det = new VolumeAnomalyDetector();
    det.train(makeStream(200, 0, 1000));
    return det;
  }

  it('rejects percent-style confidence (75 instead of 0.75)', () => {
    expect(() => trained().detect(makeStream(50, 300_000, 1000), 75)).toThrow('confidence');
  });

  it('rejects negative and NaN confidence', () => {
    const det = trained();
    expect(() => det.detect(makeStream(50, 300_000, 1000), -0.1)).toThrow('confidence');
    expect(() => det.detect(makeStream(50, 300_000, 1000), NaN)).toThrow('confidence');
  });

  it('accepts the boundaries 0 and 1', () => {
    const det = trained();
    expect(() => det.detect(makeStream(50, 300_000, 1000), 0)).not.toThrow();
    expect(() => det.detect(makeStream(50, 300_000, 1000), 1)).not.toThrow();
  });
});
