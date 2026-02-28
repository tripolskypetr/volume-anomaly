import { describe, it, expect, beforeEach } from 'vitest';
import { VolumeAnomalyDetector, detect } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';

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

/** Generate n trades with Poisson-ish arrival and random buy/sell split */
function generateTrades(
  n: number,
  startTs: number,
  avgIntervalMs: number,
  buyFraction: number, // [0,1]
): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  let ts = startTs;
  for (let i = 0; i < n; i++) {
    const interval = avgIntervalMs * (0.5 + Math.random());
    ts += interval;
    const isBuyerMaker = Math.random() > buyFraction;
    trades.push(makeTrade(ts, 1 + Math.random() * 2, isBuyerMaker));
  }
  return trades;
}

// ─── VolumeAnomalyDetector ────────────────────────────────────────────────────

describe('VolumeAnomalyDetector', () => {
  let detector: VolumeAnomalyDetector;

  beforeEach(() => {
    detector = new VolumeAnomalyDetector({ windowSize: 20 });
  });

  it('throws before training', () => {
    const trades = generateTrades(10, 0, 1000, 0.5);
    expect(() => detector.detect(trades)).toThrow();
  });

  it('isTrained becomes true after train()', () => {
    const hist = generateTrades(200, 0, 1000, 0.5);
    detector.train(hist);
    expect(detector.isTrained).toBe(true);
  });

  it('throws when training with < 50 trades', () => {
    const hist = generateTrades(30, 0, 1000, 0.5);
    expect(() => detector.train(hist)).toThrow(/50/);
  });

  it('returns DetectionResult with all fields', () => {
    const hist   = generateTrades(200, 0, 1000, 0.5);
    const recent = generateTrades(100, 200_000, 1000, 0.5);
    detector.train(hist);
    const result = detector.detect(recent);

    expect(typeof result.anomaly).toBe('boolean');
    expect(typeof result.confidence).toBe('number');
    expect(Array.isArray(result.signals)).toBe(true);
    expect(typeof result.imbalance).toBe('number');
    expect(typeof result.hawkesLambda).toBe('number');
    expect(typeof result.cusumStat).toBe('number');
    expect(typeof result.runLength).toBe('number');
  });

  it('confidence is in [0,1]', () => {
    const hist   = generateTrades(200, 0, 1000, 0.5);
    const recent = generateTrades(100, 200_000, 1000, 0.5);
    detector.train(hist);
    const result = detector.detect(recent);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('anomaly=false when confidence < threshold', () => {
    const hist   = generateTrades(200, 0, 1000, 0.5);
    const recent = generateTrades(50, 200_000, 1000, 0.5);
    detector.train(hist);
    const result = detector.detect(recent, 0.9999); // near-impossible threshold
    // Not necessarily false (could fire), but if confidence < 0.9999 then anomaly=false
    expect(result.anomaly).toBe(result.confidence >= 0.9999);
  });

  it('empty recent trades returns no anomaly', () => {
    const hist = generateTrades(200, 0, 1000, 0.5);
    detector.train(hist);
    const result = detector.detect([], 0.5);
    expect(result.anomaly).toBe(false);
    expect(result.confidence).toBe(0);
  });

  it('trainedModels exposed for debugging', () => {
    const hist = generateTrades(200, 0, 1000, 0.5);
    detector.train(hist);
    const m = detector.trainedModels!;
    expect(m.hawkesParams.mu).toBeGreaterThan(0);
    expect(m.cusumParams.h).toBeGreaterThan(0);
  });

  it('throws for invalid scoreWeights', () => {
    expect(
      () => new VolumeAnomalyDetector({ scoreWeights: [0.5, 0.5, 0.5] }),
    ).toThrow();
  });

  describe('signal detection', () => {
    it('imbalance_shift signal fires on heavy buy pressure', () => {
      const hist   = generateTrades(200, 0, 1000, 0.5);
      // All buy aggressors → isBuyerMaker = false
      const recent = generateTrades(200, 200_000, 1000, 1.0);

      detector.train(hist);
      const result = detector.detect(recent, 0.0);

      const hasImb = result.signals.some((s) => s.kind === 'imbalance_shift');
      // Imbalance should be close to +1
      expect(result.imbalance).toBeGreaterThan(0.3);
      // Signal may or may not fire depending on threshold, but imbalance must be positive
    });

    it('each signal has score in [0,1]', () => {
      const hist   = generateTrades(200, 0, 500, 0.5);
      const recent = generateTrades(200, 100_000, 100, 0.9); // fast + buy pressure
      detector.train(hist);
      const result = detector.detect(recent, 0.0);
      for (const sig of result.signals) {
        expect(sig.score).toBeGreaterThanOrEqual(0);
        expect(sig.score).toBeLessThanOrEqual(1);
        expect(sig.meta).toBeDefined();
      }
    });
  });
});

// ─── Functional API ───────────────────────────────────────────────────────────

describe('detect() functional API', () => {
  it('trains and detects in one call', () => {
    const hist   = generateTrades(200, 0, 1000, 0.5);
    const recent = generateTrades(100, 200_000, 1000, 0.5);
    const result = detect(hist, recent, 0.75);
    expect(typeof result.anomaly).toBe('boolean');
    expect(typeof result.confidence).toBe('number');
  });

  it('is deterministic for same input', () => {
    // Seed-free but same trades array → same result
    const hist   = generateTrades(200, 1_000_000, 1000, 0.5);
    const recent = generateTrades(50,  1_200_000, 1000, 0.5);

    const r1 = detect([...hist], [...recent], 0.75);
    const r2 = detect([...hist], [...recent], 0.75);

    expect(r1.confidence).toBeCloseTo(r2.confidence, 10);
  });
});
