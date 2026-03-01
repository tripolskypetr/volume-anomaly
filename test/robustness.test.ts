/**
 * robustness.test.ts — математические инварианты и property-based тесты.
 *
 * Каждый тест здесь проверяет свойство, которое ОБЯЗАНО быть истинным
 * по математике или по контракту API. Если тест падает — баг в коде,
 * не в тесте.
 *
 * Группы:
 *  1.  volumeImbalance      — диапазон, симметрия
 *  2.  hawkesAnomalyScore   — диапазон, монотонность
 *  3.  hawkesPeakLambda     — нижняя граница μ
 *  4.  hawkesFit            — params всегда положительны, stationarity < 1
 *  5.  cusumUpdate          — sPos/sNeg ≥ 0, alarm сбрасывает state
 *  6.  cusumAnomalyScore    — диапазон, монотонность по s/h
 *  7.  bocpdUpdate          — mapRL ≥ 0, cpProb ∈ [0,1], нормировка
 *  8.  bocpdUpdate          — mapRL растёт по 1 в стационарном режиме
 *  9.  bocpdAnomalyScore    — диапазон, монотонность по дропу
 * 10.  detector end-to-end  — confidence ∈ [0,1], never NaN
 * 11.  detector end-to-end  — anomaly ↔ confidence ≥ threshold
 * 12.  detector end-to-end  — детерминизм
 * 13.  detector end-to-end  — scoreWeights не-отрицательность
 * 14.  property-based       — 100 сидированных случаев, confidence ∈ [0,1]
 */

import { describe, it, expect } from 'vitest';
import {
  volumeImbalance,
  hawkesAnomalyScore,
  hawkesPeakLambda,
  hawkesFit,
} from '../src/math/hawkes.js';
import {
  cusumFit,
  cusumUpdate,
  cusumInitState,
  cusumAnomalyScore,
} from '../src/math/cusum.js';
import {
  bocpdUpdate,
  bocpdInitState,
  bocpdAnomalyScore,
  defaultPrior,
} from '../src/math/bocpd.js';
import { VolumeAnomalyDetector } from '../src/detector.js';
import type { IAggregatedTradeData } from '../src/types.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeLCG(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

let _uid = 0;
function trade(ts: number, qty: number, isBuyerMaker: boolean): IAggregatedTradeData {
  return { id: String(_uid++), price: 100, qty, timestamp: ts, isBuyerMaker };
}

/** Generates n trades with seeded PRNG */
function genTrades(n: number, seed: number, baseTs = 0, burstFrac = 0.5): IAggregatedTradeData[] {
  const rng = makeLCG(seed);
  const out: IAggregatedTradeData[] = [];
  for (let i = 0; i < n; i++) {
    out.push(trade(baseTs + i * 100, 1 + rng() * 9, rng() < burstFrac));
  }
  return out;
}

/** Minimal trained detector (200 calm trades) */
function trainedDetector(seed = 0xabcd, cfg: ConstructorParameters<typeof VolumeAnomalyDetector>[0] = {}): VolumeAnomalyDetector {
  const det = new VolumeAnomalyDetector(cfg);
  det.train(genTrades(200, seed));
  return det;
}

// ─── 1. volumeImbalance: диапазон [-1, +1] и симметрия ───────────────────────

describe('volumeImbalance: range invariant ∈ [-1, +1]', () => {
  const rng = makeLCG(0x1111);

  it('returns 0 for empty array', () => {
    expect(volumeImbalance([])).toBe(0);
  });

  it('returns +1 for all-buy window', () => {
    const trades = Array.from({ length: 20 }, (_, i) =>
      trade(i * 100, 1 + i, false),
    );
    expect(volumeImbalance(trades)).toBe(1);
  });

  it('returns -1 for all-sell window', () => {
    const trades = Array.from({ length: 20 }, (_, i) =>
      trade(i * 100, 1 + i, true),
    );
    expect(volumeImbalance(trades)).toBe(-1);
  });

  it('value ∈ [-1, +1] for 100 random finite-qty trade arrays', () => {
    for (let t = 0; t < 100; t++) {
      const n = Math.floor(rng() * 50) + 1;
      const arr = Array.from({ length: n }, () =>
        trade(0, rng() * 1e6 + 0.001, rng() < 0.5),
      );
      const v = volumeImbalance(arr);
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe('volumeImbalance: symmetry — flip isBuyerMaker negates result', () => {
  it('single trade: flip gives opposite sign', () => {
    const buy  = [trade(0, 5, false)];
    const sell = [trade(0, 5, true)];
    expect(volumeImbalance(buy)).toBe(-volumeImbalance(sell));
  });

  it('mixed window: flip gives negated imbalance', () => {
    const rng = makeLCG(0x2222);
    const n = 30;
    const original = Array.from({ length: n }, () =>
      trade(0, rng() * 10 + 0.1, rng() < 0.5),
    );
    const flipped = original.map((t) => ({ ...t, isBuyerMaker: !t.isBuyerMaker }));
    expect(volumeImbalance(original)).toBeCloseTo(-volumeImbalance(flipped), 12);
  });

  it('balanced window (equal qty buy = sell): imbalance = 0', () => {
    const arr = [
      trade(0, 3, false),
      trade(0, 3, true),
    ];
    expect(volumeImbalance(arr)).toBe(0);
  });
});

// ─── 2. hawkesAnomalyScore: диапазон [0,1], монотонность ─────────────────────

describe('hawkesAnomalyScore: range ∈ [0, 1]', () => {
  const params = { mu: 2, alpha: 0.5, beta: 3 };

  it('returns exactly 1 when branching ≥ 1 (supercritical)', () => {
    // alpha/beta = 1 → supercritical
    expect(hawkesAnomalyScore(10, { mu: 1, alpha: 2, beta: 2 }, 0)).toBe(1);
    expect(hawkesAnomalyScore(0, { mu: 1, alpha: 3, beta: 2 }, 0)).toBe(1);
  });

  it('score ∈ [0, 1] for wide range of peakLambda (0..100)', () => {
    for (let peak = 0; peak <= 100; peak += 5) {
      const s = hawkesAnomalyScore(peak, params, 0);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it('score ∈ [0, 1] for wide range of empiricalRate (0..1000)', () => {
    for (let rate = 0; rate <= 1000; rate += 50) {
      const s = hawkesAnomalyScore(0, params, rate);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it('score is never NaN for valid finite inputs', () => {
    const rates = [0, 0.001, 1, 100, 1e6];
    const peaks = [0, 0.1, 1, 10, 1e6];
    for (const r of rates) {
      for (const p of peaks) {
        expect(Number.isNaN(hawkesAnomalyScore(p, params, r))).toBe(false);
      }
    }
  });
});

describe('hawkesAnomalyScore: monotone in empiricalRate', () => {
  // Higher empirical rate should produce higher-or-equal score
  // (mu > 0 path: sig(rate/mu) is strictly increasing in rate)
  const params = { mu: 1, alpha: 0.3, beta: 2 };

  it('score(rate=10) ≥ score(rate=1) for same params', () => {
    const s1 = hawkesAnomalyScore(0, params, 1);
    const s10 = hawkesAnomalyScore(0, params, 10);
    expect(s10).toBeGreaterThanOrEqual(s1);
  });

  it('score(rate=0) ≤ score(rate=5) ≤ score(rate=100)', () => {
    const s0 = hawkesAnomalyScore(0, params, 0);
    const s5 = hawkesAnomalyScore(0, params, 5);
    const s100 = hawkesAnomalyScore(0, params, 100);
    expect(s5).toBeGreaterThanOrEqual(s0);
    expect(s100).toBeGreaterThanOrEqual(s5);
  });
});

describe('hawkesAnomalyScore: monotone in peakLambda', () => {
  const params = { mu: 1, alpha: 0.4, beta: 2 };

  it('score(peak=20) ≥ score(peak=2) for same params (meanLambda > 0)', () => {
    const s2 = hawkesAnomalyScore(2, params, 0);
    const s20 = hawkesAnomalyScore(20, params, 0);
    expect(s20).toBeGreaterThanOrEqual(s2);
  });
});

// ─── 3. hawkesPeakLambda: нижняя граница μ ────────────────────────────────────

describe('hawkesPeakLambda: peak ≥ μ', () => {
  const params = { mu: 2, alpha: 0.5, beta: 3 };

  it('returns μ for empty timestamps (no events)', () => {
    expect(hawkesPeakLambda([], params)).toBe(params.mu);
  });

  it('returns ≥ μ for any non-empty timestamps', () => {
    const ts = [0, 0.1, 0.2, 0.5, 1.0];
    expect(hawkesPeakLambda(ts, params)).toBeGreaterThanOrEqual(params.mu);
  });

  it('dense burst ≥ sparse for same params', () => {
    // Dense burst: 10 events in 0.1 s
    const dense = Array.from({ length: 10 }, (_, i) => i * 0.01);
    // Sparse: 10 events in 10 s
    const sparse = Array.from({ length: 10 }, (_, i) => i * 1.0);
    expect(hawkesPeakLambda(dense, params)).toBeGreaterThanOrEqual(
      hawkesPeakLambda(sparse, params),
    );
  });

  it('peak is always finite for finite params and finite timestamps', () => {
    const timestamps = [1e9, 1e9 + 0.001, 1e9 + 0.002, 1e9 + 0.01];
    expect(Number.isFinite(hawkesPeakLambda(timestamps, params))).toBe(true);
  });
});

// ─── 4. hawkesFit: params всегда положительны, stationarity < 1 ──────────────

describe('hawkesFit: always returns valid params for sorted timestamps', () => {
  it('n=10 identical timestamps: mu > 0, alpha > 0, beta > 0', () => {
    const ts = Array.from({ length: 10 }, () => 1_000_000);
    const r = hawkesFit(ts);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.params.alpha).toBeGreaterThan(0);
    expect(r.params.beta).toBeGreaterThan(0);
  });

  it('n=10 evenly spaced: stationarity < 1 (subcritical)', () => {
    const ts = Array.from({ length: 10 }, (_, i) => i * 0.1);
    const r = hawkesFit(ts);
    expect(r.stationarity).toBeGreaterThanOrEqual(0);
    expect(r.stationarity).toBeLessThan(1);
  });

  it('n=50 typical market data: all params positive, stationarity < 1', () => {
    const rng = makeLCG(0xf00d);
    const ts: number[] = [];
    let t = 0;
    for (let i = 0; i < 50; i++) {
      t += rng() * 0.5; // exponential-ish inter-arrival
      ts.push(t);
    }
    const r = hawkesFit(ts);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.params.alpha).toBeGreaterThan(0);
    expect(r.params.beta).toBeGreaterThan(0);
    expect(r.stationarity).toBeLessThan(1);
  });

  it('n < 10: returns converged=false (not enough data)', () => {
    const ts = [0, 0.1, 0.2, 0.3, 0.4];
    const r = hawkesFit(ts);
    expect(r.converged).toBe(false);
    // But params must still be positive
    expect(r.params.mu).toBeGreaterThan(0);
  });

  it('params are always finite', () => {
    const rng = makeLCG(0xbeef);
    for (let trial = 0; trial < 20; trial++) {
      const n = 10 + Math.floor(rng() * 40);
      const ts: number[] = [];
      let t = 0;
      for (let i = 0; i < n; i++) {
        t += rng() * 2;
        ts.push(t);
      }
      const r = hawkesFit(ts);
      expect(Number.isFinite(r.params.mu)).toBe(true);
      expect(Number.isFinite(r.params.alpha)).toBe(true);
      expect(Number.isFinite(r.params.beta)).toBe(true);
    }
  });
});

// ─── 5. cusumUpdate: sPos/sNeg ≥ 0, alarm сбрасывает state ──────────────────

describe('cusumUpdate: state invariants', () => {
  const params = { mu0: 0.3, std0: 0.1, k: 0.05, h: 0.5 };

  it('sPos ≥ 0 always', () => {
    let s = cusumInitState();
    const values = [0, 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 1.0, 0.1];
    for (const v of values) {
      const r = cusumUpdate(s, v, params);
      expect(r.state.sPos).toBeGreaterThanOrEqual(0);
      s = r.state;
    }
  });

  it('sNeg ≥ 0 always', () => {
    let s = cusumInitState();
    const values = [0, 0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 1.0, 0.1];
    for (const v of values) {
      const r = cusumUpdate(s, v, params);
      expect(r.state.sNeg).toBeGreaterThanOrEqual(0);
      s = r.state;
    }
  });

  it('when alarm fires: state.sPos = state.sNeg = state.n = 0', () => {
    // Feed values way above threshold to force alarm
    let s = cusumInitState();
    const bigParams = { mu0: 0, std0: 0.1, k: 0.01, h: 0.1 };
    let alarmFired = false;
    for (let i = 0; i < 50; i++) {
      const r = cusumUpdate(s, 1.0, bigParams);
      if (r.alarm) {
        expect(r.state.sPos).toBe(0);
        expect(r.state.sNeg).toBe(0);
        expect(r.state.n).toBe(0);
        alarmFired = true;
        break;
      }
      s = r.state;
    }
    expect(alarmFired).toBe(true);
  });

  it('alarm fires when s ≥ h: sPos reaches h before alarm', () => {
    const p = { mu0: 0, std0: 1, k: 0, h: 3 };
    let s = cusumInitState();
    // x=1 each step: sPos grows by 1 each time (1 - 0 - 0 = 1)
    for (let i = 0; i < 2; i++) {
      const r = cusumUpdate(s, 1, p);
      expect(r.alarm).toBe(false); // sPos = 1, 2 < 3
      s = r.state;
    }
    // Third step: sPos = 3 ≥ 3 → alarm
    const r3 = cusumUpdate(s, 1, p);
    expect(r3.alarm).toBe(true);
    expect(r3.preResetState.sPos).toBeGreaterThanOrEqual(p.h);
  });

  it('state.sPos and sNeg are always finite', () => {
    const rng = makeLCG(0xcafe);
    let s = cusumInitState();
    for (let i = 0; i < 200; i++) {
      const v = rng();
      const r = cusumUpdate(s, v, params);
      expect(Number.isFinite(r.state.sPos)).toBe(true);
      expect(Number.isFinite(r.state.sNeg)).toBe(true);
      s = r.state;
    }
  });
});

// ─── 6. cusumAnomalyScore: диапазон [0,1], монотонность ──────────────────────

describe('cusumAnomalyScore: range ∈ [0, 1]', () => {
  const params = { mu0: 0.3, std0: 0.1, k: 0.05, h: 0.5 };

  it('returns 0 for zero state', () => {
    expect(cusumAnomalyScore({ sPos: 0, sNeg: 0, n: 0 }, params)).toBe(0);
  });

  it('returns 1 when s = h (alarm threshold)', () => {
    expect(cusumAnomalyScore({ sPos: 0.5, sNeg: 0, n: 5 }, params)).toBe(1);
  });

  it('returns 1 when s > h (above threshold)', () => {
    expect(cusumAnomalyScore({ sPos: 2.0, sNeg: 0, n: 5 }, params)).toBe(1);
  });

  it('score ∈ [0, 1] for s ranging from 0 to 2h', () => {
    for (let s = 0; s <= 2 * params.h; s += 0.05) {
      const score = cusumAnomalyScore({ sPos: s, sNeg: 0, n: 1 }, params);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  it('score is non-decreasing as sPos increases from 0 to h', () => {
    let prev = 0;
    for (let s = 0; s <= params.h; s += 0.02) {
      const score = cusumAnomalyScore({ sPos: s, sNeg: 0, n: 1 }, params);
      expect(score).toBeGreaterThanOrEqual(prev - 1e-12); // allow tiny float error
      prev = score;
    }
  });
});

// ─── 7. bocpdUpdate: mapRL ≥ 0, cpProb ∈ [0,1], нормировка ──────────────────

describe('bocpdUpdate: state invariants', () => {
  const prior = defaultPrior(0.3, 0.05);

  it('mapRunLength ≥ 0 always', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, 0.3 + (i % 3) * 0.05, prior, 200);
      expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
      s = r.state;
    }
  });

  it('mapRunLength is an integer', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, 0.3, prior, 200);
      expect(Number.isInteger(r.mapRunLength)).toBe(true);
      s = r.state;
    }
  });

  it('cpProbability ∈ [0, 1] always', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 30; i++) {
      const r = bocpdUpdate(s, 0.3 + (i % 5) * 0.04, prior, 200);
      expect(r.cpProbability).toBeGreaterThanOrEqual(0);
      expect(r.cpProbability).toBeLessThanOrEqual(1);
      s = r.state;
    }
  });

  it('cpProbability is never NaN', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 30; i++) {
      const r = bocpdUpdate(s, 0.3, prior, 200);
      expect(Number.isNaN(r.cpProbability)).toBe(false);
      s = r.state;
    }
  });

  it('sum of exp(logProbs) ≤ 1 + 1e-9 after each step (normalization)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 50; i++) {
      const r = bocpdUpdate(s, 0.3 + (i % 4) * 0.05, prior, 200);
      const sumProb = r.state.logProbs.reduce((acc, lp) => acc + Math.exp(lp), 0);
      expect(sumProb).toBeLessThanOrEqual(1 + 1e-9);
      expect(sumProb).toBeGreaterThan(0);
      s = r.state;
    }
  });

  it('state.logProbs are never +Infinity (only finite or -Infinity)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 30; i++) {
      const r = bocpdUpdate(s, 0.3, prior, 200);
      for (const lp of r.state.logProbs) {
        expect(lp).toBeLessThanOrEqual(0); // log-probs ≤ 0 since probs ≤ 1
        expect(Number.isNaN(lp)).toBe(false);
      }
      s = r.state;
    }
  });
});

// ─── 8. bocpdUpdate: mapRL растёт по 1 в стационарном режиме ─────────────────
//
// В стационарном процессе с высоким hazardLambda (редкие changepoint'ы)
// MAP run-length должен расти на 1 каждый шаг.

describe('bocpdUpdate: mapRunLength grows by 1 per step in stable IID data', () => {
  // IID constant observations → the "no changepoint" hypothesis dominates
  const prior = defaultPrior(0.5, 0.01);
  const hazardLambda = 10_000; // очень редкие changepoint'ы

  it('mapRL = t after t steps with constant data (large hazardLambda)', () => {
    let s = bocpdInitState();
    for (let t = 1; t <= 30; t++) {
      const r = bocpdUpdate(s, 0.5, prior, hazardLambda);
      // MAP should pick the longest run (r=t) since data is constant and
      // the hazard is tiny — growth hypothesis always wins
      expect(r.mapRunLength).toBe(t);
      s = r.state;
    }
  });
});

// ─── 9. bocpdAnomalyScore: диапазон, монотонность по дропу ───────────────────

describe('bocpdAnomalyScore: range and monotonicity', () => {
  it('returns 0 when prevRunLength ≤ 0', () => {
    const dummy = { state: bocpdInitState(), mapRunLength: 1, cpProbability: 0 };
    expect(bocpdAnomalyScore(dummy, 0)).toBe(0);
    expect(bocpdAnomalyScore(dummy, -5)).toBe(0);
  });

  it('score increases as drop increases', () => {
    // Create result objects with different mapRunLengths
    const makeResult = (rl: number) => ({
      state: bocpdInitState(),
      mapRunLength: rl,
      cpProbability: 0,
    });

    const prevRL = 100;
    const scoreSmallDrop = bocpdAnomalyScore(makeResult(90), prevRL); // drop=0.1
    const scoreMedDrop   = bocpdAnomalyScore(makeResult(50), prevRL); // drop=0.5
    const scoreLargeDrop = bocpdAnomalyScore(makeResult(1),  prevRL); // drop≈0.99

    expect(scoreMedDrop).toBeGreaterThan(scoreSmallDrop);
    expect(scoreLargeDrop).toBeGreaterThan(scoreMedDrop);
  });

  it('score ∈ (0, 1) for any valid drop', () => {
    const prevRL = 100;
    for (let rl = 0; rl <= 100; rl += 10) {
      const result = { state: bocpdInitState(), mapRunLength: rl, cpProbability: 0 };
      const score = bocpdAnomalyScore(result, prevRL);
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThan(1);
    }
  });

  it('score is never NaN for any finite prevRunLength > 0', () => {
    const prevRLs = [0.001, 1, 10, 1e6];
    for (const prevRL of prevRLs) {
      const result = { state: bocpdInitState(), mapRunLength: 0, cpProbability: 0 };
      expect(Number.isNaN(bocpdAnomalyScore(result, prevRL))).toBe(false);
    }
  });
});

// ─── 10. detect(): confidence ∈ [0,1], never NaN ─────────────────────────────

describe('detector: confidence ∈ [0, 1] and never NaN', () => {
  it('confidence ∈ [0,1] for calm baseline window', () => {
    const det = trainedDetector();
    const result = det.detect(genTrades(200, 0xdead));
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(Number.isNaN(result.confidence)).toBe(false);
  });

  it('confidence ∈ [0,1] for burst window (all buys)', () => {
    const det = trainedDetector();
    const burst = Array.from({ length: 200 }, (_, i) =>
      trade(Date.now() + i, 10, false),
    );
    const result = det.detect(burst);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(Number.isNaN(result.confidence)).toBe(false);
  });

  it('confidence ∈ [0,1] for single-trade window', () => {
    const det = trainedDetector();
    const result = det.detect([trade(1_000_000, 5, false)]);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(Number.isNaN(result.confidence)).toBe(false);
  });

  it('confidence ∈ [0,1] for window with identical timestamps', () => {
    const det = trainedDetector();
    const sameTs = Array.from({ length: 100 }, () => trade(1_000_000, 1, false));
    const result = det.detect(sameTs);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(Number.isNaN(result.confidence)).toBe(false);
  });

  it('confidence ∈ [0,1] for 20 different seeds (property-like)', () => {
    const det = trainedDetector(0xaaaa);
    const rng = makeLCG(0xbbbb);
    for (let t = 0; t < 20; t++) {
      const trades = Array.from({ length: 50 + Math.floor(rng() * 150) }, (_, i) =>
        trade(i * (50 + Math.floor(rng() * 100)), rng() * 10 + 0.1, rng() < 0.5),
      );
      const r = det.detect(trades);
      expect(r.confidence).toBeGreaterThanOrEqual(0);
      expect(r.confidence).toBeLessThanOrEqual(1);
      expect(Number.isNaN(r.confidence)).toBe(false);
    }
  });
});

// ─── 11. detect(): anomaly ↔ confidence ≥ threshold ─────────────────────────

describe('detector: anomaly is exactly (confidence >= threshold)', () => {
  const thresholds = [0.5, 0.6, 0.75, 0.8, 0.9];

  it('anomaly = (confidence >= 0.75) with default threshold', () => {
    const det = trainedDetector();
    const result = det.detect(genTrades(200, 0x1234));
    expect(result.anomaly).toBe(result.confidence >= 0.75);
  });

  it('consistent for all thresholds from 0 to 1', () => {
    const det = trainedDetector();
    const trades = genTrades(200, 0x5678);
    for (const thr of thresholds) {
      const r = det.detect(trades, thr);
      expect(r.anomaly).toBe(r.confidence >= thr);
    }
  });

  it('anomaly flag is boolean (not truthy/falsy)', () => {
    const det = trainedDetector();
    const r = det.detect(genTrades(100, 0xabcd));
    expect(typeof r.anomaly).toBe('boolean');
  });
});

// ─── 12. detect(): детерминизм ────────────────────────────────────────────────

describe('detector: determinism', () => {
  it('same trades → same confidence across two calls', () => {
    const det = trainedDetector(0x9999);
    const trades = genTrades(200, 0x8888);
    const r1 = det.detect(trades);
    const r2 = det.detect(trades);
    expect(r1.confidence).toBe(r2.confidence);
    expect(r1.anomaly).toBe(r2.anomaly);
    expect(r1.imbalance).toBe(r2.imbalance);
  });

  it('same trades → same result from two independently trained detectors', () => {
    // Two detectors with identical training data should give identical results
    const hist = genTrades(200, 0x7777);
    const window = genTrades(100, 0x6666);

    const det1 = new VolumeAnomalyDetector();
    det1.train([...hist]);
    const det2 = new VolumeAnomalyDetector();
    det2.train([...hist]);

    const r1 = det1.detect(window);
    const r2 = det2.detect(window);
    expect(r1.confidence).toBe(r2.confidence);
  });
});

// ─── 13. scoreWeights: валидация отрицательных весов ─────────────────────────

describe('detector constructor: scoreWeights validation', () => {
  it('negative weight throws even if sum = 1', () => {
    // [1.5, -0.3, -0.2] sums to 1 but has negative weights
    // With these weights: combined = 1.5*score - 0.3*score - 0.2*score
    // hawkesScore=1 would give combined = 1.5 → confidence > 1
    expect(() => new VolumeAnomalyDetector({
      scoreWeights: [1.5, -0.3, -0.2],
    })).toThrow('scoreWeights');
  });

  it('negative weight throws for [0.6, -0.1, 0.5]', () => {
    expect(() => new VolumeAnomalyDetector({
      scoreWeights: [0.6, -0.1, 0.5],
    })).toThrow('scoreWeights');
  });

  it('all-zero weight fails sum check, not negativity check', () => {
    expect(() => new VolumeAnomalyDetector({
      scoreWeights: [0, 0, 0],
    })).toThrow('scoreWeights');
  });

  it('valid weights [0.5, 0.3, 0.2] do not throw', () => {
    expect(() => new VolumeAnomalyDetector({
      scoreWeights: [0.5, 0.3, 0.2],
    })).not.toThrow();
  });

  it('confidence ∈ [0,1] is guaranteed with any valid non-negative weights', () => {
    // Extreme valid weights: [1, 0, 0] — only Hawkes matters
    const det = new VolumeAnomalyDetector({ scoreWeights: [1, 0, 0] });
    det.train(genTrades(200, 0x1234));
    const r = det.detect(genTrades(200, 0x5678));
    expect(r.confidence).toBeGreaterThanOrEqual(0);
    expect(r.confidence).toBeLessThanOrEqual(1);
  });
});

// ─── 14. property-based: 100 случаев, confidence ∈ [0,1] ─────────────────────
//
// Прогоняем через детектор 100 различных торговых паттернов.
// Для каждого проверяем строгие инварианты вывода.

describe('property-based: confidence ∈ [0,1] and anomaly consistent for 100 random patterns', () => {
  const det = trainedDetector(0x1234_5678);

  it('all 100 random windows satisfy confidence ∈ [0,1] and anomaly consistency', () => {
    const rng = makeLCG(0xdead_beef);
    let failures = 0;

    for (let trial = 0; trial < 100; trial++) {
      const n = 10 + Math.floor(rng() * 190);
      const burstFrac = rng();
      const baseTs = Math.floor(rng() * 1_000_000_000);

      const trades: IAggregatedTradeData[] = [];
      for (let i = 0; i < n; i++) {
        // Vary inter-arrival times from 1ms to 10s
        const dt = Math.floor(1 + rng() * 10_000);
        trades.push(trade(baseTs + i * dt, 0.01 + rng() * 100, rng() < burstFrac));
      }

      const threshold = 0.5 + rng() * 0.4; // 0.5..0.9
      const r = det.detect(trades, threshold);

      if (Number.isNaN(r.confidence)) failures++;
      if (r.confidence < 0 || r.confidence > 1) failures++;
      if (r.anomaly !== (r.confidence >= threshold)) failures++;
    }

    expect(failures).toBe(0);
  });
});

// ─── 15. cusumFit + cusumUpdate: интеграция ───────────────────────────────────
//
// При обучении на спокойных данных и подаче таких же данных,
// CUSUM score должен оставаться низким (нет ложных тревог).

describe('cusumFit + cusumUpdate: no false alarms on in-control data', () => {
  it('trained on IID uniform: score < 1 for similar data', () => {
    const rng = makeLCG(0x1357);
    const training = Array.from({ length: 500 }, () => rng() * 0.5);
    const params = cusumFit(training);

    // Feed same distribution data
    let alarms = 0;
    let s = cusumInitState();
    for (let i = 0; i < 500; i++) {
      const r = cusumUpdate(s, rng() * 0.5, params);
      if (r.alarm) alarms++;
      s = r.state;
    }
    // h = 5σ → ARL₀ ≈ 148, so expect roughly 500/148 ≈ 3 alarms
    // Allow up to 20 alarms (3.3σ from mean for geometric count)
    expect(alarms).toBeLessThan(20);
  });

  it('score jumps to 1 immediately for out-of-control burst', () => {
    const params = cusumFit(Array.from({ length: 100 }, () => 0.1));
    // Feed x = 100× mu0 — way out of control
    let s = cusumInitState();
    let reachedOne = false;
    for (let i = 0; i < 20; i++) {
      const r = cusumUpdate(s, 10.0, params);
      if (cusumAnomalyScore(r.preResetState, params) >= 1) {
        reachedOne = true;
        break;
      }
      s = r.state;
    }
    expect(reachedOne).toBe(true);
  });
});

// ─── 16. hawkesAnomalyScore: μ=0, α=0, β=0 крайние params ────────────────────
//
// Это сценарий когда hawkesFit возвращает нулевые params (n < 10).
// Результат ОБЯЗАН быть конечным.

describe('hawkesAnomalyScore: degenerate params (mu=0 or alpha=beta=0)', () => {
  it('mu=0, alpha=0, beta=1: score ∈ [0,1]', () => {
    const s = hawkesAnomalyScore(0, { mu: 0, alpha: 0, beta: 1 }, 0);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it('mu=0, alpha=0.01, beta=1: with empiricalRate=5 → finite score', () => {
    // params from n<10 fallback: mu=0 (for n=0), alpha=0.01, beta=1
    const s = hawkesAnomalyScore(0, { mu: 0, alpha: 0.01, beta: 1 }, 5);
    expect(Number.isFinite(s)).toBe(true);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it('mu=0, peakLambda=0, empiricalRate=0: score = 0', () => {
    // No signal whatsoever → score = 0
    const s = hawkesAnomalyScore(0, { mu: 0, alpha: 0.01, beta: 1 }, 0);
    expect(s).toBe(0);
  });
});

// ─── 17. BOCPD нормировка не дрейфует за 1000 шагов ─────────────────────────

describe('bocpdUpdate: normalization stays stable over 1000 steps', () => {
  it('sum(exp(logProbs)) stays in [0.99, 1+1e-9] over 1000 steps', () => {
    const prior = defaultPrior(0.4, 0.02);
    const rng = makeLCG(0x2468);
    let s = bocpdInitState();
    let minSum = Infinity;
    let maxSum = 0;

    for (let i = 0; i < 1000; i++) {
      // IID data around the prior mean
      const x = 0.35 + rng() * 0.1;
      const r = bocpdUpdate(s, x, prior, 200);
      const sumProb = r.state.logProbs.reduce((acc, lp) => acc + Math.exp(lp), 0);
      if (sumProb < minSum) minSum = sumProb;
      if (sumProb > maxSum) maxSum = sumProb;
      s = r.state;
    }

    expect(maxSum).toBeLessThanOrEqual(1 + 1e-9);
    expect(minSum).toBeGreaterThan(0.99); // < 1% mass pruned
  });
});
