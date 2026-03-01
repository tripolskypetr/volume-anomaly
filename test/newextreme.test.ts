/**
 * newextreme.test.ts — additional "stuck at extremum" coverage.
 *
 * Each group targets a specific gap identified via static analysis of the math
 * modules that was NOT already covered by extreme.test.ts or adversarial.test.ts.
 *
 * Tests are written against CORRECT expected behaviour.
 * Tests that reveal bugs will fail until the underlying code is fixed.
 *
 *  1. cusumUpdate: x = NaN   — state must not be poisoned with NaN
 *  2. cusumAnomalyScore: h = NaN — guard must fire (currently h<=0 doesn't catch NaN)
 *  3. bocpdUpdate: x = NaN   — state collapses cleanly, cpProbability must be finite
 *  4. hawkesAnomalyScore: peakLambda = Infinity — must return 1
 *  5. hawkesAnomalyScore: peakLambda = 0, empiricalRate = 0 — returns ~0.018, not 0
 *  6. hawkesAnomalyScore: empiricalRate = Infinity — must return 1
 *  7. hawkesAnomalyScore: NaN params — must not crash
 *  8. cusumUpdate: x = ±Infinity — alarm fires immediately, clean reset
 *  9. bocpdUpdate: kappa0 = 0 — state collapses, no crash, cpProbability finite
 * 10. hawkesLogLikelihood: unsorted timestamps — documents behaviour (finite, wrong)
 * 11. hawkesFit: n = 10 exactly — boundary of the < 10 fallback branch
 * 12. nelderMead: 1D simplex — works correctly
 * 13. nelderMead: 0D simplex — converges immediately
 * 14. bocpdBatch: empty series — returns empty arrays, no crash
 * 15. bocpdAnomalyScore: prevRunLength = Infinity — drop clamped, score finite
 */

import { describe, it, expect } from 'vitest';
import {
  cusumUpdate,
  cusumInitState,
  cusumAnomalyScore,
} from '../src/math/cusum.js';
import {
  bocpdUpdate,
  bocpdInitState,
  bocpdBatch,
  bocpdAnomalyScore,
} from '../src/math/bocpd.js';
import {
  hawkesAnomalyScore,
  hawkesLogLikelihood,
  hawkesFit,
} from '../src/math/hawkes.js';
import { nelderMead } from '../src/math/optimizer.js';

// ─── 1. cusumUpdate: x = NaN ──────────────────────────────────────────────────
//
// Math.max(0, NaN) = NaN in JS — the CUSUM accumulator gets poisoned silently.
// All subsequent steps also return NaN state, and confidence becomes NaN.
//
// Correct behaviour: NaN input should not mutate the accumulator.
// Contract: after feeding NaN, state.sPos and state.sNeg must be finite.

describe('cusumUpdate: x = NaN — state must not be poisoned', () => {
  const params = { mu0: 0.2, std0: 0.1, k: 0.05, h: 0.5 };

  it('state.sPos is finite after feeding NaN (not poisoned)', () => {
    const upd = cusumUpdate(cusumInitState(), NaN, params);
    expect(Number.isFinite(upd.state.sPos)).toBe(true);
  });

  it('state.sNeg is finite after feeding NaN (not poisoned)', () => {
    const upd = cusumUpdate(cusumInitState(), NaN, params);
    expect(Number.isFinite(upd.state.sNeg)).toBe(true);
  });

  it('preResetState.sPos is finite after NaN', () => {
    const upd = cusumUpdate(cusumInitState(), NaN, params);
    expect(Number.isFinite(upd.preResetState.sPos)).toBe(true);
  });

  it('alarm does not fire on NaN input (no spurious alarm)', () => {
    const upd = cusumUpdate(cusumInitState(), NaN, params);
    expect(upd.alarm).toBe(false);
  });

  it('subsequent updates after NaN remain finite (no propagation)', () => {
    let state = cusumInitState();
    const upd0 = cusumUpdate(state, NaN, params);
    state = upd0.state;
    const upd1 = cusumUpdate(state, 0.3, params);
    expect(Number.isFinite(upd1.state.sPos)).toBe(true);
    expect(Number.isFinite(upd1.state.sNeg)).toBe(true);
  });

  it('cusumAnomalyScore after NaN update returns finite value in [0,1]', () => {
    const upd = cusumUpdate(cusumInitState(), NaN, params);
    const score = cusumAnomalyScore(upd.state, params);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});

// ─── 2. cusumAnomalyScore: h = NaN ────────────────────────────────────────────
//
// The guard `if (params.h <= 0) return 0` does NOT catch NaN:
//   NaN <= 0  →  false  (IEEE 754 — all comparisons with NaN are false)
//   min(s / NaN, 1)  →  min(NaN, 1)  →  NaN  (Math.min(NaN, x) = NaN)
//
// Correct behaviour: return 0 (or any finite value in [0,1]) when h is not
// a positive finite number.

describe('cusumAnomalyScore: h = NaN — guard must fire, result must be finite', () => {
  const state = { sPos: 0.5, sNeg: 0.3, n: 10 };

  it('returns finite value when h = NaN', () => {
    const score = cusumAnomalyScore(state, { mu0: 0, std0: 1, k: 0, h: NaN });
    expect(Number.isFinite(score)).toBe(true);
  });

  it('returns 0 when h = NaN (same contract as h <= 0)', () => {
    expect(cusumAnomalyScore(state, { mu0: 0, std0: 1, k: 0, h: NaN })).toBe(0);
  });

  it('returns finite value when h = Infinity', () => {
    // h=Infinity: min(s/Infinity, 1) = min(0, 1) = 0 — already correct in JS
    const score = cusumAnomalyScore(state, { mu0: 0, std0: 1, k: 0, h: Infinity });
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBe(0); // s/Inf = 0
  });

  it('returns 0 when h = -Infinity (covered by h <= 0 guard)', () => {
    expect(cusumAnomalyScore(state, { mu0: 0, std0: 1, k: 0, h: -Infinity })).toBe(0);
  });
});

// ─── 3. bocpdUpdate: x = NaN ──────────────────────────────────────────────────
//
// With x=NaN: studentTPredLogProb returns NaN → all jointLogProbs become NaN →
// normLogProbs = [NaN, NaN] → keep = [false, false] → state collapses to [].
// mapRunLength = 0 (correct: no hypothesis survives).
//
// The subtle bug: cpProbability = Math.exp(normLogProbs[0] ?? -Infinity) = Math.exp(NaN) = NaN.
// The ?? guard does not trigger because normLogProbs[0] = NaN, not undefined.
//
// Correct behaviour: cpProbability must be a finite number in [0,1].

describe('bocpdUpdate: x = NaN — state collapse clean, cpProbability finite', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 1 };

  it('does not crash', () => {
    expect(() => bocpdUpdate(bocpdInitState(), NaN, prior)).not.toThrow();
  });

  it('mapRunLength is a non-negative integer', () => {
    const r = bocpdUpdate(bocpdInitState(), NaN, prior);
    expect(Number.isInteger(r.mapRunLength)).toBe(true);
    expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
  });

  it('state collapses to empty (all hypotheses pruned because logProbs = NaN)', () => {
    const r = bocpdUpdate(bocpdInitState(), NaN, prior);
    expect(r.state.logProbs.length).toBe(0);
  });

  it('cpProbability is finite and in [0,1]', () => {
    const r = bocpdUpdate(bocpdInitState(), NaN, prior);
    expect(Number.isFinite(r.cpProbability)).toBe(true);
    expect(r.cpProbability).toBeGreaterThanOrEqual(0);
    expect(r.cpProbability).toBeLessThanOrEqual(1);
  });

  it('bocpdAnomalyScore does not return NaN after NaN x', () => {
    const r = bocpdUpdate(bocpdInitState(), NaN, prior);
    const score = bocpdAnomalyScore(r, 10);
    expect(Number.isFinite(score)).toBe(true);
  });
});

// ─── 4. hawkesAnomalyScore: peakLambda = Infinity ────────────────────────────
//
// sig(Infinity / meanLambda) = sig(+∞) = 1 — correct behaviour.
// Verify: no NaN, returns exactly 1.

describe('hawkesAnomalyScore: peakLambda = Infinity returns 1', () => {
  const params = { mu: 1, alpha: 0.3, beta: 2 };

  it('peakLambda = Infinity → score = 1 (not NaN)', () => {
    const score = hawkesAnomalyScore(Infinity, params, 0);
    expect(score).toBe(1);
  });

  it('peakLambda = Number.MAX_VALUE → score = 1', () => {
    const score = hawkesAnomalyScore(Number.MAX_VALUE, params, 0);
    expect(score).toBe(1);
  });

  it('peakLambda = -Infinity → score is finite and in [0,1]', () => {
    // sig(-Inf/meanLambda) = sig(-Inf) = 0 — benign
    const score = hawkesAnomalyScore(-Infinity, params, 0);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});

// ─── 5. hawkesAnomalyScore: peakLambda = 0, empiricalRate = 0 ────────────────
//
// intensityScore = sig(0 / meanLambda) = sig(0) = 1/(1+e^4) ≈ 0.018.
// rateScore = 0 (empiricalRate = 0).
// Final score ≈ 0.018 — NOT zero, NOT NaN.
//
// This matters: at baseline with no arrivals in window, the score should be
// near-zero but strictly positive (the sigmoid is centred at 2×, not at 1×).

describe('hawkesAnomalyScore: peakLambda = 0, empiricalRate = 0 → ~0.018', () => {
  const params = { mu: 1, alpha: 0.3, beta: 2 };

  it('score is > 0 (sigmoid is never exactly 0)', () => {
    const score = hawkesAnomalyScore(0, params, 0);
    expect(score).toBeGreaterThan(0);
  });

  it('score is < 0.1 (far below anomaly threshold)', () => {
    const score = hawkesAnomalyScore(0, params, 0);
    expect(score).toBeLessThan(0.1);
  });

  it('score is finite (not NaN)', () => {
    const score = hawkesAnomalyScore(0, params, 0);
    expect(Number.isFinite(score)).toBe(true);
  });

  it('score ≈ 0.018 (sig(0-2)*2 = sig(-4))', () => {
    const score = hawkesAnomalyScore(0, params, 0);
    expect(score).toBeCloseTo(1 / (1 + Math.exp(4)), 4);
  });
});

// ─── 6. hawkesAnomalyScore: empiricalRate = Infinity ─────────────────────────
//
// rateScore = sig(Infinity / mu) = sig(+∞) = 1.
// Must return 1, not NaN.

describe('hawkesAnomalyScore: empiricalRate = Infinity returns 1', () => {
  const params = { mu: 1, alpha: 0.3, beta: 2 };

  it('empiricalRate = Infinity → score = 1', () => {
    expect(hawkesAnomalyScore(params.mu, params, Infinity)).toBe(1);
  });

  it('empiricalRate = Number.MAX_VALUE → score = 1', () => {
    expect(hawkesAnomalyScore(params.mu, params, Number.MAX_VALUE)).toBe(1);
  });
});

// ─── 7. hawkesAnomalyScore: NaN params ───────────────────────────────────────
//
// With NaN params: branching = NaN → NaN >= 1 = false (guard skipped).
// meanLambda = NaN/(1-NaN) = NaN → meanLambda > 0 = false → falls to peakLambda > 0 ? 1 : 0.
// Must not crash; returns a value in [0,1].

describe('hawkesAnomalyScore: NaN params — does not crash, result in [0,1]', () => {
  it('params = all NaN, peakLambda = 1 → does not crash', () => {
    expect(() =>
      hawkesAnomalyScore(1, { mu: NaN, alpha: NaN, beta: NaN }, 0),
    ).not.toThrow();
  });

  it('result is in [0,1] for any NaN param combination', () => {
    const score = hawkesAnomalyScore(1, { mu: NaN, alpha: NaN, beta: NaN }, 0);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it('params.mu = NaN, others valid → does not produce Infinity score', () => {
    const score = hawkesAnomalyScore(2, { mu: NaN, alpha: 0.3, beta: 2 }, 0);
    expect(Number.isFinite(score)).toBe(true);
  });
});

// ─── 8. cusumUpdate: x = ±Infinity ───────────────────────────────────────────
//
// +Infinity: sPos = max(0, Inf - mu0 - k) = Inf ≥ h → alarm fires, state resets.
// -Infinity: sNeg = max(0, Inf) = Inf ≥ h → alarm fires, state resets.
// State after reset must be clean (sPos = sNeg = n = 0).

describe('cusumUpdate: x = ±Infinity — alarm fires, clean reset', () => {
  const params = { mu0: 0.2, std0: 0.1, k: 0.05, h: 0.5 };

  it('x = +Infinity → alarm fires', () => {
    const upd = cusumUpdate(cusumInitState(), Infinity, params);
    expect(upd.alarm).toBe(true);
  });

  it('x = +Infinity → state resets cleanly (sPos = sNeg = 0)', () => {
    const upd = cusumUpdate(cusumInitState(), Infinity, params);
    expect(upd.state.sPos).toBe(0);
    expect(upd.state.sNeg).toBe(0);
  });

  it('x = -Infinity → alarm fires (sNeg overflows threshold)', () => {
    const upd = cusumUpdate(cusumInitState(), -Infinity, params);
    expect(upd.alarm).toBe(true);
  });

  it('x = -Infinity → state resets cleanly', () => {
    const upd = cusumUpdate(cusumInitState(), -Infinity, params);
    expect(upd.state.sPos).toBe(0);
    expect(upd.state.sNeg).toBe(0);
  });

  it('preResetState.sPos = Infinity when x = +Infinity (captures peak before reset)', () => {
    const upd = cusumUpdate(cusumInitState(), Infinity, params);
    expect(upd.preResetState.sPos).toBe(Infinity);
  });
});

// ─── 9. bocpdUpdate: kappa0 = 0 ──────────────────────────────────────────────
//
// kappa0 = 0 → kappaN = 0 → muN = (0·mu0 + n·mean)/0 = NaN for n=0.
// cross-term = kappa0·n·(...) / (2·kappaN) = 0/0 = NaN → betaN = NaN → scale = NaN.
// All logProbs become NaN → state collapses.
//
// Same collapse pattern as x=NaN or hazardLambda < 1.
// Correct contract: no crash, cpProbability finite, mapRunLength ≥ 0.

describe('bocpdUpdate: kappa0 = 0 — collapses cleanly, cpProbability finite', () => {
  const prior0 = { mu0: 0.5, kappa0: 0, alpha0: 1, beta0: 1 };

  it('does not crash', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior0)).not.toThrow();
  });

  it('mapRunLength is non-negative integer', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior0);
    expect(Number.isInteger(r.mapRunLength)).toBe(true);
    expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
  });

  it('cpProbability is finite and in [0,1]', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior0);
    expect(Number.isFinite(r.cpProbability)).toBe(true);
    expect(r.cpProbability).toBeGreaterThanOrEqual(0);
    expect(r.cpProbability).toBeLessThanOrEqual(1);
  });

  it('10 consecutive updates do not crash', () => {
    let s = bocpdInitState();
    expect(() => {
      for (let i = 0; i < 10; i++) {
        const r = bocpdUpdate(s, 0.3 + i * 0.05, prior0);
        s = r.state;
      }
    }).not.toThrow();
  });

  it('cpProbability stays finite across 5 steps', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 5; i++) {
      const r = bocpdUpdate(s, 0.5, prior0);
      expect(Number.isFinite(r.cpProbability)).toBe(true);
      s = r.state;
    }
  });
});

// ─── 10. hawkesLogLikelihood: unsorted timestamps ─────────────────────────────
//
// The function requires ascending timestamps but does not validate order.
// With reversed input: t0 = largest value, T = last − first < 0 (negative window).
// Compensator term: (alpha/beta)*(1 - exp(-beta*(T-ti))) with T-ti < 0 adds positive mass
// instead of subtracting it, inflating the LL into large positive territory.
//
// This is documented incorrect behaviour (not a crash, but wrong math).
// The test pins the behaviour so we notice if it changes.

describe('hawkesLogLikelihood: unsorted timestamps — documents wrong-but-finite behaviour', () => {
  const params = { mu: 1, alpha: 0.3, beta: 2 };
  const sorted   = [0, 1, 2, 3, 4];
  const reversed = [4, 3, 2, 1, 0];

  it('reversed timestamps do not crash', () => {
    expect(() => hawkesLogLikelihood(reversed, params)).not.toThrow();
  });

  it('reversed timestamps return a finite number (not crash, not NaN)', () => {
    expect(Number.isFinite(hawkesLogLikelihood(reversed, params))).toBe(true);
  });

  it('reversed LL ≠ sorted LL (order matters — sanity check)', () => {
    const llSorted   = hawkesLogLikelihood(sorted,   params);
    const llReversed = hawkesLogLikelihood(reversed, params);
    expect(llReversed).not.toBeCloseTo(llSorted, 3);
  });

  it('reversed LL is unrealistically large (inflated by negative T)', () => {
    // With T < 0 the compensator contributes positive mass, inflating LL.
    // The correct LL for 5 events in 4 seconds with mu=1 is well-negative.
    const llSorted   = hawkesLogLikelihood(sorted,   params);
    const llReversed = hawkesLogLikelihood(reversed, params);
    expect(llReversed).toBeGreaterThan(llSorted);
  });
});

// ─── 11. hawkesFit: n = 10 exactly (boundary of the < 10 fallback) ────────────
//
// hawkesFit skips the optimizer for n < 10 and returns a Poisson fallback.
// At n = 10, it runs Nelder-Mead for the first time.
// Test: n=9 → converged=false (fallback); n=10 → params > 0, stationarity ≥ 0.

describe('hawkesFit: boundary at n = 10', () => {
  it('n = 9 → converged = false (Poisson fallback, no optimizer)', () => {
    const ts = Array.from({ length: 9 }, (_, i) => i * 0.5);
    const r  = hawkesFit(ts);
    expect(r.converged).toBe(false);
    expect(r.logLik).toBe(-Infinity);
  });

  it('n = 10 → runs optimizer, returns positive params and finite stationarity', () => {
    const ts = Array.from({ length: 10 }, (_, i) => i * 0.5);
    const r  = hawkesFit(ts);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.params.alpha).toBeGreaterThan(0);
    expect(r.params.beta).toBeGreaterThan(0);
    expect(Number.isFinite(r.stationarity)).toBe(true);
    expect(r.stationarity).toBeGreaterThanOrEqual(0);
  });

  it('n = 10 → stationarity < 1 (stationary process, alpha < beta)', () => {
    const ts = Array.from({ length: 10 }, (_, i) => i * 0.5);
    const r  = hawkesFit(ts);
    // Either converged with valid params, or fell back to 0.01 stationarity
    expect(r.stationarity).toBeLessThan(1);
  });

  it('n = 11 → also runs optimizer, same contracts', () => {
    const ts = Array.from({ length: 11 }, (_, i) => i * 0.5);
    const r  = hawkesFit(ts);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.stationarity).toBeGreaterThanOrEqual(0);
    expect(r.stationarity).toBeLessThan(1);
  });
});

// ─── 12. nelderMead: 1D simplex ───────────────────────────────────────────────
//
// n=1 creates a simplex with 2 vertices.  The centroid loop runs `for i<1`
// (one vertex, excluding the worst), which is correct.
// The 1D case should find the minimum of a simple quadratic.

describe('nelderMead: 1D simplex', () => {
  it('converges to minimum of (x-3)^2 from x0=[0]', () => {
    const f = ([x]: number[]) => (x! - 3) ** 2;
    const r = nelderMead(f, [0], { maxIter: 500, tol: 1e-8 });
    expect(Number.isFinite(r.fx)).toBe(true);
    expect(r.fx).toBeLessThan(0.01);
    expect(r.x[0]).toBeCloseTo(3, 1);
  });

  it('does not crash for 1D', () => {
    const f = ([x]: number[]) => x! ** 2;
    expect(() => nelderMead(f, [5], { maxIter: 200 })).not.toThrow();
  });

  it('result x has exactly 1 component', () => {
    const f = ([x]: number[]) => (x! - 7) ** 2;
    const r = nelderMead(f, [0], { maxIter: 200 });
    expect(r.x.length).toBe(1);
  });
});

// ─── 13. nelderMead: 0D simplex ───────────────────────────────────────────────
//
// n=0, simplex = [[]]. sortSimplex loops i=0..0 once. Spread = f([]) - f([]) = 0 < tol.
// Expected: converges at iter=0, returns { x: [], fx: f([]), converged: true }.

describe('nelderMead: 0D simplex (empty x0)', () => {
  it('does not crash', () => {
    expect(() => nelderMead(() => 42, [], { maxIter: 100 })).not.toThrow();
  });

  it('converges immediately (spread = 0, no dimensions to search)', () => {
    const r = nelderMead(() => 42, [], { maxIter: 100 });
    expect(r.converged).toBe(true);
    expect(r.iters).toBe(0);
  });

  it('returns fx = f([])', () => {
    const r = nelderMead(() => 7, [], { maxIter: 100 });
    expect(r.fx).toBe(7);
  });

  it('returns x = []', () => {
    const r = nelderMead(() => 0, [], { maxIter: 100 });
    expect(r.x.length).toBe(0);
  });
});

// ─── 14. bocpdBatch: empty series ────────────────────────────────────────────
//
// The for-loop over an empty array executes 0 times.
// Expected: { cpProbs: [], mapRunLengths: [] }, no crash.

describe('bocpdBatch: empty series', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 1 };

  it('does not crash on empty series', () => {
    expect(() => bocpdBatch([], prior)).not.toThrow();
  });

  it('returns empty cpProbs array', () => {
    expect(bocpdBatch([], prior).cpProbs).toHaveLength(0);
  });

  it('returns empty mapRunLengths array', () => {
    expect(bocpdBatch([], prior).mapRunLengths).toHaveLength(0);
  });
});

// ─── 15. bocpdAnomalyScore: extreme prevRunLength values ─────────────────────
//
// prevRunLength = Infinity: drop = (Inf - mapRL) / Inf.
//   If mapRL is finite: (Inf - finite)/Inf = Inf/Inf = NaN → Math.max(0, NaN) = NaN
//   → sigmoid(NaN) = NaN.  Must return a finite score.
//
// prevRunLength = NaN: guard `prevRunLength <= 0` → NaN <= 0 = false → falls through.
//   drop = (NaN - mapRL) / NaN = NaN → same NaN path.  Must return finite.

describe('bocpdAnomalyScore: extreme prevRunLength values', () => {
  const fakeResult = { mapRunLength: 5, cpProbability: 0.01, state: bocpdInitState() };

  it('prevRunLength = Infinity → finite score in [0,1]', () => {
    const score = bocpdAnomalyScore(fakeResult, Infinity);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it('prevRunLength = NaN → finite score in [0,1]', () => {
    const score = bocpdAnomalyScore(fakeResult, NaN);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it('prevRunLength = Infinity, mapRunLength = 0 → score in [0,1]', () => {
    const r = { mapRunLength: 0, cpProbability: 0.01, state: bocpdInitState() };
    const score = bocpdAnomalyScore(r, Infinity);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});
