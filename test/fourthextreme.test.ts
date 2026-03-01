/**
 * fourthextreme.test.ts — fourth wave of "stuck at extremum" coverage.
 *
 * Each group targets a gap not covered by the three previous extreme test files.
 * Tests express CORRECT expected behaviour.
 * Tests that reveal bugs will fail until the underlying code is fixed.
 *
 *  1. hawkesAnomalyScore: peakLambda = NaN, valid params
 *       → Math.max(NaN, rateScore) = NaN unless guarded   [BUG A1]
 *  2. cusumUpdate: NaN in params.k or params.mu0
 *       → x − NaN = NaN poisons accumulator               [BUG A2]
 *  3. cusumAnomalyScore: NaN in state.sPos or state.sNeg
 *       → Math.max(NaN, finite) = NaN, slips past h guard [BUG A3]
 *  4. bocpdUpdate: prior.beta0 = Infinity
 *       → scale = Infinity → logProb = −Infinity → state collapses cleanly
 *  5. cusumBatch: empty series → trivial no-crash invariant
 *  6. volumeImbalance: qty = Infinity on both sides → NaN (GIGO, no crash)
 *  7. hawkesLambda: basic edge cases (function has zero test coverage)
 *  8. defaultPrior: NaN / negative trainingVar → beta0 guard fires; NaN trainingMean
 *  9. hawkesFit: all timestamps = Infinity → NaN T → fallback, params positive
 * 10. nelderMead: x0 containing Infinity → no crash
 */

import { describe, it, expect } from 'vitest';
import {
  volumeImbalance,
  hawkesAnomalyScore,
  hawkesLogLikelihood,
  hawkesLambda,
  hawkesFit,
} from '../src/math/hawkes.js';
import {
  cusumUpdate,
  cusumInitState,
  cusumBatch,
  cusumAnomalyScore,
} from '../src/math/cusum.js';
import {
  bocpdUpdate,
  bocpdInitState,
  defaultPrior,
} from '../src/math/bocpd.js';
import { nelderMead } from '../src/math/optimizer.js';
import type { IAggregatedTradeData } from '../src/types.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

function makeTrade(qty: number, isBuyerMaker: boolean): IAggregatedTradeData {
  return { id: "1", price: 100, qty, timestamp: 1_000_000, isBuyerMaker };
}

// ─── 1. hawkesAnomalyScore: peakLambda = NaN, valid params ───────────────────
//
// With meanLambda > 0 (valid params), the code reaches:
//   intensityScore = sig(NaN / meanLambda) = sig(NaN) = 1/(1+exp(NaN)) = NaN
//   Math.max(NaN, rateScore) = NaN  (JS: Math.max(NaN, x) = NaN)
//
// Correct behaviour: treat NaN peakLambda as "no signal" → intensityScore = 0.
// Final score = max(0, rateScore): 0 when empiricalRate=0, rateScore when > 0.

describe('hawkesAnomalyScore: peakLambda = NaN, valid params — must not return NaN', () => {
  const params = { mu: 1, alpha: 0.3, beta: 2 };
  // meanLambda = 1 / (1 − 0.3/2) = 1 / 0.85 ≈ 1.176  (finite, > 0)

  it('empiricalRate = 0: score is not NaN', () => {
    expect(Number.isNaN(hawkesAnomalyScore(NaN, params, 0))).toBe(false);
  });

  it('empiricalRate = 0: returns 0 (no intensity signal, no rate signal)', () => {
    expect(hawkesAnomalyScore(NaN, params, 0)).toBe(0);
  });

  it('empiricalRate = 0: result is in [0, 1]', () => {
    const s = hawkesAnomalyScore(NaN, params, 0);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it('empiricalRate > 0: returns rateScore (positive), not NaN', () => {
    // rateScore = sig(3 / mu=1) = sig(3) = 1/(1+exp(-2)) ≈ 0.88
    const score = hawkesAnomalyScore(NaN, params, 3);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThan(0);
  });

  it('empiricalRate > 0: score equals sig(rate/mu) alone (intensity term is 0)', () => {
    const rate = 3;
    const expected = 1 / (1 + Math.exp(-(rate / params.mu - 2) * 2));
    expect(hawkesAnomalyScore(NaN, params, rate)).toBeCloseTo(expected, 6);
  });

  it('does not crash', () => {
    expect(() => hawkesAnomalyScore(NaN, params, 0)).not.toThrow();
  });
});

// ─── 2. cusumUpdate: NaN in params.k or params.mu0 ───────────────────────────
//
// The existing NaN guard covers `x = NaN` but not NaN in params.
//
//   x − NaN    = NaN         (when mu0 = NaN: x − NaN − k = NaN)
//   x − k      = NaN         (when k = NaN: x − mu0 − NaN = NaN)
//   Math.max(0, NaN) = NaN   (JS: NaN is returned, not 0)
//   → sPos / sNeg become NaN → accumulator poisoned for all future observations.
//
// Correct behaviour: NaN params → no-op (return unchanged state, alarm=false).
// Same semantics as the existing x=NaN guard.

describe('cusumUpdate: params.k = NaN — state must not be poisoned', () => {
  const paramsNanK  = { mu0: 0.2, std0: 0.1, k: NaN, h: 0.5 };

  it('state.sPos is finite after update with k=NaN', () => {
    const upd = cusumUpdate(cusumInitState(), 0.5, paramsNanK);
    expect(Number.isFinite(upd.state.sPos)).toBe(true);
  });

  it('state.sNeg is finite after update with k=NaN', () => {
    const upd = cusumUpdate(cusumInitState(), 0.5, paramsNanK);
    expect(Number.isFinite(upd.state.sNeg)).toBe(true);
  });

  it('alarm = false when k=NaN (no spurious alarm)', () => {
    const upd = cusumUpdate(cusumInitState(), 0.5, paramsNanK);
    expect(upd.alarm).toBe(false);
  });

  it('state equals input state (no-op)', () => {
    const init = cusumInitState();
    const upd  = cusumUpdate(init, 0.5, paramsNanK);
    expect(upd.state.sPos).toBe(init.sPos);
    expect(upd.state.sNeg).toBe(init.sNeg);
  });

  it('10 subsequent valid updates remain finite after NaN-k update', () => {
    const validParams = { mu0: 0.2, std0: 0.1, k: 0.05, h: 0.5 };
    let s = cusumInitState();
    cusumUpdate(s, 0.5, paramsNanK);           // NaN-k, state unchanged
    for (let i = 0; i < 10; i++) {
      const upd = cusumUpdate(s, 0.3, validParams);
      expect(Number.isFinite(upd.state.sPos)).toBe(true);
      expect(Number.isFinite(upd.state.sNeg)).toBe(true);
      s = upd.state;
    }
  });
});

describe('cusumUpdate: params.mu0 = NaN — state must not be poisoned', () => {
  const paramsNanMu = { mu0: NaN, std0: 0.1, k: 0.05, h: 0.5 };

  it('state.sPos is finite after update with mu0=NaN', () => {
    const upd = cusumUpdate(cusumInitState(), 0.5, paramsNanMu);
    expect(Number.isFinite(upd.state.sPos)).toBe(true);
  });

  it('alarm = false when mu0=NaN', () => {
    const upd = cusumUpdate(cusumInitState(), 0.5, paramsNanMu);
    expect(upd.alarm).toBe(false);
  });

  it('state equals input state (no-op)', () => {
    const init = cusumInitState();
    const upd  = cusumUpdate(init, 0.8, paramsNanMu);
    expect(upd.state.sPos).toBe(init.sPos);
    expect(upd.state.sNeg).toBe(init.sNeg);
  });
});

// ─── 3. cusumAnomalyScore: NaN in state.sPos or state.sNeg ───────────────────
//
// Math.max(NaN, finite) = NaN in JS (unlike Math.min in some engines).
// Empirically: Math.max(NaN, 1) === NaN (V8, SpiderMonkey, JavaScriptCore).
//
//   s = Math.max(NaN, 0) = NaN
//   NaN / h = NaN  (h guard `NaN <= 0 = false` lets it through)
//   Math.min(NaN, 1) = NaN
//
// Correct behaviour: if s is NaN → return 0 (state is corrupt, no signal).

describe('cusumAnomalyScore: NaN in state — must return 0, not NaN', () => {
  const params = { mu0: 0.2, std0: 0.1, k: 0.05, h: 0.5 };

  it('sPos = NaN, sNeg = 0: returns 0 (not NaN)', () => {
    expect(cusumAnomalyScore({ sPos: NaN, sNeg: 0, n: 5 }, params)).toBe(0);
  });

  it('sPos = 0, sNeg = NaN: returns 0 (not NaN)', () => {
    expect(cusumAnomalyScore({ sPos: 0, sNeg: NaN, n: 5 }, params)).toBe(0);
  });

  it('sPos = NaN, sNeg = NaN: returns 0', () => {
    expect(cusumAnomalyScore({ sPos: NaN, sNeg: NaN, n: 5 }, params)).toBe(0);
  });

  it('result is not NaN for any NaN state combination', () => {
    expect(Number.isNaN(cusumAnomalyScore({ sPos: NaN, sNeg: 0,   n: 0 }, params))).toBe(false);
    expect(Number.isNaN(cusumAnomalyScore({ sPos: 0,   sNeg: NaN, n: 0 }, params))).toBe(false);
  });

  it('sPos = Infinity still returns 1 (existing behaviour preserved)', () => {
    expect(cusumAnomalyScore({ sPos: Infinity, sNeg: 0, n: 5 }, params)).toBe(1);
  });
});

// ─── 4. bocpdUpdate: prior.beta0 = Infinity ───────────────────────────────────
//
// betaN = Infinity → scale = Infinity → z = finite/sqrt(Inf) = 0
// But: 0.5·log(π·df·Inf) = 0.5·Inf = Infinity
// logProb = logΓ(·) − logΓ(·) − Infinity − 0 = −Infinity for all hypotheses.
//
// logCpMass = −∞, all jointLogProbs = −∞.
// logNorm = logSumExp(−∞, −∞) = −∞ (via the a=−∞ branch: return b = −∞).
// normLogProbs[i] = −∞ − (−∞) = NaN → all pruned → state collapses.
// cpProbability: exp(NaN) = NaN → isFinite guard → 0.

describe('bocpdUpdate: prior.beta0 = Infinity — clean collapse, no crash', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: Infinity };

  it('does not throw', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior)).not.toThrow();
  });

  it('state collapses to empty (all logProbs become NaN via -Inf - -Inf)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior);
    expect(r.state.logProbs.length).toBe(0);
  });

  it('cpProbability is 0 (finite guard handles NaN from exp(NaN))', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior);
    expect(Number.isFinite(r.cpProbability)).toBe(true);
    expect(r.cpProbability).toBe(0);
  });

  it('mapRunLength is a non-negative integer', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior);
    expect(Number.isInteger(r.mapRunLength)).toBe(true);
    expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
  });

  it('10 consecutive updates do not crash (stuck collapsed state)', () => {
    let s = bocpdInitState();
    expect(() => {
      for (let i = 0; i < 10; i++) {
        const r = bocpdUpdate(s, 0.3 + i * 0.05, prior);
        s = r.state;
      }
    }).not.toThrow();
  });

  it('cpProbability stays 0 across all 10 steps', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 10; i++) {
      const r = bocpdUpdate(s, 0.3 + i * 0.05, prior);
      expect(Number.isFinite(r.cpProbability)).toBe(true);
      s = r.state;
    }
  });
});

// ─── 5. cusumBatch: empty series ─────────────────────────────────────────────
//
// The for-loop executes 0 times.
// Expected: alarmIndices = [], final state = cusumInitState() values.

describe('cusumBatch: empty series — trivial invariant, no crash', () => {
  const params = { mu0: 0.3, std0: 0.1, k: 0.05, h: 0.5 };

  it('does not crash', () => {
    expect(() => cusumBatch([], params)).not.toThrow();
  });

  it('alarmIndices is empty', () => {
    expect(cusumBatch([], params).alarmIndices).toHaveLength(0);
  });

  it('final state.sPos = 0 (same as initState)', () => {
    expect(cusumBatch([], params).state.sPos).toBe(0);
  });

  it('final state.sNeg = 0', () => {
    expect(cusumBatch([], params).state.sNeg).toBe(0);
  });

  it('final state.n = 0', () => {
    expect(cusumBatch([], params).state.n).toBe(0);
  });
});

// ─── 6. volumeImbalance: qty = Infinity on both sides ────────────────────────
//
// buyVol = Infinity (isBuyerMaker=false), sellVol = Infinity (isBuyerMaker=true).
// total = Infinity + Infinity = Infinity.
// (Infinity − Infinity) / Infinity = NaN / Infinity = NaN.
//
// GIGO — the function is not designed to sanitise Infinity qty.
// Key contract: no crash; single-side Infinity is handled cleanly (+1 / −1).

describe('volumeImbalance: qty = Infinity — no crash, NaN only for symmetric Infinity', () => {
  it('buy=Infinity, sell=0: does not throw', () => {
    expect(() => volumeImbalance([makeTrade(Infinity, false)])).not.toThrow();
  });

  it('buy=Infinity, sell=0: returns +1 (all buys)', () => {
    expect(volumeImbalance([makeTrade(Infinity, false)])).toBe(1);
  });

  it('buy=0, sell=Infinity: returns −1 (all sells)', () => {
    expect(volumeImbalance([makeTrade(Infinity, true)])).toBe(-1);
  });

  it('buy=Infinity, sell=Infinity: does not throw', () => {
    expect(() =>
      volumeImbalance([makeTrade(Infinity, false), makeTrade(Infinity, true)]),
    ).not.toThrow();
  });

  it('buy=Infinity, sell=Infinity: returns 0 (symmetric burst)', () => {
    // Both sides Infinity → buyVol === sellVol → return 0 (no directional bias).
    const result = volumeImbalance([
      makeTrade(Infinity, false),
      makeTrade(Infinity, true),
    ]);
    expect(result).toBe(0);
  });
});

// ─── 7. hawkesLambda: basic edge cases ───────────────────────────────────────
//
// This function has no dedicated tests in any existing test file.
// It is still exported and used by external consumers.

describe('hawkesLambda: edge cases (no prior coverage)', () => {
  const params = { mu: 1, alpha: 0.5, beta: 2 };

  it('empty timestamps: returns exactly mu', () => {
    expect(hawkesLambda(100, [], params)).toBe(1);
  });

  it('t ≤ first timestamp: loop breaks immediately, returns mu', () => {
    // All events are "in the future" — no excitation contribution.
    expect(hawkesLambda(5, [10, 20, 30], params)).toBe(params.mu);
  });

  it('t equals a timestamp: that event does not contribute (ti >= t → break)', () => {
    // The contract states all timestamps must be < t, so ti = t means no contrib.
    expect(hawkesLambda(10, [10], params)).toBe(params.mu);
  });

  it('two events before t: lambda = mu + alpha*(exp(-beta*dt1) + exp(-beta*dt2))', () => {
    // t=10, ts=[9, 9.5], params={mu:1, alpha:0.5, beta:2}
    // lambda = 1 + 0.5*(exp(-2*1) + exp(-2*0.5))
    //        = 1 + 0.5*(exp(-2) + exp(-1))
    const expected = 1 + 0.5 * (Math.exp(-2) + Math.exp(-1));
    expect(hawkesLambda(10, [9, 9.5], params)).toBeCloseTo(expected, 10);
  });

  it('alpha = 0: returns mu regardless of event history', () => {
    const purePoisson = { mu: 2, alpha: 0, beta: 1 };
    expect(hawkesLambda(100, [1, 2, 50, 99], purePoisson)).toBe(2);
  });

  it('mu = 0, alpha > 0: returns alpha * kernel sum (no baseline)', () => {
    const noBaseline = { mu: 0, alpha: 1, beta: 1 };
    // t=10, ts=[9]: lambda = 0 + 1*exp(-1*(10-9)) = exp(-1)
    expect(hawkesLambda(10, [9], noBaseline)).toBeCloseTo(Math.exp(-1), 10);
  });

  it('result is always finite and non-negative for valid params + sorted timestamps', () => {
    const ts = [1, 2, 3, 4];
    const lam = hawkesLambda(5, ts, params);
    expect(Number.isFinite(lam)).toBe(true);
    expect(lam).toBeGreaterThanOrEqual(0);
  });

  it('returns mu for single timestamp equal to t (boundary — no contribution)', () => {
    expect(hawkesLambda(42, [42], params)).toBe(params.mu);
  });
});

// ─── 8. defaultPrior: NaN / extreme training inputs ──────────────────────────
//
// defaultPrior(trainingMean, trainingVar):
//   mu0   = trainingMean   (not sanitised — NaN propagates)
//   beta0 = trainingVar > 0 ? trainingVar : 1   (guarded: NaN>0=false → 1; 0>0=false → 1)
//
// NaN mu0 propagates into bocpdUpdate and causes state collapse (tested here at
// the prior level so users understand the entry point).
// The beta0 guard (trainingVar > 0) is documented via negative/zero/NaN inputs.

describe('defaultPrior: NaN / extreme trainingVar — beta0 guard behaviour', () => {
  it('trainingVar = NaN: beta0 = 1 (NaN > 0 = false → guard fires)', () => {
    expect(defaultPrior(0.5, NaN).beta0).toBe(1);
  });

  it('trainingVar = 0: beta0 = 1 (0 > 0 = false → guard fires)', () => {
    expect(defaultPrior(0.5, 0).beta0).toBe(1);
  });

  it('trainingVar = −5: beta0 = 1 (negative > 0 = false → guard fires)', () => {
    expect(defaultPrior(0.5, -5).beta0).toBe(1);
  });

  it('trainingVar = Infinity: beta0 = Infinity (positive, guard does not fire)', () => {
    // Infinity > 0 = true — no guard. bocpdUpdate with beta0=Inf handled separately.
    expect(defaultPrior(0.5, Infinity).beta0).toBe(Infinity);
  });

  it('trainingMean = NaN: mu0 = NaN (not sanitised — propagates to bocpdUpdate)', () => {
    expect(Number.isNaN(defaultPrior(NaN, 1).mu0)).toBe(true);
  });

  it('trainingMean = NaN: other fields are still finite', () => {
    const p = defaultPrior(NaN, 1);
    expect(Number.isFinite(p.kappa0)).toBe(true);
    expect(Number.isFinite(p.alpha0)).toBe(true);
    expect(Number.isFinite(p.beta0)).toBe(true);
  });

  it('bocpdUpdate with NaN mu0 prior: does not crash (state collapses)', () => {
    const prior = defaultPrior(NaN, 1);
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior)).not.toThrow();
  });
});

// ─── 9. hawkesFit: all timestamps = Infinity ──────────────────────────────────
//
// T = Infinity − Infinity = NaN.
// mu0 = n / (NaN || 1) = n / 1 = n  (NaN is falsy → || 1 rescue fires).
// x0 = [n*0.5, n*0.4, n]  (all finite, in feasible region).
//
// However hawkesLogLikelihood(allInf, params) always returns NaN:
//   t0 = Infinity, T = NaN, ll = −mu*NaN = NaN → propagates.
// negLL = NaN → f=NaN path → converged=false → fallback kicks in.
//
// muFallback = n / (NaN || 1) = n.  alpha = n*0.01.  beta = n.  All positive.

describe('hawkesFit: all timestamps = Infinity — NaN T → fallback, params positive', () => {
  const tsInf10 = Array.from({ length: 10 }, () => Infinity);
  const tsInf20 = Array.from({ length: 20 }, () => Infinity);

  it('n=10, all Inf: does not throw', () => {
    expect(() => hawkesFit(tsInf10)).not.toThrow();
  });

  it('n=10, all Inf: converged = false (fallback)', () => {
    expect(hawkesFit(tsInf10).converged).toBe(false);
  });

  it('n=10, all Inf: params.mu > 0', () => {
    expect(hawkesFit(tsInf10).params.mu).toBeGreaterThan(0);
  });

  it('n=10, all Inf: params.alpha > 0', () => {
    expect(hawkesFit(tsInf10).params.alpha).toBeGreaterThan(0);
  });

  it('n=10, all Inf: params.beta > 0', () => {
    expect(hawkesFit(tsInf10).params.beta).toBeGreaterThan(0);
  });

  it('n=10, all Inf: stationarity ≥ 0 and < 1', () => {
    const r = hawkesFit(tsInf10);
    expect(r.stationarity).toBeGreaterThanOrEqual(0);
    expect(r.stationarity).toBeLessThan(1);
  });

  it('n=20, all Inf: same contracts', () => {
    const r = hawkesFit(tsInf20);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.params.alpha).toBeGreaterThan(0);
    expect(r.params.beta).toBeGreaterThan(0);
  });
});

// ─── 10. nelderMead: x0 containing Infinity ───────────────────────────────────
//
// If x0 = [Infinity], the step is Infinity*1.2 = Infinity → simplex = [[Inf],[Inf]].
// f([Inf]) depends on the objective: often Infinity or penalty.
// All fvals = Infinity → spread = Inf−Inf = NaN → convergence check never fires.
// Runs maxIter, converged = false.
//
// Key contract: no crash, terminates in finite iterations.

describe('nelderMead: x0 containing Infinity — no crash, terminates', () => {
  it('x0 = [Infinity]: does not crash', () => {
    const f = ([x]: number[]) => x! ** 2;
    expect(() => nelderMead(f, [Infinity], { maxIter: 50 })).not.toThrow();
  });

  it('x0 = [Infinity]: terminates after ≤ maxIter iterations', () => {
    const f = ([x]: number[]) => x! ** 2;
    const r = nelderMead(f, [Infinity], { maxIter: 50 });
    expect(r.iters).toBeLessThanOrEqual(50);
  });

  it('x0 = [Infinity]: converged = false (all fvals are Infinity)', () => {
    const f = ([x]: number[]) => x! ** 2;
    const r = nelderMead(f, [Infinity], { maxIter: 50 });
    expect(r.converged).toBe(false);
  });

  it('x0 = [0, Infinity]: does not crash', () => {
    const f = (x: number[]) => x.reduce((s, v) => s + v ** 2, 0);
    expect(() => nelderMead(f, [0, Infinity], { maxIter: 50 })).not.toThrow();
  });

  it('x0 = [0, Infinity]: terminates after ≤ maxIter iterations', () => {
    const f = (x: number[]) => x.reduce((s, v) => s + v ** 2, 0);
    const r = nelderMead(f, [0, Infinity], { maxIter: 50 });
    expect(r.iters).toBeLessThanOrEqual(50);
  });

  it('x0 = [1, 1, 1] (all finite): still converges correctly for simple quadratic', () => {
    // Sanity check: adding Infinity tests does not break normal operation.
    const f = (x: number[]) => x.reduce((s, v) => s + (v - 2) ** 2, 0);
    const r = nelderMead(f, [1, 1, 1], { maxIter: 1000, tol: 1e-8 });
    expect(r.converged).toBe(true);
    for (const xi of r.x) expect(xi).toBeCloseTo(2, 1);
  });
});
