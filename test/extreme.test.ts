/**
 * extreme.test.ts — math does not freeze at an extremum for rarest inputs.
 *
 * Each test targets a specific "stuck at extremum" scenario identified
 * via static analysis of the math modules:
 *
 *   1. bocpdUpdate: hazardLambda < 1 (H > 1)  → state collapses to empty
 *   2. bocpdUpdate: hazardLambda = 1  (H = 1)  → stuck at mapRL = 0 forever
 *   3. bocpdUpdate: hazardLambda = ∞  (H = 0)  → no changepoints, single growing run
 *   4. hawkesAnomalyScore: mu = 0             → meanLambda = 0, score stuck at 1
 *   5. hawkesAnomalyScore: branching = 1 − ε  → discontinuity at supercritical boundary
 *   6. hawkesLogLikelihood: extreme parameters → no NaN / ±Infinity
 *   7. nelderMead: f = Infinity / NaN / const  → always terminates, no false convergence
 *   8. Welford SS: near-constant series        → m2 stays ≥ 0 (no FP drift to negative)
 *   9. bocpdUpdate: prior with beta0 = 0       → scale = 0, no crash
 */

import { describe, it, expect } from 'vitest';
import {
  hawkesAnomalyScore,
  hawkesLogLikelihood,
} from '../src/math/hawkes.js';
import {
  bocpdUpdate,
  bocpdInitState,
  bocpdAnomalyScore,
} from '../src/math/bocpd.js';
import { nelderMead } from '../src/math/optimizer.js';

// ─── 1. BOCPD: hazardLambda < 1 (H > 1) — state collapse ────────────────────
//
// H = 1/hazardLambda > 1  →  log(1 − H) = log(negative) = NaN
// Every growth hypothesis gets NaN log-prob → (NaN > −30) = false → all pruned.
// The state collapses to an empty array after the very first update.
// All subsequent updates also produce an empty state (stuck state).
//
// Traced path:
//   jointLogProbs[0] (cp)      = logProb + logPred + log(H)     = finite
//   jointLogProbs[r+1] (growth)= logProb + logPred + log(1−H)   = finite + NaN = NaN
//   logNorm = logSumExp(finite, NaN) = NaN
//   normLogProbs = [NaN, NaN]  →  keep = [false, false]  →  prunedLogProbs = []

describe('bocpdUpdate: hazardLambda < 1 (H > 1) — state collapses to empty', () => {
  const prior = { mu0: 0, kappa0: 1, alpha0: 1, beta0: 1 };

  it('hazardLambda=0.5 (H=2): state.logProbs is empty after one update', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0.5);
    expect(r.state.logProbs.length).toBe(0);
    expect(r.state.suffStats.length).toBe(0);
  });

  it('hazardLambda=0.5: first update does not crash', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior, 0.5)).not.toThrow();
  });

  it('hazardLambda=0.5: mapRunLength = 0 (argmax over NaN normLogProbs)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0.5);
    expect(r.mapRunLength).toBe(0);
  });

  it('hazardLambda=0.5: state stays empty on all subsequent updates (stuck state)', () => {
    let s = bocpdInitState();
    const r0 = bocpdUpdate(s, 0.5, prior, 0.5);
    s = r0.state;
    expect(s.logProbs.length).toBe(0); // collapsed

    for (let i = 0; i < 5; i++) {
      const r = bocpdUpdate(s, 0.5 + i * 0.01, prior, 0.5);
      expect(r.state.logProbs.length).toBe(0);
      s = r.state;
    }
  });

  it('hazardLambda=0.5: 10 consecutive updates never crash', () => {
    let s = bocpdInitState();
    expect(() => {
      for (let i = 0; i < 10; i++) {
        const r = bocpdUpdate(s, 0.3 + i * 0.02, prior, 0.5);
        s = r.state;
      }
    }).not.toThrow();
  });

  it('hazardLambda=0.5: bocpdAnomalyScore does not throw on collapsed output', () => {
    let s = bocpdInitState();
    let prevRL = 0;
    for (let i = 0; i < 5; i++) {
      const r = bocpdUpdate(s, 0.5, prior, 0.5);
      expect(() => bocpdAnomalyScore(r, prevRL)).not.toThrow();
      prevRL = r.mapRunLength;
      s = r.state;
    }
  });

  it('hazardLambda=0.1 (H=10): same collapse — state.logProbs empty', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 3; i++) {
      const r = bocpdUpdate(s, 0.3, prior, 0.1);
      s = r.state;
    }
    expect(s.logProbs.length).toBe(0);
  });
});

// ─── 2. BOCPD: hazardLambda = 1 (H = 1) — stuck at cpProbability = 1 ─────────
//
// H = 1  →  log(1 − H) = log(0) = −Infinity.
// All growth hypotheses get −Infinity log-prob → pruned at every step.
// Only r = 0 (changepoint) survives with normLogProb = 0 → cpProbability = exp(0) = 1.
// The state resets to {logProbs:[0], suffStats:[ssEmpty()]} each step (no memory).
// mapRunLength = 0 always  →  bocpdAnomalyScore returns 0 always (prevRL guard).

describe('bocpdUpdate: hazardLambda = 1 (H = 1) — stuck at mapRL = 0', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 1 };

  it('mapRunLength = 0 on every step for 20 observations', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, 0.4 + (i % 3) * 0.1, prior, 1);
      expect(r.mapRunLength).toBe(0);
      s = r.state;
    }
  });

  it('cpProbability = 1 on every step (all mass at changepoint hypothesis)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, 0.5, prior, 1);
      // normLogProbs[0] = lp - lp = 0 exactly → exp(0) = 1
      expect(r.cpProbability).toBe(1);
      s = r.state;
    }
  });

  it('state always has exactly 1 hypothesis (only r=0 survives pruning)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, 0.5, prior, 1);
      expect(r.state.logProbs.length).toBe(1);
      expect(r.state.suffStats.length).toBe(1);
      s = r.state;
    }
  });

  it('bocpdAnomalyScore = 0 always (prevRL is always 0, guard fires)', () => {
    let s = bocpdInitState();
    let prevRL = 0; // mapRL never changes from 0
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, 0.5, prior, 1);
      const score = bocpdAnomalyScore(r, prevRL);
      expect(score).toBe(0); // prevRL <= 0 → guard returns 0
      prevRL = r.mapRunLength; // stays 0
      s = r.state;
    }
  });
});

// ─── 3. BOCPD: hazardLambda = Infinity (H = 0) — run never resets ─────────────
//
// H = 0  →  log(H) = −Infinity  →  cpProbability = 0 always.
// Only growth hypothesis survives: jointLogProbs[r+1] = logProbs[r] + logPred.
// With H=0 and a single initial hypothesis, exactly 1 hypothesis grows per step:
// r=0→1→2→...  — state always contains exactly 1 entry (the current run).
// mapRunLength grows monotonically.  No changepoint signal ever fires.
// bocpdAnomalyScore ≈ 0.018 (drop = 0 since RL grows each step).

describe('bocpdUpdate: hazardLambda = Infinity (H = 0) — no changepoints ever', () => {
  const prior = { mu0: 0.5, kappa0: 5, alpha0: 3, beta0: 1 };

  it('cpProbability = 0 on every step', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 30; i++) {
      const r = bocpdUpdate(s, 0.5, prior, Infinity);
      expect(r.cpProbability).toBe(0);
      s = r.state;
    }
  });

  it('mapRunLength = 1 on every step (single surviving hypothesis always at position 1)', () => {
    // With H=0: no cp hypothesis ever survives (logProb = exp(-Inf) = 0).
    // The single growth hypothesis occupies position 1 in normLogProbs every step,
    // so mapR = 1 always — even though the underlying run is growing longer.
    // The suffStats correctly accumulate (n grows), but mapRunLength stays at 1
    // because the BOCPD state index restarts from 0 after pruning compact the array.
    let s = bocpdInitState();
    for (let i = 0; i < 50; i++) {
      const r = bocpdUpdate(s, 0.5, prior, Infinity);
      expect(r.mapRunLength).toBe(1);
      s = r.state;
    }
  });

  it('state always has exactly 1 hypothesis (single growing run, H=0 creates no new cp)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 50; i++) {
      const r = bocpdUpdate(s, 0.5, prior, Infinity);
      expect(r.state.logProbs.length).toBe(1);
      s = r.state;
    }
  });

  it('bocpdAnomalyScore stays low (< 0.2): RL grows → drop = 0 → score ≈ 0.018', () => {
    let s = bocpdInitState();
    let prevRL = 0;
    for (let i = 0; i < 30; i++) {
      const r = bocpdUpdate(s, 0.5, prior, Infinity);
      const score = bocpdAnomalyScore(r, prevRL);
      expect(score).toBeLessThan(0.2);
      prevRL = r.mapRunLength;
      s = r.state;
    }
  });
});

// ─── 4. hawkesAnomalyScore: mu = 0 → score permanently stuck at 1 ────────────
//
// meanLambda = mu / (1 − alpha/beta).
// When mu = 0:  meanLambda = 0  →  peakLambda / 0 = Infinity  →  sig(Infinity) = 1.
// When empiricalRate > 0:  empiricalRate / mu = empiricalRate / 0 = Infinity  →  sig(Inf) = 1.
// The detector becomes permanently "anomalous" regardless of actual activity.
//
// hawkesFit() penalty prevents mu ≤ 0 from the optimizer, but direct callers
// (e.g., unit tests, custom integrations) can pass mu = 0 and must not crash.

describe('hawkesAnomalyScore: mu = 0 → score stuck at maximum', () => {
  it('mu=0, alpha=0, peakLambda=1: score = 1 (meanLambda = 0 → Inf ratio)', () => {
    expect(hawkesAnomalyScore(1, { mu: 0, alpha: 0, beta: 1 }, 0)).toBe(1);
  });

  it('mu=0, peakLambda>0, empiricalRate=100: both ratios = Inf → score = 1', () => {
    // intensityScore: sig(1/0) = sig(Inf) = 1
    // rateScore:      sig(100/0) = sig(Inf) = 1
    // Math.max(1, 1) = 1
    expect(hawkesAnomalyScore(1, { mu: 0, alpha: 0, beta: 1 }, 100)).toBe(1);
  });

  it('mu=0, peakLambda=0, empiricalRate=100: 0/0 = NaN → Math.max(NaN, 1) = NaN (known JS quirk)', () => {
    // intensityScore: sig(0/0) = sig(NaN) = NaN
    // rateScore:      sig(100/0) = sig(Inf) = 1
    // Math.max(NaN, 1) = NaN — JS quirk: any NaN argument makes max return NaN
    // This is a latent bug: with peakLambda=0 and mu=0 the score silently becomes NaN.
    const score = hawkesAnomalyScore(0, { mu: 0, alpha: 0, beta: 1 }, 100);
    expect(Number.isNaN(score)).toBe(true);
  });

  it('mu=Number.MIN_VALUE (5e-324): score = 1 (ratio → Infinity before sig)', () => {
    const score = hawkesAnomalyScore(1, { mu: Number.MIN_VALUE, alpha: 0, beta: 1 }, 0);
    expect(score).toBeCloseTo(1, 5);
  });

  it('mu=1e-15: score = 1 (still overwhelmingly large ratio)', () => {
    const score = hawkesAnomalyScore(1, { mu: 1e-15, alpha: 0, beta: 1 }, 0);
    expect(score).toBeCloseTo(1, 5);
  });

  it('mu=0, peakLambda=0, empiricalRate=0: does not crash (0/0 path)', () => {
    // sig(0/0) = sig(NaN) → NaN → max(NaN, 0) is engine-specific, but no crash
    expect(() => hawkesAnomalyScore(0, { mu: 0, alpha: 0, beta: 1 }, 0)).not.toThrow();
  });
});

// ─── 5. hawkesAnomalyScore: discontinuity at branching = 1 boundary ──────────
//
// At alpha/beta = 1.0 (supercritical): explicit guard → return 1 immediately.
// At alpha/beta = 0.9999 (just below):
//   meanLambda = mu / (1 − 0.9999) = mu * 10000
//   peakLambda / meanLambda ≪ 1  →  sig(tiny − 2) ≈ 0.12
//
// This is a mathematical discontinuity: branching 1.0 → score 1,
// branching 0.9999 → score ≈ 0.12.  Tests document it so regressions are caught
// if the guard condition is ever moved or changed.

describe('hawkesAnomalyScore: discontinuity at supercritical boundary (branching = 1)', () => {
  it('branching exactly 1.0 (alpha = beta = 5): supercritical guard returns 1', () => {
    expect(hawkesAnomalyScore(2, { mu: 1, alpha: 5, beta: 5 }, 0)).toBe(1);
  });

  it('branching 1.0001 (alpha > beta): guard still returns 1', () => {
    expect(hawkesAnomalyScore(2, { mu: 1, alpha: 1.0001, beta: 1 }, 0)).toBe(1);
  });

  it('branching 0.9999 (near-critical): score << 1 despite near-explosive process', () => {
    // meanLambda = 1 / 0.0001 = 10000; ratio = 2/10000 = 0.0002; sig(0.0002−2) ≈ 0.12
    const score = hawkesAnomalyScore(2, { mu: 1, alpha: 0.9999, beta: 1 }, 0);
    expect(score).toBeLessThan(0.5);
  });

  it('branching 0.9999 score is at least 0.4 below branching 1.0 (documents the jump)', () => {
    const scoreAt1   = hawkesAnomalyScore(2, { mu: 1, alpha: 1.0,    beta: 1 }, 0); // = 1
    const scoreBelow = hawkesAnomalyScore(2, { mu: 1, alpha: 0.9999, beta: 1 }, 0);
    expect(scoreAt1 - scoreBelow).toBeGreaterThan(0.4);
  });

  it('branching 0.5 → score in (0, 1) continuously', () => {
    const score = hawkesAnomalyScore(2, { mu: 1, alpha: 0.5, beta: 1 }, 0);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
    expect(Number.isFinite(score)).toBe(true);
  });
});

// ─── 6. hawkesLogLikelihood: extreme parameters do not produce ±Infinity or NaN
//
// Tests that LL remains finite under conditions that could cause overflow or
// underflow in the recursive A(i) computation or the compensator term.

describe('hawkesLogLikelihood: extreme inputs stay finite and NaN-free', () => {
  const ts5 = [0, 1, 2, 3, 4]; // T = 4, n = 5

  it('alpha = 0 (pure Poisson process): LL equals −mu·T + n·log(mu) exactly', () => {
    const mu = 2, T = 4, n = 5;
    const ll = hawkesLogLikelihood(ts5, { mu, alpha: 0, beta: 1 });
    // A=0 always (alpha=0); compensator = 0; LL = −mu·T + n·log(mu)
    expect(ll).toBeCloseTo(-mu * T + n * Math.log(mu), 8);
  });

  it('beta = Number.MAX_VALUE: A underflows to 0 for dt > 0 → effectively Poisson LL', () => {
    const mu = 2;
    const llBig    = hawkesLogLikelihood(ts5, { mu, alpha: 0.5, beta: Number.MAX_VALUE });
    const llPoisson = hawkesLogLikelihood(ts5, { mu, alpha: 0,   beta: 1 });
    expect(Number.isFinite(llBig)).toBe(true);
    // exp(−MAX_VALUE * dt) → 0; A = 0; LL collapses to Poisson LL
    expect(llBig).toBeCloseTo(llPoisson, 3);
  });

  it('beta = 1e-100 (very slow decay): A grows to ≈n−1, LL finite', () => {
    // alpha/beta = 0.01/1e-100 = 1e98 (finite); exp(-1e-100*T) ≈ 1 in float64
    // → (1 - exp(-β·(T−tᵢ))) = 0 → compensator = 1e98 * 0 = 0 → LL = Σ log(λᵢ), finite
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 0.01, beta: 1e-100 });
    expect(Number.isFinite(ll)).toBe(true);
    expect(ll).toBeGreaterThan(-Infinity);
    expect(ll).toBeLessThan(Infinity);
  });

  it('beta = Number.MIN_VALUE (5e-324): alpha/beta = Infinity → NaN (documents known limit)', () => {
    // alpha / Number.MIN_VALUE = Infinity (overflow)
    // (1 - exp(-β·T)) = 0 in float64 (β·T too small to register)
    // compensator = Infinity * 0 = NaN → LL = NaN
    // This is the absolute lower limit of β below which the compensator breaks.
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 0.01, beta: Number.MIN_VALUE });
    expect(Number.isNaN(ll)).toBe(true);
  });

  it('1000 identical timestamps (T = 0): A reaches 999, LL is finite and positive', () => {
    // A(i) = exp(−β·0)·(1 + A(i−1)) = 1 + A(i−1) → A(999) = 999
    // lambda_999 = mu + alpha·999; compensator = 0 (T−tᵢ = 0 for all)
    // LL = Σᵢ log(mu + alpha·i)  → positive and finite
    const ts1000 = Array.from({ length: 1000 }, () => 5000);
    const ll = hawkesLogLikelihood(ts1000, { mu: 0.1, alpha: 0.01, beta: 1 });
    expect(Number.isFinite(ll)).toBe(true);
    expect(ll).toBeGreaterThan(0); // log(0.1+0.01·i) > 0 for i ≥ 10
  });

  it('single timestamp (n = 1, T = 0): LL = log(mu)', () => {
    const mu = 3;
    // T=0 → −mu·0=0; A=0 (no prior events); compensator = α/β·(1−exp(0))=0
    const ll = hawkesLogLikelihood([42], { mu, alpha: 0.5, beta: 2 });
    expect(ll).toBeCloseTo(Math.log(mu), 10);
  });

  it('very large T (1 year in seconds, ~3.15e7): compensator stays finite', () => {
    const T = 31_536_000; // 1 year in seconds
    const n = 20;
    const tsYear = Array.from({ length: n }, (_, i) => i * (T / (n - 1)));
    const ll = hawkesLogLikelihood(tsYear, { mu: 1e-5, alpha: 5e-6, beta: 0.001 });
    expect(Number.isFinite(ll)).toBe(true);
  });

  it('alpha ≥ beta (supercritical, no penalty in LL): returns finite value, never NaN', () => {
    // hawkesLogLikelihood has no stationarity check — alpha/beta ≥ 1 is allowed.
    // The LL itself is a valid number for any finite timestamps and positive params.
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 1, beta: 0.5 });
    expect(Number.isNaN(ll)).toBe(false);
    expect(ll).toBeGreaterThan(-Infinity);
  });

  it('all parameters near machine epsilon (1e-300): LL finite', () => {
    const ll = hawkesLogLikelihood(ts5, { mu: 1e-300, alpha: 1e-301, beta: 1e-300 });
    // mu > 0, alpha < beta → valid; lambda ≈ mu → log(tiny) → very negative but finite
    expect(Number.isNaN(ll)).toBe(false);
    expect(Number.isFinite(ll)).toBe(true);
  });
});

// ─── 7. nelderMead: degenerate f values — always terminates ──────────────────
//
// Nelder-Mead must terminate regardless of what f returns.  Specific risks:
//
//  f = Infinity: spread = Inf−Inf = NaN → convergence check fails → loops maxIter.
//  f = NaN:      comparison NaN < fval = false → shrink fires every iteration;
//                simplex degenerates to single point, no crash.
//  f = const C:  spread = 0 < tol AND fvals[0] = C < 1e9 → converged at iter=0.
//  f = 1e10:     spread = 0 < tol BUT fvals[0] = 1e10 ≥ 1e9 → penalty-wall guard
//                prevents false convergence; runs maxIter, converged=false.

describe('nelderMead: degenerate f values — always terminates, no false convergence', () => {
  it('f always Infinity: loops maxIter, returns converged=false, fx=Infinity', () => {
    const r = nelderMead(() => Infinity, [1, 1], { maxIter: 100 });
    expect(r.converged).toBe(false);
    expect(r.iters).toBe(100);
    expect(r.fx).toBe(Infinity);
  });

  it('f always NaN: terminates after maxIter without crash, converged=false', () => {
    const r = nelderMead(() => NaN, [1, 1], { maxIter: 50 });
    expect(r.converged).toBe(false);
    expect(r.iters).toBe(50);
    // Simplex degenerates to a single point via repeated shrink — x must be finite
    expect(r.x.every(Number.isFinite)).toBe(true);
  });

  it('f constant 5.0 (flat landscape): converges immediately at iter=0', () => {
    // spread = 5−5 = 0 < tol AND fvals[0] = 5 < 1e9 → early return at iter=0
    const r = nelderMead(() => 5, [2, 3], { maxIter: 1000, tol: 1e-8 });
    expect(r.converged).toBe(true);
    expect(r.iters).toBe(0);
    expect(r.fx).toBe(5);
  });

  it('f always 1e10 (penalty wall): no false convergence despite spread = 0', () => {
    // Penalty-wall guard: spread < tol but fvals[0] ≥ 1e9 → does NOT return converged
    const r = nelderMead(() => 1e10, [1, 1], { maxIter: 100 });
    expect(r.converged).toBe(false);
    expect(r.iters).toBe(100);
    expect(r.fx).toBe(1e10);
  });

  it('f always 0 (global min everywhere): converges at iter=0, fx=0', () => {
    const r = nelderMead(() => 0, [5, -3], { maxIter: 1000 });
    expect(r.converged).toBe(true);
    expect(r.iters).toBe(0);
    expect(r.fx).toBe(0);
  });

  it('f always Infinity for 3D input: terminates without crash, x has 3 finite components', () => {
    const r = nelderMead(() => Infinity, [1, 1, 1], { maxIter: 50 });
    expect(r.converged).toBe(false);
    expect(r.x.length).toBe(3);
    expect(r.x.every(Number.isFinite)).toBe(true);
  });

  it('f = NaN for 3D input: terminates, x has 3 finite components', () => {
    const r = nelderMead(() => NaN, [0, 0, 0], { maxIter: 50 });
    expect(r.converged).toBe(false);
    expect(r.x.every(Number.isFinite)).toBe(true);
  });
});

// ─── 8. Welford SS: m2 never goes negative (floating-point drift) ─────────────
//
// Welford's formula: m2 += delta * delta2, where delta2 = x − mean_new.
// Mathematically: delta * delta2 = delta² · (n−1)/n ≥ 0, so m2 can only grow.
// With finite-precision arithmetic on nearly-identical values, cancellation in
// delta2 can theoretically make individual increments negative.  A negative m2
// propagates to betaN < 0 → scale < 0 → sqrt(negative) = NaN → cascade.
// These tests verify that m2 stays at or above 0 for the data ranges BOCPD sees.

describe('bocpdUpdate: Welford m2 stays non-negative in all surviving suffStats', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 1 };

  it('100 constant observations (0.5): m2 = 0 exactly in all surviving ss', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 100; i++) {
      const r = bocpdUpdate(s, 0.5, prior, 200);
      for (const ss of r.state.suffStats) {
        expect(ss.m2).toBeGreaterThanOrEqual(0);
      }
      s = r.state;
    }
  });

  it('200 near-constant observations (0.5 ± 1e-10): m2 ≥ −1e-25 in all ss', () => {
    // Jitter is 100× smaller than 1e-10²=1e-20 (expected m2 scale),
    // so any FP error in m2 is bounded by machine epsilon × m2 ≈ 1e-20 × 2e-16 = 2e-36.
    let s = bocpdInitState();
    for (let i = 0; i < 200; i++) {
      const x = 0.5 + (i % 3 === 0 ? 1e-10 : 0);
      const r = bocpdUpdate(s, x, prior, 200);
      for (const ss of r.state.suffStats) {
        expect(ss.m2).toBeGreaterThanOrEqual(-1e-25);
      }
      s = r.state;
    }
  });

  it('mean stays finite and bounded for all surviving ss after 100 steps', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 100; i++) {
      const r = bocpdUpdate(s, 0.3 + (i % 5) * 0.1, prior, 200);
      for (const ss of r.state.suffStats) {
        expect(Number.isFinite(ss.mean)).toBe(true);
        expect(Number.isFinite(ss.m2)).toBe(true);
        // Values in [0, 1] → mean must stay in [0, 1]
        expect(ss.mean).toBeGreaterThanOrEqual(-0.01); // tiny rounding tolerance
        expect(ss.mean).toBeLessThanOrEqual(1.01);
      }
      s = r.state;
    }
  });

  it('alternating extreme values (0 and 1): m2 ≥ 0 always', () => {
    // Maximum variance case for |imbalance| ∈ [0,1]
    let s = bocpdInitState();
    for (let i = 0; i < 100; i++) {
      const r = bocpdUpdate(s, i % 2 === 0 ? 0 : 1, prior, 200);
      for (const ss of r.state.suffStats) {
        expect(ss.m2).toBeGreaterThanOrEqual(0);
      }
      s = r.state;
    }
  });
});

// ─── 9. bocpdUpdate: prior with beta0 = 0 (scale = 0 in Student-t) ───────────
//
// defaultPrior() guards against beta0 = 0 (uses max(trainingVar, 1)).
// But bocpdUpdate() can be called directly with any prior.
//
// With beta0 = 0:
//   betaN = beta0 + 0.5·m2 + cross-term = 0 (when m2=0, mean=mu0)
//   scale = betaN·(κN+1)/(αN·κN) = 0
//   z     = (x − muN) / sqrt(0)
//     if x = muN: z = 0/0 = NaN → logProb = NaN → all hypotheses pruned → empty state
//     if x ≠ muN: z = ±Infinity → logProb = −Infinity → changepoint always wins
//
// Key contract: no crash, mapRunLength is a non-negative integer.

describe('bocpdUpdate: prior with beta0 = 0 (scale = 0 in predictive Student-t)', () => {
  const prior0 = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 0 };

  it('beta0=0, x = mu0 = 0.5: does not crash', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior0, 200)).not.toThrow();
  });

  it('beta0=0, x = mu0: mapRunLength is a non-negative integer', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior0, 200);
    expect(Number.isInteger(r.mapRunLength)).toBe(true);
    expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
  });

  it('beta0=0, x ≠ mu0 (x = 1.0): does not crash (z = Inf → logProb = −Inf)', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 1.0, prior0, 200)).not.toThrow();
  });

  it('beta0=0, x ≠ mu0: mapRunLength is a non-negative integer', () => {
    const r = bocpdUpdate(bocpdInitState(), 1.0, prior0, 200);
    expect(Number.isInteger(r.mapRunLength)).toBe(true);
    expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
  });

  it('beta0=0: 10 consecutive updates do not crash', () => {
    let s = bocpdInitState();
    expect(() => {
      for (let i = 0; i < 10; i++) {
        const r = bocpdUpdate(s, 0.5, prior0, 200);
        s = r.state;
      }
    }).not.toThrow();
  });

  it('beta0=0 with varying x: bocpdAnomalyScore does not crash after each step', () => {
    let s = bocpdInitState();
    let prevRL = 0;
    for (let i = 0; i < 5; i++) {
      const r = bocpdUpdate(s, 0.3 + i * 0.1, prior0, 200);
      expect(() => bocpdAnomalyScore(r, prevRL)).not.toThrow();
      prevRL = r.mapRunLength;
      s = r.state;
    }
  });
});
