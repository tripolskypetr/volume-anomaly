/**
 * thirdextreme.test.ts — third wave of "stuck at extremum" coverage.
 *
 * Each group targets a gap not covered by extreme.test.ts or newextreme.test.ts.
 * Tests express CORRECT expected behaviour.
 * Tests that reveal bugs will fail until the underlying code is fixed.
 *
 *  1. bocpdUpdate: hazardLambda = 0      — H = Infinity → log(1−H) = NaN → state collapses
 *  2. hawkesLogLikelihood: beta = 0      — Inf*0 = NaN in compensator → must return −Infinity
 *  3. hawkesFit: n = 0                   — empty timestamps, Poisson fallback with mu = 0
 *  4. hawkesFit: n ≥ 10, T = 0          — all identical timestamps, degenerate optimizer input
 *  5. bocpdUpdate: alpha0 = 0            — alphaN = 0 → df = 0 → scale = Inf → z²/df = NaN
 *  6. hawkesPeakLambda: n = 1            — single event, A = 0, returns exactly μ
 *  7. hawkesPeakLambda: beta = 0         — A grows as 0,1,2,…,n−1 → peak = μ + α·(n−1)
 *  8. volumeImbalance: NaN qty           — documents NaN propagation (garbage-in)
 *  9. cusumFit: NaN in values            — NaN filtered out, mu0 stays finite
 * 10. bocpdBatch: NaN in middle of series — state collapses, subsequent steps stay collapsed
 * 11. cusumBatch: NaN in middle           — NaN guard fires, subsequent valid steps work
 * 12. cusumAnomalyScore: sPos = Infinity  — min(Inf/h, 1) = 1 correctly
 * 13. nelderMead: f = −Infinity           — runs maxIter, converged=false, no crash
 * 14. hawkesAnomalyScore: beta = 0        — supercritical guard (α/0 = Inf ≥ 1) → score = 1
 */

import { describe, it, expect } from 'vitest';
import {
  volumeImbalance,
  hawkesAnomalyScore,
  hawkesLogLikelihood,
  hawkesFit,
  hawkesPeakLambda,
} from '../src/math/hawkes.js';
import {
  cusumFit,
  cusumUpdate,
  cusumInitState,
  cusumBatch,
  cusumAnomalyScore,
} from '../src/math/cusum.js';
import {
  bocpdUpdate,
  bocpdInitState,
  bocpdBatch,
  bocpdAnomalyScore,
} from '../src/math/bocpd.js';
import { nelderMead } from '../src/math/optimizer.js';
import type { IAggregatedTradeData } from '../src/types.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

function makeTrade(qty: number, isBuyerMaker: boolean): IAggregatedTradeData {
  return { id: 1, price: 100, qty, timestamp: 1_000_000, isBuyerMaker };
}

// ─── 1. bocpdUpdate: hazardLambda = 0  (H = Infinity) ─────────────────────────
//
// H = 1/0 = Infinity
// log(H) = Infinity  → logCpMass = Infinity
// log(1 − Infinity) = log(−Infinity) = NaN → all growth entries = NaN
// logNorm = logSumExp(Infinity, NaN) = NaN (Math.max(Inf, NaN) = NaN)
// normLogProbs = [Inf−NaN, NaN−NaN] = [NaN, NaN] → all pruned → state collapses.
//
// This is the "opposite extreme" to hazardLambda = Infinity (H = 0, no resets).
// The ?? guard in cpProbability does NOT catch NaN (only null/undefined),
// so Number.isFinite(rawCp) ? rawCp : 0 must handle it.

describe('bocpdUpdate: hazardLambda = 0 (H = Infinity) — state collapses', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 1 };

  it('does not throw', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior, 0)).not.toThrow();
  });

  it('state.logProbs is empty after one update (all hypotheses produce NaN log-prob)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0);
    expect(r.state.logProbs.length).toBe(0);
  });

  it('cpProbability is 0 (not NaN) — finite guard fires on Math.exp(NaN)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0);
    expect(Number.isFinite(r.cpProbability)).toBe(true);
    expect(r.cpProbability).toBe(0);
  });

  it('mapRunLength is 0 (argmax over empty / all-NaN normLogProbs)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0);
    expect(r.mapRunLength).toBe(0);
  });

  it('state stays empty on all subsequent updates (stuck state)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 10; i++) {
      const r = bocpdUpdate(s, 0.4 + i * 0.01, prior, 0);
      expect(r.state.logProbs.length).toBe(0);
      expect(Number.isFinite(r.cpProbability)).toBe(true);
      s = r.state;
    }
  });

  it('bocpdAnomalyScore does not throw on collapsed output', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0);
    expect(() => bocpdAnomalyScore(r, 10)).not.toThrow();
  });

  it('bocpdAnomalyScore returns finite score after collapse', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior, 0);
    const score = bocpdAnomalyScore(r, 10);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});

// ─── 2. hawkesLogLikelihood: beta = 0 ────────────────────────────────────────
//
// With β = 0 the exponential kernel never decays — the process is supercritical
// and the log-likelihood is undefined.  The implementation should return −Infinity
// (same contract as the λᵢ ≤ 0 guard) so the optimizer never converges there.
//
// Current bug path:
//   compensator = (α/β)·(1 − exp(−β·(T−tᵢ))) = (α/0)·(1−exp(0)) = Inf·0 = NaN
//   ll −= NaN  →  ll = NaN  (silently wrong, not −Infinity)
//
// Correct behaviour: return −Infinity.

describe('hawkesLogLikelihood: beta = 0 — must return −Infinity, not NaN', () => {
  const ts5 = [0, 1, 2, 3, 4];

  it('beta = 0 returns −Infinity (not NaN)', () => {
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 0.3, beta: 0 });
    expect(ll).toBe(-Infinity);
  });

  it('beta = 0 result is not NaN', () => {
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 0.3, beta: 0 });
    expect(Number.isNaN(ll)).toBe(false);
  });

  it('beta < 0 also returns −Infinity (kernel diverges for negative β)', () => {
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 0.3, beta: -1 });
    expect(ll).toBe(-Infinity);
  });

  it('beta = −0 (negative zero) returns −Infinity', () => {
    const ll = hawkesLogLikelihood(ts5, { mu: 1, alpha: 0.3, beta: -0 });
    expect(ll).toBe(-Infinity);
  });

  it('beta = 0 on single timestamp also returns −Infinity', () => {
    const ll = hawkesLogLikelihood([42], { mu: 2, alpha: 0.1, beta: 0 });
    expect(ll).toBe(-Infinity);
  });
});

// ─── 3. hawkesFit: n = 0 (empty timestamps) ───────────────────────────────────
//
// timestamps.length = 0 < 10 → Poisson fallback branch.
// mu = 0 / (undefined − undefined || 1) = 0 / 1 = 0.
// Returned params: { mu: 0, alpha: 0.01, beta: 1, converged: false }.
//
// Downstream risk: hawkesAnomalyScore with mu=0 returns score stuck at 1 for
// any non-zero peakLambda.  The test ensures no crash and documents the params.

describe('hawkesFit: n = 0 (empty timestamps) — Poisson fallback', () => {
  it('does not throw', () => {
    expect(() => hawkesFit([])).not.toThrow();
  });

  it('converged = false (fallback, no optimizer run)', () => {
    expect(hawkesFit([]).converged).toBe(false);
  });

  it('logLik = −Infinity (fallback sentinel)', () => {
    expect(hawkesFit([]).logLik).toBe(-Infinity);
  });

  it('params.alpha > 0', () => {
    expect(hawkesFit([]).params.alpha).toBeGreaterThan(0);
  });

  it('params.beta > 0', () => {
    expect(hawkesFit([]).params.beta).toBeGreaterThan(0);
  });
});

// ─── 4. hawkesFit: n ≥ 10, all timestamps identical (T = 0) ──────────────────
//
// T = 0 makes the compensator = (α/β)·(1−exp(−β·0)) = 0 for all events.
// The LL simplifies to Σ log(μ + α·Aᵢ), which is unbounded above (no finite MLE).
// The optimizer runs maxIter without converging → hits the invalid branch → fallback.
// Key contract: no crash, fallback params all positive.

describe('hawkesFit: n ≥ 10, T = 0 (all identical timestamps) — fallback', () => {
  const ts10same = Array.from({ length: 10 }, () => 5000);
  const ts20same = Array.from({ length: 20 }, () => 1_700_000_000);

  it('n=10 identical timestamps: does not throw', () => {
    expect(() => hawkesFit(ts10same)).not.toThrow();
  });

  it('n=10 identical timestamps: params.mu > 0', () => {
    expect(hawkesFit(ts10same).params.mu).toBeGreaterThan(0);
  });

  it('n=10 identical timestamps: params.alpha > 0', () => {
    expect(hawkesFit(ts10same).params.alpha).toBeGreaterThan(0);
  });

  it('n=10 identical timestamps: params.beta > 0', () => {
    expect(hawkesFit(ts10same).params.beta).toBeGreaterThan(0);
  });

  it('n=10 identical timestamps: stationarity in [0, 1)', () => {
    const r = hawkesFit(ts10same);
    expect(r.stationarity).toBeGreaterThanOrEqual(0);
    expect(r.stationarity).toBeLessThan(1);
  });

  it('n=20 identical Unix-epoch timestamps: does not throw', () => {
    expect(() => hawkesFit(ts20same)).not.toThrow();
  });

  it('n=20 identical timestamps: params are all positive', () => {
    const r = hawkesFit(ts20same);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.params.alpha).toBeGreaterThan(0);
    expect(r.params.beta).toBeGreaterThan(0);
  });
});

// ─── 5. bocpdUpdate: alpha0 = 0 ──────────────────────────────────────────────
//
// alphaN = alpha0 + n/2 = 0 for n=0.  df = 2·alphaN = 0.
// scale = betaN·(κN+1)/(alphaN·κN) = finite/0 = Infinity (κN > 0).
// z = (x − muN)/sqrt(Infinity) = 0.
// log(1 + z²/df) = log(1 + 0/0) = log(NaN) = NaN.
// logProb = (finite) − NaN = NaN → all hypotheses pruned → state collapses.
//
// Distinct from kappa0=0 (covered in newextreme.test.ts) — different NaN path.

describe('bocpdUpdate: alpha0 = 0 — df = 0 → NaN log-prob → collapse', () => {
  const prior0 = { mu0: 0.5, kappa0: 1, alpha0: 0, beta0: 1 };

  it('does not throw', () => {
    expect(() => bocpdUpdate(bocpdInitState(), 0.5, prior0)).not.toThrow();
  });

  it('state collapses (all hypotheses produce NaN log-prob)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior0);
    expect(r.state.logProbs.length).toBe(0);
  });

  it('cpProbability is 0 (finite guard handles NaN)', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior0);
    expect(Number.isFinite(r.cpProbability)).toBe(true);
    expect(r.cpProbability).toBe(0);
  });

  it('mapRunLength is a non-negative integer', () => {
    const r = bocpdUpdate(bocpdInitState(), 0.5, prior0);
    expect(Number.isInteger(r.mapRunLength)).toBe(true);
    expect(r.mapRunLength).toBeGreaterThanOrEqual(0);
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
});

// ─── 6. hawkesPeakLambda: n = 1 (single event) ───────────────────────────────
//
// For i=0: the `if (i > 0)` block is skipped, so A = 0 (initial value).
// λ(t₀) = μ + α·0 = μ.
// The peak is initialized to μ and the single event contributes nothing extra.
// Returns exactly μ — no self-excitation possible with one event.

describe('hawkesPeakLambda: n = 1 — returns exactly μ (A = 0)', () => {
  it('n=1: returns exactly mu', () => {
    expect(hawkesPeakLambda([100], { mu: 3, alpha: 0.5, beta: 2 })).toBe(3);
  });

  it('n=1: timestamp value does not affect result (only A matters)', () => {
    const p = { mu: 2.5, alpha: 1, beta: 0.5 };
    expect(hawkesPeakLambda([0],       p)).toBe(2.5);
    expect(hawkesPeakLambda([5000],    p)).toBe(2.5);
    expect(hawkesPeakLambda([1e9],     p)).toBe(2.5);
  });

  it('n=1: result is finite', () => {
    expect(Number.isFinite(hawkesPeakLambda([42], { mu: 1, alpha: 0.3, beta: 2 }))).toBe(true);
  });

  it('n=1 result equals n=0 result (no events → peak = μ in both cases)', () => {
    const p = { mu: 4, alpha: 0.5, beta: 1 };
    expect(hawkesPeakLambda([99], p)).toBe(hawkesPeakLambda([], p));
  });
});

// ─── 7. hawkesPeakLambda: beta = 0 ───────────────────────────────────────────
//
// exp(−0·dt) = 1 for any dt.
// A(i) = 1·(1 + A(i−1))  →  A(i) = i  (0, 1, 2, …, n−1 independent of timestamps)
// λ(tᵢ) = μ + α·i  →  peak = μ + α·(n−1).
//
// The value is large but finite.
// hawkesAnomalyScore(peak, {beta:0}, …) immediately returns 1 via the
// supercritical guard (α/β = α/0 = Infinity ≥ 1).

describe('hawkesPeakLambda: beta = 0 — A grows as 0,1,…,n−1; peak = μ + α·(n−1)', () => {
  const p = { mu: 1, alpha: 0.5, beta: 0 };

  it('n=1, beta=0: returns μ (A=0 for first event)', () => {
    expect(hawkesPeakLambda([0], p)).toBe(1);
  });

  it('n=10, beta=0: returns μ + α·9 = 1 + 0.5·9 = 5.5', () => {
    const ts = Array.from({ length: 10 }, (_, i) => i);
    expect(hawkesPeakLambda(ts, p)).toBeCloseTo(5.5, 10);
  });

  it('n=10, beta=0: timestamps do not matter (exp(−0·dt)=1 always)', () => {
    const pDense  = { mu: 1, alpha: 0.5, beta: 0 };
    const tsSparse = Array.from({ length: 10 }, (_, i) => i * 1000);
    const tsSame   = Array.from({ length: 10 }, () => 5000);
    // A(i)=i in both cases since exp(0)=1 regardless of dt
    expect(hawkesPeakLambda(tsSparse, pDense)).toBeCloseTo(5.5, 10);
    expect(hawkesPeakLambda(tsSame,   pDense)).toBeCloseTo(5.5, 10);
  });

  it('peak is finite (not Infinity)', () => {
    const ts = Array.from({ length: 100 }, (_, i) => i);
    expect(Number.isFinite(hawkesPeakLambda(ts, p))).toBe(true);
  });

  it('n=100, beta=0: peak = 1 + 0.5·99 = 50.5', () => {
    const ts = Array.from({ length: 100 }, (_, i) => i);
    expect(hawkesPeakLambda(ts, p)).toBeCloseTo(50.5, 10);
  });

  it('hawkesAnomalyScore with beta=0 returns 1 (supercritical guard)', () => {
    const ts = Array.from({ length: 10 }, (_, i) => i);
    const peak = hawkesPeakLambda(ts, p);
    expect(hawkesAnomalyScore(peak, p, 0)).toBe(1);
  });
});

// ─── 8. volumeImbalance: NaN qty ──────────────────────────────────────────────
//
// If any trade has qty = NaN: buyVol or sellVol becomes NaN, total = NaN,
// and the return value is NaN.  This is documented "garbage-in, garbage-out"
// behaviour; the function is not designed to sanitise its input.
//
// The key contract: no crash.

describe('volumeImbalance: NaN qty — garbage-in, no crash', () => {
  it('single trade with qty = NaN: does not throw', () => {
    expect(() => volumeImbalance([makeTrade(NaN, false)])).not.toThrow();
  });

  it('single trade with qty = NaN: returns NaN (propagation)', () => {
    expect(Number.isNaN(volumeImbalance([makeTrade(NaN, false)]))).toBe(true);
  });

  it('mixed valid + NaN trades: does not throw', () => {
    expect(() =>
      volumeImbalance([makeTrade(1, false), makeTrade(NaN, true)]),
    ).not.toThrow();
  });

  it('all trades qty = NaN: does not throw', () => {
    expect(() =>
      volumeImbalance([makeTrade(NaN, false), makeTrade(NaN, true)]),
    ).not.toThrow();
  });
});

// ─── 9. cusumFit: NaN in values ───────────────────────────────────────────────
//
// If values contains NaN:
//   Before fix: mu0 = NaN (NaN propagates through sum/n).
//               std0 is rescued by `|| 1e-6` (NaN is falsy) → std0 = 1e-6.
//               But mu0 = NaN means cusumUpdate later poisons the accumulator
//               via `x − NaN = NaN` → Math.max(0, NaN) = NaN → state = NaN.
//
//   After fix: NaN values are filtered before computing mean/variance.
//               cusumFit([0.3, NaN, 0.5]) → mu0 = 0.4, h/k from std of [0.3, 0.5].
//               cusumFit([NaN, NaN]) → same as empty → defaults.
//
// Correct contract: all returned params must be finite.

describe('cusumFit: NaN in values — NaN filtered, all params finite', () => {
  it('cusumFit([NaN]): mu0 is finite', () => {
    const p = cusumFit([NaN]);
    expect(Number.isFinite(p.mu0)).toBe(true);
  });

  it('cusumFit([NaN]): h is finite and positive', () => {
    const p = cusumFit([NaN]);
    expect(Number.isFinite(p.h)).toBe(true);
    expect(p.h).toBeGreaterThan(0);
  });

  it('cusumFit([0.3, NaN, 0.5]): mu0 = mean([0.3, 0.5]) = 0.4', () => {
    const p = cusumFit([0.3, NaN, 0.5]);
    expect(p.mu0).toBeCloseTo(0.4, 10);
  });

  it('cusumFit([0.3, NaN, 0.5]): all params finite', () => {
    const p = cusumFit([0.3, NaN, 0.5]);
    expect(Number.isFinite(p.mu0)).toBe(true);
    expect(Number.isFinite(p.std0)).toBe(true);
    expect(Number.isFinite(p.k)).toBe(true);
    expect(Number.isFinite(p.h)).toBe(true);
  });

  it('cusumFit([NaN, NaN, NaN]): falls back to defaults (mu0=0)', () => {
    const p = cusumFit([NaN, NaN, NaN]);
    expect(Number.isFinite(p.mu0)).toBe(true);
    expect(Number.isFinite(p.h)).toBe(true);
    expect(p.h).toBeGreaterThan(0);
  });

  it('subsequent cusumUpdate on NaN-filtered params does not poison state', () => {
    const p = cusumFit([0.3, NaN, 0.5]);
    const upd = cusumUpdate(cusumInitState(), 0.4, p);
    expect(Number.isFinite(upd.state.sPos)).toBe(true);
    expect(Number.isFinite(upd.state.sNeg)).toBe(true);
  });
});

// ─── 10. bocpdBatch: NaN in middle of series ──────────────────────────────────
//
// After bocpdUpdate with x=NaN, state collapses to {logProbs:[], …}.
// Subsequent updates on empty state produce prevN=0 → logCpMass=−Infinity →
// normLogProbs=[NaN] → collapse again → stuck collapsed forever.
// cpProbability must be 0 (finite) at every step, including post-collapse.

describe('bocpdBatch: NaN in middle — state collapses, no crash', () => {
  const prior = { mu0: 0.5, kappa0: 1, alpha0: 1, beta0: 1 };
  const series = [0.3, 0.5, NaN, 0.4, 0.6];

  it('does not throw', () => {
    expect(() => bocpdBatch(series, prior)).not.toThrow();
  });

  it('returns arrays of correct length', () => {
    const r = bocpdBatch(series, prior);
    expect(r.cpProbs.length).toBe(series.length);
    expect(r.mapRunLengths.length).toBe(series.length);
  });

  it('all cpProbs are finite (no NaN from collapsed state)', () => {
    const { cpProbs } = bocpdBatch(series, prior);
    expect(cpProbs.every(Number.isFinite)).toBe(true);
  });

  it('all mapRunLengths are non-negative integers', () => {
    const { mapRunLengths } = bocpdBatch(series, prior);
    for (const rl of mapRunLengths) {
      expect(Number.isInteger(rl)).toBe(true);
      expect(rl).toBeGreaterThanOrEqual(0);
    }
  });

  it('cpProbs after NaN index are 0 (state stays collapsed)', () => {
    const { cpProbs } = bocpdBatch(series, prior);
    // index 2 is NaN → collapse; indices 3 and 4 run on empty state → cpProb=0
    expect(cpProbs[3]).toBe(0);
    expect(cpProbs[4]).toBe(0);
  });
});

// ─── 11. cusumBatch: NaN in middle of series ──────────────────────────────────
//
// cusumUpdate guards NaN x at the start: returns unchanged state, alarm=false.
// After the NaN entry, the accumulator is unmodified and subsequent valid
// observations continue accumulating normally.

describe('cusumBatch: NaN in middle — guard fires, subsequent steps work', () => {
  const params = { mu0: 0.3, std0: 0.1, k: 0.05, h: 0.5 };
  const series = [0.3, 0.4, NaN, 0.5, 0.6];

  it('does not throw', () => {
    expect(() => cusumBatch(series, params)).not.toThrow();
  });

  it('final state.sPos is finite', () => {
    const { state } = cusumBatch(series, params);
    expect(Number.isFinite(state.sPos)).toBe(true);
  });

  it('final state.sNeg is finite', () => {
    const { state } = cusumBatch(series, params);
    expect(Number.isFinite(state.sNeg)).toBe(true);
  });

  it('NaN index (2) does not appear in alarmIndices', () => {
    const { alarmIndices } = cusumBatch(series, params);
    expect(alarmIndices).not.toContain(2);
  });

  it('same final state as running without NaN (NaN is effectively a no-op)', () => {
    const seriesClean = [0.3, 0.4, 0.5, 0.6];
    const withNaN    = cusumBatch(series,      params).state;
    const withoutNaN = cusumBatch(seriesClean, params).state;
    expect(withNaN.sPos).toBeCloseTo(withoutNaN.sPos, 10);
    expect(withNaN.sNeg).toBeCloseTo(withoutNaN.sNeg, 10);
  });
});

// ─── 12. cusumAnomalyScore: sPos = Infinity ───────────────────────────────────
//
// s = Math.max(Infinity, sNeg) = Infinity.
// h > 0 and finite → guard passes.
// Math.min(Infinity / h, 1) = Math.min(Infinity, 1) = 1 in JS (Infinity > 1).
// Must return exactly 1, not Infinity and not NaN.

describe('cusumAnomalyScore: sPos = Infinity — returns 1 correctly', () => {
  const params = { mu0: 0.2, std0: 0.1, k: 0.05, h: 0.5 };

  it('sPos = Infinity, sNeg = 0: returns 1', () => {
    expect(cusumAnomalyScore({ sPos: Infinity, sNeg: 0, n: 5 }, params)).toBe(1);
  });

  it('sPos = 0, sNeg = Infinity: returns 1', () => {
    expect(cusumAnomalyScore({ sPos: 0, sNeg: Infinity, n: 5 }, params)).toBe(1);
  });

  it('sPos = sNeg = Infinity: returns 1', () => {
    expect(cusumAnomalyScore({ sPos: Infinity, sNeg: Infinity, n: 5 }, params)).toBe(1);
  });

  it('return value is not NaN', () => {
    const score = cusumAnomalyScore({ sPos: Infinity, sNeg: 0, n: 5 }, params);
    expect(Number.isNaN(score)).toBe(false);
  });

  it('return value is not > 1 (Math.min clamps correctly)', () => {
    const score = cusumAnomalyScore({ sPos: Infinity, sNeg: 0, n: 5 }, params);
    expect(score).toBeLessThanOrEqual(1);
  });
});

// ─── 13. nelderMead: f always returns −Infinity ───────────────────────────────
//
// Complementary to the +Infinity and NaN cases in extreme.test.ts.
//
// With all fvals = −Infinity:
//   spread = fvals[n] − fvals[0] = −Inf − (−Inf) = NaN → NaN < tol = false
//   fr = −Infinity:  fr < fvals[0] = −Inf < −Inf = false → no expansion
//                    fr < fvals[n−1] = −Inf < −Inf = false → contraction
//   fc = −Infinity:  fc < fvals[n] = −Inf < −Inf = false → shrink
//   Shrink: simplex collapses; fvals all stay −Infinity.
//   Repeats maxIter times → converged=false, fx=−Infinity.

describe('nelderMead: f = −Infinity — runs maxIter, converged=false, no crash', () => {
  it('1D: does not throw', () => {
    expect(() => nelderMead(() => -Infinity, [1], { maxIter: 50 })).not.toThrow();
  });

  it('2D: does not throw', () => {
    expect(() => nelderMead(() => -Infinity, [1, 2], { maxIter: 100 })).not.toThrow();
  });

  it('2D: converged = false (no finite minimum)', () => {
    const r = nelderMead(() => -Infinity, [1, 2], { maxIter: 100 });
    expect(r.converged).toBe(false);
  });

  it('2D: runs exactly maxIter iterations', () => {
    const r = nelderMead(() => -Infinity, [1, 2], { maxIter: 77 });
    expect(r.iters).toBe(77);
  });

  it('2D: fx = −Infinity (documents the return value)', () => {
    const r = nelderMead(() => -Infinity, [1, 2], { maxIter: 50 });
    expect(r.fx).toBe(-Infinity);
  });

  it('3D: x components are all finite after maxIter (simplex has collapsed to a point)', () => {
    const r = nelderMead(() => -Infinity, [1, 2, 3], { maxIter: 50 });
    expect(r.x.every(Number.isFinite)).toBe(true);
  });
});

// ─── 14. hawkesAnomalyScore: beta = 0 ────────────────────────────────────────
//
// branching = α / β = α / 0 = Infinity.
// Infinity ≥ 1 → supercritical guard fires immediately → return 1.
// This is consistent with hawkesPeakLambda(beta=0) returning a potentially
// huge but finite peak, and the score correctly signalling an anomaly.

describe('hawkesAnomalyScore: beta = 0 — supercritical guard fires, returns 1', () => {
  it('alpha > 0, beta = 0: branching = Infinity ≥ 1 → returns 1', () => {
    expect(hawkesAnomalyScore(2, { mu: 1, alpha: 0.5, beta: 0 }, 0)).toBe(1);
  });

  it('returns 1 regardless of peakLambda (guard fires before any ratio)', () => {
    expect(hawkesAnomalyScore(0,        { mu: 1, alpha: 1, beta: 0 }, 0)).toBe(1);
    expect(hawkesAnomalyScore(Infinity, { mu: 1, alpha: 1, beta: 0 }, 0)).toBe(1);
    expect(hawkesAnomalyScore(NaN,      { mu: 1, alpha: 1, beta: 0 }, 0)).toBe(1);
  });

  it('returns 1 regardless of empiricalRate', () => {
    expect(hawkesAnomalyScore(1, { mu: 1, alpha: 1, beta: 0 }, Infinity)).toBe(1);
    expect(hawkesAnomalyScore(1, { mu: 1, alpha: 1, beta: 0 }, 0)).toBe(1);
  });

  it('result is not NaN', () => {
    expect(Number.isNaN(hawkesAnomalyScore(1, { mu: 1, alpha: 0.3, beta: 0 }, 0))).toBe(false);
  });
});
