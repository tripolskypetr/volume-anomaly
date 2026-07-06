/**
 * Bivariate Hawkes process (buy-aggressor / sell-aggressor events,
 * exponential kernel with a SHARED decay β):
 *
 *   λ_b(t) = μ_b + α_bb·A_b(t) + α_bs·A_s(t)
 *   λ_s(t) = μ_s + α_sb·A_b(t) + α_ss·A_s(t)
 *   A_y(t) = Σ_{tᵢ < t, type(i)=y} exp(−β·(t − tᵢ))
 *
 * The cross-excitation asymmetry (α_bb vs α_sb etc.) is the microstructure
 * signature of aggressive directional flow: in a genuine directional burst,
 * buys excite more buys.  The shared β keeps the MLE at 7 parameters —
 * separate decays double the compute for no measured benefit on trade data.
 *
 * Fitting: Nelder-Mead MLE (same approach as the univariate hawkesFit), LL in
 * O(n) via the recursive A_b/A_s trick.  Stationarity constraint: the Perron
 * root of the branching matrix Γ = α/β must stay < 1.
 */

import type { IAggregatedTradeData } from '../types.js';
import { nelderMead } from './optimizer.js';

export interface BiHawkesParams {
  /** Background rates (buy / sell), > 0 */
  muB:  number;
  muS:  number;
  /** Excitation: first index = target type, second = source type; ≥ 0 */
  aBB:  number;
  aBS:  number;
  aSB:  number;
  aSS:  number;
  /** Shared decay rate, > 0 */
  beta: number;
}

export interface BiHawkesFitResult {
  params:       BiHawkesParams;
  logLik:       number;
  /** Perron root of the branching matrix Γ = α/β; < 1 → stationary */
  branching:    number;
  converged:    boolean;
}

/** Perron (largest) eigenvalue of the non-negative 2×2 branching matrix. */
export function biBranching(p: BiHawkesParams): number {
  const g11 = p.aBB / p.beta, g12 = p.aBS / p.beta;
  const g21 = p.aSB / p.beta, g22 = p.aSS / p.beta;
  const tr  = g11 + g22;
  const det = g11 * g22 - g12 * g21;
  const disc = Math.max(0, tr * tr - 4 * det);
  return (tr + Math.sqrt(disc)) / 2;
}

/**
 * Log-likelihood of a merged, time-sorted event stream.
 * times in seconds; isBuy[i] = true for buy-aggressor events.
 */
export function biHawkesLogLikelihood(
  times: number[],
  isBuy: boolean[],
  p:     BiHawkesParams,
): number {
  const n = times.length;
  if (n === 0) return 0;
  const { muB, muS, aBB, aBS, aSB, aSS, beta } = p;
  if (beta <= 0 || muB <= 0 || muS <= 0
    || aBB < 0 || aBS < 0 || aSB < 0 || aSS < 0) return -Infinity;

  const t0 = times[0]!;
  const T  = times[n - 1]! - t0;
  let ll   = -(muB + muS) * T;
  let Ab = 0, As = 0; // decayed event sums by SOURCE type, at the current event

  for (let i = 0; i < n; i++) {
    const ti = times[i]! - t0;
    if (i > 0) {
      const dt = ti - (times[i - 1]! - t0);
      const d  = Math.exp(-beta * dt);
      Ab *= d;
      As *= d;
      if (isBuy[i - 1]) Ab += d; else As += d;
    }
    const lam = isBuy[i]
      ? muB + aBB * Ab + aBS * As
      : muS + aSB * Ab + aSS * As;
    if (lam <= 0 || !Number.isFinite(lam)) return -Infinity;
    ll += Math.log(lam);
    // compensator: event i feeds BOTH target intensities until T
    const decay = (1 - Math.exp(-beta * (T - ti))) / beta;
    ll -= (isBuy[i] ? aBB + aSB : aBS + aSS) * decay;
  }
  return ll;
}

/**
 * Fit via Nelder-Mead MLE on the merged stream.
 * trades must be time-sorted; timestamps converted to seconds internally.
 *
 * maxIter is deliberately modest (default 300): the direction statistic
 * downstream needs the excitation ASYMMETRY, not deep likelihood convergence,
 * and the fit runs per detection baseline.
 */
export function biHawkesFit(
  trades:  IAggregatedTradeData[],
  maxIter: number = 300,
): BiHawkesFitResult {
  const times = trades.map((t) => t.timestamp / 1000);
  const isBuy = trades.map((t) => !t.isBuyerMaker);
  const n = times.length;
  const T = n >= 2 ? times[n - 1]! - times[0]! : 0;
  const nB = isBuy.filter(Boolean).length;
  const nS = n - nB;
  const rateB = nB / (T || 1);
  const rateS = nS / (T || 1);

  const fallback = (): BiHawkesFitResult => {
    const beta = Math.max(rateB + rateS, 1e-3);
    return {
      params: {
        muB: Math.max(rateB, 1e-6), muS: Math.max(rateS, 1e-6),
        aBB: beta * 0.01, aBS: beta * 0.01, aSB: beta * 0.01, aSS: beta * 0.01,
        beta,
      },
      logLik: -Infinity, branching: 0.02, converged: false,
    };
  };
  if (n < 20 || nB === 0 || nS === 0 || T <= 0) return fallback();

  const negLL = (x: number[]): number => {
    const p: BiHawkesParams = {
      muB: x[0]!, muS: x[1]!,
      aBB: x[2]!, aBS: x[3]!, aSB: x[4]!, aSS: x[5]!,
      beta: x[6]!,
    };
    if (p.muB <= 0 || p.muS <= 0 || p.beta <= 0
      || p.aBB < 0 || p.aBS < 0 || p.aSB < 0 || p.aSS < 0) return 1e10;
    if (biBranching(p) >= 0.98) return 1e10;
    const ll = biHawkesLogLikelihood(times, isBuy, p);
    return Number.isFinite(ll) ? -ll : 1e10;
  };

  const beta0 = Math.max(rateB + rateS, 1e-3);
  const x0 = [
    rateB * 0.5, rateS * 0.5,
    beta0 * 0.3, beta0 * 0.1, beta0 * 0.1, beta0 * 0.3,
    beta0,
  ];
  const r = nelderMead(negLL, x0, { maxIter, tol: 1e-6 });
  const p: BiHawkesParams = {
    muB: r.x[0]!, muS: r.x[1]!,
    aBB: r.x[2]!, aBS: r.x[3]!, aSB: r.x[4]!, aSS: r.x[5]!,
    beta: r.x[6]!,
  };
  const invalid = r.fx >= 1e9
    || p.muB <= 0 || p.muS <= 0 || p.beta <= 0
    || p.aBB < 0 || p.aBS < 0 || p.aSB < 0 || p.aSS < 0
    || biBranching(p) >= 1;
  if (invalid) return fallback();
  return { params: p, logLik: -r.fx, branching: biBranching(p), converged: r.converged };
}

/**
 * Excitation share at time t: (λ_b − λ_s) / (λ_b + λ_s) ∈ (−1, +1).
 *
 * The model-based analogue of order-flow imbalance: recent events are
 * weighted by the fitted kernel (the burst onset dominates naturally) and
 * routed through the fitted cross-excitation matrix, so a buy burst that
 * historically excites follow-on buying reads stronger than one that doesn't.
 * Events with time ≤ t contribute (decayed to t); times in seconds.
 */
export function biExcitationShare(
  times: number[],
  isBuy: boolean[],
  t:     number,
  p:     BiHawkesParams,
): number {
  const { muB, muS, aBB, aBS, aSB, aSS, beta } = p;
  let Ab = 0, As = 0;
  for (let i = 0; i < times.length; i++) {
    if (times[i]! > t) break;
    const w = Math.exp(-beta * (t - times[i]!));
    if (isBuy[i]) Ab += w; else As += w;
  }
  const lamB = muB + aBB * Ab + aBS * As;
  const lamS = muS + aSB * Ab + aSS * As;
  const tot  = lamB + lamS;
  return tot > 0 ? (lamB - lamS) / tot : 0;
}
