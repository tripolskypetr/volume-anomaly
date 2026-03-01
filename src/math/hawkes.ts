/**
 * Hawkes process (univariate, exponential kernel) + volume imbalance.
 *
 * Model:  λ(t) = μ + Σ_{tᵢ < t} α · exp(−β · (t − tᵢ))
 *
 * Fitting: maximum likelihood via Nelder-Mead (same approach as garch lib).
 * The LL is computed in O(n) using the recursive A(i) trick:
 *
 *   A(i) = exp(−β · Δᵢ) · (1 + A(i−1))
 *   ln L = −μ·T + (α/β)·(Σ exp(−β·(T−tᵢ)) − n) + Σ ln λ(tᵢ)
 */

import type { HawkesParams, IAggregatedTradeData } from '../types.js';
import { nelderMead } from './optimizer.js';

// ─── Volume imbalance ────────────────────────────────────────────────────────

/**
 * Compute normalised buy/sell volume imbalance over a trade window.
 * Returns value in [-1, +1]:  +1 = all buys,  -1 = all sells.
 */
export function volumeImbalance(trades: IAggregatedTradeData[]): number {
  let buyVol  = 0;
  let sellVol = 0;
  for (const t of trades) {
    // isBuyerMaker = true  → sell aggressor (taker sold into bid)
    // isBuyerMaker = false → buy  aggressor (taker bought ask)
    if (t.isBuyerMaker) {
      sellVol += t.qty;
    } else {
      buyVol  += t.qty;
    }
  }
  const total = buyVol + sellVol;
  if (total === 0) return 0;
  return (buyVol - sellVol) / total;
}

// ─── Log-likelihood (O(n) recursive) ────────────────────────────────────────

/**
 * Ogata (1988) log-likelihood for univariate Hawkes with exponential kernel.
 * timestamps must be sorted ascending, in seconds (or any consistent unit).
 */
export function hawkesLogLikelihood(
  timestamps: number[],
  params: HawkesParams,
): number {
  const { mu, alpha, beta } = params;
  const n = timestamps.length;
  if (n === 0) return 0;

  // Use observation window length, not absolute time, so the LL is invariant
  // to timestamp origin (works for both t0=0 and Unix-epoch seconds).
  const t0 = timestamps[0]!;
  const T  = timestamps[n - 1]! - t0;  // window length
  let ll   = -mu * T;
  let A    = 0; // recursive compensator

  for (let i = 0; i < n; i++) {
    const ti = timestamps[i]! - t0;    // shift to origin

    if (i > 0) {
      const dt = ti - (timestamps[i - 1]! - t0);
      A = Math.exp(-beta * dt) * (1 + A);
    }

    const lambda_i = mu + alpha * A;
    if (lambda_i <= 0) return -Infinity;

    ll += Math.log(lambda_i);
    // compensator contribution for this event
    ll -= (alpha / beta) * (1 - Math.exp(-beta * (T - ti)));
  }

  return ll;
}

// ─── MLE fitting ─────────────────────────────────────────────────────────────

export interface HawkesFitResult {
  params:      HawkesParams;
  logLik:      number;
  /** α/β < 1 → subcritical (stationary process) */
  stationarity: number;
  converged:   boolean;
}

/**
 * Fit Hawkes(1,exp) via Nelder-Mead MLE.
 * timestamps – sorted array of trade times in **seconds**.
 */
export function hawkesFit(timestamps: number[]): HawkesFitResult {
  if (timestamps.length < 10) {
    // Not enough data — return flat-rate Poisson
    const mu = timestamps.length / (timestamps[timestamps.length - 1]! - timestamps[0]! || 1);
    return {
      params:       { mu, alpha: 0.01, beta: 1.0 },
      logLik:       -Infinity,
      stationarity: 0,
      converged:    false,
    };
  }

  const negLL = ([mu, alpha, beta]: number[]) => {
    if (mu! <= 0 || alpha! <= 0 || beta! <= 0 || alpha! >= beta!) return 1e10;
    return -hawkesLogLikelihood(timestamps, { mu: mu!, alpha: alpha!, beta: beta! });
  };

  // Starting point: empirical rate, branching ratio 0.5
  const T   = timestamps[timestamps.length - 1]! - timestamps[0]!;
  const mu0 = timestamps.length / (T || 1);
  const x0  = [mu0 * 0.5, mu0 * 0.4, mu0];

  const result = nelderMead(negLL, x0, { maxIter: 1000, tol: 1e-8 });
  const [mu, alpha, beta] = result.x;

  // If the optimizer landed in the penalty region, fall back to a safe
  // near-Poisson parameterisation so downstream scoring stays conservative.
  const invalid = !result.converged || result.fx >= 1e9
    || mu! <= 0 || alpha! <= 0 || beta! <= 0 || alpha! >= beta!;

  if (invalid) {
    const muFallback = timestamps.length / (T || 1);
    return {
      params:       { mu: muFallback, alpha: muFallback * 0.01, beta: muFallback },
      logLik:       -Infinity,
      stationarity: 0.01,
      converged:    false,
    };
  }

  const params: HawkesParams = { mu: mu!, alpha: alpha!, beta: beta! };
  return {
    params,
    logLik:       -result.fx,
    stationarity: alpha! / beta!,
    converged:    result.converged,
  };
}

// ─── Conditional intensity at time t ─────────────────────────────────────────

/**
 * Compute λ(t) — conditional intensity at time t given history.
 * timestamps must be sorted ascending and all < t.
 */
export function hawkesLambda(
  t: number,
  timestamps: number[],
  params: HawkesParams,
): number {
  const { mu, alpha, beta } = params;
  let sum = 0;
  for (const ti of timestamps) {
    if (ti >= t) break;
    sum += Math.exp(-beta * (t - ti));
  }
  return mu + alpha * sum;
}

/**
 * Compute the peak conditional intensity over all events in the window.
 *
 * Uses the O(n) recursive A(i) trick from the log-likelihood:
 *   A(i) = exp(−β·Δᵢ) · (1 + A(i−1))
 *   λ(tᵢ) = μ + α·A(i)
 *
 * Taking the maximum over all events (not just the last one) is essential
 * for detecting bursts that occur in the middle of a detection window — at
 * the last event the kernel has already decayed, so λ(t_last) can be close
 * to μ even when a spike occurred earlier.
 *
 * timestamps must be sorted ascending (in seconds, same unit as params).
 */
export function hawkesPeakLambda(
  timestamps: number[],
  params: HawkesParams,
): number {
  const { mu, alpha, beta } = params;
  const n = timestamps.length;
  if (n === 0) return mu;

  let A    = 0;
  let peak = mu; // λ before first event = μ

  for (let i = 0; i < n; i++) {
    if (i > 0) {
      const dt = timestamps[i]! - timestamps[i - 1]!;
      A = Math.exp(-beta * dt) * (1 + A);
    }
    const lam = mu + alpha * A;
    if (lam > peak) peak = lam;
  }
  return peak;
}

// ─── Anomaly score from Hawkes ────────────────────────────────────────────────

/**
 * Normalised score [0,1]: how much the arrival rate exceeds the baseline.
 *
 * Two complementary signals are combined with max():
 *
 *  1. Intensity ratio: peakLambda / E[λ].
 *     E[λ] = μ/(1−α/β) — the unconditional mean of the fitted process.
 *     Captures self-excitation bursts when the MLE branching ratio is large.
 *
 *  2. Empirical rate ratio: empiricalRate / μ.
 *     Compares the raw arrival density in the detection window to the fitted
 *     baseline rate μ. This is model-agnostic and fires even when the MLE
 *     assigns alpha ≈ 0 (Poisson baseline), where the intensity ratio stays
 *     near 1 regardless of how many events arrived.
 *
 * Both ratios are fed through the same sigmoid centred at 2× baseline, so
 * the score is 0 at baseline rate, 0.5 at 2×, and approaches 1 at ≥ 4×.
 *
 * @param peakLambda    Peak λ(tᵢ) over the detection window (from hawkesPeakLambda).
 * @param params        Fitted Hawkes parameters.
 * @param empiricalRate Observed arrival rate in the detection window (events/s).
 *                      Pass 0 to use only the intensity ratio.
 */
export function hawkesAnomalyScore(
  peakLambda: number,
  params: HawkesParams,
  empiricalRate = 0,
): number {
  const branching  = params.alpha / params.beta;
  if (branching >= 1) return 1; // supercritical → always anomalous
  const meanLambda = params.mu / (1 - branching);

  // sigmoid centred at 2× baseline
  const sig = (ratio: number) => 1 / (1 + Math.exp(-(ratio - 2) * 2));

  // meanLambda = 0 when mu = 0: ratio = peakLambda / 0 = Infinity (score=1) when
  // peakLambda > 0, or NaN (0/0) when peakLambda = 0.  Guard the NaN case.
  const intensityScore = meanLambda > 0 ? sig(peakLambda / meanLambda)
    : peakLambda > 0 ? 1 : 0;
  const rateScore      = empiricalRate > 0
    ? (params.mu > 0 ? sig(empiricalRate / params.mu) : 1)
    : 0;

  return Math.max(intensityScore, rateScore);
}
