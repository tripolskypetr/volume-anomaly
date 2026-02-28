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

  const T = timestamps[n - 1]!;
  let ll   = -mu * T;
  let A    = 0; // recursive compensator

  for (let i = 0; i < n; i++) {
    const ti = timestamps[i]!;

    if (i > 0) {
      const dt = ti - timestamps[i - 1]!;
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

// ─── Anomaly score from Hawkes ────────────────────────────────────────────────

/**
 * Normalised score [0,1]: how much the current λ exceeds the unconditional mean.
 * Unconditional mean: E[λ] = μ / (1 − α/β)  (subcritical Hawkes)
 */
export function hawkesAnomalyScore(
  lambda: number,
  params: HawkesParams,
): number {
  const branching = params.alpha / params.beta;
  if (branching >= 1) return 1; // supercritical → always anomalous
  const meanLambda = params.mu / (1 - branching);
  // sigmoid centred at 2× mean intensity
  return 1 / (1 + Math.exp(-(lambda / meanLambda - 2) * 2));
}
