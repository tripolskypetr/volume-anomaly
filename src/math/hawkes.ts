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
  // When total = Infinity (an overflowed qty) the division (buyVol−sellVol)/Infinity
  // is NaN even for a one-sided burst (Inf/Inf = NaN in IEEE 754).
  // Compare sides directly to get the correct ±1 / 0 answer.
  // NaN total (from NaN qty) falls through to the regular division — GIGO.
  if (total === Infinity) {
    if (buyVol === sellVol) return 0;   // both Infinity — symmetric burst
    return buyVol > sellVol ? 1 : -1;
  }
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
  // β ≤ 0: kernel exp(−β·dt) does not decay (diverges or flat).
  // Compensator = (α/β)·(1−exp(−β·(T−tᵢ))) → Inf·0 = NaN when β=0.
  // Return −Infinity so the optimizer treats this as an infeasible region.
  if (beta <= 0) return -Infinity;

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

// ─── Compensator excess (time-rescaling channel) ─────────────────────────────

/**
 * Rolling "excess events" statistic: for each event i with a full lookback,
 * compare the OBSERVED count N in (tᵢ − horizon, tᵢ] with the model's
 * conditional compensator Λ (expected events given history and fitted
 * self-excitation), as a Poisson-standardized excess:
 *
 *   excess(i) = (N − Λ) / √max(Λ, ε)
 *
 * Λ(a, b) = μ·(b−a) + (α/β)·[ A(a)·(1 − e^{−β(b−a)}) + Σ_{a<tⱼ≤b}(1 − e^{−β(b−tⱼ)}) ]
 * where A(a) = Σ_{tⱼ≤a} e^{−β(a−tⱼ)}.
 *
 * This is the time-rescaling idea in windowed form: a follow-on cluster that
 * the fitted kernel already predicts (the decay tail of a burst) has high Λ
 * and scores LOW; an exogenous escalation the model cannot explain scores
 * HIGH.  A raw rate z cannot make this distinction.
 *
 * Both sums are carried by monotone recursions (the in-window decayed sum via
 * B(i), the boundary sum via a decayed accumulator on the trailing pointer),
 * so the whole series is O(n).  Only full-horizon windows are scored — same
 * contract as the rolling rate statistic; when the array spans less than the
 * horizon, a single whole-window sample is returned (firstIdx = -1).
 *
 * timestamps sorted ascending, in seconds (same unit as params).
 */
export function hawkesExcessSeries(
  timestamps: number[],
  params:     HawkesParams,
  horizon:    number,
): { excess: number[]; firstIdx: number } {
  const { mu, alpha, beta } = params;
  const n = timestamps.length;
  const excess: number[] = [];
  let firstIdx = -1;
  if (n < 2 || beta <= 0) return { excess, firstIdx };

  const t0   = timestamps[0]!;
  const span = timestamps[n - 1]! - t0;

  // B(i) = Σ_{j≤i} e^{−β(tᵢ−tⱼ)}  (includes event i itself)
  const B = new Array<number>(n);
  B[0] = 1;
  for (let i = 1; i < n; i++) {
    B[i] = 1 + B[i - 1]! * Math.exp(-beta * (timestamps[i]! - timestamps[i - 1]!));
  }

  const lambdaOf = (a: number, b: number, Aa: number, N: number, Wb: number): number => {
    const inWin = N - (Wb - Aa * Math.exp(-beta * (b - a)));
    return mu * (b - a) + (alpha / beta) * (Aa * (1 - Math.exp(-beta * (b - a))) + inWin);
  };

  if (span >= horizon) {
    let j = 0;          // first event index with t > a  (a = tᵢ − horizon)
    let Aa = 0;         // Σ_{t≤a} e^{−β(a−t)}
    let aPrev = t0;     // time Aa is currently decayed to
    for (let i = 0; i < n; i++) {
      if (timestamps[i]! - t0 < horizon) continue;
      if (firstIdx < 0) firstIdx = i;
      const b = timestamps[i]!;
      const a = b - horizon;
      // advance the boundary accumulator to a: decay, then absorb events ≤ a
      Aa *= Math.exp(-beta * (a - aPrev));
      while (j < n && timestamps[j]! <= a) {
        Aa += Math.exp(-beta * (a - timestamps[j]!));
        j++;
      }
      aPrev = a;
      const N   = i - j + 1;
      const lam = lambdaOf(a, b, Aa, N, B[i]!);
      excess.push((N - lam) / Math.sqrt(Math.max(lam, 1e-9)));
    }
  }
  if (excess.length === 0) {
    // whole-window fallback (span < horizon): cold start, A(a) = 0
    const b = timestamps[n - 1]!;
    const lam = lambdaOf(t0, b, 0, n, B[n - 1]!);
    excess.push((n - lam) / Math.sqrt(Math.max(lam, 1e-9)));
    firstIdx = -1;
  }
  return { excess, firstIdx };
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
 * Both ratios are fed through the same sigmoid centred at 2× baseline:
 * score ≈ 0.12 at baseline rate (1×), 0.5 at 2×, and approaches 1 at ≥ 4×.
 * Note the ≈ 0.12 floor at baseline — the score never reaches exactly 0.
 *
 * NOTE: these THEORETICAL baselines (E[λ], μ) are adequate for near-Poisson
 * synthetic data only.  On real trade streams rates fluctuate several-fold
 * between adjacent windows of a perfectly normal market, so "2× the fitted μ"
 * is routine noise — VolumeAnomalyDetector does not use this function; it
 * scores robust z of rolling rates against baselines measured on the training
 * window instead.
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
  // sigmoid centred at 2× baseline
  const sig = (ratio: number) => 1 / (1 + Math.exp(-(ratio - 2) * 2));

  const branching  = params.alpha / params.beta;
  if (branching >= 1) return 1; // supercritical → always anomalous
  const meanLambda = params.mu / (1 - branching);

  // meanLambda = 0 when mu = 0: ratio = peakLambda / 0 = Infinity (score=1) when
  // peakLambda > 0, or NaN (0/0) when peakLambda = 0.  Guard the NaN case.
  // NaN peakLambda (e.g. timestamps contained NaN): treat as "no signal" → 0.
  const intensityScore = meanLambda > 0
    ? (Number.isNaN(peakLambda) ? 0 : sig(peakLambda / meanLambda))
    : peakLambda > 0 ? 1 : 0;
  const rateScore      = empiricalRate > 0
    ? (params.mu > 0 ? sig(empiricalRate / params.mu) : 1)
    : 0;

  return Math.max(intensityScore, rateScore);
}
