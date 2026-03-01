/**
 * CUSUM — Cumulative Sum Control Chart (Page, 1954).
 *
 * Detects a persistent shift of size δ in the mean of a series.
 * Classic two-sided formulation:
 *
 *   S⁺ₜ = max(0,  S⁺_{t-1} + xₜ − (μ₀ + k))
 *   S⁻ₜ = max(0,  S⁻_{t-1} − xₜ + (μ₀ − k))
 *
 * Alarm fires when Sₜ ≥ h.
 *
 * Applied to volume imbalance:
 *   xₜ  = |imbalance(window)| — we track absolute deviation, so one-sided S⁺ suffices.
 *   μ₀  = baseline mean imbalance magnitude (from training window)
 *   k   = allowable slack  = δ/2  (typically δ = 1 std-dev)
 *   h   = alarm threshold (tuned to ARL₀ — average run length under H₀)
 */

import type { CusumState } from '../types.js';

// ─── Training ─────────────────────────────────────────────────────────────────

export interface CusumParams {
  /** In-control mean */
  mu0:  number;
  /** In-control std-dev */
  std0: number;
  /** Allowable slack = δ/2  (default δ = 1σ → k = 0.5σ) */
  k:    number;
  /**
   * Alarm threshold h.
   * Rule of thumb: h ≈ 4–5 σ gives ARL₀ ≈ 500–1000.
   * Default: 4 std0.
   */
  h:    number;
}

/**
 * Estimate CUSUM parameters from a baseline (in-control) series of values.
 * values — e.g. array of |imbalance| from a calm training window.
 */
export function cusumFit(values: number[], kSigmas = 0.5, hSigmas = 4): CusumParams {
  if (values.length === 0) {
    return { mu0: 0, std0: 1, k: kSigmas, h: hSigmas };
  }
  const n    = values.length;
  const mu0  = values.reduce((s, x) => s + x, 0) / n;
  const var0 = values.reduce((s, x) => s + (x - mu0) ** 2, 0) / Math.max(n - 1, 1);
  const std0 = Math.sqrt(var0) || 1e-6;
  return {
    mu0,
    std0,
    k: kSigmas * std0,
    h: hSigmas * std0,
  };
}

// ─── Update ───────────────────────────────────────────────────────────────────

export interface CusumUpdateResult {
  state: CusumState;
  /** true when either S⁺ or S⁻ reached the alarm threshold h */
  alarm: boolean;
  /**
   * Accumulator values after applying the update formula but BEFORE the alarm
   * reset.  Useful for scoring: when alarm=true, state.sPos/sNeg are zeroed,
   * but preResetState reflects the actual peak reached.
   */
  preResetState: CusumState;
}

/**
 * Process one new observation, return updated state + alarm flag.
 * Pure function — does not mutate input.
 */
export function cusumUpdate(
  state: CusumState,
  x:     number,
  params: CusumParams,
): CusumUpdateResult {
  // Non-finite x (NaN, ±Infinity that doesn't trigger alarm) would poison the
  // accumulators via Math.max(0, NaN) = NaN.  Skip the update entirely for NaN;
  // ±Infinity is handled naturally (Inf ≥ h → alarm fires and resets state).
  if (Number.isNaN(x)) {
    return { alarm: false, preResetState: state, state };
  }
  const { mu0, k, h } = params;
  const sPos  = Math.max(0, state.sPos + (x - mu0) - k);
  const sNeg  = Math.max(0, state.sNeg - (x - mu0) - k);
  const alarm = sPos >= h || sNeg >= h;
  const preResetState: CusumState = { sPos, sNeg, n: state.n + 1 };
  return {
    alarm,
    preResetState,
    state: {
      sPos: alarm ? 0 : sPos,
      sNeg: alarm ? 0 : sNeg,
      n:    alarm ? 0 : state.n + 1,
    },
  };
}

export function cusumInitState(): CusumState {
  return { sPos: 0, sNeg: 0, n: 0 };
}

// ─── Score ────────────────────────────────────────────────────────────────────

/**
 * Normalised anomaly score [0,1] based on how close S⁺ is to alarm threshold h.
 */
export function cusumAnomalyScore(state: CusumState, params: CusumParams): number {
  const s = Math.max(state.sPos, state.sNeg);
  // NaN <= 0 is false in IEEE 754, so guard explicitly against non-finite h.
  if (params.h <= 0 || !Number.isFinite(params.h)) return 0;
  return Math.min(s / params.h, 1);
}

// ─── Batch update ─────────────────────────────────────────────────────────────

/**
 * Run CUSUM on a full series, returning final state + alarm timestamps.
 */
export function cusumBatch(
  series: number[],
  params: CusumParams,
): { state: CusumState; alarmIndices: number[] } {
  let state = cusumInitState();
  const alarmIndices: number[] = [];

  for (let i = 0; i < series.length; i++) {
    const result = cusumUpdate(state, series[i]!, params);
    if (result.alarm) alarmIndices.push(i);
    state = result.state;
  }

  return { state, alarmIndices };
}
