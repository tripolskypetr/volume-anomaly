// ─── Input ───────────────────────────────────────────────────────────────────

export interface IAggregatedTradeData {
  /** Binance aggTradeId */
  id:           string;
  /** Execution price */
  price:        number;
  /** Trade size (base asset) */
  qty:          number;
  /** Unix timestamp in milliseconds */
  timestamp:    number;
  /** true  → buyer is maker (sell aggressor)
   *  false → buyer is taker (buy aggressor)  */
  isBuyerMaker: boolean;
}

// ─── Prediction result ───────────────────────────────────────────────────────

/** Trade direction inferred from order-flow imbalance. */
export type Direction = 'long' | 'short' | 'neutral';

export interface PredictionResult {
  /** true when combined confidence ≥ requested threshold */
  anomaly:    boolean;
  /** Composite anomaly score [0,1] */
  confidence: number;
  /**
   * Directional signal derived from imbalance:
   * - `'long'`    — anomaly + imbalance >  imbalanceThreshold (buy aggression);
   *                 always implies imbalance > 0
   * - `'short'`   — anomaly + imbalance < −imbalanceThreshold (sell aggression);
   *                 always implies imbalance < 0
   * - `'neutral'` — no anomaly, or anomaly with balanced order flow (rate-only spike)
   *
   * The threshold (trained p75 of rolling signed imbalance, or the explicit
   * override) is clamped at zero before the symmetric ± comparison.
   */
  direction:  Direction;
  /** Signed imbalance [-1,+1] over the full window. Positive = buy-side pressure. */
  imbalance:  number;
  /**
   * Imbalance inside the peak burst window (see DetectionResult.burstImbalance).
   * `direction` is derived from THIS value — the full-window `imbalance`
   * dilutes a burst's onset direction with surrounding two-way flow.
   */
  burstImbalance: number;
  /** Predictive ranking score for forward price response (see DetectionResult.moveScore). */
  moveScore:      number;
}

// ─── Detection result ─────────────────────────────────────────────────────────

export type AnomalyKind =
  | 'volume_spike'          // Hawkes λ surge
  | 'imbalance_shift'       // |imbalance| crossed threshold
  | 'cusum_alarm'           // CUSUM h-boundary hit
  | 'bocpd_changepoint';    // BOCPD run-length reset

export interface AnomalySignal {
  kind:        AnomalyKind;
  /** Normalised [0,1] strength of evidence for this sub-detector */
  score:       number;
  /** Detector-specific metadata */
  meta:        Record<string, number>;
}

export interface DetectionResult {
  /** true when combined confidence ≥ requested threshold */
  anomaly:        boolean;
  /** Probability [0,1] that the current window contains an anomaly */
  confidence:     number;
  /**
   * Raw sub-detector scores [0,1] regardless of signal thresholds.
   * confidence = scoreWeights · [hawkes, cusum, bocpd] (weights renormalized
   * when the window is too short for the rolling detectors to run).
   */
  scores:         { hawkes: number; cusum: number; bocpd: number };
  /**
   * Raw volume/rate statistics behind scores.hawkes:
   * robust z ("σ above recent typical") of the peak rolling arrival rate and
   * volume rate at the fast/slow horizons, plus the peak-λ ratio vs training.
   */
  stats:          {
    zRate:       number;
    zVol:        number;
    zRateSlow:   number;
    zVolSlow:    number;
    lambdaRatio: number;
    /**
     * Per-scale peak robust z, one entry per trained horizon
     * (TrainedModels.horizonsSec, ascending).  zRate/zVol/zRateSlow/zVolSlow
     * above are the fastest/slow entries of these, kept for compatibility.
     */
    zRates:      number[];
    zVols:       number[];
  };
  /** Per-detector signals that fired */
  signals:        AnomalySignal[];
  /** Estimated imbalance [-1,+1]: positive = buy pressure */
  imbalance:      number;
  /**
   * Order-flow imbalance [-1,+1] measured INSIDE the peak burst window (the
   * rolling window that produced the winning volume/rate channel), not over
   * the whole detection window — a burst's onset direction gets diluted by
   * surrounding two-way flow.  Shrunk toward the training buy/sell balance by
   * effective sample size (Kish n_eff over qty), so a few-trade or one-whale
   * window carries little directional weight.  Falls back to the shrunk
   * full-window imbalance when no channel produced a trade-aligned peak.
   * predict() derives `direction` from this field.
   */
  burstImbalance: number;
  /**
   * Predictive ranking score [0,1) for FORWARD price response — how strongly
   * this window's volume statistic has historically preceded near-term price
   * movement.  Built from the peak long-scale (slow horizon and above)
   * VOLUME z through a fixed universal mapping: no per-baseline adaptation,
   * so values are comparable across windows and across time.  Use it to RANK
   * alerts (position sizing, prioritization), not as a detection threshold —
   * that is `confidence`'s job.  Measured on the full-day benchmark: ranks
   * forward 1-min range at AUC ≈ 0.64 vs ≈ 0.60 for confidence.
   */
  moveScore:      number;
  /**
   * Timestamp (ms) of the last trade of the peak burst window — when the
   * anomaly actually peaked inside the detection window.  Last trade of the
   * window when no channel produced a trade-aligned peak; 0 for an empty
   * window.
   */
  peakTs:         number;
  /** Peak Hawkes conditional intensity λ(tᵢ) seen across all trades in the detection window */
  hawkesLambda:   number;
  /** CUSUM statistic (+ side) at last observation */
  cusumStat:      number;
  /** BOCPD: most probable run length (periods since last changepoint) */
  runLength:      number;
}

// ─── Internal model state (exposed via math exports for testing) ──────────────

export interface HawkesParams {
  /** Background rate  μ > 0 */
  mu:    number;
  /** Excitation factor  0 < α < β */
  alpha: number;
  /** Decay rate  β > 0 */
  beta:  number;
}

export interface CusumState {
  /** Positive CUSUM accumulator */
  sPos: number;
  /** Negative CUSUM accumulator */
  sNeg: number;
  /** Observations since last reset */
  n:    number;
}

/** Normal-Gamma conjugate sufficient statistics for one BOCPD segment */
export interface NormalGammaSS {
  /** Number of observations in segment */
  n:      number;
  /** Running mean */
  mean:   number;
  /** Sum of squared deviations (M2 from Welford) */
  m2:     number;
}
