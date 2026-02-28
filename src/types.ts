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
  /** Per-detector signals that fired */
  signals:        AnomalySignal[];
  /** Estimated imbalance [-1,+1]: positive = buy pressure */
  imbalance:      number;
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
