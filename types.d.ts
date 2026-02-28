interface IAggregatedTradeData {
    /** Binance aggTradeId */
    id: string;
    /** Execution price */
    price: number;
    /** Trade size (base asset) */
    qty: number;
    /** Unix timestamp in milliseconds */
    timestamp: number;
    /** true  → buyer is maker (sell aggressor)
     *  false → buyer is taker (buy aggressor)  */
    isBuyerMaker: boolean;
}
type AnomalyKind = 'volume_spike' | 'imbalance_shift' | 'cusum_alarm' | 'bocpd_changepoint';
interface AnomalySignal {
    kind: AnomalyKind;
    /** Normalised [0,1] strength of evidence for this sub-detector */
    score: number;
    /** Detector-specific metadata */
    meta: Record<string, number>;
}
interface DetectionResult {
    /** true when combined confidence ≥ requested threshold */
    anomaly: boolean;
    /** Probability [0,1] that the current window contains an anomaly */
    confidence: number;
    /** Per-detector signals that fired */
    signals: AnomalySignal[];
    /** Estimated imbalance [-1,+1]: positive = buy pressure */
    imbalance: number;
    /** Hawkes conditional intensity at last observed trade */
    hawkesLambda: number;
    /** CUSUM statistic (+ side) at last observation */
    cusumStat: number;
    /** BOCPD: most probable run length (periods since last changepoint) */
    runLength: number;
}
interface HawkesParams {
    /** Background rate  μ > 0 */
    mu: number;
    /** Excitation factor  0 < α < β */
    alpha: number;
    /** Decay rate  β > 0 */
    beta: number;
}

/**
 * Bayesian Online Changepoint Detection
 * Adams & MacKay, 2007  (https://arxiv.org/abs/0710.3742)
 *
 * Run-length posterior:
 *   P(rₜ | x₁:ₜ) ∝ Σ_{rₜ₋₁} P(xₜ | r_{t-1}, x_{t-r:t}) · P(rₜ | rₜ₋₁) · P(rₜ₋₁ | x₁:ₜ₋₁)
 *
 * Underlying model: Gaussian observations with Normal-Gamma conjugate prior.
 * Hazard function H(r) = 1/λ  (geometric / memoryless changepoint gaps, λ = expected gap).
 *
 * Each run-length hypothesis maintains sufficient statistics (Welford online mean + M2).
 */

interface NormalGammaPrior {
    /** Prior mean */
    mu0: number;
    /** Prior pseudo-observations (strength of mean belief) */
    kappa0: number;
    /** Prior shape (α₀, must be > 0) */
    alpha0: number;
    /** Prior rate  (β₀, must be > 0) */
    beta0: number;
}

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

interface CusumParams {
    /** In-control mean */
    mu0: number;
    /** In-control std-dev */
    std0: number;
    /** Allowable slack = δ/2  (default δ = 1σ → k = 0.5σ) */
    k: number;
    /**
     * Alarm threshold h.
     * Rule of thumb: h ≈ 4–5 σ gives ARL₀ ≈ 500–1000.
     * Default: 4 std0.
     */
    h: number;
}

/**
 * VolumeAnomalyDetector
 *
 * Wraps Hawkes + CUSUM + BOCPD into a single object.
 * Workflow:
 *   1. detector.train(historicalTrades)   — fits all models
 *   2. detector.detect(recentTrades, confidence)  — returns DetectionResult
 *
 * All math is screened behind this interface.
 * For unit testing, import individual functions from '#math'.
 */

interface DetectorConfig {
    /**
     * Window size (number of trades) for computing per-step imbalance.
     * Smaller = more reactive, larger = smoother signal.
     */
    windowSize?: number;
    /** Expected gap between changepoints (for BOCPD hazard rate). */
    hazardLambda?: number;
    /** CUSUM k multiplier in σ units (default 0.5σ). */
    cusumKSigmas?: number;
    /** CUSUM h alarm threshold in σ units (default 4σ). */
    cusumHSigmas?: number;
    /**
     * Weights for combining sub-detector scores into a final confidence.
     * Must be 3 values [hawkes, cusum, bocpd] summing to 1.
     */
    scoreWeights?: [number, number, number];
}
interface TrainedModels {
    hawkesParams: HawkesParams;
    cusumParams: CusumParams;
    bocpdPrior: NormalGammaPrior;
}
declare class VolumeAnomalyDetector {
    private readonly cfg;
    private models;
    private cusumState;
    private bocpdState;
    constructor(config?: DetectorConfig);
    /**
     * Fit all models to historical (in-control) trade data.
     * Must be called before detect().
     */
    train(trades: IAggregatedTradeData[]): void;
    /**
     * Detect volume anomaly in a recent trade window.
     *
     * @param trades     Recent trades (e.g. last 200–500 trades).
     * @param confidence Required confidence threshold [0,1]. Default 0.75.
     *
     * This is a **snapshot** call — does not carry state between calls.
     * For streaming (sequential) use, call detectNext() per-trade instead.
     */
    detect(trades: IAggregatedTradeData[], confidence?: number): DetectionResult;
    private rollingImbalance;
    private emptyResult;
    get isTrained(): boolean;
    /** Expose fitted parameters (for debugging / serialization) */
    get trainedModels(): Readonly<TrainedModels> | null;
}

/**
 * volume-anomaly — public API
 *
 * @example
 * ```typescript
 * import { detect, VolumeAnomalyDetector } from 'volume-anomaly';
 *
 * // One-shot (convenience wrapper, no state):
 * const result = detect(historicalTrades, recentTrades, 0.75);
 * if (result.anomaly) {
 *   console.log('Entry signal!', result.imbalance, result.confidence);
 * }
 *
 * // Stateful (recommended for production):
 * const detector = new VolumeAnomalyDetector({ windowSize: 50 });
 * detector.train(historicalTrades);
 * const r = detector.detect(recentTrades, 0.75);
 * ```
 */

/**
 * Convenience function: train + detect in one call.
 *
 * @param historical  Long baseline window (≥ 50 trades) — used for model training.
 * @param recent      Short recent window — evaluated for anomalies.
 * @param confidence  Required confidence to flag anomaly [0,1]. Default 0.75.
 */
declare function detect(historical: IAggregatedTradeData[], recent: IAggregatedTradeData[], confidence?: number): DetectionResult;

export { VolumeAnomalyDetector, detect };
export type { AnomalyKind, AnomalySignal, DetectionResult, DetectorConfig, IAggregatedTradeData };
