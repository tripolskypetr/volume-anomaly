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
/** Trade direction inferred from order-flow imbalance. */
type Direction = 'long' | 'short' | 'neutral';
interface PredictionResult {
    /** true when combined confidence ≥ requested threshold */
    anomaly: boolean;
    /** Composite anomaly score [0,1] */
    confidence: number;
    /**
     * Directional signal derived from imbalance:
     * - `'long'`    — anomaly + imbalance >  imbalanceThreshold (buy aggression)
     * - `'short'`   — anomaly + imbalance < −imbalanceThreshold (sell aggression)
     * - `'neutral'` — no anomaly, or anomaly with balanced order flow (rate-only spike)
     */
    direction: Direction;
    /** Signed imbalance [-1,+1]. Positive = buy-side pressure. */
    imbalance: number;
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
    /**
     * Raw sub-detector scores [0,1] regardless of signal thresholds.
     * confidence = scoreWeights · [hawkes, cusum, bocpd] (weights renormalized
     * when the window is too short for the rolling detectors to run).
     */
    scores: {
        hawkes: number;
        cusum: number;
        bocpd: number;
    };
    /**
     * Raw volume/rate statistics behind scores.hawkes:
     * robust z ("σ above recent typical") of the peak rolling arrival rate and
     * volume rate at the fast/slow horizons, plus the peak-λ ratio vs training.
     */
    stats: {
        zRate: number;
        zVol: number;
        zRateSlow: number;
        zVolSlow: number;
        lambdaRatio: number;
    };
    /** Per-detector signals that fired */
    signals: AnomalySignal[];
    /** Estimated imbalance [-1,+1]: positive = buy pressure */
    imbalance: number;
    /** Peak Hawkes conditional intensity λ(tᵢ) seen across all trades in the detection window */
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
 *   xₜ  = |imbalance(window)| — S⁺ catches pressure buildup (|imb| rising above
 *         baseline), S⁻ catches collapse toward balance (|imb| falling below
 *         baseline); both are regime changes, so the score uses max(S⁺, S⁻).
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
    /** CUSUM h alarm threshold in σ units (default 5σ, ARL₀ ≈ 148). */
    cusumHSigmas?: number;
    /**
     * Weights for combining sub-detector scores into a final confidence.
     * Must be 3 values [hawkes, cusum, bocpd] summing to 1.
     *
     * Default [1, 0, 0]: on real trade data the volume/rate channel (hawkes)
     * is the only one that discriminates volume anomalies; the imbalance-based
     * CUSUM/BOCPD channels track order-flow regime shifts (a different
     * phenomenon) and any additive weight on them measurably dilutes recall and
     * adds false alarms (see test/eval.test.ts).  Their scores are still
     * computed and reported in `scores`/`signals` — re-weight only if your use
     * case specifically targets flow-shift events.
     */
    scoreWeights?: [number, number, number];
    /**
     * Fast time horizon (seconds) for the rate / volume-rate statistic:
     * "trades per rateHorizonSec" and "qty per rateHorizonSec".  Catches brief
     * intense bursts.
     *
     * When omitted, chosen on the fly: max(5 s, 25 × median inter-trade gap),
     * capped by the training span — so sparse instruments automatically get a
     * horizon that contains enough trades for the statistic to mean anything.
     * Pass an explicit value to pin it.
     */
    rateHorizonSec?: number;
    /**
     * Slow time horizon (seconds) for the same statistic.  Catches sustained
     * volume waves that are spread too evenly to concentrate at the fast
     * horizon.  When omitted: 6 × the (auto) fast horizon, floored at 30 s and
     * capped by the training span.  Pass an explicit value to pin it.
     */
    slowHorizonSec?: number;
    /**
     * Percentile (0–100) of the training rolling signed imbalance distribution
     * used as the directional threshold inside predict().
     * p75 means: direction=long only when imbalance exceeds the 75th percentile
     * of the training imbalance series; direction=short when below the 25th.
     * Default 75.
     */
    imbalancePercentile?: number;
}
interface TrainedModels {
    hawkesParams: HawkesParams;
    cusumParams: CusumParams;
    bocpdPrior: NormalGammaPrior;
    /** p(imbalancePercentile) of the training rolling signed imbalance series */
    imbalanceThreshold: number;
    /**
     * Self-calibrated ceilings measured on the training window.
     *
     * Theory-driven thresholds ("2× fitted μ", "h = 5σ") misfire on real trade
     * streams: arrival rates fluctuate several-fold between adjacent windows and
     * the rolling |imbalance| series is heavily autocorrelated, so those levels
     * correspond to routine noise.  Instead each detector is calibrated against
     * what the (in-control) training window itself actually reached — an anomaly
     * must exceed the baseline's own extremes with margin.
     */
    /**
     * Robust location/scale of the rolling arrival rate (events/s) and rolling
     * volume rate (qty/s), per horizon.  detect() scores its peak rolling rate
     * as a robust z against these: z = (peak − med) / (1.4826 · MAD).
     * "fast" = brief bursts, "slow" = sustained volume waves spread too evenly
     * to concentrate at the fast horizon.
     */
    rateStats: {
        fast: RobustStats;
        slow: RobustStats;
    };
    volStats: {
        fast: RobustStats;
        slow: RobustStats;
    };
    /**
     * Effective horizons chosen at train() time (seconds).  Equal to the config
     * values when pinned explicitly; otherwise scaled up on the fly for sparse
     * streams so a horizon window contains enough trades to carry a rate.
     */
    fastHorizonSec: number;
    slowHorizonSec: number;
    /**
     * Per-channel score calibration from the null distribution of window maxima
     * over the training data (see ChannelCalib / calibrateChannel).
     */
    channelCalib: {
        rateFast: ChannelCalib;
        volFast: ChannelCalib;
        rateSlow: ChannelCalib;
        volSlow: ChannelCalib;
    };
    /** Peak λ(tᵢ) over the training window under the fitted Hawkes params */
    lambdaBaseline: number;
    /** Peak BOCPD anomaly score over the training series (noise floor) */
    bocpdNoiseFloor: number;
}
/** Robust location/scale: median + MAD. */
interface RobustStats {
    med: number;
    mad: number;
}
/**
 * Per-channel score mapping, self-calibrated from the null distribution of
 * window maxima on the training data (see calibrateChannel):
 *
 *   score(z) = σ((z − c − u·spanShift) · ln3 / u)
 *
 *   c = P90 of the maxima the baseline itself produced    → score 0.5
 *   c + u (one tail unit, u = P99 − P90 of those maxima)  → score 0.75
 *   each further u beyond that                            → 0.9, 0.96, …
 *
 * Both anchors are floored by the universal mapping validated on a full day
 * of real data (CALIB_FALLBACK_RATE / CALIB_FALLBACK_VOL): the baseline's own
 * null distribution can
 * only make the detector stricter (hot/noisy baseline ⇒ higher, wider
 * mapping), never more trigger-happy — a 15–30 min window yields too few
 * null stretches to trust a low estimate of the normal tail.
 */
interface ChannelCalib {
    /** Score-0.5 level: max(null P90 of window maxima, universal floor) */
    c: number;
    /** Universal tail unit; one u beyond c ⇒ 0.75 */
    u: number;
    /**
     * Quantile ladder of the null window-maxima distribution
     * [P50, P75, P80, P85, P90, P95, P97, P99] — exposed for introspection and
     * threshold research (empty when the fallback mapping is in effect).
     */
    nullQ: number[];
}
declare class VolumeAnomalyDetector {
    private readonly cfg;
    private models;
    /** User pinned the horizon explicitly — skip on-the-fly selection */
    private readonly explicitFast;
    private readonly explicitSlow;
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
     */
    detect(trades: IAggregatedTradeData[], confidence?: number): DetectionResult;
    /**
     * Arrival rate (events/s) and volume rate (qty/s) over a rolling TIME
     * window of rateHorizonSec, one sample per trade.  This is the shared
     * statistic for calibration: train() takes P99 of it as the baseline
     * ceiling, detect() takes its max as the detection statistic — same horizon
     * on both sides, so the ratio is apples-to-apples.
     *
     * The horizon must be a time span, not a trade count: a fixed-count window
     * ("last 50 trades") is rate-invariant by construction — during a burst it
     * simply shrinks in duration and its count/duration reflects matching-engine
     * micro-batching, not the burst.  A volume anomaly is "more trades / more
     * quantity per unit TIME", so the statistic has to be measured per unit time.
     *
     * Only full-horizon windows are scored: near the start of the array the
     * lookback would be truncated to data that isn't there, and dividing a
     * burst of leading trades by a floored duration reads as a phantom rate
     * spike that full-lookback training samples never contain.  When the whole
     * array spans less than the horizon, a single whole-window sample is
     * returned instead (duration floored at min(1 s, horizon) so that a clump
     * of same-millisecond trades still yields a finite, comparable rate).
     */
    private rollingRates;
    /**
     * Build the score mapping for one channel from the null distribution of its
     * window maxima on the training data.
     *
     * The training z-series is cut into sliding stretches of windowSec (the
     * slow-horizon alert timescale) at every offset multiple of windowSec/16;
     * the maximum z of each stretch is one null sample — "the worst this
     * baseline does in one alert window".  The fine step matters: coarse
     * stepping quantizes window edges and a peak sitting at a boundary between
     * coarse positions biases the null quantiles.  The mapping anchors on the
     * quantiles of those maxima; a noisy instrument gets a higher level, a
     * quiet one keeps the universal floor, and a baseline that itself contains
     * recurring bursts absorbs them into its null quantiles (a repeat of a
     * known pattern scores ≈ 0.5, not 1.0).
     *
     * Falls back to the fixed real-data calibration when the training span
     * yields fewer than 8 stretches.
     */
    private calibrateChannel;
    private rollingAbsImbalance;
    private rollingSignedImbalance;
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
/**
 * One-shot anomaly detection with directional signal.
 *
 * Wraps `detect()` and adds a `direction` field derived from `imbalance`:
 * - `'long'`    — anomaly detected + buy-side order flow dominates
 * - `'short'`   — anomaly detected + sell-side order flow dominates
 * - `'neutral'` — no anomaly, or anomaly is a pure rate spike with balanced flow
 *
 * The directional threshold is derived automatically from training data:
 * `imbalanceThreshold = p75 of the rolling signed imbalance series` (configurable
 * via `DetectorConfig.imbalancePercentile`). Pass an explicit number to override.
 *
 * @param historical          Baseline window (≥ 50 trades) for model training.
 * @param recent              Recent window to evaluate.
 * @param confidence          Anomaly threshold [0,1]. Default 0.75.
 * @param imbalanceThreshold  Override the trained threshold. Omit to use p75 from training.
 */
declare function predict(historical: IAggregatedTradeData[], recent: IAggregatedTradeData[], confidence?: number, imbalanceThreshold?: number): PredictionResult;

export { VolumeAnomalyDetector, detect, predict };
export type { AnomalyKind, AnomalySignal, DetectionResult, DetectorConfig, Direction, IAggregatedTradeData, PredictionResult };
