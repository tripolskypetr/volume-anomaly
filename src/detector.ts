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

import type { IAggregatedTradeData, DetectionResult, AnomalySignal } from './types.js';
import type { NormalGammaPrior }                 from './math/bocpd.js';
import type { HawkesParams }                     from './types.js';
import type { CusumState }                       from './types.js';

import type { CusumParams } from './math/cusum.js';

import { volumeImbalance, hawkesFit, hawkesPeakLambda } from './math/hawkes.js';
import { cusumFit, cusumUpdate, cusumInitState, cusumAnomalyScore }         from './math/cusum.js';
import { bocpdUpdate, bocpdInitState, bocpdAnomalyScore, defaultPrior }  from './math/bocpd.js';

// ─── Configuration ────────────────────────────────────────────────────────────

export interface DetectorConfig {
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
   * intense bursts.  Default 5 s.
   */
  rateHorizonSec?: number;
  /**
   * Slow time horizon (seconds) for the same statistic.  Catches sustained
   * volume waves that are spread too evenly to concentrate at the fast
   * horizon.  Default 30 s.
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

const DEFAULTS: Required<DetectorConfig> = {
  windowSize:          50,
  rateHorizonSec:      5,
  slowHorizonSec:      30,
  hazardLambda:        200,
  cusumKSigmas:        0.5,
  cusumHSigmas:        5,
  scoreWeights:        [1, 0, 0],
  imbalancePercentile: 75,
};

// ─── Trained model bundle ─────────────────────────────────────────────────────

interface TrainedModels {
  hawkesParams:        HawkesParams;
  cusumParams:         CusumParams;
  bocpdPrior:          NormalGammaPrior;
  /** p(imbalancePercentile) of the training rolling signed imbalance series */
  imbalanceThreshold:  number;
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
   * "fast" = rateHorizonSec (brief bursts), "slow" = slowHorizonSec (sustained
   * volume waves spread too evenly to concentrate at the fast horizon).
   */
  rateStats:           { fast: RobustStats; slow: RobustStats };
  volStats:            { fast: RobustStats; slow: RobustStats };
  /** Peak λ(tᵢ) over the training window under the fitted Hawkes params */
  lambdaBaseline:      number;
  /** Peak BOCPD anomaly score over the training series (noise floor) */
  bocpdNoiseFloor:     number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Linear-interpolation quantile (standard type 7). p in [0, 100]. */
function quantile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx    = (p / 100) * (sorted.length - 1);
  const lo     = Math.floor(idx);
  const hi     = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (idx - lo);
}

/** Robust location/scale: median + MAD. */
interface RobustStats {
  med: number;
  mad: number;
}

function robustStats(xs: number[]): RobustStats {
  if (xs.length === 0) return { med: 0, mad: 0 };
  const med = quantile(xs, 50);
  const mad = quantile(xs.map((x) => Math.abs(x - med)), 50);
  return { med, mad };
}

/**
 * Robust z-score of x against training stats.
 * MAD is floored at 10% of the median: rate distributions are heavy-tailed
 * and a near-zero raw MAD would turn routine wiggles into huge z values.
 *
 * Deliberately NOT tail-aware (no P99-based scale term): widening the scale
 * by the baseline's own upper tail would silence recurring-pattern repeats,
 * but it equally silences genuine escalating events whose baseline is already
 * hot (the two are indistinguishable to any local statistic — verified on the
 * real-data benchmark, where a tail-aware scale crushed the day's largest
 * confirmed spikes).  Consequence: a burst that recurs inside the training
 * window IS re-flagged; if your instrument has a known periodic pattern,
 * lengthen the baseline or de-duplicate alerts downstream.
 */
function robustZ(x: number, s: RobustStats): number {
  const scale = 1.4826 * Math.max(s.mad, 0.1 * Math.abs(s.med), 1e-12);
  return (x - s.med) / scale;
}

// ─── Detector class ───────────────────────────────────────────────────────────

export class VolumeAnomalyDetector {
  private readonly cfg: Required<DetectorConfig>;
  private models:       TrainedModels | null = null;

  constructor(config: DetectorConfig = {}) {
    this.cfg = { ...DEFAULTS, ...config };
    if (config.scoreWeights) {
      if (!config.scoreWeights.every(Number.isFinite)) {
        throw new Error(`scoreWeights must be finite numbers, got ${config.scoreWeights}`);
      }
      if (config.scoreWeights.some((w) => w < 0)) {
        throw new Error(`scoreWeights must be non-negative, got ${config.scoreWeights}`);
      }
      const sum = config.scoreWeights.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1) > 1e-6) {
        throw new Error(`scoreWeights must sum to 1, got ${sum}`);
      }
    }
  }

  // ─── Training ───────────────────────────────────────────────────────────────

  /**
   * Fit all models to historical (in-control) trade data.
   * Must be called before detect().
   */
  train(trades: IAggregatedTradeData[]): void {
    if (trades.length < 50) {
      throw new Error(`Need at least 50 trades for training, got ${trades.length}`);
    }

    // Sort by time
    const sorted = [...trades].sort((a, b) => a.timestamp - b.timestamp);

    // ── Self-calibrated rate baselines: robust median/MAD of the rolling
    // arrival rate (events/s) and rolling volume rate (qty/s) at both time
    // horizons, taken over the FULL historical span.  detect() scores its
    // peak rolling rates as robust z against these — "how many robust σ above
    // the recent typical level", the standard operational definition of a
    // volume anomaly.
    //
    // The span matters more than the trade count: a baseline of "the last N
    // trades" adapts its duration to market pace, so right before a burst it
    // covers seconds of already-hot market and the baseline inflates to mask
    // the very anomaly being detected, while in calm periods it covers a few
    // narrow minutes and routine micro-clusters look anomalous.  Feed train()
    // a fixed TIME span (≥ 15–30 min of normal market) so the baselines are
    // stable regardless of pace.
    const allTimestamps = sorted.map((t) => t.timestamp / 1000);
    const fast = this.rollingRates(sorted, allTimestamps, this.cfg.rateHorizonSec);
    const slow = this.rollingRates(sorted, allTimestamps, this.cfg.slowHorizonSec);
    const rateStats = { fast: robustStats(fast.rates),    slow: robustStats(slow.rates) };
    const volStats  = { fast: robustStats(fast.volRates), slow: robustStats(slow.volRates) };

    // ── Hawkes: fit to trade arrival times (in seconds).
    // MLE cost grows with n and the fit only needs recent arrival structure —
    // cap to the most recent trades while the cheap O(n) ceilings above use
    // the whole span.
    const hawkesSlice = allTimestamps.slice(-2000);
    const { params: hawkesParams } = hawkesFit(hawkesSlice);

    // ── Self-calibrated intensity ceiling: peak λ the baseline itself reached.
    const lambdaBaseline = hawkesPeakLambda(hawkesSlice, hawkesParams);

    // ── CUSUM + BOCPD: fit to rolling |imbalance| series from training data.
    // Both detectors operate on absolute imbalance so that buy-side and
    // sell-side pressure are treated symmetrically.
    // One rolling pass: |imbalance| is derived from the signed series that the
    // directional threshold below needs anyway.
    // Capped to recent trades: the stride-1 rolling pass is O(n·windowSize)
    // and flow-regime statistics only need recent structure.
    const imbSlice        = sorted.slice(-1000);
    const signedImbalance = this.rollingSignedImbalance(imbSlice);
    const absImbalance    = signedImbalance.map(Math.abs);
    const cusumFitted     = cusumFit(absImbalance, this.cfg.cusumKSigmas, this.cfg.cusumHSigmas);

    // ── CUSUM: empirical alarm threshold.
    // The theoretical "h = 5σ ⇒ ARL₀ ≈ …" calibration assumes independent
    // observations; the stride-1 rolling series is ~98% autocorrelated, so the
    // accumulator legitimately wanders far past 5σ on perfectly normal data.
    // Probe the training series itself (alarms disabled) and require detection
    // excursions to double the maximum the baseline ever reached.
    const probeParams = { ...cusumFitted, h: Infinity };
    let probeState    = cusumInitState();
    let maxExcursion  = 0;
    for (const v of absImbalance) {
      probeState = cusumUpdate(probeState, v, probeParams).state;
      const m = Math.max(probeState.sPos, probeState.sNeg);
      if (m > maxExcursion) maxExcursion = m;
    }
    const cusumParams: CusumParams = {
      ...cusumFitted,
      h: Math.max(cusumFitted.h, 2 * maxExcursion),
    };

    // ── BOCPD: noise floor.
    // On an autocorrelated real-market series the run length collapses
    // routinely, so the raw drop score is nonzero even in-control.  Record the
    // worst score the training series produces; detect() rescales its scores
    // to count only what exceeds this floor.
    const n    = absImbalance.length;
    const mean = n > 0 ? absImbalance.reduce((s, x) => s + x, 0) / n : 0;
    const vari = n > 0 ? absImbalance.reduce((s, x) => s + (x - mean) ** 2, 0) / n : 0;
    const bocpdPrior = defaultPrior(mean, vari);

    let floorResult     = { mapRunLength: 0, cpProbability: 0, state: bocpdInitState() };
    let bocpdNoiseFloor = 0;
    for (const v of absImbalance) {
      const prevRL = floorResult.mapRunLength;
      floorResult  = bocpdUpdate(floorResult.state, v, bocpdPrior, this.cfg.hazardLambda);
      const s      = bocpdAnomalyScore(floorResult, prevRL);
      if (s > bocpdNoiseFloor) bocpdNoiseFloor = s;
    }

    // ── Directional threshold: p(imbalancePercentile) of rolling signed imbalance.
    // Uses signed (not absolute) series so trending markets produce an elevated
    // threshold that reflects actual baseline buy/sell bias.
    const imbalanceThreshold = signedImbalance.length > 0
      ? quantile(signedImbalance, this.cfg.imbalancePercentile)
      : 0.3;

    this.models = {
      hawkesParams,
      cusumParams,
      bocpdPrior,
      imbalanceThreshold,
      rateStats,
      volStats,
      lambdaBaseline,
      bocpdNoiseFloor,
    };
  }

  // ─── Detection ──────────────────────────────────────────────────────────────

  /**
   * Detect volume anomaly in a recent trade window.
   *
   * @param trades     Recent trades (e.g. last 200–500 trades).
   * @param confidence Required confidence threshold [0,1]. Default 0.75.
   */
  detect(
    trades:     IAggregatedTradeData[],
    confidence: number = 0.75,
  ): DetectionResult {
    if (!this.models) {
      throw new Error('Call train() before detect()');
    }
    if (trades.length === 0) {
      return this.emptyResult();
    }

    const sorted = [...trades].sort((a, b) => a.timestamp - b.timestamp);
    const {
      hawkesParams, cusumParams, bocpdPrior,
      rateStats, volStats, lambdaBaseline, bocpdNoiseFloor,
    } = this.models;
    const [wH, wC, wB] = this.cfg.scoreWeights;

    // ── 1. Volume/rate channel.
    // Peak rolling arrival rate (events/s) and volume rate (qty/s) at both
    // time horizons, scored as robust z against the training median/MAD:
    // "how many robust σ above the recent typical level".  The volume-rate
    // channels catch "few huge trades" anomalies invisible to arrival-time
    // statistics; the slow horizon catches sustained waves spread too evenly
    // to concentrate at the fast horizon.  A single trade yields no
    // meaningful rate → channels stay off.
    //
    // Sigmoid centred at z = 12 with slope 0.4 — calibrated on a full day of
    // real BTCUSDT aggTrades (see test/eval.test.ts): peak-over-window robust
    // z of normal windows sits at ≈ 6–7 (P95) while genuine anomalies reach
    // z ≈ 14–150.  This mapping gives z=3 → 0.03, z=12 → 0.5, z≈14.7 → 0.75
    // (the default confidence), z=20 → 0.96.
    // Measured at confidence 0.75: ≈90% of locally-strong (z≥8 vs trailing
    // hour) buckets and ≈94% of anomaly events detected at ≈2.5% false-alarm
    // rate on normal buckets.
    const timestamps = sorted.map((t) => t.timestamp / 1000);
    const lambda     = hawkesPeakLambda(timestamps, hawkesParams);
    const zSig = (z: number) => 1 / (1 + Math.exp(-(z - 12) * 0.4));

    let zRate = 0, zVol = 0, zRateSlow = 0, zVolSlow = 0;
    if (timestamps.length >= 2) {
      const fast = this.rollingRates(sorted, timestamps, this.cfg.rateHorizonSec);
      const slow = this.rollingRates(sorted, timestamps, this.cfg.slowHorizonSec);
      zRate     = robustZ(Math.max(...fast.rates),    rateStats.fast);
      zVol      = robustZ(Math.max(...fast.volRates), volStats.fast);
      zRateSlow = robustZ(Math.max(...slow.rates),    rateStats.slow);
      zVolSlow  = robustZ(Math.max(...slow.volRates), volStats.slow);
    }
    // λ ratio (peak Hawkes intensity vs the training peak) is reported in
    // stats/meta for transparency but deliberately kept OUT of the score: it
    // correlates with the rate z-channels and only added false-positive tail
    // in the real-data evaluation.
    const lambdaRatio = lambdaBaseline > 0 && Number.isFinite(lambda) ? lambda / lambdaBaseline : 0;
    const hawkesScore = Math.max(zSig(zRate), zSig(zVol), zSig(zRateSlow), zSig(zVolSlow));

    // ── 2. Current imbalance (full window, signed — for direction reporting)
    const imbalance = volumeImbalance(sorted);
    const absImb    = Math.abs(imbalance);

    // ── 3. CUSUM on |imbalance| rolling series.
    // Track the peak S/h ratio seen during the run, including just before any
    // alarm reset.  This captures evidence even when the alarm fires mid-window
    // and the accumulator resets to zero before the last observation.
    let cusumState: CusumState = cusumInitState();
    const absImbSeries         = this.rollingAbsImbalance(sorted);
    let peakCusumScore         = 0;
    let peakCusumState: CusumState = cusumState;
    for (const v of absImbSeries) {
      const upd = cusumUpdate(cusumState, v, cusumParams);
      // Score against preResetState so alarm events are not lost:
      // when alarm=true, upd.state is zeroed but preResetState holds the peak.
      const scoreNow = cusumAnomalyScore(upd.preResetState, cusumParams);
      if (scoreNow > peakCusumScore) {
        peakCusumScore = scoreNow;
        peakCusumState = upd.preResetState;
      }
      cusumState = upd.state;
    }
    const cusumScore = peakCusumScore;

    // ── 4. BOCPD on |imbalance| rolling series — same space as training prior.
    // cpProbability is always ≈ H = 1/hazardLambda (a prior constant) and does
    // not spike at a genuine changepoint.  The real signal is mapRunLength: in
    // a stable process it grows monotonically; a changepoint resets it to ≈ 0.
    // bocpdAnomalyScore measures the relative drop from the previous step, so
    // a reset from 90 → 1 scores ≈ 1 while gradual growth scores near 0.
    // We take the peak score over the window to catch changepoints that
    // happened before the last observation.
    let bocpdResult  = { mapRunLength: 0, cpProbability: 0, state: bocpdInitState() };
    let rawBocpdPeak = 0;
    for (const v of absImbSeries) {
      const prevRL = bocpdResult.mapRunLength;
      bocpdResult  = bocpdUpdate(bocpdResult.state, v, bocpdPrior, this.cfg.hazardLambda);
      const s      = bocpdAnomalyScore(bocpdResult, prevRL);
      if (s > rawBocpdPeak) rawBocpdPeak = s;
    }
    // Rescale against the training noise floor: on an autocorrelated real
    // series run-length collapses happen in-control, so only the score mass
    // above what the baseline itself produced counts as evidence.
    const bocpdScore = bocpdNoiseFloor >= 1
      ? 0
      : Math.max(0, (rawBocpdPeak - bocpdNoiseFloor) / (1 - bocpdNoiseFloor));

    // ── 5. Combine scores.
    // When the window is shorter than windowSize the rolling series is empty:
    // CUSUM and BOCPD never ran, so their weights would silently cap the
    // combined score at wH (0.4 by default) — an anomaly could never fire.
    // Renormalize over the detectors that actually saw data (Hawkes only).
    const combined = absImbSeries.length > 0
      ? wH! * hawkesScore + wC! * cusumScore + wB! * bocpdScore
      : (wH! > 0 ? hawkesScore : 0);

    // ── 6. Build signals list
    const signals: AnomalySignal[] = [];

    if (hawkesScore > 0.5) {
      signals.push({
        kind:  'volume_spike',
        score: hawkesScore,
        meta:  { lambda, zRate, zVol, zRateSlow, zVolSlow, lambdaRatio },
      });
    }
    if (absImb > 0.4) {
      signals.push({
        kind:  'imbalance_shift',
        score: absImb,
        meta:  { imbalance, absImbalance: absImb },
      });
    }
    if (cusumScore > 0.7) {
      signals.push({
        kind:  'cusum_alarm',
        score: cusumScore,
        // Peak pre-reset accumulators, not the final state: when the alarm
        // fired mid-window the final state has already been zeroed and would
        // report sPos = sNeg = 0 for the very event being signalled.
        meta:  { sPos: peakCusumState.sPos, sNeg: peakCusumState.sNeg, h: cusumParams.h },
      });
    }
    if (bocpdScore > 0.3) {
      signals.push({
        kind:  'bocpd_changepoint',
        score: bocpdScore,
        meta:  { cpProbability: bocpdResult.cpProbability, runLength: bocpdResult.mapRunLength },
      });
    }

    return {
      anomaly:      combined >= confidence,
      confidence:   combined,
      scores:       { hawkes: hawkesScore, cusum: cusumScore, bocpd: bocpdScore },
      stats:        { zRate, zVol, zRateSlow, zVolSlow, lambdaRatio },
      signals,
      imbalance,
      hawkesLambda: lambda,
      cusumStat:    Math.max(cusumState.sPos, cusumState.sNeg),
      runLength:    bocpdResult.mapRunLength,
    };
  }

  // ─── Rolling rate helper ────────────────────────────────────────────────────

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
  private rollingRates(
    sorted:     IAggregatedTradeData[],
    timestamps: number[],
    horizon:    number,
  ): { rates: number[]; volRates: number[] } {
    const n = sorted.length;
    const qtyPrefix = new Array<number>(n + 1).fill(0);
    for (let i = 0; i < n; i++) {
      qtyPrefix[i + 1] = qtyPrefix[i]! + sorted[i]!.qty;
    }
    const rates:    number[] = [];
    const volRates: number[] = [];
    const t0   = timestamps[0]!;
    const span = timestamps[n - 1]! - t0;
    if (span >= horizon) {
      let j = 0; // two-pointer: first trade inside the window (tᵢ − horizon, tᵢ]
      for (let i = 0; i < n; i++) {
        if (timestamps[i]! - t0 < horizon) continue; // truncated lookback
        while (timestamps[j]! <= timestamps[i]! - horizon) j++;
        rates.push((i - j + 1) / horizon);
        volRates.push((qtyPrefix[i + 1]! - qtyPrefix[j]!) / horizon);
      }
    }
    if (rates.length === 0 && n >= 2) {
      const dur = Math.max(span, Math.min(1, horizon));
      rates.push(n / dur);
      volRates.push(qtyPrefix[n]! / dur);
    }
    return { rates, volRates };
  }

  // ─── Rolling |imbalance| helper ───────────────────────────────────────────

  private rollingAbsImbalance(sorted: IAggregatedTradeData[]): number[] {
    const w   = this.cfg.windowSize;
    const out: number[] = [];
    for (let i = w; i <= sorted.length; i++) {
      out.push(Math.abs(volumeImbalance(sorted.slice(i - w, i))));
    }
    return out;
  }

  private rollingSignedImbalance(sorted: IAggregatedTradeData[]): number[] {
    const w   = this.cfg.windowSize;
    const out: number[] = [];
    for (let i = w; i <= sorted.length; i++) {
      out.push(volumeImbalance(sorted.slice(i - w, i)));
    }
    return out;
  }

  private emptyResult(): DetectionResult {
    return {
      anomaly:      false,
      confidence:   0,
      scores:       { hawkes: 0, cusum: 0, bocpd: 0 },
      stats:        { zRate: 0, zVol: 0, zRateSlow: 0, zVolSlow: 0, lambdaRatio: 0 },
      signals:      [],
      imbalance:    0,
      hawkesLambda: 0,
      cusumStat:    0,
      runLength:    0,
    };
  }

  // ─── Introspection ────────────────────────────────────────────────────────

  get isTrained(): boolean {
    return this.models !== null;
  }

  /** Expose fitted parameters (for debugging / serialization) */
  get trainedModels(): Readonly<TrainedModels> | null {
    return this.models;
  }
}
