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

import { volumeImbalance, hawkesFit, hawkesPeakLambda, hawkesAnomalyScore } from './math/hawkes.js';
import { cusumFit, cusumUpdate, cusumInitState, cusumAnomalyScore, CusumParams }      from './math/cusum.js';
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
   */
  scoreWeights?: [number, number, number];
}

const DEFAULTS: Required<DetectorConfig> = {
  windowSize:   50,
  hazardLambda: 200,
  cusumKSigmas: 0.5,
  cusumHSigmas: 5,
  scoreWeights: [0.4, 0.3, 0.3],
};

// ─── Trained model bundle ─────────────────────────────────────────────────────

interface TrainedModels {
  hawkesParams:  HawkesParams;
  cusumParams:   CusumParams;
  bocpdPrior:    NormalGammaPrior;
}

// ─── Detector class ───────────────────────────────────────────────────────────

export class VolumeAnomalyDetector {
  private readonly cfg: Required<DetectorConfig>;
  private models:       TrainedModels | null = null;

  constructor(config: DetectorConfig = {}) {
    this.cfg = { ...DEFAULTS, ...config };
    if (config.scoreWeights) {
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

    // ── Hawkes: fit to trade arrival times (in seconds)
    const timestamps = sorted.map((t) => t.timestamp / 1000);
    const { params: hawkesParams } = hawkesFit(timestamps);

    // ── CUSUM + BOCPD: fit to rolling |imbalance| series from training data.
    // Both detectors operate on absolute imbalance so that buy-side and
    // sell-side pressure are treated symmetrically.
    const absImbalance = this.rollingAbsImbalance(sorted);
    const cusumParams  = cusumFit(absImbalance, this.cfg.cusumKSigmas, this.cfg.cusumHSigmas);

    const mean = absImbalance.reduce((s, x) => s + x, 0) / absImbalance.length;
    const vari = absImbalance.reduce((s, x) => s + (x - mean) ** 2, 0) / absImbalance.length;
    const bocpdPrior = defaultPrior(mean, vari);

    this.models = { hawkesParams, cusumParams, bocpdPrior };
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
    const { hawkesParams, cusumParams, bocpdPrior } = this.models;
    const [wH, wC, wB] = this.cfg.scoreWeights;

    // ── 1. Peak Hawkes intensity over the detection window.
    // hawkesPeakLambda captures the maximum λ(tᵢ) seen at any event in the
    // window (using the O(n) recursive A-trick), so a burst that decayed by
    // the last event is still detected.
    // empiricalRate provides a model-agnostic fallback: even when MLE assigns
    // alpha ≈ 0 (Poisson baseline), a 1000× arrival surge is clearly anomalous
    // and the rate ratio fires where the intensity ratio would not.
    const timestamps    = sorted.map((t) => t.timestamp / 1000);
    const windowSec     = (timestamps[timestamps.length - 1]! - timestamps[0]!) || 1;
    const empiricalRate = timestamps.length / windowSec;
    const lambda        = hawkesPeakLambda(timestamps, hawkesParams);
    const hawkesScore   = hawkesAnomalyScore(lambda, hawkesParams, empiricalRate);

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
    for (const v of absImbSeries) {
      const upd = cusumUpdate(cusumState, v, cusumParams);
      // Score before applying reset so alarm events are not lost
      const scoreNow = cusumAnomalyScore(
        upd.alarm ? { sPos: Math.max(cusumState.sPos + (v - cusumParams.mu0) - cusumParams.k, 0),
                      sNeg: Math.max(cusumState.sNeg - (v - cusumParams.mu0) - cusumParams.k, 0),
                      n: cusumState.n } : upd.state,
        cusumParams,
      );
      if (scoreNow > peakCusumScore) peakCusumScore = scoreNow;
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
    let bocpdScore   = 0;
    for (const v of absImbSeries) {
      const prevRL = bocpdResult.mapRunLength;
      bocpdResult  = bocpdUpdate(bocpdResult.state, v, bocpdPrior, this.cfg.hazardLambda);
      const s      = bocpdAnomalyScore(bocpdResult, prevRL);
      if (s > bocpdScore) bocpdScore = s;
    }

    // ── 5. Combine scores
    const combined = wH! * hawkesScore + wC! * cusumScore + wB! * bocpdScore;

    // ── 6. Build signals list
    const signals: AnomalySignal[] = [];

    if (hawkesScore > 0.5) {
      signals.push({
        kind:  'volume_spike',
        score: hawkesScore,
        meta:  { lambda, mu: hawkesParams.mu, branching: hawkesParams.alpha / hawkesParams.beta },
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
        meta:  { sPos: cusumState.sPos, sNeg: cusumState.sNeg, h: cusumParams.h },
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
      signals,
      imbalance,
      hawkesLambda: lambda,
      cusumStat:    Math.max(cusumState.sPos, cusumState.sNeg),
      runLength:    bocpdResult.mapRunLength,
    };
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

  private emptyResult(): DetectionResult {
    return {
      anomaly:      false,
      confidence:   0,
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
