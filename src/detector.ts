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
   * of the training imbalance series; direction=short when below its negation.
   * The threshold is clamped at zero before use, so a trended baseline whose
   * quantile crosses zero can never flip long/short semantics.
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

export interface TrainedModels {
  hawkesParams:        HawkesParams;
  cusumParams:         CusumParams;
  bocpdPrior:          NormalGammaPrior;
  /**
   * p(imbalancePercentile) of the training rolling signed imbalance series.
   * predict() applies it as a symmetric ±threshold clamped at zero: on a
   * sell-trended baseline the raw quantile goes negative, and an unclamped
   * "imbalance > p75" fired 'long' on perfectly balanced flow while making
   * 'neutral' unreachable.
   */
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
   * volume rate (qty/s), one entry per horizon in `horizonsSec`.  detect()
   * scores its peak rolling rate at each scale as a robust z against these:
   * z = (peak − med) / (1.4826 · MAD).  Short horizons catch brief bursts,
   * long ones sustained volume waves spread too evenly to concentrate at the
   * short scales.
   */
  rateStats:           RobustStats[];
  volStats:            RobustStats[];
  /**
   * Multi-scale horizon family chosen at train() time (seconds, ascending):
   * fast, an optional geometric-mean mid scale, slow, and an optional
   * extended scale (up to 3× slow, capped by the training span).  Fast/slow
   * equal the config values when pinned explicitly; otherwise scaled up on
   * the fly for sparse streams so a horizon window contains enough trades to
   * carry a rate.
   */
  horizonsSec:         number[];
  /** = horizonsSec[0] — kept for introspection compatibility */
  fastHorizonSec:      number;
  /** The canonical alert timescale (slow horizon; not the extended scale) */
  slowHorizonSec:      number;
  /**
   * Per-channel score calibration from the null distribution of window maxima
   * over the training data (see ChannelCalib / calibrateChannel), one entry
   * per horizon in `horizonsSec`.
   */
  channelCalib: {
    rate: ChannelCalib[];
    vol:  ChannelCalib[];
  };
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

/**
 * Math.max(...xs) without the spread: each spread element becomes a call
 * argument, so a detection window of ~10⁶ trades overflows the call stack
 * (RangeError) before any math runs.
 */
function arrayMax(xs: number[]): number {
  let m = -Infinity;
  for (const x of xs) if (x > m) m = x;
  return m;
}

/** Index of the maximum element (first occurrence); -1 for an empty array. */
function argMax(xs: number[]): number {
  let m = -Infinity, idx = -1;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i]! > m) { m = xs[i]!; idx = i; }
  }
  return idx;
}

/**
 * Prior strength (in effective trades) for the burst-imbalance shrinkage:
 * the burst window's buy fraction is pulled toward the training buy fraction
 * as if the baseline had contributed this many trades.  A 5-trade burst is
 * shrunk ~2/3 toward baseline (±1 imbalance by chance stops reading as
 * directional conviction); a 100-trade burst keeps ~90% of its signal.
 */
const BURST_PRIOR_STRENGTH = 10;

/**
 * Qty-weighted imbalance of trades[from..to), shrunk toward NEUTRAL (buy
 * fraction 0.5).  Effective sample size is Kish's n_eff = (Σq)²/Σq², so a
 * "burst" that is one whale trade (n_eff ≈ 1) carries almost no directional
 * evidence regardless of its size, while many similar trades count fully.
 * This is the estimation form of a binomial significance test: strength of
 * evidence scales the deviation instead of hard-gating it.
 *
 * The prior center is deliberately 0.5, NOT the training buy fraction:
 * direction semantics are absolute ("buy aggression" ⇒ imbalance > 0), and a
 * baseline-centered prior would make every low-evidence window inherit the
 * baseline's own bias — a balanced burst after a sell-heavy baseline read as
 * conviction 'short'.  Relativity to the baseline is already encoded in the
 * directional threshold (trained imbalance quantile) applied by predict().
 */
function shrunkImbalance(
  trades: IAggregatedTradeData[],
  from:   number,
  to:     number,
): number {
  let buyVol = 0, sellVol = 0, sumQ2 = 0;
  for (let i = from; i < to; i++) {
    const q = trades[i]!.qty;
    if (trades[i]!.isBuyerMaker) sellVol += q; else buyVol += q;
    sumQ2 += q * q;
  }
  const total = buyVol + sellVol;
  if (!(total > 0) || !(sumQ2 > 0)) return 0;
  const nEff  = (total * total) / sumQ2;
  const p     = buyVol / total;
  const pStar = (nEff * p + BURST_PRIOR_STRENGTH * 0.5) / (nEff + BURST_PRIOR_STRENGTH);
  return 2 * pStar - 1;
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

/**
 * Per-channel score mapping, self-calibrated from the null distribution of
 * window maxima on the training data (see calibrateChannel).  Standardized
 * exceedance t = (z − c − u·spanShift) / u is mapped through the rational
 * sigmoid 0.5 + 0.5·t/(1+|t|):
 *
 *   z = c   (the calibrated level)   → t = 0 → score 0.5
 *   z = c+u (one tail unit beyond)   → t = 1 → score 0.75
 *   further tail units               → 0.83, 0.875, … approaching 1 slowly
 *
 * The level c = max(min(empirical q85 of the null maxima, Gumbel-left-fit
 * q85), universal floor): the Gumbel term makes it robust to events inside
 * the training window (contamination inflates only the upper quantiles), the
 * floor is the mapping validated on a full day of real data
 * (CALIB_FALLBACK_RATE / CALIB_FALLBACK_VOL) — a 15–30 min window yields too
 * few null stretches to trust a very low estimate of the normal tail.
 */
interface ChannelCalib {
  /** Score-0.5 level: max(min(null q85, Gumbel-left q85), universal floor) */
  c: number;
  /** Universal tail unit; one u beyond c ⇒ 0.75 */
  u: number;
  /**
   * Quantile ladder of the null window-maxima distribution at NULLQ_PCTS —
   * exposed for introspection and threshold research (empty when the fallback
   * mapping is in effect).
   */
  nullQ: number[];
}

/**
 * Percentiles reported in ChannelCalib.nullQ.  The lower half exists for
 * contamination-robust research: an event inside the training window inflates
 * only the UPPER quantiles of the window-maxima distribution, so a clean-tail
 * estimate must anchor on the left part.
 */
export const NULLQ_PCTS = [5, 10, 25, 50, 75, 80, 85, 90, 95, 97, 99] as const;

/**
 * Universal floor/fallback mappings, found by dense brute force over the
 * (anchor quantile × rate floor × vol floor × tail unit) space on the
 * real-data benchmark (test/eval.test.ts) and verified to sit on a stable
 * plateau (±1–2 in any parameter moves metrics ≤ 1–2 points).
 *
 * Rate and volume channels get DIFFERENT floors because they separate
 * differently on real data: peak volume z of normal windows stays low while
 * anomalies reach z ≥ 13 (clean separation ⇒ low floor 6.5), whereas peak
 * arrival-rate z of normal windows overlaps the anomaly range much more
 * (⇒ high floor 14).  The tail unit u is shared.
 *
 * At the default confidence 0.75 (fires at z ≥ c + u) this means: volume
 * z ≥ 12, arrival-rate z ≥ 19.5 — measured 95.2% event recall at 2.45%
 * false alarms, dominating the previous single-floor mapping on all three
 * benchmark metrics.
 */
const CALIB_FALLBACK_RATE: ChannelCalib = { c: 14,  u: 5.5, nullQ: [] };
const CALIB_FALLBACK_VOL:  ChannelCalib = { c: 6.5, u: 5.5, nullQ: [] };

/**
 * Floor surcharge for the non-canonical scan scales (mid / extended): the
 * universal floors above were validated for the fast/slow scales, and every
 * extra scale is an extra look at the same stream, so the cross-scale max
 * needs a mild multiplicity correction.  Measured Pareto-best at 1.15 on the
 * full-day benchmark (grid 1.0 / 1.15 / 1.3 / 1.5).
 */
const NONCANONICAL_FLOOR_MULT = 1.15;

// ─── Serialization ────────────────────────────────────────────────────────────

/**
 * JSON-friendly snapshot of a detector: configuration + trained models.
 * Produced by toJSON() (and therefore by JSON.stringify(detector)); consumed
 * by VolumeAnomalyDetector.fromJSON().  Every value is a plain finite number
 * or boolean, so the snapshot survives a JSON round-trip losslessly.
 */
export interface DetectorSnapshot {
  /** Snapshot format version; current writers emit 1 */
  version:      number;
  config:       Required<DetectorConfig>;
  /** Whether the fast/slow horizons were pinned explicitly (vs auto-chosen) */
  explicitFast: boolean;
  explicitSlow: boolean;
  models:       TrainedModels | null;
}

/**
 * Deep copy for snapshot payloads.  Trained models are plain trees of finite
 * numbers, so a JSON round-trip is lossless here — and it stays inside the
 * "lib": ["ES2022"] surface (no structuredClone dependency).
 */
function deepClone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T;
}

/** Top-level keys a serialized TrainedModels must carry. */
const MODEL_KEYS: readonly (keyof TrainedModels)[] = [
  'hawkesParams', 'cusumParams', 'bocpdPrior', 'imbalanceThreshold',
  'rateStats', 'volStats', 'fastHorizonSec', 'slowHorizonSec',
  'channelCalib', 'lambdaBaseline', 'bocpdNoiseFloor',
];

/**
 * Structural validation for deserialized models: all top-level keys present
 * and every leaf a finite number.  JSON.stringify silently turns NaN/Infinity
 * into null, so a corrupted or hand-edited snapshot surfaces here with a
 * path in the message instead of as NaN confidence deep inside detect().
 */
function assertModelShape(models: unknown): asserts models is TrainedModels {
  if (models === null || typeof models !== 'object' || Array.isArray(models)) {
    throw new Error('snapshot.models must be an object or null');
  }
  for (const key of MODEL_KEYS) {
    if (!(key in models)) throw new Error(`snapshot.models.${key} is missing`);
  }
  const walk = (v: unknown, path: string): void => {
    if (typeof v === 'number') {
      if (!Number.isFinite(v)) throw new Error(`snapshot.models: non-finite number at ${path}`);
      return;
    }
    if (Array.isArray(v)) {
      v.forEach((x, i) => walk(x, `${path}[${i}]`));
      return;
    }
    if (v !== null && typeof v === 'object') {
      for (const [k, x] of Object.entries(v)) walk(x, `${path}.${k}`);
      return;
    }
    throw new Error(`snapshot.models: unexpected ${v === null ? 'null' : typeof v} at ${path}`);
  };
  walk(models, 'models');
}

// ─── Detector class ───────────────────────────────────────────────────────────

export class VolumeAnomalyDetector {
  private readonly cfg: Required<DetectorConfig>;
  private models:       TrainedModels | null = null;
  /** User pinned the horizon explicitly — skip on-the-fly selection */
  private readonly explicitFast: boolean;
  private readonly explicitSlow: boolean;

  constructor(config: DetectorConfig = {}) {
    this.cfg = { ...DEFAULTS, ...config };
    this.explicitFast = config.rateHorizonSec !== undefined;
    this.explicitSlow = config.slowHorizonSec !== undefined;
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

    // Numeric config options: a bad value here never crashes — it silently
    // produces a miscalibrated or dead detector (e.g. hazardLambda ≤ 1 makes
    // H = 1/λ ≥ 1, log(1 − H) = −∞/NaN, and the BOCPD run-length state
    // collapses to empty on the first update) — so reject it loudly instead.
    const check = (name: string, v: number, ok: boolean, expected: string) => {
      if (!ok) throw new Error(`${name} must be ${expected}, got ${v}`);
    };
    const { windowSize, hazardLambda, cusumKSigmas, cusumHSigmas,
            rateHorizonSec, slowHorizonSec, imbalancePercentile } = this.cfg;
    check('windowSize',     windowSize,     Number.isInteger(windowSize) && windowSize >= 1, 'an integer >= 1');
    check('hazardLambda',   hazardLambda,   Number.isFinite(hazardLambda) && hazardLambda > 1, 'a finite number > 1');
    check('cusumKSigmas',   cusumKSigmas,   Number.isFinite(cusumKSigmas) && cusumKSigmas > 0, 'a finite number > 0');
    check('cusumHSigmas',   cusumHSigmas,   Number.isFinite(cusumHSigmas) && cusumHSigmas > 0, 'a finite number > 0');
    check('rateHorizonSec', rateHorizonSec, Number.isFinite(rateHorizonSec) && rateHorizonSec > 0, 'a finite number > 0');
    check('slowHorizonSec', slowHorizonSec, Number.isFinite(slowHorizonSec) && slowHorizonSec > 0, 'a finite number > 0');
    check('imbalancePercentile', imbalancePercentile,
      Number.isFinite(imbalancePercentile) && imbalancePercentile >= 0 && imbalancePercentile <= 100,
      'a finite number in [0, 100]');
  }

  /**
   * trades[].timestamp must be Unix MILLISECONDS.  A wrong unit never crashes —
   * it silently rescales every time horizon 1000× (the most damaging
   * integration mistake possible) — so the two real-world mix-ups are rejected
   * here.  Only epoch-like values can be judged; relative/synthetic timestamps
   * pass through untouched:
   *   epoch seconds → [1e9, 4e9) covers years 2001–2096, where this mistake
   *     actually lives; as relative ms that's a 12–46 day origin — narrow
   *     enough not to collide with synthetic data (kept deliberately tighter
   *     than the full seconds range so arbitrary synthetic origins < 1e9 pass);
   *   epoch µs      → ≥ 1e14; as ms that's year 5138+, colliding with nothing.
   */
  private static assertMillis(t0: number): void {
    if (t0 >= 1e9 && t0 < 4e9) {
      throw new Error(
        `timestamps look like Unix SECONDS (first = ${t0}); timestamp must be in milliseconds`,
      );
    }
    if (t0 >= 1e14) {
      throw new Error(
        `timestamps look like Unix MICROseconds (first = ${t0}); timestamp must be in milliseconds`,
      );
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
    VolumeAnomalyDetector.assertMillis(sorted[0]!.timestamp);

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
    const spanSec = allTimestamps[allTimestamps.length - 1]! - allTimestamps[0]!;

    // ── Horizons chosen on the fly (unless pinned in config).
    // The config values act as floors; for sparse streams the horizons are
    // scaled up from the median inter-trade gap so a horizon window carries
    // ≥ ~25 trades and the "rate" statistic actually measures something.
    // Both are capped by the training span (the slow horizon needs several
    // independent stretches inside the baseline to be calibratable).
    const gaps: number[] = [];
    for (let i = 1; i < allTimestamps.length; i++) {
      gaps.push(allTimestamps[i]! - allTimestamps[i - 1]!);
    }
    const medianGap = gaps.length > 0 ? quantile(gaps, 50) : 1;
    let fastHorizonSec = this.explicitFast
      ? this.cfg.rateHorizonSec
      : Math.min(
          Math.max(this.cfg.rateHorizonSec, 25 * medianGap),
          Math.max(spanSec / 10, this.cfg.rateHorizonSec),
        );
    let slowHorizonSec = this.explicitSlow
      ? this.cfg.slowHorizonSec
      : Math.min(
          Math.max(this.cfg.slowHorizonSec, 6 * fastHorizonSec),
          Math.max(spanSec / 4, this.cfg.slowHorizonSec),
        );
    if (slowHorizonSec < fastHorizonSec) slowHorizonSec = fastHorizonSec;

    // ── Multi-scale horizon family (scan statistic).
    // Two fixed horizons are a two-tooth comb: a burst living at ~2× the fast
    // horizon or a wave at ~3× the slow one dilutes at both and loses z.  Add
    // a geometric-mean MID scale between fast and slow (when they are far
    // enough apart to leave a gap) and an EXTENDED scale above slow (when the
    // training span can still calibrate it: same span/4 cap as slow).  Every
    // scale runs the same statistic → robust z → null-calibrated mapping; the
    // final score is the calibrated max over all scales (each scale's own
    // null absorbs its multiple-look cost).
    const horizonsSec: number[] = [fastHorizonSec];
    if (slowHorizonSec / fastHorizonSec >= 4) {
      horizonsSec.push(Math.sqrt(fastHorizonSec * slowHorizonSec));
    }
    if (slowHorizonSec > horizonsSec[horizonsSec.length - 1]!) {
      horizonsSec.push(slowHorizonSec);
    }
    const extendedSec = Math.min(3 * slowHorizonSec, Math.max(spanSec / 4, slowHorizonSec));
    if (extendedSec >= 2 * slowHorizonSec) horizonsSec.push(extendedSec);

    // ── Per-scale stats + score-mapping calibration from the null
    // distribution: what peak z does this baseline produce on its own, in
    // alert-sized stretches?
    const rateStats: RobustStats[] = [];
    const volStats:  RobustStats[] = [];
    const calibRate: ChannelCalib[] = [];
    const calibVol:  ChannelCalib[] = [];
    for (const h of horizonsSec) {
      const roll = this.rollingRates(sorted, allTimestamps, h);
      const rs   = robustStats(roll.rates);
      const vs   = robustStats(roll.volRates);
      rateStats.push(rs);
      volStats.push(vs);
      const ts = roll.firstIdx >= 0 ? allTimestamps.slice(roll.firstIdx) : [];
      // Non-canonical scales (mid/extended) pay a floor surcharge: the
      // universal floors were brute-force-validated FOR the fast/slow scales,
      // and extra scales are extra looks at the same stream — the cross-scale
      // max needs a mild multiplicity correction.  ×1.15 measured Pareto-best
      // on the full-day benchmark (same recall/events as ×1.0, −0.25% FP;
      // ×1.3 starts trading recall away).
      const mult     = h === fastHorizonSec || h === slowHorizonSec ? 1 : NONCANONICAL_FLOOR_MULT;
      const fbRate   = mult === 1 ? CALIB_FALLBACK_RATE : { ...CALIB_FALLBACK_RATE, c: CALIB_FALLBACK_RATE.c * mult };
      const fbVol    = mult === 1 ? CALIB_FALLBACK_VOL  : { ...CALIB_FALLBACK_VOL,  c: CALIB_FALLBACK_VOL.c  * mult };
      calibRate.push(this.calibrateChannel(
        roll.rates.map((x) => robustZ(x, rs)), ts, slowHorizonSec, fbRate,
      ));
      calibVol.push(this.calibrateChannel(
        roll.volRates.map((x) => robustZ(x, vs)), ts, slowHorizonSec, fbVol,
      ));
    }
    const channelCalib = { rate: calibRate, vol: calibVol };

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
    // threshold that reflects actual baseline buy/sell bias.  Applied by
    // predict() as symmetric ±threshold clamped at zero (see TrainedModels).
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
      horizonsSec,
      fastHorizonSec,
      slowHorizonSec,
      channelCalib,
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
    // !(…) also catches NaN.  The most common real mistake is percent instead
    // of a fraction (75 for 0.75) — anomaly would then silently never fire.
    if (!(confidence >= 0 && confidence <= 1)) {
      throw new Error(`confidence must be in [0, 1], got ${confidence}`);
    }
    if (trades.length === 0) {
      return this.emptyResult();
    }

    const sorted = [...trades].sort((a, b) => a.timestamp - b.timestamp);
    VolumeAnomalyDetector.assertMillis(sorted[0]!.timestamp);
    const {
      hawkesParams, cusumParams, bocpdPrior,
      rateStats, volStats, horizonsSec, slowHorizonSec, channelCalib,
      lambdaBaseline, bocpdNoiseFloor,
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
    // Each z is mapped to a score through the channel's self-calibrated null
    // mapping (see ChannelCalib): 0.5 at the P90 of the maxima the baseline
    // itself produced in alert-sized stretches, 0.75 at their P99.  A window
    // longer than the calibration stretch gets more independent looks at the
    // maximum, so the bar rises by one tail unit per decade of extra span
    // (exponential-tail return-level correction).
    const timestamps = sorted.map((t) => t.timestamp / 1000);
    const lambda     = hawkesPeakLambda(timestamps, hawkesParams);
    const spanSec    = timestamps[timestamps.length - 1]! - timestamps[0]!;
    const spanShift  = Math.log10(Math.max(1, spanSec / slowHorizonSec));
    // Standardized channel exceedance: t = 0 at the calibrated level (score
    // 0.5), t = 1 one tail unit beyond it (score 0.75).
    const chT = (z: number, cal: ChannelCalib) => (z - cal.c - cal.u * spanShift) / cal.u;
    // Rational (algebraic) sigmoid t → score.  Same anchors as the previous
    // exponential sigmoid — 0.5 at t=0, 0.75 at t=1, so decisions at the
    // default 0.75 threshold are identical — but the top approaches 1
    // harmonically instead of exponentially: no double-precision saturation,
    // so the score keeps RANKING extreme events (predictive use) instead of
    // collapsing everything strong into ties at ≈1.  Never reaches 1, which
    // preserves the "confidence 1.0 never fires" semantics without a clamp.
    const ratSig = (t: number) => 0.5 + (0.5 * t) / (1 + Math.abs(t));

    let zRate = 0, zVol = 0, zRateSlow = 0, zVolSlow = 0;
    let hawkesScore = 0;
    // Peak burst window of the winning channel: trade index of the rolling
    // window END whose statistic won the score, plus that channel's horizon.
    let peakIdx = -1, peakHorizonSec = 0;
    const zRates: number[] = [];
    const zVols:  number[] = [];
    if (timestamps.length >= 2) {
      // Multi-scale scan: every horizon in the trained family contributes a
      // rate and a volume channel; each channel is standardized against its
      // OWN null calibration, so scales are comparable and the max is fair.
      let tRateBest = -Infinity, tVolBest = -Infinity;
      let bestT = -Infinity;
      const iSlow = horizonsSec.indexOf(slowHorizonSec);
      for (let k = 0; k < horizonsSec.length; k++) {
        const roll = this.rollingRates(sorted, timestamps, horizonsSec[k]!);
        for (const isRate of [true, false]) {
          const xs    = isRate ? roll.rates    : roll.volRates;
          const stats = isRate ? rateStats[k]! : volStats[k]!;
          const cal   = isRate ? channelCalib.rate[k]! : channelCalib.vol[k]!;
          const ai = argMax(xs);
          const z  = ai >= 0 ? robustZ(xs[ai]!, stats) : 0;
          const t  = ai >= 0 ? chT(z, cal) : -Infinity;
          if (isRate) { zRates.push(z); if (t > tRateBest) tRateBest = t; }
          else        { zVols.push(z);  if (t > tVolBest)  tVolBest  = t; }
          if (t > bestT) {
            bestT = t;
            // firstIdx < 0 → single whole-window fallback sample, not trade-aligned
            peakIdx        = roll.firstIdx >= 0 ? roll.firstIdx + ai : -1;
            peakHorizonSec = horizonsSec[k]!;
          }
          // stats reporting keeps the fastest/slow scales (API compatibility)
          if (k === 0)     { if (isRate) zRate     = z; else zVol     = z; }
          if (k === iSlow) { if (isRate) zRateSlow = z; else zVolSlow = z; }
        }
      }
      // Corroboration (Stouffer): a moderate arrival-rate excess AND a
      // moderate volume excess together are stronger evidence than either
      // alone, which a pure max() discards.  Combined strictly ACROSS types
      // (best rate-t with best vol-t): scales of the SAME type see the same
      // underlying wiggle at different bandwidths, and letting them
      // corroborate each other double-counts one piece of evidence
      // (measured: same-type cross-scale corroboration pushed borderline
      // normal buckets over the threshold and raised FP with no recall gain).
      // Only positive support from the other type counts, and the leader
      // alone is the floor.
      const tLead  = Math.max(tRateBest, tVolBest);
      const tOther = Math.min(tRateBest, tVolBest);
      const combined = Number.isFinite(tLead)
        ? Math.max(tLead, (tLead + Math.max(Number.isFinite(tOther) ? tOther : 0, 0)) / Math.SQRT2)
        : -Infinity;
      hawkesScore = Number.isFinite(combined) ? ratSig(combined) : 0;
    }
    // λ ratio (peak Hawkes intensity vs the training peak) is reported in
    // stats/meta for transparency but deliberately kept OUT of the score: it
    // correlates with the rate z-channels and only added false-positive tail
    // in the real-data evaluation.
    const lambdaRatio = lambdaBaseline > 0 && Number.isFinite(lambda) ? lambda / lambdaBaseline : 0;

    // ── 2. Current imbalance (full window, signed — for direction reporting)
    const imbalance = volumeImbalance(sorted);
    const absImb    = Math.abs(imbalance);

    // ── 2b. Burst-local imbalance: order flow inside the winning channel's
    // peak rolling window, not the whole detection window.  A burst's onset
    // direction gets diluted by post-burst two-way flow when measured over
    // the full window (measured on real data: −0.42 full-window vs −0.9 at
    // the burst).  Shrunk toward the training buy fraction by effective
    // sample size (see shrunkImbalance) so a few-trade or one-whale "burst"
    // does not fake directional conviction.
    const n = sorted.length;
    let burstImbalance: number;
    let peakTs = sorted[n - 1]!.timestamp;
    if (peakIdx >= 0) {
      const tEnd = timestamps[peakIdx]!;
      let j = peakIdx;
      while (j > 0 && timestamps[j - 1]! > tEnd - peakHorizonSec) j--;
      burstImbalance = shrunkImbalance(sorted, j, peakIdx + 1);
      peakTs         = sorted[peakIdx]!.timestamp;
    } else {
      burstImbalance = shrunkImbalance(sorted, 0, n);
    }

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
      stats:        { zRate, zVol, zRateSlow, zVolSlow, lambdaRatio, zRates, zVols },
      signals,
      imbalance,
      burstImbalance,
      peakTs,
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
  ): { rates: number[]; volRates: number[]; firstIdx: number } {
    const n = sorted.length;
    const qtyPrefix = new Array<number>(n + 1).fill(0);
    for (let i = 0; i < n; i++) {
      qtyPrefix[i + 1] = qtyPrefix[i]! + sorted[i]!.qty;
    }
    const rates:    number[] = [];
    const volRates: number[] = [];
    let firstIdx = -1; // trade index of the first full-horizon sample
    const t0   = timestamps[0]!;
    const span = timestamps[n - 1]! - t0;
    if (span >= horizon) {
      let j = 0; // two-pointer: first trade inside the window (tᵢ − horizon, tᵢ]
      for (let i = 0; i < n; i++) {
        if (timestamps[i]! - t0 < horizon) continue; // truncated lookback
        if (firstIdx < 0) firstIdx = i;
        while (timestamps[j]! <= timestamps[i]! - horizon) j++;
        rates.push((i - j + 1) / horizon);
        volRates.push((qtyPrefix[i + 1]! - qtyPrefix[j]!) / horizon);
      }
    }
    if (rates.length === 0 && n >= 2) {
      const dur = Math.max(span, Math.min(1, horizon));
      rates.push(n / dur);
      volRates.push(qtyPrefix[n]! / dur);
      firstIdx = -1; // single whole-window sample — not aligned to a trade
    }
    return { rates, volRates, firstIdx };
  }

  // ─── Score-mapping calibration ──────────────────────────────────────────────

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
  private calibrateChannel(
    zSamples:  number[],
    sampleTs:  number[],
    windowSec: number,
    fallback:  ChannelCalib,
  ): ChannelCalib {
    if (zSamples.length === 0 || zSamples.length !== sampleTs.length) return fallback;
    const SUB  = 16;            // sub-buckets per window
    const step = windowSec / SUB;
    const t0   = sampleTs[0]!;
    const nBuckets = Math.floor((sampleTs[sampleTs.length - 1]! - t0) / step) + 1;
    if (nBuckets < 2 * SUB) return fallback;

    // Max per step-bucket, then sliding window max = max of SUB consecutive
    // buckets at every bucket offset (window length stays exactly windowSec).
    const bucketMax = new Array<number>(nBuckets).fill(-Infinity);
    for (let i = 0; i < zSamples.length; i++) {
      const b = Math.min(nBuckets - 1, Math.floor((sampleTs[i]! - t0) / step));
      if (zSamples[i]! > bucketMax[b]!) bucketMax[b] = zSamples[i]!;
    }
    const maxima: number[] = [];
    for (let b = 0; b + SUB - 1 < nBuckets; b++) {
      let m = -Infinity;
      for (let k = 0; k < SUB; k++) {
        if (bucketMax[b + k]! > m) m = bucketMax[b + k]!;
      }
      if (m !== -Infinity) maxima.push(m);
    }
    if (maxima.length < 8) return fallback;

    const nullQ = NULLQ_PCTS.map((p) => quantile(maxima, p));
    const q85 = nullQ[NULLQ_PCTS.indexOf(85)]!;

    // ── Contamination-robust level: an event INSIDE the training window
    // inflates only the upper quantiles of the window-maxima distribution, so
    // the empirical q85 of a "hot" baseline deafens the detector to the very
    // escalation it precedes.  Fit a Gumbel (the max-domain law for
    // light-tailed maxima) to the LEFT quantiles [P25, P50, P75] — where
    // contamination does not live — and extrapolate a clean q85 from it.
    // Level = min(empirical q85, Gumbel q85): equal on clean baselines,
    // Gumbel wins on contaminated ones.  Measured on the full-day benchmark:
    // reaches 96.8% event recall at 2.66% FP where the empirical-only level
    // needed 3.45%.
    const xg = (p: number) => -Math.log(-Math.log(p));
    const pts: Array<[number, number]> = [25, 50, 75].map(
      (p) => [xg(p / 100), nullQ[NULLQ_PCTS.indexOf(p as 25 | 50 | 75)]!] as [number, number],
    );
    const mx = (pts[0]![0] + pts[1]![0] + pts[2]![0]) / 3;
    const my = (pts[0]![1] + pts[1]![1] + pts[2]![1]) / 3;
    let sxy = 0, sxx = 0;
    for (const [x, y] of pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; }
    const beta    = sxy / sxx;                 // Gumbel scale
    const cGumbel = (my - beta * mx) + beta * xg(0.85);

    // Only the LEVEL adapts; the tail unit stays universal.  Measured on the
    // real-data benchmark:
    //  - the fallback level acts as a FLOOR, not just a fallback — a 15–30 min
    //    baseline yields only ~10² correlated null stretches, so its upper
    //    quantiles routinely UNDERestimate the true normal tail, and trusting
    //    a low estimate raised false alarms;
    //  - anchoring the level at null P99 (instead of P85) crushed escalating
    //    events whose baseline was already hot (spike_2 fixture);
    //  - adapting the tail unit u to the null spread (empirical P99 − P90
    //    OR the Gumbel-fitted spread) traded away score comparability across
    //    baselines: forward-move ranking (AUC) dropped measurably, so u stays
    //    fixed even though the adaptive variant bought ~2 recall points.
    // The P85 anchor and the per-channel-type floors/tail unit come from the
    // dense brute-force search (see CALIB_FALLBACK_RATE/_VOL).
    return {
      c: Math.max(Math.min(q85, cGumbel), fallback.c),
      u: fallback.u,
      nullQ,
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
      stats:        { zRate: 0, zVol: 0, zRateSlow: 0, zVolSlow: 0, lambdaRatio: 0, zRates: [], zVols: [] },
      signals:      [],
      imbalance:    0,
      burstImbalance: 0,
      peakTs:       0,
      hawkesLambda: 0,
      cusumStat:    0,
      runLength:    0,
    };
  }

  // ─── Serialization ──────────────────────────────────────────────────────────

  /**
   * Snapshot of configuration + trained models (deep-copied — mutating the
   * result cannot poison the detector).  Called automatically by
   * JSON.stringify(detector).  Restore with VolumeAnomalyDetector.fromJSON():
   * train once (e.g. in a worker on a schedule), serialize, detect anywhere
   * else without re-training.
   */
  toJSON(): DetectorSnapshot {
    return {
      version:      1,
      config:       { ...this.cfg, scoreWeights: [...this.cfg.scoreWeights] },
      explicitFast: this.explicitFast,
      explicitSlow: this.explicitSlow,
      models:       this.models && deepClone(this.models),
    };
  }

  /**
   * Reconstruct a detector from a toJSON() snapshot (object or JSON string).
   * The snapshot is validated structurally — config through the constructor,
   * models leaf-by-leaf — so corrupted or hand-edited state fails loudly here
   * rather than as NaN confidence inside detect().
   */
  static fromJSON(snapshot: DetectorSnapshot | string): VolumeAnomalyDetector {
    const s = (typeof snapshot === 'string'
      ? JSON.parse(snapshot)
      : snapshot) as DetectorSnapshot;
    if (s === null || typeof s !== 'object') {
      throw new Error('snapshot must be an object or a JSON string');
    }
    if (s.version !== 1) {
      throw new Error(`unsupported snapshot version: ${s.version}`);
    }
    if (s.config === null || typeof s.config !== 'object') {
      throw new Error('snapshot.config is missing');
    }
    // Auto-chosen horizons must stay auto after restore: the constructor
    // derives the explicit-pin flags from key presence, so drop the keys that
    // were filled from DEFAULTS rather than by the user.
    const cfg: DetectorConfig = { ...s.config };
    if (!s.explicitFast) delete cfg.rateHorizonSec;
    if (!s.explicitSlow) delete cfg.slowHorizonSec;
    const det = new VolumeAnomalyDetector(cfg); // re-runs config validation
    if (s.models !== null && s.models !== undefined) {
      assertModelShape(s.models);
      det.models = deepClone(s.models);
    }
    return det;
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
