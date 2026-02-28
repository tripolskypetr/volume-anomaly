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

import type { NormalGammaSS } from '../types.js';

// ─── Prior hyperparameters ────────────────────────────────────────────────────

export interface NormalGammaPrior {
  /** Prior mean */
  mu0:     number;
  /** Prior pseudo-observations (strength of mean belief) */
  kappa0:  number;
  /** Prior shape (α₀, must be > 0) */
  alpha0:  number;
  /** Prior rate  (β₀, must be > 0) */
  beta0:   number;
}

/**
 * Default weakly-informative prior.
 * mu0 derived from training data; alpha0/beta0 encode variance belief.
 */
export function defaultPrior(trainingMean: number, trainingVar: number): NormalGammaPrior {
  return {
    mu0:    trainingMean,
    kappa0: 1,
    alpha0: 1,
    beta0:  trainingVar > 0 ? trainingVar : 1,
  };
}

// ─── Predictive probability p(xₜ | rₜ₋₁, x_{t-r:t}) ─────────────────────────

/**
 * Student-t predictive log-probability for a Normal-Gamma model.
 * Sufficient statistics updated via Welford's online algorithm.
 */
function studentTPredLogProb(x: number, ss: NormalGammaSS, prior: NormalGammaPrior): number {
  const { mu0, kappa0, alpha0, beta0 } = prior;
  const { n, mean, m2 } = ss;

  const kappaN  = kappa0  + n;
  const alphaN  = alpha0  + n / 2;
  const muN     = (kappa0 * mu0 + n * mean) / kappaN;
  const betaN   = beta0   + 0.5 * m2 + (kappa0 * n * (mean - mu0) ** 2) / (2 * kappaN);

  // Predictive: Student-t(2αN, muN, βN(κN+1)/(αN·κN))
  const df    = 2 * alphaN;
  const scale = (betaN * (kappaN + 1)) / (alphaN * kappaN);
  const z     = (x - muN) / Math.sqrt(scale);

  // ln p(x) = ln Γ((df+1)/2) − ln Γ(df/2) − 0.5 ln(π·df·scale) − (df+1)/2 · ln(1 + z²/df)
  return (
    logGamma((df + 1) / 2) -
    logGamma(df / 2) -
    0.5 * Math.log(Math.PI * df * scale) -
    ((df + 1) / 2) * Math.log(1 + (z * z) / df)
  );
}

// ─── Sufficient statistics update (Welford) ───────────────────────────────────

function ssUpdate(ss: NormalGammaSS, x: number): NormalGammaSS {
  const n    = ss.n + 1;
  const delta = x - ss.mean;
  const mean  = ss.mean + delta / n;
  const delta2 = x - mean;
  const m2   = ss.m2 + delta * delta2;
  return { n, mean, m2 };
}

function ssEmpty(): NormalGammaSS {
  return { n: 0, mean: 0, m2: 0 };
}

// ─── BOCPD state ──────────────────────────────────────────────────────────────

export interface BocpdState {
  /** log P(rₜ = r | x₁:ₜ) for r = 0, 1, ..., t  (unnormalized) */
  logProbs:  number[];
  /** Sufficient statistics for each run-length hypothesis */
  suffStats: NormalGammaSS[];
  /** Total observations processed */
  t:         number;
}

export function bocpdInitState(): BocpdState {
  return {
    logProbs:  [0],           // P(r₀ = 0) = 1  →  log = 0
    suffStats: [ssEmpty()],
    t:         0,
  };
}

// ─── Update ───────────────────────────────────────────────────────────────────

export interface BocpdUpdateResult {
  state:      BocpdState;
  /** Most probable run length (MAP) */
  mapRunLength: number;
  /** P(changepoint at t) = P(rₜ = 0 | x₁:ₜ) */
  cpProbability: number;
}

/**
 * Process one new observation, return updated state + diagnostics.
 * hazardLambda — expected gap between changepoints (periods).
 */
export function bocpdUpdate(
  state:        BocpdState,
  x:            number,
  prior:        NormalGammaPrior,
  hazardLambda: number = 200,
): BocpdUpdateResult {
  const H   = 1 / hazardLambda; // hazard rate
  const prevN = state.logProbs.length;

  // Step 1: compute predictive log-probs for each old run-length hypothesis
  const logPredProbs = state.logProbs.map((_, r) =>
    studentTPredLogProb(x, state.suffStats[r]!, prior),
  );

  // Step 2: compute joint log-probs
  const jointLogProbs: number[] = new Array(prevN + 1).fill(-Infinity);

  // changepoint: new run length = 0
  let logCpMass = -Infinity;
  for (let r = 0; r < prevN; r++) {
    const contrib = state.logProbs[r]! + logPredProbs[r]! + Math.log(H);
    logCpMass = logSumExp(logCpMass, contrib);
  }
  jointLogProbs[0] = logCpMass;

  // growth: run length increments by 1
  for (let r = 0; r < prevN; r++) {
    jointLogProbs[r + 1] = state.logProbs[r]! + logPredProbs[r]! + Math.log(1 - H);
  }

  // Step 3: normalise (log-domain)
  const logNorm = jointLogProbs.reduce(logSumExp, -Infinity);
  const normLogProbs = jointLogProbs.map((lp) => lp - logNorm);

  // Step 4: update sufficient statistics
  const newSuffStats: NormalGammaSS[] = [ssEmpty()];
  for (let r = 0; r < prevN; r++) {
    newSuffStats.push(ssUpdate(state.suffStats[r]!, x));
  }

  // Step 5: prune negligible hypotheses (log prob < -30 ≈ prob < 1e-13)
  const PRUNE_THRESH = -30;
  const keep = normLogProbs.map((lp) => lp > PRUNE_THRESH);
  const prunedLogProbs  = normLogProbs.filter((_, i) => keep[i]);
  const prunedSuffStats = newSuffStats.filter(  (_, i) => keep[i]);

  const newState: BocpdState = {
    logProbs:  prunedLogProbs,
    suffStats: prunedSuffStats,
    t:         state.t + 1,
  };

  // MAP run length (highest probability)
  let mapR = 0;
  let mapLP = -Infinity;
  for (let r = 0; r < normLogProbs.length; r++) {
    if (normLogProbs[r]! > mapLP) { mapLP = normLogProbs[r]!; mapR = r; }
  }

  return {
    state:          newState,
    mapRunLength:   mapR,
    cpProbability:  Math.exp(normLogProbs[0] ?? -Infinity),
  };
}

// ─── Score ────────────────────────────────────────────────────────────────────

/**
 * Anomaly score [0,1]: how much evidence there is for a recent changepoint.
 *
 * The MAP run length is the primary signal.  In a stable process with no
 * changepoints, mapRunLength grows monotonically with each new observation
 * and after t steps mapRunLength ≈ t.  A changepoint resets it.  The ratio
 *
 *   gap = (t − mapRunLength) / t
 *
 * is therefore 0 in a stable process and close to 1 immediately after a
 * changepoint (mapRunLength ≪ t).  We pass this through a sigmoid centred
 * at gap = 0.5 so that the score becomes meaningful only after the run
 * length has fallen to less than half the total elapsed steps.
 *
 * Additionally cpProbability provides a complementary signal when it spikes
 * above the prior hazard H = 1/hazardLambda.
 *
 * @param result  Output from bocpdUpdate (must include state.t via BocpdState).
 * @param hazardLambda  Expected gap between changepoints.
 */
export function bocpdAnomalyScore(result: BocpdUpdateResult, hazardLambda = 200): number {
  const t = result.state.t;
  if (t < 2) return 0;

  // Gap score: fraction of elapsed steps NOT in the current run.
  const gap = (t - result.mapRunLength) / t;
  const gapScore = 1 / (1 + Math.exp(-(gap - 0.5) * 8)); // sigmoid, steep at gap=0.5

  // cpProb score: normalised by prior hazard
  const H        = 1 / hazardLambda;
  const ratio    = result.cpProbability / H;
  const cpScore  = 1 / (1 + Math.exp(-(ratio - 3)));

  return Math.max(gapScore, cpScore);
}

// ─── Batch ────────────────────────────────────────────────────────────────────

/**
 * Run BOCPD on a series, returning per-observation changepoint probabilities.
 */
export function bocpdBatch(
  series:       number[],
  prior:        NormalGammaPrior,
  hazardLambda: number = 200,
): { cpProbs: number[]; mapRunLengths: number[] } {
  let state = bocpdInitState();
  const cpProbs: number[] = [];
  const mapRunLengths: number[] = [];

  for (const x of series) {
    const r = bocpdUpdate(state, x, prior, hazardLambda);
    state = r.state;
    cpProbs.push(r.cpProbability);
    mapRunLengths.push(r.mapRunLength);
  }

  return { cpProbs, mapRunLengths };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function logSumExp(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** Lanczos log-gamma (g=5, n=6) — sufficient precision for df > 2 */
function logGamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
  let sum = 1.000000000190015;
  for (let j = 0; j < c.length; j++) sum += c[j]! / (x + j + 1);
  return (
    Math.log(Math.sqrt(2 * Math.PI)) +
    (x + 0.5) * Math.log(x + 5.5) -
    (x + 5.5) +
    Math.log(sum)
  );
}
