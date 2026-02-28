<p align="center">
  <strong>Volume anomaly detection for trade streams</strong><br>
  Hawkes Process · CUSUM · Bayesian Online Changepoint Detection<br>
  TypeScript. Zero dependencies.
</p>

## Installation

```bash
npm install volume-anomaly
```

## Overview

The library detects **abnormal surges in trade flow** — sudden acceleration of arrivals, buy/sell imbalance shifts, and structural regime changes — from a raw stream of aggregated trades. The direction of the trade must come from your own analysis (fundamental, technical). This library answers a narrower question: **is right now a statistically unusual moment in market microstructure?**

Three independent detectors run in parallel. Each produces a score in [0, 1]. The scores are combined into a single `confidence` value that you compare against your threshold.

---

## API

### `detect(historical, recent, confidence?)`

One-shot convenience function. Trains on historical data, then evaluates the recent window. Returns a `DetectionResult`.

```typescript
import { detect } from 'volume-anomaly';
import type { IAggregatedTradeData } from 'volume-anomaly';

const historical: IAggregatedTradeData[] = await getAggregatedTrades('BTCUSDT', 2000);
const recent:     IAggregatedTradeData[] = await getAggregatedTrades('BTCUSDT', 300);

const result = detect(historical, recent, 0.75);
// {
//   anomaly:      true,
//   confidence:   0.81,          // weighted composite score
//   imbalance:    0.72,          // buy-side dominance
//   hawkesLambda: 4.3,           // current intensity (trades/sec)
//   cusumStat:    3.1,           // CUSUM accumulator (σ units)
//   runLength:    2,             // periods since last changepoint
//   signals: [
//     { kind: 'volume_spike',   score: 0.88, meta: { lambda: 4.3, mu: 1.1, branching: 0.61 } },
//     { kind: 'imbalance_shift', score: 0.72, meta: { imbalance: 0.72, absImbalance: 0.72 } },
//     { kind: 'bocpd_changepoint', score: 0.44, meta: { cpProbability: 0.088, runLength: 2 } },
//   ]
// }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `historical` | `IAggregatedTradeData[]` | required | Baseline window for training (≥ 50 trades). Should represent calm, in-control market conditions |
| `recent` | `IAggregatedTradeData[]` | required | Window to evaluate. Typically 100–500 trades |
| `confidence` | `number` | `0.75` | Threshold in (0, 1). `result.anomaly = result.confidence >= confidence` |

**Returns:** `DetectionResult`

```typescript
interface DetectionResult {
  anomaly:      boolean;       // confidence >= threshold
  confidence:   number;        // composite score [0,1]
  signals:      AnomalySignal[]; // which sub-detectors fired
  imbalance:    number;        // buy/sell balance [-1, +1]
  hawkesLambda: number;        // conditional intensity at last trade (trades/sec)
  cusumStat:    number;        // max(S⁺, S⁻) — CUSUM accumulator
  runLength:    number;        // MAP run length — periods since last changepoint
}
```

---

### `new VolumeAnomalyDetector(config?)`

Stateful class. Use when you need to re-use fitted models across multiple `detect()` calls without re-training, or when you want to tune individual model parameters.

```typescript
import { VolumeAnomalyDetector } from 'volume-anomaly';

const detector = new VolumeAnomalyDetector({
  windowSize:   50,          // trades per imbalance window
  hazardLambda: 200,         // expected periods between changepoints
  cusumKSigmas: 0.5,         // CUSUM slack k = 0.5 · σ
  cusumHSigmas: 4,           // CUSUM alarm h = 4 · σ
  scoreWeights: [0.4, 0.3, 0.3], // [Hawkes, CUSUM, BOCPD]
});

detector.train(historicalTrades);
const result = detector.detect(recentTrades, 0.75);

// Inspect fitted parameters for debugging:
const { hawkesParams, cusumParams, bocpdPrior } = detector.trainedModels!;
```

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windowSize` | `number` | `50` | Number of trades per rolling imbalance window. Smaller = more reactive to local shifts, larger = smoother signal |
| `hazardLambda` | `number` | `200` | Expected number of windows between changepoints (BOCPD hazard rate H = 1/λ). Set lower for more frequent regime changes |
| `cusumKSigmas` | `number` | `0.5` | CUSUM allowable slack k in σ units. Controls sensitivity: lower k = faster response but more false positives |
| `cusumHSigmas` | `number` | `4` | CUSUM alarm threshold h in σ units. Higher h = fewer but more confident alarms (ARL₀ ≈ 500–1000 at h = 4–5σ) |
| `scoreWeights` | `[n, n, n]` | `[0.4, 0.3, 0.3]` | Weights for [Hawkes, CUSUM, BOCPD] scores. Must sum to 1 |

---

## `confidence` — how it works (critical difference from garch)

> **This is fundamentally different from `garch`'s `confidence` parameter.**

In `garch`, `confidence` is a two-sided probability passed through `probit` to get a z-score, which then scales a log-normal price corridor: `confidence → z = probit((1+confidence)/2) → P·exp(±z·σ)`. The `confidence` there controls **band width**, not a classification threshold.

Here, `confidence` is a **hard threshold on a composite score**. It has no probabilistic interpretation from a normal distribution. The formula is:

```
score_final = w_H · score_hawkes + w_C · score_cusum + w_B · score_bocpd
anomaly     = score_final >= confidence
```

Each sub-score is mapped independently to [0, 1] through its own non-linear function (sigmoid for Hawkes, linear ratio for CUSUM, amplified probability for BOCPD). The `confidence` you pass is the minimum weighted average you require before calling the moment an anomaly.

**Practical guidance:**

| `confidence` | Sensitivity | False positive rate | Use case |
|-------------|-------------|---------------------|----------|
| `0.5` | Very high | High | Research / signal exploration |
| `0.65` | High | Moderate | Aggressive entries, many signals |
| `0.75` (default) | Balanced | Low | Standard trading use |
| `0.85` | Low | Very low | High-conviction entries only |
| `0.95` | Very low | Near zero | Stress testing / rare events |

Unlike `garch`, raising `confidence` does not widen a corridor — it raises the bar for all three detectors simultaneously. A result with `confidence = 0.74` at a threshold of `0.75` means the moment is borderline: borderline intense arrival rate, borderline imbalance shift, or borderline regime change — but not all three firing hard.

---

## Input data

```typescript
interface IAggregatedTradeData {
  id:           string;   // Binance aggTradeId
  price:        number;   // Execution price
  qty:          number;   // Trade size (base asset)
  timestamp:    number;   // Unix milliseconds
  isBuyerMaker: boolean;  // true  → sell aggressor (taker sold into bid)
                          // false → buy  aggressor (taker bought ask)
}
```

**`isBuyerMaker` semantics** — this field trips people up. In a limit order book, the maker posts the resting order. When `isBuyerMaker = true`, the buyer is the maker (passive bid), meaning the *seller* was the aggressive taker. From an order flow perspective: `isBuyerMaker = true` → **sell aggression**, `isBuyerMaker = false` → **buy aggression**.

---

## Math

### Volume Imbalance

The first quantity derived from raw trades — used as the input series for both CUSUM and BOCPD.

```
buyVol  = Σ qty_i   for all i where isBuyerMaker = false
sellVol = Σ qty_i   for all i where isBuyerMaker = true
imbalance = (buyVol - sellVol) / (buyVol + sellVol)
```

Result is in [-1, +1]. `+1` = all volume is buy-aggressor. `-1` = all volume is sell-aggressor. `0` = balanced. Empty input returns `0`.

The key design choice: **weighted by qty, not trade count**. A single 50 BTC block trade counts 50× more than a 1 BTC retail fill. This makes the imbalance measure resistant to spoofing via many tiny orders.

Each call to `detect()` computes the imbalance for the full recent window (directional signal, returned as `result.imbalance`), and also a **rolling imbalance series** with `windowSize`-trade sliding windows (used as input to CUSUM and BOCPD):

```
rolling[i] = imbalance(trades[i - windowSize : i])   for i = windowSize, ..., n
```

The rolling series converts a raw trade stream into a time series of local imbalances, making it suitable for the sequential change detectors below.

---

### Hawkes Process

**Model:** Univariate Hawkes process with exponential kernel (Hawkes, 1971).

```
λ(t) = μ + Σ_{tᵢ < t} α · exp(−β · (t − tᵢ))
```

- **μ > 0** — background intensity (trades/sec in quiet market)
- **α > 0** — excitation magnitude (how much each trade boosts future intensity)
- **β > 0** — decay rate (how fast the excitement fades)
- **α < β** — stationarity constraint (subcritical process)

The model says: trades arrive at a baseline rate μ, but each arriving trade triggers a burst of additional arrivals that decays exponentially. This captures the empirical clustering of order flow — a large trade tends to be followed by a flurry of reactive orders.

**Unconditional mean:**

```
E[λ] = μ / (1 − α/β)
```

The ratio `α/β` is the **branching ratio** — the expected number of secondary events triggered by one primary event. At `α/β = 0.6`, each trade triggers on average 0.6 follow-on trades. At `α/β → 1` the process becomes supercritical (explosive).

**Log-likelihood — O(n) recursive computation:**

The naive LL is O(n²) since each event sees all previous events. Ogata (1988) reduced this to O(n) with the recursive compensator trick:

```
A(0) = 0
A(i) = exp(−β · (tᵢ − tᵢ₋₁)) · (1 + A(i−1))

λ(tᵢ) = μ + α · A(i)

ln L = −μ·T − (α/β)·Σᵢ(1 − exp(−β·(T−tᵢ))) + Σᵢ ln λ(tᵢ)
```

The second term is the **compensator** — the expected number of events the model predicts over [0,T]. The third term is the data likelihood under the fitted intensity. Maximum likelihood balances these two.

**Fitting via Nelder-Mead MLE:**

Parameters [μ, α, β] are estimated by minimising the negative log-likelihood. Starting point:

```
T    = t_n − t_0          (observation window length)
μ₀   = 0.5 · n/T          (half the empirical rate)
α₀   = 0.4 · n/T          (40% excitation share)
β₀   = n/T                (rate = decay)
```

Constraints enforced inside the objective: if `μ ≤ 0` or `α ≤ 0` or `β ≤ 0` or `α ≥ β`, return `1e10` (hard wall). This keeps the optimizer in the subcritical stationary region.

**Anomaly score — sigmoid centred at 2× unconditional mean:**

```
meanLambda = μ / (1 − α/β)
score_hawkes = σ(2 · (λ(t_last) / meanLambda − 2))
            = 1 / (1 + exp(−2 · (λ / meanLambda − 2)))
```

The sigmoid is centred so that `score = 0.5` when `λ = 2 · meanLambda` (twice the expected quiet rate). At `λ = meanLambda` (normal) the score is `1/(1+e⁴) ≈ 0.018`. At `λ = 3·meanLambda` the score is `1/(1+e⁻²) ≈ 0.88`.

If the fitted branching ratio `α/β ≥ 1`, the process is supercritical and the score is clamped to `1` unconditionally.

**What the Hawkes score captures:** arrival rate acceleration. A flash crash preceded by 10× normal trade frequency will drive a high Hawkes score even before price moves significantly. It is blind to the *direction* of trades — only their timing.

---

### CUSUM — Sequential Change Detection

**Model:** Cumulative Sum Control Chart (Page, 1954). Applied to the rolling imbalance series.

The input series is `xₜ = |imbalance(window_t)|` — absolute imbalance magnitude. The two-sided CUSUM tracks:

```
S⁺ₜ = max(0,  S⁺_{t-1} + xₜ − μ₀ − k)
S⁻ₜ = max(0,  S⁻_{t-1} − xₜ + μ₀ − k)
```

- **μ₀** — in-control mean of |imbalance| (from training data)
- **k** — allowable slack = `cusumKSigmas · σ₀` (filters out noise below k)
- **h** — alarm threshold = `cusumHSigmas · σ₀` (fires when S ≥ h)

`S⁺` accumulates evidence that the series has shifted **above** its historical mean. `S⁻` accumulates evidence of a downward shift. Both are reset to zero after an alarm.

**Why absolute imbalance:** using |imbalance| instead of signed imbalance means both extreme buy pressure and extreme sell pressure register as anomalies. The direction comes from `result.imbalance` (signed), not from CUSUM.

**Training — parameter estimation:**

```
μ₀   = mean(|imbalance|)       over the training window
σ₀²  = var(|imbalance|)        sample variance
k    = cusumKSigmas · σ₀       (default 0.5σ)
h    = cusumHSigmas · σ₀       (default 4σ)
```

**Average run length under H₀ (ARL₀):** the expected number of observations before a false alarm. For Gaussian series, the approximate relationship between h, k and ARL₀ is:

```
ARL₀ ≈ exp(2·k·h / σ₀²)
```

At the defaults `k = 0.5σ`, `h = 4σ`: `ARL₀ ≈ exp(4) ≈ 55`. Raising `cusumHSigmas` to 5 gives `ARL₀ ≈ exp(5) ≈ 148`. Raising to 6 gives `ARL₀ ≈ e⁶ ≈ 403`. This is why the default of `4` fires relatively quickly after a shift, while `6` gives rare, high-confidence alarms only.

**CUSUM anomaly score:**

```
score_cusum = min(max(S⁺, S⁻) / h,  1)
```

Linear: `0` at no accumulation, `1` at the alarm boundary. The score reaches `1` exactly when CUSUM would fire, then resets. Between resets it linearly grows with accumulated evidence.

**Important nuance — auto-reset on alarm:** when the alarm fires, both `S⁺` and `S⁻` are reset to zero and observation counter `n` resets. The score thus drops to 0 right after a confirmed alarm. This means `score_cusum = 1` is momentary: the next observation after a fire starts fresh. If you see `cusumStat` close to `h` but not quite there, the moment is building.

---

### BOCPD — Bayesian Online Changepoint Detection

**Model:** Adams & MacKay (2007). Computes the posterior distribution over **run lengths** — the number of observations since the last changepoint — updated online with each new observation.

The fundamental recursion (from the paper):

```
P(rₜ | x₁:ₜ) ∝ Σ_{rₜ₋₁} P(xₜ | rₜ₋₁, x_{t-r:t}) · P(rₜ | rₜ₋₁) · P(rₜ₋₁ | x₁:ₜ₋₁)
```

There are two possible transitions at each step:
- **Growth** `P(rₜ = r+1 | rₜ₋₁ = r) = 1 − H` — run continues, length grows
- **Changepoint** `P(rₜ = 0 | rₜ₋₁ = r) = H` — run resets to zero

The hazard function `H = 1 / hazardLambda` is constant (geometric / memoryless gaps between changepoints). `hazardLambda = 200` means the model expects a changepoint every 200 windows on average.

**Underlying observation model — Normal-Gamma conjugate:**

Each run-length hypothesis r maintains a separate Normal-Gamma posterior over the mean and precision of `{xₜ₋ᵣ, ..., xₜ}`. The predictive probability of a new observation given run length r is a Student-t:

```
p(xₜ | rₜ₋₁ = r, x_{t-r:t}) = Student-t(2αN, μN, βN(κN+1)/(αN·κN))
```

Where the posterior hyperparameters after n = r observations are updated from prior (μ₀, κ₀, α₀, β₀):

```
κN = κ₀ + n
αN = α₀ + n/2
μN = (κ₀·μ₀ + n·x̄) / κN
βN = β₀ + 0.5·M₂ + κ₀·n·(x̄ − μ₀)² / (2·κN)
```

M₂ = Σ(xᵢ − x̄)² is maintained via Welford's online algorithm (numerically stable, O(1) per update).

**Prior hyperparameters — derived from training:**

```
μ₀    = mean(|imbalance|)       training window mean
κ₀    = 1                       weak prior (1 pseudo-observation)
α₀    = 1                       weak prior on precision
β₀    = var(|imbalance|)        training window variance
```

A prior with `κ₀ = 1` means the prior contributes the equivalent of one observation. After 10 real observations the likelihood dominates; the prior is only important for brand-new run-length hypotheses (segments just started).

**Log-domain computation for numerical stability:**

All probabilities are maintained as log-probabilities. The changepoint mass is accumulated via log-sum-exp:

```
log P(rₜ = 0) = logSumExp over r of [log P(rₜ₋₁ = r) + log p(xₜ | r) + log H]
log P(rₜ = r+1) = log P(rₜ₋₁ = r) + log p(xₜ | r) + log(1-H)
```

After each update, all log-probs are normalised by subtracting `logSumExp(all)`. This keeps the distribution proper and prevents underflow.

**Pruning:** hypotheses with `log P(rₜ = r) < −30` (probability < `1e-13`) are discarded. This bounds memory and computation: in practice the active set stays O(1) to O(few hundred) even after thousands of observations.

**Diagnostics returned:**

- `cpProbability = P(rₜ = 0 | x₁:ₜ)` — probability that a changepoint occurred exactly at observation t
- `mapRunLength` — the run length with highest posterior probability (MAP estimator)

**BOCPD anomaly score:**

```
score_bocpd = min(cpProbability · 5,  1)
```

The ×5 amplifier compensates for the fact that `cpProbability` is typically small even during genuine changepoints (the model distributes probability across many run-length hypotheses). Empirically, `cpProbability = 0.1–0.2` at a real changepoint, so ×5 maps that to `0.5–1.0`. The `min(..., 1)` caps it.

**What BOCPD captures:** regime shifts — moments where the *distribution* of imbalance itself changes, not just its current level. A market transitioning from choppy balanced flow to sustained directional flow will register here, often before the imbalance crosses an absolute threshold.

---

### Composite Score and Signal Thresholds

The three scores are linearly combined:

```
confidence_score = w_H · score_hawkes + w_C · score_cusum + w_B · score_bocpd
                 = 0.4 · score_hawkes + 0.3 · score_cusum + 0.3 · score_bocpd  (defaults)
```

The `anomaly` flag is:

```
anomaly = confidence_score >= confidence_threshold
```

**Signals** are individual detector firings appended to `result.signals` when:

| Signal kind | Fires when | Score attached |
|-------------|-----------|----------------|
| `volume_spike` | `score_hawkes > 0.5` | Hawkes sigmoid value |
| `imbalance_shift` | `\|imbalance\| > 0.4` | Raw absolute imbalance |
| `cusum_alarm` | `score_cusum > 0.7` | Linear ratio S/h |
| `bocpd_changepoint` | `score_bocpd > 0.3` | Amplified CP probability |

A signal in `result.signals` does **not** require `result.anomaly = true`. You can have partial signals (e.g. only Hawkes firing) with `confidence_score < threshold`. The signals let you understand *why* the composite score is what it is.

**Score combination example** with defaults `[0.4, 0.3, 0.3]`:

| scenario | Hawkes | CUSUM | BOCPD | composite | anomaly at 0.75 |
|----------|--------|-------|-------|-----------|-----------------|
| quiet market | 0.02 | 0.05 | 0.03 | 0.033 | ✗ |
| arrival spike only | 0.90 | 0.10 | 0.05 | 0.39 | ✗ |
| spike + imbalance | 0.90 | 0.75 | 0.20 | 0.645 | ✗ |
| all three fire | 0.90 | 0.90 | 0.90 | 0.90 | ✓ |
| CUSUM + BOCPD, calm arrivals | 0.15 | 0.95 | 0.95 | 0.63 | ✗ |

This shows a key design property: **no single detector can exceed the threshold alone at default weights**, since max single contribution is `0.4 · 1.0 = 0.40`. At least two detectors must agree. Raise Hawkes weight to `0.8` if you want arrival rate alone to be sufficient.

---

## Math internals — exported for testing

All internal functions are accessible via the `math` export path for unit testing and experimentation:

```typescript
import {
  // Hawkes
  volumeImbalance,
  hawkesLogLikelihood,
  hawkesFit,
  hawkesLambda,
  hawkesAnomalyScore,

  // CUSUM
  cusumFit,
  cusumUpdate,       // returns { state, alarm }
  cusumInitState,
  cusumAnomalyScore,
  cusumBatch,

  // BOCPD
  bocpdUpdate,
  bocpdInitState,
  bocpdAnomalyScore,
  bocpdBatch,
  defaultPrior,

  // Optimizer
  nelderMead,
} from 'volume-anomaly/math';
```

### `hawkesLogLikelihood(timestamps, params)`

Raw Ogata log-likelihood. `timestamps` must be sorted ascending in **seconds**. Returns `-Infinity` if params are invalid (μ ≤ 0, α ≤ 0, β ≤ 0). Does **not** enforce `α < β` — that constraint is applied only inside `hawkesFit`.

### `hawkesFit(timestamps)`

Returns `{ params, logLik, stationarity, converged }`. `stationarity = α/β`. If `timestamps.length < 10`, returns a flat Poisson fallback with `converged: false`.

### `hawkesLambda(t, timestamps, params)`

Evaluates `λ(t)` at a specific time given a history of prior events. All timestamps must be `< t`.

### `cusumUpdate(state, x, params)`

Pure function. Returns `{ state: CusumState, alarm: boolean }`. Does **not** mutate the input state.

### `bocpdUpdate(state, x, prior, hazardLambda?)`

Returns `{ state, mapRunLength, cpProbability }`. The returned state contains pruned log-probability arrays. Pass `hazardLambda` in windows (same units as your observation index).

### `nelderMead(f, x0, options?)`

Nelder-Mead simplex optimizer. Used internally by Hawkes fitting. Returns `{ x, fx, iters, converged }`.

---

## Optimization

Hawkes parameter estimation uses **single-start Nelder-Mead** (3 parameters: μ, α, β). The starting point is empirically derived from the data (empirical rate as μ₀) so a single start is typically sufficient. CUSUM and BOCPD use closed-form estimation (sample mean/variance for CUSUM, Welford update for BOCPD) — no optimizer needed.

| Component | Parameters | Optimization | Complexity |
|-----------|-----------|--------------|------------|
| Hawkes | 3 (μ, α, β) | Nelder-Mead, 1 start, 1000 iter | O(n) per LL eval |
| CUSUM | 4 (μ₀, σ₀, k, h) | Closed-form (sample stats) | O(n) |
| BOCPD | 4 (μ₀, κ₀, α₀, β₀) | Closed-form (sample stats) | O(n) per update, pruned |

BOCPD update is technically O(r_max) where r_max is the number of surviving run-length hypotheses. The pruning threshold `log P < −30` keeps this bounded in practice (typically < 300 hypotheses even after 10,000 observations).

---

## Training data guidance

| Trades in historical window | Quality |
|----------------------------|---------|
| < 50 | Rejected (throws) |
| 50–200 | Minimal — CUSUM μ₀/σ₀ estimates unreliable |
| 200–500 | Adequate for typical use |
| 500–2000 | Good — stable Hawkes MLE, representative CUSUM baseline |
| 2000+ | Best — especially important for low-activity pairs |

The training window should represent **normal, in-control market conditions**. Fitting on data that already contains anomalies will inflate the baseline and reduce sensitivity. If your market opens with a gap or major event, use a calmer historical window from the previous session.

**`windowSize` guidance** — the number of trades per rolling imbalance step:

| `windowSize` | Trades in window | Sensitivity | Lag |
|-------------|-----------------|-------------|-----|
| 20 | 20 | Very high | Low |
| 50 (default) | 50 | Balanced | Moderate |
| 100 | 100 | Lower | Higher |
| 200 | 200 | Low | High |

On high-volume pairs (BTC/USDT perpetual), 50 trades may span only 1–2 seconds. On low-volume pairs, 50 trades may span minutes. Calibrate to the effective time scale that matters for your entry.

---

## Integration with garch

Typical workflow combining both libraries:

```typescript
import { predict }                  from 'garch';
import { detect, VolumeAnomalyDetector } from 'volume-anomaly';

// 1. Train volume detector once per session
const detector = new VolumeAnomalyDetector({ windowSize: 50 });
detector.train(await getAggregatedTrades('BTCUSDT', 2000));

// 2. On each new candle close:
async function onCandle(candles: Candle[], recentTrades: IAggregatedTradeData[]) {
  // Volume anomaly — entry timing
  const vol = detector.detect(recentTrades, 0.75);

  if (!vol.anomaly) return; // not an anomalous moment

  // Directional filter from your fundamental analysis
  const isBuySignal = vol.imbalance > 0.3 && myFundamentalBullish();
  if (!isBuySignal) return;

  // garch — TP/SL sizing
  const { upperPrice, lowerPrice, sigma, reliable } = predict(candles, '15m');
  if (!reliable) return;

  const entry = currentPrice;
  const tp    = upperPrice;         // +1σ target (default confidence 0.6827)
  const sl    = lowerPrice;         // -1σ stop

  // Or use 95% VaR for wider stop:
  const { lowerPrice: sl95 } = predict(candles, '15m', undefined, 0.95);

  placeOrder({ entry, tp, sl: sl95 });
}
```

`garch.predict` answers: *how big is the next normal move?* `volume-anomaly.detect` answers: *is this moment abnormal enough to act on?* They are complementary and independent.

---

## Tests

**59 tests** across **4 test files**. All passing.

| File | Tests | Coverage |
|------|-------|----------|
| `hawkes.test.ts` | 20 | Imbalance formula, LL computation, MLE fitting, λ evaluation and decay, anomaly score monotonicity and supercritical clamp |
| `cusum.test.ts` | 15 | Parameter estimation, state update (pure function), accumulation, alarm + reset, score range, batch detection |
| `bocpd.test.ts` | 11 | Init state, t increment, probability normalisation, run length growth in stable regime, CP spike on distribution shift, immutability, batch changepoint detection |
| `detector.test.ts` | 13 | Pre-train guard, isTrained flag, minimum training size, DetectionResult fields, confidence range, empty window, signal score range, functional API determinism |

```bash
npm test
```

---

## License

MIT