<p align="center">
  <img src="https://github.com/tripolskypetr/volume-anomaly/raw/master/assets/logo.png" height="115px" alt="garch" />
</p>

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

The library detects **abnormal surges in trade flow** — sudden acceleration of arrivals and volume waves — from a raw stream of aggregated trades. The direction of the trade must come from your own analysis (fundamental, technical). This library answers a narrower question: **is right now a statistically unusual moment in market microstructure?**

The primary statistic is a **self-calibrated robust z-score of trade rate and volume rate**: the detector measures "trades per 5 s / per 30 s" and "quantity per 5 s / per 30 s" in the detection window and scores the peaks against the median/MAD of the same statistic over the training window — *how many robust σ above the recent typical level*. Thresholds are calibrated on a full day of real BTCUSDT aggTrades (1.49 M trades, see `test/eval.test.ts`): at the default `confidence = 0.75` the detector catches ≈ 92 % of locally-strong volume anomalies (robust z ≥ 8 vs the trailing hour) and ≈ 95 % of anomaly events, with ≈ 2.5 % false alarms on normal 30-second windows.

Two secondary detectors (CUSUM and BOCPD on order-flow imbalance) are computed and reported in `scores`/`signals`, but by default carry **zero weight** in the combined confidence: on real data they track flow-regime shifts — a different phenomenon — and any additive weight on them measurably dilutes recall. Re-weight via `scoreWeights` if your use case targets flow shifts specifically.

---

## API

### `detect(historical, recent, confidence?)`

One-shot convenience function. Trains on historical data, then evaluates the recent window. Returns a `DetectionResult`.

```typescript
import { detect } from 'volume-anomaly';
import type { IAggregatedTradeData } from 'volume-anomaly';

// historical: 15–30 MINUTES of trades (time span matters more than count —
// the baseline must cover enough market time to define "typical")
const historical: IAggregatedTradeData[] = await getAggregatedTrades('BTCUSDT', 10_000);
const recent:     IAggregatedTradeData[] = await getAggregatedTrades('BTCUSDT', 300);

const result = detect(historical, recent, 0.75);
// {
//   anomaly:      true,
//   confidence:   0.98,          // combined score (volume channel by default)
//   scores:       { hawkes: 0.98, cusum: 0.61, bocpd: 0.00 },  // raw sub-scores
//   stats:        { zRate: 22.3, zVol: 104.1, zRateSlow: 25.3, zVolSlow: 87.2, lambdaRatio: 3.1 },
//   imbalance:    0.72,          // buy-side dominance
//   hawkesLambda: 61.8,          // peak intensity (trades/sec)
//   cusumStat:    3.1,           // CUSUM accumulator
//   runLength:    2,             // periods since last changepoint
//   signals: [
//     { kind: 'volume_spike',    score: 0.98, meta: { lambda: 61.8, zRate: 22.3, zVol: 104.1, ... } },
//     { kind: 'imbalance_shift', score: 0.72, meta: { imbalance: 0.72, absImbalance: 0.72 } },
//   ]
// }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `historical` | `IAggregatedTradeData[]` | required | Baseline window for training. **Must span 15–30 minutes of market time** (code minimum: 50 trades). A count-based baseline ("last 500 trades") adapts its duration to market pace and masks the very burst being detected |
| `recent` | `IAggregatedTradeData[]` | required | Window to evaluate. Typically 100–500 trades; must span at least `rateHorizonSec` (5 s) for the fast channel to engage |
| `confidence` | `number` | `0.75` | Threshold in (0, 1). `result.anomaly = result.confidence >= confidence` |

**Returns:** `DetectionResult`

```typescript
interface DetectionResult {
  anomaly:      boolean;       // confidence >= threshold
  confidence:   number;        // composite score [0,1]
  scores:       { hawkes: number; cusum: number; bocpd: number }; // raw sub-scores
  stats:        { zRate: number; zVol: number;                    // robust z of peak
                  zRateSlow: number; zVolSlow: number;            // rolling rates
                  lambdaRatio: number };                          // peak λ vs training
  signals:      AnomalySignal[]; // which sub-detectors fired
  imbalance:    number;        // buy/sell balance [-1, +1]
  hawkesLambda: number;        // conditional intensity at last trade (trades/sec)
  cusumStat:    number;        // max(S⁺, S⁻) — CUSUM accumulator
  runLength:    number;        // MAP run length — periods since last changepoint
}
```

---

### `predict(historical, recent, confidence?, imbalanceThreshold?)`

One-shot convenience function. Wraps `detect()` and adds a directional signal derived from the **burst-local** order flow.

```typescript
import { predict } from 'volume-anomaly';

const result = predict(historical, recent, 0.75, 0.3);
// {
//   anomaly:        true,
//   confidence:     0.81,
//   direction:      'long',    // 'long' | 'short' | 'neutral'
//   imbalance:      0.42,      // full-window flow balance
//   burstImbalance: 0.72,      // flow balance INSIDE the peak burst window
// }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `historical` | `IAggregatedTradeData[]` | required | Baseline window for training (≥ 50 trades) |
| `recent` | `IAggregatedTradeData[]` | required | Window to evaluate |
| `confidence` | `number` | `0.75` | Anomaly threshold [0,1] |
| `imbalanceThreshold` | `number` | *(trained)* | Override the trained directional threshold. Omit to use the value derived automatically from training data (p75 of the rolling signed imbalance series) |

**Direction logic:**

```
thr = max(0, imbalanceThreshold)  (if provided explicitly)
    = max(0, detector.trainedModels.imbalanceThreshold)  (otherwise — p75 from training)

direction = 'long'    if anomaly && burstImbalance >  +thr
direction = 'short'   if anomaly && burstImbalance < −thr
direction = 'neutral' otherwise (no anomaly, or balanced flow at the burst)
```

Direction reads the **burst-local** imbalance — the order flow inside the rolling window that actually produced the anomaly score — not the full-window average, which dilutes a burst's onset with surrounding two-way flow (measured on real data: a pure sell burst reads −0.9 at the burst but only −0.42 over the full bucket). The burst imbalance is additionally shrunk toward neutral by effective sample size (Kish n_eff over qty), so a three-trade or single-whale "burst" cannot fake directional conviction.

On a neutral/balanced market `thr` will be near 0 (most windows have close-to-zero imbalance, p75 ≈ 0.1–0.2). On a trending market the p75 shifts upward with the trend, so the bar for `direction=long` rises accordingly — preventing chronic false long signals during a bull run where sustained buy imbalance is normal, not anomalous. The threshold is clamped at zero, so `'long'` always implies buy-side burst flow and `'short'` sell-side.

**Returns:** `PredictionResult`

```typescript
interface PredictionResult {
  anomaly:        boolean;    // confidence >= threshold
  confidence:     number;     // composite score [0,1]
  direction:      Direction;  // 'long' | 'short' | 'neutral'
  imbalance:      number;     // full-window buy/sell balance [-1, +1]
  burstImbalance: number;     // buy/sell balance inside the peak burst window
}
```

**Practical usage with `getAggregatedTrades` from `backtest-kit`:**

> ⚠️ **Never pass the same trades in both `historical` and `recent`.** Training calibrates the baseline. If `recent` overlaps with `historical`, any anomaly in that period is absorbed into the baseline — the detector learns to treat it as normal and misses it. Always slice so that `recent` starts where `historical` ends.

```typescript
import { predict } from 'volume-anomaly';
import type { IAggregatedTradeData } from 'volume-anomaly';

// Your data-fetching function — returns the last `limit` trades, oldest first:
declare function getAggregatedTrades(
  symbol: string,
  limit:  number,
): Promise<IAggregatedTradeData[]>;

// ── One-shot (single API call, zero overlap) ──────────────────────────────────

const N_train  = 1200;  // calibration window
const N_detect = 200;   // window to evaluate

const all        = await getAggregatedTrades('BTCUSDT', N_train + N_detect);
const historical = all.slice(0, N_train);  // older 1200 trades — baseline
const recent     = all.slice(N_train);     // newest 200 trades — no overlap

const result = predict(historical, recent, 0.75);
// {
//   anomaly:    true,
//   confidence: 0.83,
//   direction:  'long',   // 'long' | 'short' | 'neutral'
//   imbalance:  0.61,
// }

if (result.anomaly) {
  console.log(`direction=${result.direction}  confidence=${result.confidence.toFixed(2)}`);
}
```

`predict()` trains a fresh detector on every call. For continuous monitoring (many `detect()` calls from one trained model) use `VolumeAnomalyDetector` directly — see the class API below.

---

### `new VolumeAnomalyDetector(config?)`

Stateful class. Use when you need to re-use fitted models across multiple `detect()` calls without re-training, or when you want to tune individual model parameters.

```typescript
import { VolumeAnomalyDetector } from 'volume-anomaly';

const detector = new VolumeAnomalyDetector({
  windowSize:     50,        // trades per imbalance window (CUSUM/BOCPD)
  rateHorizonSec: 5,         // fast rate/volume horizon (seconds)
  slowHorizonSec: 30,        // slow rate/volume horizon (seconds)
  hazardLambda:   200,       // expected periods between changepoints
  cusumKSigmas:   0.5,       // CUSUM slack k = 0.5 · σ
  cusumHSigmas:   5,         // CUSUM alarm h floor = 5 · σ
  scoreWeights:   [1, 0, 0], // [volume channel, CUSUM, BOCPD]
});

detector.train(historicalTrades);
const result = detector.detect(recentTrades, 0.75);

// Inspect fitted parameters for debugging:
const { hawkesParams, cusumParams, bocpdPrior } = detector.trainedModels!;
```

**Config parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `windowSize` | `number` | `50` | Number of trades per rolling imbalance window (CUSUM/BOCPD). Smaller = more reactive to local shifts, larger = smoother signal |
| `rateHorizonSec` | `number` | auto (≥ 5) | Fast time horizon for the volume channel: "trades per horizon" / "qty per horizon". Catches brief intense bursts. When omitted, chosen on the fly: max(5 s, 25 × median inter-trade gap), capped by the training span — sparse instruments automatically get a horizon that carries a meaningful rate. Pass a value to pin it |
| `slowHorizonSec` | `number` | auto (≥ 30) | Slow horizon for the same statistic. Catches sustained volume waves spread too evenly to concentrate at the fast horizon. Auto: 6 × fast horizon, floored at 30 s, capped by the training span |
| `hazardLambda` | `number` | `200` | Expected number of windows between changepoints (BOCPD hazard rate H = 1/λ). Set lower for more frequent regime changes |
| `cusumKSigmas` | `number` | `0.5` | CUSUM allowable slack k in σ units. Controls sensitivity: lower k = faster response but more false positives |
| `cusumHSigmas` | `number` | `5` | CUSUM alarm threshold floor in σ units; the effective `h` is self-calibrated as `max(hSigmas·σ, 2 × max training excursion)` because the rolling series is heavily autocorrelated |
| `scoreWeights` | `[n, n, n]` | `[1, 0, 0]` | Weights for [volume channel, CUSUM, BOCPD]. Must sum to 1. The default gives the flow-shift detectors zero weight — on real data they dilute volume-anomaly recall (see `test/eval.test.ts`); their scores remain available in `result.scores` |
| `imbalancePercentile` | `number` | `75` | Percentile of the training rolling signed imbalance used as the directional threshold. p75 = direction fires only when imbalance exceeds the 75th percentile of the training distribution. Clamped at zero before use, so `'long'` always implies imbalance > 0 and `'short'` implies imbalance < 0 |

Every numeric option is validated at construction — a bad value (e.g. `hazardLambda: 0.5`, which would silently collapse the BOCPD state, or a non-integer `windowSize`) throws immediately with the option name and received value instead of producing a miscalibrated detector. Two input mistakes that never crash but silently rescale every time horizon 1000× are also rejected: timestamps that look like Unix **seconds** or **microseconds** (they must be milliseconds), and a percent-style `confidence` (`75` instead of `0.75`).

### `toJSON()` / `VolumeAnomalyDetector.fromJSON(snapshot)`

Trained models are plain finite numbers, so a detector serializes losslessly: train once (e.g. in a worker on a schedule), ship the snapshot anywhere, and detect without re-training.

```typescript
const detector = new VolumeAnomalyDetector();
detector.train(historicalTrades);

const snapshot = JSON.stringify(detector);          // uses toJSON() automatically

// … elsewhere (another process, a file, Redis, …):
const restored = VolumeAnomalyDetector.fromJSON(snapshot); // string or parsed object
const result   = restored.detect(recentTrades, 0.75);      // identical to the original
```

`fromJSON` validates the snapshot structurally — config re-passes constructor validation, and every numeric leaf of the models must be finite (JSON silently turns `NaN`/`Infinity` into `null`, so a corrupted snapshot fails loudly here, with the offending path in the message, rather than as `NaN` confidence inside `detect()`). Auto-chosen horizons stay auto after restore: a restored detector re-derives them if you call `train()` again.

---

## `confidence` — how it works (critical difference from garch)

> **This is fundamentally different from `garch`'s `confidence` parameter.**

In `garch`, `confidence` is a two-sided probability passed through `probit` to get a z-score, which then scales a log-normal price corridor: `confidence → z = probit((1+confidence)/2) → P·exp(±z·σ)`. The `confidence` there controls **band width**, not a classification threshold.

Here, `confidence` is a **hard threshold on a composite score**. The formula is:

```
score_final = w_H · score_volume + w_C · score_cusum + w_B · score_bocpd
            = score_volume                                    (default weights [1,0,0])
anomaly     = score_final >= confidence
```

The volume-channel score is built in three steps:

1. **Multi-scale scan.** The peak rolling arrival rate and volume rate are measured at every horizon in the trained family (fast ≈ 5 s, a geometric-mean mid scale, slow ≈ 30 s, and an extended scale up to 3× slow when the training span allows) and standardized per channel: `t = (z − c − u·spanShift) / u`, where `z` is the robust z of the peak against the training median/MAD.
2. **Cross-type corroboration (Stouffer).** The best arrival-rate `t` and the best volume `t` combine as `max(t_lead, (t_lead + max(t_other, 0)) / √2)`: a moderate rate excess AND a moderate volume excess together outrank either alone. Scales of the *same* type never corroborate each other — they see the same wiggle at different bandwidths.
3. **Rational sigmoid.** `score = 0.5 + 0.5·t/(1+|t|)`: 0.5 at the calibrated level, 0.75 one tail unit beyond it, approaching 1 harmonically — the top of the scale keeps *ranking* extreme events instead of saturating into ties, which matters when the score is used predictively.

The level `c` is chosen **on the fly at `train()` time**: the P85 of the peak-z values the baseline itself produced in alert-sized stretches, made robust to events inside the training window by a Gumbel fit to the LEFT quantiles of those maxima (contamination inflates only the upper quantiles; the level is `min(empirical P85, Gumbel-extrapolated P85)`), and floored by per-channel universal levels found by dense brute-force search on a full day of real BTCUSDT trades (`test/eval.test.ts`): 14 for arrival-rate channels, 6.5 for volume channels (volume z separates anomalies much more cleanly, so it earns a lower bar). The tail unit `u = 5.5` is universal in z-space. On a typical baseline `confidence = 0.75` fires at volume z ≥ 12 or rate z ≥ 19.5; a hotter/noisier baseline automatically raises the bar.

**Practical guidance** (measured on the real-data benchmark, 30 s buckets):

| `confidence` | Fires at (t units beyond level) | Recall (strong anomalies) | False alarms (normal buckets) |
|-------------|----------------------------------|---------------------------|-------------------------------|
| `0.5` | t ≥ 0 (vol z ≥ 6.5 / rate z ≥ 14) | higher | higher |
| `0.75` (default) | t ≥ 1 (vol z ≥ 12 / rate z ≥ 19.5) | ≈ 96 % (events ≈ 97 %) | ≈ 3.2 % |
| `0.9` | t ≥ 4 (vol z ≥ 28.5 / rate z ≥ 36) | lower | lower |

A result with `confidence = 0.74` at a threshold of `0.75` means the moment is borderline — the combined statistic sits just under one tail unit beyond the calibrated level.

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

**Peak intensity over the detection window:**

Instead of evaluating λ at the last event only, the detector takes the **maximum** λ(tᵢ) seen at any event in the window using the same O(n) recursive trick:

```
A(0) = 0
A(i) = exp(−β · (tᵢ − tᵢ₋₁)) · (1 + A(i−1))
λ(tᵢ) = μ + α · A(i)

peakLambda = max over i of λ(tᵢ)
```

This ensures that a burst occurring in the middle of the window is detected even after the kernel has decayed by the last event.

**How the detector actually scores (self-calibrated z-channels):**

The theoretical baselines above (`E[λ]`, fitted `μ`) are exported in the math API
(`hawkesAnomalyScore`) but are **not what `VolumeAnomalyDetector` uses** — on real
trade streams rates fluctuate several-fold between adjacent windows of a
perfectly normal market, so "2× the fitted μ" is routine noise (measured: 28 %
false alarms on real BTCUSDT).  Instead the detector scores robust z of rolling
rate and volume-rate statistics against baselines measured on the training
window itself:

```
for horizon h ∈ { rateHorizonSec (5 s), slowHorizonSec (30 s) }:
  rate_i    = trades in (tᵢ − h, tᵢ] / h        (one sample per trade)
  volRate_i = qty    in (tᵢ − h, tᵢ] / h

  z = (max over detection window − median over training) /
      (1.4826 · max(MAD_training, 0.1 · median))

score_volume = max over 4 channels of  σ((z − 12) · 0.4)
```

The fitted Hawkes model still provides `hawkesLambda` / `lambdaRatio`
diagnostics (peak conditional intensity vs the training peak), reported in
`result.stats` and signal meta for transparency.

**What the volume channel captures:** arrival-rate acceleration (many trades
per second) *and* volume waves (few huge trades) at both brief and sustained
horizons. It is blind to the *direction* of trades — direction comes from
`imbalance`.

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
h    = cusumHSigmas · σ₀       (default 5σ)
```

**Average run length under H₀ (ARL₀):** the expected number of observations before a false alarm. For Gaussian series, the approximate relationship between h, k and ARL₀ is:

```
ARL₀ ≈ exp(2·k·h / σ₀²)
```

At the defaults `k = 0.5σ`, `h = 5σ`: `ARL₀ ≈ exp(5) ≈ 148`. Raising `cusumHSigmas` to 6 gives `ARL₀ ≈ e⁶ ≈ 403`. Lowering to 4 gives `ARL₀ ≈ exp(4) ≈ 55` — fires quickly but with more false positives.

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

**BOCPD anomaly score — relative run-length drop:**

`cpProbability` is approximately equal to the constant prior hazard `H = 1/hazardLambda` and does **not** spike at genuine changepoints — it is dominated by the prior, not the data. The real signal is `mapRunLength`: in a stable process it grows monotonically; a changepoint resets it to near zero.

The score measures the *relative drop* from the previous step:

```
drop = clamp((prevRunLength − mapRunLength) / prevRunLength,  0, 1)

score_bocpd = 1 / (1 + exp(−(drop − 0.5) · 8))
```

Typical values:
- drop = 0 (run length grew — stable)     → score ≈ 0.018
- drop = 0.5 (run length halved)          → score = 0.50
- drop ≥ 0.9 (e.g. 90 → 1 after reset)   → score ≈ 0.98

The sigmoid is centred at `drop = 0.5` with steepness 8. The score is taken as the **peak over the entire detection window**, so changepoints that occurred mid-window are still captured.

**What BOCPD captures:** regime shifts — moments where the *distribution* of imbalance itself changes, not just its current level. A market transitioning from choppy balanced flow to sustained directional flow will register here, often before the imbalance crosses an absolute threshold.

---

### Composite Score and Signal Thresholds

The volume channel is the score that matters by default:

```
score_volume = max over channels of  σ((z − c_ch − u·spanShift) · ln3 / u)
  channels:  z of peak "trades per fast horizon"   (fast arrival bursts)
             z of peak "qty per fast horizon"      (fast volume bursts, block trades)
             z of peak "trades per slow horizon"   (sustained arrival waves)
             z of peak "qty per slow horizon"      (sustained volume waves)

  z    = (peak_detect − median_train) / (1.4826 · max(MAD_train, 0.1·median_train))
  c_ch = max(P85 of the peak-z the baseline itself produced in slow-horizon
             stretches, universal per-channel floor:
             14 for rate channels, 6.5 for volume channels)  → score 0.5
  u    = 5.5 (universal tail unit); c_ch + u                  → score 0.75
  spanShift = log10(max(1, detectionSpan / slowHorizon))      (longer window ⇒
              more independent looks at the max ⇒ higher bar)

confidence_score = w_H · score_volume + w_C · score_cusum + w_B · score_bocpd
                 = score_volume                          (default weights [1, 0, 0])
anomaly          = confidence_score >= confidence_threshold
```

Everything instrument-specific is chosen **on the fly at `train()` time**:

- **Horizons** — scaled up from the median inter-trade gap for sparse streams (config values are floors; pass explicit values to pin them).
- **Yardstick (median/MAD)** — measured per channel on the training window.
- **Score level `c_ch`** — the baseline's own null distribution of window maxima, floored by the universal mapping validated on a full day of real BTCUSDT trades. A baseline that itself contains recurring bursts absorbs them into its null quantiles, so a repeat of a known pattern scores ≈ 0.5 rather than 1.0.

There are no theoretical thresholds ("2× the fitted μ", "5σ under i.i.d.") — those were tested on real data and misfire badly (28 % false alarms), because real trade streams are heavily autocorrelated and heavy-tailed. The remaining fixed anchors (per-channel floor levels and the tail unit) live in standardized z-space, were found by a dense brute-force search over the full mapping space on the benchmark (44 100 configurations; the chosen point sits on a stable Pareto plateau), and act only as a sensitivity *limit* — the data can make the detector stricter, never more trigger-happy.

**Signals** are individual detector firings appended to `result.signals` when:

| Signal kind | Fires when | Score attached |
|-------------|-----------|----------------|
| `volume_spike` | `score_volume > 0.5` | max of the four z-channel sigmoids |
| `imbalance_shift` | `\|imbalance\| > 0.4` | Raw absolute imbalance |
| `cusum_alarm` | `score_cusum > 0.7` | Linear ratio max(S⁺, S⁻) / h (h self-calibrated) |
| `bocpd_changepoint` | `score_bocpd > 0.3` | Run-length-drop score above the training noise floor |

A signal in `result.signals` does **not** require `result.anomaly = true`. Signals (and the raw `result.scores` / `result.stats`) let you understand *why* the composite score is what it is.

**Design property:** at default weights the anomaly decision is purely "is the trade rate or volume rate many robust σ above what this instrument did in the last 15–30 minutes". CUSUM/BOCPD track order-flow *imbalance* regime shifts — a different phenomenon that correlates poorly with volume anomalies on real data (measured: any additive weight on them reduces recall and adds false alarms). Give them weight only if flow shifts are what you actually want to detect.

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
  hawkesPeakLambda,  // max λ(tᵢ) over window — used by the detector
  hawkesAnomalyScore,

  // CUSUM
  cusumFit,
  cusumUpdate,       // returns { state, alarm, preResetState }
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

### `hawkesPeakLambda(timestamps, params)`

Returns the **maximum** `λ(tᵢ)` over all events in `timestamps` using the O(n) recursive A(i) trick. This is what the detector uses internally instead of `hawkesLambda` — a burst that decayed by the last event is still captured. `hawkesLambda` evaluates at a single point; `hawkesPeakLambda` scans the full window.

### `cusumUpdate(state, x, params)`

Pure function. Returns `{ state: CusumState, alarm: boolean, preResetState: CusumState }`. Does **not** mutate the input state. `preResetState` holds the accumulator values *before* the alarm reset — use it for scoring, since `state.sPos/sNeg` are zeroed when `alarm = true`.

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

## Training and detection window sizes

### `train()` — historical window

**The time span matters more than the trade count.** The volume-channel baselines (median/MAD of rolling rates) are meaningful only if the training window covers enough market time to define "typical":

| `historical` spans | Volume-channel baselines | Notes |
|--------------------|--------------------------|-------|
| < 1 minute | Unreliable | The slow horizon (30 s) has almost no independent samples; a hot pre-burst minute masks the burst itself |
| 1–5 minutes | Weak | Usable for the fast horizon only |
| **15–30 minutes** | **Good** | **Recommended.** This is what the calibration in `test/eval.test.ts` uses |
| Multiple hours | Robust but stale | Beware regime staleness — baselines may span very different market conditions |

Never build the baseline as "the last N trades": a count-based window adapts its *duration* to market pace — right before a burst it covers seconds of already-hot market and the inflated baseline masks the very anomaly being detected, while in a calm period it covers narrow minutes and routine wiggles look anomalous. Feed a **fixed time span** of recent history instead (internally the expensive fits are capped: Hawkes MLE uses the most recent 2 000 trades, the imbalance series the most recent 1 000; the O(n) rate baselines use everything).

`train()` throws below 50 trades. The training window should represent **normal market conditions**: an anomaly inside the baseline inflates the yardstick and reduces sensitivity to similar events.

### `detect()` — recent window

The volume channel needs the recent window to span at least `rateHorizonSec` (5 s default) — shorter windows fall back to a single whole-window rate sample. CUSUM/BOCPD additionally need `trades ≥ windowSize` for their rolling imbalance series; when the window is shorter, score weights are renormalized over the detectors that actually ran, so the volume channel can still reach confidence 1 alone.

| Trades in `recent` | Notes |
|--------------------|-------|
| < 5 s of market time | Fast channel falls back to whole-window rate; slow channel off |
| 100–300 trades | **Recommended** — what the real-data calibration uses |
| 300–1000 trades | Fine; peak-based statistics don't dilute with window growth |

**Rule of thumb:** call `detect()` every few seconds with the last ~300 trades; re-`train()` every few minutes with the trailing 15–30 minutes.

### `windowSize` guidance

| `windowSize` | Sensitivity | Lag | Minimum `train()` | Minimum `detect()` for full signal |
|-------------|-------------|-----|-------------------|-------------------------------------|
| 20 | Very high | Low | 50 trades (code minimum) | 40 trades |
| 50 (default) | Balanced | Moderate | 100 trades (recommended) | 100 trades |
| 100 | Lower | Higher | 200 trades | 200 trades |
| 200 | Low | High | 400 trades | 400 trades |

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

**735 tests** across **18 test files**. All passing. 100% statement/function/line coverage, 98.72% branch (two unreachable `??` guards).

| File | Tests | What is covered |
|------|-------|-----------------|
| `hawkes.test.ts` | 20 | Imbalance formula, LL computation, MLE fitting, λ evaluation and decay, anomaly score monotonicity and supercritical clamp |
| `cusum.test.ts` | 15 | Parameter estimation, state update (pure function), accumulation, alarm + reset, score range, batch detection |
| `bocpd.test.ts` | 13 | Init state, update, probability normalisation, run length growth in stable regime, CP spike on distribution shift, immutability, batch |
| `detector.test.ts` | 20 | Pre-train guard, isTrained flag, minimum training size, DetectionResult fields, confidence range, empty window, signal score range |
| `detect.test.ts` | 36 | End-to-end anomaly detection, confidence thresholds, signal composition, edge inputs |
| `seeded.test.ts` | 67 | Deterministic seeded scenarios covering long/short/neutral bursts across parameter space |
| `predict.test.ts` | 24 | Direction assignment, trained imbalanceThreshold, imbalancePercentile config, trending vs balanced threshold |
| `invariants.test.ts` | 29 | Monotonicity, score bounds, immutability, score weight validation |
| `adversarial.test.ts` | 58 | Adversarial inputs: NaN propagation, extreme values, Inf timestamps, zero-qty trades |
| `falsepositive.test.ts` | 18 | Scenarios that must NOT trigger: gradual drift, HFT clusters, trending market, whale trades, overnight gaps |
| `edgecases.test.ts` | 80 | Boundary conditions, signal threshold exact values (strict >), detect < windowSize bypass, train twice, cusumBatch multiple alarms |
| `realdata.test.ts` | 23 | Real BTCUSDT-2025-03-01 data: 4 spike windows + 1 calm baseline |
| `robustness.test.ts` | 66 | Mathematical invariants: range/symmetry/monotonicity for all functions, BOCPD normalisation Σexp(lp) ≤ 1, 100-case property-based detector test |
| `extreme.test.ts` | 52 | Stuck-at-extremum: hazardLambda edge cases, μ = 0, extreme β, degenerate Nelder-Mead, Welford drift, β₀ = 0 |
| `newextreme.test.ts` | 58 | NaN propagation in CUSUM/BOCPD, hawkesAnomalyScore extremes, cusumAnomalyScore h = NaN, prevRL = Inf/NaN, kappa0 = 0 |
| `thirdextreme.test.ts` | 74 | hazardLambda = 0 collapse, β ≤ 0, hawkesFit n = 0/T = 0, hawkesPeakLambda n = 1/β = 0, volumeImbalance NaN qty, cusumFit NaN filter |
| `fourthextreme.test.ts` | 63 | hawkesAnomalyScore NaN peak + valid params, cusumUpdate NaN params, cusumAnomalyScore NaN state, bocpdUpdate beta0 = Inf, Infinity qty |
| `perf.test.ts` | 19 | Latency P95 bounds, throughput (detect(200) ≥ 800/s), scaling ratios, stability over 500 sequential calls |

```bash
npm test
```

---

## License

MIT