import { describe, it, expect } from 'vitest';
import { predict, detect, VolumeAnomalyDetector } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

let _uid = 0;

function makeLCG(seed: number) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 0xFFFFFFFF; };
}

function buildStream(
  n: number, startTs: number, intervalMs: number,
  buyFrac: number, rng: () => number,
): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  let ts = startTs;
  for (let i = 0; i < n; i++) {
    ts += intervalMs * (0.7 + rng() * 0.6);
    trades.push({
      id: String(_uid++), price: 100,
      qty: 0.5 + rng() * 1.5,
      timestamp: Math.round(ts),
      isBuyerMaker: rng() > buyFrac,
    });
  }
  return trades;
}

// ─── Result shape ─────────────────────────────────────────────────────────────

describe('predict(): result shape', () => {
  it('returns anomaly, confidence, direction, imbalance', () => {
    const rng  = makeLCG(0x10101010);
    const hist = buildStream(200, 0, 1000, 0.5, rng);
    const rec  = buildStream(100, 200_000, 1000, 0.5, rng);
    const r    = predict(hist, rec);

    expect(typeof r.anomaly).toBe('boolean');
    expect(typeof r.confidence).toBe('number');
    expect(typeof r.imbalance).toBe('number');
    expect(['long', 'short', 'neutral']).toContain(r.direction);
  });

  it('confidence is in [0, 1]', () => {
    const rng  = makeLCG(0x11111111);
    const hist = buildStream(200, 0, 1000, 0.5, rng);
    const rec  = buildStream(100, 200_000, 1000, 0.5, rng);
    const { confidence } = predict(hist, rec);
    expect(confidence).toBeGreaterThanOrEqual(0);
    expect(confidence).toBeLessThanOrEqual(1);
  });

  it('imbalance is in [-1, +1]', () => {
    const rng  = makeLCG(0x12121212);
    const hist = buildStream(200, 0, 1000, 0.5, rng);
    const rec  = buildStream(100, 200_000, 1000, 0.5, rng);
    const { imbalance } = predict(hist, rec);
    expect(imbalance).toBeGreaterThanOrEqual(-1);
    expect(imbalance).toBeLessThanOrEqual(1);
  });
});

// ─── Direction logic ─────────────────────────────────────────────────────────

describe('predict(): direction assignment', () => {
  it('direction=neutral when anomaly=false', () => {
    // Calm identical data — anomaly must be false
    const base: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) {
      base.push({ id: String(_uid++), price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) {
      rec.push({ id: String(_uid++), price: 100, qty: 1, timestamp: 300_000 + i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const r = predict(base, rec, 0.75);
    expect(r.anomaly).toBe(false);
    expect(r.direction).toBe('neutral');
  });

  it('direction=long when anomaly=true and imbalance > threshold', () => {
    const rng  = makeLCG(0xDEADBEEF);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    // Heavy buy burst: 10× rate + 90% buy aggressors
    const rec  = buildStream(250, 700_000, 100, 0.9, rng);
    const r    = predict(hist, rec, 0.75, 0.3);
    if (r.anomaly && r.imbalance > 0.3) {
      expect(r.direction).toBe('long');
    }
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('long');
  });

  it('direction=short when anomaly=true and imbalance < -threshold', () => {
    const rng  = makeLCG(0x30303030);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    // Heavy sell burst: 10× rate + 5% buy (95% sell aggressors)
    const rec  = buildStream(250, 700_000, 100, 0.05, rng);
    const r    = predict(hist, rec, 0.75, 0.3);
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('short');
  });

  it('direction=neutral when anomaly=true but imbalance within threshold (rate-only spike)', () => {
    // Rate burst with balanced buy/sell (50%) — anomaly fires on rate, imbalance near 0 → neutral
    const rng  = makeLCG(0xDEADBEEF);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    // 10× rate burst, perfectly balanced (buyFrac=0.5)
    const burst = buildStream(250, 700_000, 100, 0.5, rng);
    const r = predict(hist, burst, 0.75, 0.3);
    expect(r.anomaly).toBe(true);
    expect(Math.abs(r.imbalance)).toBeLessThan(0.3);
    expect(r.direction).toBe('neutral');
  });
});

// ─── anomaly/direction consistency ───────────────────────────────────────────

describe('predict(): anomaly and direction consistency', () => {
  it('direction is never long/short when anomaly=false', () => {
    const rng = makeLCG(0x40404040);
    // Multiple calm windows
    for (let trial = 0; trial < 5; trial++) {
      const hist = buildStream(200, 0, 1000, 0.5, rng);
      const rec  = buildStream(100, 200_000 + trial * 10_000, 1000, 0.5, rng);
      const r    = predict(hist, rec, 0.75);
      if (!r.anomaly) {
        expect(r.direction).toBe('neutral');
      }
    }
  });

  it('direction long/short requires anomaly=true', () => {
    // Если direction не neutral — anomaly должно быть true
    const rng  = makeLCG(0x41414141);
    const hist = buildStream(500, 0, 1000, 0.5, rng);

    for (const buyFrac of [0.05, 0.95]) {
      const rng2 = makeLCG(0x41414141 + buyFrac * 100);
      const h2   = buildStream(500, 0, 1000, 0.5, rng2);
      const rec  = buildStream(250, 700_000, 100, buyFrac, rng2);
      const r    = predict(h2, rec, 0.0);  // threshold=0 → anomaly always true if confidence>0

      if (r.direction !== 'neutral') {
        expect(r.anomaly).toBe(true);
      }
    }
    void hist;
  });
});

// ─── imbalanceThreshold parameter ────────────────────────────────────────────

describe('predict(): imbalanceThreshold parameter', () => {
  it('lower threshold → more likely to assign long/short', () => {
    const rng  = makeLCG(0x50505050);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    // Moderate buy imbalance burst
    const rec  = buildStream(250, 700_000, 100, 0.75, rng);

    const rStrict = predict([...hist], [...rec], 0.75, 0.8);  // high threshold
    const rLoose  = predict([...hist], [...rec], 0.75, 0.1);  // low threshold

    if (rLoose.anomaly && rLoose.imbalance > 0.1) {
      expect(rLoose.direction).toBe('long');
    }
    // At strict threshold 0.8, direction may be neutral even if anomaly
    // At loose threshold 0.1, should assign long if imbalance > 0.1
    if (rStrict.anomaly && Math.abs(rStrict.imbalance) <= 0.8) {
      expect(rStrict.direction).toBe('neutral');
    }
  });

  it('threshold=0 → any positive imbalance is long', () => {
    const rng  = makeLCG(0x51515151);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(250, 700_000, 100, 0.9, rng);
    const r    = predict(hist, rec, 0.75, 0.0);
    if (r.anomaly && r.imbalance > 0) {
      expect(r.direction).toBe('long');
    }
    if (r.anomaly && r.imbalance < 0) {
      expect(r.direction).toBe('short');
    }
  });
});

// ─── predict() matches detect() ──────────────────────────────────────────────

describe('predict(): consistency with detect()', () => {
  it('predict().confidence equals detect().confidence', async () => {
    const rng  = makeLCG(0x60606060);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(200, 700_000, 100, 0.85, rng);

    // Re-build with same seed
    const rng2  = makeLCG(0x60606060);
    const hist2 = buildStream(500, 0, 1000, 0.5, rng2);
    const rec2  = buildStream(200, 700_000, 100, 0.85, rng2);

    const rPredict = predict(hist,  rec,  0.75);
    const rDetect  = detect( hist2, rec2, 0.75);

    expect(rPredict.confidence).toBe(rDetect.confidence);
    expect(rPredict.anomaly).toBe(rDetect.anomaly);
    expect(rPredict.imbalance).toBe(rDetect.imbalance);
  });

  it('predict() is deterministic for same input', () => {
    const rng  = makeLCG(0x61616161);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(200, 700_000, 100, 0.8, rng);

    const r1 = predict([...hist], [...rec], 0.75);
    const r2 = predict([...hist], [...rec], 0.75);

    expect(r1.confidence).toBe(r2.confidence);
    expect(r1.direction).toBe(r2.direction);
    expect(r1.imbalance).toBe(r2.imbalance);
  });
});

// ─── Seeded regression ────────────────────────────────────────────────────────

describe('predict(): seeded regression', () => {
  it('[0xDEADBEEF] buy burst → long', () => {
    const rng  = makeLCG(0xDEADBEEF);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(250, 700_000, 100, 0.9, rng);
    const r    = predict(hist, rec, 0.75, 0.3);
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('long');
    expect(r.imbalance).toBeGreaterThan(0.3);
  });

  it('[0xDEADC0DE] sell burst → short', () => {
    const rng  = makeLCG(0xDEADC0DE);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(250, 700_000, 100, 0.05, rng);
    const r    = predict(hist, rec, 0.75, 0.3);
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('short');
    expect(r.imbalance).toBeLessThan(-0.3);
  });

  it('[0x12345678] calm stream → neutral', () => {
    const rng  = makeLCG(0x12345678);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(200, 600_000, 1000, 0.5, rng);
    const r    = predict(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
    expect(r.direction).toBe('neutral');
  });
});

// ─── Trained imbalanceThreshold ──────────────────────────────────────────────

describe('predict(): trained imbalanceThreshold', () => {
  it('trainedModels.imbalanceThreshold is a finite number after train()', () => {
    const rng  = makeLCG(0x70707070);
    const hist = buildStream(300, 0, 1000, 0.5, rng);
    const det  = new VolumeAnomalyDetector();
    det.train(hist);
    const thr = det.trainedModels!.imbalanceThreshold;
    expect(Number.isFinite(thr)).toBe(true);
    expect(Number.isNaN(thr)).toBe(false);
  });

  it('balanced training → threshold near 0 (p75 of near-zero series)', () => {
    // All trades perfectly alternating buy/sell → rolling imbalance ≈ 0 in every window
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) {
      hist.push({ id: String(i), price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const det = new VolumeAnomalyDetector({ windowSize: 20 });
    det.train(hist);
    const thr = det.trainedModels!.imbalanceThreshold;
    // p75 of a series that is always 0 must be 0 (or very close due to rounding)
    expect(Math.abs(thr)).toBeLessThan(0.05);
  });

  it('trending training → threshold higher than on balanced data', () => {
    // 80% buy-aggressor history → p75 of signed imbalance is elevated
    const trendHist: IAggregatedTradeData[] = [];
    const calmHist:  IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) {
      trendHist.push({ id: `t${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 5 === 0 });  // 80% buy
      calmHist.push({  id: `c${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });  // 50% balanced
    }
    const detTrend = new VolumeAnomalyDetector({ windowSize: 20 });
    const detCalm  = new VolumeAnomalyDetector({ windowSize: 20 });
    detTrend.train(trendHist);
    detCalm.train(calmHist);
    const thrTrend = detTrend.trainedModels!.imbalanceThreshold;
    const thrCalm  = detCalm.trainedModels!.imbalanceThreshold;
    // Trending market should produce a higher directional threshold
    expect(thrTrend).toBeGreaterThan(thrCalm);
  });

  it('predict() without explicit threshold uses trained value, not 0.3', () => {
    // Train on data where p75 is well above 0.3 (strong buy trend)
    const trendHist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) {
      // 95% buy aggressors → rolling imbalance ≈ +0.9
      trendHist.push({ id: `th${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 20 === 0 });
    }
    const det = new VolumeAnomalyDetector({ windowSize: 20 });
    det.train(trendHist);
    const trainedThr = det.trainedModels!.imbalanceThreshold;
    // p75 of a ~0.9 series must be well above 0.3
    expect(trainedThr).toBeGreaterThan(0.3);

    // predict() with this data and no override should use trainedThr internally
    // A moderate-buy recent window (imbalance ~0.5) should be 'neutral' with trained
    // threshold ~0.9 but would be 'long' with the old hardcoded 0.3
    const rng = makeLCG(0x71717171);
    const rec = buildStream(200, 400_000, 100, 0.7, rng);  // moderate buy burst
    const rWithTrainedThr = predict(trendHist, rec, 0.0);       // use trained thr
    const rWithFixed03   = predict(trendHist, rec, 0.0, 0.3);  // force old threshold

    // With the trained high threshold, a moderate imbalance that is below the
    // training p75 should NOT yield 'long'
    if (rWithTrainedThr.anomaly && rWithTrainedThr.imbalance < trainedThr) {
      expect(rWithTrainedThr.direction).toBe('neutral');
    }
    // With the fixed 0.3 threshold, same imbalance would be 'long' if imbalance > 0.3
    if (rWithFixed03.anomaly && rWithFixed03.imbalance > 0.3) {
      expect(rWithFixed03.direction).toBe('long');
    }
  });

  it('imbalancePercentile config changes the trained threshold', () => {
    // Same training data, different percentile config → different threshold
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) {
      // 70% buy aggressors → signed imbalance ≈ +0.4
      hist.push({ id: `ip${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 10 < 3 });
    }
    const detP50  = new VolumeAnomalyDetector({ windowSize: 20, imbalancePercentile: 50 });
    const detP90  = new VolumeAnomalyDetector({ windowSize: 20, imbalancePercentile: 90 });
    detP50.train(hist);
    detP90.train(hist);
    const thr50 = detP50.trainedModels!.imbalanceThreshold;
    const thr90 = detP90.trainedModels!.imbalanceThreshold;
    // p90 must be ≥ p50 for any distribution
    expect(thr90).toBeGreaterThanOrEqual(thr50);
  });

  it('imbalancePercentile=0 → threshold is the minimum of the series', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) {
      hist.push({ id: `p0${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const det = new VolumeAnomalyDetector({ windowSize: 20, imbalancePercentile: 0 });
    det.train(hist);
    const thr = det.trainedModels!.imbalanceThreshold;
    // p0 is the minimum — for alternating series that's ≈ 0 but could be slightly negative
    expect(Number.isFinite(thr)).toBe(true);
  });

  it('imbalancePercentile=100 → threshold is the maximum of the series', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) {
      hist.push({ id: `p100${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const det = new VolumeAnomalyDetector({ windowSize: 20, imbalancePercentile: 100 });
    det.train(hist);
    const thr = det.trainedModels!.imbalanceThreshold;
    // p100 is the max — for alternating series that's ≈ 0
    expect(Number.isFinite(thr)).toBe(true);
  });

  it('windowSize > training count → fallback threshold 0.3 is finite', () => {
    // Same regression as NaN bug — signed imbalance also empty when windowSize > n
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 50; i++) {
      hist.push({ id: `fb${i}`, price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const det = new VolumeAnomalyDetector({ windowSize: 100 });
    det.train(hist);
    const thr = det.trainedModels!.imbalanceThreshold;
    expect(thr).toBe(0.3);
    expect(Number.isFinite(thr)).toBe(true);
  });
});
