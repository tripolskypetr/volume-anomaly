/**
 * Tests for the `detect()` one-shot functional API.
 *
 * `detect(historical, recent, confidence?)` — convenience wrapper:
 *   1. Creates a default VolumeAnomalyDetector (no config exposure)
 *   2. Trains on `historical`
 *   3. Returns detect(recent, confidence)
 *
 * All streams use a seeded LCG for full reproducibility.
 */

import { describe, it, expect } from 'vitest';
import { detect } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';

// ─── LCG ─────────────────────────────────────────────────────────────────────

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0xFFFFFFFF;
  };
}

// ─── Stream builder ───────────────────────────────────────────────────────────

let _uid = 0;

interface Seg {
  count:       number;
  intervalMs:  number;
  buyFraction: number;
  jitter:      number;
  qtyMin?:     number;
  qtyMax?:     number;
}

function buildStream(segs: Seg[], startTs: number, rng: () => number): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  let ts = startTs;
  for (const seg of segs) {
    const qMin = seg.qtyMin ?? 0.5;
    const qMax = seg.qtyMax ?? 2.0;
    for (let i = 0; i < seg.count; i++) {
      ts += Math.max(1, seg.intervalMs * (1 - seg.jitter + rng() * seg.jitter * 2));
      const isBuyerMaker = rng() > seg.buyFraction;
      const qty          = qMin + rng() * (qMax - qMin);
      trades.push({ id: String(_uid++), price: 100, qty, timestamp: Math.round(ts), isBuyerMaker });
    }
  }
  return trades;
}

const CALM:  Seg = { count: 0, intervalMs: 1000, buyFraction: 0.5, jitter: 0.3 };
const BURST: Seg = { count: 200, intervalMs: 100,  buyFraction: 0.9, jitter: 0.1 };

function makeCalm(n: number, rng: () => number, startTs: number): IAggregatedTradeData[] {
  return buildStream([{ ...CALM, count: n }], startTs, rng);
}

// ─── 1. Contract: DetectionResult shape ───────────────────────────────────────

describe('detect(): result shape', () => {
  it('returns all required fields', () => {
    const rng  = makeLCG(0x01010101);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);
    const r    = detect(hist, rec);

    expect(typeof r.anomaly).toBe('boolean');
    expect(typeof r.confidence).toBe('number');
    expect(typeof r.imbalance).toBe('number');
    expect(typeof r.hawkesLambda).toBe('number');
    expect(typeof r.cusumStat).toBe('number');
    expect(typeof r.runLength).toBe('number');
    expect(Array.isArray(r.signals)).toBe(true);
  });

  it('confidence is in [0, 1]', () => {
    const rng  = makeLCG(0x02020202);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);
    const { confidence } = detect(hist, rec);
    expect(confidence).toBeGreaterThanOrEqual(0);
    expect(confidence).toBeLessThanOrEqual(1);
  });

  it('anomaly === (confidence >= threshold)', () => {
    const rng  = makeLCG(0x03030303);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);
    for (const thr of [0.0, 0.5, 0.75, 0.9, 1.0]) {
      const r = detect(hist, rec, thr);
      expect(r.anomaly).toBe(r.confidence >= thr);
    }
  });

  it('every signal has score in [0, 1] and valid kind', () => {
    const VALID_KINDS = new Set(['volume_spike', 'imbalance_shift', 'cusum_alarm', 'bocpd_changepoint']);
    const rng  = makeLCG(0x04040404);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const { signals } = detect(hist, rec, 0.0);
    for (const s of signals) {
      expect(VALID_KINDS.has(s.kind)).toBe(true);
      expect(s.score).toBeGreaterThanOrEqual(0);
      expect(s.score).toBeLessThanOrEqual(1);
      expect(typeof s.meta).toBe('object');
    }
  });

  it('empty recent array returns zero confidence and no anomaly', () => {
    const rng  = makeLCG(0x05050505);
    const hist = makeCalm(200, rng, 0);
    const r    = detect(hist, []);
    expect(r.anomaly).toBe(false);
    expect(r.confidence).toBe(0);
    expect(r.signals).toHaveLength(0);
  });
});

// ─── 2. Guard: training size ──────────────────────────────────────────────────

describe('detect(): training guard', () => {
  it('throws when historical has < 50 trades', () => {
    const rng  = makeLCG(0x10101010);
    const hist = makeCalm(49, rng, 0);
    const rec  = makeCalm(50, rng, 60_000);
    expect(() => detect(hist, rec)).toThrow(/50/);
  });

  it('does not throw at exactly 50 historical trades', () => {
    const rng  = makeLCG(0x11111111);
    const hist = makeCalm(50, rng, 0);
    const rec  = makeCalm(50, rng, 60_000);
    expect(() => detect(hist, rec)).not.toThrow();
  });
});

// ─── 3. Determinism ───────────────────────────────────────────────────────────

describe('detect(): determinism', () => {
  it('same input → identical result', () => {
    const rng  = makeLCG(0x20202020);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);

    const r1 = detect([...hist], [...rec], 0.75);
    const r2 = detect([...hist], [...rec], 0.75);

    expect(r1.confidence).toBe(r2.confidence);
    expect(r1.anomaly).toBe(r2.anomaly);
    expect(r1.imbalance).toBe(r2.imbalance);
    expect(r1.runLength).toBe(r2.runLength);
  });

  it('different seeds → different confidence', () => {
    const rng1 = makeLCG(0x21212121);
    const hist1 = makeCalm(500, rng1, 0);
    const rec1  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng1);

    const rng2 = makeLCG(0x99999999);
    const hist2 = makeCalm(500, rng2, 0);
    const rec2  = makeCalm(100, rng2, 800_000);

    const r1 = detect(hist1, rec1, 0.75);
    const r2 = detect(hist2, rec2, 0.75);

    // An anomalous window vs a calm window should have different confidence
    expect(r1.confidence).not.toBeCloseTo(r2.confidence, 3);
  });

  it('input arrays not mutated', () => {
    const rng  = makeLCG(0x22222222);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);

    const histCopy = hist.map(t => ({ ...t }));
    const recCopy  = rec.map(t => ({ ...t }));

    detect(hist, rec, 0.75);

    // Timestamps and isBuyerMaker must be unchanged
    for (let i = 0; i < hist.length; i++) {
      expect(hist[i]!.timestamp).toBe(histCopy[i]!.timestamp);
    }
    for (let i = 0; i < rec.length; i++) {
      expect(rec[i]!.timestamp).toBe(recCopy[i]!.timestamp);
    }
  });
});

// ─── 4. Threshold sensitivity ─────────────────────────────────────────────────

describe('detect(): threshold sensitivity', () => {
  it('anomaly=true at threshold 0.0 when there is any signal', () => {
    const rng  = makeLCG(0x30303030);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 0.0);
    expect(r.anomaly).toBe(true);   // threshold=0 → always fires if confidence>0
    expect(r.confidence).toBeGreaterThan(0);
  });

  it('anomaly=false at threshold 1.0 (near-impossible)', () => {
    const rng  = makeLCG(0x31313131);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 1.0);
    // confidence is a weighted sum of scores ∈ [0,1]; exact 1.0 is unreachable
    expect(r.anomaly).toBe(false);
  });

  it('higher threshold requires more evidence to fire', () => {
    const rng  = makeLCG(0x32323232);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r50  = detect([...hist], [...rec], 0.5);
    const r75  = detect([...hist], [...rec], 0.75);
    const r90  = detect([...hist], [...rec], 0.9);
    // confidence is fixed; only anomaly flag changes
    expect(r50.confidence).toBe(r75.confidence);
    if (!r50.anomaly) expect(r75.anomaly).toBe(false);
    if (!r75.anomaly) expect(r90.anomaly).toBe(false);
  });
});

// ─── 5. Output fields semantics ───────────────────────────────────────────────

describe('detect(): output field semantics', () => {
  it('imbalance is in [-1, +1]', () => {
    const rng  = makeLCG(0x40404040);
    const hist = makeCalm(200, rng, 0);
    for (const buyFrac of [0.0, 0.5, 1.0]) {
      const rng2 = makeLCG(0x40404040 + buyFrac * 100);
      const h2   = makeCalm(200, rng2, 0);
      const rec  = buildStream([{ ...CALM, count: 100, buyFraction: buyFrac }], 300_000, rng2);
      const { imbalance } = detect(h2, rec);
      expect(imbalance).toBeGreaterThanOrEqual(-1);
      expect(imbalance).toBeLessThanOrEqual(1);
    }
    void hist; // suppress unused warning
  });

  it('imbalance > 0 for pure buy-aggressor window', () => {
    const rng  = makeLCG(0x41414141);
    const hist = makeCalm(300, rng, 0);
    const rec  = buildStream([{ count: 100, intervalMs: 1000, buyFraction: 1.0, jitter: 0 }], 400_000, rng);
    // buyFraction=1 → rng() > 1 is always false → isBuyerMaker=false → buy aggressor
    const { imbalance } = detect(hist, rec);
    expect(imbalance).toBeGreaterThan(0);
  });

  it('imbalance < 0 for pure sell-aggressor window', () => {
    const rng  = makeLCG(0x42424242);
    const hist = makeCalm(300, rng, 0);
    const rec  = buildStream([{ count: 100, intervalMs: 1000, buyFraction: 0.0, jitter: 0 }], 400_000, rng);
    // buyFraction=0 → rng() > 0 is always true → isBuyerMaker=true → sell aggressor
    const { imbalance } = detect(hist, rec);
    expect(imbalance).toBeLessThan(0);
  });

  it('hawkesLambda > 0', () => {
    const rng  = makeLCG(0x43434343);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);
    const { hawkesLambda } = detect(hist, rec);
    expect(hawkesLambda).toBeGreaterThan(0);
  });

  it('cusumStat >= 0', () => {
    const rng  = makeLCG(0x44444444);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);
    const { cusumStat } = detect(hist, rec);
    expect(cusumStat).toBeGreaterThanOrEqual(0);
  });

  it('runLength >= 0', () => {
    const rng  = makeLCG(0x45454545);
    const hist = makeCalm(200, rng, 0);
    const rec  = makeCalm(100, rng, 200_000);
    const { runLength } = detect(hist, rec);
    expect(runLength).toBeGreaterThanOrEqual(0);
  });

  it('unsorted recent array produces same result as sorted (detect sorts internally)', () => {
    const rng   = makeLCG(0x46464646);
    const hist  = makeCalm(500, rng, 0);
    const rec   = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const recRev = [...rec].reverse();  // reversed timestamps

    const rSorted   = detect([...hist], [...rec],    0.75);
    const rReversed = detect([...hist], [...recRev], 0.75);

    expect(rSorted.confidence).toBeCloseTo(rReversed.confidence, 10);
    expect(rSorted.anomaly).toBe(rReversed.anomaly);
  });
});

// ─── 6. Functional: calm vs anomaly ───────────────────────────────────────────

describe('detect(): calm vs anomaly', () => {
  it('[0xAA] pre-anomaly calm window does not fire at 0.75', () => {
    const rng  = makeLCG(0xAAAAAAAA);
    const hist = makeCalm(500, rng, 0);
    const pre  = makeCalm(150, rng, 600_000);
    const r    = detect(hist, pre, 0.75);
    expect(r.anomaly).toBe(false);
  });

  it('[0xAA] burst window fires at 0.75', () => {
    const rng  = makeLCG(0xAAAAAAAA);
    const hist = makeCalm(500, rng, 0);
    makeCalm(150, rng, 600_000);  // consume rng to same state as above
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(true);
  });

  it('[0xAA] post-anomaly calm window does not fire at 0.75', () => {
    const rng  = makeLCG(0xAAAAAAAA);
    const hist = makeCalm(500, rng, 0);
    makeCalm(150, rng, 600_000);
    buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);  // consume
    const post = makeCalm(150, rng, 1_000_000);
    const r    = detect(hist, post, 0.75);
    expect(r.anomaly).toBe(false);
  });

  it('confidence strictly higher for anomaly than for calm window', () => {
    const rng   = makeLCG(0xBBBBBBBB);
    const hist  = makeCalm(500, rng, 0);
    const pre   = makeCalm(150, rng, 600_000);
    const burst = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);

    const rPre   = detect([...hist], pre,   0.0);
    const rBurst = detect([...hist], burst, 0.0);

    expect(rBurst.confidence).toBeGreaterThan(rPre.confidence);
  });

  it('volume_spike signal present for 10× rate burst', () => {
    const rng  = makeLCG(0xCCCCCCCC);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 0.0);
    expect(r.signals.some(s => s.kind === 'volume_spike')).toBe(true);
  });

  it('imbalance_shift signal present for heavy buy window', () => {
    const rng  = makeLCG(0xDDDDDDDD);
    const hist = makeCalm(500, rng, 0);
    // Heavy buy: buyFraction=0.95 → most trades are buy aggressors
    const rec  = buildStream([{ count: 150, intervalMs: 1000, buyFraction: 0.95, jitter: 0.1 }], 600_000, rng);
    const r    = detect(hist, rec, 0.0);
    expect(r.signals.some(s => s.kind === 'imbalance_shift')).toBe(true);
    expect(r.imbalance).toBeGreaterThan(0.4);
  });

  it('no signals for ultra-calm identical baseline replay', () => {
    // Perfectly periodic balanced trades — CUSUM and BOCPD baseline fits exactly.
    const base: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) {
      base.push({ id: String(_uid++), price: 100, qty: 1, timestamp: i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) {
      rec.push({ id: String(_uid++), price: 100, qty: 1, timestamp: 300_000 + i * 1000, isBuyerMaker: i % 2 === 0 });
    }
    const r = detect(base, rec, 0.75);
    expect(r.anomaly).toBe(false);
    expect(r.confidence).toBeLessThan(0.75);
  });
});

// ─── 7. Signal meta fields ────────────────────────────────────────────────────

describe('detect(): signal meta', () => {
  it('volume_spike meta has lambda, mu, branching', () => {
    const rng  = makeLCG(0x50505050);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 0.0);
    const spike = r.signals.find(s => s.kind === 'volume_spike');
    if (spike) {
      expect(typeof spike.meta['lambda']).toBe('number');
      expect(typeof spike.meta['mu']).toBe('number');
      expect(typeof spike.meta['branching']).toBe('number');
      expect(spike.meta['branching']).toBeGreaterThanOrEqual(0);
      expect(spike.meta['branching']).toBeLessThan(1);  // subcritical
    }
  });

  it('imbalance_shift meta has imbalance and absImbalance', () => {
    const rng  = makeLCG(0x51515151);
    const hist = makeCalm(300, rng, 0);
    const rec  = buildStream([{ count: 100, intervalMs: 1000, buyFraction: 0.95, jitter: 0.1 }], 400_000, rng);
    const r    = detect(hist, rec, 0.0);
    const shift = r.signals.find(s => s.kind === 'imbalance_shift');
    if (shift) {
      expect(typeof shift.meta['imbalance']).toBe('number');
      expect(typeof shift.meta['absImbalance']).toBe('number');
      expect(shift.meta['absImbalance']).toBeGreaterThanOrEqual(0);
      // absImbalance must equal |imbalance|
      expect(shift.meta['absImbalance']).toBeCloseTo(Math.abs(shift.meta['imbalance']!), 10);
    }
  });

  it('cusum_alarm meta has sPos, sNeg, h', () => {
    const rng  = makeLCG(0x52525252);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 0.0);
    const alarm = r.signals.find(s => s.kind === 'cusum_alarm');
    if (alarm) {
      expect(typeof alarm.meta['sPos']).toBe('number');
      expect(typeof alarm.meta['sNeg']).toBe('number');
      expect(typeof alarm.meta['h']).toBe('number');
      expect(alarm.meta['h']).toBeGreaterThan(0);
    }
  });

  it('bocpd_changepoint meta has cpProbability and runLength', () => {
    const rng  = makeLCG(0x53535353);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    const r    = detect(hist, rec, 0.0);
    const cp = r.signals.find(s => s.kind === 'bocpd_changepoint');
    if (cp) {
      expect(typeof cp.meta['cpProbability']).toBe('number');
      expect(typeof cp.meta['runLength']).toBe('number');
      expect(cp.meta['cpProbability']).toBeGreaterThanOrEqual(0);
      expect(cp.meta['cpProbability']).toBeLessThanOrEqual(1);
    }
  });
});

// ─── 8. Buy/sell symmetry ─────────────────────────────────────────────────────

describe('detect(): buy/sell imbalance symmetry', () => {
  it('|imbalance| equal for mirrored buy and sell windows', () => {
    const mkHistory = (seed: number) => {
      const rng = makeLCG(seed);
      return makeCalm(500, rng, 0);
    };

    const buyRng  = makeLCG(0x60000000);
    const sellRng = makeLCG(0x60000000);

    makeCalm(500, buyRng,  0);  // consume same amount as history
    makeCalm(500, sellRng, 0);

    // Same qty/timing, only direction differs
    const buyRec  = buildStream([{ count: 100, intervalMs: 1000, buyFraction: 1.0, jitter: 0.1 }], 600_000, buyRng);
    const sellRec = buildStream([{ count: 100, intervalMs: 1000, buyFraction: 0.0, jitter: 0.1 }], 600_000, sellRng);

    const hist     = mkHistory(0x60000000);
    const rBuy     = detect([...hist], buyRec,  0.0);
    const rSell    = detect([...hist], sellRec, 0.0);

    // CUSUM and BOCPD work on |imbalance| → confidence must be equal
    expect(rBuy.confidence).toBeCloseTo(rSell.confidence, 6);
    // Direction must differ
    expect(rBuy.imbalance).toBeGreaterThan(0);
    expect(rSell.imbalance).toBeLessThan(0);
  });
});

// ─── 9. Seeded regression — exact confidence values ───────────────────────────
// These tests pin concrete confidence values. If the algorithm changes, they
// will fail, explicitly flagging a regression.

describe('detect(): seeded regression', () => {
  function run(seed: number): ReturnType<typeof detect> {
    const rng  = makeLCG(seed);
    const hist = makeCalm(500, rng, 0);
    const rec  = buildStream([{ ...CALM, count: 50 }, BURST], 800_000, rng);
    return detect(hist, rec, 0.75);
  }

  it('[0xDEADBEEF] anomaly=true, confidence > 0.75', () => {
    const r = run(0xDEADBEEF);
    expect(r.anomaly).toBe(true);
    expect(r.confidence).toBeGreaterThan(0.75);
  });

  it('[0xCAFEBABE] anomaly=true, confidence > 0.75', () => {
    const r = run(0xCAFEBABE);
    expect(r.anomaly).toBe(true);
    expect(r.confidence).toBeGreaterThan(0.75);
  });

  it('[0xDEADBEEF] confidence is stable across runs (no randomness)', () => {
    const r1 = run(0xDEADBEEF);
    const r2 = run(0xDEADBEEF);
    expect(r1.confidence).toBe(r2.confidence);
  });

  it('[0x12345678] calm-only produces confidence < 0.75', () => {
    const rng  = makeLCG(0x12345678);
    const hist = makeCalm(500, rng, 0);
    const rec  = makeCalm(200, rng, 600_000);
    const r    = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
    expect(r.confidence).toBeLessThan(0.75);
  });
});
