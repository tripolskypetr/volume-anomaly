/**
 * serialization.test.ts — toJSON() / VolumeAnomalyDetector.fromJSON().
 *
 * Contract: train once → JSON.stringify → fromJSON elsewhere → detect()
 * behaves identically to the original detector.  Corrupted snapshots must
 * fail loudly at fromJSON() time, never as NaN confidence inside detect().
 */

import { describe, it, expect } from 'vitest';
import { VolumeAnomalyDetector } from '../src/index.js';
import type { IAggregatedTradeData, DetectorSnapshot } from '../src/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

let _id = 0;

function makeTrade(timestamp: number, qty: number, isBuyerMaker: boolean): IAggregatedTradeData {
  return { id: String(_id++), price: 100, qty, timestamp, isBuyerMaker };
}

/** Deterministic mixed stream: variable qty, mostly-alternating sides */
function makeStream(n: number, startTs: number, intervalMs: number): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  for (let i = 0; i < n; i++) {
    trades.push(makeTrade(startTs + i * intervalMs, 1 + (i % 5) * 0.3, i % 3 !== 0));
  }
  return trades;
}

function trainedDetector(config = {}): VolumeAnomalyDetector {
  const det = new VolumeAnomalyDetector(config);
  det.train(makeStream(500, 1_700_000_000_000, 1000));
  return det;
}

const RECENT      = makeStream(200, 1_700_000_600_000, 100);   // 10× baseline rate
const RECENT_CALM = makeStream(200, 1_700_000_600_000, 1000);  // baseline rate

// ─── Round trip ───────────────────────────────────────────────────────────────

describe('toJSON → fromJSON round trip', () => {
  it('restored detector produces bit-identical detect() results', () => {
    const original = trainedDetector();
    const restored = VolumeAnomalyDetector.fromJSON(JSON.parse(JSON.stringify(original)));

    for (const rec of [RECENT, RECENT_CALM]) {
      const a = original.detect(rec, 0.75);
      const b = restored.detect(rec, 0.75);
      expect(b.confidence).toBe(a.confidence);
      expect(b.anomaly).toBe(a.anomaly);
      expect(b.imbalance).toBe(a.imbalance);
      expect(b.hawkesLambda).toBe(a.hawkesLambda);
      expect(b.scores).toEqual(a.scores);
      expect(b.stats).toEqual(a.stats);
    }
  });

  it('accepts a JSON string directly', () => {
    const original = trainedDetector();
    const restored = VolumeAnomalyDetector.fromJSON(JSON.stringify(original));
    expect(restored.isTrained).toBe(true);
    expect(restored.detect(RECENT, 0.75).confidence)
      .toBe(original.detect(RECENT, 0.75).confidence);
  });

  it('trained models survive the round trip field-for-field', () => {
    const original = trainedDetector();
    const restored = VolumeAnomalyDetector.fromJSON(JSON.stringify(original));
    expect(restored.trainedModels).toEqual(original.trainedModels);
  });

  it('non-default config survives the round trip', () => {
    const original = trainedDetector({ windowSize: 20, scoreWeights: [0.5, 0.25, 0.25] as [number, number, number] });
    const restored = VolumeAnomalyDetector.fromJSON(JSON.stringify(original));
    const a = original.detect(RECENT, 0.5);
    const b = restored.detect(RECENT, 0.5);
    expect(b.confidence).toBe(a.confidence);
    expect(b.scores).toEqual(a.scores);
  });

  it('untrained detector round-trips as untrained', () => {
    const restored = VolumeAnomalyDetector.fromJSON(JSON.stringify(new VolumeAnomalyDetector()));
    expect(restored.isTrained).toBe(false);
    expect(() => restored.detect(RECENT)).toThrow('train()');
  });

  it('JSON.stringify(detector) uses toJSON automatically', () => {
    const snapshot = JSON.parse(JSON.stringify(trainedDetector())) as DetectorSnapshot;
    expect(snapshot.version).toBe(1);
    expect(snapshot.models).not.toBeNull();
    expect(snapshot.config.windowSize).toBe(50);
  });
});

// ─── Isolation ────────────────────────────────────────────────────────────────

describe('snapshot isolation (pure-function discipline)', () => {
  it('mutating the snapshot does not poison the source detector', () => {
    const det  = trainedDetector();
    const snap = det.toJSON();
    snap.models!.lambdaBaseline = -1;
    snap.models!.hawkesParams.mu = NaN;
    expect(det.trainedModels!.lambdaBaseline).toBeGreaterThan(0);
    expect(Number.isFinite(det.trainedModels!.hawkesParams.mu)).toBe(true);
  });

  it('mutating the snapshot after fromJSON does not poison the restored detector', () => {
    const snap = trainedDetector().toJSON();
    const det  = VolumeAnomalyDetector.fromJSON(snap);
    snap.models!.cusumParams.h = NaN;
    expect(Number.isFinite(det.trainedModels!.cusumParams.h)).toBe(true);
  });
});

// ─── Horizon pinning semantics ────────────────────────────────────────────────

describe('explicit-horizon flags survive restore', () => {
  it('auto horizons stay auto: restored detector re-derives them on retrain', () => {
    const det = trainedDetector(); // no explicit horizons
    expect(det.toJSON().explicitFast).toBe(false);
    expect(det.toJSON().explicitSlow).toBe(false);

    const restored = VolumeAnomalyDetector.fromJSON(JSON.stringify(det));
    // Sparse stream (60 s gaps) → auto fast horizon must scale up past the
    // 5 s default; a snapshot that wrongly pinned the horizon would keep 5.
    restored.train(makeStream(100, 1_700_000_000_000, 60_000));
    expect(restored.trainedModels!.fastHorizonSec).toBeGreaterThan(5);
  });

  it('explicit horizons stay pinned after restore', () => {
    const det = trainedDetector({ rateHorizonSec: 7 });
    expect(det.toJSON().explicitFast).toBe(true);

    const restored = VolumeAnomalyDetector.fromJSON(JSON.stringify(det));
    restored.train(makeStream(100, 1_700_000_000_000, 60_000)); // sparse
    expect(restored.trainedModels!.fastHorizonSec).toBe(7);
  });
});

// ─── Corrupted snapshots fail loudly ──────────────────────────────────────────

describe('fromJSON rejects corrupted snapshots', () => {
  const validSnap = (): DetectorSnapshot => trainedDetector().toJSON();

  it('null / non-object', () => {
    expect(() => VolumeAnomalyDetector.fromJSON(null as never)).toThrow();
    expect(() => VolumeAnomalyDetector.fromJSON('42')).toThrow();
  });

  it('malformed JSON string', () => {
    expect(() => VolumeAnomalyDetector.fromJSON('{not json')).toThrow();
  });

  it('unsupported version', () => {
    const s = validSnap();
    (s as { version: number }).version = 2;
    expect(() => VolumeAnomalyDetector.fromJSON(s)).toThrow('version');
  });

  it('missing config', () => {
    const s = validSnap();
    delete (s as Partial<DetectorSnapshot>).config;
    expect(() => VolumeAnomalyDetector.fromJSON(s)).toThrow('config');
  });

  it('missing top-level model key', () => {
    const s = validSnap();
    delete (s.models as Partial<NonNullable<DetectorSnapshot['models']>>).hawkesParams;
    expect(() => VolumeAnomalyDetector.fromJSON(s)).toThrow('hawkesParams');
  });

  it('NaN smuggled into a model leaf', () => {
    const s = validSnap();
    s.models!.cusumParams.h = NaN;
    expect(() => VolumeAnomalyDetector.fromJSON(s)).toThrow('non-finite');
  });

  it('Infinity → null JSON corruption is caught with a path', () => {
    // JSON.stringify(Infinity) === 'null' — simulate a snapshot written by a
    // buggy producer.
    const s = JSON.parse(JSON.stringify(validSnap())) as DetectorSnapshot;
    (s.models!.rateStats.fast as { med: number | null }).med = null;
    expect(() => VolumeAnomalyDetector.fromJSON(s)).toThrow('rateStats.fast.med');
  });

  it('invalid config in snapshot is rejected by constructor validation', () => {
    const s = validSnap();
    s.config.windowSize = -5;
    expect(() => VolumeAnomalyDetector.fromJSON(s)).toThrow('windowSize');
  });
});
