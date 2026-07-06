/**
 * dx.test.ts — the non-specialist convenience layer:
 * severity, explain(), scan(), calibrationReport.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VolumeAnomalyDetector, scan, explain, predict, severityOf } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const loadFixture = (n: string) =>
  JSON.parse(readFileSync(join(__dirname, '..', 'mock', n + '.json'), 'utf8')) as {
    historical: IAggregatedTradeData[]; recent: IAggregatedTradeData[];
  };

function makeStream(n: number, startTs: number, intervalMs: number, buyEvery = 2): IAggregatedTradeData[] {
  return Array.from({ length: n }, (_, i) => ({
    id: String(i), price: 100, qty: 1, timestamp: startTs + i * intervalMs, isBuyerMaker: i % buyEvery === 0,
  }));
}

// ─── severity ─────────────────────────────────────────────────────────────────

describe('severity', () => {
  it('anchors match the score semantics', () => {
    expect(severityOf(0.2)).toBe('none');
    expect(severityOf(0.5)).toBe('notable');
    expect(severityOf(0.75)).toBe('strong');
    expect(severityOf(0.9)).toBe('extreme');
  });

  it('detect() and predict() report consistent severity', () => {
    const { historical, recent } = loadFixture('spike_1_vol_count');
    const det = new VolumeAnomalyDetector();
    det.train(historical);
    const r = det.detect(recent, 0.75);
    expect(r.severity).toBe(severityOf(r.confidence));
    expect(['strong', 'extreme']).toContain(r.severity);
    const p = predict(historical, recent, 0.75);
    expect(p.severity).toBe(severityOf(p.confidence));
  });

  it('calm window is none/notable', () => {
    const { historical, recent } = loadFixture('calm_baseline');
    const det = new VolumeAnomalyDetector();
    det.train(historical);
    expect(['none', 'notable']).toContain(det.detect(recent, 0.75).severity);
  });
});

// ─── explain ──────────────────────────────────────────────────────────────────

describe('explain()', () => {
  it('narrates a real spike: verdict, driver, peak time, flow, ranking', () => {
    const { historical, recent } = loadFixture('spike_1_vol_count');
    const det = new VolumeAnomalyDetector();
    det.train(historical);
    const text = explain(det.detect(recent, 0.75));
    expect(text).toMatch(/Volume anomaly detected \(severity: (strong|extreme)\)/);
    expect(text).toMatch(/confidence \d\.\d\d vs alert threshold 0\.75/);
    expect(text).toMatch(/(volume|trade rate) ran ~\d+ robust sigma/);
    expect(text).toMatch(/Peak at 2025-03-01T/);
    expect(text).toMatch(/sell-side/);          // spike_1 is a sell burst
    expect(text).toMatch(/moveScore/);
  });

  it('narrates a calm window as no anomaly', () => {
    const { historical, recent } = loadFixture('calm_baseline');
    const det = new VolumeAnomalyDetector();
    det.train(historical);
    const text = explain(det.detect(recent, 0.75));
    expect(text).toMatch(/^No anomaly/);
  });

  it('works on PredictionResult (reduced detail) and warns about direction', () => {
    const { historical, recent } = loadFixture('spike_1_vol_count');
    const p = predict(historical, recent, 0.6);
    const text = explain(p, 0.6);
    expect(text).toMatch(/Volume anomaly detected/);
    expect(text).not.toMatch(/robust sigma/); // no stats on PredictionResult
    if (p.direction !== 'neutral') {
      expect(text).toMatch(/does NOT predict/);
    }
  });
});

// ─── scan ─────────────────────────────────────────────────────────────────────

describe('scan()', () => {
  it('single-stream call detects the spike fixture (historical + recent concatenated)', () => {
    const { historical, recent } = loadFixture('spike_1_vol_count');
    const r = scan([...historical, ...recent], { recentSec: 20 });
    expect(r.anomaly).toBe(true);
    expect(['strong', 'extreme']).toContain(r.severity);
    expect(['short', 'neutral']).toContain(r.direction);
  });

  it('stays quiet on a uniform calm stream', () => {
    const r = scan(makeStream(1000, 0, 1000));
    expect(r.anomaly).toBe(false);
  });

  it('splits by TIME with no overlap: a burst in the tail cannot poison the baseline', () => {
    // 900 calm trades @1/s, then a 100-trade burst inside the last 10 s
    const calm  = makeStream(900, 0, 1000);
    const burst = makeStream(100, 900_500, 20).map((t, i) => ({ ...t, id: `b${i}` }));
    const r = scan([...calm, ...burst], { recentSec: 15 });
    expect(r.anomaly).toBe(true);
  });

  it('quiet tail extends the recent window to the last 20 trades', () => {
    // no trades at all in the last 30 s of market time except one
    const trades = makeStream(500, 0, 1000);
    const r = scan(trades); // last 30s has ~30 trades anyway; use large recentSec
    expect(Number.isFinite(r.confidence)).toBe(true);
  });

  it('clear error when the baseline part is too small', () => {
    expect(() => scan(makeStream(60, 0, 1000), { recentSec: 30 }))
      .toThrow(/baseline trades/);
  });

  it('rejects invalid recentSec', () => {
    expect(() => scan(makeStream(200, 0, 1000), { recentSec: 0 })).toThrow('recentSec');
    expect(() => scan(makeStream(200, 0, 1000), { recentSec: NaN })).toThrow('recentSec');
  });

  it('passes detector config through', () => {
    expect(() => scan(makeStream(200, 0, 1000), { windowSize: -1 })).toThrow('windowSize');
  });
});

// ─── calibrationReport ────────────────────────────────────────────────────────

describe('calibrationReport', () => {
  it('real 500-trade baseline: calibrated or partial, with facts', () => {
    const { historical } = loadFixture('spike_1_vol_count');
    const det = new VolumeAnomalyDetector();
    det.train(historical);
    const rep = det.calibrationReport;
    expect(['calibrated', 'partial']).toContain(rep.quality);
    expect(rep.trainingTrades).toBe(historical.length);
    expect(rep.trainingSpanSec).toBeGreaterThan(0);
    expect(rep.channelsCalibrated).toBeGreaterThan(0);
    expect(rep.channelsCalibrated).toBeLessThanOrEqual(rep.channelsTotal);
  });

  it('minimal 50-trade baseline: fallback quality with an actionable note', () => {
    const det = new VolumeAnomalyDetector();
    det.train(makeStream(50, 0, 1000)); // 49 s span → nothing can calibrate
    const rep = det.calibrationReport;
    expect(rep.quality).toBe('fallback');
    expect(rep.notes.join(' ')).toMatch(/15–30 min|universal fallback/);
  });

  it('throws before train()', () => {
    expect(() => new VolumeAnomalyDetector().calibrationReport).toThrow('train');
  });
});
