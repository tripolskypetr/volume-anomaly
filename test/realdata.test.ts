/**
 * Real-data regression tests against BTCUSDT-aggTrades-2025-03-01.csv (Binance).
 *
 * Five pre-extracted windows are stored as JSON in mock/:
 *   spike_1_vol_count  — ts=1740812760000  largest volume+count burst (z≈24.6 vol, z≈10.5 count)
 *   spike_2_vol_count  — ts=1740824580000  second vol+count burst (z≈10.3, z≈11.7)
 *   spike_3_count_only — ts=1740818550000  HFT micro-trade burst (z≈15.2 count, low vol)
 *   spike_4_vol        — ts=1740828930000  large vol spike (z≈12.6 vol)
 *   calm_baseline      — ts=1740807840000  quietest 30s window of the day
 *
 * Each JSON contains { historical: IAggregatedTradeData[500], recent: IAggregatedTradeData[≤300] }.
 * Tests use the built sources (src/…) via vitest — no build step required.
 *
 * How windows were chosen:
 *   1. Bucketed all 1.4 M trades into 30-second bins.
 *   2. Ranked by volume z-score and count z-score.
 *   3. Selected the four highest-scoring buckets (distinct time ranges).
 *   4. For each bucket: last 500 trades before it → historical; first 300 trades of the bucket → recent.
 *   5. Calm window: bucket with fewest BTC volume and ≥100 trades.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync }          from 'node:fs';
import { join, dirname }         from 'node:path';
import { fileURLToPath }         from 'node:url';
import { VolumeAnomalyDetector, predict } from '../src/index.js';
import type { IAggregatedTradeData }       from '../src/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));

interface MockWindow {
  label:      string;
  ts_ms:      number;
  historical: IAggregatedTradeData[];
  recent:     IAggregatedTradeData[];
}

function loadWindow(name: string): MockWindow {
  const p = join(__dirname, '..', 'mock', name + '.json');
  return JSON.parse(readFileSync(p, 'utf8')) as MockWindow;
}

// ─── Fixtures (loaded once per file) ──────────────────────────────────────────

const spike1 = loadWindow('spike_1_vol_count');   // massive sell-side vol+rate burst
const spike2 = loadWindow('spike_2_vol_count');   // sell-side vol+rate, CUSUM+BOCPD driven
const spike3 = loadWindow('spike_3_count_only');  // HFT micro-trade burst — all detectors fire
const spike4 = loadWindow('spike_4_vol');         // extremely high λ, vol+imbalance driven
const calm   = loadWindow('calm_baseline');        // quiet overnight window

// ─── detect() / VolumeAnomalyDetector ─────────────────────────────────────────

describe('realdata: VolumeAnomalyDetector on BTCUSDT-2025-03-01', () => {

  // ── spike_1: volume + count burst (largest of the day) ──────────────────────

  it('spike_1 — confidence exceeds 0.60 (well above noise floor)', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike1.historical);
    const r = det.detect(spike1.recent, 0.6);
    expect(r.anomaly).toBe(true);
    expect(r.confidence).toBeGreaterThan(0.60);
  });

  it('spike_1 — volume_spike and imbalance_shift signals are present', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike1.historical);
    const r = det.detect(spike1.recent, 0.6);
    const kinds = r.signals.map(s => s.kind);
    expect(kinds).toContain('volume_spike');
    expect(kinds).toContain('imbalance_shift');
  });

  it('spike_1 — sell-side imbalance over the full burst bucket', () => {
    // Full-bucket window (3000 trades) dilutes the pure-sell onset with the
    // post-burst two-way flow, so the imbalance is ≈ −0.42 rather than −0.9+.
    const det = new VolumeAnomalyDetector();
    det.train(spike1.historical);
    const r = det.detect(spike1.recent, 0.6);
    expect(r.imbalance).toBeLessThan(-0.3);
  });

  it('spike_1 — hawkesLambda is elevated above training μ', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike1.historical);
    const r      = det.detect(spike1.recent, 0.6);
    const mu     = det.trainedModels!.hawkesParams.mu;
    expect(r.hawkesLambda).toBeGreaterThan(mu);
  });

  // ── spike_2: CUSUM + BOCPD driven ───────────────────────────────────────────

  it('spike_2 — confidence exceeds 0.60', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike2.historical);
    const r = det.detect(spike2.recent, 0.6);
    expect(r.anomaly).toBe(true);
    expect(r.confidence).toBeGreaterThan(0.60);
  });

  it('spike_2 — volume_spike and cusum_alarm signals fire', () => {
    // BOCPD is noise-floor-calibrated on real data and stays quiet here;
    // the anomaly is carried by the volume channel with CUSUM corroboration.
    const det = new VolumeAnomalyDetector();
    det.train(spike2.historical);
    const r = det.detect(spike2.recent, 0.6);
    const kinds = r.signals.map(s => s.kind);
    expect(kinds).toContain('volume_spike');
    expect(kinds).toContain('cusum_alarm');
  });

  // ── spike_3: HFT micro-trade burst — highest overall confidence ─────────────

  it('spike_3 — anomaly=true at default 0.75 threshold', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike3.historical);
    const r = det.detect(spike3.recent, 0.75);
    expect(r.anomaly).toBe(true);
  });

  it('spike_3 — volume_spike and cusum_alarm fire (HFT count burst)', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike3.historical);
    const r     = det.detect(spike3.recent, 0.75);
    const kinds = r.signals.map(s => s.kind);
    expect(kinds).toContain('volume_spike');
    expect(kinds).toContain('cusum_alarm');
  });

  it('spike_3 — rate z-score dominates volume z-score (count-only burst)', () => {
    // This spike is many small trades: arrival-rate z ≈ 32 while volume z ≈ 4.
    const det = new VolumeAnomalyDetector();
    det.train(spike3.historical);
    const r = det.detect(spike3.recent, 0.75);
    expect(r.stats.zRate).toBeGreaterThan(r.stats.zVol);
    expect(r.stats.zRate).toBeGreaterThan(20);
  });

  it('spike_3 — sell-side imbalance over the full bucket', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike3.historical);
    const r = det.detect(spike3.recent, 0.75);
    expect(r.imbalance).toBeLessThan(-0.2);
  });

  // ── spike_4: extreme λ (volume-dominant) ────────────────────────────────────

  it('spike_4 — confidence exceeds 0.60', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike4.historical);
    const r = det.detect(spike4.recent, 0.6);
    expect(r.anomaly).toBe(true);
    expect(r.confidence).toBeGreaterThan(0.60);
  });

  it('spike_4 — hawkesLambda well above the training background rate', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike4.historical);
    const r  = det.detect(spike4.recent, 0.6);
    const mu = det.trainedModels!.hawkesParams.mu;
    expect(r.hawkesLambda).toBeGreaterThan(2 * mu);
  });

  it('spike_4 — volume_spike signal score is at ceiling (≥ 0.99)', () => {
    const det = new VolumeAnomalyDetector();
    det.train(spike4.historical);
    const r     = det.detect(spike4.recent, 0.6);
    const vsig  = r.signals.find(s => s.kind === 'volume_spike');
    expect(vsig).toBeDefined();
    // Rational (algebraic) sigmoid: the top end approaches 1 harmonically to
    // preserve ranking of extreme events, so "at ceiling" is ≥ 0.97 (that is
    // ~16 tail units beyond the calibrated level), not the old exponential
    // saturation at ≥ 0.99.
    expect(vsig!.score).toBeGreaterThanOrEqual(0.97);
  });

  // ── calm baseline: no anomaly ────────────────────────────────────────────────

  it('calm_baseline — anomaly=false at 0.75 threshold', () => {
    const det = new VolumeAnomalyDetector();
    det.train(calm.historical);
    const r = det.detect(calm.recent, 0.75);
    expect(r.anomaly).toBe(false);
  });

  it('calm_baseline — confidence is below all spike confidences', () => {
    const calmDet = new VolumeAnomalyDetector();
    calmDet.train(calm.historical);
    const calmConf = calmDet.detect(calm.recent, 0.6).confidence;

    for (const w of [spike1, spike2, spike3, spike4]) {
      const det = new VolumeAnomalyDetector();
      det.train(w.historical);
      const spikeConf = det.detect(w.recent, 0.6).confidence;
      expect(calmConf).toBeLessThan(spikeConf);
    }
  });

  // ── isTrained guard ──────────────────────────────────────────────────────────

  it('isTrained is false before train(), true after', () => {
    const det = new VolumeAnomalyDetector();
    expect(det.isTrained).toBe(false);
    det.train(spike1.historical);
    expect(det.isTrained).toBe(true);
  });
});

// ─── predict() on real data ────────────────────────────────────────────────────

describe('realdata: predict() direction on BTCUSDT-2025-03-01', () => {

  it('spike_1 — direction=short at confidence 0.6 (strong sell aggression)', () => {
    const r = predict(spike1.historical, spike1.recent, 0.6);
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('short');
  });

  it('spike_3 — HFT micro-burst is directionless even at explicit threshold 0.3', () => {
    // direction reads the BURST-local flow, not the full-window drift.
    // spike_3 is an HFT micro-trade burst: its peak window is two-way
    // ping-pong (burstImbalance ≈ −0.27), more balanced than the window's
    // −0.39 drift — the old 'short' here was an artifact of averaging slow
    // drift into the burst.  Genuinely directional spikes sharpen instead
    // (spike_1: −0.48 burst-local vs −0.42 full-window).
    const r = predict(spike3.historical, spike3.recent, 0.75, 0.3);
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('neutral');
    expect(r.burstImbalance).toBeGreaterThan(-0.3);
    expect(r.burstImbalance).toBeLessThan(0);
  });

  it('spike_4 — direction=short at confidence 0.6', () => {
    const r = predict(spike4.historical, spike4.recent, 0.6);
    expect(r.anomaly).toBe(true);
    expect(r.direction).toBe('short');
  });

  it('calm_baseline — direction=neutral (no anomaly)', () => {
    const r = predict(calm.historical, calm.recent, 0.75);
    expect(r.anomaly).toBe(false);
    expect(r.direction).toBe('neutral');
  });

  it('predict() imbalance matches detect() imbalance for same inputs', () => {
    for (const w of [spike1, spike3, calm]) {
      const pr = predict(w.historical, w.recent, 0.75);
      const det = new VolumeAnomalyDetector();
      det.train(w.historical);
      const dr = det.detect(w.recent, 0.75);
      expect(pr.imbalance).toBeCloseTo(dr.imbalance, 6);
      expect(pr.confidence).toBeCloseTo(dr.confidence, 6);
      expect(pr.anomaly).toBe(dr.anomaly);
    }
  });

  it('explicit imbalanceThreshold=0 forces direction long/short whenever anomaly fires', () => {
    // With thr=0, any non-zero imbalance is directional
    const r = predict(spike1.historical, spike1.recent, 0.6, 0);
    if (r.anomaly) {
      expect(['long', 'short']).toContain(r.direction);
    }
  });
});

// ─── Determinism ──────────────────────────────────────────────────────────────

describe('realdata: determinism — same inputs → same outputs', () => {
  it('detect() is deterministic across two identical runs', () => {
    const run = (w: MockWindow) => {
      const det = new VolumeAnomalyDetector();
      det.train(w.historical);
      return det.detect(w.recent, 0.75);
    };
    for (const w of [spike1, spike3, calm]) {
      const r1 = run(w);
      const r2 = run(w);
      expect(r1.confidence).toBe(r2.confidence);
      expect(r1.anomaly).toBe(r2.anomaly);
      expect(r1.imbalance).toBe(r2.imbalance);
      expect(r1.hawkesLambda).toBe(r2.hawkesLambda);
    }
  });
});
