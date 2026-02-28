import { describe, it, expect } from 'vitest';
import {
  cusumFit,
  cusumUpdate,
  cusumInitState,
  cusumAnomalyScore,
  cusumBatch,
} from '../src/math/index.js';

describe('cusumFit', () => {
  it('returns safe defaults for empty array', () => {
    const p = cusumFit([]);
    expect(p.mu0).toBe(0);
    expect(p.std0).toBeGreaterThan(0);
  });

  it('estimates mean correctly', () => {
    const values = [0.1, 0.2, 0.3, 0.2, 0.1, 0.2];
    const p = cusumFit(values);
    expect(p.mu0).toBeCloseTo(values.reduce((a, b) => a + b) / values.length, 5);
  });

  it('k = kSigmas * std0', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const p = cusumFit(values, 0.5, 4);
    expect(p.k).toBeCloseTo(0.5 * p.std0, 10);
  });

  it('h = hSigmas * std0', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const p = cusumFit(values, 0.5, 4);
    expect(p.h).toBeCloseTo(4 * p.std0, 10);
  });
});

describe('cusumInitState', () => {
  it('starts at zero', () => {
    const s = cusumInitState();
    expect(s.sPos).toBe(0);
    expect(s.sNeg).toBe(0);
    expect(s.n).toBe(0);
  });
});

describe('cusumUpdate', () => {
  it('accumulates positive shift', () => {
    // h is very large so no alarm triggered
    const params = { mu0: 0, std0: 1, k: 0.5, h: 100 };
    let state = cusumInitState();
    for (let i = 0; i < 5; i++) {
      state = cusumUpdate(state, 2, params).state;
    }
    expect(state.sPos).toBeGreaterThan(0);
  });

  it('fires alarm and resets when sPos reaches h', () => {
    // h=1, k=0 → alarm on first obs x=2: sPos = 2 >= 1
    const params = { mu0: 0, std0: 1, k: 0, h: 1 };
    let state = cusumInitState();
    const result = cusumUpdate(state, 2, params);
    expect(result.alarm).toBe(true);
    expect(result.state.sPos).toBe(0);
    expect(result.state.n).toBe(0);
  });

  it('accumulates sNeg on negative shift (large h prevents reset)', () => {
    // h=1000 so alarm never fires in 10 steps
    const params = { mu0: 0, std0: 1, k: 0, h: 1000 };
    let state = cusumInitState();
    for (let i = 0; i < 10; i++) {
      state = cusumUpdate(state, -5, params).state;
    }
    expect(state.sPos).toBe(0);
    expect(state.sNeg).toBeGreaterThan(0);
  });

  it('is a pure function (does not mutate input state)', () => {
    const params = { mu0: 0, std0: 1, k: 0.5, h: 4 };
    const before = cusumInitState();
    cusumUpdate(before, 3, params);
    expect(before.sPos).toBe(0);
  });

  it('returns alarm=false for in-control observation', () => {
    const params = { mu0: 0, std0: 1, k: 0.5, h: 100 };
    const result = cusumUpdate(cusumInitState(), 0, params);
    expect(result.alarm).toBe(false);
  });
});

describe('cusumAnomalyScore', () => {
  it('returns 0 at init state', () => {
    const params = { mu0: 0, std0: 1, k: 0.5, h: 4 };
    expect(cusumAnomalyScore(cusumInitState(), params)).toBe(0);
  });

  it('returns value in [0,1]', () => {
    const params = { mu0: 0, std0: 1, k: 0, h: 20 };
    let state = cusumInitState();
    for (let i = 0; i < 4; i++) {
      state = cusumUpdate(state, 2, params).state;
    }
    const score = cusumAnomalyScore(state, params);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });
});

describe('cusumBatch', () => {
  it('detects alarm in clearly shifted series', () => {
    // Baseline: zeros, then spike to 10
    const baseline = Array(20).fill(0);
    const spike    = Array(20).fill(10);
    const series   = [...baseline, ...spike];

    const params = cusumFit(baseline, 0.5, 2); // h = 2 * std → fires quickly
    const { alarmIndices } = cusumBatch(series, params);

    expect(alarmIndices.length).toBeGreaterThan(0);
  });

  it('no alarms on steady in-control series', () => {
    // Use a series with real variation so std0 > 0 and h is meaningful
    // Seed a deterministic "random" series by sine wave
    const values = Array.from({ length: 100 }, (_, i) => 0.2 + 0.05 * Math.sin(i * 0.3));
    const params = cusumFit(values, 0.5, 10); // h = 10σ → very conservative
    const { alarmIndices } = cusumBatch(values, params);
    expect(alarmIndices.length).toBe(0);
  });

  it('alarmIndices are within bounds of series', () => {
    const series = [...Array(50).fill(0), ...Array(50).fill(5)];
    const params = cusumFit(Array(50).fill(0), 0.5, 2);
    const { alarmIndices } = cusumBatch(series, params);
    for (const idx of alarmIndices) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(series.length);
    }
  });
});

