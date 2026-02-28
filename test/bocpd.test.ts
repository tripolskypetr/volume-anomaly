import { describe, it, expect } from 'vitest';
import {
  bocpdInitState,
  bocpdUpdate,
  bocpdAnomalyScore,
  bocpdBatch,
  defaultPrior,
} from '../src/math/index.js';

const PRIOR = defaultPrior(0, 1);

describe('bocpdInitState', () => {
  it('starts with single run-length hypothesis r=0, log prob=0', () => {
    const s = bocpdInitState();
    expect(s.logProbs).toHaveLength(1);
    expect(s.logProbs[0]).toBe(0);
    expect(s.t).toBe(0);
  });
});

describe('bocpdUpdate', () => {
  it('increments t on each observation', () => {
    let s = bocpdInitState();
    let r = bocpdUpdate(s, 0, PRIOR);
    expect(r.state.t).toBe(1);
    r = bocpdUpdate(r.state, 0, PRIOR);
    expect(r.state.t).toBe(2);
  });

  it('cpProbability is in [0,1]', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 20; i++) {
      const r = bocpdUpdate(s, Math.random(), PRIOR);
      expect(r.cpProbability).toBeGreaterThanOrEqual(0);
      expect(r.cpProbability).toBeLessThanOrEqual(1);
      s = r.state;
    }
  });

  it('log probs normalise (sum of probs ≈ 1)', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 10; i++) {
      const r = bocpdUpdate(s, i * 0.1, PRIOR);
      const totalProb = r.state.logProbs.reduce((acc, lp) => acc + Math.exp(lp), 0);
      expect(totalProb).toBeCloseTo(1, 4);
      s = r.state;
    }
  });

  it('MAP run length grows during stable regime', () => {
    let s = bocpdInitState();
    const runLengths: number[] = [];
    for (let i = 0; i < 30; i++) {
      const r = bocpdUpdate(s, 0.5 + Math.random() * 0.01, PRIOR, 10000); // very low hazard
      runLengths.push(r.mapRunLength);
      s = r.state;
    }
    // Run length should generally increase during a stable regime
    expect(runLengths[runLengths.length - 1]!).toBeGreaterThan(runLengths[0]!);
  });

  it('cpProbability spikes on sudden distribution shift', () => {
    let s = bocpdInitState();
    const prior = defaultPrior(0, 0.01);

    // Stable regime: feed 50 observations near 0
    for (let i = 0; i < 50; i++) {
      const r = bocpdUpdate(s, 0 + Math.random() * 0.01, prior, 200);
      s = r.state;
    }

    // Sudden shift: feed observation far from prior
    const r = bocpdUpdate(s, 100, prior, 200);
    // After a big surprise, CP probability should be elevated
    expect(r.cpProbability).toBeGreaterThan(0.001);
  });

  it('does not mutate input state', () => {
    const s = bocpdInitState();
    const origLen = s.logProbs.length;
    bocpdUpdate(s, 1, PRIOR);
    expect(s.logProbs).toHaveLength(origLen);
    expect(s.t).toBe(0);
  });
});

describe('bocpdAnomalyScore', () => {
  it('returns value in [0,1]', () => {
    let s = bocpdInitState();
    for (let i = 0; i < 10; i++) {
      const r = bocpdUpdate(s, Math.random(), PRIOR);
      const score = bocpdAnomalyScore(r);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
      s = r.state;
    }
  });
});

describe('bocpdBatch', () => {
  it('returns arrays of same length as input', () => {
    const series = Array.from({ length: 50 }, () => Math.random());
    const { cpProbs, mapRunLengths } = bocpdBatch(series, PRIOR);
    expect(cpProbs).toHaveLength(50);
    expect(mapRunLengths).toHaveLength(50);
  });

  it('detects regime change: cpProb elevated after mean shift', () => {
    const stable = Array.from({ length: 100 }, () => 0 + Math.random() * 0.05);
    const shift  = Array.from({ length: 20  }, () => 5 + Math.random() * 0.05);
    const series = [...stable, ...shift];

    const prior = defaultPrior(0, 0.01);
    const { cpProbs } = bocpdBatch(series, prior, 500);

    // Max CP prob in shift region should be higher than in stable region
    const maxStable = Math.max(...cpProbs.slice(0, 100));
    const maxShift  = Math.max(...cpProbs.slice(100));
    expect(maxShift).toBeGreaterThan(maxStable);
  });

  it('run length resets near 0 after changepoint', () => {
    const stable = Array.from({ length: 50 }, () => 0.5);
    const shift  = Array.from({ length: 30 }, () => 100);
    const series = [...stable, ...shift];

    const { mapRunLengths } = bocpdBatch(series, defaultPrior(0.5, 0.001), 200);

    // Run length should be large before shift, small right after
    const rlBeforeShift = mapRunLengths[49]!;
    const rlAfterShift  = mapRunLengths[51]!;
    expect(rlBeforeShift).toBeGreaterThan(rlAfterShift);
  });
});
