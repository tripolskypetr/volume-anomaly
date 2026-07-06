/**
 * excess.test.ts — hawkesExcessSeries (windowed time-rescaling statistic).
 *
 * Key property under test: the statistic distinguishes EXPLAINED clustering
 * (follow-on trades the fitted kernel predicts — a burst's decay tail) from
 * EXOGENOUS excess (arrivals beyond what history justifies), which no raw
 * rate statistic can do.
 */

import { describe, it, expect } from 'vitest';
import { hawkesExcessSeries } from '../src/math/index.js';
import type { HawkesParams } from '../src/index.js';

const POISSON: HawkesParams = { mu: 1, alpha: 0.001, beta: 1 };

/** Regular 1 event/s stream */
function regular(n: number, gap = 1): number[] {
  return Array.from({ length: n }, (_, i) => i * gap);
}

describe('hawkesExcessSeries: basics', () => {
  it('near-zero excess on a stream matching the fitted rate', () => {
    const { excess, firstIdx } = hawkesExcessSeries(regular(200), POISSON, 10);
    expect(firstIdx).toBeGreaterThan(0);
    expect(excess.length).toBeGreaterThan(100);
    const mean = excess.reduce((s, x) => s + x, 0) / excess.length;
    expect(Math.abs(mean)).toBeLessThan(1);
    for (const e of excess) expect(Number.isFinite(e)).toBe(true);
  });

  it('burst produces a large positive excess under a Poisson model', () => {
    const ts = [...regular(100)];
    let t = 100;
    for (let k = 0; k < 50; k++) { t += 0.02; ts.push(t); } // 50 trades in 1s
    const { excess } = hawkesExcessSeries(ts, POISSON, 10);
    expect(Math.max(...excess)).toBeGreaterThan(5);
  });

  it('deficit (gap) produces negative excess', () => {
    const ts = [...regular(100), 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160];
    const { excess } = hawkesExcessSeries(ts, POISSON, 10);
    expect(Math.min(...excess)).toBeLessThan(-1);
  });

  it('span shorter than horizon → single whole-window sample, firstIdx = -1', () => {
    const r = hawkesExcessSeries(regular(10, 0.1), POISSON, 100);
    expect(r.excess.length).toBe(1);
    expect(r.firstIdx).toBe(-1);
  });

  it('empty and single-event inputs are safe', () => {
    expect(hawkesExcessSeries([], POISSON, 10).excess.length).toBe(0);
    expect(hawkesExcessSeries([5], POISSON, 10).excess.length).toBe(0);
  });

  it('beta <= 0 yields no samples instead of NaN', () => {
    expect(hawkesExcessSeries(regular(50), { mu: 1, alpha: 0.5, beta: 0 }, 10).excess.length).toBe(0);
  });
});

describe('hawkesExcessSeries: excitation-awareness (the whole point)', () => {
  it('the same follow-on cluster scores LOWER under a self-exciting model', () => {
    // Initial spike, then a follow-on cluster inside its decay window.
    const ts: number[] = [...regular(60)];
    let t = 60;
    for (let k = 0; k < 30; k++) { t += 0.03; ts.push(t); }  // initial spike
    for (let k = 0; k < 20; k++) { t += 0.15; ts.push(t); }  // follow-on cluster

    // Same μ; one model explains clustering (α/β = 0.8), one cannot (α ≈ 0).
    const excited: HawkesParams = { mu: 1, alpha: 1.6, beta: 2 };
    const poisson: HawkesParams = { mu: 1, alpha: 1e-6, beta: 2 };

    const zEx = hawkesExcessSeries(ts, excited, 5).excess;
    const zPo = hawkesExcessSeries(ts, poisson, 5).excess;
    // Peak excess of the follow-on region: the excited model expects the
    // cluster (high Λ) → materially lower excess than the Poisson model.
    expect(Math.max(...zEx)).toBeLessThan(Math.max(...zPo) * 0.8);
    // ...but the exogenous initial spike still registers under both.
    expect(Math.max(...zEx)).toBeGreaterThan(2);
  });

  it('O(n) recursion matches a brute-force compensator on a small case', () => {
    const ts = [0, 0.5, 1.1, 2.0, 2.05, 2.1, 3.0, 4.5, 5.0, 6.2, 7.7, 8.1, 9.9, 10.4, 11.0];
    const p: HawkesParams = { mu: 0.8, alpha: 0.6, beta: 1.5 };
    const h = 4;
    const { excess, firstIdx } = hawkesExcessSeries(ts, p, h);
    // brute force for each full window
    let k = 0;
    for (let i = 0; i < ts.length; i++) {
      if (ts[i]! - ts[0]! < h) continue;
      const b = ts[i]!, a = b - h;
      let lam = p.mu * h;
      for (const tj of ts) {
        if (tj > b) break;
        if (tj <= a) lam += (p.alpha / p.beta) * (Math.exp(-p.beta * (a - tj)) - Math.exp(-p.beta * (b - tj)));
        else         lam += (p.alpha / p.beta) * (1 - Math.exp(-p.beta * (b - tj)));
      }
      let N = 0;
      for (const tj of ts) if (tj > a && tj <= b) N++;
      const want = (N - lam) / Math.sqrt(Math.max(lam, 1e-9));
      expect(excess[k]!).toBeCloseTo(want, 8);
      k++;
    }
    expect(firstIdx).toBeGreaterThanOrEqual(0);
    expect(k).toBe(excess.length);
  });
});
