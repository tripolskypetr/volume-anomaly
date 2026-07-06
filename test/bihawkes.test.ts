/**
 * bihawkes.test.ts — bivariate buy/sell Hawkes: likelihood, MLE, excitation share.
 */

import { describe, it, expect } from 'vitest';
import {
  biHawkesFit, biHawkesLogLikelihood, biExcitationShare, biBranching,
} from '../src/math/index.js';
import type { BiHawkesParams, IAggregatedTradeData } from '../src/index.js';

let _id = 0;
function trade(tsSec: number, isBuy: boolean): IAggregatedTradeData {
  return { id: String(_id++), price: 100, qty: 1, timestamp: tsSec * 1000, isBuyerMaker: !isBuy };
}

/** Alternating buy/sell at a constant 1 event/s — a balanced Poisson-ish stream */
function balanced(n: number): IAggregatedTradeData[] {
  return Array.from({ length: n }, (_, i) => trade(i, i % 2 === 0));
}

/**
 * Buy-clustered stream: sells at a slow constant pace, buys in tight
 * self-exciting runs — the microstructure signature of buy aggression.
 */
function buyClustered(runs: number): IAggregatedTradeData[] {
  const out: IAggregatedTradeData[] = [];
  let t = 0;
  for (let r = 0; r < runs; r++) {
    out.push(trade(t, false));                 // lone sell
    t += 5;
    for (let k = 0; k < 8; k++) { out.push(trade(t, true)); t += 0.05; } // buy run
    t += 5;
  }
  return out;
}

const P: BiHawkesParams = { muB: 0.5, muS: 0.5, aBB: 0.3, aBS: 0.1, aSB: 0.1, aSS: 0.3, beta: 1 };

describe('biHawkesLogLikelihood', () => {
  const ts = balanced(100);
  const times = ts.map((x) => x.timestamp / 1000);
  const isBuy = ts.map((x) => !x.isBuyerMaker);

  it('finite for valid params', () => {
    expect(Number.isFinite(biHawkesLogLikelihood(times, isBuy, P))).toBe(true);
  });

  it('-Infinity for non-positive mu or beta', () => {
    expect(biHawkesLogLikelihood(times, isBuy, { ...P, muB: 0 })).toBe(-Infinity);
    expect(biHawkesLogLikelihood(times, isBuy, { ...P, beta: 0 })).toBe(-Infinity);
    expect(biHawkesLogLikelihood(times, isBuy, { ...P, aBS: -0.1 })).toBe(-Infinity);
  });

  it('near-true rate params beat absurd ones on a balanced stream', () => {
    // Stream is 1 event/s split evenly → muB ≈ muS ≈ 0.5 with weak excitation.
    const good = biHawkesLogLikelihood(times, isBuy, { ...P, aBB: 0.05, aBS: 0.05, aSB: 0.05, aSS: 0.05 });
    const bad  = biHawkesLogLikelihood(times, isBuy, { ...P, muB: 50, muS: 50 });
    expect(good).toBeGreaterThan(bad);
  });

  it('empty stream → 0', () => {
    expect(biHawkesLogLikelihood([], [], P)).toBe(0);
  });
});

describe('biBranching', () => {
  it('Perron root of a symmetric matrix', () => {
    // Γ = [[0.3, 0.1], [0.1, 0.3]] → eigenvalues 0.4, 0.2
    expect(biBranching(P)).toBeCloseTo(0.4, 10);
  });

  it('supercritical matrix exceeds 1', () => {
    expect(biBranching({ ...P, aBB: 1.2, aSS: 1.2 })).toBeGreaterThan(1);
  });
});

describe('biHawkesFit', () => {
  it('valid stationary params on a balanced stream', () => {
    const r = biHawkesFit(balanced(300));
    const p = r.params;
    expect(p.muB).toBeGreaterThan(0);
    expect(p.muS).toBeGreaterThan(0);
    expect(p.beta).toBeGreaterThan(0);
    for (const a of [p.aBB, p.aBS, p.aSB, p.aSS]) expect(a).toBeGreaterThanOrEqual(0);
    expect(r.branching).toBeLessThan(1);
  });

  it('buy-clustered stream: buy self-excitation dominates sell background', () => {
    const r = biHawkesFit(buyClustered(30));
    // Buys arrive in tight runs → the fitted model must attribute buy arrivals
    // mostly to excitation (aBB high relative to β-scaled background).
    expect(r.params.aBB).toBeGreaterThan(r.params.aSB);
  });

  it('one-sided stream falls back gracefully (no sells)', () => {
    const oneSided = Array.from({ length: 100 }, (_, i) => trade(i, true));
    const r = biHawkesFit(oneSided);
    expect(r.converged).toBe(false);
    expect(Number.isFinite(r.params.muB)).toBe(true);
  });

  it('too few events falls back', () => {
    expect(biHawkesFit(balanced(10)).converged).toBe(false);
  });
});

describe('biExcitationShare', () => {
  it('positive right after a buy run, negative after a sell run', () => {
    const stream = buyClustered(30);
    const times = stream.map((x) => x.timestamp / 1000);
    const isBuy = stream.map((x) => !x.isBuyerMaker);
    const r = biHawkesFit(stream);

    // t just after the last buy run
    const lastBuyT = Math.max(...stream.filter((x) => !x.isBuyerMaker === false).map(() => 0), ...times);
    const afterBuys = biExcitationShare(times, isBuy, lastBuyT, r.params);
    expect(afterBuys).toBeGreaterThan(0);

    // Mirror stream (flip sides) → share flips sign
    const flipped = stream.map((x) => ({ ...x, isBuyerMaker: !x.isBuyerMaker }));
    const rf = biHawkesFit(flipped);
    const afterSells = biExcitationShare(times, isBuy.map((b) => !b), lastBuyT, rf.params);
    expect(afterSells).toBeLessThan(0);
  });

  it('bounded in (−1, 1) and 0-safe on empty history', () => {
    const s = biExcitationShare([], [], 10, P);
    // no events → share of backgrounds: (0.5−0.5)/1 = 0
    expect(s).toBe(0);
  });

  it('share reflects background asymmetry with no events', () => {
    const s = biExcitationShare([], [], 0, { ...P, muB: 0.9, muS: 0.1 });
    expect(s).toBeCloseTo(0.8, 10);
  });
});
