/**
 * perf.test.ts — performance tests.
 *
 * Цели:
 *  1. Latency bounds — P95 не превышает установленный лимит (×5 от baseline).
 *  2. Throughput    — минимальное количество операций в секунду.
 *  3. Scaling       — detect(500) не медленнее detect(200) более чем в разумное число раз.
 *  4. Stability     — latency detect() не растёт со временем (нет накопления состояния).
 *
 * ВАЖНО: если тест падает — проблема в коде (регрессия или утечка), не в тесте.
 *
 * Baseline (Windows 10, V8 / Node 20, 2025-03 AMD Ryzen):
 *   train(200)               P50=1.1ms   P95=1.4ms
 *   train(500)               P50=2.1ms   P95=2.5ms
 *   detect(50)               P50=0.008ms P95=0.09ms
 *   detect(200)              P50=1.4ms   P95=2.5ms
 *   detect(500)              P50=4.7ms   P95=6.3ms
 *   hawkesFit(200)           P50=0.85ms  P95=1.2ms
 *   hawkesFit(500)           P50=1.8ms   P95=2.2ms
 *   hawkesPeakLambda(500)    P50=0.002ms P95=0.003ms
 *   hawkesAnomalyScore       P50=0.0ms   P95=0.001ms
 *   volumeImbalance(1000)    P50=0.055ms P95=0.07ms
 *   cusumBatch(500)          P50=0.02ms  P95=0.08ms
 *   bocpdBatch(200)          P50=3.9ms   P95=4.8ms
 */

import { describe, it, expect } from 'vitest';
import {
  hawkesFit,
  hawkesAnomalyScore,
  hawkesPeakLambda,
  volumeImbalance,
} from '../src/math/hawkes.js';
import { cusumFit, cusumBatch } from '../src/math/cusum.js';
import { bocpdBatch, defaultPrior } from '../src/math/bocpd.js';
import { VolumeAnomalyDetector } from '../src/detector.js';
import type { IAggregatedTradeData } from '../src/types.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function makeTrades(n: number, seed: number, baseTs = 0): IAggregatedTradeData[] {
  const rng = lcg(seed);
  return Array.from({ length: n }, (_, i) => ({
    id: String(i),
    price: 100,
    qty: 0.1 + rng() * 9.9,
    timestamp: baseTs + i * 100,
    isBuyerMaker: rng() < 0.5,
  }));
}

function makeTimestamps(n: number, seed: number): number[] {
  const rng = lcg(seed);
  const ts: number[] = [];
  let t = 0;
  for (let i = 0; i < n; i++) {
    t += rng() * 2 + 0.001; // exponential-ish inter-arrival
    ts.push(t);
  }
  return ts;
}

/** Runs fn N times (after 5 warmup calls), returns sorted ms timings. */
function timings(fn: () => void, N = 40): number[] {
  for (let i = 0; i < 5; i++) fn(); // JIT warmup
  const out: number[] = [];
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    fn();
    out.push(performance.now() - t0);
  }
  return out.sort((a, b) => a - b);
}

function p(sorted: number[], q: number): number {
  return sorted[Math.floor(sorted.length * q)]!;
}

/** Pretty-prints a timing row; returns { p50, p95, p99 } for assertions. */
function report(label: string, sorted: number[]): { p50: number; p95: number; p99: number } {
  const r = { p50: p(sorted, 0.5), p95: p(sorted, 0.95), p99: p(sorted, 0.99) };
  console.log(
    `  ${label.padEnd(40)} ` +
    `P50=${r.p50.toFixed(3).padStart(8)}ms  ` +
    `P95=${r.p95.toFixed(3).padStart(8)}ms  ` +
    `P99=${r.p99.toFixed(3).padStart(8)}ms`,
  );
  return r;
}

// ─── Shared fixtures (built once for the whole file) ─────────────────────────

const HIST_200  = makeTrades(200, 0xabc1);
const HIST_500  = makeTrades(500, 0xabc2);
const WIN_50    = makeTrades(50,  0xabc3, 200_000);
const WIN_200   = makeTrades(200, 0xabc4, 200_000);
const WIN_500   = makeTrades(500, 0xabc5, 200_000);
const TS_50     = makeTimestamps(50,  0x111);
const TS_200    = makeTimestamps(200, 0x222);
const TS_500    = makeTimestamps(500, 0x333);

const RNG_SERIES = (() => {
  const rng = lcg(0x999);
  return Array.from({ length: 500 }, () => rng() * 0.5);
})();

const HAWKES_PARAMS = hawkesFit(TS_200).params;
const CUSUM_PARAMS  = cusumFit(RNG_SERIES.slice(0, 200));
const BOCPD_PRIOR   = defaultPrior(0.3, 0.05);

// Detectors trained once and reused across tests
const DET_200 = new VolumeAnomalyDetector();
DET_200.train(HIST_200);

const DET_500 = new VolumeAnomalyDetector();
DET_500.train(HIST_500);

// ─── 1. Компонентные бенчмарки ────────────────────────────────────────────────

describe('perf: component latency', () => {
  console.log('\n── Component latency ─────────────────────────────────────────');

  it('volumeImbalance(1000 trades): P95 < 1ms', () => {
    const trades = makeTrades(1000, 0x001);
    const t = timings(() => volumeImbalance(trades));
    const r = report('volumeImbalance(1000)', t);
    expect(r.p95).toBeLessThan(1);
  });

  it('hawkesAnomalyScore(scalar): P95 < 0.1ms', () => {
    const t = timings(() => hawkesAnomalyScore(10, HAWKES_PARAMS, 5));
    const r = report('hawkesAnomalyScore', t);
    expect(r.p95).toBeLessThan(0.1);
  });

  it('hawkesPeakLambda(500 ts): P95 < 0.5ms', () => {
    const t = timings(() => hawkesPeakLambda(TS_500, HAWKES_PARAMS));
    const r = report('hawkesPeakLambda(500)', t);
    expect(r.p95).toBeLessThan(0.5);
  });

  it('hawkesFit(50 ts): P95 < 5ms', () => {
    const t = timings(() => hawkesFit(TS_50));
    const r = report('hawkesFit(50 ts)', t);
    expect(r.p95).toBeLessThan(5);
  });

  it('hawkesFit(200 ts): P95 < 10ms', () => {
    const t = timings(() => hawkesFit(TS_200));
    const r = report('hawkesFit(200 ts)', t);
    expect(r.p95).toBeLessThan(10);
  });

  it('hawkesFit(500 ts): P95 < 20ms', () => {
    const t = timings(() => hawkesFit(TS_500));
    const r = report('hawkesFit(500 ts)', t);
    expect(r.p95).toBeLessThan(20);
  });

  it('cusumBatch(500): P95 < 2ms', () => {
    const t = timings(() => cusumBatch(RNG_SERIES, CUSUM_PARAMS));
    const r = report('cusumBatch(500)', t);
    expect(r.p95).toBeLessThan(2);
  });

  it('bocpdBatch(200): P95 < 30ms', () => {
    const t = timings(() => bocpdBatch(RNG_SERIES.slice(0, 200), BOCPD_PRIOR), 20);
    const r = report('bocpdBatch(200)', t);
    expect(r.p95).toBeLessThan(30);
  });
}, 30_000);

// ─── 2. Детектор: train / detect ──────────────────────────────────────────────

describe('perf: detector train / detect latency', () => {
  console.log('\n── Detector train / detect ───────────────────────────────────');

  it('train(200 trades): P95 < 15ms', () => {
    const t = timings(() => {
      const d = new VolumeAnomalyDetector();
      d.train(makeTrades(200, 0x1));
    });
    const r = report('train(200 trades)', t);
    expect(r.p95).toBeLessThan(15);
  });

  it('train(500 trades): P95 < 20ms', () => {
    const t = timings(() => {
      const d = new VolumeAnomalyDetector();
      d.train(makeTrades(500, 0x2));
    }, 30);
    const r = report('train(500 trades)', t);
    expect(r.p95).toBeLessThan(20);
  });

  it('detect(50 trades): P95 < 5ms', () => {
    const t = timings(() => DET_200.detect(WIN_50));
    const r = report('detect(50 trades)', t);
    expect(r.p95).toBeLessThan(5);
  });

  it('detect(200 trades): P95 < 15ms', () => {
    const t = timings(() => DET_200.detect(WIN_200));
    const r = report('detect(200 trades)', t);
    expect(r.p95).toBeLessThan(15);
  });

  it('detect(500 trades): P95 < 40ms', () => {
    const t = timings(() => DET_500.detect(WIN_500), 30);
    const r = report('detect(500 trades)', t);
    expect(r.p95).toBeLessThan(40);
  });
}, 30_000);

// ─── 3. Throughput ────────────────────────────────────────────────────────────
//
// Типовой сценарий реального использования: детектор обучен один раз,
// затем получает N тиков подряд.

describe('perf: throughput', () => {
  console.log('\n── Throughput ─────────────────────────────────────────────────');

  it('detect(200 trades) ×100 total < 500ms (≥200 detections/s)', () => {
    const N = 100;
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      DET_200.detect(WIN_200);
    }
    const elapsed = performance.now() - t0;
    const dps = (N / elapsed * 1000).toFixed(0);
    console.log(`  detect(200) ×100: total=${elapsed.toFixed(1)}ms  throughput=${dps} det/s`);
    // 200 detections/s → 5ms/det → generous bound
    expect(elapsed).toBeLessThan(500);
  });

  it('detect(50 trades) ×1000 total < 500ms (≥2000 detections/s)', () => {
    const N = 1000;
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      DET_200.detect(WIN_50);
    }
    const elapsed = performance.now() - t0;
    const dps = (N / elapsed * 1000).toFixed(0);
    console.log(`  detect(50) ×1000: total=${elapsed.toFixed(1)}ms  throughput=${dps} det/s`);
    expect(elapsed).toBeLessThan(500);
  });
}, 30_000);

// ─── 4. Scaling ───────────────────────────────────────────────────────────────
//
// detect(500) не должен быть больше чем в 8× медленнее detect(200).
// Если соотношение выше — алгоритмическая сложность деградировала (O(n²) вместо O(n)).

describe('perf: scaling', () => {
  console.log('\n── Scaling ────────────────────────────────────────────────────');

  it('detect(500) / detect(200) ratio P50 < 8 (near-linear scaling)', () => {
    const t200 = timings(() => DET_200.detect(WIN_200));
    const t500 = timings(() => DET_500.detect(WIN_500));
    const ratio = p(t500, 0.5) / Math.max(p(t200, 0.5), 0.01);
    console.log(
      `  detect(200) P50=${p(t200, 0.5).toFixed(3)}ms` +
      `  detect(500) P50=${p(t500, 0.5).toFixed(3)}ms` +
      `  ratio=${ratio.toFixed(1)}×`,
    );
    // Ideal O(n) ratio = 2.5× (150→450 BOCPD steps). Allow 8× for jitter.
    expect(ratio).toBeLessThan(8);
  });

  it('train(500) / train(200) ratio P50 < 5', () => {
    const t200 = timings(() => {
      const d = new VolumeAnomalyDetector(); d.train(makeTrades(200, 0x1));
    });
    const t500 = timings(() => {
      const d = new VolumeAnomalyDetector(); d.train(makeTrades(500, 0x2));
    }, 30);
    const ratio = p(t500, 0.5) / Math.max(p(t200, 0.5), 0.01);
    console.log(
      `  train(200) P50=${p(t200, 0.5).toFixed(3)}ms` +
      `  train(500) P50=${p(t500, 0.5).toFixed(3)}ms` +
      `  ratio=${ratio.toFixed(1)}×`,
    );
    expect(ratio).toBeLessThan(5);
  });

  it('hawkesFit(500) / hawkesFit(200) ratio P50 < 5 (O(n) LL per iteration)', () => {
    const t200 = timings(() => hawkesFit(TS_200));
    const t500 = timings(() => hawkesFit(TS_500));
    const ratio = p(t500, 0.5) / Math.max(p(t200, 0.5), 0.001);
    console.log(
      `  hawkesFit(200) P50=${p(t200, 0.5).toFixed(3)}ms` +
      `  hawkesFit(500) P50=${p(t500, 0.5).toFixed(3)}ms` +
      `  ratio=${ratio.toFixed(1)}×`,
    );
    // 500/200 = 2.5× events per LL eval, same NM iterations → expect ≈2.5×
    expect(ratio).toBeLessThan(5);
  });
}, 30_000);

// ─── 5. Stability — latency не растёт при последовательных вызовах ─────────────
//
// detect() сбрасывает BOCPD/CUSUM state каждый раз. Если состояние каким-то
// образом накапливается, latency будет расти. Тест делает 500 вызовов и
// сравнивает первые 50 с последними 50.

describe('perf: latency stability over 500 sequential detect() calls', () => {
  it('last-50 P50 ≤ 3× first-50 P50 (no state leak)', () => {
    const first50: number[] = [];
    const last50:  number[] = [];

    // JIT warmup
    for (let i = 0; i < 10; i++) DET_200.detect(WIN_200);

    for (let i = 0; i < 500; i++) {
      const t0 = performance.now();
      DET_200.detect(WIN_200);
      const dt = performance.now() - t0;
      if (i <  50) first50.push(dt);
      if (i >= 450) last50.push(dt);
    }

    first50.sort((a, b) => a - b);
    last50.sort((a, b)  => a - b);

    const p50_first = p(first50, 0.5);
    const p50_last  = p(last50,  0.5);
    const ratio = p50_last / Math.max(p50_first, 0.001);

    console.log(
      `\n── Stability ─────────────────────────────────────────────────\n` +
      `  first-50 calls P50=${p50_first.toFixed(3)}ms` +
      `  last-50 calls P50=${p50_last.toFixed(3)}ms` +
      `  ratio=${ratio.toFixed(2)}×`,
    );
    expect(ratio).toBeLessThan(3);
  });
}, 30_000);
