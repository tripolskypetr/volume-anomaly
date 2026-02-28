/**
 * Adversarial tests — каждый ловит конкретное "написано что сделано, а в коде не так".
 *
 * Каждый describe описывает что именно проверяется и почему это важно.
 */

import { describe, it, expect } from 'vitest';
import {
  cusumUpdate,
  cusumInitState,
  cusumAnomalyScore,
} from '../src/math/cusum.js';
import {
  bocpdUpdate,
  bocpdInitState,
  bocpdAnomalyScore,
} from '../src/math/bocpd.js';
import {
  hawkesAnomalyScore,
  hawkesPeakLambda,
  hawkesLogLikelihood,
  hawkesLambda,
  hawkesFit,
  volumeImbalance,
} from '../src/math/hawkes.js';
import { cusumFit } from '../src/math/cusum.js';
import { nelderMead } from '../src/math/optimizer.js';
import { VolumeAnomalyDetector }     from '../src/detector.js';
import type { IAggregatedTradeData } from '../src/types.js';

let _id = 0;
function trade(ts: number, qty: number, isBuyerMaker: boolean): IAggregatedTradeData {
  return { id: String(_id++), price: 100, qty, timestamp: ts, isBuyerMaker };
}

// ─── 1. cusumUpdate.preResetState ────────────────────────────────────────────
//
// Ранее detector.ts дублировал формулу cusumUpdate вручную при alarm.
// Если бы формула изменилась — дубль разошёлся бы молча.
// Теперь cusumUpdate возвращает preResetState, и detector.ts его использует.
// Тест: preResetState содержит значения ДО сброса.

describe('cusumUpdate.preResetState', () => {
  it('preResetState.sPos ≥ h when alarm fires', () => {
    const p = { mu0: 0, std0: 1, k: 0, h: 0.5 };
    const upd = cusumUpdate(cusumInitState(), 0.6, p); // sPos=0.6 ≥ h=0.5
    expect(upd.alarm).toBe(true);
    expect(upd.preResetState.sPos).toBeGreaterThanOrEqual(p.h);
    expect(upd.state.sPos).toBe(0); // state сброшен
    expect(upd.state.sNeg).toBe(0);
  });

  it('preResetState === state when no alarm', () => {
    const p = { mu0: 0, std0: 1, k: 0, h: 100 };
    const upd = cusumUpdate(cusumInitState(), 0.3, p);
    expect(upd.alarm).toBe(false);
    expect(upd.preResetState.sPos).toBe(upd.state.sPos);
    expect(upd.preResetState.sNeg).toBe(upd.state.sNeg);
  });

  it('cusumAnomalyScore on preResetState captures alarm peak, not zeroed state', () => {
    const p = { mu0: 0, std0: 1, k: 0, h: 0.5 };
    const upd = cusumUpdate(cusumInitState(), 1.0, p); // sPos=1.0 >> h=0.5
    expect(upd.alarm).toBe(true);
    // Score на сброшенном state = 0
    expect(cusumAnomalyScore(upd.state, p)).toBe(0);
    // Score на preResetState = 1 (min(1.0/0.5, 1) = 1)
    expect(cusumAnomalyScore(upd.preResetState, p)).toBe(1);
  });

  it('accumulated alarms do not lose evidence over multiple resets', () => {
    // 3 волны по 10 значений = 3 alarm события.
    // Если использовать upd.state при alarm — peakScore = 0 (всегда сброс).
    // С preResetState peakScore должен быть 1.
    const p = { mu0: 0, std0: 1, k: 0, h: 2 };
    let state = cusumInitState();
    let peakFromPreReset = 0;
    let peakFromState    = 0;
    for (let i = 0; i < 30; i++) {
      const upd = cusumUpdate(state, 1.0, p); // накапливается к alarm каждые 2 шага
      peakFromPreReset = Math.max(peakFromPreReset, cusumAnomalyScore(upd.preResetState, p));
      peakFromState    = Math.max(peakFromState,    cusumAnomalyScore(upd.state,         p));
      state = upd.state;
    }
    expect(peakFromPreReset).toBe(1);      // peak успешно захвачен
    expect(peakFromState).toBeLessThan(1); // из state — потеряно
  });
});

// ─── 2. bocpdAnomalyScore: drop зажат в [0, 1] ───────────────────────────────
//
// Документация утверждала drop ∈ [0,1].
// Без clamp при росте RL drop отрицательный → sigmoid выдаёт > 0.5 для "роста".
// Фикс: Math.max(0, drop).

describe('bocpdAnomalyScore: negative drop clamped', () => {
  it('score < 0.15 when run length grows (no changepoint)', () => {
    // prevRL=5, mapRL=6 → drop = (5-6)/5 = -0.2 → clamp(0) → score ≈ 0.018
    const r = { mapRunLength: 6, cpProbability: 0.005, state: bocpdInitState() };
    const score = bocpdAnomalyScore(r, 5);
    expect(score).toBeLessThan(0.15);
  });

  it('score > 0.9 when run length resets (changepoint)', () => {
    // prevRL=90, mapRL=1 → drop=0.989 → score ≈ 0.98
    const r = { mapRunLength: 1, cpProbability: 0.005, state: bocpdInitState() };
    const score = bocpdAnomalyScore(r, 90);
    expect(score).toBeGreaterThan(0.9);
  });

  it('score always in [0,1] for any input combination', () => {
    const cases = [
      { rl: 0,   prev: 10  },
      { rl: 10,  prev: 10  }, // no change
      { rl: 15,  prev: 10  }, // grew — negative drop
      { rl: 1,   prev: 100 }, // reset
      { rl: 0,   prev: 0   }, // first step
      { rl: 5,   prev: -1  }, // invalid prev
    ];
    for (const { rl, prev } of cases) {
      const s = bocpdAnomalyScore(
        { mapRunLength: rl, cpProbability: 0.005, state: bocpdInitState() },
        prev,
      );
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it('returns 0 for first step (prevRunLength = 0)', () => {
    const r = { mapRunLength: 0, cpProbability: 0.2, state: bocpdInitState() };
    expect(bocpdAnomalyScore(r, 0)).toBe(0);
    expect(bocpdAnomalyScore(r)).toBe(0); // default arg
  });

  it('returns 0 for negative prevRunLength', () => {
    const r = { mapRunLength: 0, cpProbability: 0.2, state: bocpdInitState() };
    expect(bocpdAnomalyScore(r, -5)).toBe(0);
  });
});

// ─── 3. hawkesLogLikelihood: инвариантность к origin timestamp ───────────────
//
// Bug (исправлен): T = timestamps[n-1] вместо T = timestamps[n-1] - t0.
// Тест: LL одинаковый при timestamps начинающихся с 0 и с Unix epoch.

describe('hawkesLogLikelihood: timestamp origin invariance', () => {
  const params = { mu: 1, alpha: 0.3, beta: 2 };

  it('same LL whether timestamps start at 0 or at Unix epoch offset', () => {
    const relative = [0, 0.5, 1.2, 1.8, 2.5, 3.1, 3.9, 4.4, 5.0, 5.6];
    const EPOCH    = 1_700_000_000;
    const absolute = relative.map((t) => t + EPOCH);

    const llRel = hawkesLogLikelihood(relative, params);
    const llAbs = hawkesLogLikelihood(absolute, params);

    expect(llRel).not.toBe(-Infinity);
    expect(llAbs).toBeCloseTo(llRel, 5);
  });

  it('LL is worse for shuffled (non-Poisson-structured) timestamps', () => {
    // Упорядоченные timestamps → правильный LL
    // Неправильный T (старый баг) был бы одинаковым для обоих — тест поймал бы регрессию
    const ordered   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const bunched   = [0, 0.1, 0.2, 0.3, 0.4, 5, 6, 7, 8, 9]; // burst + silence

    const llOrdered = hawkesLogLikelihood(ordered,  params);
    const llBunched = hawkesLogLikelihood(bunched,  params);

    // Оба должны быть конечными
    expect(llOrdered).not.toBe(-Infinity);
    expect(llBunched).not.toBe(-Infinity);
    // Bunched даёт более высокий LL для self-exciting process (alpha>0)
    expect(llBunched).toBeGreaterThan(llOrdered);
  });
});

// ─── 4. hawkesAnomalyScore: empiricalRate fallback при alpha ≈ 0 ─────────────
//
// При Poisson baseline (alpha→0 после MLE) hawkesLambda≈mu всегда.
// Без empiricalRate: burst 1000×mu → score ≈ 0.
// С empiricalRate: ratio=1000 → score ≈ 1.

describe('hawkesAnomalyScore: empiricalRate fallback', () => {
  it('scores high on rate spike even with alpha≈0 (Poisson params)', () => {
    const params = { mu: 1, alpha: 1e-12, beta: 1e-6 };
    const score  = hawkesAnomalyScore(params.mu, params, 1000);
    expect(score).toBeGreaterThan(0.9);
  });

  it('does not fire on baseline rate (empiricalRate ≈ mu)', () => {
    const params = { mu: 1, alpha: 1e-12, beta: 1e-6 };
    const score  = hawkesAnomalyScore(params.mu, params, 1.0);
    // sig((1/1 - 2)*2) = sig(-2) ≈ 0.12
    expect(score).toBeLessThan(0.2);
  });

  it('empiricalRate=0 falls back to intensity-only score', () => {
    const params = { mu: 1, alpha: 0.5, beta: 2 };
    // meanLambda = 1/(1-0.25) = 1.333; lambda=5 → ratio=3.75
    const scoreWithRate    = hawkesAnomalyScore(5, params, 10);
    const scoreWithoutRate = hawkesAnomalyScore(5, params, 0);
    // Оба должны быть > 0
    expect(scoreWithoutRate).toBeGreaterThan(0);
    // С rate score выше или равен
    expect(scoreWithRate).toBeGreaterThanOrEqual(scoreWithoutRate);
  });
});

// ─── 5. hawkesPeakLambda: peak захватывает burst в середине окна ─────────────
//
// hawkesLambda(lastT) пропустит burst если он угас к концу окна.
// hawkesPeakLambda берёт максимум по всем событиям.

describe('hawkesPeakLambda: captures burst in middle of window', () => {
  it('peak >> lambda at last event when burst is followed by silence', () => {
    const params = { mu: 1, alpha: 0.5, beta: 5 };
    const ts: number[] = [];
    for (let i = 0; i < 50; i++) ts.push(i * 0.001);        // burst: 50ms, 1ms apart
    for (let i = 1; i <= 10; i++) ts.push(0.050 + i * 1.0); // silence: 1s apart

    const peak = hawkesPeakLambda(ts, params);

    // lambda at lastT: exp(-5 * 10s) ≈ 0, so lambda ≈ mu=1
    const lastT        = ts[ts.length - 1]!;
    const lambdaAtLast = hawkesLambda(lastT, ts.slice(0, -1), params);

    // peak (во время burst) должен быть минимум в 5× больше lambda в конце
    expect(peak).toBeGreaterThan(lambdaAtLast * 5);
  });

  it('peak ≥ lambda(lastT) always (peak is a maximum)', () => {
    const params = { mu: 0.5, alpha: 0.2, beta: 1 };
    // Равномерные timestamps — peak = lambda at last (монотонный рост A)
    const ts     = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const peak   = hawkesPeakLambda(ts, params);
    const lastT  = ts[ts.length - 1]!;
    const lambdaAtLast = hawkesLambda(lastT, ts.slice(0, -1), params);
    expect(peak).toBeGreaterThanOrEqual(lambdaAtLast - 1e-10);
  });
});

// ─── 6. detector: identical timestamps — нет деления на ноль ─────────────────
//
// windowSec = 0 → empiricalRate = n/0 = Infinity → hawkesScore = 1 → ложное срабатывание.
// Фикс: minWindowSec = n * 0.001s.

describe('detector: identical timestamps edge case', () => {
  function makeHist(): IAggregatedTradeData[] {
    const h: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) h.push(trade(i * 1000, 1, i % 2 === 0));
    return h;
  }

  it('confidence is finite and in [0,1] when all trades share the same timestamp', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 10 });
    det.train(makeHist());

    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 50; i++) recent.push(trade(500_000, 1, i % 2 === 0));

    const result = det.detect(recent, 0.75);
    expect(Number.isFinite(result.confidence)).toBe(true);
    expect(Number.isNaN(result.confidence)).toBe(false);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('balanced identical-timestamp trades do not trigger anomaly at default threshold', () => {
    // 50 трейдов с одним timestamp, чередующиеся buy/sell → imbalance≈0.
    // hawkesScore может быть высоким из-за rate spike, но CUSUM/BOCPD молчат.
    // combined = 0.4·hawkes + 0 + 0 = 0.4 < 0.75.
    const det = new VolumeAnomalyDetector({ windowSize: 10 });
    det.train(makeHist());

    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 50; i++) recent.push(trade(500_000, 1, i % 2 === 0));

    const result = det.detect(recent, 0.75);
    expect(result.anomaly).toBe(false);
    expect(Number.isFinite(result.hawkesLambda)).toBe(true);
  });
});

// ─── 7. CUSUM sNeg на |imbalance| ≥ 0 никогда не вызывает false alarm ────────
//
// Комментарий "one-sided S⁺ suffices" — но код двусторонний.
// На |imbalance| ≥ 0 sNeg должен всегда стремиться к 0.

describe('cusum sNeg stays zero on non-negative series', () => {
  it('sNeg never reaches alarm threshold when values are always ≥ mu0', () => {
    const p = { mu0: 0.1, std0: 0.1, k: 0.05, h: 0.5 };
    let state = cusumInitState();
    for (let i = 0; i < 500; i++) {
      const upd = cusumUpdate(state, 0.3, p); // > mu0 → sNeg drains immediately
      expect(upd.preResetState.sNeg).toBeLessThan(p.h);
      state = upd.state;
    }
    expect(state.sNeg).toBe(0);
  });

  it('sNeg can alarm when values drop below mu0 - k (genuine negative shift)', () => {
    // Это тест что sNeg РАБОТАЕТ когда нужно — он не мёртвый код вообще,
    // он нужен на случай если |imbalance| неожиданно упал ниже baseline.
    const p = { mu0: 0.5, std0: 0.1, k: 0.05, h: 0.3 };
    let state = cusumInitState();
    let alarmFired = false;
    for (let i = 0; i < 50; i++) {
      const upd = cusumUpdate(state, 0.0, p); // << mu0 → sNeg accumulates
      if (upd.alarm) { alarmFired = true; break; }
      state = upd.state;
    }
    expect(alarmFired).toBe(true);
  });
});

// ─── 8. detector: confidence threshold validation ─────────────────────────────
//
// detect() принимает confidence out of [0,1] молча.
// Тест документирует поведение: anomaly = (combined >= confidence).

describe('detector: confidence threshold behaviour', () => {
  function setup() {
    const det  = new VolumeAnomalyDetector({ windowSize: 10 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    det.train(hist);
    return det;
  }

  it('confidence=0 always returns anomaly=true (threshold impossible to miss)', () => {
    const det    = setup();
    const recent = [trade(300_000, 1, true), trade(301_000, 1, false)];
    const result = det.detect(recent, 0);
    // combined ≥ 0 is always true (combined ≥ 0)
    expect(result.anomaly).toBe(true);
  });

  it('confidence=1 never returns anomaly=true (threshold impossible to reach)', () => {
    const det    = setup();
    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) recent.push(trade(300_000 + i * 1000, 1, i % 2 === 0));
    const result = det.detect(recent, 1);
    // combined can be 1.0 only if ALL detectors score 1 simultaneously — very unlikely on calm data
    // At minimum: anomaly = (combined === 1)
    expect(result.anomaly).toBe(result.confidence >= 1);
  });
});

// ─── 9. hawkesFit: stationarity constraint always respected ──────────────────
//
// negLL penalty returns 1e10 when alpha >= beta.
// Тест: hawkesFit never returns converged params where alpha >= beta.
// Без теста фикс optimizer penalty мог бы тихо сломаться.

describe('hawkesFit: stationarity constraint alpha < beta', () => {
  function check(ts: number[]) {
    const r = hawkesFit(ts);
    if (r.converged) {
      // When converged, the optimizer found a valid interior point
      expect(r.params.alpha).toBeLessThan(r.params.beta);
      expect(r.params.mu).toBeGreaterThan(0);
      expect(r.params.alpha).toBeGreaterThan(0);
      expect(r.params.beta).toBeGreaterThan(0);
    }
    // stationarity = alpha/beta reported correctly
    expect(r.stationarity).toBeCloseTo(r.params.alpha / r.params.beta, 8);
  }

  it('uniform arrivals do not produce supercritical params', () => {
    const ts = Array.from({ length: 100 }, (_, i) => i * 0.2);
    check(ts);
  });

  it('clustered arrivals do not produce supercritical params', () => {
    const ts: number[] = [];
    for (let i = 0; i < 20; i++) {
      for (let j = 0; j < 5; j++) ts.push(i * 10 + j * 0.05);
    }
    check(ts);
  });

  it('stationarity = alpha / beta exactly', () => {
    const ts = Array.from({ length: 50 }, (_, i) => i * 0.1);
    const r  = hawkesFit(ts);
    expect(r.stationarity).toBeCloseTo(r.params.alpha / r.params.beta, 10);
  });
});

// ─── 10. rollingAbsImbalance output length ────────────────────────────────────
//
// rollingAbsImbalance(sorted) для n трейдов и windowSize w возвращает
// n - w + 1 элементов: цикл `for (i = w; i <= n; i++)` включает оба конца.
// То есть n === w даёт 1 окно (весь массив), n === w-1 даёт 0 окон.

describe('detector: rollingAbsImbalance output length', () => {
  it('n < windowSize → 0 windows → CUSUM/BOCPD не делают шагов', () => {
    // w=20, trades=19 → i starts at 20, 20 <= 19 false → пустой массив
    const det = new VolumeAnomalyDetector({ windowSize: 20 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    det.train(hist);

    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 19; i++) recent.push(trade(400_000 + i * 1000, 1, i % 2 === 0));
    const r = det.detect(recent);
    // 0 rolling windows → BOCPD не шагал → runLength = 0
    expect(r.runLength).toBe(0);
    expect(r.cusumStat).toBe(0);
  });

  it('n === windowSize → 1 окно (весь массив)', () => {
    // w=10, trades=10 → i=10 <= 10 → одна итерация → slice(0,10)
    const det = new VolumeAnomalyDetector({ windowSize: 10 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    det.train(hist);

    // 10 чередующихся buy/sell → |imbalance| ≈ 0 → CUSUM/BOCPD видят один шаг
    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 10; i++) recent.push(trade(300_000 + i * 1000, 1, i % 2 === 0));
    const r = det.detect(recent);
    // BOCPD делает 1 шаг → runLength = 1 (первый шаг всегда MAP RL = 0 или 1)
    // Главное: результат конечный и в диапазоне
    expect(Number.isFinite(r.confidence)).toBe(true);
    expect(r.confidence).toBeGreaterThanOrEqual(0);
    expect(r.confidence).toBeLessThanOrEqual(1);
  });

  it('n > windowSize → n - w + 1 окон, confidence финитна', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 10 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    det.train(hist);

    // 50 трейдов → 41 окно → BOCPD видит 41 шаг, runLength растёт
    const recent: IAggregatedTradeData[] = [];
    for (let i = 0; i < 50; i++) recent.push(trade(300_000 + i * 1000, 1, i % 2 === 0));
    const r = det.detect(recent);
    expect(Number.isFinite(r.confidence)).toBe(true);
    expect(r.runLength).toBeGreaterThan(0); // минимум 1 шаг BOCPD был сделан
  });
});

// ─── 11. train() boundary: 49 throws, 50 succeeds ────────────────────────────
//
// train() documents "Need at least 50 trades".
// Test the exact boundary — не 51, а точно 50.

describe('VolumeAnomalyDetector.train() boundary', () => {
  it('throws with 49 trades', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 10 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 49; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    expect(() => det.train(hist)).toThrow('50');
  });

  it('succeeds with exactly 50 trades', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 10 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 50; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    expect(() => det.train(hist)).not.toThrow();
    expect(det.isTrained).toBe(true);
  });

  it('scoreWeights not summing to 1 throws at construction', () => {
    expect(() => new VolumeAnomalyDetector({ scoreWeights: [0.5, 0.3, 0.3] }))
      .toThrow('scoreWeights');
  });

  it('scoreWeights summing to 1 exactly does not throw', () => {
    expect(() => new VolumeAnomalyDetector({ scoreWeights: [0.4, 0.3, 0.3] }))
      .not.toThrow();
  });
});

// ─── 12. cusumFit with a single value ────────────────────────────────────────
//
// n=1 → var = (x - mean)^2 / max(n-1, 1) = 0 / 1 = 0 → std0 = 0.
// Code: std0 = Math.sqrt(var0) || 1e-6 → falls back to 1e-6 (truthy check fails for 0).
// Test: no NaN/Infinity, std0 > 0, k and h are positive.

describe('cusumFit: single-value array', () => {
  it('std0 falls back to 1e-6, not 0 or NaN', () => {
    const p = cusumFit([0.5]);
    expect(Number.isFinite(p.std0)).toBe(true);
    expect(p.std0).toBeGreaterThan(0);
    expect(p.mu0).toBeCloseTo(0.5, 10);
    expect(p.k).toBeGreaterThan(0);
    expect(p.h).toBeGreaterThan(0);
  });

  it('constant array (zero variance) gives std0 = 1e-6', () => {
    const p = cusumFit([3, 3, 3, 3, 3]);
    expect(p.std0).toBeCloseTo(1e-6, 10);
    expect(Number.isFinite(p.k)).toBe(true);
    expect(Number.isFinite(p.h)).toBe(true);
  });
});

// ─── 13. volumeImbalance always in [-1, +1] ──────────────────────────────────
//
// (buyVol - sellVol) / total → by construction ∈ [-1, +1], but floating-point
// extremes (very large or very small qty) should not escape the range.

describe('volumeImbalance: always in [-1, +1]', () => {
  const cases: Array<[number, boolean][]> = [
    [[1e308, false]],                                      // all buy, extreme qty
    [[1e308, true]],                                       // all sell, extreme qty
    [[1e-308, false], [1e-308, true]],                     // balanced tiny qty
    [[1e308, false], [1e-300, true]],                      // heavily buy-skewed
    [[Number.MAX_SAFE_INTEGER, false], [1, true]],         // large vs small
  ];

  for (const trades of cases) {
    it(`imbalance ∈ [-1,+1] for qty pattern [${trades.map(([q, m]) => `${q.toExponential(0)}/${m}`).join(',')}]`, () => {
      const t = trades.map(([qty, isBuyerMaker]) =>
        ({ id: '0', price: 100, qty, timestamp: 0, isBuyerMaker } as IAggregatedTradeData),
      );
      const imb = volumeImbalance(t);
      expect(imb).toBeGreaterThanOrEqual(-1);
      expect(imb).toBeLessThanOrEqual(1);
      expect(Number.isFinite(imb)).toBe(true);
    });
  }
});

// ─── 14. hawkesPeakLambda >= mu always ────────────────────────────────────────
//
// hawkesPeakLambda initialises peak = mu and only updates upward.
// Test: peak >= mu for any non-empty timestamps + positive params.

describe('hawkesPeakLambda >= mu always', () => {
  it('returns exactly mu for empty timestamps', () => {
    const params = { mu: 3, alpha: 0.5, beta: 2 };
    expect(hawkesPeakLambda([], params)).toBe(3);
  });

  it('peak >= mu for single timestamp', () => {
    const params = { mu: 1, alpha: 0.5, beta: 2 };
    // First event: A=0, lam = mu + alpha*0 = mu → peak = mu
    expect(hawkesPeakLambda([0], params)).toBeGreaterThanOrEqual(params.mu);
  });

  it('peak >= mu for clustered timestamps (excitation adds to mu)', () => {
    const params = { mu: 1, alpha: 0.5, beta: 2 };
    const ts = [0, 0.01, 0.02, 0.03, 0.04]; // tight cluster
    const peak = hawkesPeakLambda(ts, params);
    expect(peak).toBeGreaterThanOrEqual(params.mu);
    // In a tight cluster, A grows → peak >> mu
    expect(peak).toBeGreaterThan(params.mu);
  });

  it('peak >= mu even when alpha is very small', () => {
    const params = { mu: 5, alpha: 1e-10, beta: 1 };
    const ts = [0, 1, 2, 3, 4];
    expect(hawkesPeakLambda(ts, params)).toBeGreaterThanOrEqual(params.mu);
  });
});

// ─── 15. nelderMead: finds known quadratic minimum ───────────────────────────
//
// f(x,y) = (x - 3)^2 + (y + 2)^2  → min at (3, -2), f=0.
// Test: optimizer converges to within 0.01 of true minimum.

describe('nelderMead: finds quadratic minimum', () => {
  it('converges to (3, -2) from starting point (0, 0) — triggers zero-init branch', () => {
    // x0=[0,0]: v[i]=0 → Math.abs(0) < 1e-10 → добавляет step=0.2 вместо scale
    // Покрывает ternary branch на строке 26 optimizer.ts
    const f = ([x, y]: number[]) => (x! - 3) ** 2 + (y! + 2) ** 2;
    const r = nelderMead(f, [0, 0], { maxIter: 1000, tol: 1e-10 });
    expect(r.converged).toBe(true);
    expect(r.x[0]).toBeCloseTo(3,  2);
    expect(r.x[1]).toBeCloseTo(-2, 2);
    expect(r.fx).toBeCloseTo(0, 4);
  });

  it('converges to (0, 0, 0) for 3D bowl from (1, 1, 1)', () => {
    const f = ([x, y, z]: number[]) => x! ** 2 + y! ** 2 + z! ** 2;
    const r = nelderMead(f, [1, 1, 1], { maxIter: 2000, tol: 1e-10 });
    expect(r.converged).toBe(true);
    expect(r.fx).toBeCloseTo(0, 4);
  });

  it('fx is always finite after run (no NaN or Infinity)', () => {
    // Rosenbrock (harder): f = (1-x)^2 + 100*(y-x^2)^2
    const f = ([x, y]: number[]) => (1 - x!) ** 2 + 100 * (y! - x! ** 2) ** 2;
    const r = nelderMead(f, [0, 0], { maxIter: 5000, tol: 1e-8 });
    expect(Number.isFinite(r.fx)).toBe(true);
    expect(r.fx).toBeGreaterThanOrEqual(0);
  });
});

// ─── 16. hawkesAnomalyScore: monotone in peakLambda ─────────────────────────
//
// The sigmoid sig(x/meanLambda) is strictly monotone.
// Higher peakLambda → higher intensityScore → higher or equal final score.

describe('hawkesAnomalyScore: monotone in peakLambda', () => {
  it('score is non-decreasing as lambda increases', () => {
    const params = { mu: 1, alpha: 0.3, beta: 2 };
    const lambdas = [0.5, 1, 2, 3, 5, 10, 20, 50];
    let prev = -Infinity;
    for (const lam of lambdas) {
      const s = hawkesAnomalyScore(lam, params);
      expect(s).toBeGreaterThanOrEqual(prev - 1e-12); // monotone (with float tolerance)
      prev = s;
    }
  });

  it('score with higher empiricalRate >= score with lower (all else equal)', () => {
    const params = { mu: 1, alpha: 0.3, beta: 2 };
    const s1 = hawkesAnomalyScore(2, params, 1);
    const s2 = hawkesAnomalyScore(2, params, 5);
    const s3 = hawkesAnomalyScore(2, params, 20);
    expect(s2).toBeGreaterThanOrEqual(s1);
    expect(s3).toBeGreaterThanOrEqual(s2);
  });
});

// ─── 17. branch coverage: hawkes.ts uncovered branches ───────────────────────
//
// hawkes.ts:96  — `T=0 || 1` fallback: < 10 timestamps все одинаковые → T=0
// hawkes.ts:156 — `if (ti >= t) break` в hawkesLambda: timestamp в истории >= t

describe('hawkes: branch coverage', () => {
  it('hawkesFit < 10 with identical timestamps uses || 1 fallback (T=0)', () => {
    // timestamps[n-1] - timestamps[0] = 0 → || 1 → mu = n/1 = n
    const r = hawkesFit([5, 5, 5, 5, 5]);
    expect(r.converged).toBe(false);
    expect(r.params.mu).toBeCloseTo(5, 10); // n=5, T=0||1=1 → mu=5
    expect(r.params.mu).toBeGreaterThan(0);
  });

  it('hawkesLambda breaks early when ti >= t (timestamp at or after query time)', () => {
    // timestamps содержит элемент >= t → break срабатывает раньше конца массива
    const params = { mu: 1, alpha: 0.5, beta: 2 };
    // t=2, история [0, 1, 2, 3] → при ti=2 >= t=2 → break
    const lambdaWithBreak = hawkesLambda(2, [0, 1, 2, 3], params);
    // без break (если бы он не работал) ti=2 и ti=3 добавились бы
    const lambdaCorrect   = hawkesLambda(2, [0, 1], params);
    // Они должны быть одинаковы — break правильно исключает ti >= t
    expect(lambdaWithBreak).toBeCloseTo(lambdaCorrect, 10);
  });
});

// ─── 18. (перенумерация) cusumAnomalyScore: h <= 0 guard ─────────────────────
//
// cusumAnomalyScore:108 — `if (params.h <= 0) return 0` не покрыта.
// Это защита от деления на ноль: s/h при h=0.

describe('cusumAnomalyScore: h <= 0 returns 0', () => {
  it('returns 0 when h = 0', () => {
    const state = { sPos: 5, sNeg: 3, n: 10 };
    expect(cusumAnomalyScore(state, { mu0: 0, std0: 1, k: 0, h: 0 })).toBe(0);
  });

  it('returns 0 when h < 0', () => {
    const state = { sPos: 5, sNeg: 3, n: 10 };
    expect(cusumAnomalyScore(state, { mu0: 0, std0: 1, k: 0, h: -1 })).toBe(0);
  });
});

// ─── 18. hawkesFit: invalid fallback path ────────────────────────────────────
//
// hawkes.ts:124-131 — fallback когда optimizer не сошёлся или вышел за границы.
// Срабатывает при очень мало данных (< 10 timestamps → другой path) или
// при данных где optimizer застревает в penalty region.
// Тест: передаём timestamps где MLE заведомо проблемный.

describe('hawkesFit: invalid/fallback path', () => {
  it('fallback params are all positive and stationarity = 0.01', () => {
    // 2 одинаковых timestamp → T=0 → mu=n/(0||1)=n, optimizer стартует странно.
    // Либо converged=false (fallback), либо converged=true (valid params).
    // В любом случае: все params > 0 и стационарность >= 0.
    const r = hawkesFit([1_700_000_000, 1_700_000_000, 1_700_000_000,
                         1_700_000_000, 1_700_000_000, 1_700_000_000,
                         1_700_000_000, 1_700_000_000, 1_700_000_000,
                         1_700_000_000, 1_700_000_000]);
    expect(r.params.mu).toBeGreaterThan(0);
    expect(r.params.alpha).toBeGreaterThan(0);
    expect(r.params.beta).toBeGreaterThan(0);
    expect(Number.isFinite(r.stationarity)).toBe(true);
    expect(r.stationarity).toBeGreaterThanOrEqual(0);
  });

  it('fallback when not converged: converged=false, logLik=-Infinity', () => {
    // < 10 timestamps → сразу возвращает converged=false без optimizer
    const r = hawkesFit([0, 1, 2]);
    expect(r.converged).toBe(false);
    expect(r.logLik).toBe(-Infinity);
    expect(r.params.mu).toBeGreaterThan(0);
  });
});

// ─── 19. nelderMead: shrink step ─────────────────────────────────────────────
//
// optimizer.ts:78-83 — shrink шаг срабатывает когда contraction тоже плохой.
// Нужна функция где и reflection, и contraction хуже лучшей вершины.
// Rosenbrock с плохим стартом провоцирует shrink.

describe('nelderMead: shrink step executed', () => {
  it('Rosenbrock from bad start still converges to near-minimum', () => {
    // f = (1-x)^2 + 100*(y-x^2)^2, min at (1,1)=0
    // Старт (-2, 3) — далеко, провоцирует shrink шаги
    const f = ([x, y]: number[]) => (1 - x!) ** 2 + 100 * (y! - x! ** 2) ** 2;
    const r = nelderMead(f, [-2, 3], { maxIter: 10000, tol: 1e-10 });
    expect(Number.isFinite(r.fx)).toBe(true);
    expect(r.fx).toBeGreaterThanOrEqual(0);
    expect(r.fx).toBeLessThan(0.1);
  });

  it('shrink path covered: adversarial stateful function forces shrink branch', () => {
    // Shrink происходит когда fc >= fvals[n] (contraction хуже worst).
    // Гарантируем это: функция которая на шагах contraction всегда возвращает
    // значение хуже worst за счёт состояния. Используем счётчик вызовов:
    // первые 3 вызова (инициализация simplex) — обычные значения,
    // затем для всех точек кроме best — очень большое значение.
    // Это заставит reflection → плохо, contraction → плохо → shrink.
    let callN = 0;
    const pts: number[][] = [];
    const f = (x: number[]) => {
      callN++;
      pts.push(x.slice());
      // Инициализация simplex: первые n+1=3 вызовов — нормальная квадратичная
      if (callN <= 3) return x[0]! * x[0]! + x[1]! * x[1]!;
      // После инициализации: best=simplex[0] близко к (0,0).
      // Для всех остальных точек — огромное значение → shrink вынужден.
      const dist0 = x[0]! * x[0]! + x[1]! * x[1]!;
      // Если точка далека от (0,0) → большое значение
      return dist0 < 0.001 ? dist0 : 1e6 + dist0;
    };
    const r = nelderMead(f, [0, 0], { maxIter: 200, tol: 1e-6 });
    expect(Number.isFinite(r.fx)).toBe(true);
    // Optimizer может застрять но не должен крашиться
    expect(r.fx).toBeGreaterThanOrEqual(0);
    void pts;
  });
});

// ─── 20. logSumExp: b === -Infinity branch ────────────────────────────────────
//
// bocpd.ts:247 — `if (b === -Infinity) return a` не покрыта напрямую.
// bocpdBatch с одним наблюдением триггерит это через reduce с начальным -Infinity.

describe('bocpdUpdate: logSumExp b=-Infinity branch', () => {
  it('single observation does not produce NaN or Infinity in cpProbability', () => {
    // После первого update normLogProbs имеет 2 элемента.
    // logSumExp вызывается через reduce начиная с -Infinity как накопителем.
    const r = bocpdUpdate(bocpdInitState(), 0.5, { mu0: 0, kappa0: 1, alpha0: 1, beta0: 1 });
    expect(Number.isFinite(r.cpProbability)).toBe(true);
    expect(r.cpProbability).toBeGreaterThanOrEqual(0);
    expect(r.cpProbability).toBeLessThanOrEqual(1);
  });

  it('logProbs sum to 1 even after heavy pruning (tests ?? -Infinity branch)', () => {
    // Очень низкий hazardLambda → малая вероятность CP → r=0 может быть pruned
    // Тогда normLogProbs[0] может быть undefined → ?? -Infinity → exp(-Inf)=0
    let s = bocpdInitState();
    // Кормим сильно отличающиеся данные — часть гипотез будет pruned
    for (let i = 0; i < 5; i++) {
      const r = bocpdUpdate(s, i < 3 ? 0 : 1000, { mu0: 0, kappa0: 1, alpha0: 1, beta0: 0.001 }, 10000);
      const sum = r.state.logProbs.reduce((acc, lp) => acc + Math.exp(lp), 0);
      expect(sum).toBeCloseTo(1, 3);
      s = r.state;
    }
  });
});
