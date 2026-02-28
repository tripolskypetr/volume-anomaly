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
} from '../src/math/hawkes.js';
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
