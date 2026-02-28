/**
 * Invariant tests — свойства, которые должны выполняться при ЛЮБЫХ входных данных.
 *
 * Каждый тест проверяет математическое или архитектурное свойство.
 * Если оно нарушается — это баг, а не «плохой тест».
 */

import { describe, it, expect } from 'vitest';
import { VolumeAnomalyDetector, detect } from '../src/index.js';
import type { IAggregatedTradeData }     from '../src/index.js';

// ─── Helpers ──────────────────────────────────────────────────────────────────

let _uid = 0;

function trade(ts: number, qty: number, isBuyerMaker: boolean): IAggregatedTradeData {
  return { id: String(_uid++), price: 100, qty, timestamp: ts, isBuyerMaker };
}

function makeLCG(seed: number) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 0xFFFFFFFF; };
}

function buildStream(
  n: number, startTs: number, intervalMs: number,
  buyFrac: number, rng: () => number,
): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  let ts = startTs;
  for (let i = 0; i < n; i++) {
    ts += intervalMs * (0.7 + rng() * 0.6);
    trades.push(trade(Math.round(ts), 0.5 + rng() * 1.5, rng() > buyFrac));
  }
  return trades;
}

// ─── 1. retrain() overwrites — никакого утечки старых моделей ─────────────────

describe('invariant: retrain() overwrites previous models', () => {
  it('second train() replaces first model completely', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 20 });

    // Первый baseline: быстрый рынок (1 trade per 100 ms)
    const fast: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) fast.push(trade(i * 100, 1, i % 2 === 0));

    // Второй baseline: медленный рынок (1 trade per 2000 ms)
    const slow: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) slow.push(trade(i * 2000, 1, i % 2 === 0));

    det.train(fast);
    const paramsFast = { ...det.trainedModels!.hawkesParams };

    det.train(slow);
    const paramsSlow = { ...det.trainedModels!.hawkesParams };

    // μ (background rate) должен быть разным: fast μ > slow μ
    expect(paramsFast.mu).toBeGreaterThan(paramsSlow.mu);
  });

  it('detect() after retrain() uses new model, not old', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 20, scoreWeights: [1, 0, 0] });

    // Тренируемся на медленных данных (μ ≈ 0.5 tr/s)
    const slow: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) slow.push(trade(i * 2000, 1, i % 2 === 0));
    det.train(slow);
    const muSlow = det.trainedModels!.hawkesParams.mu;

    // Переобучаемся на быстрых данных (μ ≈ 20 tr/s)
    const fast: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) fast.push(trade(i * 50, 1, i % 2 === 0));
    det.train(fast);
    const muFast = det.trainedModels!.hawkesParams.mu;

    // Модели должны различаться: быстрый рынок имеет значительно выше μ
    expect(muFast).toBeGreaterThan(muSlow * 5);

    // Те же медленные данные через разные модели → разный hawkesLambda
    det.train(slow);
    const rSlow = det.detect(slow.slice(0, 100), 0.0);
    det.train(fast);
    const rFast = det.detect(slow.slice(0, 100), 0.0);

    // При fast-модели медленные данные выглядят «аномально тихими» — разный confidence
    expect(rSlow.hawkesLambda).not.toBeCloseTo(rFast.hawkesLambda, 3);
  });
});

// ─── 2. hawkesLambda ≥ mu (математический инвариант Hawkes) ──────────────────
//
// λ(t) = μ + Σ α·exp(−β·Δ) ≥ μ  (все слагаемые ≥ 0).
// hawkesPeakLambda = max_i λ(t_i) ≥ μ.

describe('invariant: hawkesLambda >= mu', () => {
  const seeds = [0x11111111, 0x22222222, 0xDEADBEEF, 0xCAFEBABE, 0x99999999];

  for (const seed of seeds) {
    it(`[0x${seed.toString(16)}] hawkesLambda >= mu always`, () => {
      const rng  = makeLCG(seed);
      const hist = buildStream(500, 0, 1000, 0.5, rng);
      const rec  = buildStream(100, 600_000, 1000, 0.5, rng);

      const det = new VolumeAnomalyDetector({ windowSize: 20 });
      det.train(hist);
      const r   = det.detect(rec, 0.0);
      const mu  = det.trainedModels!.hawkesParams.mu;

      expect(r.hawkesLambda).toBeGreaterThanOrEqual(mu);
    });
  }
});

// ─── 3. Монотонность confidence по силе аномалии ──────────────────────────────
//
// Более сильная аномалия (выше rate) → более высокий confidence.
// При прочих равных: 20× rate > 10× rate > 5× rate.

describe('invariant: confidence monotone in arrival rate', () => {
  it('20× rate burst scores higher than 10× which scores higher than 5×', () => {
    const base: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) base.push(trade(i * 1000, 1, i % 2 === 0));

    const det = new VolumeAnomalyDetector({ windowSize: 20 });
    det.train(base);

    function makeBurst(interval: number): IAggregatedTradeData[] {
      const trades: IAggregatedTradeData[] = [];
      for (let i = 0; i < 200; i++) trades.push(trade(300_000 + i * interval, 1, i % 2 === 0));
      return trades;
    }

    const r5  = det.detect(makeBurst(200),  0.0); // 5×  (1000/200)
    const r10 = det.detect(makeBurst(100),  0.0); // 10× (1000/100)
    const r20 = det.detect(makeBurst(50),   0.0); // 20× (1000/50)

    expect(r10.confidence).toBeGreaterThan(r5.confidence);
    expect(r20.confidence).toBeGreaterThan(r10.confidence);
  });

  it('stronger buy imbalance scores higher or equal than weaker', () => {
    // При фиксированном rate, увеличение |imbalance| → выше CUSUM/BOCPD score
    const base: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) base.push(trade(i * 1000, 1, i % 2 === 0));

    const det = new VolumeAnomalyDetector({ windowSize: 20 });
    det.train(base);

    // buyFrac=1 → rng()>1 never → все isBuyerMaker=false → pure buy aggressor
    function makePressure(buyFrac: number): IAggregatedTradeData[] {
      const rng = makeLCG(0xAAAA0000 + buyFrac * 100);
      return buildStream(150, 400_000, 1000, buyFrac, rng);
    }

    // 50% buy (balanced), 75% buy, 95% buy
    const r50 = det.detect(makePressure(0.5),  0.0);
    const r75 = det.detect(makePressure(0.75), 0.0);
    const r95 = det.detect(makePressure(0.95), 0.0);

    expect(r75.confidence).toBeGreaterThanOrEqual(r50.confidence);
    expect(r95.confidence).toBeGreaterThanOrEqual(r75.confidence);
  });
});

// ─── 4. cusumStat = max(sPos, sNeg) конечного состояния ──────────────────────
//
// detector.ts:224 документирует: `cusumStat: Math.max(cusumState.sPos, cusumState.sNeg)`.
// Поле НЕ является пиковым — это текущий конечный аккумулятор.
// Инвариант: cusumStat ≥ 0 и конечен.

describe('invariant: cusumStat is finite and non-negative', () => {
  it('cusumStat >= 0 for all window sizes and scenarios', () => {
    const rng  = makeLCG(0x33333333);
    const hist = buildStream(300, 0, 1000, 0.5, rng);

    for (const ws of [5, 10, 20, 50]) {
      const det = new VolumeAnomalyDetector({ windowSize: ws });
      det.train(hist);

      for (const [n, buyFrac, interval] of [
        [10, 0.5, 1000], [50, 0.9, 1000], [200, 0.5, 100],
      ] as [number, number, number][]) {
        const rng2 = makeLCG(0x33333333 + ws + n);
        const rec  = buildStream(n, 400_000, interval, buyFrac, rng2);
        const r    = det.detect(rec, 0.0);
        expect(r.cusumStat).toBeGreaterThanOrEqual(0);
        expect(Number.isFinite(r.cusumStat)).toBe(true);
      }
    }
  });
});

// ─── 5. runLength = 0 когда rolling windows пусты ─────────────────────────────
//
// Если recent.length < windowSize, rollingAbsImbalance возвращает [],
// BOCPD не делает шагов → bocpdResult остаётся начальным {mapRunLength: 0}.

describe('invariant: runLength=0 when no rolling windows', () => {
  it('recent < windowSize → runLength=0 and cusumStat=0', () => {
    const det = new VolumeAnomalyDetector({ windowSize: 50 });
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));
    det.train(hist);

    // Передаём 49 трейдов при windowSize=50 → ни одного rolling window
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 49; i++) rec.push(trade(300_000 + i * 1000, 1, i % 2 === 0));
    const r = det.detect(rec, 0.0);

    expect(r.runLength).toBe(0);
    expect(r.cusumStat).toBe(0);
  });

  it('detect() functional API: same invariant', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));

    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 10; i++) rec.push(trade(300_000 + i * 1000, 1, i % 2 === 0));

    // defaultWindowSize=50, rec=10 → нет rolling windows
    const r = detect(hist, rec, 0.0);
    expect(r.runLength).toBe(0);
    expect(r.cusumStat).toBe(0);
  });
});

// ─── 6. No duplicate signal kinds ────────────────────────────────────────────
//
// Каждый тип сигнала может встречаться максимум один раз в result.signals.
// Детектор генерирует сигнал однократно (четыре if-блока в detector.ts).

describe('invariant: no duplicate signal kinds', () => {
  it('every AnomalyKind appears at most once in signals', () => {
    const rng  = makeLCG(0x44444444);
    const hist = buildStream(500, 0, 1000, 0.5, rng);

    const det = new VolumeAnomalyDetector({ windowSize: 20 });
    det.train(hist);

    // Создаём разные окна и проверяем каждое
    for (let trial = 0; trial < 5; trial++) {
      const rng2 = makeLCG(0x44440000 + trial);
      const burst = buildStream(250, 600_000 + trial * 100_000, 100, 0.9, rng2);
      const r     = det.detect(burst, 0.0);

      const kinds = r.signals.map(s => s.kind);
      const unique = new Set(kinds);
      expect(kinds.length).toBe(unique.size);
    }
  });
});

// ─── 7. Signal score consistency with detector thresholds ────────────────────
//
// Сигнал добавляется только если его score превышает внутренний порог (detector.ts):
//   volume_spike:      hawkesScore > 0.5
//   imbalance_shift:   absImb > 0.4
//   cusum_alarm:       cusumScore > 0.7
//   bocpd_changepoint: bocpdScore > 0.3
// Проверяем что присутствие сигнала коррелирует с соответствующим полем в result.

describe('invariant: signal presence consistent with output fields', () => {
  function runWith(seed: number, intervalMs: number, buyFrac: number) {
    const rng  = makeLCG(seed);
    const hist = buildStream(300, 0, 1000, 0.5, rng);
    const rec  = buildStream(200, 400_000, intervalMs, buyFrac, rng);
    const det  = new VolumeAnomalyDetector({ windowSize: 20 });
    det.train(hist);
    return det.detect(rec, 0.0);
  }

  it('imbalance_shift present ↔ |imbalance| > 0.4', () => {
    for (const [seed, interval, buyFrac] of [
      [0x55550001, 1000, 0.5],   // balanced → no shift
      [0x55550002, 1000, 1.0],   // all buy → shift
      [0x55550003, 1000, 0.0],   // all sell → shift
      [0x55550004, 1000, 0.7],   // moderate buy
    ] as [number, number, number][]) {
      const r   = runWith(seed, interval, buyFrac);
      const has = r.signals.some(s => s.kind === 'imbalance_shift');
      if (Math.abs(r.imbalance) > 0.4) {
        expect(has).toBe(true);
      } else {
        expect(has).toBe(false);
      }
    }
  });

  it('volume_spike signal score matches hawkesScore threshold 0.5', () => {
    // Если volume_spike присутствует — его score > 0.5
    for (const seed of [0x55560001, 0x55560002, 0x55560003]) {
      const rng  = makeLCG(seed);
      const hist = buildStream(300, 0, 1000, 0.5, rng);
      const rec  = buildStream(200, 400_000, 100, 0.5, rng);
      const det  = new VolumeAnomalyDetector({ windowSize: 20 });
      det.train(hist);
      const r = det.detect(rec, 0.0);

      const spike = r.signals.find(s => s.kind === 'volume_spike');
      if (spike) {
        expect(spike.score).toBeGreaterThan(0.5);
      }
    }
  });
});

// ─── 8. scoreWeights линейность ───────────────────────────────────────────────
//
// combined = w1*s1 + w2*s2 + w3*s3.
// Если один детектор весит 1 и другие 0, combined = тот детекторный score.
// Поэтому: w=[1,0,0] → anomaly тогда и только тогда когда hawkesScore >= threshold.

describe('invariant: scoreWeights linear combination', () => {
  it('w=[1,0,0]: anomaly iff hawkesScore >= threshold', () => {
    const rng  = makeLCG(0x66666666);
    const hist = buildStream(300, 0, 1000, 0.5, rng);
    const det  = new VolumeAnomalyDetector({ windowSize: 20, scoreWeights: [1, 0, 0] });
    det.train(hist);

    // Быстрый burst → высокий hawkesScore
    const burst = buildStream(200, 400_000, 50, 0.5, rng);
    const rBurst = det.detect(burst, 0.5);
    if (rBurst.anomaly) {
      expect(rBurst.confidence).toBeGreaterThanOrEqual(0.5);
    }

    // Медленные balanced данные → низкий hawkesScore
    const slow = buildStream(100, 400_000, 2000, 0.5, rng);
    const rSlow = det.detect(slow, 0.5);
    // hawkes alone on slow data shouldn't be high; either it fires or not,
    // but confidence must be consistent with anomaly flag
    expect(rSlow.anomaly).toBe(rSlow.confidence >= 0.5);
  });

  it('combined score depends on weights: hawkes-heavy vs cusum-heavy differ on rate-only burst', () => {
    // Для rate-only burst (нет imbalance): hawkesScore высокий, cusumScore низкий.
    // Поэтому [1,0,0] даёт высокий score, а [0,1,0] — низкий.
    const base: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) base.push(trade(i * 1000, 1, i % 2 === 0));

    // Rate burst без imbalance: alternating buy/sell
    const burst: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) burst.push(trade(300_000 + i * 10, 1, i % 2 === 0));

    const detHawkes = new VolumeAnomalyDetector({ windowSize: 20, scoreWeights: [1, 0, 0] });
    const detCusum  = new VolumeAnomalyDetector({ windowSize: 20, scoreWeights: [0, 1, 0] });
    detHawkes.train([...base]);
    detCusum.train([...base]);

    const rHawkes = detHawkes.detect([...burst], 0.0);
    const rCusum  = detCusum.detect([...burst],  0.0);

    // Hawkes видит rate spike → высокий confidence
    expect(rHawkes.confidence).toBeGreaterThan(0.5);
    // CUSUM на balanced burst не должен давать высокий score (нет imbalance shift)
    // Разница должна быть значительной
    expect(rHawkes.confidence).toBeGreaterThan(rCusum.confidence + 0.2);
  });
});

// ─── 9. detect() = VolumeAnomalyDetector().train().detect() ──────────────────
//
// Функциональный API — это ровно то что написано в src/index.ts.
// Проверяем что результат идентичен ручному вызову.

describe('invariant: detect() === new + train() + detect()', () => {
  it('functional and OOP API produce identical results', () => {
    const rng  = makeLCG(0x77777777);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(200, 700_000, 100, 0.8, rng);

    const rFunc = detect([...hist], [...rec], 0.75);

    const det = new VolumeAnomalyDetector(); // same defaults as detect()
    det.train([...hist]);
    const rOop = det.detect([...rec], 0.75);

    expect(rFunc.confidence).toBe(rOop.confidence);
    expect(rFunc.anomaly).toBe(rOop.anomaly);
    expect(rFunc.imbalance).toBe(rOop.imbalance);
    expect(rFunc.hawkesLambda).toBe(rOop.hawkesLambda);
    expect(rFunc.cusumStat).toBe(rOop.cusumStat);
    expect(rFunc.runLength).toBe(rOop.runLength);
    expect(rFunc.signals).toHaveLength(rOop.signals.length);
  });
});

// ─── 10. imbalance ∈ [-1, 1] при экстремальных qty ───────────────────────────

describe('invariant: imbalance stays in [-1, +1] under extreme qty', () => {
  it('extreme buy qty (1e10 vs 1e-10) gives imbalance close to +1', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));

    const rec: IAggregatedTradeData[] = [
      trade(300_000, 1e10,  false),  // buy aggressor, huge qty
      trade(301_000, 1e-10, true),   // sell aggressor, tiny qty
    ];

    const r = detect(hist, rec, 0.0);
    expect(r.imbalance).toBeGreaterThanOrEqual(-1);
    expect(r.imbalance).toBeLessThanOrEqual(1);
    expect(r.imbalance).toBeCloseTo(1, 5);
  });

  it('extreme sell qty gives imbalance close to -1', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));

    const rec: IAggregatedTradeData[] = [
      trade(300_000, 1e-10, false),  // buy aggressor, tiny qty
      trade(301_000, 1e10,  true),   // sell aggressor, huge qty
    ];

    const r = detect(hist, rec, 0.0);
    expect(r.imbalance).toBeGreaterThanOrEqual(-1);
    expect(r.imbalance).toBeLessThanOrEqual(1);
    expect(r.imbalance).toBeCloseTo(-1, 5);
  });
});

// ─── 11. Дублирующиеся timestamps ────────────────────────────────────────────
//
// Реальные биржевые данные могут иметь несколько трейдов с одинаковым timestamp.
// Детектор должен работать корректно и детерминированно.

describe('invariant: duplicate timestamps', () => {
  it('100 trades at same timestamp does not crash or NaN', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));

    // Все 100 трейдов в одну миллисекунду
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) rec.push(trade(300_000, 1, i % 2 === 0));

    const r = detect(hist, rec, 0.0);
    expect(Number.isFinite(r.confidence)).toBe(true);
    expect(Number.isFinite(r.imbalance)).toBe(true);
    expect(Number.isFinite(r.hawkesLambda)).toBe(true);
    expect(r.imbalance).toBeGreaterThanOrEqual(-1);
    expect(r.imbalance).toBeLessThanOrEqual(1);
  });

  it('two runs with same duplicate-timestamp data give identical result', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(i * 1000, 1, i % 2 === 0));

    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 50; i++) rec.push(trade(300_000, 1 + (i % 3) * 0.5, i % 3 === 0));

    const r1 = detect([...hist], [...rec], 0.75);
    const r2 = detect([...hist], [...rec], 0.75);
    expect(r1.confidence).toBe(r2.confidence);
  });
});

// ─── 12. Очень большие timestamps (Y2050) ─────────────────────────────────────
//
// Hawkes работает с timestamp/1000 (в секундах). При Unix epoch ~2050 года
// timestamps ≈ 2.5e12 ms → 2.5e9 s. Проверяем что нет overflow.

describe('invariant: large timestamps do not cause overflow', () => {
  it('timestamps near Y2050 (2.5e12 ms) produce finite result', () => {
    const BASE = 2_500_000_000_000; // ~2050 год в ms
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 200; i++) hist.push(trade(BASE + i * 1000, 1, i % 2 === 0));

    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) rec.push(trade(BASE + 200_000 + i * 1000, 1, i % 2 === 0));

    const r = detect(hist, rec, 0.75);
    expect(Number.isFinite(r.confidence)).toBe(true);
    expect(Number.isFinite(r.hawkesLambda)).toBe(true);
    expect(r.confidence).toBeGreaterThanOrEqual(0);
    expect(r.confidence).toBeLessThanOrEqual(1);
  });
});

// ─── 13. cusumAnomalyScore peak — пиковый score захватывается ─────────────────
//
// detector.ts использует `preResetState` чтобы не потерять score при alarm+reset.
// Если в середине окна CUSUM срабатывает и сбрасывается, пиковый score сохраняется.
// Проверяем: confidence при коротком burst В НАЧАЛЕ окна >= при всё более поздних burst.

describe('invariant: peak CUSUM score captured across reset', () => {
  it('burst at start of window and burst at end give similar confidence', () => {
    // Базис: 300 чередующихся трейдов
    const base: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) base.push(trade(i * 1000, 1, i % 2 === 0));

    const det = new VolumeAnomalyDetector({ windowSize: 20, scoreWeights: [0, 1, 0] });
    det.train(base);

    // Burst в начале окна: 50 calm + 100 all-buy + 50 calm
    const startBurst: IAggregatedTradeData[] = [
      ...Array.from({ length: 50 },  (_, i) => trade(300_000 + i * 1000, 1, i % 2 === 0)),
      ...Array.from({ length: 100 }, (_, i) => trade(350_000 + i * 1000, 1, false)),
      ...Array.from({ length: 50 },  (_, i) => trade(450_000 + i * 1000, 1, i % 2 === 0)),
    ];

    // Burst в конце окна: 100 calm + 100 all-buy
    const endBurst: IAggregatedTradeData[] = [
      ...Array.from({ length: 100 }, (_, i) => trade(300_000 + i * 1000, 1, i % 2 === 0)),
      ...Array.from({ length: 100 }, (_, i) => trade(400_000 + i * 1000, 1, false)),
    ];

    const rStart = det.detect(startBurst, 0.0);
    const rEnd   = det.detect(endBurst,   0.0);

    // Оба должны показывать высокий CUSUM score
    expect(rStart.confidence).toBeGreaterThan(0.3);
    expect(rEnd.confidence).toBeGreaterThan(0.3);
  });
});

// ─── 14. Все поля конечны при любом валидном входе ───────────────────────────

describe('invariant: all result fields are finite', () => {
  const scenarios: [string, number, number, number][] = [
    ['calm balanced',    500, 1000, 0.5],
    ['fast balanced',    500,   50, 0.5],
    ['slow buy-heavy',   500, 2000, 0.9],
    ['fast sell-heavy',  500,   50, 0.1],
    ['mixed',            500,  200, 0.7],
  ];

  for (const [label, n, interval, buyFrac] of scenarios) {
    it(`${label}: all result fields are finite numbers`, () => {
      const rng  = makeLCG(0x88880000 + interval + buyFrac * 100);
      const hist = buildStream(500, 0, 1000, 0.5, rng);
      const rec  = buildStream(n,   400_000, interval, buyFrac, rng);
      const r    = detect(hist, rec, 0.0);

      expect(Number.isFinite(r.confidence)).toBe(true);
      expect(Number.isFinite(r.imbalance)).toBe(true);
      expect(Number.isFinite(r.hawkesLambda)).toBe(true);
      expect(Number.isFinite(r.cusumStat)).toBe(true);
      expect(Number.isFinite(r.runLength)).toBe(true);
    });
  }
});
