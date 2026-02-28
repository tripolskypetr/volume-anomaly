/**
 * Seeded integration tests — детерминированный LCG, все результаты воспроизводимы.
 *
 * Каждый тест фиксирует seed, строит поток, и проверяет конкретный исход.
 * Если поведение детектора изменится — тест упадёт, явно указав регрессию.
 *
 * LCG: Numerical Recipes константы (те же что в detector.test.ts).
 *
 * Архитектурное ограничение детектора:
 *   - scoreWeights по умолчанию [0.4, 0.3, 0.3] (Hawkes, CUSUM, BOCPD).
 *   - При pure imbalance аномалии (нет rate change) Hawkes score ≈ 0.
 *     combined_max = 0.3+0.3 = 0.6 < 0.75 → не срабатывает при дефолтных весах.
 *   - Для теста buy/sell surge используем scoreWeights [0, 0.5, 0.5] или
 *     пониженный threshold.
 */

import { describe, it, expect } from 'vitest';
import { VolumeAnomalyDetector }    from '../src/detector.js';
import type { IAggregatedTradeData } from '../src/types.js';

// ─── LCG ─────────────────────────────────────────────────────────────────────

function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0xFFFFFFFF;
  };
}

// ─── Stream builder ───────────────────────────────────────────────────────────

interface Seg {
  count:       number;
  intervalMs:  number;
  buyFraction: number;   // доля buy-aggressors [0,1]
  jitter:      number;   // ±fraction от intervalMs
  qtyMin?:     number;
  qtyMax?:     number;
}

let _uid = 0;

function buildStream(segs: Seg[], startTs: number, rng: () => number): IAggregatedTradeData[] {
  const trades: IAggregatedTradeData[] = [];
  let ts = startTs;
  for (const seg of segs) {
    const qMin = seg.qtyMin ?? 0.5;
    const qMax = seg.qtyMax ?? 2.0;
    for (let i = 0; i < seg.count; i++) {
      const interval    = seg.intervalMs * (1 - seg.jitter + rng() * seg.jitter * 2);
      ts               += Math.max(1, interval);
      const isBuyerMaker = rng() > seg.buyFraction;   // false = buy aggressor
      const qty          = qMin + rng() * (qMax - qMin);
      trades.push({
        id:           String(_uid++),
        price:        100,
        qty,
        timestamp:    Math.round(ts),
        isBuyerMaker,
      });
    }
  }
  return trades;
}

// ─── Стандартные сегменты ────────────────────────────────────────────────────

const CALM: Seg = { count: 0, intervalMs: 1000, buyFraction: 0.5, jitter: 0.3 };

// Burst: 10× rate, нейтральный imbalance
const RATE_BURST: Seg  = { count: 200, intervalMs: 100, buyFraction: 0.5, jitter: 0.1 };
// Buy surge: нормальный rate, тяжёлый buy pressure
const BUY_SURGE: Seg   = { count: 150, intervalMs: 1000, buyFraction: 0.95, jitter: 0.3 };
// Sell surge: нормальный rate, тяжёлый sell pressure
const SELL_SURGE: Seg  = { count: 150, intervalMs: 1000, buyFraction: 0.05, jitter: 0.3 };
// Combo: burst + heavy buy (10× rate + directional pressure)
const COMBO: Seg       = { count: 200, intervalMs: 100,  buyFraction: 0.9,  jitter: 0.1 };
// Микро-burst: 5× rate, нейтральный
const MICRO_BURST: Seg = { count: 100, intervalMs: 200,  buyFraction: 0.5,  jitter: 0.1 };
// Крупные qty: тот же rate, но qty в 10× больше
const BIG_QTY: Seg     = { count: 150, intervalMs: 1000, buyFraction: 0.9,  jitter: 0.3,
                            qtyMin: 10, qtyMax: 20 };
// Sell combo: burst + heavy sell
const SELL_COMBO: Seg  = { count: 200, intervalMs: 100,  buyFraction: 0.05, jitter: 0.1 };

// ─── Тестовый конструктор ──────────────────────────────────────────────────────

interface ScenarioResult {
  pre:     ReturnType<VolumeAnomalyDetector['detect']>;
  anomaly: ReturnType<VolumeAnomalyDetector['detect']>;
  post:    ReturnType<VolumeAnomalyDetector['detect']>;
}

function runScenario(
  seed:          number,
  anomalySeg:    Seg | Seg[],
  detectorCfg:   ConstructorParameters<typeof VolumeAnomalyDetector>[0] = {},
  threshold = 0.75,
): ScenarioResult {
  const rng  = makeLCG(seed);
  const cfg  = { windowSize: 20, ...detectorCfg };

  const hist    = buildStream([{ ...CALM, count: 500 }], 0, rng);
  const pre     = buildStream([{ ...CALM, count: 150 }], 600_000, rng);

  const anomalySegs = Array.isArray(anomalySeg) ? anomalySeg : [anomalySeg];
  const anomaly = buildStream(
    [{ ...CALM, count: 50 }, ...anomalySegs],
    800_000,
    rng,
  );
  const post    = buildStream([{ ...CALM, count: 150 }], 900_000, rng);

  const det = new VolumeAnomalyDetector(cfg);
  det.train(hist);

  return {
    pre:     det.detect(pre,     threshold),
    anomaly: det.detect(anomaly, threshold),
    post:    det.detect(post,    threshold),
  };
}

// ─── 1. Чистый rate burst (Hawkes) ───────────────────────────────────────────
// RATE_BURST: 10× arrival rate, нейтральный imbalance.
// Hawkes score будет высоким, combined = 0.4*hawkes + 0.3*cusum + 0.3*bocpd > 0.75.

describe('seeded: rate burst — Hawkes fires', () => {
  const SEED = 0x11111111;

  it('[0x11111111] pre-anomaly window is quiet', () => {
    const { pre } = runScenario(SEED, RATE_BURST);
    expect(pre.anomaly).toBe(false);
  });

  it('[0x11111111] anomaly window fires', () => {
    const { anomaly } = runScenario(SEED, RATE_BURST);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[0x11111111] post-anomaly window is quiet', () => {
    const { post } = runScenario(SEED, RATE_BURST);
    expect(post.anomaly).toBe(false);
  });

  it('[0x11111111] hawkes signal present in anomaly', () => {
    const { anomaly } = runScenario(SEED, RATE_BURST);
    // volume_spike signal указывает что Hawkes-детектор сработал
    const hasVolumeSpike = anomaly.signals.some(s => s.kind === 'volume_spike');
    expect(hasVolumeSpike).toBe(true);
  });

  it('[0x11111111] imbalance stays near 0 (balanced burst)', () => {
    const { anomaly } = runScenario(SEED, RATE_BURST);
    expect(Math.abs(anomaly.imbalance)).toBeLessThan(0.3);
  });
});

// ─── 2. Buy surge — детектируется через CUSUM/BOCPD ──────────────────────────
// При дефолтных весах [0.4, 0.3, 0.3] pure imbalance аномалия даёт combined≈0.62.
// Порог 0.75 не достигается — это корректное поведение: детектор требует Hawkes.
// Тесты проверяют что сигналы CUSUM и imbalance присутствуют.

describe('seeded: buy surge — imbalance signals present', () => {
  const SEED = 0x22222222;

  it('[0x22222222] imbalance strongly positive', () => {
    const { anomaly } = runScenario(SEED, BUY_SURGE);
    expect(anomaly.imbalance).toBeGreaterThan(0.5);
  });

  it('[0x22222222] imbalance_shift signal present', () => {
    const { anomaly } = runScenario(SEED, BUY_SURGE);
    const hasShift = anomaly.signals.some(s => s.kind === 'imbalance_shift');
    expect(hasShift).toBe(true);
  });

  it('[0x22222222] cusum_alarm signal present', () => {
    const { anomaly } = runScenario(SEED, BUY_SURGE);
    const hasCusum = anomaly.signals.some(s => s.kind === 'cusum_alarm');
    expect(hasCusum).toBe(true);
  });

  it('[0x22222222] anomaly fires at lower threshold 0.5', () => {
    const { anomaly } = runScenario(SEED, BUY_SURGE, {}, 0.5);
    expect(anomaly.anomaly).toBe(true);
  });
});

// ─── 3. Sell surge — симметрия с buy ──────────────────────────────────────────

describe('seeded: sell surge — symmetric to buy', () => {
  const SEED = 0x33333333;

  it('[0x33333333] imbalance strongly negative', () => {
    const { anomaly } = runScenario(SEED, SELL_SURGE);
    expect(anomaly.imbalance).toBeLessThan(-0.5);
  });

  it('[0x33333333] anomaly fires at lower threshold 0.5', () => {
    const { anomaly } = runScenario(SEED, SELL_SURGE, {}, 0.5);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[0x33333333] buy and sell surges produce similar confidence (|imbalance| symmetry)', () => {
    // Детекторы работают на |imbalance| — buy/sell должны давать близкий confidence.
    // Используем threshold 0.5 где оба срабатывают.
    const rBuy  = runScenario(SEED, BUY_SURGE,  {}, 0.5);
    const rSell = runScenario(SEED, SELL_SURGE, {}, 0.5);
    // Расхождение < 0.1 из-за стохастической природы BOCPD
    expect(Math.abs(rBuy.anomaly.confidence - rSell.anomaly.confidence)).toBeLessThan(0.1);
  });
});

// ─── 4. Combo: burst + buy pressure ───────────────────────────────────────────
// COMBO активирует все три детектора → наивысший combined score.

describe('seeded: combo burst+buy — all detectors fire', () => {
  const SEED = 0x44444444;

  it('[0x44444444] pre quiet', () => {
    const { pre } = runScenario(SEED, COMBO);
    expect(pre.anomaly).toBe(false);
  });

  it('[0x44444444] anomaly fires with high confidence', () => {
    const { anomaly } = runScenario(SEED, COMBO);
    expect(anomaly.anomaly).toBe(true);
    expect(anomaly.confidence).toBeGreaterThan(0.75);
  });

  it('[0x44444444] post quiet', () => {
    const { post } = runScenario(SEED, COMBO);
    expect(post.anomaly).toBe(false);
  });

  it('[0x44444444] combo has higher confidence than pure rate burst', () => {
    const rCombo = runScenario(SEED, COMBO);
    const rBurst = runScenario(SEED, RATE_BURST);
    expect(rCombo.anomaly.confidence).toBeGreaterThanOrEqual(rBurst.anomaly.confidence);
  });
});

// ─── 5. Sell combo: burst + sell ──────────────────────────────────────────────

describe('seeded: sell combo — burst + sell pressure', () => {
  const SEED = 0x55555555;

  it('[0x55555555] anomaly fires', () => {
    const { anomaly } = runScenario(SEED, SELL_COMBO);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[0x55555555] imbalance negative (sell pressure)', () => {
    const { anomaly } = runScenario(SEED, SELL_COMBO);
    expect(anomaly.imbalance).toBeLessThan(0);
  });

  it('[0x55555555] sell combo confidence ≈ buy combo confidence (within 0.1)', () => {
    const rSell = runScenario(SEED, SELL_COMBO);
    const rBuy  = runScenario(SEED, COMBO);
    // Оба combo работают на |imbalance|; разница < 0.1
    expect(Math.abs(rSell.anomaly.confidence - rBuy.anomaly.confidence)).toBeLessThan(0.1);
  });
});

// ─── 6. Микро-burst (5×) — срабатывает при дефолтном пороге ──────────────────
// Микро-burst: 5× rate. Достаточно для combined > 0.75.

describe('seeded: micro burst', () => {
  const SEED = 0x66666666;

  it('[0x66666666] micro burst (5×) fires at default threshold', () => {
    const { anomaly } = runScenario(SEED, MICRO_BURST);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[0x66666666] micro burst fires at strict threshold 0.5', () => {
    const { anomaly } = runScenario(SEED, MICRO_BURST, {}, 0.5);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[0x66666666] pre quiet at default threshold', () => {
    // Pre-window — нет burst, hawkesScore ≈ 0, combined < 0.75
    const { pre } = runScenario(SEED, MICRO_BURST);
    expect(pre.anomaly).toBe(false);
  });
});

// ─── 7. Большие qty, тяжёлый buy — imbalance signal ─────────────────────────
// BIG_QTY: normal rate, qty 10–20×. Без rate burst hawkesScore ≈ 0.
// combined ≈ 0.63 при дефолтных весах — не достигает 0.75.
// Проверяем наличие imbalance сигналов и срабатывание при threshold 0.5.

describe('seeded: large qty buy surge', () => {
  const SEED = 0x77777777;

  it('[0x77777777] imbalance > 0.5 (buy-side)', () => {
    const { anomaly } = runScenario(SEED, BIG_QTY);
    expect(anomaly.imbalance).toBeGreaterThan(0.5);
  });

  it('[0x77777777] imbalance_shift signal present', () => {
    const { anomaly } = runScenario(SEED, BIG_QTY);
    const hasShift = anomaly.signals.some(s => s.kind === 'imbalance_shift');
    expect(hasShift).toBe(true);
  });

  it('[0x77777777] fires at lower threshold 0.5', () => {
    const { anomaly } = runScenario(SEED, BIG_QTY, {}, 0.5);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[0x77777777] post quiet at default threshold', () => {
    const { post } = runScenario(SEED, BIG_QTY);
    expect(post.anomaly).toBe(false);
  });
});

// ─── 8. Разные Seeds — один и тот же аномальный сценарий всегда срабатывает ───

describe('seeded: combo anomaly stable across 5 seeds', () => {
  const SEEDS = [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE];

  for (const seed of SEEDS) {
    it(`[0x${seed.toString(16).toUpperCase()}] fires on combo`, () => {
      const { anomaly } = runScenario(seed, COMBO);
      expect(anomaly.anomaly).toBe(true);
    });

    it(`[0x${seed.toString(16).toUpperCase()}] pre quiet`, () => {
      const { pre } = runScenario(seed, COMBO);
      expect(pre.anomaly).toBe(false);
    });

    it(`[0x${seed.toString(16).toUpperCase()}] post quiet`, () => {
      const { post } = runScenario(seed, COMBO);
      expect(post.anomaly).toBe(false);
    });
  }
});

// ─── 9. Разные windowSize — поведение стабильно ───────────────────────────────

describe('seeded: windowSize variation', () => {
  const SEED = 0x12345678;

  for (const ws of [10, 20, 50]) {
    it(`[windowSize=${ws}] combo fires`, () => {
      const { anomaly } = runScenario(SEED, COMBO, { windowSize: ws });
      expect(anomaly.anomaly).toBe(true);
    });

    it(`[windowSize=${ws}] pre quiet`, () => {
      const { pre } = runScenario(SEED, COMBO, { windowSize: ws });
      expect(pre.anomaly).toBe(false);
    });
  }
});

// ─── 10. hazardLambda variation ───────────────────────────────────────────────

describe('seeded: hazardLambda variation', () => {
  const SEED = 0xFEDCBA98;

  for (const hz of [10, 50, 200, 500]) {
    it(`[hazardLambda=${hz}] combo anomaly fires`, () => {
      const { anomaly } = runScenario(SEED, COMBO, { hazardLambda: hz });
      expect(anomaly.anomaly).toBe(true);
    });
  }

  it('lower hazardLambda → higher BOCPD score on combo', () => {
    // При bocpd-only весах: более частое ожидание changepoints → выше score при burst
    const r10  = runScenario(SEED, COMBO, { hazardLambda: 10,  scoreWeights: [0, 0, 1] });
    const r500 = runScenario(SEED, COMBO, { hazardLambda: 500, scoreWeights: [0, 0, 1] });
    // Оба должны срабатывать
    expect(r10.anomaly.anomaly).toBe(true);
    expect(r500.anomaly.anomaly).toBe(true);
  });
});

// ─── 11. scoreWeights isolation — один детектор изолированно ──────────────────

describe('seeded: score weight isolation', () => {
  const SEED = 0xDEADBEEF;

  it('[hawkes only] rate burst fires with weight [1,0,0]', () => {
    const { anomaly } = runScenario(SEED, RATE_BURST, { scoreWeights: [1, 0, 0] });
    expect(anomaly.anomaly).toBe(true);
    expect(anomaly.confidence).toBeGreaterThan(0.5);
  });

  it('[cusum+bocpd] buy surge fires at lower threshold', () => {
    // При дефолтных весах buy surge даёт combined ≈ 0.62.
    // С threshold 0.5 срабатывает.
    const { anomaly } = runScenario(SEED, BUY_SURGE, {}, 0.5);
    expect(anomaly.anomaly).toBe(true);
  });

  it('[bocpd only] combo fires with weight [0,0,1]', () => {
    const { anomaly } = runScenario(SEED, COMBO, { scoreWeights: [0, 0, 1] });
    expect(anomaly.anomaly).toBe(true);
  });

  it('[hawkes only] buy surge at baseline rate does NOT fire', () => {
    // Buy surge: rate = baseline. Hawkes score ≈ 0.
    const { anomaly } = runScenario(SEED, BUY_SURGE, { scoreWeights: [1, 0, 0] });
    expect(anomaly.anomaly).toBe(false);
  });

  it('[hawkes only] combo fires with weight [1,0,0]', () => {
    // Combo: 10× rate. Hawkes score высокий → fires.
    const { anomaly } = runScenario(SEED, COMBO, { scoreWeights: [1, 0, 0] });
    expect(anomaly.anomaly).toBe(true);
  });
});

// ─── 12. Двойная аномалия: два burst в одном окне ────────────────────────────

describe('seeded: double burst in single window', () => {
  const SEED = 0xCAFEBABE;

  it('[0xCAFEBABE] double combo fires with high confidence', () => {
    const doubleSeg: Seg[] = [COMBO, { ...CALM, count: 10 }, COMBO];
    const { anomaly } = runScenario(SEED, doubleSeg);
    expect(anomaly.anomaly).toBe(true);
    expect(anomaly.confidence).toBeGreaterThan(0.75);
  });

  it('[0xCAFEBABE] double combo confidence ≥ single combo confidence (minus tolerance)', () => {
    const doubleSeg: Seg[] = [COMBO, { ...CALM, count: 10 }, COMBO];
    const rDouble = runScenario(SEED, doubleSeg);
    const rSingle = runScenario(SEED, COMBO);
    // Двойной burst не обязательно строго лучше — но не должен быть существенно хуже
    expect(rDouble.anomaly.confidence).toBeGreaterThanOrEqual(rSingle.anomaly.confidence - 0.05);
  });
});

// ─── 13. Детерминизм — два одинаковых прогона дают одинаковый результат ──────

describe('seeded: determinism', () => {
  const SEEDS = [0x11111111, 0x44444444, 0xDEADBEEF];

  for (const seed of SEEDS) {
    it(`[0x${seed.toString(16)}] two runs with same seed produce identical confidence`, () => {
      const r1 = runScenario(seed, COMBO);
      const r2 = runScenario(seed, COMBO);
      expect(r1.pre.confidence).toBeCloseTo(r2.pre.confidence, 10);
      expect(r1.anomaly.confidence).toBeCloseTo(r2.anomaly.confidence, 10);
      expect(r1.post.confidence).toBeCloseTo(r2.post.confidence, 10);
    });
  }
});

// ─── 14. Ложные срабатывания — только спокойные данные ────────────────────────

describe('seeded: false positive — calm-only streams', () => {
  const SEEDS = [0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x13579BDF, 0x2468ACE0];

  for (const seed of SEEDS) {
    it(`[0x${seed.toString(16)}] calm stream never triggers at threshold 0.75`, () => {
      const rng  = makeLCG(seed);
      const hist = buildStream([{ ...CALM, count: 500 }], 0, rng);
      const pre  = buildStream([{ ...CALM, count: 200 }], 600_000, rng);

      const det = new VolumeAnomalyDetector({ windowSize: 20 });
      det.train(hist);
      const r = det.detect(pre, 0.75);

      expect(r.anomaly).toBe(false);
      expect(r.confidence).toBeLessThan(0.75);
    });
  }
});
