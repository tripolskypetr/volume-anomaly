/**
 * False-positive tests — сценарии, где детектор НЕ должен срабатывать.
 *
 * Каждый describe описывает один реальный рыночный паттерн который выглядит
 * подозрительно, но аномалией не является. Тест документирует, что библиотека
 * с этим справляется.
 *
 * Если тест начинает падать — это регрессия: детектор стал выдавать сигнал
 * на нормальных данных.
 */

import { describe, it, expect } from 'vitest';
import { detect, VolumeAnomalyDetector } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';

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

// ─── 1. Лёгкий дрейф объёма (постепенное увеличение, не скачок) ──────────────
//
// Рынок постепенно оживает: каждые 60 секунд добавляется +10% объёма.
// Это нормальная сессионная динамика — не аномалия.
// Детектор должен молчать, потому что Hawkes обучен на той же «медленной»
// истории; CUSUM/BOCPD не видят резкого сдвига imbalance.

describe('false positive: gradual volume drift', () => {
  it('slow volume ramp-up does not trigger at threshold 0.75', () => {
    const hist: IAggregatedTradeData[] = [];
    // История: 400 трейдов, объём линейно растёт от 1 до 2 за 400 секунд
    for (let i = 0; i < 400; i++) {
      const qty = 1 + (i / 400);
      hist.push(trade(i * 1000, qty, i % 2 === 0));
    }

    // Текущее окно: продолжение того же дрейфа (объём ~2)
    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 100; i++) {
      hist.push(trade(i * 1000, qty_for(i, 400, 500), i % 2 === 0));
    }
    function qty_for(i: number, base: number, total: number) { return 1 + ((base + i) / total); }
    for (let i = 0; i < 80; i++) {
      rec.push(trade(400_000 + i * 1000, qty_for(i, 400, 500), i % 2 === 0));
    }

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 2. Периодические всплески в рамках нормы (высокочастотный маркет) ────────
//
// На ликвидном рынке каждые ~5 секунд бывает кластер из 3–5 трейдов.
// Это нормальный паттерн HFT/маркетмейкеров.
// Если обучающая история содержит такую же периодичность — детектор не должен
// срабатывать на ней же в detection window.

describe('false positive: periodic micro-clusters (HFT pattern)', () => {
  it('recurring 3-trade clusters every 5s do not trigger when baseline is the same', () => {
    // Строим историю с паттерном: 3 трейда в 10ms, затем тишина 5000ms
    function makeClusteredStream(n: number, startTs: number): IAggregatedTradeData[] {
      const trades: IAggregatedTradeData[] = [];
      let ts = startTs;
      for (let i = 0; i < n; i++) {
        // кластер из 3
        trades.push(trade(ts,     1, true));
        trades.push(trade(ts + 5, 1, false));
        trades.push(trade(ts + 10, 1, true));
        ts += 5_000; // 5 секунд до следующего кластера
      }
      return trades;
    }

    const hist = makeClusteredStream(200, 0);       // ~200 кластеров = 600 трейдов
    const rec  = makeClusteredStream(30, 1_000_000); // тот же паттерн в detection

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 3. Односторонний поток без rate-аномалии (медленный trending market) ─────
//
// Trending рынок: 75% buy aggressors, но rate нормальный (1 трейд/сек).
// imbalance будет ~0.5, но это «новая норма» если обучающая история такая же.
// Детектор не должен кричать «аномалия», потому что CUSUM обучен на той же норме.

describe('false positive: trending market (high imbalance, normal rate)', () => {
  it('persistent 75% buy bias does not trigger when baseline is equally biased', () => {
    const rng = makeLCG(0xABCDABCD);

    // История: 500 трейдов, 75% buy aggressors, интервал ~1000ms
    const hist = buildStream(500, 0, 1000, 0.75, rng);

    // Текущее окно: тот же режим (75% buy, нормальный rate)
    const rec  = buildStream(150, 600_000, 1000, 0.75, rng);

    const r = detect(hist, rec, 0.75);
    // imbalance будет ~0.5, но CUSUM обучен на этом же уровне
    expect(r.anomaly).toBe(false);
  });
});

// ─── 4. Разовый крупный трейд («кит»), после которого рынок спокойный ─────────
//
// Один трейд с qty в 100× больше среднего. Классическая «свеча» кита.
// После него — ничего. Такой трейд сдвигает imbalance за один шаг,
// но CUSUM и BOCPD видят это как единичный всплеск в окне —
// нет sustained deviation (нет аномалии ставки, нет смены режима).

describe('false positive: single whale trade', () => {
  it('single 100× trade followed by calm does not trigger', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 400; i++) {
      hist.push(trade(i * 1000, 1, i % 2 === 0));
    }

    // Detection window: 49 нормальных трейдов + 1 кит в начале
    const rec: IAggregatedTradeData[] = [];
    rec.push(trade(400_000, 100, false));  // кит — buy aggressor, qty=100
    for (let i = 1; i < 80; i++) {
      rec.push(trade(400_000 + i * 1000, 1, i % 2 === 0)); // спокойно после
    }

    const r = detect(hist, rec, 0.75);
    // Один кит — нет rate spike (windowSec≈80s, 80 трейдов → нормальный rate)
    // imbalance смещён, но CUSUM/BOCPD видят 1 аномальный step из ~70 → нет sustained
    expect(r.anomaly).toBe(false);
  });
});

// ─── 5. Открытие рынка (первые трейды дня — быстрый burst без истории) ────────
//
// В первые секунды торгов накапливаются отложенные заявки → первые 10 трейдов
// идут очень быстро. НО если обучение происходило на таких же «открытиях»,
// это нормально.
// Здесь тест: detection window содержит 10 быстрых трейдов + 90 нормальных,
// история содержит тот же паттерн «старт + норма».

describe('false positive: market open burst within known pattern', () => {
  it('fast start + calm tail does not trigger when baseline has same pattern', () => {
    // История: каждые 400 трейдов — 10 быстрых (100ms) + 390 нормальных (1000ms)
    function makeMarketOpenStream(cycles: number, startTs: number): IAggregatedTradeData[] {
      const trades: IAggregatedTradeData[] = [];
      let ts = startTs;
      for (let c = 0; c < cycles; c++) {
        for (let i = 0; i < 10; i++) { trades.push(trade(ts, 1, i % 2 === 0)); ts += 100; }
        for (let i = 0; i < 90; i++) { trades.push(trade(ts, 1, i % 2 === 0)); ts += 1000; }
      }
      return trades;
    }

    const hist = makeMarketOpenStream(5, 0);         // 5 циклов = 500 трейдов
    const rec  = makeMarketOpenStream(1, 500_000);   // ещё один такой же цикл

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 6. Симметричный всплеск объёма без imbalance (liquidity event) ───────────
//
// Крупная публичная новость вызывает одновременно buy и sell — volume растёт,
// но imbalance остаётся ≈0. В detection window в 2× больше трейдов, но 50/50.
// Должна срабатывать rate-аномалия, но НЕ должна выдавать direction.
// Как false-positive тест: direction должен быть 'neutral'.

describe('false positive: symmetric volume spike (balanced news event)', () => {
  it('2× rate with 50/50 imbalance gives confidence < 0.75 or direction=neutral', () => {
    const rng = makeLCG(0x99887766);
    const hist = buildStream(500, 0, 1000, 0.5, rng);

    // 2× rate, сбалансированный
    const rec  = buildStream(200, 700_000, 500, 0.5, rng);

    const r = detect(hist, rec, 0.75);
    // Либо не аномалия совсем, либо если срабатывает — imbalance должен быть около 0
    if (r.anomaly) {
      expect(Math.abs(r.imbalance)).toBeLessThan(0.4);
    } else {
      expect(r.anomaly).toBe(false);
    }
  });
});

// ─── 7. Стабильный overnight рынок (очень редкие трейды, низкий rate) ─────────
//
// Ночью на малоликвидном рынке: 1 трейд в минуту.
// Утром начинается обычная сессия: 1 трейд в секунду (60× rate).
// НО история обучения специально содержит утренние данные → Hawkes знает этот rate.

describe('false positive: low-to-normal rate transition (trained on same)', () => {
  it('60× intraday rate vs overnight baseline does not trigger when trained on intraday', () => {
    // История: 500 трейдов, 1 трейд/сек (нормальная сессия)
    const rng = makeLCG(0x11223344);
    const hist = buildStream(500, 0, 1000, 0.5, rng);

    // Detection window: тот же 1 трейд/сек
    const rec  = buildStream(100, 600_000, 1000, 0.5, rng);

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 8. Повторяющийся паттерн imbalance (алгоритмический маркетмейкер) ─────────
//
// Маркетмейкер работает циклами: 10 buy → 10 sell → 10 buy → ...
// В любом окне imbalance периодически бывает ~±0.9, но это нормально.
// Если обучающая история содержит тот же цикл — CUSUM обучен на нём.

describe('false positive: cyclical imbalance (algorithmic market maker)', () => {
  it('alternating buy/sell blocks within baseline range do not trigger', () => {
    // История: чередующиеся блоки по 20 трейдов buy, затем 20 sell
    function makeCyclicStream(n: number, blockSize: number, startTs: number): IAggregatedTradeData[] {
      const trades: IAggregatedTradeData[] = [];
      for (let i = 0; i < n; i++) {
        const block  = Math.floor(i / blockSize);
        const isBuy  = block % 2 === 0;
        trades.push(trade(startTs + i * 1000, 1, !isBuy)); // isBuyerMaker=true → sell aggressor
      }
      return trades;
    }

    const hist = makeCyclicStream(400, 20, 0);
    const rec  = makeCyclicStream(100, 20, 500_000);

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 9. Разные периоды — detection window длиннее базовой ─────────────────────
//
// Детектор вызван с очень длинным detection window (500 трейдов вместо типичных 100).
// Длинное окно само по себе не должно вызывать аномалию на спокойных данных.

describe('false positive: oversized detection window', () => {
  it('500-trade detection window on calm data does not trigger', () => {
    const rng  = makeLCG(0xAABBCCDD);
    const hist = buildStream(500, 0, 1000, 0.5, rng);
    const rec  = buildStream(500, 600_000, 1000, 0.5, rng); // такой же по размеру

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 10. Случайный шум — 5 независимых сидов, ни один не даёт аномалию ────────
//
// Чистый случайный поток (50% buy/50% sell, нормальный rate) не должен
// срабатывать на пороге 0.75. Тест на 5 семенах — чтобы отловить случайные
// ложные тревоги из-за случайности самих данных.

describe('false positive: random baseline streams (5 seeds)', () => {
  const SEEDS = [0x10000001, 0x20000002, 0x30000003, 0x40000004, 0x50000005];

  for (const seed of SEEDS) {
    it(`[0x${seed.toString(16)}] calm stream → anomaly=false`, () => {
      const rng  = makeLCG(seed);
      const hist = buildStream(500, 0, 1000, 0.5, rng);
      const rec  = buildStream(150, 600_000, 1000, 0.5, rng);
      const r    = detect(hist, rec, 0.75);
      expect(r.anomaly).toBe(false);
    });
  }
});

// ─── 11. Повторный вызов detect() — второй вызов не получает «остаточный» сигнал
//
// Класс создаётся один раз, train вызывается один раз, detect вызывается дважды
// с одинаковыми спокойными данными. Оба вызова должны дать одинаковый результат.
// Тест проверяет отсутствие side-effect состояния между вызовами detect().

describe('false positive: no state bleed between consecutive detect() calls', () => {
  it('two consecutive calm detect() calls return same result', () => {
    const rng  = makeLCG(0xCACACACA);
    const hist = buildStream(500, 0, 1000, 0.5, rng);

    const det = new VolumeAnomalyDetector();
    det.train(hist);

    const rng2 = makeLCG(0xCACACACA);
    buildStream(500, 0, 1000, 0.5, rng2); // consume same state as hist build
    const rec  = buildStream(100, 600_000, 1000, 0.5, rng2);

    const r1 = det.detect([...rec], 0.75);
    const r2 = det.detect([...rec], 0.75);

    expect(r1.anomaly).toBe(r2.anomaly);
    expect(r1.confidence).toBe(r2.confidence);
    expect(r1.imbalance).toBe(r2.imbalance);
    // Оба не должны быть аномалией
    expect(r1.anomaly).toBe(false);
  });
});

// ─── 12. Аномалия в истории не влияет на baseline ─────────────────────────────
//
// История содержит один burst в середине (реальная аномалия), затем возвращается
// к норме. Обучение на такой истории не должно «сломать» модель так, чтобы
// нормальный detection window вызывал ложную тревогу.

describe('false positive: anomaly in training history does not corrupt baseline', () => {
  it('model trained on history with mid-burst stays accurate on calm detection', () => {
    const rng  = makeLCG(0xF1F2F3F4);

    // 200 нормальных трейдов
    const hist1 = buildStream(200, 0, 1000, 0.5, rng);
    // burst в середине: 50 быстрых трейдов
    const burst = buildStream(50, 200_000, 50, 0.9, rng);
    // ещё 250 нормальных трейдов
    const hist2 = buildStream(250, 204_000, 1000, 0.5, rng);

    const fullHist = [...hist1, ...burst, ...hist2];

    // Спокойный detection window
    const rec = buildStream(100, 700_000, 1000, 0.5, rng);

    const r = detect(fullHist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});

// ─── 13. Очень мало трейдов в detection window (пограничный случай) ────────────
//
// Маленькое окно (10 трейдов) не должно само по себе вызывать аномалию.
// Hawkes score для 10 нормальных трейдов должен быть низким.

describe('false positive: very small detection window', () => {
  it('10 calm trades in detection window do not trigger', () => {
    const hist: IAggregatedTradeData[] = [];
    for (let i = 0; i < 300; i++) {
      hist.push(trade(i * 1000, 1, i % 2 === 0));
    }

    const rec: IAggregatedTradeData[] = [];
    for (let i = 0; i < 10; i++) {
      rec.push(trade(400_000 + i * 1000, 1, i % 2 === 0));
    }

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
    expect(Number.isFinite(r.confidence)).toBe(true);
  });
});

// ─── 14. Перекрёстные сессии (разрыв в timestamps) ────────────────────────────
//
// История за вчера (0–8h), detection window за сегодня (начинается через 16h).
// Большой временной разрыв не должен вызывать аномалию — это нормальная ночная пауза.
// hawkesLambda к концу разрыва = mu (экспоненциально затухает).

describe('false positive: cross-session gap (overnight pause)', () => {
  it('16-hour timestamp gap between history and detection does not trigger', () => {
    const rng = makeLCG(0x00112233);
    // История: 8 часов, 500 трейдов (1 трейд/57.6 сек ≈ 1 трейд/мин)
    const hist = buildStream(500, 0, 57_600, 0.5, rng);

    // Detection: следующий день (+16h = 57_600_000ms gap)
    const gapMs = 57_600_000; // 16 часов
    const lastHistTs = hist[hist.length - 1]!.timestamp;
    const rec  = buildStream(100, lastHistTs + gapMs, 57_600, 0.5, rng);

    const r = detect(hist, rec, 0.75);
    expect(r.anomaly).toBe(false);
  });
});
