/**
 * eval.test.ts — full-day real-data evaluation sweep (opt-in, not part of CI).
 *
 * Run with:  EVAL=1 npx vitest run test/eval.test.ts
 *
 * Protocol (mimics operational usage from README):
 *   1. Parse mock/BTCUSDT-aggTrades-2025-03-01.csv (1.49 M trades, timestamps in µs).
 *   2. Bucket the day into 30-second bins; compute robust z-scores (median/MAD)
 *      of per-bucket volume and trade count over the whole day.
 *      Ground truth: STRONG anomaly = zVol ≥ 8 or zCnt ≥ 8; NORMAL = both < 3;
 *      buckets in between are a gray zone excluded from both metrics.
 *   3. For every bucket: train on the 500 trades preceding it, detect on the
 *      first ≤300 trades inside it (same shape as the realdata fixtures).
 *   4. Report: recall on STRONG, false-positive rate on NORMAL, alert rate,
 *      confidence percentiles, and which sub-detector drives false positives.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname }         from 'node:path';
import { fileURLToPath }         from 'node:url';
import { VolumeAnomalyDetector } from '../src/index.js';
import type { IAggregatedTradeData } from '../src/index.js';

const RUN = process.env['EVAL'] === '1';
const __dirname = dirname(fileURLToPath(import.meta.url));

// ─── CSV loading ──────────────────────────────────────────────────────────────

interface Columns {
  ts:    Float64Array;  // ms
  qty:   Float64Array;
  price: Float64Array;
  maker: Uint8Array;    // isBuyerMaker
  n:     number;
}

function loadCsv(): Columns {
  const raw   = readFileSync(join(__dirname, '..', 'mock', 'BTCUSDT-aggTrades-2025-03-01.csv'), 'utf8');
  const lines = raw.split('\n');
  const n     = lines.length;
  const ts    = new Float64Array(n);
  const qty   = new Float64Array(n);
  const price = new Float64Array(n);
  const maker = new Uint8Array(n);
  let count = 0;
  for (const line of lines) {
    if (!line) continue;
    // aggTradeId,price,qty,firstId,lastId,timestamp_us,isBuyerMaker,isBestMatch
    const parts = line.split(',');
    if (parts.length < 7) continue;
    ts[count]    = Number(parts[5]) / 1000; // µs → ms
    qty[count]   = Number(parts[2]);
    price[count] = Number(parts[1]);
    maker[count] = parts[6] === 'True' ? 1 : 0;
    count++;
  }
  return { ts, qty, price, maker, n: count };
}

function makeWindow(c: Columns, from: number, to: number): IAggregatedTradeData[] {
  const out: IAggregatedTradeData[] = [];
  for (let i = from; i < to; i++) {
    out.push({ id: String(i), price: 0, qty: c.qty[i]!, timestamp: c.ts[i]!, isBuyerMaker: c.maker[i] === 1 });
  }
  return out;
}

// ─── Robust stats ─────────────────────────────────────────────────────────────

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function robustZ(xs: number[]): number[] {
  const med = median(xs);
  const mad = median(xs.map((x) => Math.abs(x - med))) || 1e-12;
  return xs.map((x) => (x - med) / (1.4826 * mad));
}

function pct(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]!;
}

/**
 * Mann-Whitney AUC: probability that a random positive outranks a random
 * negative (ties get 0.5).  Score-free — depends only on the ranking, so any
 * monotone re-mapping of the score leaves it unchanged.
 */
function auc(scores: number[], labels: boolean[]): number {
  const idx = scores.map((_, i) => i).sort((a, b) => scores[a]! - scores[b]!);
  // average ranks with ties
  const ranks = new Array<number>(scores.length).fill(0);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && scores[idx[j + 1]!] === scores[idx[i]!]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]!] = avg;
    i = j + 1;
  }
  let posRankSum = 0, nPos = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i]) { posRankSum += ranks[i]!; nPos++; }
  }
  const nNeg = labels.length - nPos;
  if (nPos === 0 || nNeg === 0) return NaN;
  return (posRankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

// ─── Sweep ────────────────────────────────────────────────────────────────────

const BUCKET_MS  = 30_000;
const HIST_MS    = 30 * 60_000; // 30-minute time-based baseline
const HIST_MIN   = 500;         // skip buckets with less history than this
const HIST_CAP   = 50_000;      // bound object churn in the hottest periods
const RECENT_MAX = 300;

describe.runIf(RUN)('eval: full-day sweep on BTCUSDT-2025-03-01', () => {
  it('sweep report', () => {
    const c = loadCsv();
    console.log(`trades: ${c.n}`);

    // ── Buckets + ground truth
    const t0 = c.ts[0]!;
    const nBuckets = Math.ceil((c.ts[c.n - 1]! - t0) / BUCKET_MS);
    const vol = new Array<number>(nBuckets).fill(0);
    const cnt = new Array<number>(nBuckets).fill(0);
    const firstIdx = new Array<number>(nBuckets).fill(-1);
    const hi    = new Array<number>(nBuckets).fill(-Infinity);
    const lo    = new Array<number>(nBuckets).fill(Infinity);
    const close = new Array<number>(nBuckets).fill(0);
    const open  = new Array<number>(nBuckets).fill(0);
    for (let i = 0; i < c.n; i++) {
      const b = Math.min(nBuckets - 1, Math.floor((c.ts[i]! - t0) / BUCKET_MS));
      vol[b]! += c.qty[i]!;
      cnt[b]! += 1;
      if (firstIdx[b] === -1) { firstIdx[b] = i; open[b] = c.price[i]!; }
      const p = c.price[i]!;
      if (p > hi[b]!) hi[b] = p;
      if (p < lo[b]!) lo[b] = p;
      close[b] = p;
    }
    // Fill empty buckets forward so ranges/returns are well-defined
    for (let b = 1; b < nBuckets; b++) {
      if (close[b] === 0) { close[b] = close[b - 1]!; open[b] = close[b - 1]!; hi[b] = close[b - 1]!; lo[b] = close[b - 1]!; }
    }

    // ── Forward price response (predictive ground truth).
    // fwdRange_K(b) = (max high − min low over buckets b+1..b+K) / close(b):
    // "how much does price actually travel in the K buckets AFTER this one".
    // Scored as robust z against the trailing hour of the same statistic, so
    // "big move" means big relative to what the recent market was doing —
    // the same locally-adaptive yardstick as the volume GT below.
    const FWD_HORIZONS = [2, 10] as const;         // 1 min, 5 min
    const fwdR = FWD_HORIZONS.map(() => new Array<number>(nBuckets).fill(0));
    for (let b = 0; b < nBuckets; b++) {
      for (let k = 0; k < FWD_HORIZONS.length; k++) {
        const K = FWD_HORIZONS[k]!;
        let h = -Infinity, l = Infinity;
        for (let j = b + 1; j <= Math.min(b + K, nBuckets - 1); j++) {
          if (hi[j]! > h) h = hi[j]!;
          if (lo[j]! < l) l = lo[j]!;
        }
        fwdR[k]![b] = h > l && close[b]! > 0 ? (h - l) / close[b]! : 0;
      }
    }
    const TRAIL_FWD = 120;
    const fwdZ = FWD_HORIZONS.map(() => new Array<number>(nBuckets).fill(NaN));
    for (let k = 0; k < FWD_HORIZONS.length; k++) {
      for (let b = TRAIL_FWD; b < nBuckets - FWD_HORIZONS[k]!; b++) {
        const w   = fwdR[k]!.slice(b - TRAIL_FWD, b);
        const m   = median(w);
        const mad = Math.max(median(w.map((x) => Math.abs(x - m))), 0.1 * m, 1e-12);
        fwdZ[k]![b] = (fwdR[k]![b]! - m) / (1.4826 * mad);
      }
    }
    // Ground truth: robust z-score vs the TRAILING hour, not the global day.
    // A global yardstick labels "NY hours are busier than the overnight
    // median" as anomalous — that is a regime, not an anomaly, and no locally
    // adaptive detector (nor a human trader) should fire on it.  Anomaly =
    // deviation from the recent local norm.
    const TRAIL = 120; // 1 hour of 30s buckets
    const zVol = new Array<number>(nBuckets).fill(0);
    const zCnt = new Array<number>(nBuckets).fill(0);
    const zOk  = new Array<boolean>(nBuckets).fill(false);
    for (let b = TRAIL; b < nBuckets; b++) {
      const wv = vol.slice(b - TRAIL, b);
      const wc = cnt.slice(b - TRAIL, b);
      const mv = median(wv), mc = median(wc);
      // MAD floored at 10% of the median: heavy-tailed volume makes raw MAD
      // tiny, which would inflate z for routine wiggles.
      const madv = Math.max(median(wv.map((x) => Math.abs(x - mv))), 0.1 * mv, 1e-9);
      const madc = Math.max(median(wc.map((x) => Math.abs(x - mc))), 0.1 * mc, 1e-9);
      zVol[b] = (vol[b]! - mv) / (1.4826 * madv);
      zCnt[b] = (cnt[b]! - mc) / (1.4826 * madc);
      zOk[b]  = true;
    }

    type Label = 'strong' | 'normal' | 'gray';
    const label = (b: number): Label =>
      !zOk[b] ? 'gray'
      : zVol[b]! >= 8 || zCnt[b]! >= 8 ? 'strong'
      : zVol[b]! < 3 && zCnt[b]! < 3 ? 'normal'
      : 'gray';

    const nStrong = zVol.filter((_, b) => label(b) === 'strong').length;
    const nNormal = zVol.filter((_, b) => label(b) === 'normal').length;
    console.log(`buckets: ${nBuckets}  strong: ${nStrong}  normal: ${nNormal}  gray: ${nBuckets - nStrong - nNormal}`);

    // ── Sweep
    const CONF = 0.75;
    let tp = 0, fnCount = 0, fp = 0, tn = 0;
    const normalConf: number[] = [];
    const strongConf: number[] = [];
    const fpSignals: Record<string, number> = {};
    const missed: Array<{ b: number; conf: number; zv: number; zc: number }> = [];
    const branchings: number[] = [];
    const strongFlagged = new Map<number, boolean>();
    const rows: Array<Record<string, unknown>> = [];
    const sub = {
      normal: { hawkes: [] as number[], cusum: [] as number[], bocpd: [] as number[] },
      strong: { hawkes: [] as number[], cusum: [] as number[], bocpd: [] as number[] },
    };
    const ratios = {
      normal: { rate: [] as number[], vol: [] as number[] },
      strong: { rate: [] as number[], vol: [] as number[] },
    };

    // Per-bucket predictive join: detector confidence vs forward response.
    const pred: Array<{ b: number; lab: Label; conf: number; anomaly: boolean; ret: number; move: number }> = [];

    const started = Date.now();
    // Progress bar: one line per ~5% of the day (vitest streams stdout live,
    // so a long sweep shows movement instead of silence until the report).
    const PROG_STEP = Math.max(1, Math.floor(nBuckets / 20));
    const progress = (b: number) => {
      const frac    = b / nBuckets;
      const BAR     = 24;
      const filled  = Math.round(frac * BAR);
      const elapsed = (Date.now() - started) / 1000;
      const eta     = elapsed * (1 - frac) / Math.max(frac, 1e-9);
      console.log(
        `[${'█'.repeat(filled)}${'░'.repeat(BAR - filled)}] ${(frac * 100).toFixed(0).padStart(3)}%` +
        `  bucket ${b}/${nBuckets}  elapsed ${Math.round(elapsed)}s  ETA ${Math.round(eta)}s`,
      );
    };

    for (let b = 0; b < nBuckets; b++) {
      if (b > 0 && b % PROG_STEP === 0) progress(b);
      const lab = label(b);
      // Gray buckets are excluded from the gate metrics (ambiguous GT) but
      // still evaluated: the predictive benchmark needs full coverage.
      const fi = firstIdx[b]!;
      if (fi === -1) continue;

      // historical = trades from the 30 minutes preceding the bucket start —
      // a TIME-based baseline.  A count-based baseline ("last 500 trades")
      // adapts its duration to market pace and masks the very burst being
      // detected (hot pre-burst market → inflated ceilings).
      const bucketStartTs = t0 + b * BUCKET_MS;
      let histLo = fi;
      while (histLo > 0 && c.ts[histLo - 1]! >= bucketStartTs - HIST_MS) histLo--;
      histLo = Math.max(histLo, fi - HIST_CAP);
      if (fi - histLo < HIST_MIN) continue;

      // Operational protocol: the detector is called repeatedly as trades
      // stream in.  Emulate with sliding ≤300-trade windows stepping 150
      // trades through the bucket; the bucket counts as flagged if ANY call
      // fires.  (A single call at bucket end sees only the cooldown of a
      // burst that peaked mid-bucket; a single call at bucket start sees
      // nothing of a burst that starts later.)
      let end = fi;
      const bucketEnd = t0 + (b + 1) * BUCKET_MS;
      while (end < c.n && c.ts[end]! < bucketEnd) end++;
      if (end - fi < 5) continue; // nearly-empty bucket, nothing to detect on

      const det = new VolumeAnomalyDetector();
      det.train(makeWindow(c, histLo, fi));
      const m = det.trainedModels!;
      branchings.push(m.hawkesParams.alpha / m.hawkesParams.beta);

      let r = det.detect(makeWindow(c, Math.max(fi, end - RECENT_MAX), end), CONF);
      let move = r.moveScore; // peak predictive ranking score over calls
      let rr = 0, vr = 0; // peak fast-horizon robust z over calls (rate, vol)
      const calls: Array<{ s: number; z: number[] }> = [];
      for (let e = Math.min(fi + RECENT_MAX, end); ; e += 150) {
        e = Math.min(e, end);
        const start = Math.max(fi, e - RECENT_MAX);
        const rc = det.detect(makeWindow(c, start, e), CONF);
        if (rc.confidence > r.confidence) r = rc;
        if (rc.moveScore > move) move = rc.moveScore;
        rr = Math.max(rr, rc.stats.zRate, rc.stats.zRateSlow);
        vr = Math.max(vr, rc.stats.zVol,  rc.stats.zVolSlow);
        const spanSec = (c.ts[e - 1]! - c.ts[start]!) / 1000;
        calls.push({
          s: Math.log10(Math.max(1, spanSec / m.slowHorizonSec)),
          z: [rc.stats.zRate, rc.stats.zVol, rc.stats.zRateSlow, rc.stats.zVolSlow],
          // full per-scale vectors (one entry per m.horizonsSec)
          zr: rc.stats.zRates, zv2: rc.stats.zVols,
        });
        if (e >= end) break;
      }

      if (lab !== 'gray') {
        const bucket = lab === 'strong' ? sub.strong : sub.normal;
        bucket.hawkes.push(r.scores.hawkes);
        bucket.cusum.push(r.scores.cusum);
        bucket.bocpd.push(r.scores.bocpd);
        const rat = lab === 'strong' ? ratios.strong : ratios.normal;
        rat.rate.push(rr);
        rat.vol.push(vr);
      }
      const retBps = close[b]! > 0 && open[b]! > 0 ? Math.abs(Math.log(close[b]! / open[b]!)) * 1e4 : 0;
      pred.push({ b, lab, conf: r.confidence, anomaly: r.anomaly, ret: retBps, move });
      rows.push({
        b, lab,
        h: r.scores.hawkes, c: r.scores.cusum, p: r.scores.bocpd,
        rr, vr, zv: zVol[b]!, zc: zCnt[b]!,
        // forward price response (predictive GT): z of forward 1min/5min range
        fz1: fwdZ[0]![b]!, fz5: fwdZ[1]![b]!,
        // current-bucket |return| in bps — the naive momentum baseline
        ret: retBps,
        calls,
        // Null quantile ladders (NULLQ_PCTS percentiles): rate then vol
        // channel per horizon, plus the horizon family itself
        hs: m.horizonsSec,
        nq: m.horizonsSec.flatMap((_, k) => [
          m.channelCalib.rate[k]!.nullQ,
          m.channelCalib.vol[k]!.nullQ,
        ]),
      });

      if (lab === 'strong') {
        strongConf.push(r.confidence);
        strongFlagged.set(b, r.anomaly);
        if (r.anomaly) tp++;
        else { fnCount++; missed.push({ b, conf: r.confidence, zv: zVol[b]!, zc: zCnt[b]! }); }
      } else if (lab === 'normal') {
        normalConf.push(r.confidence);
        if (r.anomaly) {
          fp++;
          for (const s of r.signals) fpSignals[s.kind] = (fpSignals[s.kind] ?? 0) + 1;
        } else tn++;
      }
    }
    console.log(`sweep time: ${((Date.now() - started) / 1000).toFixed(1)}s`);

    // ── Report
    const recall = tp / Math.max(tp + fnCount, 1);
    const fpRate = fp / Math.max(fp + tn, 1);
    console.log('');
    console.log(`STRONG  (z≥8):  ${tp + fnCount}   detected: ${tp}   recall: ${(recall * 100).toFixed(1)}%`);
    console.log(`NORMAL  (z<3):  ${fp + tn}   false alarms: ${fp}   FP rate: ${(fpRate * 100).toFixed(2)}%`);

    // Event-level recall: consecutive strong buckets (gap ≤ 4) form one event;
    // an event counts as caught if ANY of its buckets was flagged.
    const strongBuckets = [...strongFlagged.keys()].sort((a, b2) => a - b2);
    let events = 0, eventsCaught = 0;
    for (let i = 0; i < strongBuckets.length; ) {
      let j2 = i, caught = false;
      while (j2 < strongBuckets.length && (j2 === i || strongBuckets[j2]! - strongBuckets[j2 - 1]! <= 4)) {
        if (strongFlagged.get(strongBuckets[j2]!)) caught = true;
        j2++;
      }
      events++;
      if (caught) eventsCaught++;
      i = j2;
    }
    console.log(`EVENTS (strong clusters): ${events}   caught: ${eventsCaught}   event recall: ${events ? ((eventsCaught / events) * 100).toFixed(1) : '—'}%`);
    console.log('');
    console.log(`confidence on NORMAL: P50=${pct(normalConf, 50).toFixed(3)}  P95=${pct(normalConf, 95).toFixed(3)}  P99=${pct(normalConf, 99).toFixed(3)}  max=${Math.max(...normalConf).toFixed(3)}`);
    console.log(`confidence on STRONG: P10=${pct(strongConf, 10).toFixed(3)}  P50=${pct(strongConf, 50).toFixed(3)}`);
    console.log(`FP signal breakdown: ${JSON.stringify(fpSignals)}`);
    for (const k of ['rate', 'vol'] as const) {
      const nrm = ratios.normal[k];
      const str = ratios.strong[k];
      console.log(
        `ratio ${k.padEnd(4)} NORMAL P50=${pct(nrm, 50).toFixed(2)} P95=${pct(nrm, 95).toFixed(2)} P99=${pct(nrm, 99).toFixed(2)} max=${Math.max(...nrm).toFixed(1)}` +
        `   STRONG P10=${pct(str, 10).toFixed(2)} P25=${pct(str, 25).toFixed(2)} P50=${pct(str, 50).toFixed(2)}`,
      );
    }
    for (const det of ['hawkes', 'cusum', 'bocpd'] as const) {
      const nrm = sub.normal[det];
      const str = sub.strong[det];
      console.log(
        `${det.padEnd(6)} NORMAL P50=${pct(nrm, 50).toFixed(3)} P95=${pct(nrm, 95).toFixed(3)} P99=${pct(nrm, 99).toFixed(3)}` +
        `   STRONG P10=${pct(str, 10).toFixed(3)} P50=${pct(str, 50).toFixed(3)}`,
      );
    }
    console.log(`fitted branching α/β: P50=${pct(branchings, 50).toFixed(3)}  P90=${pct(branchings, 90).toFixed(3)}`);
    if (missed.length) {
      console.log(`missed strong buckets (${missed.length}):`);
      for (const m2 of missed.slice(0, 15)) {
        console.log(`  bucket ${m2.b}  conf=${m2.conf.toFixed(3)}  zVol=${m2.zv.toFixed(1)}  zCnt=${m2.zc.toFixed(1)}`);
      }
    }

    // ── Predictive benchmark: does confidence rank FUTURE price movement?
    // The volume GT above is nearly the detector's own statistic, so recall
    // against it has a self-agreement ceiling.  Here the target is external:
    // forward 1min/5min price range, z-scored vs the trailing hour.  Reported
    // vs two reference rankers: the GT volume z itself (what a perfect
    // volume-anomaly detector could achieve) and the naive momentum baseline
    // (current-bucket |return| — "price is already moving").
    console.log('');
    console.log('── predictive (forward price response) ──');
    for (let k = 0; k < FWD_HORIZONS.length; k++) {
      const horizon = FWD_HORIZONS[k]! === 2 ? '1min' : '5min';
      const rowsOk  = pred.filter((p) => Number.isFinite(fwdZ[k]![p.b]!));
      const zs      = rowsOk.map((p) => fwdZ[k]![p.b]!);
      for (const THR of [2, 4]) {
        const moved  = zs.map((z) => z >= THR);
        const nMoved = moved.filter(Boolean).length;
        const aucConf = auc(rowsOk.map((p) => p.conf), moved);
        const aucMove = auc(rowsOk.map((p) => p.move), moved);
        const aucGt   = auc(rowsOk.map((p) => Math.max(zVol[p.b]!, zCnt[p.b]!)), moved);
        const aucRet  = auc(rowsOk.map((p) => p.ret), moved);
        let movedAndFlagged = 0, flagged = 0;
        for (let i = 0; i < rowsOk.length; i++) {
          if (rowsOk[i]!.anomaly) { flagged++; if (moved[i]) movedAndFlagged++; }
        }
        const pMovedGivenAnomaly = movedAndFlagged / Math.max(flagged, 1);
        console.log(
          `fwd ${horizon} z≥${THR}: base rate ${(100 * nMoved / zs.length).toFixed(1)}%  ` +
          `P(move|anomaly)=${(100 * pMovedGivenAnomaly).toFixed(1)}%  ` +
          `AUC conf=${aucConf.toFixed(3)}  AUC moveScore=${aucMove.toFixed(3)}  AUC gtZ=${aucGt.toFixed(3)}  AUC |ret|=${aucRet.toFixed(3)}`,
        );
      }
    }

    const dumpPath = process.env['EVAL_DUMP'];
    if (dumpPath) {
      writeFileSync(dumpPath, JSON.stringify(rows));
      console.log(`per-bucket dump: ${dumpPath} (${rows.length} rows)`);
    }

    // ── Benchmark gate (brute-force-calibrated 2026-07: recall 92.5% /
    // events 95.2% / FP 2.45%).  If a change trips these, it made the
    // detector measurably worse on real data — fix the change, not the
    // thresholds.
    expect(recall).toBeGreaterThanOrEqual(0.85);
    expect(eventsCaught / events).toBeGreaterThanOrEqual(0.9);
    expect(fpRate).toBeLessThanOrEqual(0.035);
  }, 600_000);
});
