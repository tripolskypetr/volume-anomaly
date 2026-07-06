/**
 * quickstart.mjs — volume-anomaly in 30 seconds, no math required.
 *
 * Run from the repo root (uses the committed real BTCUSDT data + the built
 * library, so no install or build step is needed):
 *
 *   node examples/quickstart.mjs
 *
 * It replays a slice of real Binance aggTrades around a known volume spike,
 * calls scan() on the raw stream (no manual window slicing), and prints the
 * plain-language explanation.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { scan, explain, VolumeAnomalyDetector } from '../build/index.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── 1. Load ~35 minutes of real trades around a known spike ──────────────────
// CSV columns: aggTradeId,price,qty,firstId,lastId,timestamp_us,isBuyerMaker,isBestMatch
// NOTE: this file's timestamps are in MICROseconds; the library wants ms.
const SPIKE_TS_MS = 1_740_812_760_000;             // 2025-03-01, largest burst of the day
const FROM_MS     = SPIKE_TS_MS - 34 * 60_000;     // 34 min of baseline before it
const TO_MS       = SPIKE_TS_MS + 20_000;          // ...through the burst itself

const trades = [];
for (const line of readFileSync(join(__dirname, '..', 'mock', 'BTCUSDT-aggTrades-2025-03-01.csv'), 'utf8').split('\n')) {
  const p = line.split(',');
  if (p.length < 7) continue;
  const ts = Number(p[5]) / 1000; // µs → ms
  if (ts < FROM_MS) continue;
  if (ts > TO_MS) break;
  trades.push({ id: p[0], price: Number(p[1]), qty: Number(p[2]), timestamp: ts, isBuyerMaker: p[6] === 'True' });
}
console.log(`loaded ${trades.length} trades spanning ${(((TO_MS - FROM_MS)) / 60_000).toFixed(0)} min\n`);

// ── 2. One call: scan the stream (it splits baseline/recent by time itself) ──
const result = scan(trades);

// ── 3. Read the answer like a human ──────────────────────────────────────────
console.log(explain(result));
console.log('');
console.log(`raw fields: anomaly=${result.anomaly}  severity=${result.severity}  confidence=${result.confidence.toFixed(3)}  moveScore=${result.moveScore.toFixed(3)}`);

// ── 4. (optional) How well could the detector calibrate on your data? ────────
const det = new VolumeAnomalyDetector();
det.train(trades.slice(0, -300));
const rep = det.calibrationReport;
console.log(`\ncalibration: ${rep.quality} (${rep.channelsCalibrated}/${rep.channelsTotal} channels, ${(rep.trainingSpanSec / 60).toFixed(0)} min of data)`);
for (const n of rep.notes) console.log(`  note: ${n}`);
