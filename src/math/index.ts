// ─── Math internals — exported for unit testing ───────────────────────────────
// Import via package.json "exports" → "volume-anomaly/math"
// or directly in tests: import { ... } from '../src/math/index.js'

export {
  volumeImbalance,
  hawkesLogLikelihood,
  hawkesFit,
  hawkesLambda,
  hawkesAnomalyScore,
} from './hawkes.js';

export type { HawkesFitResult } from './hawkes.js';

export {
  cusumFit,
  cusumUpdate,
  cusumInitState,
  cusumAnomalyScore,
  cusumBatch,
} from './cusum.js';

export type { CusumUpdateResult } from './cusum.js';

export {
  bocpdUpdate,
  bocpdInitState,
  bocpdAnomalyScore,
  bocpdBatch,
  defaultPrior,
} from './bocpd.js';

export type { BocpdState, BocpdUpdateResult, NormalGammaPrior } from './bocpd.js';

export { nelderMead } from './optimizer.js';
export type { NMResult } from './optimizer.js';
