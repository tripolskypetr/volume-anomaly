/**
 * Minimal Nelder-Mead simplex optimizer.
 * Sufficient for low-dimensional problems (Hawkes: 3 params).
 */

export interface NMResult {
  x:         number[];
  fx:        number;
  iters:     number;
  converged: boolean;
}

export function nelderMead(
  f:       (x: number[]) => number,
  x0:      number[],
  options: { maxIter?: number; tol?: number } = {},
): NMResult {
  const { maxIter = 1000, tol = 1e-8 } = options;
  const n    = x0.length;
  const step = 0.2;

  // Build initial simplex
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const v = x0.slice();
    v[i] = (v[i] ?? 0) * (1 + step) + (Math.abs(v[i] ?? 0) < 1e-10 ? step : 0);
    simplex.push(v);
  }
  let fvals = simplex.map(f);

  const sortSimplex = () => {
    const idx = fvals.map((_, i) => i).sort((a, b) => fvals[a]! - fvals[b]!);
    fvals = idx.map((i) => fvals[i]!);
    for (let i = 0; i <= n; i++) simplex[i] = idx.map((j) => simplex[j]![i] as number);
    // rebuild rows
    const rows: number[][] = idx.map((j) => simplex[j] as unknown as number[]);
    for (let i = 0; i <= n; i++) simplex[i] = rows[i]!;
  };

  // NM parameters
  const alpha_r = 1, gamma_e = 2, rho = 0.5, sigma = 0.5;

  for (let iter = 0; iter < maxIter; iter++) {
    sortSimplex();

    // Convergence check
    const spread = fvals[n]! - fvals[0]!;
    if (spread < tol) {
      return { x: simplex[0]!, fx: fvals[0]!, iters: iter, converged: true };
    }

    // Centroid (exclude worst)
    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j] += simplex[i]![j]! / n;
    }

    // Reflection
    const xr = centroid.map((c, j) => c + alpha_r * (c - simplex[n]![j]!));
    const fr = f(xr);

    if (fr < fvals[0]!) {
      // Expansion
      const xe = centroid.map((c, j) => c + gamma_e * (xr[j]! - c));
      const fe = f(xe);
      simplex[n] = fe < fr ? xe : xr;
      fvals[n]   = fe < fr ? fe : fr;
    } else if (fr < fvals[n - 1]!) {
      simplex[n] = xr;
      fvals[n]   = fr;
    } else {
      // Contraction
      const xc = centroid.map((c, j) => c + rho * (simplex[n]![j]! - c));
      const fc = f(xc);
      if (fc < fvals[n]!) {
        simplex[n] = xc;
        fvals[n]   = fc;
      } else {
        // Shrink
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i]!.map((v, j) => simplex[0]![j]! + sigma * (v - simplex[0]![j]!));
          fvals[i]   = f(simplex[i]!);
        }
      }
    }
  }

  sortSimplex();
  return { x: simplex[0]!, fx: fvals[0]!, iters: maxIter, converged: false };
}
