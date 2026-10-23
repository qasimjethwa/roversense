/* RoverSense in-browser scorer.
 *
 * Runs the EXACT IsolationForest trained by train.py (every tree, split and
 * leaf exported to model.json), plus the identical feature engineering and
 * guardrail layers. No backend, no approximation — bit-faithful port of the
 * scikit-learn scoring path (verified by tests/parity.js against the
 * Python model).
 */

const GAMMA = 0.5772156649015329; // Euler-Mascheroni

function avgPathLength(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + GAMMA) - (2 * (n - 1)) / n;
}

/** Paper anomaly score s in (0,1]; higher = more anomalous. */
export function scoreSample(model, x) {
  const { trees, meta } = model;
  const cMax = avgPathLength(meta.max_samples);
  let sumH = 0;
  for (let t = 0; t < trees.length; t++) {
    const tr = trees[t];
    let node = 0;
    while (tr.left[node] !== -1) {
      node = x[tr.feature[node]] <= tr.threshold[node]
        ? tr.left[node]
        : tr.right[node];
    }
    // tree.depth is 1-indexed (root=1, like sklearn); -1 -> edge count
    sumH += tr.depth[node] + avgPathLength(tr.n_node_samples[node]) - 1.0;
  }
  return Math.pow(2, -sumH / (trees.length * cMax));
}

const N_CH = 6;
const ROLL = 12;

export class Detector {
  constructor(model) {
    const m = model.meta;
    this.model = model;
    this.med = m.base_median;
    this.madX = m.base_mad_x;
    this.madD1 = m.base_mad_d1;
    this.thr = m.threshold;
    this.alpha = m.detection ? 0.3 : 0.3;
    this.hist = Array.from({ length: N_CH }, () => []); // imputed raws
    this.prev = new Array(N_CH).fill(null);
    this.ewma = null;
    this.mlRun = 0;
    this.stuckRun = new Array(N_CH).fill(0);
    this.inIncident = false;
    this.quiet = 0;
    this.t = 0;
  }

  /** Push one 6-channel sample (NaN allowed). Returns detection state. */
  push(raw) {
    const t = this.t++;
    const imp = new Array(N_CH);
    for (let c = 0; c < N_CH; c++) {
      let v = raw[c];
      if (!Number.isFinite(v)) {
        v = this.prev[c] !== null ? this.prev[c] : this.med[c];
        if (!Number.isFinite(v)) v = this.med[c];
      }
      imp[c] = v;
    }
    const F = new Array(N_CH * 3);
    for (let c = 0; c < N_CH; c++) {
      const v = imp[c];
      const d1 = this.prev[c] === null ? 0 : Math.abs(v - this.prev[c]);
      const h = this.hist[c];
      h.push(v);
      if (h.length > ROLL) h.shift();
      const k = h.length;
      let mean = 0;
      for (let i = 0; i < k; i++) mean += h[i];
      mean /= k;
      let va = 0;
      for (let i = 0; i < k; i++) va += (h[i] - mean) * (h[i] - mean);
      va /= k;
      const still = -Math.log10(Math.sqrt(Math.max(va, 0)) + 1e-6);
      F[c * 3] = Math.abs(v - this.med[c]);
      F[c * 3 + 1] = still;
      F[c * 3 + 2] = d1;
      this.prev[c] = v;
    }

    const score = scoreSample(this.model, F);
    this.ewma = this.ewma === null ? score : this.alpha * score + (1 - this.alpha) * this.ewma;

    const ready = t >= ROLL;
    this.mlRun = this.ewma > this.thr ? this.mlRun + 1 : 0;
    const trigML = ready && this.mlRun >= 3;

    let trigZ = false, trigSpike = false, trigNan = false, worstCh = -1, worstZ = 0;
    for (let c = 0; c < N_CH; c++) {
      const z = Math.abs(raw[c] - this.med[c]) / Math.max(this.madX[c], 1e-12);
      if (Number.isFinite(raw[c]) && z > worstZ) { worstZ = z; worstCh = c; }
      if (z > 8.0) trigZ = true;
      const z1 = F[c * 3 + 2] / Math.max(this.madD1[c], 1e-12);
      if (z1 > 8.0) trigSpike = true;
      if (!Number.isFinite(raw[c])) trigNan = true;
    }
    let trigStuck = false;
    for (let c = 0; c < N_CH; c++) {
      this.stuckRun[c] = F[c * 3 + 1] > 4.5 ? this.stuckRun[c] + 1 : 0;
      if (this.stuckRun[c] >= 12) trigStuck = true;
    }
    trigZ = trigZ && ready; trigSpike = trigSpike && ready;
    trigNan = trigNan && ready; trigStuck = trigStuck && ready;

    const layers = [];
    if (trigML) layers.push("ML");
    if (trigZ) layers.push("Z");
    if (trigStuck) layers.push("STUCK");
    if (trigSpike) layers.push("SPIKE");
    if (trigNan) layers.push("NAN");
    const trig = layers.length > 0;

    if (!this.inIncident && trig) { this.inIncident = true; this.quiet = 0; }
    else if (this.inIncident) {
      if (trig) this.quiet = 0;
      else if (++this.quiet >= 3) this.inIncident = false;
    }

    return {
      t, score, ewma: this.ewma, layers, incident: this.inIncident,
      worstCh, worstZ, raw: imp.slice(), hasNan: trigNan, calibrating: !ready,
    };
  }
}
