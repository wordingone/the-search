/**
 * α sweep for active inference blend contract.
 * actionGain=10.0 fixed. Varies α in [0.1, 0.3, 0.5, 0.7, 0.9].
 * activeReality = normalize(α * externalReality + (1-α) * action)
 * RNG reset to same seed before each trial for fair comparison.
 */

// ── Utilities (copied from fluxcore_entity.mjs) ───────────────────────────
let _rng = 42;
const rand = () => { _rng = (Math.imul(1664525, _rng) + 1013904223) >>> 0; return (_rng >>> 0) / 4294967296; };
const randN = () => { const u = rand() + 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const normalize = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; const n = Math.sqrt(s) + 1e-12; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
const randUnit = d => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = randN(); return normalize(v); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const clone = v => new Float64Array(v);
const avg = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
function makeOrtho(base, dim) {
  const r = randUnit(dim), p = dot(r, base), b = new Float64Array(dim);
  for (let i = 0; i < dim; i++) b[i] = r[i] - p * base[i];
  return normalize(b);
}

// ── DynamicAttractorField ─────────────────────────────────────────────────
class DynamicAttractorField {
  constructor(dim, opts = {}) {
    this.dim = dim;
    this.memories = [];
    this.spawnThreshold = opts.spawnThreshold || 0.5;
    this.mergeThreshold = opts.mergeThreshold || 0.95;
    this.pruneThreshold = opts.pruneThreshold || 500;
    this.memoryMeta = [];
    this.tick = 0;
  }
  selectAndUpdate(reality, self, memLr) {
    this.tick++;
    let surprise = 0;
    for (let i = 0; i < this.dim; i++) surprise += Math.abs(self[i] - reality[i]);
    surprise /= this.dim;
    let bestIdx = -1, bestSim = -1;
    for (let i = 0; i < this.memories.length; i++) {
      const sim = Math.abs(dot(this.memories[i], reality));
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }
    if (bestSim < this.spawnThreshold) {
      const newMem = clone(reality);
      const newIdx = this.memories.length;
      this.memories.push(newMem);
      this.memoryMeta[newIdx] = { lastUsed: this.tick, useCount: 1, birthTick: this.tick };
      return { memory: newMem, idx: newIdx, surprise, action: 'spawn', totalMemories: this.memories.length };
    }
    const activeMem = this.memories[bestIdx];
    for (let i = 0; i < this.dim; i++) this.memories[bestIdx][i] += memLr * reality[i];
    normalize(this.memories[bestIdx]);
    this.memoryMeta[bestIdx] = {
      lastUsed: this.tick,
      useCount: (this.memoryMeta[bestIdx]?.useCount || 0) + 1,
      birthTick: this.memoryMeta[bestIdx]?.birthTick || this.tick
    };
    this.mergeConvergedMemories();
    this.pruneStaleMemories();
    return { memory: activeMem, idx: bestIdx, surprise, action: 'use', totalMemories: this.memories.length };
  }
  mergeConvergedMemories() {
    for (let i = 0; i < this.memories.length; i++) {
      for (let j = i + 1; j < this.memories.length; j++) {
        if (Math.abs(dot(this.memories[i], this.memories[j])) > this.mergeThreshold) {
          const mi = this.memoryMeta[i] || { useCount: 1 };
          const mj = this.memoryMeta[j] || { useCount: 1 };
          const total = mi.useCount + mj.useCount;
          for (let k = 0; k < this.dim; k++)
            this.memories[i][k] = (mi.useCount * this.memories[i][k] + mj.useCount * this.memories[j][k]) / total;
          normalize(this.memories[i]);
          this.memoryMeta[i] = { lastUsed: Math.max(mi.lastUsed, mj.lastUsed), useCount: total, birthTick: Math.min(mi.birthTick, mj.birthTick) };
          this.memories.splice(j, 1); this.memoryMeta.splice(j, 1); j--;
        }
      }
    }
  }
  pruneStaleMemories() {
    for (let i = this.memories.length - 1; i >= 0; i--) {
      if (this.memoryMeta[i] && this.tick - this.memoryMeta[i].lastUsed > this.pruneThreshold) {
        this.memories.splice(i, 1); this.memoryMeta.splice(i, 1);
      }
    }
  }
}

// ── ActiveInferenceEngine ─────────────────────────────────────────────────
class ActiveInferenceEngine {
  constructor(dim) {
    this.dim = dim;
    this.self = randUnit(dim);
    this.velocity = new Float64Array(dim);
    this.prevSelf = clone(this.self);
    this.attractors = new DynamicAttractorField(dim);
    this.actionGain = 10.0;
    this.lastPredicted = null;
    this.activeMemForAction = null;
    this.predictionErrorMagnitude = 0;
  }
  fold(reality, baseLr, memLr, k, memW, velDecay, velGain) {
    const dim = this.dim, s = this.self, r = reality;
    let surprise = 0;
    for (let i = 0; i < dim; i++) surprise += Math.abs(s[i] - r[i]);
    surprise /= dim;
    if (this.lastPredicted) {
      let pe = 0;
      for (let i = 0; i < dim; i++) pe += Math.abs(r[i] - this.lastPredicted[i]);
      this.predictionErrorMagnitude = pe / dim;
    } else {
      this.predictionErrorMagnitude = surprise;
    }
    const attractorResult = this.attractors.selectAndUpdate(r, s, memLr);
    this.activeMemForAction = attractorResult.memory;
    const alr = baseLr * (1 + k * surprise);
    for (let idx = 0; idx < dim; idx++) {
      const si = s[idx], ri = r[idx], mi = this.activeMemForAction[idx];
      const d = Math.abs(si - ri);
      const left = s[(idx + dim - 1) % dim], right = s[(idx + 1) % dim];
      const grad = (si - left) - (si - right);
      s[idx] = si + alr * ri + (alr * 0.5) * d * grad + memW * mi;
    }
    normalize(s);
    for (let i = 0; i < dim; i++)
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    this.prevSelf = clone(s);
    return { surprise };
  }
  act(velScale) {
    const output = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) output[i] = this.self[i] + velScale * this.velocity[i];
    normalize(output);
    this.lastPredicted = clone(output);
    if (this.activeMemForAction && this.predictionErrorMagnitude > 0.02) {
      const action = clone(this.activeMemForAction);
      normalize(action);
      for (let i = 0; i < this.dim; i++) action[i] *= this.predictionErrorMagnitude * this.actionGain;
      return action;
    }
    return new Float64Array(this.dim);
  }
}

// ── Sweep ─────────────────────────────────────────────────────────────────
const DIM = 64;
const SEED = 9999;  // fixed seed, set before each trial
const ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9];

// Pre-generate shared environment vectors at a fixed seed so all trials
// see identical external reality streams.
_rng = SEED;
const targetDist = randUnit(DIM);
const perturbation = randUnit(DIM);

// Generate all external reality ticks upfront (500 warmup + 200 test)
const warmupRealities = [];
for (let i = 0; i < 500; i++) {
  const v = new Float64Array(DIM);
  for (let j = 0; j < DIM; j++) v[j] = targetDist[j] + 0.01 * randN();
  warmupRealities.push(normalize(v));
}
const testRealities = [];
for (let i = 0; i < 200; i++) {
  const mixRatio = i < 100 ? 0.5 : 0.0;
  const v = new Float64Array(DIM);
  for (let j = 0; j < DIM; j++) v[j] = (1 - mixRatio) * targetDist[j] + mixRatio * perturbation[j];
  testRealities.push(normalize(v));
}

console.log('FluxCore — α sweep (actionGain=10.0, gate=0.02, DIM=64)');
console.log('='.repeat(60));
console.log(`${'α'.padStart(5)} | ${'advantage%'.padEnd(12)} | ${'passive'.padEnd(8)} | ${'active'.padEnd(8)} | result`);
console.log('-'.repeat(60));

const rows = [];

for (const alpha of ALPHAS) {
  // Reset agent RNG seed so each trial starts with same internal state
  _rng = SEED;
  const passiveAgent = new ActiveInferenceEngine(DIM);
  _rng = SEED + 1;
  const activeAgent = new ActiveInferenceEngine(DIM);

  // Warmup
  let activeAction = new Float64Array(DIM);
  for (let i = 0; i < 500; i++) {
    const ext = warmupRealities[i];
    passiveAgent.fold(ext, 0.08, 0.015, 20, 0.15, 0.95, 0.05);

    const aMag = Math.sqrt(activeAction.reduce((s, x) => s + x * x, 0));
    const aInput = new Float64Array(DIM);
    for (let j = 0; j < DIM; j++) {
      aInput[j] = aMag > 1e-9
        ? alpha * ext[j] + (1 - alpha) * activeAction[j]
        : ext[j];
    }
    normalize(aInput);
    activeAgent.fold(aInput, 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    activeAction = activeAgent.act(12.5);
  }

  // Test
  const passiveSurprises = [], activeSurprises = [];
  for (let i = 0; i < 200; i++) {
    const ext = testRealities[i];
    const pRes = passiveAgent.fold(ext, 0.08, 0.015, 20, 0.15, 0.95, 0.05);

    const aMag = Math.sqrt(activeAction.reduce((s, x) => s + x * x, 0));
    const aInput = new Float64Array(DIM);
    for (let j = 0; j < DIM; j++) {
      aInput[j] = aMag > 1e-9
        ? alpha * ext[j] + (1 - alpha) * activeAction[j]
        : ext[j];
    }
    normalize(aInput);
    const aRes = activeAgent.fold(aInput, 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    activeAction = activeAgent.act(12.5);

    passiveSurprises.push(pRes.surprise);
    activeSurprises.push(aRes.surprise);
  }

  const pAvg = avg(passiveSurprises.slice(100, 200));
  const aAvg = avg(activeSurprises.slice(100, 200));
  const adv = pAvg > 0 ? ((pAvg - aAvg) / pAvg * 100) : 0;
  const pass = aAvg < pAvg;

  // Per-window breakdown
  const windows = [];
  for (let w = 0; w < 4; w++) {
    const pW = avg(passiveSurprises.slice(w * 50, (w + 1) * 50));
    const aW = avg(activeSurprises.slice(w * 50, (w + 1) * 50));
    windows.push({ pW, aW, adv: pW > 0 ? ((pW - aW) / pW * 100) : 0 });
  }

  rows.push({ alpha, pAvg, aAvg, adv, pass, windows });
  console.log(`${alpha.toFixed(1).padStart(5)} | ${adv.toFixed(2).padStart(10)}%  | ${pAvg.toFixed(4).padEnd(8)} | ${aAvg.toFixed(4).padEnd(8)} | ${pass ? 'PASS ✓' : 'FAIL ✗'}`);
}

console.log('='.repeat(60));
console.log('\nPer-window breakdown (ticks 0-49, 50-99, 100-149, 150-199):');
for (const r of rows) {
  console.log(`\n  α=${r.alpha.toFixed(1)}:`);
  r.windows.forEach((w, i) => {
    console.log(`    ticks ${i*50}-${(i+1)*50-1}: passive=${w.pW.toFixed(4)} active=${w.aW.toFixed(4)} adv=${w.adv.toFixed(1)}%`);
  });
}
