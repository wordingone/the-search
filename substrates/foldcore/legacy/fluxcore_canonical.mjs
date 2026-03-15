/**
 * FluxCore Canonical — Step 16
 *
 * The canonical fold: gradient term removed, self-derived thresholds only.
 * Combines Step 14 (gradient ablation) + Step 15 (autothresh).
 *
 * Canonical fold:
 *   u[i] = s[i] + alr * r[i] + memW * m[i]
 *   s = normalize(u)
 *
 * No hand-tuned threshold fallback. Self-derived is the only mode.
 */

import { writeFileSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD — self-derived thresholds only
// ════════════════════════════════════════════════════════════════════════════
class DynamicAttractorField {
  constructor(dim) {
    this.dim = dim;
    this.memories = [];
    this.memoryMeta = [];
    this.tick = 0;

    // Spawn: running mean/std of max memory similarity per tick
    this.simMu = 0.5;
    this.simSigma = 0.1;
    this.simEmaAlpha = 0.01;
    this.simVarEma = 0.01;
    this.simCount = 0;

    // Merge: 1 - meanSurprise
    this.surpriseEma = 0.05;

    // Prune: per-memory surprise contribution
    this.memoryContrib = [];
    this.surpriseNoiseFloor = Infinity;
  }

  selectAndUpdate(reality, self, memLr) {
    this.tick++;

    // Global surprise (L1)
    let surprise = 0;
    for (let i = 0; i < this.dim; i++) surprise += Math.abs(self[i] - reality[i]);
    surprise /= this.dim;

    // Update surprise EMA
    this.surpriseEma = 0.99 * this.surpriseEma + 0.01 * surprise;

    // Update surprise noise floor
    if (surprise < this.surpriseNoiseFloor) {
      this.surpriseNoiseFloor = surprise;
    }

    // Find best-matching memory
    let bestIdx = -1, bestSim = -1;
    for (let i = 0; i < this.memories.length; i++) {
      const sim = Math.abs(dot(this.memories[i], reality));
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }

    // Update running similarity stats
    if (this.memories.length > 0) {
      this.simCount++;
      if (this.simCount <= 5) {
        this.simMu = ((this.simCount - 1) * this.simMu + bestSim) / this.simCount;
        const diff = bestSim - this.simMu;
        this.simVarEma = ((this.simCount - 1) * this.simVarEma + diff * diff) / this.simCount;
        this.simSigma = Math.sqrt(this.simVarEma);
      } else {
        this.simMu = (1 - this.simEmaAlpha) * this.simMu + this.simEmaAlpha * bestSim;
        const diff = bestSim - this.simMu;
        this.simVarEma = (1 - this.simEmaAlpha) * this.simVarEma + this.simEmaAlpha * diff * diff;
        this.simSigma = Math.sqrt(this.simVarEma);
      }
    }

    // Self-derived thresholds
    let effectiveSpawn = this.simMu - 2 * this.simSigma;
    effectiveSpawn = Math.max(0.05, Math.min(0.95, effectiveSpawn));

    let effectiveMerge = 1 - this.surpriseEma;
    effectiveMerge = Math.max(0.8, Math.min(0.995, effectiveMerge));

    // SPAWN
    if (bestSim < effectiveSpawn) {
      const newMem = clone(reality);
      const newIdx = this.memories.length;
      this.memories.push(newMem);
      this.memoryMeta[newIdx] = { lastUsed: this.tick, useCount: 1, birthTick: this.tick };
      this.memoryContrib[newIdx] = 0;
      return {
        memory: newMem, idx: newIdx, surprise, action: 'spawn',
        totalMemories: this.memories.length,
        effectiveSpawn, effectiveMerge
      };
    }

    // Use best memory
    const activeMem = this.memories[bestIdx];

    // Contribution tracking
    const contribution = bestSim * (1 - surprise);
    const contribAlpha = 0.05;
    this.memoryContrib[bestIdx] = (1 - contribAlpha) * (this.memoryContrib[bestIdx] || 0) + contribAlpha * contribution;

    // Update memory toward reality
    for (let i = 0; i < this.dim; i++) this.memories[bestIdx][i] += memLr * reality[i];
    normalize(this.memories[bestIdx]);

    this.memoryMeta[bestIdx] = {
      lastUsed: this.tick,
      useCount: (this.memoryMeta[bestIdx]?.useCount || 0) + 1,
      birthTick: this.memoryMeta[bestIdx]?.birthTick || this.tick
    };

    this._mergeConverged(effectiveMerge);
    this._pruneMemories();

    return {
      memory: activeMem, idx: bestIdx, surprise, action: 'use',
      totalMemories: this.memories.length,
      effectiveSpawn, effectiveMerge
    };
  }

  _mergeConverged(threshold) {
    for (let i = 0; i < this.memories.length; i++) {
      for (let j = i + 1; j < this.memories.length; j++) {
        if (Math.abs(dot(this.memories[i], this.memories[j])) > threshold) {
          const mi = this.memoryMeta[i] || { useCount: 1 };
          const mj = this.memoryMeta[j] || { useCount: 1 };
          const total = mi.useCount + mj.useCount;
          for (let k = 0; k < this.dim; k++)
            this.memories[i][k] = (mi.useCount * this.memories[i][k] + mj.useCount * this.memories[j][k]) / total;
          normalize(this.memories[i]);
          this.memoryMeta[i] = { lastUsed: Math.max(mi.lastUsed, mj.lastUsed), useCount: total, birthTick: Math.min(mi.birthTick, mj.birthTick) };
          this.memoryContrib[i] = (this.memoryContrib[i] || 0) + (this.memoryContrib[j] || 0);
          this.memories.splice(j, 1); this.memoryMeta.splice(j, 1); this.memoryContrib.splice(j, 1); j--;
        }
      }
    }
  }

  _pruneMemories() {
    const noiseFloor = this.surpriseNoiseFloor * 1.5;
    const staleLimit = 5000;

    for (let i = this.memories.length - 1; i >= 0; i--) {
      const meta = this.memoryMeta[i];
      if (!meta) continue;
      const age = this.tick - meta.birthTick;
      const idle = this.tick - meta.lastUsed;
      const contrib = this.memoryContrib[i] || 0;
      if (age < 50) continue;
      if ((contrib < noiseFloor && idle > 100) || idle > staleLimit) {
        this.memories.splice(i, 1); this.memoryMeta.splice(i, 1); this.memoryContrib.splice(i, 1);
      }
    }
  }

  getStats() {
    return {
      count: this.memories.length,
      ages: this.memories.map((_, i) => this.memoryMeta[i] ? this.tick - this.memoryMeta[i].birthTick : 0),
      useCounts: this.memoryMeta.map(m => m?.useCount || 0),
      simMu: this.simMu,
      simSigma: this.simSigma,
      surpriseEma: this.surpriseEma,
      noiseFloor: this.surpriseNoiseFloor
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE ENGINE — canonical fold (no gradient term)
// ════════════════════════════════════════════════════════════════════════════
class ActiveInferenceEngine {
  constructor(dim, env) {
    this.dim = dim;
    this.environment = env;
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
    const activeMem = attractorResult.memory;
    this.activeMemForAction = activeMem;
    const alr = baseLr * (1 + k * surprise);

    // CANONICAL FOLD: no gradient term
    // u[i] = s[i] + alr * r[i] + memW * m[i]
    for (let idx = 0; idx < dim; idx++) {
      s[idx] = s[idx] + alr * r[idx] + memW * activeMem[idx];
    }
    normalize(s);

    for (let i = 0; i < dim; i++) {
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    }
    this.prevSelf = clone(s);

    return {
      surprise,
      predictionErrorMagnitude: this.predictionErrorMagnitude,
      attractorAction: attractorResult.action,
      memoryCount: attractorResult.totalMemories,
      effectiveSpawn: attractorResult.effectiveSpawn,
      effectiveMerge: attractorResult.effectiveMerge
    };
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

// ════════════════════════════════════════════════════════════════════════════
// ENVIRONMENT
// ════════════════════════════════════════════════════════════════════════════
class DynamicEnvironment {
  constructor(dim) {
    this.dim = dim;
    this.currentDist = 0;
    this.distributions = [];
    this.tick = 0;
    let base = randUnit(dim);
    for (let i = 0; i < 4; i++) {
      this.distributions.push(clone(base));
      base = makeOrtho(base, dim);
    }
  }
  sense() {
    if (this.tick % 500 === 0 && this.tick > 0) {
      this.currentDist = (this.currentDist + 1) % this.distributions.length;
    }
    this.tick++;
    const dist = this.distributions[this.currentDist];
    const noise = 0.001;
    const v = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) v[i] = dist[i] + noise * randN();
    return normalize(v);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// RESULTS ACCUMULATOR
// ════════════════════════════════════════════════════════════════════════════
const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

log('='.repeat(78));
log('FluxCore Canonical — Validation Suite');
log('Canonical fold: u[i] = s[i] + alr*r[i] + memW*m[i] (no gradient)');
log('Thresholds: self-derived only (spawn=mu-2sigma, merge=1-surprise, prune=contrib)');
log('='.repeat(78));

const results = {};

// ════════════════════════════════════════════════════════════════════════════
// TEST 1: Attractor Genesis (A->B->C->D->A, DIM=64)
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('TEST 1: Attractor Genesis (A->B->C->D->A, DIM=64)');
  log('-'.repeat(78));

  _rng = 42;
  const dim = 64;
  const env = new DynamicEnvironment(dim);
  const entity = new ActiveInferenceEngine(dim, null);

  log('\nRunning 2000 ticks with 4 distribution switches...\n');

  for (let tick = 0; tick < 2000; tick++) {
    const reality = env.sense();
    const result = entity.fold(reality, 0.08, 0.015, 20, 0.15, 0.95, 0.05);

    if (tick % 250 === 0) {
      const stats = entity.attractors.getStats();
      log(`  Tick ${String(tick).padStart(4)}: ${stats.count} memories, surprise=${result.surprise.toFixed(4)}, action=${result.attractorAction}, spawn_th=${result.effectiveSpawn.toFixed(4)}, merge_th=${result.effectiveMerge.toFixed(4)}`);
    }

    entity.act(12.5);
  }

  const memCount = entity.attractors.getStats().count;
  const pass = memCount >= 2 && memCount <= 8;
  results['T1_genesis'] = pass;
  log(`\n  Final: ${memCount} memories`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'} (expected 2-8 memories, got ${memCount})`);
}

// ════════════════════════════════════════════════════════════════════════════
// T1-T4 EQUIVALENT at DIM=64, 512, 8192
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('T1-T4 MULTI-DIM: DIM=64, 512, 8192');
  log('-'.repeat(78));

  for (const dim of [64, 512, 8192]) {
    // --- Attractor genesis ---
    _rng = 42;
    const env = new DynamicEnvironment(dim);
    const entity = new ActiveInferenceEngine(dim, null);

    for (let tick = 0; tick < 2000; tick++) {
      const reality = env.sense();
      entity.fold(reality, 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      entity.act(12.5);
    }

    const stats = entity.attractors.getStats();
    const genesisPass = stats.count >= 2 && stats.count <= 8;
    results[`genesis_${dim}`] = genesisPass;

    log(`\n  DIM=${dim}:`);
    log(`    Genesis: ${stats.count} memories -> ${genesisPass ? 'PASS' : 'FAIL'}`);
    log(`    simMu=${stats.simMu.toFixed(4)}, simSigma=${stats.simSigma.toFixed(4)}, surpriseEma=${stats.surpriseEma.toFixed(4)}`);

    // --- Reacquisition A->B->A ---
    _rng = 123;
    const NOISE = 0.0003;
    const bA = randUnit(dim);
    const bB = makeOrtho(bA, dim);
    const makeStatic = (base) => () => {
      const v = new Float64Array(dim);
      for (let i = 0; i < dim; i++) v[i] = base[i] + NOISE * randN();
      return normalize(v);
    };
    const gA = makeStatic(bA), gB = makeStatic(bB);
    const eng = new ActiveInferenceEngine(dim, null);

    const ea1 = [];
    for (let t = 0; t < 1000; t++) {
      const res = eng.fold(gA(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      if (t < 50) ea1.push(res.surprise);
    }
    for (let t = 0; t < 500; t++) eng.fold(gB(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    const ea2 = [];
    for (let t = 0; t < 1000; t++) {
      const res = eng.fold(gA(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      if (t < 50) ea2.push(res.surprise);
    }

    const m1 = avg(ea1), m2 = avg(ea2);
    const reacqPass = m2 < m1;
    results[`reacq_${dim}`] = reacqPass;
    log(`    Reacquisition: A1=${m1.toFixed(6)} A2=${m2.toFixed(6)} -> ${reacqPass ? 'PASS' : 'FAIL'}`);

    // --- Multi-dist A->B->C->D->A ---
    _rng = 456;
    const bases = [randUnit(dim)];
    for (let i = 1; i < 4; i++) bases.push(makeOrtho(bases[i-1], dim));
    const gens = bases.map(b => makeStatic(b));
    const eng2 = new ActiveInferenceEngine(dim, null);

    const ea1m = [];
    for (let t = 0; t < 1000; t++) {
      const res = eng2.fold(gens[0](), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      if (t < 50) ea1m.push(res.surprise);
    }
    for (let p = 1; p < 4; p++) {
      for (let t = 0; t < 500; t++) eng2.fold(gens[p](), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    }
    const ea2m = [];
    for (let t = 0; t < 1000; t++) {
      const res = eng2.fold(gens[0](), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      if (t < 50) ea2m.push(res.surprise);
    }

    const m1m = avg(ea1m), m2m = avg(ea2m);
    const multiPass = m2m < m1m;
    results[`multi_${dim}`] = multiPass;
    log(`    Multi-dist:    A1=${m1m.toFixed(6)} A2=${m2m.toFixed(6)} -> ${multiPass ? 'PASS' : 'FAIL'}`);
    log(`    Final memories: ${eng2.attractors.getStats().count}`);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// REACQUISITION TEST (standalone, A->B->A at DIM=64)
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('REACQUISITION TEST (A->B->A, DIM=64, standalone)');
  log('-'.repeat(78));

  const DIM = 64;
  const NOISE = 0.0003;
  _rng = 123;

  const bA = randUnit(DIM);
  const bB = makeOrtho(bA, DIM);
  const makeStatic = (base) => () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = base[i] + NOISE * randN();
    return normalize(v);
  };
  const gA = makeStatic(bA), gB = makeStatic(bB);
  const eng = new ActiveInferenceEngine(DIM, null);

  // Phase A1: 1000 ticks
  const ea1 = [];
  for (let t = 0; t < 1000; t++) {
    const res = eng.fold(gA(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    if (t < 50) ea1.push(res.surprise);
  }

  // Phase B: 500 ticks
  for (let t = 0; t < 500; t++) eng.fold(gB(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
  const statsAfterB = eng.attractors.getStats();
  log(`\n  After A+B: ${statsAfterB.count} memories`);

  // Phase A2: 1000 ticks
  const ea2 = [];
  for (let t = 0; t < 1000; t++) {
    const res = eng.fold(gA(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    if (t < 50) ea2.push(res.surprise);
  }

  const m1 = avg(ea1), m2 = avg(ea2);
  const pass = m2 < m1;
  results['reacq_standalone'] = pass;
  log(`  A1 early surprise (50 ticks): ${m1.toFixed(7)}`);
  log(`  A2 early surprise (50 ticks): ${m2.toFixed(7)}`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'}`);
}

// ════════════════════════════════════════════════════════════════════════════
// CSI REAL DATA TEST
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('CSI REAL DATA TEST (DIM=384, 1920 embeddings)');
  log('-'.repeat(78));

  const embedded = JSON.parse(readFileSync(path.join(__dirname, 'data', 'csi_embedded.json'), 'utf8'));
  const centers = JSON.parse(readFileSync(path.join(__dirname, 'data', 'csi_division_centers.json'), 'utf8'));
  const divNames = Object.keys(centers).sort();

  log(`\n  Loaded ${embedded.length} records, ${divNames.length} divisions`);

  _rng = 42;
  const DIM = 384;
  const engine = new ActiveInferenceEngine(DIM, null);

  log('  Running FluxCore canonical on CSI embeddings...\n');

  const surpriseWindow = [];
  let spawnCount = 0;

  for (let tick = 0; tick < embedded.length; tick++) {
    const rec = embedded[tick];
    const vec = new Float64Array(rec.vec);
    const result = engine.fold(vec, 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    surpriseWindow.push(result.surprise);
    if (result.attractorAction === 'spawn') spawnCount++;

    if ((tick + 1) % 200 === 0 || tick === embedded.length - 1) {
      const avgSurp = surpriseWindow.slice(-100).reduce((s, x) => s + x, 0) / Math.min(100, surpriseWindow.length);
      log(`  Tick ${String(tick + 1).padStart(4)}: mem=${result.memoryCount}, surp=${result.surprise.toFixed(4)}, avg100=${avgSurp.toFixed(4)}, spawn_th=${result.effectiveSpawn.toFixed(4)}, merge_th=${result.effectiveMerge.toFixed(4)}`);
    }
  }

  log(`\n  Total spawns: ${spawnCount}`);

  // Memory-division alignment
  const mems = engine.attractors.memories;
  log(`  Final memory count: ${mems.length}`);

  let passCount = 0;
  let totalBestSim = 0;
  let totalSpec = 0;
  const alignments = [];

  for (let m = 0; m < mems.length; m++) {
    const mem = mems[m];
    const sims = divNames.map(div => ({
      div,
      sim: dot(mem, new Float64Array(centers[div]))
    })).sort((a, b) => b.sim - a.sim);
    const best = sims[0], second = sims[1];
    const specificity = best.sim - second.sim;
    if (best.sim > 0.3) passCount++;
    totalBestSim += best.sim;
    totalSpec += specificity;
    alignments.push({ mem: m, best, second, specificity });
  }

  const meanSim = mems.length ? totalBestSim / mems.length : 0;
  const meanSpec = mems.length ? totalSpec / mems.length : 0;

  log(`\n  Alignment: ${passCount}/${mems.length} pass (dot > 0.3)`);
  log(`  Mean best sim: ${meanSim.toFixed(4)}`);
  log(`  Mean specificity: ${meanSpec.toFixed(4)}`);

  // Per-memory detail
  log(`\n  ${'Mem'.padEnd(4)} ${'Best div'.padEnd(10)} ${'Sim'.padEnd(8)} ${'2nd div'.padEnd(10)} ${'2nd sim'.padEnd(8)} Specificity`);
  log('  ' + '-'.repeat(60));
  for (const a of alignments) {
    log(`  ${String(a.mem).padEnd(4)} div ${a.best.div.padEnd(5)} ${a.best.sim.toFixed(4).padEnd(8)} div ${a.second.div.padEnd(5)} ${a.second.sim.toFixed(4).padEnd(8)} +${a.specificity.toFixed(4)}`);
  }

  log('\n  --- Comparison ---');
  log(`  Hand-tuned baseline:  357/359 pass, mean sim=0.5710, 359 memories`);
  log(`  AutoThresh (Step 15): 21/21 pass, mean sim=0.5895, 21 memories`);
  log(`  Canonical (Step 16):  ${passCount}/${mems.length} pass, mean sim=${meanSim.toFixed(4)}, ${mems.length} memories`);

  const csiPass = passCount === mems.length && mems.length > 0;
  results['csi_realdata'] = csiPass;

  const stats = engine.attractors.getStats();
  log(`\n  Final state: simMu=${stats.simMu.toFixed(4)}, simSigma=${stats.simSigma.toFixed(4)}, surpriseEma=${stats.surpriseEma.toFixed(4)}, noiseFloor=${stats.noiseFloor.toFixed(6)}`);
}

// ════════════════════════════════════════════════════════════════════════════
// SUMMARY
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '='.repeat(78));
  log('CANONICAL FOLD SUMMARY');
  log('='.repeat(78));

  log('\nFold equation:');
  log('  u[i] = s[i] + alr * r[i] + memW * m[i]');
  log('  s = normalize(u)');
  log('  where alr = baseLr * (1 + k * surprise)');
  log('');
  log('Thresholds (all self-derived):');
  log('  Spawn:  mu - 2*sigma of running max-similarity stats');
  log('  Merge:  1 - surpriseEMA');
  log('  Prune:  contribution EMA < noiseFloor * 1.5');
  log('');
  log('Removed:');
  log('  - Gradient term: (alr * 0.5) * |s[i] - r[i]| * grad[i]');
  log('  - Hand-tuned threshold fallback');

  log('\n' + '-'.repeat(78));
  log('TEST RESULTS');
  log('-'.repeat(78));

  let allPass = true;
  for (const [test, pass] of Object.entries(results)) {
    log(`  ${test.padEnd(25)} ${pass ? 'PASS' : 'FAIL'}`);
    if (!pass) allPass = false;
  }

  log('\n' + '-'.repeat(78));
  const totalTests = Object.keys(results).length;
  const passedTests = Object.values(results).filter(p => p).length;
  log(`  Total: ${passedTests}/${totalTests} passed`);
  log(`  Overall: ${allPass ? 'ALL PASS' : 'SOME FAILURES'}`);
  log('='.repeat(78));
}

log('\n');

writeFileSync('B:/M/avir/research/fluxcore/results/canonical_validation.txt', allLines.join('\n') + '\n');
log('Captured to results/canonical_validation.txt');
