/**
 * FluxCore Entity — Auto-Threshold (Step 15)
 *
 * Surprise-derived thresholds vs hand-tuned baseline.
 * AUTO_THRESH=true uses self-derived spawn/merge/prune.
 * AUTO_THRESH=false falls back to original hand-tuned values.
 */

import { writeFileSync } from 'fs';

const AUTO_THRESH = true;

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
const std = a => { const m = avg(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length || 1)); };
function makeOrtho(base, dim) {
  const r = randUnit(dim), p = dot(r, base), b = new Float64Array(dim);
  for (let i = 0; i < dim; i++) b[i] = r[i] - p * base[i];
  return normalize(b);
}

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD — with self-derived thresholds
// ════════════════════════════════════════════════════════════════════════════
class DynamicAttractorField {
  constructor(dim, opts = {}) {
    this.dim = dim;
    this.memories = [];
    this.memoryMeta = [];
    this.tick = 0;

    // Hand-tuned fallback values
    this.spawnThreshold = opts.spawnThreshold ?? 0.5;
    this.mergeThreshold = opts.mergeThreshold ?? 0.95;
    this.pruneThreshold = opts.pruneThreshold ?? 500;

    // ── Auto-threshold state ──
    this.autoThresh = opts.autoThresh ?? AUTO_THRESH;

    // Spawn: running mean/std of max memory similarity per tick
    this.simMu = 0.5;     // initial prior
    this.simSigma = 0.1;  // initial prior
    this.simEmaAlpha = 0.01; // EMA decay for running stats
    this.simVarEma = 0.01;   // EMA of variance
    this.simCount = 0;        // ticks seen

    // Merge: dynamic threshold = 1 - currentMeanSurprise
    this.surpriseEma = 0.05;  // running EMA of surprise

    // Prune: per-memory surprise contribution tracking
    // memoryContrib[i] = EMA of how much memory i reduces surprise
    this.memoryContrib = [];
    this.surpriseNoiseFloor = Infinity; // running min of global surprise
    this.surpriseNoiseEma = 0.05;       // smoothed noise floor
  }

  selectAndUpdate(reality, self, memLr) {
    this.tick++;

    // Global surprise (L1)
    let surprise = 0;
    for (let i = 0; i < this.dim; i++) surprise += Math.abs(self[i] - reality[i]);
    surprise /= this.dim;

    // Update surprise EMA (used for merge threshold)
    this.surpriseEma = 0.99 * this.surpriseEma + 0.01 * surprise;

    // Update surprise noise floor (running min * 1.5)
    if (surprise < this.surpriseNoiseFloor) {
      this.surpriseNoiseFloor = surprise;
    }
    this.surpriseNoiseEma = 0.999 * this.surpriseNoiseEma + 0.001 * surprise;

    // Find best-matching memory
    let bestIdx = -1, bestSim = -1;
    for (let i = 0; i < this.memories.length; i++) {
      const sim = Math.abs(dot(this.memories[i], reality));
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }

    // Update running similarity stats (for spawn threshold)
    if (this.memories.length > 0) {
      this.simCount++;
      if (this.simCount <= 5) {
        // Warmup: simple accumulation
        this.simMu = ((this.simCount - 1) * this.simMu + bestSim) / this.simCount;
        // Variance accumulation
        const diff = bestSim - this.simMu;
        this.simVarEma = ((this.simCount - 1) * this.simVarEma + diff * diff) / this.simCount;
        this.simSigma = Math.sqrt(this.simVarEma);
      } else {
        // EMA update
        this.simMu = (1 - this.simEmaAlpha) * this.simMu + this.simEmaAlpha * bestSim;
        const diff = bestSim - this.simMu;
        this.simVarEma = (1 - this.simEmaAlpha) * this.simVarEma + this.simEmaAlpha * diff * diff;
        this.simSigma = Math.sqrt(this.simVarEma);
      }
    }

    // ── Compute effective thresholds ──
    let effectiveSpawn, effectiveMerge;
    if (this.autoThresh) {
      // Spawn: when max_sim < mu - 2*sigma, input is novel
      effectiveSpawn = this.simMu - 2 * this.simSigma;
      // Clamp to reasonable range
      effectiveSpawn = Math.max(0.05, Math.min(0.95, effectiveSpawn));

      // Merge: 1 - meanSurprise. Low surprise -> high threshold (only near-duplicates)
      effectiveMerge = 1 - this.surpriseEma;
      effectiveMerge = Math.max(0.8, Math.min(0.995, effectiveMerge));
    } else {
      effectiveSpawn = this.spawnThreshold;
      effectiveMerge = this.mergeThreshold;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SPAWN: If no good match, create new attractor
    // ═══════════════════════════════════════════════════════════════════════
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

    // ── Prune: surprise contribution tracking ──
    // How much does having this memory reduce surprise?
    // Approximate: bestSim is high when memory is useful (reduces surprise)
    // contribution = bestSim - (what surprise would be without this memory)
    // Proxy: contribution ~ bestSim * (1 - surprise)
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

    // ═══════════════════════════════════════════════════════════════════════
    // MERGE: If two memories converge, fuse them
    // ═══════════════════════════════════════════════════════════════════════
    this._mergeConverged(effectiveMerge);

    // ═══════════════════════════════════════════════════════════════════════
    // PRUNE: Remove stale/useless memories
    // ═══════════════════════════════════════════════════════════════════════
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
          // Merge contribution: sum
          this.memoryContrib[i] = (this.memoryContrib[i] || 0) + (this.memoryContrib[j] || 0);
          this.memories.splice(j, 1); this.memoryMeta.splice(j, 1); this.memoryContrib.splice(j, 1); j--;
        }
      }
    }
  }

  _pruneMemories() {
    if (!this.autoThresh) {
      // Hand-tuned: prune by staleness only
      for (let i = this.memories.length - 1; i >= 0; i--) {
        const meta = this.memoryMeta[i];
        if (meta && this.tick - meta.lastUsed > this.pruneThreshold) {
          this.memories.splice(i, 1); this.memoryMeta.splice(i, 1); this.memoryContrib.splice(i, 1);
        }
      }
      return;
    }

    // Auto-thresh: prune when contribution EMA drops below noise floor
    const noiseFloor = this.surpriseNoiseFloor * 1.5;
    // Also keep staleness as a safety net (much longer: 5000 ticks)
    const staleLimit = 5000;

    for (let i = this.memories.length - 1; i >= 0; i--) {
      const meta = this.memoryMeta[i];
      if (!meta) continue;

      const age = this.tick - meta.birthTick;
      const idle = this.tick - meta.lastUsed;
      const contrib = this.memoryContrib[i] || 0;

      // Don't prune very young memories (give them a chance)
      if (age < 50) continue;

      // Prune if: contribution below noise floor AND idle for a while
      // OR if extremely stale
      if ((contrib < noiseFloor && idle > 100) || idle > staleLimit) {
        this.memories.splice(i, 1); this.memoryMeta.splice(i, 1); this.memoryContrib.splice(i, 1);
      }
    }
  }

  getStats() {
    return {
      count: this.memories.length,
      ages: this.memories.map((_, i) => {
        const meta = this.memoryMeta[i];
        return meta ? this.tick - meta.birthTick : 0;
      }),
      useCounts: this.memoryMeta.map(m => m?.useCount || 0),
      simMu: this.simMu,
      simSigma: this.simSigma,
      surpriseEma: this.surpriseEma,
      noiseFloor: this.surpriseNoiseFloor
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE ENGINE (unchanged except for opts passthrough)
// ════════════════════════════════════════════════════════════════════════════
class ActiveInferenceEngine {
  constructor(dim, env, opts = {}) {
    this.dim = dim;
    this.environment = env;
    this.self = randUnit(dim);
    this.velocity = new Float64Array(dim);
    this.prevSelf = clone(this.self);
    this.attractors = new DynamicAttractorField(dim, opts);
    this.actionGain = 10.0;
    this.surpriseTarget = 0.01;
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

    for (let idx = 0; idx < dim; idx++) {
      const si = s[idx], ri = r[idx], mi = activeMem[idx];
      const d = Math.abs(si - ri);
      const left = s[(idx + dim - 1) % dim], right = s[(idx + 1) % dim];
      const grad = (si - left) - (si - right);
      s[idx] = si + alr * ri + (alr * 0.5) * d * grad + memW * mi;
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
log(`FluxCore Entity — Threshold Self-Derivation (AUTO_THRESH=${AUTO_THRESH})`);
log('='.repeat(78));

// ════════════════════════════════════════════════════════════════════════════
// TEST 1: Attractor Genesis A->B->C->D->A at DIM=64
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('TEST 1: Attractor Genesis (A->B->C->D->A, DIM=64)');
  log('-'.repeat(78));

  _rng = 42;
  const dim = 64;
  const env = new DynamicEnvironment(dim);
  const entity = new ActiveInferenceEngine(dim, null, { autoThresh: AUTO_THRESH });

  log('\nRunning 2000 ticks with 4 distribution switches...');
  log('Expected: ~4 memories should form\n');

  for (let tick = 0; tick < 2000; tick++) {
    const reality = env.sense();
    const result = entity.fold(reality, 0.08, 0.015, 20, 0.15, 0.95, 0.05);

    if (tick % 250 === 0) {
      const stats = entity.attractors.getStats();
      log(`  Tick ${String(tick).padStart(4)}: ${stats.count} memories, surprise=${result.surprise.toFixed(4)}, action=${result.attractorAction}, spawn_th=${result.effectiveSpawn?.toFixed(4)}, merge_th=${result.effectiveMerge?.toFixed(4)}`);
      log(`    simMu=${stats.simMu.toFixed(4)}, simSigma=${stats.simSigma.toFixed(4)}, surpriseEma=${stats.surpriseEma.toFixed(4)}`);
    }

    entity.act(12.5);
  }

  const finalStats = entity.attractors.getStats();
  const memCount = finalStats.count;
  const pass = memCount >= 2 && memCount <= 8;
  log(`\n  Final: ${memCount} memories`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'} (expected 2-8 memories, got ${memCount})`);
}

// ════════════════════════════════════════════════════════════════════════════
// MULTI-DIM SWEEP: T1-T4 equivalent at DIM=64, 512, 8192
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('MULTI-DIM SWEEP: DIM=64, 512, 8192');
  log('-'.repeat(78));

  for (const dim of [64, 512, 8192]) {
    _rng = 42;
    const env = new DynamicEnvironment(dim);
    const entity = new ActiveInferenceEngine(dim, null, { autoThresh: AUTO_THRESH });

    // A->B->C->D->A: 2000 ticks
    for (let tick = 0; tick < 2000; tick++) {
      const reality = env.sense();
      entity.fold(reality, 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      entity.act(12.5);
    }

    const stats = entity.attractors.getStats();
    log(`\n  DIM=${dim}:`);
    log(`    Memories: ${stats.count}`);
    log(`    simMu=${stats.simMu.toFixed(4)}, simSigma=${stats.simSigma.toFixed(4)}`);
    log(`    surpriseEma=${stats.surpriseEma.toFixed(4)}, noiseFloor=${stats.noiseFloor.toFixed(6)}`);

    // Reacquisition test: A->B->A
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
    const eng = new ActiveInferenceEngine(dim, null, { autoThresh: AUTO_THRESH });

    // Phase A1
    const ea1 = [];
    for (let t = 0; t < 1000; t++) {
      const res = eng.fold(gA(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      if (t < 50) ea1.push(res.surprise);
    }
    // Phase B
    for (let t = 0; t < 500; t++) eng.fold(gB(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    // Phase A2
    const ea2 = [];
    for (let t = 0; t < 1000; t++) {
      const res = eng.fold(gA(), 0.08, 0.015, 20, 0.15, 0.95, 0.05);
      if (t < 50) ea2.push(res.surprise);
    }

    const m1 = avg(ea1), m2 = avg(ea2);
    const reacqPass = m2 < m1;
    log(`    Reacquisition: A1=${m1.toFixed(6)} A2=${m2.toFixed(6)} -> ${reacqPass ? 'PASS' : 'FAIL'}`);

    // Multi-dist test: A->B->C->D->A
    _rng = 456;
    const bases = [randUnit(dim)];
    for (let i = 1; i < 4; i++) bases.push(makeOrtho(bases[i-1], dim));
    const gens = bases.map(b => makeStatic(b));
    const eng2 = new ActiveInferenceEngine(dim, null, { autoThresh: AUTO_THRESH });

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
    log(`    Multi-dist:    A1=${m1m.toFixed(6)} A2=${m2m.toFixed(6)} -> ${multiPass ? 'PASS' : 'FAIL'}`);
    log(`    Final memories: ${eng2.attractors.getStats().count}`);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// CSI REAL DATA TEST
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('CSI REAL DATA TEST');
  log('-'.repeat(78));

  const { readFileSync } = await import('fs');
  const { fileURLToPath } = await import('url');
  const path = await import('path');
  const __dirname = path.dirname(fileURLToPath(import.meta.url));

  const embedded = JSON.parse(readFileSync(path.join(__dirname, 'data', 'csi_embedded.json'), 'utf8'));
  const centers = JSON.parse(readFileSync(path.join(__dirname, 'data', 'csi_division_centers.json'), 'utf8'));
  const divNames = Object.keys(centers).sort();

  log(`\nLoaded ${embedded.length} records, ${divNames.length} divisions`);

  _rng = 42;
  const DIM = 384;
  const engine = new ActiveInferenceEngine(DIM, null, {
    autoThresh: AUTO_THRESH,
    spawnThreshold: 0.5,
    pruneThreshold: 2000
  });

  log('Running FluxCore on CSI embeddings...\n');

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
      log(`  Tick ${String(tick + 1).padStart(4)}: mem=${result.memoryCount}, surp=${result.surprise.toFixed(4)}, avg100=${avgSurp.toFixed(4)}, spawn_th=${result.effectiveSpawn?.toFixed(4)}, merge_th=${result.effectiveMerge?.toFixed(4)}`);
    }
  }

  log(`\nTotal spawns: ${spawnCount}`);

  // Memory-division alignment
  const mems = engine.attractors.memories;
  const meta = engine.attractors.memoryMeta;
  log(`Final memory count: ${mems.length}`);

  let passCount = 0;
  let totalBestSim = 0;
  let totalSpec = 0;

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
  }

  const meanSim = mems.length ? totalBestSim / mems.length : 0;
  const meanSpec = mems.length ? totalSpec / mems.length : 0;

  log(`\nAlignment: ${passCount}/${mems.length} pass (dot > 0.3)`);
  log(`Mean best sim: ${meanSim.toFixed(4)}`);
  log(`Mean specificity: ${meanSpec.toFixed(4)}`);

  // Compare to baseline
  log('\n--- vs Baseline (hand-tuned) ---');
  log(`Baseline: 357/359 pass, mean sim=0.5710`);
  log(`AutoThresh: ${passCount}/${mems.length} pass, mean sim=${meanSim.toFixed(4)}`);

  const stats = engine.attractors.getStats();
  log(`\nFinal auto-threshold state:`);
  log(`  simMu=${stats.simMu.toFixed(4)}, simSigma=${stats.simSigma.toFixed(4)}`);
  log(`  surpriseEma=${stats.surpriseEma.toFixed(4)}, noiseFloor=${stats.noiseFloor.toFixed(6)}`);
}

// ════════════════════════════════════════════════════════════════════════════
// ANALYSIS: Which thresholds self-derived cleanly
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '='.repeat(78));
  log('THRESHOLD SELF-DERIVATION ANALYSIS');
  log('='.repeat(78));

  log('\n1. SPAWN THRESHOLD (mu - 2*sigma of max similarity):');
  log('   Self-derived from running statistics of how well memories match inputs.');
  log('   When a new input is > 2 std deviations below the mean match quality,');
  log('   it represents genuine novelty. This adapts automatically to the data');
  log('   distribution and dimensionality.');

  log('\n2. MERGE THRESHOLD (1 - meanSurprise):');
  log('   Self-derived from global surprise level.');
  log('   When the system is tracking well (low surprise), it only merges');
  log('   near-duplicates (high threshold). When adapting (high surprise),');
  log('   it becomes more aggressive about merging.');

  log('\n3. PRUNE THRESHOLD (contribution EMA < noise floor * 1.5):');
  log('   Self-derived from per-memory surprise contribution tracking.');
  log('   Each memory is evaluated by how much it reduces surprise when active.');
  log('   Memories whose contribution drops below the noise floor are pruned.');
}

log('\n');

// Write results
writeFileSync('B:/M/avir/research/fluxcore/results/ablation_thresholds.txt', allLines.join('\n') + '\n');
log('Captured to results/ablation_thresholds.txt');
