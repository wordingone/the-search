# FluxCore — Honest Claims Summary
*Generated 2026-03-13. Every number from results/. Basis for Step 13 README.*

## CLAIM 1 — VERIFIED
The fold converges to track reality.
**Evidence:** Surprise drops from ~0.14 to <0.001 within 250 ticks at DIM=64 (baseline_entity.txt TEST 1). T1 passes at DIM=64 (avg gain 0.00758), DIM=512 (0.00208), DIM=8192 (0.00022) (baseline_true.txt). CUDA at DIM=64: tick 0=0.1438, steady-state=0.0003. CUDA at DIM=512: tick 0=0.0532, steady-state=0.0003.
**Public statement:** The fold converges to track arbitrary unit-vector distributions within hundreds of ticks across dimensions from 64 to 8192, verified independently in JavaScript and CUDA on an RTX 4090.

## CLAIM 2 — VERIFIED
Dynamic attractors form a self-organizing codebook.
**Evidence:** TEST 1 (baseline_entity.txt): memories spawn on each distribution switch. TEST 5 (entity_multidist.txt): A→B→C→D→A produces 4 memories with perfect alignment A=1.0000, B=1.0000, C=1.0000, D=1.0000. CUDA DIM=64 and DIM=512 both produce correct 1→2→3→4 spawn counts.
**Public statement:** The attractor pool grows one memory per novel distribution encountered, achieves perfect alignment with each distribution, and correctly separates up to 4 orthogonal distributions in both JavaScript and CUDA.

## CLAIM 3 — VERIFIED
The fold remembers — accelerated reacquisition on return.
**Evidence:** T2 (baseline_true.txt): DIM=64: 6.1× faster (A2=0.0094 vs A1=0.0572); DIM=512: 5.5×; DIM=8192: 5.7×. Entity T4 (entity_reacquisition.txt): A2=0.0125 vs A1=0.0136 (8% faster). T5 (entity_multidist.txt): A2=0.0122 vs A1=0.0124 after 4-distribution cycle.
**Public statement:** After encountering a distribution, returning to it produces 6× faster convergence in the dual-memory baseline (verified at three dimensions); the dynamic attractor pool shows ~8% faster reacquisition.

## CLAIM 4 — VERIFIED
The fold is not reducible to EMA.
**Evidence:** T1 (anticipation), T3 (shift detection), T4 (adaptive LR) pass at DIM=64/512/8192 (baseline_true.txt). T3 shift detection produces spikes of 1882σ (DIM=64), 3154σ (DIM=512), 6044σ (DIM=8192) above stable baseline.
**Public statement:** The fold's surprise-adaptive learning rate and velocity tracking produce behaviors — anticipation, distributional shift detection, accelerated reacquisition — that cannot be reproduced by an EMA.

## CLAIM 5 — VERIFIED (QUALIFIED)
Prediction-error-driven steering produces measurable advantage over passive observation.
**Evidence:** New mechanism: prediction storage → predictionError = reality - lastPredicted → action = normalize(activeMem) × predictionErrorMagnitude × actionGain (gate: errMag > 0.02). Blend contract: activeReality = normalize(α × externalReality + (1-α) × actionVec). actionGain=10.0. α=0.5: 2.6% advantage. α sweep (active_alpha_sweep.txt): α=0.1→24.66%, α=0.3→7.08%, α=0.5→2.76%, α=0.7→1.08%, α=0.9→0.25%. Monotonic controllability curve.
**Qualification:** Advantage proportional to control authority. Attractor lag (~50 ticks at memLr=0.015) causes oscillatory overshoot during memory transitions. This is prediction-error-driven memory-directed steering, not full free-energy active inference.
**Public statement:** The active agent shows measurably lower surprise than passive across full controllability range (α=0.1–0.9). Advantage is proportional to control authority and not yet validated on real-world data.

## CLAIM 6 — VERIFIED
The hierarchy extracts temporal abstractions at multiple levels.
**Evidence:** After normalization bug fix (hierarchy_fixed.txt): L0=2, L1=2, L2=2 memories (vs pre-fix L1=100, L2=64). Audit (hierarchy_audit.txt): each memory maps 1:1 to distribution epochs (500 uses each). dot(L2_mem1, L0_mem0)=-0.852 shows L2 references raw distribution space.
**Public statement:** After fixing a normalization bug, the three-level hierarchy forms one memory per distribution epoch at every level, with L2 attractors encoding abstract patterns that reference the underlying distribution geometry.

## CLAIM 7 — VERIFIED
The fold scales to high dimensions on GPU.
**Evidence:** Compiles clean (sm_89, RTX 4090) after 6 bug fixes (5 in entity.cu/impl.cu + missed NormalizeKernel inter-warp reduction). DIM=64: steady-state=0.0003, attractor genesis 1→2→3→4 correct. DIM=512: steady-state=0.0003, correct. DIM=4096: steady-state=0.0003, correct. Tick-0 surprise follows √(1/dim) scaling.
**Public statement:** The CUDA implementation compiles and runs correctly at DIM=64, DIM=512, and DIM=4096 on an RTX 4090, producing convergence behavior matching the JavaScript reference.

## CLAIM 8 — VERIFIED
The fold retains information on real-world data.
**Evidence:** CSI construction specification corpus: 1920 records, 33 divisions, DIM=384 (all-MiniLM-L6-v2, L2-normalized). 359 memories spawned. 357/359 (99.4%) align with division centers (dot > 0.3). Mean best similarity 0.5710 (random baseline ≈ 0.0). All 33 divisions represented. Surprise plateau ~0.038 (vs 0.0003 synthetic) — reflects genuine within-division semantic variance. div01/div26 cross-alignment (0.8777/0.8331) reflects real semantic overlap, not a bug. 2 failures: mem175 (div13, sim=0.2909) and mem209 (div28, sim=0.2852) — borderline transition points.
**GNG comparison (gng_comparison.txt):** GNG final nodes=21, pass=21/21 (100%), mean sim=0.8765. FluxCore: 359 memories, 99.4% pass, mean sim=0.5710. GNG produces compact cluster representatives; FluxCore discovers granular subcategory structure. Different regimes — not directly comparable as better/worse.
**Public statement:** The fold spawns semantically coherent memories on real text embeddings. 99.4% of 359 memories align with recognizable CSI division centers across 33 construction specification categories.

---
*All 8 claims verified as of 2026-03-13. CUDA cosmetic bug: uninitialized age display for unused memory slots (does not affect computation).*
