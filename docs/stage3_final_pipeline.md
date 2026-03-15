# Stage 3 Experiment Pipeline: Final Version

## Executive Summary

After adversarial critique and resp_z contamination analysis, **2 candidates** advance to Phase 2:
1. **#3: Phi Stability (delta_stability)** — Priority 1
2. **#5: Prediction Error Alignment (delta_correlation)** — Priority 2

**3 candidates REJECTED:** #1 (resp_z-derived, direction ambiguous), #2 (orthogonal vars), #4 (redundant)

**Total pipeline:** 2-3 weeks, $20-40, produces 0-2 viable Stage 3 signals

---

## Phase 1 Final Results

| Candidate | R1-R5 | F1-F6 | Direction Rule | resp_z Contamination Risk | Verdict |
|-----------|-------|-------|----------------|---------------------------|---------|
| #3: Phi Stability (delta_stability) | PASS | PASS | Δ>0→↓eta, Δ<0→↑eta | LOW (phi upstream of resp_z) | **PASS** |
| #5: Error Alignment (delta_correlation) | PASS | PASS | Δ<0→↓eta, Δ≥0→neutral | LOW (error independent, push hybrid) | **PASS** |
| #1: Push Variance (cv_push) | PASS | PASS | **UNDEFINED** | **HIGH** (push = f(resp_z)) | **REJECT** |
| #4: Response Autocorrelation | FAIL | PASS | Ambiguous | **HIGH** (autocorr(resp) = autocorr(resp_z)) | **REJECT** |
| #2: Plasticity Consensus | FAIL (R5) | PASS | Ambiguous | LOW | **REJECT** |

### Rejection Rationale

**#1: Push Variance**
- **Direction ambiguity unresolved:** High CV = good specialization OR bad chaos? No principled rule without additional context.
- **resp_z contamination:** push computed from resp_z (line 182: `push = eta * tanh(resp_z) * direction`). Variance of push = variance of f(resp_z). Likely |r| > 0.7 in Phase 2 test.
- **Temporal derivative adds frozen element:** delta_cv requires history storage + combination rule → violates R5.

**#4: Response Autocorrelation**
- **Redundant information:** autocorr(response) ≈ autocorr(resp_z) ≈ 0.98 (c012 already measured). Autocorrelation invariant under z-scoring.
- **No new signal family:** violates c013 spirit.

**#2: Plasticity Consensus**
- **Orthogonal variables:** Attention measures state similarity, push measures alpha updates. Alignment meaning unclear.
- **Threshold introduces frozen hyperparameter:** violates R5.

---

## Phase 2 Protocol: FINAL

### Candidates

**#3: Phi Stability (delta_stability)**
- **Signal definition:**
  ```python
  # Step t:
  phi_change[t] = sum((phi_sig[t][i][k] - phi_sig[t-1][i][k])^2 for all i,k)
  stability[t] = 1.0 / (1.0 + sqrt(phi_change[t]))

  # Meta-signal (requires t-1 history):
  delta_stability[t] = stability[t] - stability[t-1]
  ```
- **Eta update rule:**
  ```python
  eta_new = clip(eta * (1 + alpha * (-delta_stability)), 0.001, 0.1)
  # alpha ∈ {0.01, 0.05, 0.1} tuned in Phase 2
  # Negative sign: settling (Δ>0) decreases eta, destabilizing (Δ<0) increases eta
  ```
- **Variables from the_living_seed.py:**
  - `phi_sig[i][k]`: lines 135-145 (signal-modulated tanh output)
  - Requires storing: `phi_sig_prev` (72 floats), `stability_prev` (1 float)

**#5: Prediction Error Alignment (delta_correlation)**
- **Signal definition:**
  ```python
  # Per cell i:
  bare_diff[i] = [phi_bare[i][k] - xs[i][k] for k in D]  # line 212
  error_mag[i] = ||bare_diff[i]||  # vnorm, line 213
  push_mag[i] = sum(|push[i][k]| for k in D)  # alpha update magnitude

  # Across all cells:
  correlation[t] = pearson(error_mag, push_mag)

  # Meta-signal (requires t-1 history):
  delta_correlation[t] = correlation[t] - correlation[t-1]
  ```
- **Eta update rule:**
  ```python
  eta_new = clip(eta * (1 + alpha * min(0, delta_correlation)), 0.001, 0.1)
  # alpha ∈ {0.01, 0.05, 0.1} tuned in Phase 2
  # One-way: only decrease eta when alignment degrades (Δ<0), never increase
  ```
- **Variables from the_living_seed.py:**
  - `bare_diff[i]`: line 212
  - `phi_bare[i][k]`: lines 122-131
  - `xs[i][k]`: current state
  - `push[i][k]`: lines 178-185 (alpha update, not stored but computable)
  - Requires storing: `correlation_prev` (1 float)

### Step 2.1: Implementation (4 hours each)

Engineer implements in harness.py:
1. Signal computation function
2. History storage (adaptive state, not frozen hyperparameters)
3. Eta update rule with alpha meta-rate parameter
4. Logging: signal value, delta value, eta value, alpha stats per step

### Step 2.2: Signal Statistics (2 hours, 3 seeds)

**Configuration:**
- Seeds: [42, 137, 2024]
- Protocol: n_perm=4, n_trials=3, K=[6,8] (cheap)
- Duration: 100 steps
- Log per signal-bearing step

**Measurements:**

1. **Dynamic range:** mean(|signal|), std(signal), mean(|delta|), std(delta)
2. **Temporal stability:** lag-1 autocorr(delta)
3. **Eta convergence:** mean(eta), std(eta) in final 20 steps
4. **Computational overhead:** signal computation time / step time
5. **resp_z Contamination Check (NEW):**
   ```python
   # Per signal-bearing step:
   resp_z_flat = [resp_z[i][k] for all i,k]  # line 171, flatten to 72-vector

   # If signal is scalar (delta_stability, delta_correlation):
   r_contamination = pearson(signal_history, mean(resp_z_flat)_history)

   # Average |r| across steps and seeds
   ```

**Pass criteria:**
- Dynamic range: mean(|delta|) > 0.01, std(delta) > 0.001 (D3)
- Temporal stability: autocorr(delta) ∈ [0.1, 0.8] (D2: not noise, not constant)
- Eta convergence: eta stabilizes to non-zero value or oscillates predictably
- Overhead: < 10% (D1)
- **Contamination: |r| < 0.7** (independent of resp_z, satisfies c013)

**Fail criteria:**
- **|r| ≥ 0.7:** Signal is disguised resp_z derivative → REJECT, document in constraints.json
- Dynamic range too small or autocorr out of range → REVISE or REJECT

### Step 2.3: Ground Truth Smoke Test (1 hour, 3 seeds)

Run ground truth test on adaptive-eta version:
- Same 3 seeds
- Verify distinguishability on novel inputs
- Verify persistence after input removal

**Pass criterion:** 3/3 seeds pass ground truth (R4)

---

## Phase 3 Protocol: FINAL

### Phase 3 Prerequisite: 2-Seed Sanity Check (Regime Validation)

**Purpose:** Validate HIGH/LOW regime stratification under improved 2× exposure protocol before expensive Phase 3.

**Rationale:** Current stratification based on CV=29% protocol (Entry 030). Need to confirm it holds under CV≈20% improved protocol (c015: n_perm=8, n_trials=6).

**Procedure:**
1. Select boundary seeds: 31337 (HIGH, canonical +0.2291), 1618 (LOW, canonical +0.0955)
2. Run canonical baseline with 2× exposure protocol
3. Measure training gap for both seeds
4. Compute amplitude ratio: 31337_gap / 1618_gap

**Pass criterion:**
- Ratio ∈ [1.8, 2.1]× (validates 1.87× regime separation)
- Both measurements stable (low variance under repeated eval)

**Fail criterion (triggers re-clustering):**
- Ratio < 1.5× or > 2.5× (regime structure changed under improved protocol)
- **Action on fail:** Re-run all 10 seeds with 2× exposure to re-cluster HIGH/MID/LOW before Phase 3

**Cost:** ~20 minutes compute

---

### Step 3.1: Baseline + Adaptive Comparison (10 seeds)

**Baseline (Fixed Eta = 0.01):**
- Seeds: [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
- Protocol: n_perm=8, n_trials=6, K=[4,6,8,10] (2× exposure, c015/c017)
- Measure: training gap, novel gap

**Adaptive (Candidate Signal):**
- Eta init: 0.01
- Alpha: selected from Phase 2 sweep (best of {0.01, 0.05, 0.1})
- Same seeds, protocol, measurements

**Analysis:**
- Paired t-test (same seed for baseline and adaptive)
- Effect size: Cohen's d
- P-value threshold: p < 0.05
- Ground truth: 10/10 seeds must pass

### Step 3.2: Seed Regime Robustness (NEW)

**Requirement:** Signal must work across HIGH/MID/LOW seed regimes, ensuring true regime invariance.

**Seed Stratification (from analyst Entry 030):**
- **HIGH regimes** (n=5): seeds [42, 2024, 314, 31337, 2718], mean training gap +0.2019
- **MID regimes** (n=2): seeds [7, 9999], mean training gap +0.1460
- **LOW regimes** (n=3): seeds [137, 1618, 3141], mean training gap +0.1078
- Amplitude ratio HIGH:LOW = 1.87×

**Procedure:**
1. Run Phase 3 Step 3.1 with all 10 seeds
2. Post-hoc analysis: stratify paired deltas (adaptive - fixed) by regime
3. Compute paired t-test p-value PER REGIME

**Pass criterion (STRICT - ALL THREE):**
- p < 0.05 in HIGH regime (n=5)
- **AND** p < 0.05 in MID regime (n=2)
- **AND** p < 0.05 in LOW regime (n=3)
- **AND** overall p < 0.05 across all 10 seeds

**Fail criterion (ANY ONE triggers REJECT):**
- p ≥ 0.05 in HIGH regime (fails strongest regime)
- **OR** p ≥ 0.05 in MID regime (fails mid-range)
- **OR** p ≥ 0.05 in LOW regime (fails weakest regime)
- **OR** HIGH/LOW delta ratio > 2.5× (regime sensitivity despite passing p-tests)

**Rationale for strict AND logic:**
- Signals failing on LOW but passing on HIGH are regime-dependent, not self-generated properties
- Accepting regime-specific signals adds frozen element: "which regime am I in?"
- True Stage 3 signals adapt to THE SYSTEM across all initialization conditions, not favorable subsets

### Step 3.3: Convergence Analysis

For passing candidates:
- Plot eta evolution over time (all 10 seeds)
- Measure convergence rate
- Check for oscillation, saturation, runaway
- Verify eta distribution is non-trivial (R3: not uniform, not zero, not at bounds)

---

## Success Criteria: COMPLETE

A candidate is a **successful Stage 3 solution** if and only if:

1. ✓ Phase 1: Passes R1-R5, violates none of F1-F6, has defined directional rule
2. ✓ Phase 2: Dynamic range > 0.01, autocorr ∈ [0.1, 0.8], **|r_contamination| < 0.7**, ground truth 3/3
3. ✓ Phase 3: Training gap p < 0.05 (d > 0.3), novel gap p < 0.05 (d > 0.3), ground truth 10/10
4. ✓ **Regime robustness:** p < 0.05 in HIGH, MID, AND LOW regimes separately
5. ✓ Eta convergence: non-trivial distribution (not uniform, not zero, not at bounds)

**Constitution Stage 3 exit criterion:**
> "Adaptive adaptation beats fixed adaptation. Ground truth still passes. The rates converge to non-trivial values (not all equal, not all zero)."

---

## Compute Budget: FINAL

| Phase | Candidates | Runs | Time | Cost |
|-------|-----------|------|------|------|
| Phase 1 | 5 → 2 PASS | 0 | 1 hour | $0 |
| Phase 2 | 2 | 60 runs @ 2 min | 2 hours | $10-20 |
| Phase 3 | 1-2 | 29-58 runs @ 4 min | 2-4 hours | $10-20 |
| **Total** | - | **89-118 runs** | **2-3 weeks** | **$20-40** |

**Cost per successful Stage 3 signal:** $40-80 (assuming 50% Phase 3 success rate)

---

## Expected Outcomes

### Best Case (2 candidates succeed Phase 3)

**Result:** Two viable Stage 3 signals with different properties:
- #3 (delta_stability): Two-way adaptation (can increase or decrease eta)
- #5 (delta_correlation): One-way adaptation (only decreases eta)

**Next action:** Compare effect sizes. Select stronger signal as canonical. Document weaker as alternative. Test if combination (weighted average) is even stronger.

### Nominal Case (1 candidate succeeds Phase 3)

**Result:** One viable Stage 3 signal.

**Next action:** Document as canonical Stage 3 implementation. Update frozen_frame.json (eta: frozen → adaptive). Begin Stage 4 design (beta/gamma structural parameters).

### Failure Case (0 candidates succeed Phase 3)

**Result:** Neither signal produces statistically significant improvement over fixed eta.

**Interpretation:** The 2 candidate signal families tested are insufficient for Stage 3. Possible reasons:
1. **Stage 2 incomplete:** Alpha adaptation has not converged to stable distribution, making eta adaptation premature
2. **Different signal family needed:** Phi-based and error-based signals are insufficient. Need cross-cell coordination signals (e.g., synchrony, information flow) or higher-level abstractions
3. **Architectural limitation:** Current plasticity rule (lines 169-190) may be fundamentally limited. Stage 3 may require Stage 6 (functional form adaptation) first
4. **Constitution violation:** Stage 3 definition may be wrong. Adaptive adaptation rates may require external evaluation (violating Principle II)

**Next action:** Return to Phase 1 with expanded candidate families OR escalate to constitution review.

---

## Critical Implementation Notes

### For Engineer (Task 2)

**History storage must be implemented carefully:**

```python
class Organism:
    def __init__(self, ...):
        # For #3: Phi Stability
        self.phi_sig_prev = None  # NC × D floats, init to None
        self.stability_prev = None  # 1 float, init to None

        # For #5: Error Alignment
        self.correlation_prev = None  # 1 float, init to None

    def step(self, xs, signal=None):
        # ... compute phi_sig, phi_bare, etc.

        # Compute candidate signals ONLY when signal-bearing step AND history exists
        if self.alive and signal and self.phi_sig_prev is not None:
            # Compute delta_stability or delta_correlation
            # Update eta
            pass

        # Update history at END of step
        if signal:
            self.phi_sig_prev = copy.deepcopy(phi_sig)
            self.stability_prev = computed_stability
            self.correlation_prev = computed_correlation
```

**Alpha meta-rate parameter:**
- Default: 0.01
- Tunable in Phase 2 via command-line arg: `--meta_alpha 0.05`
- Must be logged alongside eta for analysis

### For QA (Task 6)

**Validation checklist:**
1. Verify temporal derivatives computed correctly (t vs t-1, not t vs t-2)
2. Verify history storage does not leak across runs (reset in __init__)
3. Verify eta clip bounds enforced (0.001, 0.1)
4. Verify signal computation skipped when history is None (first step)
5. **NEW:** Verify resp_z contamination measurement uses correct correlation (signal vs mean(resp_z), not signal vs individual resp_z[i][k])

### For Analyst

**Pre-Phase-3 requirements:**
1. **Phase 3 Prerequisite Task:** Run 2-seed sanity check (31337 HIGH, 1618 LOW) with 2× exposure to validate regime stratification
   - If ratio ∈ [1.8, 2.1]×: proceed with historical stratification
   - If ratio outside range: re-run all 10 seeds to re-cluster before Phase 3
2. Seed stratification (from Entry 030, pending sanity check validation):
   - HIGH (n=5): [42, 2024, 314, 31337, 2718], mean +0.2019
   - MID (n=2): [7, 9999], mean +0.1460
   - LOW (n=3): [137, 1618, 3141], mean +0.1078

**Phase 3 post-hoc analysis:**
- Stratify paired deltas (adaptive - fixed) by regime
- Compute paired t-test p-value per regime (within-regime test)
- Report: overall p, HIGH p, MID p, LOW p, HIGH/LOW delta ratio
- **Pass requires:** p < 0.05 in ALL three regimes (AND logic, not OR)

---

## Open Questions: RESOLVED

### Q1: Is Per-Cell Eta Necessary?

**RESOLVED:** NO. Constraint c011 established per-cell eta does not improve performance. Eta remains GLOBAL (single scalar).

### Q2: One-Way vs Two-Way Adaptation?

**UNRESOLVED:** #3 (two-way) vs #5 (one-way). Will be answered empirically by Phase 3 results.

### Q3: Temporal Derivative Lag?

**IMPLEMENTATION DECISION:** Use lag-1 for signal-bearing steps only. If signal-bearing step frequency < 20%, Phase 2 statistics will reveal this via low autocorr. If needed, increase signal frequency in harness (currently ~1 every 3-5 steps).

---

## Document Hierarchy

This is the FINAL pipeline. Supersedes:
- `stage3_experiment_pipeline.md` (original Task 7 deliverable, pre-adversarial)
- `stage3_pipeline_revision.md` (post-adversarial, still had #1 as conditional)

**Canonical reference order:**
1. **This document** (stage3_final_pipeline.md) — executable pipeline
2. `stage3_adversarial_analysis.md` — justification for rejections
3. `stage3_signal_spec.md` — requirements framework (Task 1)
4. `stage3_candidates.md` — original candidate list (Task 3)

---

*This final pipeline incorporates all adversarial critiques, resolves direction ambiguities via temporal derivatives, adds resp_z contamination checks, and reduces candidates from 5 → 2. Ready for implementation.*
