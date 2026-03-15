# Stage 3 Experiment Pipeline: Eta Adaptation Signal Testing

## Purpose

This document defines the concrete experiment pipeline for testing Stage 3 candidate signals. It synthesizes the requirements specification (stage3_signal_spec.md) with the candidate list (stage3_candidates.md) to create a phased validation protocol.

## Pipeline Overview

Three-phase protocol per the Experiment Gate decision (state.md):
- **Phase 1:** Theoretical compliance check (1 hour, kills bad ideas cheap)
- **Phase 2:** Computational feasibility test (1 day, validates implementation)
- **Phase 3:** Statistical validation (3-5 days, proves performance)

**Critical rule:** No candidate advances to Phase N+1 without passing Phase N.

---

## Phase 1: Requirements Filter (1 Hour Per Candidate)

### Objective
Screen all 5 candidates against the formal requirements spec. Eliminate candidates that violate REQUIRED or FORBIDDEN properties.

### Process

For each candidate signal, verify:

#### Checklist A: REQUIRED Properties (R1-R5)

| Property | Test | Pass Criterion |
|----------|------|----------------|
| R1: Self-Generated | Can signal be computed in forward pass with local state? No external evaluator? | YES |
| R2: Performance Driver | Does theory suggest signal correlates with adaptation quality? | PLAUSIBLE |
| R3: Non-Trivial Convergence | Can signal produce heterogeneous, non-zero, bounded eta values? | YES |
| R4: Ground Truth | Does implementation preserve core dynamics? | YES |
| R5: Frozen Frame | Does signal introduce new frozen elements? | NO NEW ELEMENTS |

#### Checklist B: FORBIDDEN Properties (F1-F6)

| Property | Test | Pass Criterion |
|----------|------|----------------|
| F1: External Measurement | Does signal require MI/loss evaluation? | NO |
| F2: Higher-Order resp_z | Is signal d²resp_z/dt² or higher? | NO |
| F3: delta_rz Per-Cell | Is signal delta_rz for per-cell eta? (c011 violation) | NO |
| F4: Self-Reference | Does eta multiply its own update? (c008 violation) | NO |
| F5: Beta/Gamma Gradients | Does signal use ∂MI/∂β or ∂MI/∂γ? (c001/c002 violation) | NO |
| F6: Local Global Proxy | Does signal decompose beta/gamma per-cell? (c003/c006 violation) | NO |

#### Deliverable

For each candidate:
- **PASS:** Satisfies all R1-R5, violates none of F1-F6 → Advance to Phase 2
- **FAIL:** Violates any R or satisfies any F → Reject with reason
- **UNCERTAIN:** Requires implementation to determine → Note for Phase 2 test

### Execution

Strategist performs manual review with written justification per candidate.

**Estimated time:** 1 hour (12 min/candidate × 5)

**Cost:** $0 (no compute)

---

## Phase 2: Computational Feasibility (1 Day Per Candidate)

### Objective
Implement passing candidates, measure signal statistics, verify computational properties.

### Prerequisites
- Candidate passed Phase 1
- Improved eval protocol implemented (Task 2 complete)

### Process

#### Step 2.1: Implementation (4 hours)

Implement signal computation in harness.py:
1. Add signal computation function
2. Add eta update rule: `eta_new = eta * (1 + alpha * signal)`
3. Add signal logging to track temporal evolution
4. Verify no new frozen elements introduced

**Guard rails:**
- Alpha parameter for meta-learning rate must have default (e.g., 0.01)
- Eta must have clip bounds (e.g., [0.001, 0.1])
- Implementation must not modify core dynamics (phi computation)

#### Step 2.2: Signal Statistics (2 hours, 3 seeds)

Run canonical baseline with signal logging:
- Seeds: [42, 137, 2024]
- Config: n_perm=4, n_trials=3, K=[6,8] (cheap protocol)
- Duration: 100 steps per run
- Log per-step: signal value, eta value, alpha statistics

**Measurements:**
1. **Dynamic range:** mean(|signal|), std(signal)
2. **Temporal stability:** lag-1 autocorrelation of signal
3. **Eta convergence:** does eta reach stable distribution or oscillate?
4. **Computational overhead:** signal computation time / total step time

**Pass criteria (from D1-D7):**
- Mean absolute value > 0.01 (D3: non-zero dynamic range)
- Std > 0.001 (D3: non-zero dynamic range)
- Autocorrelation in [0.1, 0.95] (D2: temporal stability, not noise, not constant)
- Eta converges to non-zero, non-uniform values (R3)
- Computational overhead < 10% (D1: practical)
- Similar statistics across 3 seeds (D4: robust to initialization)

#### Step 2.3: Ground Truth Smoke Test (1 hour, 3 seeds)

Run ground truth test on adaptive-eta version:
- Same 3 seeds
- Verify distinguishability on novel inputs
- Verify persistence after input removal

**Pass criterion:** All 3 seeds pass ground truth (R4)

#### Deliverable

Per candidate:
- **PASS:** All statistics in range, ground truth passes → Advance to Phase 3
- **FAIL:** Statistics out of range OR ground truth fails → Reject with diagnostics
- **REVISE:** Minor implementation issue fixable → Fix and re-run Step 2.2-2.3

### Execution

Engineer implements, QA validates, Analyst computes statistics.

**Estimated time:** 1 day (4h implement + 2h stats + 1h ground truth + 1h review)

**Cost:** ~30 seed-runs @ 2 min/run = 1 hour compute = $5-10 GPU time

---

## Phase 3: Statistical Validation (3-5 Days Per Candidate)

### Objective
Prove candidate signal produces statistically significant improvement over fixed eta on both training and novel tasks.

### Prerequisites
- Candidate passed Phase 2
- Improved eval protocol validated (Task 6 complete)

### Process

#### Step 3.1: Protocol Configuration

**Baseline (Fixed Eta):**
- Eta: 0.01 (canonical value)
- Seeds: 10 (minimum per c019: [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337])
- Protocol: n_perm=8, n_trials=6, K=[4,6,8,10] (2× exposure per c015/c017)
- Duration: 100 steps
- Measure: training gap (learned signals) + novel gap (unseen signals)

**Adaptive (Candidate Signal):**
- Eta initialization: 0.01
- Eta adaptation: `eta_new = clip(eta * (1 + alpha * signal), 0.001, 0.1)`
- Alpha: [To be determined per candidate in Phase 2 based on signal magnitude]
- Same seeds, protocol, duration, measurements as baseline

#### Step 3.2: Hyperparameter Sweep (2 days, if needed)

If Phase 2 reveals signal requires tuning (e.g., alpha meta-rate, smoothing window):
- Test 3-5 hyperparameter values
- Use 3 seeds for sweep (cheap)
- Select best configuration for full 10-seed test

**Example for Candidate #1 (Alpha Trajectory Variance):**
- Alpha ∈ {0.001, 0.01, 0.1}
- 3 seeds × 3 alphas = 9 runs
- Select alpha with highest mean training gap

**Guard rail:** If performance is highly sensitive to hyperparameter, signal may be unstable (violates D4). Consider rejection.

#### Step 3.3: 10-Seed Paired Comparison (1 day)

Run both baseline and adaptive on all 10 seeds:
- **Pairing:** Same seed for baseline and adaptive (reduces variance)
- **Metrics:** Training gap, novel gap, ground truth pass/fail
- **Analysis:** Paired t-test, effect size (Cohen's d), p-value

**Pass criteria (R2, R3, R4):**
- Training gap: adaptive > baseline, p < 0.05 (R2: drives performance)
- Novel gap: adaptive > baseline, p < 0.05 (R2: generalizes)
- Effect size: d > 0.3 (medium effect, detectable above noise)
- Ground truth: All 10 seeds pass for adaptive (R4: maintains ground truth)
- Eta distribution: Not all equal, not all zero, not at clip bounds (R3: non-trivial)

**Failure modes:**
- Training passes, novel fails: Overfitting, signal not self-generated (violates R1/R2)
- Both fail p-value: Signal does not drive adaptation (violates R2)
- Eta degenerates: Signal drives pathological convergence (violates R3)
- Ground truth fails: Signal corrupts core dynamics (violates R4)

#### Step 3.4: Convergence Analysis (1 day)

For passing candidates, analyze eta evolution:
- Plot eta distribution over time (per-cell or global)
- Measure convergence rate
- Check for oscillation, saturation, or runaway
- Compare eta trajectory across seeds

**Questions:**
- Does eta stabilize or continue adapting?
- Do different seeds converge to similar eta distributions? (D4: robustness)
- Is convergence smooth or oscillatory? (D2: stability)

#### Deliverable

Per candidate:
- **SUCCESS:** Passes all criteria → Candidate is viable Stage 3 signal
- **PARTIAL:** Training succeeds, novel fails → Document as Stage 2+ (improved fixed adaptation, not meta-adaptation)
- **FAIL:** Neither passes → Reject, document failure mode

**Success deliverable:**
- Statistical report: p-values, effect sizes, confidence intervals
- Eta convergence plots
- Ground truth verification table
- Recommendation: proceed to Stage 4 or iterate on Stage 3

### Execution

Engineer runs experiments, Analyst computes statistics, QA validates methodology, Strategist interprets results.

**Estimated time:** 3-5 days (0.5d setup + 1-2d hyperparam sweep + 1d 10-seed + 0.5d analysis + 0.5d report)

**Cost:** ~200 seed-runs @ 4 min/run (2× exposure) = 13 hours compute = $50-100 GPU time

---

## Candidate Priority Queue (Phase 1 Pre-Screen)

Based on Phase 1 requirements filter:

### TIER 1: Advance to Phase 2 Immediately

**Candidate #1: Alpha Trajectory Variance (sigma_eta)**
- **R1 (self-generated):** ✓ Computed from push magnitudes (intrinsic)
- **R2 (performance):** ✓ Variance measures ensemble health
- **R3 (non-trivial):** ✓ Can produce heterogeneous eta
- **R4 (ground truth):** ✓ Does not modify phi dynamics
- **R5 (frozen frame):** ✓ No new frozen elements (uses existing push array)
- **F1-F6 violations:** NONE
- **Assessment:** PASS. Simplest, most direct. Highest priority.

**Candidate #3: Phi Stability (phi_stability)**
- **R1 (self-generated):** ✓ Computed from phi evolution (intrinsic)
- **R2 (performance):** ✓ Stability measures computational coherence
- **R3 (non-trivial):** ✓ Can drive eta adaptation
- **R4 (ground truth):** ✓ Does not modify phi dynamics
- **R5 (frozen frame):** ⚠️ Requires storing phi_prev (NC×D floats) — is this a new frozen element?
  - **Decision:** Storage is adaptive state (changes every step), NOT a frozen hyperparameter. PASS.
- **F1-F6 violations:** NONE
- **Assessment:** PASS. Different timescale from push variance. High priority.

### TIER 2: Advance to Phase 2 With Caution

**Candidate #4: Response Autocorrelation (response_coherence)**
- **R1 (self-generated):** ✓ Computed from response history (intrinsic)
- **R2 (performance):** ? Unclear if autocorrelation correlates with MI
- **R3 (non-trivial):** ✓ Can produce heterogeneous eta
- **R4 (ground truth):** ✓ Does not modify phi dynamics
- **R5 (frozen frame):** ⚠️ Requires storing response_prev (NC×D floats)
  - **Decision:** Adaptive state, not frozen. PASS.
- **F1-F6 violations:** ⚠️ F2 concern — is this a derivative of resp_z?
  - **Analysis:** Response is |phi_sig - phi_bare|, not resp_z (z-scored response). Autocorrelation of response is NOT a derivative of resp_z. Uses raw magnitudes, not normalized values. PASS.
- **Assessment:** PASS with caution. Theory less clear than #1/#3. Medium priority.

**Candidate #5: Prediction Error Alignment (eta_calibration)**
- **R1 (self-generated):** ✓ Computed from bare_diff and push (intrinsic)
- **R2 (performance):** ✓ Alignment measures learning appropriateness
- **R3 (non-trivial):** ? Pearson correlation can be unstable
- **R4 (ground truth):** ✓ Does not modify phi dynamics
- **R5 (frozen frame):** ✓ Uses existing bare_diff and push
- **F1-F6 violations:** NONE
- **Assessment:** PASS with caution. Pearson correlation stability concerns. Medium priority.

### TIER 3: Revise or Reject

**Candidate #2: Plasticity Consensus (eta_mismatch)**
- **R1 (self-generated):** ✓ Computed from weights and push (intrinsic)
- **R2 (performance):** ❓ Unclear if mismatch correlates with improvement
  - **Concern:** May reward frozen consensus (bad) over productive disagreement (good)
  - **Concern:** "Alignment" requires threshold tuning — introduces new hyperparameter
- **R3 (non-trivial):** ? Depends on threshold
- **R4 (ground truth):** ✓ Does not modify phi dynamics
- **R5 (frozen frame):** ⚠️ Threshold for "aligned" is a new frozen hyperparameter
  - **Violation:** Introducing threshold without removing two frozen elements violates R5
- **F1-F6 violations:** NONE (but R5 violation)
- **Assessment:** CONDITIONAL REJECT. Violates R5 unless threshold can be made adaptive or eliminated. Lowest priority.

---

## Phase 1 Results Summary

| Candidate | R1-R5 | F1-F6 | Decision | Priority |
|-----------|-------|-------|----------|----------|
| #1: Alpha Trajectory Variance | PASS | PASS | **ADVANCE** | 1 |
| #3: Phi Stability | PASS | PASS | **ADVANCE** | 2 |
| #4: Response Autocorrelation | PASS | PASS | **ADVANCE** | 3 |
| #5: Prediction Error Alignment | PASS | PASS | **ADVANCE** | 4 |
| #2: Plasticity Consensus | FAIL (R5) | PASS | **REJECT** | 5 |

**Recommendation:** Proceed to Phase 2 with Candidates #1, #3, #4, #5 in priority order.

---

## Phase 2 Experiment Schedule

Assuming sequential execution (one candidate at a time to avoid resource contention):

| Day | Candidate | Activity | Deliverable |
|-----|-----------|----------|-------------|
| 1 | #1: Alpha Trajectory Variance | Implement + test | Statistics report + ground truth |
| 2 | #3: Phi Stability | Implement + test | Statistics report + ground truth |
| 3 | #4: Response Autocorrelation | Implement + test | Statistics report + ground truth |
| 4 | #5: Prediction Error Alignment | Implement + test | Statistics report + ground truth |

**Phase 2 gate:** Only candidates passing Phase 2 advance to Phase 3.

**Expected outcome:** 2-3 candidates pass Phase 2 (typical ~50-75% pass rate based on Session 5 local proxy search).

---

## Phase 3 Experiment Schedule

Assuming 2 candidates pass Phase 2:

| Week | Candidate | Activity | Deliverable |
|------|-----------|----------|-------------|
| 1 | #1 (if passed) | Hyperparam sweep + 10-seed validation | Statistical report |
| 2 | #3 (if passed) | Hyperparam sweep + 10-seed validation | Statistical report |

**Phase 3 gate:** Only candidates with p < 0.05 on both training and novel advance to Stage 4.

**Expected outcome:** 0-1 candidates pass Phase 3 (typical ~25-50% pass rate based on Sessions 6-8 delta_rz failure).

---

## Compute Budget Estimate

### Phase 1
- Time: 1 hour
- Cost: $0

### Phase 2 (4 candidates)
- Time: 4 days
- Seed-runs: 4 candidates × 30 runs = 120 runs
- Compute: 120 runs × 2 min = 4 hours
- Cost: $20-40

### Phase 3 (2 candidates)
- Time: 2 weeks
- Seed-runs: 2 candidates × (9 hyperparam + 20 final) = 58 runs
- Compute: 58 runs × 4 min = 4 hours
- Cost: $20-40

### Total Pipeline
- Time: 2-3 weeks
- Compute: ~8 hours GPU
- Cost: $40-80

**Cost per successful Stage 3 signal:** $80-160 (assuming 50% success rate)

---

## Success Criteria Checklist

A candidate signal is declared a **successful Stage 3 solution** if and only if:

1. ✓ Passes Phase 1 requirements filter (R1-R5, F1-F6)
2. ✓ Passes Phase 2 feasibility (statistics in range, ground truth passes 3 seeds)
3. ✓ Passes Phase 3 statistical validation:
   - Training gap: p < 0.05, d > 0.3 (adaptive > fixed)
   - Novel gap: p < 0.05, d > 0.3 (adaptive > fixed)
   - Ground truth: 10/10 seeds pass
   - Eta convergence: non-trivial distribution (not uniform, not zero, not at bounds)

**Constitution Stage 3 exit criterion:**
> "Adaptive adaptation beats fixed adaptation. Ground truth still passes. The rates converge to non-trivial values (not all equal, not all zero)."

This pipeline directly tests that criterion.

---

## Failure Handling

### If all candidates fail Phase 3:

**Interpretation:** The candidate signal family is insufficient. Stage 3 may require:
1. Combination signals (e.g., weighted sum of multiple candidates)
2. Different signal family not yet considered
3. Architectural change (e.g., Stage 2 incomplete, beta/gamma must adapt first)
4. Fundamental limitation (Stage 3 may require representation change)

**Next action:** Return to Phase 1 with expanded candidate list OR escalate to constitution review.

### If one candidate succeeds:

**Next action:** Document as canonical Stage 3 implementation. Update frozen_frame.json (eta status: frozen → adaptive). Begin Stage 4 design (structural parameters: beta/gamma).

### If multiple candidates succeed:

**Next action:** Compare effect sizes. Select strongest signal as canonical. Document others as alternatives. Investigate if combination signals are even stronger (requires new Phase 3 experiment).

---

## Implementation Notes

### Engineer (Task 2 + Phase 2)
- Implement improved eval protocol first (n_perm=8, n_trials=6)
- Create signal computation module (signal_library.py) with all 5 candidates
- Create eta adaptation module (meta_plasticity.py) with configurable rules
- Add logging for signal/eta evolution

### QA (Task 6 + Phase 2)
- Validate eval protocol matches spec (c015/c017/c019 compliance)
- Validate signal implementations match mathematical definitions
- Verify ground truth test correctness

### Analyst (Phase 2 + Phase 3)
- Compute temporal statistics (autocorrelation, dynamic range)
- Run paired t-tests, compute effect sizes
- Generate convergence plots
- Detect failure modes (oscillation, saturation, degeneracy)

### Strategist (Phase 1 + Phase 3 interpretation)
- Filter candidates (this document)
- Interpret statistical results
- Recommend next actions based on outcomes

---

## Critical Reminders

1. **One variable per experiment:** Test ONE signal at a time. Do not combine signals in Phase 3 initial tests.
2. **Ground truth first:** Every experiment MUST verify ground truth on all seeds.
3. **No skipping phases:** Phase 2 failure = do not run Phase 3 (waste of compute).
4. **Document negative results:** Failed candidates go into constraints.json with failure mode.
5. **Seed discipline:** Use fixed seed set [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337] for all Phase 3 tests (c019).

---

## Appendix: Open Questions (From Signal Spec)

### Q1: Is Per-Cell Eta Necessary?

**Candidate implications:**
- Candidates #1, #3, #4, #5 can produce either global (scalar) or per-cell (NC-dimensional) signals
- Candidate #2 inherently produces per-cell signal

**Phase 2 test:** Implement both global and per-cell versions for candidates that support both. Compare in Phase 2 statistics. If per-cell version shows no benefit (similar to c011 result for delta_rz), prefer global version (simpler, fewer parameters).

### Q2: What Relationship Between Alpha and Eta?

**Candidate implications:**
- Candidate #1: Measures alpha dynamics directly (push variance)
- Candidate #3: Measures computation (phi stability), independent of alpha
- Candidate #4: Measures response (related to alpha updates)
- Candidate #5: Measures alpha-error alignment

**Phase 3 test:** If #1 (alpha-dependent) and #3 (alpha-independent) both succeed, compare effect sizes. This reveals whether Stage 3 signal MUST monitor Stage 2 state or can be independent.

### Q3: Does Stage 3 Require Stage 2 Convergence?

**Current assumption:** Alpha has converged to stable distribution before eta adaptation begins (implicit in 100-step protocol).

**Phase 2 test:** Measure alpha statistics in first 20 steps vs. last 20 steps. If alpha is still changing significantly in final 20 steps, Stage 2 may not be converged. This could confound Stage 3 signal interpretation.

**Mitigation:** If alpha convergence is slow, extend run duration or implement staged activation (alpha adapts for 50 steps, then eta adaptation activates).

---

*This pipeline implements the 3-phase validation protocol from the Experiment Gate decision. It synthesizes requirements spec (Task 1) with candidate list (Task 3) to create concrete, executable experiments with defined success criteria, cost estimates, and failure handling.*
