# Stage 3 Experiment Pipeline: Post-Adversarial Revision

## Summary of Changes

After team-lead's adversarial critique, the experiment pipeline has been revised. See `stage3_adversarial_analysis.md` for full justification.

## Phase 1 Results: REVISED

### PASS → Advance to Phase 2

**#3: Phi Stability (REVISED as delta_stability)**
- **Signal:** `delta_stability = stability[t] - stability[t-1]` where `stability = 1/(1 + sqrt(sum((phi_sig[t] - phi_sig[t-1])^2)))`
- **Directional rule:** delta_stability > 0 (settling) → decrease eta; delta_stability < 0 (destabilizing) → increase eta
- **Eta update:** `eta_new = eta * (1 + alpha * (-delta_stability))` where alpha ∈ [0.01, 0.1] (tune in Phase 2)
- **Family:** Independent of resp_z (phi is upstream)
- **Priority:** 1

**#5: Prediction Error Alignment (REVISED as delta_correlation)**
- **Signal:** `delta_corr = corr(error_mag, push_mag)[t] - corr[t-1]` where `error_mag[i] = ||phi_bare[i] - xs[i]||` and `push_mag[i] = sum(|push[i][k]|)`
- **Directional rule:** delta_corr < 0 (degrading alignment) → decrease eta; delta_corr ≥ 0 → neutral
- **Eta update:** `eta_new = eta * (1 + alpha * min(0, delta_corr))` (only decrease eta, never increase)
- **Family:** Hybrid (error independent, push resp_z-derived) — cross-correlation may escape c013
- **Priority:** 2

### CONDITIONAL → Phase 2 Autocorr Test Required

**#1: Push Variance (REVISED as cv_push with autocorr gate)**
- **Signal:** `cv_push = std(push_magnitudes) / mean(push_magnitudes)` where `push[i][k]` is alpha update magnitude
- **Autocorr test:** Measure lag-1 autocorr(cv_push) over 10 steps, 3 seeds
  - If autocorr > 0.8: REJECT (same family as resp_z = 0.98, violates c013)
  - If autocorr < 0.5: ADVANCE to full Phase 2 (different timescale)
- **Directional rule:** DEFERRED pending autocorr test (ambiguous without temporal context)
- **Priority:** 3

### REJECT

**#4: Response Autocorrelation**
- **Reason:** autocorr(response) ≈ autocorr(resp_z) ≈ 0.98 (c012 already measured this). No new information. Violates c013 spirit.

**#2: Plasticity Consensus**
- **Reason:** Attention (state similarity) and push (alpha updates) are orthogonal variables. Directional interpretation ambiguous. Requires threshold (violates R5).

## Phase 2 Schedule: REVISED

| Day | Candidate | Activity | Gate Criterion |
|-----|-----------|----------|----------------|
| 1 | #3: delta_stability | Implement + stats + ground truth | Dynamic range > 0.01, autocorr ∈ [0.1, 0.95], ground truth 3/3 |
| 2 | #5: delta_correlation | Implement + stats + ground truth | Dynamic range > 0.01, autocorr ∈ [0.1, 0.95], ground truth 3/3 |
| 3 | #1: cv_push | Autocorr test ONLY | If autocorr(cv_push) < 0.5: PASS, proceed to Day 4; else REJECT |
| 4 | #1: cv_push (if passed Day 3) | Full stats + ground truth | (same as Day 1-2) |

**Expected Phase 2 → Phase 3 pass rate:** 1-2 candidates (50-67%, revised down from 75%)

## Phase 3 Changes: REVISED

### Hyperparam Sweep (Step 3.2)

For temporal derivative signals (#3, #5), the hyperparameter is `alpha` (meta-learning rate multiplier):

**#3: delta_stability**
- Test alpha ∈ {0.01, 0.05, 0.1} (scaling factor for eta adjustment)
- Update rule: `eta_new = clip(eta * (1 + alpha * (-delta_stability)), 0.001, 0.1)`
- Negative sign because delta_stability > 0 should DECREASE eta

**#5: delta_correlation**
- Test alpha ∈ {0.01, 0.05, 0.1}
- Update rule: `eta_new = clip(eta * (1 + alpha * min(0, delta_correlation)), 0.001, 0.1)`
- Only decrease eta (one-way adaptation)

**#1: cv_push (if it passes Day 3)**
- Directional rule TBD based on autocorr results
- If autocorr is low (< 0.3), signal may be too noisy → requires smoothing (EMA) → introduces frozen decay constant (violates R5 risk)

### Additional Phase 3 Validation: Seed Regime Robustness

Per team-lead note: "Seeds cluster into HIGH/MID/LOW regimes. Signal must work across all three."

**New requirement for Phase 3:**
- Analyst provides seed stratification before Phase 3 begins
- 10-seed validation must include ≥3 seeds from EACH regime
- Post-hoc analysis: stratify p-values by regime
- **Fail criterion:** If p < 0.05 for HIGH regime but p > 0.2 for LOW regime → REJECT (regime-specific, not robust)

## Compute Budget: REVISED

### Phase 2
- Day 1-2: 2 candidates × 30 runs = 60 runs @ 2 min = 2 hours = $10-20
- Day 3-4: 1 candidate (conditional) × 30 runs = 30 runs @ 2 min = 1 hour = $5-10
- **Total Phase 2:** 3-4 days, $15-30

### Phase 3
- 1-2 candidates × (9 hyperparam + 20 final) = 29-58 runs @ 4 min = 2-4 hours = $10-20
- **Total Phase 3:** 1-2 weeks, $10-20

### Total Pipeline: REVISED
- Time: 2-3 weeks (unchanged)
- Compute: ~4-6 hours GPU (down from 8 hours)
- Cost: $25-50 (down from $40-80)

**Cost per successful Stage 3 signal:** $50-100 (assuming 50% Phase 3 success rate)

## Critical Implementation Notes

### For Engineer (Task 2)

When implementing temporal derivative signals, you MUST store history:

**#3: delta_stability**
- Store `phi_sig_prev` (NC × D = 72 floats) from previous step
- Store `stability_prev` (1 float) from previous signal-bearing step
- Reset history on run init

**#5: delta_correlation**
- Store `correlation_prev` (1 float) from previous signal-bearing step
- Compute `error_mag[i]` from existing `bare_diff` (line 212-213)
- Compute `push_mag[i]` from alpha updates (lines 178-190)

**#1: cv_push (conditional)**
- Store `cv_push` history for 10 steps (10 floats) for autocorr measurement
- If autocorr test passes, implement full eta adaptation with rule TBD

### For Analyst (Task 4)

**Required before Phase 3:**
1. Seed regime stratification: classify 10 seeds into HIGH/MID/LOW based on Session 8 data
2. For each regime, provide:
   - Mean training gap (canonical baseline)
   - Std training gap (canonical baseline)
   - Typical MI range
3. This enables Phase 3 post-hoc regime analysis

**If regime data not available:**
- Run 10-seed canonical baseline with 2× exposure protocol
- Cluster seeds by training gap (3 clusters via k-means)
- Document cluster boundaries for future reference

## Success Criteria: UNCHANGED

A candidate signal is a **successful Stage 3 solution** if:
1. Training gap: adaptive > fixed, p < 0.05, d > 0.3
2. Novel gap: adaptive > fixed, p < 0.05, d > 0.3
3. Ground truth: 10/10 seeds pass
4. Eta convergence: non-trivial (not uniform, not zero, not at bounds)
5. **NEW:** Regime robustness: p < 0.05 in ALL three regimes (HIGH/MID/LOW)

## Open Questions: UPDATED

### Q1: Is Per-Cell Eta Necessary?

**Status:** Deferred. Candidates #3 and #5 can produce global (scalar) eta adjustments. Test global first. If both fail, revisit per-cell hypothesis.

### Q2: One-Way vs Two-Way Adaptation

**Observation:** Candidate #5 (delta_correlation) only DECREASES eta (one-way). Candidate #3 (delta_stability) can increase OR decrease eta (two-way).

**Question:** Is two-way adaptation necessary for Stage 3, or is one-way sufficient?

**Test:** If #5 passes Phase 3 but #3 fails, one-way may be sufficient. If #3 passes but #5 fails, two-way may be necessary. If both pass, compare effect sizes.

### Q3: Temporal Derivative Lag

**Question:** Should delta be computed lag-1 (consecutive steps) or lag-N (every N steps)?

**Current assumption:** Lag-1 for signal-bearing steps only (not every step, since signal is None during non-signal steps).

**Risk:** Signal-bearing steps may be sparse (e.g., 1 every 3-5 steps). Lag-1 autocorr may be artificially low due to sparsity.

**Mitigation:** Phase 2 stats must report signal-bearing step frequency. If < 20% of steps have signal, consider lag-N derivative or EMA smoothing.

---

## Next Actions

1. **Engineer:** Implement improved eval protocol (Task 2), then implement #3 and #5 per Phase 2 specs
2. **Analyst:** Provide seed regime stratification before Phase 3
3. **QA:** Validate temporal derivative implementations match mathematical definitions
4. **Strategist:** Monitor Phase 2 results, interpret autocorr tests, revise directional rules if needed

---

*This revision addresses all adversarial challenges raised by team-lead. 2 candidates advance (PASS), 1 conditional (autocorr gate), 2 rejected. All passing candidates now use temporal derivatives to resolve directional ambiguity.*
