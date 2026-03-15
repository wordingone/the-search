# Stage 3 Signal Requirements Specification

## Purpose

This document specifies the formal requirements for a Stage 3 adaptation signal that governs the meta-rate of eta (the Stage 2 adaptation rate). Stage 3 makes the adaptation rate itself adaptive.

## Constitutional Context

**Stage 3 Definition (Constitution §88-94):**
- **Frozen frame shrinks by:** Adaptation hyperparameters (eta) become adaptive state
- **Exit criterion:** Adaptive adaptation beats fixed adaptation. Ground truth still passes. The rates converge to non-trivial values (not all equal, not all zero).

**Stage 3 Target:** Eta (adaptation rate hyperparameter governing alpha updates)

**Current frozen frame status:** Eta is frozen at a designer-chosen constant (default 0.01 in canonical implementation).

---

## I. REQUIRED Properties

These properties are **mandatory** — derived directly from Constitution Principles I-V and the Stage 3 definition. A signal lacking any of these cannot satisfy Stage 3.

### R1. Self-Generated (Principle II)

The signal MUST be a byproduct of the system's own computation, not a separate measurement taken by a separate evaluator.

**Test:** Can the signal be computed during the forward pass using only locally available state (cell activations, plasticity state, input history)? If it requires an external oracle (MI measurement, performance evaluation, gradient through time), it violates Principle II.

**Rationale:** Principle II states: "If you can remove the adaptation mechanism without changing the computation, they are separate. They must not be separate."

### R2. Drives Performance Improvement (Stage 3 Exit Criterion)

Using the signal to adapt eta MUST produce measurable performance improvement over fixed eta on the ground truth task.

**Test:** Adaptive-eta version beats fixed-eta version on both training task (learned input patterns) and novel task (unseen input patterns). Statistical significance required: p < 0.05 with effect size detectable above noise floor (CV ≈ 20%).

**Rationale:** Stage 3 exit criterion: "Adaptive adaptation beats fixed adaptation. Ground truth still passes. Advantage holds on novel inputs."

### R3. Convergence to Non-Trivial Values (Stage 3 Exit Criterion)

The adapted eta values MUST converge to non-trivial, non-uniform distributions.

**Test:** After adaptation, eta values across cells must satisfy:
- Not all equal (heterogeneity required)
- Not all zero (non-degenerate required)
- Not all at clip bounds (stability required)

**Rationale:** Stage 3 exit criterion: "The rates converge to non-trivial values (not all equal, not all zero)."

### R4. Maintains Ground Truth (All Stages)

The system with adaptive eta MUST still pass the ground truth test.

**Test:** "The system produces distinguishable final states for distinguishable input sequences, AND this distinguishability persists after the input is removed, AND this persistence arises from the system's dynamics alone without external memory."

**Rationale:** Constitution §144: "Test after every change. The ground truth test takes seconds. Run it after every modification. No exceptions."

### R5. Shrinks Frozen Frame Monotonically (Principle IV)

Introducing the Stage 3 signal MUST NOT add new frozen elements without eliminating at least two existing frozen elements.

**Test:** Enumerate all frozen elements before and after Stage 3 implementation. The list must be strictly shorter.

**Rationale:** Principle IV: "If a new frozen element was introduced, it must be accompanied by the elimination of at least two existing frozen elements."

---

## II. FORBIDDEN Properties

These properties are **prohibited** — derived from the 19 experimentally-validated constraints. A signal with any of these properties is known to fail.

### F1. External Performance Measurement (c004)

The signal MUST NOT be computed by evaluating MI, loss, or any other performance metric that requires running the system and comparing outputs to a target.

**Source:** c004 — "Finite-diff MI gradient violates Principle II (external measurement, not self-generated)"

**Rationale:** External measurement requires a separate evaluator, creating a second frozen frame. This is exactly what Principle II forbids.

### F2. Higher-Order Derivatives of resp_z (c012, c013)

The signal MUST NOT be a second-order or higher derivative of resp_z (e.g., delta_delta_rz, third derivative).

**Source:**
- c012 — "resp_z derivative tower collapses at order 1 — higher-order derivatives cannot drive adaptation"
- c013 — "Stage 3+ requires a fundamentally different signal family, not derivatives of resp_z"

**Rationale:** The resp_z derivative tower has autocorrelation: order-0 (+0.98), order-1 (+0.11), order-2 (-0.48, anti-correlated). Order 2+ are oscillatory noise, not signals.

### F3. First-Order resp_z Derivative (delta_rz) for Per-Cell Eta (c011)

The signal MUST NOT use delta_rz to adapt per-cell eta values.

**Source:** c011 — "Per-cell eta adaptation does not improve performance — the system is insensitive to learning rate heterogeneity"

**Rationale:** Session 6 Exp B (delta_rz eta adaptation) failed 10-seed validation. Training: p=0.999 (d=-0.001), Novel: p=0.688 (d=0.19). Statistically identical to fixed eta.

### F4. Multiplicative Self-Reference (c008, c009)

The signal MUST NOT create multiplicative self-reference where eta modulates its own update (e.g., push_eta = eta × f(signal)).

**Source:**
- c008 — "Multiplicative self-referential meta-rates cause bang-bang oscillation (inherently unstable)"
- c009 — "Stage 3 meta-rate must be EXTERNAL to eta, not eta itself"

**Rationale:** Session 6 Exp A failed catastrophically. Self-referential updates create exponential growth/decay → bang-bang oscillation between clip bounds.

### F5. Analytical Gradients of Beta/Gamma (c001, c002)

The signal MUST NOT use analytical gradients derived from beta/gamma parameters.

**Source:**
- c001 — "Response-weighted analytical gradients are not reliable proxies for MI direction"
- c002 — "Analytical gradient magnitude is too weak for beta/gamma without 100-1000× learning rate boost"

**Rationale:** Z-score cancellation (mean=0, var=1) prevents collective signal. NC×D averaging dilutes gradient by factor of 72.

### F6. Local Proxies for Global Parameters (c003, c006, c007)

The signal MUST NOT attempt to decompose global coupling parameters (beta/gamma) into per-cell proxies.

**Source:**
- c003 — "No local proxy for beta/gamma exists (7 tested, best r=0.44, none >0.7)"
- c006 — "Per-cell decomposition of global coupling parameters destroys performance (53% MI loss)"
- c007 — "Beta/gamma coupling is fundamentally global, not locally decomposable"

**Rationale:** Beta/gamma define global information flow across the 1D ring. Local statistics cannot capture global coupling effects.

---

## III. DESIRABLE Properties

These properties are **beneficial but not mandatory** — derived from experimental evidence and practical considerations. They improve likelihood of success but are not strict requirements.

### D1. Computational Locality

**Preference:** Signals computable using only local state (current cell, k=1 neighbors, own plasticity history) are preferable to signals requiring global aggregation.

**Rationale:** Reduces implementation complexity, increases biological plausibility, avoids introducing global communication as a new frozen element.

### D2. Temporal Stability

**Preference:** Signals with autocorrelation in range [0.3, 0.8] are preferable to highly oscillatory (< 0.1) or nearly constant (> 0.95) signals.

**Rationale:**
- High autocorrelation (> 0.95): signal is nearly constant, provides little adaptation information (resp_z = 0.98)
- Low autocorrelation (< 0.1): signal is noisy, adaptation will be erratic (delta_rz = 0.11)
- Medium autocorrelation (0.3-0.8): signal has structure but varies meaningfully over time

### D3. Non-Zero Dynamic Range

**Preference:** Signals with mean absolute value > 0.01 and standard deviation > 0.001 are preferable.

**Rationale:** Extremely weak signals (< 0.001) may require large learning rate multipliers, effectively reintroducing a frozen hyperparameter (the multiplier).

### D4. Robustness to Initialization

**Preference:** Signals that produce qualitatively similar adaptation trajectories across different random seeds are preferable.

**Rationale:** Seed variance (CV = 20% under improved protocol) is the dominant source of noise. Signals that are sensitive to initialization will be drowned out by this variance.

### D5. Compatibility with Improved Evaluation Protocol

**Preference:** Signals that can be evaluated using the improved protocol (n_perm=8, n_trials=6, K=[4,6,8,10]) are preferable.

**Rationale:**
- c015 — "Future experiments must use n_perm=8, n_trials=6 (2× exposure) to achieve CV≈20%"
- c017 — "Adding extreme K values (K=3,12) increases variance — stick to K=[4,6,8,10]"
- c019 — "Future search must use 10+ seeds with 2× exposure protocol"

### D6. Avoids Multi-Local-Maxima Landscapes

**Preference:** If the signal involves optimization over a parameter space, that space should have a single global optimum (or at least a convex basin).

**Rationale:**
- c005 — "Beta/gamma MI landscape has multiple local maxima (no unique optimum)"
- Multi-modal landscapes make search intractable without external guidance (which violates Principle II).

### D7. Interpretability

**Preference:** Signals with clear physical or computational interpretation are preferable to opaque combinations.

**Rationale:** If a signal fails, understanding *why* requires knowing what it measures. Black-box signals (e.g., high-order polynomial combinations of many features) obscure failure modes.

---

## IV. Signal Families to Explore

Based on the REQUIRED and FORBIDDEN properties, these are candidate signal families for Stage 3:

### Candidate Family 1: Cross-Cell Coherence Measures

**Rationale:** Stage 2 uses resp_z (single-cell response strength). Stage 3 could use cross-cell correlation or coherence (multi-cell coordination).

**Example signals:**
- Lag-1 cross-correlation between neighboring cells
- Phase synchrony between cell activation trajectories
- Mutual information between cell states (NOT between input/output, which is external)

**Satisfies:**
- R1 (self-generated): computable from cell activations during forward pass
- Not F2 (higher-order derivative): uses activations directly, not derivatives of resp_z
- Not F3 (delta_rz per-cell): uses cross-cell statistics, not single-cell derivatives

**Risks:**
- May require global aggregation (conflicts with D1 if all-pairs correlation is used)
- May be computationally expensive for large NC

### Candidate Family 2: Plasticity State Statistics

**Rationale:** Alpha is adaptive (Stage 2). Statistics of alpha distribution (variance, range, skewness) could drive eta.

**Example signals:**
- Variance of alpha across cells
- Rate of change of alpha variance
- Sparsity of alpha distribution (L1/L2 ratio)

**Satisfies:**
- R1 (self-generated): computable from alpha history
- Not F2, F3 (resp_z derivatives): uses alpha, not resp_z
- Not F4 (self-reference): uses alpha state, not eta state

**Risks:**
- Indirect connection to ground truth performance (may not satisfy R2)
- May require careful normalization to avoid degenerate convergence (R3 risk)

### Candidate Family 3: Input-Reconstruction Error

**Rationale:** The ground truth requires persistent distinguishable states. A signal measuring how well past input can be reconstructed from current state measures persistence quality.

**Example signals:**
- L2 distance between input history and activation-predicted input
- Prediction error variance across cells
- Temporal decorrelation of prediction error

**Satisfies:**
- R1 (self-generated): computable during forward pass
- R2 (performance): directly related to ground truth (persistence/distinguishability)
- Not F1 (external measurement): uses internal state, not external evaluation

**Risks:**
- May require storing input history (conflicts with ground truth's "no external memory" requirement)
- Reconstruction mechanism itself may be a frozen element (violates R5)

### Candidate Family 4: Activity Entropy / Diversity

**Rationale:** Uniform activity (all cells the same) fails ground truth. Diversity measures could drive eta to maintain heterogeneity.

**Example signals:**
- Shannon entropy of cell activation distribution
- Gini coefficient of activation magnitudes
- Effective dimension of activation space

**Satisfies:**
- R1 (self-generated): computable from activations
- R2 (performance): related to distinguishability requirement
- Not F2, F3 (resp_z derivatives): uses activations directly

**Risks:**
- May be too indirect (entropy can be high with random noise, which fails ground truth)
- May not satisfy R2 if diversity and performance are not correlated

---

## V. Validation Requirements

Any proposed Stage 3 signal must be validated in this order:

### Phase 1: Theoretical Compliance (1 hour)

**Check:**
1. Does it satisfy all 5 REQUIRED properties (R1-R5)?
2. Does it violate any of the 6 FORBIDDEN properties (F1-F6)?
3. Does it introduce new frozen elements? (If yes, reject immediately per R5)

**Deliverable:** Written justification for R1-R5 satisfaction and F1-F6 avoidance.

### Phase 2: Computational Feasibility (1 day)

**Check:**
1. Implement signal computation in harness.py
2. Measure computational overhead (< 10% of forward pass time preferred)
3. Verify signal has non-trivial dynamic range (mean > 0.01, std > 0.001 per D3)
4. Measure temporal stability (autocorrelation per D2)

**Deliverable:** Signal statistics across 3 seeds, 10 timesteps.

### Phase 3: Adaptation Test (3 days, 10 seeds minimum per c019)

**Check:**
1. Implement eta adaptation rule driven by signal
2. Run improved protocol: n_perm=8, n_trials=6, K=[4,6,8,10], 10 seeds (c015, c017, c019)
3. Compare adaptive-eta vs fixed-eta on training + novel tasks
4. Verify ground truth still passes (R4)
5. Check convergence properties (R3): not all equal, not all zero, not at bounds

**Deliverable:**
- Statistical comparison (paired t-test, p-value, effect size)
- Ground truth pass/fail for all 10 seeds
- Eta distribution histograms at convergence

**Rejection criteria:**
- p > 0.05 on either training or novel task (fails R2)
- Ground truth fails on any seed (fails R4)
- Eta converges to uniform values or all zeros (fails R3)

---

## VI. Known Open Questions

These questions are unresolved and may affect signal design:

### Q1. Is Per-Cell Eta Necessary?

c011 shows per-cell delta_rz eta adaptation fails. Does this mean:
- (A) Eta should remain global (single value for all cells)?
- (B) Per-cell eta is necessary but delta_rz is the wrong signal?
- (C) Eta heterogeneity doesn't matter for this architecture?

**Impact:** If (A), Stage 3 signal only needs to produce a scalar. If (B), signal needs NC-dimensional output.

### Q2. What Is the Relationship Between Alpha and Eta?

Alpha is Stage 2's adaptive parameter. Eta governs alpha's adaptation rate. Should Stage 3 signal:
- (A) Measure alpha's state (e.g., variance, sparsity)?
- (B) Measure alpha's dynamics (e.g., rate of change)?
- (C) Be independent of alpha (measure computation directly)?

**Impact:** Determines which signal families are most likely to succeed.

### Q3. Does Stage 3 Require Stage 2 to Be Fully Converged?

The constitution says "do not skip stages." Does Stage 3 require:
- (A) Alpha has converged to a stable distribution before eta adaptation begins?
- (B) Alpha and eta adapt simultaneously from the start?
- (C) Alpha reaches a performance threshold before eta adaptation activates?

**Impact:** Affects experiment design and convergence criteria.

---

## VII. Summary Table

| Property ID | Type | Description | Source |
|-------------|------|-------------|--------|
| R1 | Required | Self-generated (Principle II) | Constitution §29-35 |
| R2 | Required | Drives performance improvement | Constitution §94 |
| R3 | Required | Converges to non-trivial values | Constitution §94 |
| R4 | Required | Maintains ground truth | Constitution §144 |
| R5 | Required | Shrinks frozen frame monotonically | Constitution §43-49 |
| F1 | Forbidden | External performance measurement | c004 |
| F2 | Forbidden | Higher-order resp_z derivatives | c012, c013 |
| F3 | Forbidden | delta_rz for per-cell eta | c011 |
| F4 | Forbidden | Multiplicative self-reference | c008, c009 |
| F5 | Forbidden | Analytical gradients of beta/gamma | c001, c002 |
| F6 | Forbidden | Local proxies for global parameters | c003, c006, c007 |
| D1 | Desirable | Computational locality | Practical |
| D2 | Desirable | Temporal stability (autocorr 0.3-0.8) | c012 analysis |
| D3 | Desirable | Non-zero dynamic range | c002 analysis |
| D4 | Desirable | Robustness to initialization | c010, c014 |
| D5 | Desirable | Improved protocol compatibility | c015, c017, c019 |
| D6 | Desirable | Avoids multi-local-maxima | c005 |
| D7 | Desirable | Interpretability | Operational |

---

## VIII. Recommended Next Steps

1. **Researcher:** Survey codebase for existing signals matching candidate families 1-4
2. **Analyst:** Compute temporal statistics (autocorrelation, dynamic range) for existing signals in experiment logs
3. **Strategist (this task):** Design experiment pipeline once candidate signals are identified
4. **Engineer + QA:** Implement and validate improved evaluation protocol before running Phase 3 tests

---

*This specification is derived from Constitution v2, 19 constraints, and 5 sessions of experimental evidence. Any proposed Stage 3 signal must satisfy all REQUIRED properties and avoid all FORBIDDEN properties. Desirable properties improve likelihood of success but are not strict requirements.*
