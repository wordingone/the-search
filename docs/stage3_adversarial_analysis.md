# Stage 3 Candidate Signals: Adversarial Analysis

## Purpose

Team-lead raised critical challenges to the 5 candidate signals from researcher's Task #3. This document addresses each challenge with concrete mathematical definitions using ACTUAL variables from `the_living_seed.py` and resolves directional ambiguities.

---

## The Three Adversarial Tests

For each candidate:
1. **Grounding:** What is the EXACT mathematical formula using ACTUAL variables from the_living_seed.py?
2. **Direction:** When this signal is high, should eta go UP or DOWN? WHY?
3. **Family:** Is this truly a new signal family or a transform of resp_z?

---

## Candidate #1: Alpha Trajectory Variance

### Challenge: "Direction ambiguity. High variance = good specialization or bad chaos?"

### Grounding (Test 1)

**Exact formula from the_living_seed.py lines 169-190:**

```python
# Inside step(), after plasticity block:
push[i][k] = the actual alpha update computed in lines 176-185

For each (i,k):
  if abs(dev) < 0.01:
      push[i][k] = eta * 0.3 * gauss(0,1)
  elif resp_z > 0:
      direction = 1.0 if dev > 0 else -1.0
      push[i][k] = eta * tanh(resp_z) * direction * 0.5
  else:
      push[i][k] = eta * 0.1 * gauss(0,1)

# Line 190: total_alpha_shift += abs(alpha_new - alpha_old)
```

**Proposed Stage 3 signal:**

```python
push_magnitudes = [abs(push[i][k]) for all i in NC, k in D]  # 72 values
mean_push = sum(push_magnitudes) / 72
variance_push = sum((p - mean_push)^2 for p in push_magnitudes) / 72
std_push = sqrt(variance_push)

signal = std_push / (mean_push + 1e-10)  # Coefficient of variation
```

**Variables used:**
- `push[i][k]`: Computed line 178, 182, 185 (not stored in code, but computable)
- `self.alpha[i][k]`: State variable lines 110-113, 188-189

### Direction (Test 2)

**When signal is HIGH (high CV of push magnitudes):**

**Interpretation:** Cells are updating alpha at vastly different rates. Some cells have large push, others have small push. This indicates heterogeneity in learning.

**Two scenarios:**
- **Good:** Early in adaptation, cells discovering different specializations → KEEP or INCREASE eta to encourage exploration
- **Bad:** Late in adaptation, some cells saturated, others thrashing → DECREASE eta to stabilize

**AMBIGUITY CONFIRMED.** The signal alone cannot distinguish good heterogeneity from bad divergence.

**Resolution:** Require TEMPORAL CONTEXT.

**Revised signal formula:**

```python
# Compute current CV (as above)
cv_current = std_push / (mean_push + 1e-10)

# Store from previous signal-bearing step
cv_previous = stored_value

# Temporal derivative
delta_cv = cv_current - cv_previous

# Directional rule:
if delta_cv > 0:
    # CV increasing = divergence accelerating
    # If mean_push is also high: productive exploration → KEEP eta
    # If mean_push is low: cells stuck in different modes → DECREASE eta
    if mean_push > threshold:
        eta_adjustment = 0  # neutral
    else:
        eta_adjustment = -1  # decrease eta
else:
    # CV decreasing = convergence
    # If mean_push is also decreasing: settling into equilibrium → KEEP eta
    # If mean_push is high but CV decreasing: cells aligning → KEEP eta
    eta_adjustment = 0  # neutral
```

**This is now a 2D signal:** (cv_current, delta_cv) or (cv_current, mean_push).

**Problem:** This introduces a NEW frozen element (the threshold or the combination rule). Violates R5.

### Family (Test 3)

**Is this a transform of resp_z?**

**Analysis:**
- `resp_z` computed line 171: `resp_z = (response[i][k] - overall_mean) / overall_std`
- `push` computed lines 178-185: uses `resp_z` as INPUT (line 182: `push = eta * tanh(resp_z) * direction * 0.5`)
- **Push IS a function of resp_z**

**Chain:** input → phi → response → resp_z → push → alpha

**Variance of push = variance of f(resp_z)** where f includes tanh, direction, and stochastic terms.

**Verdict:** Push variance is a DERIVED signal from resp_z, not an independent family.

**Is it a different ENOUGH family?**

Constraint c012 says resp_z autocorrelation is 0.98 (highly persistent). If push autocorrelation is DIFFERENT (e.g., < 0.5), then push statistics may carry information resp_z does not.

**Needs empirical test:** Measure autocorrelation of push variance across steps. If autocorr(push_variance) < 0.5 while autocorr(resp_z) = 0.98, then push_variance is a DIFFERENT timescale signal, not just a transform.

### Verdict

**Phase 1:** CONDITIONAL PASS
- Requires Phase 2 test: measure autocorr(push_variance) vs. autocorr(resp_z)
- If autocorr(push_variance) < 0.5: PASS (different timescale)
- If autocorr(push_variance) > 0.8: FAIL (same family as resp_z, violates c013)

**Directional rule:** UNDEFINED without additional context (CV alone is ambiguous)

**Recommended revision:** Abandon pure CV signal. Use push_variance + mean_push as 2D signal OR measure temporal derivative of CV (requires history storage).

---

## Candidate #2: Plasticity Consensus (Mismatch)

### Challenge: "The Living Seed has no attention mechanism. What are 'attention weights'?"

### Grounding (Test 1)

**Actual variables from the_living_seed.py lines 193-205:**

```python
# Line 193-205: ATTENTION block
weights = []
for i in range(NC):
    raw = []
    for j in range(NC):
        if i == j:
            raw.append(-1e10)  # Mask self
        else:
            d = sum(xs[i][k] * xs[j][k] for k in range(D))  # Dot product
            raw.append(d / (D * tau))
    mx = max(raw)
    exps = [exp(min(v - mx, 50)) for v in raw]
    s = sum(exps) + 1e-15
    weights.append([e / s for e in exps])  # Softmax

# Line 222: pull[k] += weights[i][j] * (phi_bare[j][k] - phi_bare[i][k])
```

**THE SYSTEM DOES HAVE ATTENTION.** Line 193-205 computes softmax attention over cells based on state similarity. Line 222 uses weights to pull toward attended neighbors.

**Researcher's proposed signal:**

```python
# For each cell i:
update_dir[i] = sign(sum(push[i][k] for k in D))  # Net push direction
attn_weight[i] = max(weights[i])  # Strongest attended neighbor

# Alignment check:
# If cell i attends to cell j, and both are pushing in same direction → aligned
# If cell i attends to cell j, but pushing opposite directions → misaligned
```

**Problem:** `push[i][k]` is per-dimension. `weights[i][j]` is per-neighbor. These are different granularities.

**Also:** `push` is only computed when `alive and signal` (line 157). Attention is computed every step (line 193). They don't have the same temporal coverage.

### Direction (Test 2)

**When mismatch is HIGH:**

Researcher claims: cells updating contrary to attended neighbors → reduce eta (chaos)

**But:** Attention pulls state toward neighbors (line 222). Push updates alpha (plasticity strength). These are ORTHOGONAL operations. State and alpha are different variables.

**The alignment of state similarity (what attention measures) and plasticity direction (what push measures) is not obviously meaningful.**

**Counter-scenario:** A cell might attend to a neighbor because they have similar CURRENT state, but that cell might need to adapt alpha DIFFERENTLY to specialize for a different role. High mismatch could be PRODUCTIVE specialization, not chaos.

**AMBIGUITY CONFIRMED.** The signal conflates two independent adaptation mechanisms.

### Family (Test 3)

Push is derived from resp_z (as established in Candidate #1 analysis). Attention is computed from state dot products (independent of resp_z). The correlation between them is a CROSS-FAMILY signal.

**Verdict:** Not a pure resp_z derivative, but requires push (which is resp_z-derived).

### Verdict

**Phase 1:** REJECT

**Reasons:**
1. Attention and push operate on different variables (state vs. alpha) → unclear why alignment matters
2. Attention and push have different temporal coverage (every step vs. signal-bearing steps only)
3. Directional interpretation is ambiguous (high mismatch = productive specialization or destructive chaos?)
4. Original rejection reason stands: requires threshold for "aligned" (introduces frozen hyperparameter, violates R5)

---

## Candidate #3: Phi Stability

### Challenge: "Stable phi = learning complete (lower eta) or stuck (raise eta)? Ambiguous."

### Grounding (Test 1)

**Exact formula from the_living_seed.py lines 134-147:**

```python
# phi_sig computed lines 135-145
phi_sig[i][k] = tanh(alpha[i][k] * xs[i][k]
                     + beta * (xs[i][kp] + gamma*signal[kp])
                            * (xs[i][km] + gamma*signal[km]))

# Proposed signal: L2 distance between consecutive phi_sig values
phi_sig_prev = stored from previous step
phi_sig_current = computed this step

phi_change = sum((phi_sig_current[i][k] - phi_sig_prev[i][k])^2
                 for all i,k)
stability = 1.0 / (1.0 + sqrt(phi_change))
```

**Variables used:**
- `phi_sig[i][k]`: Computed lines 135-145
- Requires storing `phi_sig` from previous step (NC × D = 72 floats)

### Direction (Test 2)

**When stability is HIGH (phi changing slowly):**

**Two interpretations:**
- **Good:** System has converged to a stable attractor → learning complete → DECREASE eta (no more adaptation needed)
- **Bad:** System is stuck in a local minimum → INCREASE eta to escape

**When stability is LOW (phi changing rapidly):**

**Two interpretations:**
- **Good:** System is actively exploring → KEEP or INCREASE eta (productive search)
- **Bad:** System is chaotic/diverging → DECREASE eta to stabilize

**AMBIGUITY CONFIRMED.** Stability alone cannot distinguish convergence from stagnation, or exploration from chaos.

**Resolution attempt:** Combine with performance.

**Revised signal:**

```python
# Compute stability (as above)
# Compute performance proxy: training gap or MI (requires external measurement)
# If stability HIGH and performance HIGH: converged successfully → DECREASE eta
# If stability HIGH and performance LOW: stuck in bad attractor → INCREASE eta
# If stability LOW and performance INCREASING: productive exploration → KEEP eta
# If stability LOW and performance DECREASING: chaos → DECREASE eta
```

**Problem:** Performance measurement (MI, training gap) violates F1 (external measurement, c004). We cannot use it.

**Alternative resolution:** Combine with ground truth.

**Revised signal #2:**

```python
# Compute stability (as above)
# Run ground truth test every N steps (cheap, binary)
# If stability HIGH and ground truth PASS: converged → DECREASE eta
# If stability HIGH and ground truth FAIL: stuck → INCREASE eta
```

**Problem:** Ground truth is a PASS/FAIL test, not a continuous signal. Running it every step is computationally expensive and violates the "fast" requirement (Principle V).

**Alternative resolution #3:** Temporal derivative of stability.

```python
stability_current = (as above)
stability_prev = stored
delta_stability = stability_current - stability_prev

# If delta_stability > 0: system is settling (phi change decreasing)
#   → Learning slowing → DECREASE eta (adaptation complete)
# If delta_stability < 0: system is accelerating (phi change increasing)
#   → Learning starting or destabilizing → INCREASE eta
```

**This is plausible but requires empirical validation.**

### Family (Test 3)

**Is phi stability a transform of resp_z?**

**Chain:** input → phi

**resp_z chain:** input → phi → response → resp_z

**Phi stability is UPSTREAM of resp_z.** It measures raw computational dynamics, not response to signal.

**Verdict:** Different family. NOT a resp_z derivative.

### Verdict

**Phase 1:** CONDITIONAL PASS

**Requirement:** Must define DIRECTIONAL rule. Three options:

1. **Use temporal derivative:** delta_stability (requires history storage)
2. **Use combined 2D signal:** (stability, some other intrinsic variable)
3. **Accept ambiguity and test empirically:** Try both `eta ∝ stability` and `eta ∝ 1/stability` in Phase 2, see which works

**Recommendation:** Option 1 (temporal derivative) is most principled. Delta_stability > 0 = settling → decrease eta. Delta_stability < 0 = destabilizing → increase eta.

**Family:** PASS (not resp_z-derived)

---

## Candidate #4: Response Autocorrelation

### Challenge: "DANGER — this may violate c013. Autocorrelation OF response is still in resp_z family."

### Grounding (Test 1)

**Exact formula from the_living_seed.py lines 158-161:**

```python
# Line 158-161: response computation
if self.alive and signal:
    response = []
    for i in range(NC):
        response.append([abs(phi_sig[i][k] - phi_bare[i][k]) for k in range(D)])
```

**Researcher's proposed signal:**

```python
# Store response from previous signal-bearing step
response_prev = stored

# Current signal-bearing step
response_current = (as above)

# Flatten both
response_flat_prev = [response_prev[i][k] for all i,k]
response_flat_current = [response_current[i][k] for all i,k]

# Lag-1 autocorrelation
autocorr = pearson_correlation(response_flat_current, response_flat_prev)
signal = max(0, autocorr)  # Clamp to [0,1]
```

### Challenge Analysis (Test 3)

**Constraint c012 finding:** "resp_z derivative tower collapses at order 1 — higher-order derivatives cannot drive adaptation"

**Constraint c013:** "Stage 3+ requires a fundamentally different signal family, not derivatives of resp_z"

**Key question:** Is autocorrelation(response) a derivative of resp_z?

**Definitions:**
- `response[i][k]` = |phi_sig[i][k] - phi_bare[i][k]|  (raw magnitude, line 160)
- `resp_z[i][k]` = (response[i][k] - overall_mean) / overall_std  (z-scored, line 171)

**resp_z is a TRANSFORM of response, not a derivative.**

**Autocorrelation of response ≠ derivative of resp_z.**

**But:** Is autocorrelation of response in the SAME FAMILY as resp_z?

**Family hierarchy:**
- Order 0: `response[i][k]` (raw magnitude)
- Transform of Order 0: `resp_z[i][k]` (z-scored response)
- Temporal statistic of Order 0: `autocorr(response)` (lag-1 correlation)
- Order 1 derivative: `delta_rz = resp_z[t] - resp_z[t-1]`  (c012: autocorr = 0.11)
- Order 2 derivative: `delta_delta_rz = delta_rz[t] - delta_rz[t-1]`  (c012: autocorr = -0.48)

**Constraint c012 showed:** autocorr(resp_z) = 0.98 (order 0), autocorr(delta_rz) = 0.11 (order 1).

**Proposed signal computes:** autocorr(response), which is autocorr(pre-transform resp_z).

**This is the AUTOCORRELATION ITSELF, not a value in the tower.**

**Analogy:**
- Measuring temperature over time: [70°F, 72°F, 71°F, ...]
- First derivative (rate of change): [+2°F, -1°F, ...]
- Autocorrelation: "How similar is today's temperature to yesterday's?"

**Autocorrelation is a PROPERTY of the signal, not a point in the derivative tower.**

**However:** The c012 experiment ALREADY MEASURED autocorrelation of resp_z (0.98). Researcher is proposing to measure autocorrelation of response (pre-z-scoring).

**Is pre-z-scored response different enough from resp_z?**

Z-scoring is a linear transform: `z = (x - μ) / σ`. Autocorrelation is INVARIANT under linear transforms (Pearson correlation removes mean and scales by std).

**Therefore:** autocorr(response) = autocorr(resp_z) = 0.98 (approximately)

**VERDICT:** This signal is REDUNDANT with c012 measurement. It's not a derivative, but it's not NEW information either.

### Direction (Test 2)

**When autocorr(response) is HIGH:**

Researcher claims: "same cells/dimensions responsive across steps = stable learning structure (good)"

**But:** c012 showed autocorr(resp_z) = 0.98 already. This is saying "resp_z is highly persistent." We already know this.

**What should eta do when autocorr is HIGH?**

If cells consistently respond to the same patterns → specialization is stable → DECREASE eta (no more adaptation needed)?

Or → cells are stuck responding to the same patterns → INCREASE eta to break out?

**AMBIGUITY CONFIRMED.**

### Verdict

**Phase 1:** REJECT

**Reasons:**
1. Autocorr(response) ≈ autocorr(resp_z) ≈ 0.98 (c012 already measured this) → NO NEW INFORMATION
2. Signal is REDUNDANT with known resp_z property → violates c013 spirit (must be fundamentally different family)
3. Directional interpretation ambiguous (high autocorr = good stability or bad stagnation?)
4. Even if we use this, it's a SCALAR derived from resp_z family → doesn't escape the family

---

## Candidate #5: Prediction Error Alignment

### Challenge: "What is 'error' when there's no loss function? (Principle I)"

### Grounding (Test 1)

**Exact formula from the_living_seed.py lines 212-214:**

```python
# Line 212-214: prediction error computation
bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
fp_d = vnorm(bare_diff) / max(vnorm(xs[i]), 1.0)
plast = exp(-(fp_d^2) / 0.0225)
```

**This is PREDICTION ERROR.** `phi_bare` is the system's prediction (forward dynamics without signal). `xs` is the current state. `bare_diff` is the mismatch.

**This is NOT a loss function.** It's computed internally as part of the dynamics (line 214: `plast` gates the attention-driven state update, lines 216-223).

**Principle I states:** "Remove all external loss functions... Does it still produce distinguishable outputs?"

**bare_diff is INTERNAL, not external.** It exists in the computation, not as an external evaluator.

**GROUNDING CONFIRMED.**

**Researcher's proposed signal:**

```python
# For each cell i:
error_mag[i] = vnorm(bare_diff[i])  # Line 213: fp_d * vnorm(xs[i])
push_mag[i] = sum(abs(push[i][k]) for k in D)  # Total alpha update

# Across all cells:
correlation = pearson(error_mag, push_mag)
signal = abs(correlation)
```

**Variables used:**
- `bare_diff[i]`: Computed line 212
- `push[i][k]`: Computed lines 178-185 (as in Candidate #1)

### Direction (Test 2)

**When correlation is HIGH (positive):**

Interpretation: Cells with high prediction error are learning more (high push).

**Should eta go UP or DOWN?**

**If correlation is high (0.8+):** The plasticity rule is APPROPRIATELY targeting errors. Learning is calibrated. → KEEP eta (working well)

**If correlation drops (from 0.8 to 0.3):** The plasticity rule is NO LONGER targeting errors. Learning is becoming random or misaligned. → This indicates the plasticity rule is FAILING.

**But:** The plasticity rule is frozen (lines 169-190). We're in Stage 2, not Stage 6 (functional form adaptation). The rule doesn't change.

**So why would correlation drop?**

- Errors might be decreasing globally (system converging) → less variance in error_mag → correlation becomes noisy
- Errors might be increasing globally (system diverging) → more variance in error_mag → push might saturate or become noisy

**Directional rule:**

```python
if correlation > 0.7:
    # Learning is appropriately error-driven
    # KEEP eta (rule working as intended)
    eta_adjustment = 0
elif correlation < 0.3:
    # Learning has become decorrelated from errors
    # Could mean: (1) errors are too small (converged), or (2) errors are too large (diverging)
    # Need additional context: check mean(error_mag)
    if mean(error_mag) < threshold_low:
        # Converged: reduce eta
        eta_adjustment = -1
    else:
        # Diverging or saturated: reduce eta
        eta_adjustment = -1
```

**Problem:** Requires threshold (frozen hyperparameter). Violates R5.

**Alternative:** Use temporal derivative of correlation.

```python
correlation_current = (as above)
correlation_prev = stored
delta_correlation = correlation_current - correlation_prev

if delta_correlation < 0:
    # Correlation degrading = learning misalignment increasing
    # DECREASE eta (learning rule losing effectiveness)
    eta_adjustment = -1
else:
    # Correlation stable or improving
    # KEEP eta
    eta_adjustment = 0
```

**This is plausible.**

### Family (Test 3)

**Is error-push correlation a transform of resp_z?**

**Components:**
- `error_mag[i]` = vnorm(phi_bare[i] - xs[i])  (independent of resp_z, computed from phi and state)
- `push_mag[i]` = sum(abs(push[i][k]))  (derived from resp_z, as shown in Candidate #1)

**This is a CROSS-FAMILY signal:** correlation between a resp_z-independent variable (error) and a resp_z-dependent variable (push).

**Is this a new family?** Partially. Error is independent, but push is resp_z-derived. The CORRELATION is a new operation, but one component is still from the resp_z family.

**Verdict:** HYBRID family. Not pure resp_z derivative, but still uses resp_z-derived push.

### Verdict

**Phase 1:** CONDITIONAL PASS

**Requirements:**
1. Must define directional rule (temporal derivative of correlation recommended)
2. Must verify error_mag and push_mag have sufficient variance for Pearson correlation to be stable (D3: non-zero dynamic range)
3. Must acknowledge this is a HYBRID signal (partially resp_z-derived) and justify why the cross-correlation escapes c013

**Recommendation:** Advance to Phase 2 with caution. Empirical test required to verify correlation is stable and directional rule works.

---

## Summary Table: Adversarial Verdicts

| Candidate | Grounding | Direction | Family | Phase 1 Verdict |
|-----------|-----------|-----------|--------|-----------------|
| #1: Alpha Trajectory Variance | ✓ (push CV) | ❌ Ambiguous | ⚠️ Derived from resp_z | CONDITIONAL: Needs autocorr test |
| #2: Plasticity Consensus | ✓ (attention exists) | ❌ Ambiguous | ⚠️ Hybrid | REJECT: Orthogonal variables, R5 violation |
| #3: Phi Stability | ✓ (phi L2 change) | ⚠️ Needs derivative | ✓ Independent | CONDITIONAL: Use delta_stability |
| #4: Response Autocorrelation | ✓ (response lag-1) | ❌ Ambiguous | ❌ Redundant (≈0.98) | REJECT: No new info, violates c013 spirit |
| #5: Prediction Error Alignment | ✓ (error-push corr) | ⚠️ Needs derivative | ⚠️ Hybrid (partial resp_z) | CONDITIONAL: Use delta_correlation |

---

## Revised Phase 1 Pass List

### PASS (Advance to Phase 2)

**Candidate #3: Phi Stability (REVISED)**
- **Signal:** `delta_stability = stability_current - stability_prev`
- **Directional rule:** delta_stability > 0 (settling) → decrease eta; delta_stability < 0 (destabilizing) → increase eta
- **Family:** Independent of resp_z (upstream in computation)
- **New frozen elements:** Requires storing phi_prev (72 floats) — adaptive state, not hyperparameter

**Candidate #5: Prediction Error Alignment (REVISED)**
- **Signal:** `delta_correlation = correlation(error_mag, push_mag)_current - correlation_prev`
- **Directional rule:** delta_correlation < 0 (degrading alignment) → decrease eta; delta_correlation > 0 (improving) → neutral
- **Family:** Hybrid (error independent, push resp_z-derived) — cross-family correlation may escape c013
- **New frozen elements:** Requires storing correlation_prev (1 float) — adaptive state, not hyperparameter

### CONDITIONAL (Requires Phase 2 Empirical Test)

**Candidate #1: Alpha Trajectory Variance (REVISED)**
- **Signal:** `cv_push = std(push_magnitudes) / mean(push_magnitudes)`
- **Empirical test:** Measure autocorr(cv_push) across 10 steps, 3 seeds
  - If autocorr < 0.5: PASS (different timescale from resp_z = 0.98)
  - If autocorr > 0.8: FAIL (same family as resp_z, violates c013)
- **Directional rule:** DEFERRED pending autocorr test
  - If autocorr is low (< 0.5), signal is noisy → may need smoothing → rule TBD

### REJECT

**Candidate #2: Plasticity Consensus**
- Attention and push operate on orthogonal variables (state vs. alpha)
- Directional interpretation ambiguous
- Requires threshold hyperparameter (violates R5)

**Candidate #4: Response Autocorrelation**
- Redundant with c012 finding (autocorr(resp_z) = 0.98 already measured)
- No new information (autocorr invariant under z-scoring)
- Violates c013 spirit (must be fundamentally different family)

---

## Critical Insight: The Directional Ambiguity Problem

**All Stage 3 signals face the same challenge:** A scalar signal measuring a system property cannot unambiguously determine whether to increase or decrease eta WITHOUT ADDITIONAL CONTEXT.

**Three solutions:**

1. **Temporal derivative:** Use rate of change of signal, not absolute value
   - delta_stability, delta_correlation, delta_cv
   - Requires history storage (1-72 floats per signal)
   - Assumes derivative direction correlates with desirable eta adjustment

2. **2D signal:** Combine two complementary signals to disambiguate
   - (stability, ground_truth_pass) — but ground truth is expensive
   - (cv_push, mean_push) — but requires combination rule (new frozen element)
   - This risks violating R5 (introducing new frozen hyperparameters)

3. **Empirical search:** Test both directions in Phase 2, select winner
   - Try eta ∝ signal AND eta ∝ 1/signal
   - Whichever produces better training/novel gap in Phase 2 wins
   - This is EMPIRICAL, not theoretically justified, but pragmatic

**Recommendation:** Use Solution 1 (temporal derivative) for candidates #3 and #5. Use Solution 3 (empirical search) for candidate #1 pending autocorr test.

---

## Revised Phase 2 Experiment Queue

| Priority | Candidate | Signal Definition | Directional Rule | Empirical Test |
|----------|-----------|-------------------|------------------|----------------|
| 1 | #3: Phi Stability | delta_stability | Δ > 0 → ↓eta, Δ < 0 → ↑eta | Stability + delta_stability stats |
| 2 | #5: Error Alignment | delta_correlation | Δ < 0 → ↓eta, Δ ≥ 0 → neutral | Correlation stats + variance |
| 3 | #1: Push Variance | cv_push (conditional) | TBD after autocorr | Autocorr(cv_push) vs. autocorr(resp_z) |

**Total Phase 2 candidates:** 3 (down from 5)

**Expected Phase 2 → Phase 3 pass rate:** 1-2 candidates (50-67%)

---

## Analyst Note: Seed Regime Robustness

Team-lead mentioned: "Seeds cluster into HIGH/MID/LOW regimes (2× amplitude range). Any Stage 3 signal must be robust across all three regimes."

**Implication for Phase 2:** When measuring signal statistics (autocorr, dynamic range), stratify by seed regime:
- HIGH regime seeds: [?? to be identified by analyst]
- MID regime seeds: [?? to be identified by analyst]
- LOW regime seeds: [?? to be identified by analyst]

**Pass criterion:** Signal must have similar autocorr and dynamic range across all three regimes. If signal works on HIGH but not LOW, it's regime-specific (FAIL).

**Request to analyst:** Provide seed stratification before Phase 2 begins.

---

*This adversarial analysis resolves ambiguities in the original candidate list. 3 candidates advance (2 PASS, 1 CONDITIONAL). 2 candidates rejected. All passing candidates now have concrete mathematical definitions, directional rules, and family justifications.*
