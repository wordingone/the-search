# Stage 3 Candidate Signals for Eta Adaptation

## Scope

Stage 3 requires a signal family fundamentally different from resp_z derivatives (constraint c013). This analysis identifies intrinsic, computable signals that could drive adaptation of eta (the learning rate itself), making the adaptation RATE adaptive.

**Core requirements (Principle II):**
- Signal must be computed by the system's own dynamics, not by an external evaluator
- Signal must be computable in O(N) time per step (N = 72 cells × dimensions)
- Signal must capture QUALITY of adaptation, not just magnitude
- Must pass ground truth test after any implementation

---

## Intermediate Values Available (from the_living_seed.py)

Every step produces:
1. **phi_bare** (NC×D): tanh output without signal
2. **phi_sig** (NC×D): tanh output with signal
3. **response** (NC×D): |phi_sig - phi_bare|
4. **resp_z** (NC×D): normalized response (z-score)
5. **col_mean** (D): mean alpha per dimension
6. **dev** (NC×D): individual alpha - column mean
7. **push** (NC×D): actual alpha update magnitude
8. **weights** (NC×NC): attention matrix (softmax of dot products)
9. **bare_diff** (NC×D): phi_bare - xs (prediction error)
10. **plast** (NC): plasticity factor (state change magnitude)

---

## Candidate Signal #1: Alpha Trajectory Variance

**Name:** `sigma_eta` — Normalized alpha update variance across population

**Mathematical definition:**
```
push[i,k] = alpha_update magnitude for cell i, dimension k
push_flat = [push[i,k] for all i,k]
mean_push = mean(push_flat)
sigma_push = std(push_flat)
alpha_quality = sigma_push / (mean_push + 1e-10)  # diversity-to-magnitude ratio
```

**Principle II compliance:** YES
- Computed directly from push magnitudes generated during plasticity block (lines 178-190)
- Not derived from resp_z; it's a POPULATION-LEVEL property of the updates
- Intrinsic to the dynamics: removing the computation of push removes this signal

**Computational cost:** O(NC × D) = O(72)
- Single pass through push array after all updates
- Two reductions (sum, sum-of-squares)

**Why it works for Stage 3:**
- High variance = cells diverging in alpha trajectories = potential for specialization
- Low variance = cells clustering = saturation or stuck equilibrium
- Signal is QUALITY metric: captures whether the ensemble is maintaining plasticity
- Natural meta-signal: if all cells push equally, learning is becoming uniform (bad)
- If variance drops to near-zero, eta might need to increase (break symmetry)

**Biological intuition:**
- Population-level entropy of adaptation direction
- Reflects whether system maintains adaptive capacity across the ensemble
- Unlike resp_z (which saturates at ±1), push variance can drive continuous feedback

**Risks/concerns:**
- Variance itself is noisy, especially early in runs
- May need low-pass filtering to avoid oscillation
- Could confound with specialization (want variance, but not infinite divergence)

---

## Candidate Signal #2: Plasticity Consensus (Mismatch Between Attention and Update Direction)

**Name:** `eta_mismatch` — Alignment of attention weights with alpha update direction

**Mathematical definition:**
```
For each cell i:
  update_dir[i] = sign(sum(push[i,k] for k in D))  # Net direction of push across dimensions
  attn_weight[i] = max(weights[i])  # Strongest attended neighbor

  Can_cells_agree[i] = 1 if attn_weight[i] and update_dir[i] align, else 0

agreement_score = mean(Can_cells_agree) ∈ [0,1]
eta_mismatch = 1 - agreement_score  # Low when cells that attend similar others also update similarly
```

**Principle II compliance:** YES
- Both attention weights and push magnitudes are computed in the step function
- No external evaluator; purely intrinsic observation of dynamics
- If you remove either attention (lines 173-186) or plasticity (lines 149-191), signal disappears

**Computational cost:** O(NC) = O(6)
- Max operation per cell, comparison operation
- Negligible compared to other step operations

**Why it works for Stage 3:**
- When mismatch is high: cells are updating contrary to their attended neighbors
- Could mean the system is failing to benefit from attention signal
- Could indicate need to slow learning (cells thrashing against each other)
- Natural self-diagnostic: is the adaptation COHERENT with the network structure?

**Biological intuition:**
- Checks whether plastic changes are respecting the information flow (weights)
- Detects conflict: "I'm attending to cell j, but I'm updating opposite to what j is doing"
- High mismatch = poor signal coordination = reduce eta

**Risks/concerns:**
- May not correlate with actual MI improvement
- Needs careful thresholding (what counts as "aligned"?)
- Could be dominated by initialization artifacts early in runs
- Might reward frozen consensus (bad) over productive disagreement (good)

---

## Candidate Signal #3: Phi Stability (Computational Steady State)

**Name:** `phi_stability` — L2 distance between consecutive phi values (cross-step coherence)

**Mathematical definition:**
```
Store phi_sig from previous step as phi_prev
In current step:
  phi_sig_new computed normally

phi_change[i,k] = (phi_sig_new[i,k] - phi_prev[i,k])^2
sum_change = sum of all phi_change[i,k]
stability = 1.0 / (1.0 + sqrt(sum_change))  # Sigmoid of change magnitude

Interpretation: stability ≈ 1 = phi changing slowly, stability ≈ 0 = phi chaotic
```

**Principle II compliance:** YES
- Computed from phi values that are generated in every step (lines 122-147)
- No external reference; entirely self-contained measurement
- Requires only storing one previous phi (NC×D floats)

**Computational cost:** O(NC × D) = O(72)
- Element-wise subtraction and squaring
- Single sum reduction

**Why it works for Stage 3:**
- High stability = system converging to fixed points or limit cycles (potentially bad: stuck)
- Low stability = system chaotic or rapidly exploring (potentially good, but could overshoot)
- INTERMEDIATE stability may indicate healthy exploration with integration
- Natural meta-signal: "Is the computation settling into a productive pattern?"
- Captures different timescale than response: response measures SIGNAL-DRIVEN change; phi_stability measures OVERALL coherence

**Biological intuition:**
- Nervous system equilibrium: too stable = comatose, too chaotic = seizure
- Healthy system maintains dynamic but bounded exploration
- Meta-learning signal: if stability is too high, eta could increase to escape plateaus
- If stability is too low, eta could decrease to allow integration

**Risks/concerns:**
- Trivially high in early steps (phi is changing by design)
- May have natural oscillations that don't correlate with learning quality
- Could penalize rapid transitions that are actually beneficial
- Needs long-window smoothing to avoid noise-driven oscillations

---

## Candidate Signal #4: Response Autocorrelation (Temporal Consistency of Plasticity)

**Name:** `response_coherence` — Lag-1 autocorrelation of response magnitude (excluding resp_z derivative)

**Mathematical definition:**
```
Store response_magnitude from previous signal-bearing step (when signal ≠ None)
In current signal-bearing step:
  response_flat_new = [response[i,k] for all i,k]
  response_flat_old = [stored response from prior signal step]

  (requires both steps to have signal)

autocorr = pearson_correlation(response_flat_new, response_flat_old)
response_coherence = max(0, autocorr)  # Clamp to [0,1], we care about positive correlation
```

**Principle II compliance:** YES (with caveat)
- NOT a derivative of resp_z (which we already showed collapses)
- IS a temporal property computed on the raw response magnitude
- The autocorrelation itself (not the correlation direction) is the signal
- Intrinsic: cannot be computed by external observer without storing previous state

**Computational cost:** O(NC × D) = O(72) + O(1) correlation
- Two flattening operations (cheap)
- Single correlation calculation

**Why it works for Stage 3:**
- High autocorr = response pattern stable over time (same cells/dimensions responsive across steps)
- Low autocorr = response pattern random or anti-correlated (system erratic)
- DIFFERENT from resp_z autocorr: we're looking at signal magnitude structure, not z-score
- Captures: "Are the same regions of the system responding consistently?"
- If cells that were plastic yesterday are still plastic today = system maintaining specialization

**Biological intuition:**
- Memory of plasticity patterns: where did we learn last step?
- If the same regions keep being sensitive = stable learning structure (good)
- If response jumps around randomly = maybe eta is too high (destabilizing)

**Risks/concerns:**
- Only defined when signal is present (only during signal-bearing steps)
- May conflate low autocorr with good exploration vs. bad noise
- Needs careful initialization (first signal-bearing step has no prior)
- Constraint c012 ruled out resp_z derivatives; this is different but needs verification it truly is different

---

## Candidate Signal #5: Prediction Error Alignment (Phi Accuracy vs. Learning)

**Name:** `eta_calibration` — Alignment of prediction error with plasticity strength

**Mathematical definition:**
```
For each cell i:
  bare_diff[i] = phi_bare[i] - xs[i]  (prediction error from current dynamics)
  error_mag[i] = norm(bare_diff[i])

  plasticity[i] = exp(-(fp_d[i]^2) / 0.0225)  (from line 195)
  push_mag[i] = sum(|push[i,k]| for k in D)  (total alpha update magnitude)

For all cells:
  correlation = pearson(error_mag, push_mag)
  calibration = abs(correlation)  # We want moderate positive correlation

eta_calibration = calibration  # Signal: are cells learning where they're failing?
```

**Principle II compliance:** YES
- All components (bare_diff, phi, push) computed internally
- Correlation is intrinsic metric on internal state
- Not an external evaluator observing from outside

**Computational cost:** O(NC × D) for error norms + O(NC) for correlation
- Prediction error already computed (line 213)
- Plasticity already computed (line 195)

**Why it works for Stage 3:**
- Ideally: cells with high prediction error SHOULD be learning more (high push)
- If correlation is low: cells are learning uniformly regardless of error
- If correlation is negative: cells are learning where they're succeeding (bad)
- Signal captures: "Is the plasticity rule RESPONDING APPROPRIATELY to errors?"
- Natural meta-signal: if calibration drops, learning rule may be miscalibrated → adjust eta

**Biological intuition:**
- Error-driven learning: bigger mistakes → bigger plastic changes
- If a cell has zero error (prediction perfect), it shouldn't be learning much
- If a cell is making big errors, it SHOULD be learning
- Meta-signal: "Are we learning in the right places?"

**Risks/concerns:**
- Requires storing error_mag and push_mag across a full step
- Pearson correlation can be unstable with low variance in either variable
- May not correlate with actual performance (can have good internal alignment with poor external performance)
- Early in runs, correlation may be random due to initialization

---

## Ranking for Immediate Experimentation

Based on Principle II compliance, implementability, and theoretical grounding:

### TIER 1 (Highest Priority)
1. **Alpha Trajectory Variance** (#1): Simplest, most directly computable, captures ensemble-level adaptation health
2. **Phi Stability** (#3): Captures computational coherence; different timescale from response; no derivative dependence

### TIER 2 (High Priority, Implementation Dependent)
3. **Response Autocorrelation** (#4): Captures temporal structure WITHOUT using resp_z derivatives; needs careful implementation
4. **Prediction Error Alignment** (#5): Captures learning appropriateness; good candidate for calibration feedback

### TIER 3 (Secondary, Validation First)
5. **Plasticity Consensus** (#2): Requires threshold tuning; may reward wrong behaviors; lower confidence on phase transition detection

---

## Implementation Recommendations

1. **Start with Candidate #1 (Alpha Trajectory Variance)**
   - Implement as: `eta_next = eta * (1 + alpha * sigma_push_ratio)`
   - Try alpha ∈ {0.001, 0.01, 0.1} with 10+ seed validation
   - Ground truth test: measure gap on novel signals; should be ≥ baseline

2. **Implement Candidate #3 (Phi Stability) as Second Signal**
   - Compute per-step; use low-pass filter (e.g., exponential moving average)
   - Try mixed signal: `eta_next = eta * (1 + alpha1 * sigma_push + alpha2 * phi_stability)`
   - Test if combination captures different aspects of learning quality

3. **For Candidates #4 and #5**
   - Implement after establishing #1 works
   - These are more complex; validate theory first before expensive 10-seed runs

4. **All candidates require**
   - 10+ seed validation (constraint c015)
   - 2× exposure protocol: n_perm=8, n_trials=6
   - Novel signal generalization test
   - Ground truth pass on all seeds

---

## Critical Notes

- **No derivative tower escalation**: All candidates explicitly avoid stacking derivatives (c012, c013)
- **Intrinsic only**: Each can be computed inside the step function; none require external MI measurement
- **One variable per experiment**: Test ONE candidate signal at a time with ONE eta meta-parameter
- **Ground truth first**: Before expensive search, verify all candidates pass ground truth on 3 seeds
- **Avoid overfitting**: Novel signal generalization is mandatory
