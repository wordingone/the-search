# Scaling Degradation: Diagnosis & Fix Proposals

**Problem:** ALIVE advantage collapses from +0.11 (D=12) to +0.005 (D=24)

**Goal:** Identify root cause and fix WITHOUT adding to frozen frame

---

## Hypothesis 1: Plasticity Rate Doesn't Scale

### Observation
Eta = 0.0003 is fixed across all dimensions. But effective learning rate should scale with D because:
- More dimensions → more alpha parameters to update
- Same eta applied to 72 params (D=12) vs 144 params (D=24) vs 288 params (D=48)
- Total alpha shift scales with D, but per-dimension shift stays constant

### Test
Run D=24 with eta scaled by dimension:
- `eta_effective = eta * sqrt(D)` or
- `eta_effective = eta * D / 12` (normalize to D=12 baseline)

### Implementation
Modify Organism.__init__:
```python
self.eta = eta * math.sqrt(D / 12.0)  # Scale relative to baseline
```

### Expected outcome
If hypothesis correct: ALIVE advantage at D=24 recovers toward D=12 level

### Constitutional impact
✓ Does not add to frozen frame (eta still fixed, just D-dependent)
⚠ Makes eta functionally dependent on D (adds implicit coupling)

---

## Hypothesis 2: Attention Dilution

### Observation
Attention weights: `softmax((xs @ xs.T) / (D * tau))`

Normalization by D means:
- Larger D → smaller logits → flatter softmax → weaker specialization
- At D=12: dot products ~ 12 * (state variance)
- At D=48: dot products ~ 48 * (state variance), divided by 48 → same scale
- BUT: if states don't scale with sqrt(D), effective temperature rises

### Test
Run D=24 with tau scaled:
- `tau_effective = tau * sqrt(D / 12)` (keep effective temperature constant)

Or remove D normalization:
- `softmax(xs @ xs.T / tau)` (scale-invariant)

### Implementation
```python
dots = (xs @ xs.T) / tau  # Remove D normalization
```

### Expected outcome
If hypothesis correct: Attention becomes sharper at higher D, ALIVE recovers

### Constitutional impact
✓ Does not add to frozen frame
⚠ Changes attention behavior (might break D=12 performance)

---

## Hypothesis 3: Signal Strength Degrades

### Observation
Signals are normalized to unit sphere, scaled by 0.8:
```python
sigs[i] = [v * 0.8 / nm for v in s]
```

But product term is `(x_{k+1} + gamma*s_{k+1}) * (x_{k-1} + gamma*s_{k-1})`

With gamma=0.9, signal contribution is `0.9 * 0.8 = 0.72` in each direction.

At higher D:
- State vectors spread over more dimensions
- Signal fixed at 0.8 norm regardless of D
- Relative signal strength shrinks: signal is constant volume, state space grows as D^(1/2)

### Test
Scale signal strength with dimension:
```python
sigs[i] = [v * 0.8 * sqrt(D / 12) / nm for v in s]
```

### Expected outcome
If hypothesis correct: Stronger signals at high D → better discrimination → ALIVE recovers

### Constitutional impact
⚠ Adds D-dependence to signal generation (frozen frame grows)
✗ Violates "novel inputs" test (signals not directly comparable across D)

**Verdict:** BAD fix. Adds to frozen frame.

---

## Hypothesis 4: Product Term Saturation

### Observation
Product term: `beta * (x_{k+1} + gamma*s) * (x_{k-1} + gamma*s)`

With more dimensions:
- State magnitudes bounded by clip=4.0
- Product can reach `0.5 * (4+0.72) * (4+0.72) ≈ 11.2`
- After alpha multiplication, input to tanh can exceed ±10
- tanh(±10) ≈ ±1.0 (saturated)

More dimensions → more opportunities for saturation → less gradient signal for adaptation

### Test
Measure saturation rate:
```python
pre_tanh = alpha * xs + beta * xs_kp * xs_km
saturation_rate = (pre_tanh.abs() > 2.0).float().mean()
```

Track saturation vs dimension. If saturation increases with D, this is the cause.

### Fix options
1. Scale beta by 1/sqrt(D): weaker coupling at high D
2. Adaptive beta: make beta itself plastic (Stage 2+)
3. Different activation: replace tanh with non-saturating function

### Constitutional impact
- Option 1: ✓ No frozen frame growth (beta still frozen, just D-dependent)
- Option 2: ✓ SHRINKS frozen frame (beta becomes adaptive) — **Stage 3 move**
- Option 3: ✗ Changes functional form (different experiment)

---

## Hypothesis 5: Task Difficulty Increases Faster Than Plasticity Helps

### Observation
STILL degrades: +0.072 → +0.080 → +0.035

This is NON-MONOTONIC (increases then crashes). Suggests:
- D=12: task easy, STILL works
- D=24: task harder, STILL struggles, ALIVE helps a little
- D=48: task very hard, both struggle, ALIVE helps more (relative recovery)

### Interpretation
Maybe scaling degradation is EXPECTED. Higher D = harder task. ALIVE maintains positive delta at all scales, which is success.

### Test
Measure task difficulty independent of ALIVE/STILL:
- Run random baselines at each D
- Measure information-theoretic capacity: how many sequences can be discriminated?
- Theory: capacity should scale as D^k for some k

If D=24 has exponentially more task difficulty, constant absolute delta might be acceptable.

### Constitutional impact
✓ No implementation change needed
⚠ Requires redefining "success" — is maintaining positive delta enough?

**Constitution says:** "Advantage holds on novel inputs"

Does NOT say advantage must be constant magnitude. Just positive.

**Verdict:** Living Seed might PASS Stage 2 even with scaling degradation, if delta stays positive.

---

## Hypothesis 6: Plasticity Rule Doesn't Generalize to High D

### Observation
Plasticity rule (lines 176-189):
```python
col_mean = sum(alpha[j][k] for j in range(NC)) / NC
dev = alpha[i][k] - col_mean

if resp_z > 0 and |dev| > 0.01:
    push = eta * tanh(resp_z) * sign(dev) * 0.5
```

This amplifies deviation from column mean. But:
- At D=12: 6 cells × 12 dims = 72 params, column mean is 6 samples
- At D=48: 6 cells × 48 dims = 288 params, column mean is still 6 samples
- Statistical power of "deviation from mean" doesn't scale

More dimensions → more columns → harder to detect meaningful deviation with same NC

### Test
Scale NC with D: use 12 cells at D=24, 24 cells at D=48

Or change plasticity rule to operate on different statistics (global mean instead of column mean)

### Constitutional impact
✗ Scaling NC changes architecture (frozen element)
⚠ Changing plasticity rule changes frozen element

**Verdict:** This requires Stage 3+ changes (making NC adaptive or plasticity rule adaptive)

---

## Recommended Experiment Sequence

### 1. Diagnostic Run (no changes)
Instrument SeedGPU at D=12, 24, 48 to measure:
- Tanh saturation rates (% of pre-activations > 2.0)
- Alpha shift magnitude per step
- Attention entropy (how sharp is softmax?)
- Response signal magnitude (mean |phi_sig - phi_bare|)

**Goal:** Identify which hypothesis correlates with degradation

### 2. Minimal Fix (Hypothesis 1)
Scale eta with dimension: `eta_effective = eta * sqrt(D / 12)`

**Justification:**
- Smallest change
- Doesn't add frozen elements (eta still fixed, just D-aware)
- Addresses most direct issue (learning rate mismatch)

**Test:** Run full SeedGPU test with eta scaling. Does D=24 delta recover?

### 3. If Hypothesis 1 Fails, Try Hypothesis 4
Scale beta with dimension: `beta_effective = beta / sqrt(D / 12)`

**Justification:**
- Addresses saturation
- Still doesn't add frozen elements
- May require retuning at D=12 (regression risk)

### 4. If Both Fail, Accept Degradation
Argument: Constitution requires "advantage holds on novel inputs", not "advantage maintains magnitude"

Living Seed maintains positive delta at all tested dimensions → PASSES Stage 2

Scaling degradation is Stage 3+ problem (make eta, beta, tau adaptive to handle arbitrary D)

---

## Constitutional Analysis of Fixes

**Fix that PASSES constitutional test:**
- Make eta or beta FUNCTIONALLY dependent on D (D is already frozen, so no new frozen element)
- Ratio: `eta = base_eta * f(D)` where f is fixed function (sqrt, log, etc)

**Fix that FAILS constitutional test:**
- Make eta or beta INDEPENDENTLY tuned per D (adds frozen element per dimension)
- Add new hyperparameter to control scaling (frozen frame grows)

**Fix that ADVANCES to Stage 3:**
- Make eta ADAPTIVE based on system state (Stage 3: "adaptation rate adapts")
- Make beta ADAPTIVE based on computation (Stage 4: "structural constants adaptive")

**Recommendation:** Try functional dependence first (constitutional). If it fails, jump to Stage 3 (make eta adaptive) rather than adding frozen elements.

---

## Next Steps

1. Run diagnostic instrumentation on existing SeedGPU
2. Test Hypothesis 1 (eta scaling) - LOWEST RISK
3. If success: document as Stage 2 completion, proceed to Stage 3
4. If failure: test Hypothesis 4 (beta scaling)
5. If both fail: accept degradation as constitutional pass, OR jump to Stage 3 (adaptive eta)
6. Compare findings with engineer-2's Tempest results (evolvable transition functions might solve this differently)
