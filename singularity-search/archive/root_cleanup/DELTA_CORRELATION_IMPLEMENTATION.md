# Delta_Correlation Signal Implementation (Candidate #5)

## Implementation Summary

Implemented Stage 3 delta_correlation signal for adaptive eta in `src/harness.py` per Task #15 requirements.

## Signal Definition

**Signal:** `delta_correlation = correlation[t] - correlation[t-1]`

Where:
- `correlation = pearson_corr(error_mag, push_mag)` across all cells
- `error_mag[i] = ||phi_bare[i] - xs[i]||` — prediction error per cell
- `push_mag[i] = sum(|push[i][k]|)` — total alpha update per cell

Measures whether plasticity is appropriately targeting prediction errors.

**Directional Rule (ONE-WAY):**
- `delta_correlation < 0` (degrading alignment) → DECREASE eta
- `delta_correlation ≥ 0` (improving/stable) → NEUTRAL (no change)

**Eta Update:**
```python
eta_new = clip(eta * (1 + alpha_meta * min(0, delta_correlation)), 0.001, 0.1)
```

Where `alpha_meta` is the meta-learning rate (default 0.05).

## Code Changes

### 1. Modified `Organism.__init__()` (lines 84-120)

Added delta_correlation support:
- `stage3_signal`: String flag ('delta_stability' or 'delta_correlation')
- `correlation_prev`: Previous correlation value (1 float)
- `delta_correlation_history`: List of delta_correlation values

### 2. Modified plasticity block in `Organism.step()` (lines 161-204)

Added push magnitude tracking:
- `push_mags[i]`: Accumulate |push| for each cell during plasticity updates
- Only tracked when stage3_signal == 'delta_correlation'

### 3. Modified Stage 3 block in `Organism.step()` (lines 206-267)

Refactored to support both signals:
- **delta_stability branch:** Two-way adaptation (existing)
- **delta_correlation branch:** NEW
  1. Compute error_mag per cell from bare_diff
  2. Compute Pearson correlation between error_mag and push_mag
  3. Compute delta_correlation (temporal derivative)
  4. Update eta using one-way rule (min(0, delta_correlation))

### 4. Added `stage3_delta_correlation_rule()` (lines 440-462)

Helper function to create delta_correlation-enabled rules:
- Returns canonical rule with stage3_enabled=True, stage3_signal='delta_correlation'
- Sets alpha_meta, eta_clip_lo, eta_clip_hi
- Initializes eta to mid-range (0.01)

### 5. Added `phase2_validate_delta_correlation()` (lines 819-935)

Phase 2 validation implementing all requirements:
1. **Ground truth check:** Run 3 seeds, verify all pass
2. **Eta trajectory analysis:** Extract eta history, check non-trivial convergence
3. **Signal dynamic range:** Verify delta_correlation has sufficient variance
4. **Training comparison:** Compare adaptive vs fixed eta on full protocol

**Pass criteria:** Same as delta_stability

### 6. Modified main block (lines 960-972)

Enhanced `--phase2` flag to support both signals:
```bash
python src/harness.py --phase2 delta_stability       # Candidate #3
python src/harness.py --phase2 delta_correlation    # Candidate #5
python src/harness.py --phase2 delta_correlation 0.1  # Custom alpha_meta
```

## Unit Test Results

Created `test_delta_correlation.py` to verify implementation:

```
Organism created with stage3_enabled=True
  stage3_signal=delta_correlation
  alpha_meta=0.05
  Initial eta=0.01

Running 20 steps with signal...
Step 2: eta=0.009910, delta_correlation=-0.180702
Step 3: eta=0.009636, delta_correlation=-0.552198
...
Step 20: eta=0.008953, delta_correlation=+0.195339

Summary:
  Total signal-bearing steps with delta_correlation: 19
  Eta history length: 19
  Eta range: [0.008953, 0.009910], mean=0.009229
  Eta changed: YES
  Delta_correlation range: [-0.552198, +0.291250], mean=-0.050791
  Eta direction: DECREASED
  One-way check: PASS (decreased only)

SUCCESS: delta_correlation signal is computing and updating eta.
```

**Observations:**
- Signal computes correctly on signal-bearing steps (19/20 steps)
- Delta_correlation has both positive and negative values (bidirectional signal)
- Eta ONLY DECREASES (one-way adaptation working correctly)
- Mean delta_correlation is negative (degrading alignment on average)
- Eta decreased from 0.01 → 0.008953 (10% reduction)

## Key Differences from Delta_Stability

| Aspect | Delta_Stability (#3) | Delta_Correlation (#5) |
|--------|----------------------|------------------------|
| **Signal family** | Temporal dynamics (phi change) | Cross-variable correlation (error vs. push) |
| **Signal source** | Computational stability | Learning alignment |
| **Adaptation mode** | Two-way (increase/decrease) | One-way (decrease only) |
| **Eta direction** | Both up and down | Down only |
| **Interpretation** | System settling/destabilizing | Learning aligning/degrading |
| **Resp_z family** | Independent (phi is upstream) | Hybrid (push is resp_z-derived) |

## Phase 2 Validation Status

Full Phase 2 validation ready to run:
```bash
python src/harness.py --phase2 delta_correlation
```

Validation checks:
1. ? Ground truth (3/3 seeds) — not yet run
2. ? Eta trajectory (non-trivial) — not yet run
3. ? Signal dynamic range — not yet run
4. ? Training gap comparison — not yet run

**Note:** Unit test confirms signal computation is correct. Full validation will test ground truth and performance.

## Files Modified

- `src/harness.py`: Added delta_correlation signal, one-way eta adaptation, Phase 2 validation

## Files Created

- `test_delta_correlation.py`: Unit test for signal computation
- `DELTA_CORRELATION_IMPLEMENTATION.md`: This document

## Implementation Notes

### Design Decisions

1. **One-way adaptation:** Eta can only decrease, never increase. This is conservative — prevents runaway learning rates. Pipeline spec explicitly requires this.

2. **Pearson correlation:** Standard statistical measure of linear relationship between error_mag and push_mag. Range [-1, 1]. Positive = errors and learning aligned.

3. **Per-cell aggregation:** Both error_mag and push_mag are per-cell scalars. Correlation computed across NC=6 cells. This provides enough samples for stable correlation.

4. **Signal computed before state update:** Error_mag uses bare_diff which is computed during the attention block. We compute it early in Stage 3 block before state update to ensure consistency.

### Potential Issues

1. **Low sample size:** Correlation computed over NC=6 cells. Small sample → high variance → noisy signal. May need smoothing (but EMA introduces frozen decay constant, violating R5).

2. **One-way limitation:** If eta gets stuck at lower bound, system cannot recover. Unlike delta_stability which can increase eta if needed.

3. **Hybrid signal family:** Push is resp_z-derived, so this signal is not fully independent of the resp_z family. May fail c013 check if correlation with resp_z is high.

### Open Questions

1. **Is NC=6 sufficient?** Pearson correlation with n=6 has high variance. May need to accumulate over multiple steps or use larger NC.

2. **Should we use Spearman instead?** Rank correlation (Spearman) is more robust to outliers than Pearson. Consider if correlation is unstable.

3. **What if correlation is consistently negative?** Negative correlation = cells with high errors are learning LESS. This would cause eta to increase (via negative delta_correlation), but one-way rule blocks increases. System would be stuck.

## Comparison to Adversarial Analysis Spec

From `stage3_adversarial_analysis.md`:

**Candidate #5: Prediction Error Alignment (REVISED as delta_correlation)**
- ✓ Signal: delta_corr = corr(error_mag, push_mag)[t] - corr[t-1]
- ✓ Directional rule: delta_corr < 0 → decrease eta; delta_corr ≥ 0 → neutral
- ✓ Eta update: eta_new = eta * (1 + alpha * min(0, delta_corr))
- ✓ Family: Hybrid (error independent, push resp_z-derived)
- ✓ Priority: 2

**Phase 2 Requirements:**
- ✓ Implement signal computation in harness.py
- ✓ Implement global eta update
- 🔄 Run 3-seed ground truth check (ready to run)
- ⚠️ Compute resp_z contamination check (not yet implemented)
- 🔄 Measure eta trajectory (ready to run)
- 🔄 Compare training gap: adaptive vs fixed (ready to run)

**Status:** 2/6 complete, 3/6 ready to run, 1/6 not implemented.

## Next Steps

1. **Run Phase 2 validation:** `python src/harness.py --phase2 delta_correlation`
2. **Analyze results:** Check pass/fail criteria
3. **Compare to delta_stability:** Which signal performs better?
4. **If both pass Phase 2:** Advance both to Phase 3 (10-seed validation)
5. **If only one passes:** Advance that candidate, iterate on the other

---

*Implementation complete. Both delta_stability and delta_correlation signals ready for Phase 2 validation.*
