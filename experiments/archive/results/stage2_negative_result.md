# Stage 2 Negative Result: Beta/Gamma Adaptation Fails

**Date:** 2026-02-23
**Investigator:** engineer-1
**Status:** HYPOTHESIS REJECTED

## Summary

Gradient-based adaptation of structural parameters (beta, gamma) does NOT converge to a unique attractor and does NOT outperform grid-optimized static values. The frozen frame floor is NOT zero for structural parameters.

## Experiment Design

**Randomized Initialization Test:**
- 5 diverse starting points for (beta, gamma)
- 3 seeds per initialization
- 800 total training steps
- Response-weighted analytical gradients (Option D)
- Fine grid search baseline (10×10)

## Results

### Eta Scaling Investigation

| eta_structural | Typical shift | Convergence | Result |
|----------------|---------------|-------------|---------|
| 0.0003 (1×)    | 0.0000-0.0011 | FROZEN      | No movement |
| 0.003 (10×)    | 0.001-0.01    | FROZEN      | Minimal movement |
| 0.03 (100×)    | 0.01-0.10     | LOCAL       | Some movement |
| 0.30 (1000×)   | 0.02-1.66     | LOCAL       | Substantial movement |

### Final Convergence (eta=0.30)

| Init (β, γ) | Final (β, γ) | Total Shift | MI |
|-------------|--------------|-------------|-----|
| (0.10, 0.10) | (0.11, 0.10) | 0.02 | 1.86 |
| (1.00, 0.50) | (1.03, 0.44) | 0.43 | 2.50 |
| (0.30, 1.50) | (0.43, 1.65) | 1.15 | 2.18 |
| (1.50, 0.10) | (1.51, 0.10) | 0.10 | 2.36 |
| (0.05, 2.00) | (0.45, 1.98) | 1.66 | 2.36 |

**Range:** β=1.40, γ=1.88 (>> 0.05 convergence threshold)

**Adaptive mean MI:** 2.25
**Best static MI (grid):** 2.45 (β=0.30, γ=0.70)
**Disadvantage:** -0.19

## Gradient Diagnostic

**Root cause of initial failures (eta < 0.03):**

1. **Response dilution**: |φ_sig - φ_bare| ≈ 0.03 (3% modulation)
2. **Averaging over NC×D=72**: Individual gradients ~0.01 → averaged ~1e-4
3. **Eta mismatch**: Alpha uses 0.0003, but structural params need 100-1000× larger

**At birth values (β=0.5, γ=0.9) with eta=0.0003:**
- Raw gradient: ~1e-4
- Effective step: ~1e-8
- Steps to change by 1.0: 6 million

## Interpretation

### Why No Global Attractor?

1. **Rugged landscape**: Response function creates multiple local optima
2. **Weak constraints**: MI maximization insufficient to uniquely determine β, γ
3. **Multiple stable regimes**:
   - Low signal (β, γ ≈ 0.1): minimal modulation
   - Mid range (β ≈ 0.3-1.0, γ ≈ 0.5-1.5): balanced
   - High gamma (γ → 2.0): strong signal coupling

### Theoretical Implications

**The Strong Thesis (frozen frame → 0) is FALSIFIED for structural parameters.**

Reasons:
- No thermodynamic ground truth for β, γ
- Multiple configurations achieve similar MI
- Static optimization beats adaptive
- Gradient descent finds local minima, not global optimum

## Conclusions

### What Failed

- ✗ Convergence to single (β₀, γ₀) attractor
- ✗ Adaptive beats static baseline
- ✗ Initialization independence
- ✗ Frozen frame reduction in Stage 2

### What Succeeded

- ✓ Gradient implementation correct
- ✓ Eta scaling diagnosed and fixed
- ✓ Rigorous falsification framework
- ✓ Negative result is publishable

## Recommendations

1. **Accept Stage 2 failure**: Document as boundary of singularity approach
2. **Investigate why**: What makes β, γ different from α?
   - α: per-channel, local optimization works
   - β, γ: global coupling, multiple equilibria
3. **Alternative objectives**: MI may be wrong target for structural params
   - Prediction error minimization?
   - Energy conservation?
   - Specific dynamical regime?
4. **Move to Stage 3**: Test Layer 1 params (eps, tau, delta)
5. **Theoretical work**: Formalize when gradient descent finds unique optimum

## Files

- Implementation: `src/living_seed_stage2_gradient.py`
- Initialization test: `src/stage2_init_test.py`
- Gradient diagnostic: `src/gradient_diagnostic.py`
- Results: `results/stage2_negative_result.md` (this file)

## Next Steps

Await team-lead decision on:
- Report negative result and move to Stage 3?
- Revisit Stage 2 with different objective function?
- Theoretical analysis of when frozen frame floor exists?
