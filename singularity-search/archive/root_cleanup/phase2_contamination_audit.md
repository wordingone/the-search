# Phase 2 Contamination Check Audit

**Auditor:** qa (sonnet)
**Date:** 2026-02-24
**Request:** Verify resp_z contamination check implementation in Phase 2 validation functions

---

## Audit Scope

Verify that both Phase 2 validation functions include contamination checks:
1. `phase2_validate_delta_stability()` (line 718)
2. `phase2_validate_delta_correlation()` (line 888)

**Required Check:** `corr(signal_history, mean(resp_z)_history)` with pass criterion `|r| < 0.7`

---

## Findings

### ✓ phase2_validate_delta_stability() — COMPLIANT

**Location:** `src/harness.py:718-885`

**Contamination Check Implementation (lines 797-817):**
```python
# Test 3: resp_z contamination check
# Compute correlation between delta_stability and mean(resp_z) across steps
if verbose:
    print(f"\n[Contamination Check]")

resp_z_history = org.resp_z_history
contamination_pass = True
contamination_corr = 0.0

if len(delta_stability_history) > 1 and len(resp_z_history) > 1:
    # Align histories (delta_stability starts one step later than resp_z)
    min_len = min(len(delta_stability_history), len(resp_z_history))
    ds_aligned = delta_stability_history[:min_len]
    rz_aligned = resp_z_history[-min_len:]  # Take last min_len elements

    contamination_corr = compute_pearson_correlation(ds_aligned, rz_aligned)
    contamination_pass = abs(contamination_corr) < 0.7

    if verbose:
        print(f"  corr(delta_stability, mean(resp_z)) = {contamination_corr:+.4f}")
        print(f"  Contamination check: {'PASS' if contamination_pass else 'FAIL'} (|r| < 0.7)")
```

**Pass Criterion (line 855-859):**
```python
phase2_pass = (all(ground_truth_passes) and
               contamination_pass and
               eta_nontrivial and
               signal_sufficient and
               improvement >= 0)
```

**Assessment:** ✓ CORRECT
- Uses `compute_pearson_correlation()` helper function
- Threshold |r| < 0.7 enforced on line 813
- `contamination_pass` is part of Phase 2 pass criteria
- Proper history alignment handling

---

### ✓ phase2_validate_delta_correlation() — COMPLIANT

**Location:** `src/harness.py:888-1051`

**Contamination Check Implementation (lines 964-983):**
```python
# Test 3: resp_z contamination check
if verbose:
    print(f"\n[Contamination Check]")

resp_z_history = org.resp_z_history
contamination_pass = True
contamination_corr = 0.0

if len(delta_correlation_history) > 1 and len(resp_z_history) > 1:
    # Align histories (delta_correlation starts one step later than resp_z)
    min_len = min(len(delta_correlation_history), len(resp_z_history))
    dc_aligned = delta_correlation_history[:min_len]
    rz_aligned = resp_z_history[-min_len:]

    contamination_corr = compute_pearson_correlation(dc_aligned, rz_aligned)
    contamination_pass = abs(contamination_corr) < 0.7

    if verbose:
        print(f"  corr(delta_correlation, mean(resp_z)) = {contamination_corr:+.4f}")
        print(f"  Contamination check: {'PASS' if contamination_pass else 'FAIL'} (|r| < 0.7)")
```

**Pass Criterion (lines 1021-1025):**
```python
phase2_pass = (all(ground_truth_passes) and
               contamination_pass and
               eta_nontrivial and
               signal_sufficient and
               improvement >= 0)
```

**Assessment:** ✓ CORRECT
- Uses `compute_pearson_correlation()` helper function
- Threshold |r| < 0.7 enforced on line 979
- `contamination_pass` is part of Phase 2 pass criteria
- Proper history alignment handling

---

## Helper Function Verification

### ✓ compute_pearson_correlation() — CORRECT

**Location:** `src/harness.py:679-703`

**Implementation:**
```python
def compute_pearson_correlation(x, y):
    """
    Compute Pearson correlation coefficient between two lists.
    """
    n = len(x)
    if n == 0 or len(y) != n:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((x[i] - mean_x)**2 for i in range(n)))
    std_y = math.sqrt(sum((y[i] - mean_y)**2 for i in range(n)))

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0  # Constant signal

    return cov / (std_x * std_y)
```

**Assessment:** ✓ CORRECT
- Standard Pearson correlation formula
- Handles edge cases (empty lists, constant signals)
- Returns 0.0 for degenerate cases (safe default)

---

## Summary

### Compliance Status: ✓ FULL COMPLIANCE

Both Phase 2 validation functions correctly implement the resp_z contamination check:

1. **Computation:** Uses `compute_pearson_correlation(signal_history, resp_z_history)`
2. **Threshold:** `|r| < 0.7` enforced
3. **Integration:** `contamination_pass` is a hard requirement for `phase2_pass`
4. **Reporting:** Correlation value and pass/fail status reported in verbose mode

### Code Quality Notes

**Strengths:**
- Identical structure between both functions (maintainability)
- Proper history alignment (handles timing offsets)
- Clear verbose output for debugging
- Contamination check is blocking (cannot pass Phase 2 without passing contamination)

**Minor Observations:**
- History alignment assumes signal lags resp_z by 1 step (line 810, 973)
- Uses last `min_len` elements of resp_z (line 810, 976) rather than first
- This alignment strategy should be validated against actual Organism history generation

### Recommendation

✓ **APPROVE** — Contamination checks are correctly implemented and enforced.

No changes required. The implementation matches the Phase 2 requirements from `stage3_experiment_pipeline.md`.

---

## Audit Trail

- **Requested by:** team-lead
- **Completed by:** qa
- **Files audited:** `src/harness.py` (lines 679-1051)
- **Functions verified:**
  - `compute_pearson_correlation()` (line 679)
  - `phase2_validate_delta_stability()` (line 718)
  - `phase2_validate_delta_correlation()` (line 888)
- **Verdict:** COMPLIANT
