# Evaluation Protocol Update — 2× Exposure

## Implementation Summary

Updated `src/harness.py` to implement the 2× exposure evaluation protocol required by constraint c015.

### Changes Made

1. **Updated defaults in `measure_gap()` (line 280)**
   - `n_perm`: 4 → 8 (2× permutations)
   - `n_trials`: 3 → 6 (2× trials per permutation)

2. **Updated defaults in `run_comparison()` (lines 330-337)**
   - Added `n_perm=8` and `n_trials=6` as explicit parameters
   - Threaded these parameters through all `measure_gap()` calls

3. **Added quick mode for fast iteration**
   - New function `quick_eval()` (line 520) uses old defaults (n_perm=4, n_trials=3)
   - Command-line flag `--quick` enables quick mode
   - Usage: `python src/harness.py --quick`

### Backward Compatibility

All changes are backward compatible:
- Parameters can be overridden via function arguments
- Quick mode available for fast iteration during development
- No changes to core computation logic

### Verification

Created `smoke_test.py` to verify defaults:
```
measure_gap defaults:
  n_perm = 8 (expected: 8)
  n_trials = 6 (expected: 6)

run_comparison defaults:
  n_perm = 8 (expected: 8)
  n_trials = 6 (expected: 6)

[PASS] measure_gap defaults CORRECT
[PASS] run_comparison defaults CORRECT

SUCCESS: All defaults verified as n_perm=8, n_trials=6
```

### Expected Impact

From constraint c015 analysis:
- **OLD protocol**: n_perm=4, n_trials=3 → CV≈29% (underpowered)
- **NEW protocol**: n_perm=8, n_trials=6 → CV≈20% (adequate power)

This ~30% reduction in coefficient of variation should provide sufficient statistical power to detect Stage 3 signal effects.

### Runtime Impact

Expected runtime increase:
- Training signals: 4K×3T = 12 measurements → 8K×6T = 48 measurements (4× longer)
- Novel signals: Same 4× factor
- Total wall time: ~4× increase per evaluation

Quick mode (`--quick`) provides the old performance for rapid iteration.

### Usage Examples

```bash
# Full evaluation with new protocol (default)
python src/harness.py

# Quick evaluation for development
python src/harness.py --quick

# Programmatic usage with custom parameters
from harness import run_comparison, canonical_rule
result = run_comparison(canonical_rule(), n_perm=16, n_trials=12)  # Even higher exposure
```

### Files Modified

- `src/harness.py`: Updated defaults, added quick mode, threaded parameters

### Files Created

- `smoke_test.py`: Verification script for defaults
- `test_harness_protocol.py`: Full validation test (optional)
- `PROTOCOL_UPDATE.md`: This document

### Next Steps

QA team (Task #6) should:
1. Run full validation with new defaults on canonical baseline
2. Verify CV reduction to ~20% as predicted
3. Confirm ground truth still passes
4. Validate quick mode produces consistent results (just lower precision)
