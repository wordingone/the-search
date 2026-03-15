"""
Run full Phase 2 validation sweep for both Stage 3 signals.
Tests alpha_meta = [0.01, 0.05, 0.1] for each signal.
"""

import sys
sys.path.insert(0, 'src')

from harness import phase2_validate_delta_stability, phase2_validate_delta_correlation

alpha_meta_values = [0.01, 0.05, 0.1]

print("="*80)
print("  PHASE 2 VALIDATION SWEEP - DELTA_STABILITY")
print("="*80)

ds_results = []
for alpha_meta in alpha_meta_values:
    print(f"\n{'='*80}")
    print(f"  Testing alpha_meta = {alpha_meta}")
    print(f"{'='*80}\n")

    result = phase2_validate_delta_stability(alpha_meta=alpha_meta, verbose=True)
    ds_results.append(result)

print(f"\n{'='*80}")
print(f"  DELTA_STABILITY SUMMARY")
print(f"{'='*80}")
for i, alpha_meta in enumerate(alpha_meta_values):
    r = ds_results[i]
    status = 'PASS' if r['phase2_pass'] else 'FAIL'
    print(f"  alpha_meta={alpha_meta:5.2f}: {status:4s} | "
          f"contam={r['contamination_corr']:+.3f} | "
          f"improvement={r['improvement']:+.4f}")

print("\n" + "="*80)
print("  PHASE 2 VALIDATION SWEEP - DELTA_CORRELATION")
print("="*80)

dc_results = []
for alpha_meta in alpha_meta_values:
    print(f"\n{'='*80}")
    print(f"  Testing alpha_meta = {alpha_meta}")
    print(f"{'='*80}\n")

    result = phase2_validate_delta_correlation(alpha_meta=alpha_meta, verbose=True)
    dc_results.append(result)

print(f"\n{'='*80}")
print(f"  DELTA_CORRELATION SUMMARY")
print(f"{'='*80}")
for i, alpha_meta in enumerate(alpha_meta_values):
    r = dc_results[i]
    status = 'PASS' if r['phase2_pass'] else 'FAIL'
    print(f"  alpha_meta={alpha_meta:5.2f}: {status:4s} | "
          f"contam={r['contamination_corr']:+.3f} | "
          f"improvement={r['improvement']:+.4f}")

print("\n" + "="*80)
print("  FINAL SUMMARY - BOTH SIGNALS")
print("="*80)
print("\nDELTA_STABILITY:")
for i, alpha_meta in enumerate(alpha_meta_values):
    r = ds_results[i]
    status = '✓ PASS' if r['phase2_pass'] else '✗ FAIL'
    print(f"  {status} | alpha_meta={alpha_meta:.2f} | contam_corr={r['contamination_corr']:+.4f}")

print("\nDELTA_CORRELATION:")
for i, alpha_meta in enumerate(alpha_meta_values):
    r = dc_results[i]
    status = '✓ PASS' if r['phase2_pass'] else '✗ FAIL'
    print(f"  {status} | alpha_meta={alpha_meta:.2f} | contam_corr={r['contamination_corr']:+.4f}")

print("\n" + "="*80)
