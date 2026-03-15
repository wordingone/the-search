"""
Standalone script to run Phase 2 validation for delta_stability.
Saves output to file for review.
"""

import sys
sys.path.insert(0, 'src')

from harness import phase2_validate_delta_stability

# Run validation with alpha_meta = 0.05
print("Starting Phase 2 validation for delta_stability (alpha_meta=0.05)...")
result = phase2_validate_delta_stability(alpha_meta=0.05, verbose=True)

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)
print(f"Phase 2 Pass: {result['phase2_pass']}")
print(f"Contamination corr: {result['contamination_corr']:+.4f}")
print(f"Improvement: {result['improvement']:+.4f}")
