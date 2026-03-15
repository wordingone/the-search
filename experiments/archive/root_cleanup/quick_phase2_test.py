"""Quick Phase 2 test - single alpha_meta value for delta_stability"""
import sys
sys.path.insert(0, 'src')
from harness import phase2_validate_delta_stability

print("Starting delta_stability Phase 2 validation (alpha_meta=0.05)...")
print("This will take ~3-5 minutes with 2× exposure protocol.\n")

result = phase2_validate_delta_stability(alpha_meta=0.05, verbose=True)

print("\n" + "="*80)
print("FINAL RESULT:")
print("="*80)
print(f"Phase 2 Pass: {result['phase2_pass']}")
print(f"Contamination corr: {result['contamination_corr']:+.4f} ({'PASS' if result['contamination_pass'] else 'FAIL'})")
print(f"Improvement: {result['improvement']:+.4f}")
print("="*80)

# Write to file for retrieval
with open('phase2_delta_stability_result.txt', 'w') as f:
    f.write(f"Phase 2 Pass: {result['phase2_pass']}\n")
    f.write(f"Contamination corr: {result['contamination_corr']:+.4f}\n")
    f.write(f"Contamination pass: {result['contamination_pass']}\n")
    f.write(f"Improvement: {result['improvement']:+.4f}\n")

print("\nResults saved to phase2_delta_stability_result.txt")
