import sys
sys.path.insert(0, 'src')
from harness import phase2_validate_delta_stability

print("Starting Phase 2 validation...")
print("This may take several minutes with 2x exposure protocol.\n")
result = phase2_validate_delta_stability(alpha_meta=0.05, verbose=True)
print("\n=== FINAL RESULT ===")
print(f"Phase 2 Pass: {result['phase2_pass']}")
print(f"Contamination: {result['contamination_corr']:+.4f}")
