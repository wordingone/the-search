"""
step0800b_ls20.py -- EpsilonActionChange800b on LS20 (R3_cf).

80% argmax per-action EMA change + 20% random. Prevents argmax collapse.
Compare: step800 (collapse, 0/seed), step806v2 (seed-0 artifact), random (36.4/seed).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0800b import EpsilonActionChange800b
from r3cf_runner import run_r3cf

result = run_r3cf(EpsilonActionChange800b, "Step800b_EpsilonActionChange_LS20",
                  measure_prediction=False)

print()
print("STEP 800b LS20 DONE")
print(f"R3_cf (L1): {'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')}")
print(f"cold={result['total_cold']}  warm={result['total_warm']}")
print(f"Random baseline: 36.4/seed. 800 (pure argmax): 0/seed (collapse).")
