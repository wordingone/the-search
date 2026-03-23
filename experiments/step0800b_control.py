"""
step0800b_control.py -- Control for step800b: different substrate seeds.

step800b shows cold=327/seed (9x random baseline). L1 R3_cf PASS (p=0.013).
Must verify not a substrate_seed=0 artifact like step806v2 was.

Tests substrate_seed = 1, 2, 3. Identical methodology as step806v2_control.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

from substrates.step0800b import EpsilonActionChange800b
from r3cf_runner import run_r3cf

SUBSTRATE_SEEDS = [1, 2, 3]

print("=" * 70)
print("STEP 800b CONTROL — substrate_seed=1,2,3")
print("=" * 70)
print("Verifying cold=327/seed result is robust, not substrate_seed=0 artifact.")
print()

for ss in SUBSTRATE_SEEDS:
    print(f"\n--- substrate_seed={ss} ---")
    result = run_r3cf(EpsilonActionChange800b, f"Step800b_ctrl_seed{ss}",
                      measure_prediction=False, substrate_seed=ss)
    l1_status = 'PASS' if result['r3_cf_pass'] else ('FAIL' if result['r3_cf_pass'] is False else 'INCONCLUSIVE')
    print(f"  L1: {l1_status}  cold={result['total_cold']} ({result['total_cold']//5}/seed)  warm={result['total_warm']} ({result['total_warm']//5}/seed)")

print()
print("STEP 800b CONTROL DONE")
