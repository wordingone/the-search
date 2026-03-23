"""
step0813_anti_convergence.py -- Epsilon sweep + Boltzmann for step800b architecture.

Tests epsilon in [0.05, 0.10, 0.20, 0.50] and Boltzmann(tau=0.1) on LS20.
step800b baseline: epsilon=0.20, cold=237-377/seed (6-10x random).
Random baseline: 36.4/seed.

Only cold performance (no pretrain) — warm transfer was inconsistent.
Uses substrate_seed=0 (consistent with step800b original).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0813 import EpsilonActionChange813, BoltzmannActionChange813
from r3cf_runner import run_r3cf
import functools

print("=" * 70)
print("STEP 813 — ANTI-CONVERGENCE SWEEP (LS20)")
print("=" * 70)
print("Baseline: step800b cold=327/seed (epsilon=0.20), random=36.4/seed")
print()

results = {}

# Epsilon sweep
for eps in [0.05, 0.10, 0.20, 0.50]:
    # Create a subclass with fixed epsilon
    SubClass = type(
        f"EpsilonActionChange813_eps{int(eps*100)}",
        (EpsilonActionChange813,),
        {"__init__": lambda self, n_actions=4, seed=0, _eps=eps:
         EpsilonActionChange813.__init__(self, n_actions=n_actions, seed=seed, epsilon=_eps)}
    )
    label = f"eps={eps}"
    print(f"\n--- epsilon={eps} ---")
    result = run_r3cf(SubClass, f"Step813_eps{int(eps*100)}", measure_prediction=False)
    cold = result['total_cold'] // 5
    warm = result['total_warm'] // 5
    results[label] = {"cold": cold, "warm": warm}
    print(f"  cold={cold}/seed  warm={warm}/seed")

# Boltzmann
print(f"\n--- Boltzmann (tau=0.1) ---")
SubBoltz = type(
    "BoltzmannActionChange813_tau01",
    (BoltzmannActionChange813,),
    {"__init__": lambda self, n_actions=4, seed=0:
     BoltzmannActionChange813.__init__(self, n_actions=n_actions, seed=seed, tau=0.1)}
)
result = run_r3cf(SubBoltz, "Step813_boltzmann_tau01", measure_prediction=False)
cold = result['total_cold'] // 5
warm = result['total_warm'] // 5
results["boltzmann_tau0.1"] = {"cold": cold, "warm": warm}
print(f"  cold={cold}/seed  warm={warm}/seed")

print()
print("=" * 70)
print("STEP 813 SUMMARY")
print("=" * 70)
print(f"  Random baseline: 36.4/seed")
print(f"  step800b (eps=0.20): 327/seed (seed=0 reference)")
for label, r in results.items():
    print(f"  {label}: cold={r['cold']}/seed  warm={r['warm']}/seed")
print()
print("STEP 813 DONE")
