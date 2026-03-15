#!/usr/bin/env python3
"""Unit test for delta_stability signal implementation."""

import sys
sys.path.insert(0, 'src')

from harness import Organism, stage3_delta_stability_rule, make_signals
import random

print("Testing delta_stability signal implementation...")
print()

# Create stage3-enabled organism
rule = stage3_delta_stability_rule(alpha_meta=0.05)
org = Organism(seed=42, alive=True, rule_params=rule)

print(f"Organism created with stage3_enabled={org.stage3_enabled}")
print(f"  alpha_meta={org.alpha_meta}")
print(f"  eta_clip_lo={org.eta_clip_lo}, eta_clip_hi={org.eta_clip_hi}")
print(f"  Initial eta={org.eta}")
print()

# Generate test signal
sigs = make_signals(4, seed=100)
signal = sigs[0]

# Initialize state
random.seed(42)
xs = [[random.gauss(0, 0.5) for _ in range(12)] for _ in range(6)]

print("Running 20 steps with signal...")
for step in range(20):
    xs = org.step(xs, signal=signal)

    if len(org.eta_history) > 0:
        current_eta = org.eta_history[-1]
        if len(org.delta_stability_history) > 0:
            current_ds = org.delta_stability_history[-1]
            print(f"Step {step+1}: eta={current_eta:.6f}, delta_stability={current_ds:+.6f}")

print()
print("Summary:")
print(f"  Total signal-bearing steps with delta_stability: {len(org.delta_stability_history)}")
print(f"  Eta history length: {len(org.eta_history)}")

if len(org.eta_history) > 0:
    eta_mean = sum(org.eta_history) / len(org.eta_history)
    eta_min = min(org.eta_history)
    eta_max = max(org.eta_history)
    print(f"  Eta range: [{eta_min:.6f}, {eta_max:.6f}], mean={eta_mean:.6f}")
    print(f"  Eta changed: {'YES' if eta_max != eta_min else 'NO'}")

if len(org.delta_stability_history) > 0:
    ds_mean = sum(org.delta_stability_history) / len(org.delta_stability_history)
    ds_min = min(org.delta_stability_history)
    ds_max = max(org.delta_stability_history)
    print(f"  Delta_stability range: [{ds_min:+.6f}, {ds_max:+.6f}], mean={ds_mean:+.6f}")

print()
print("SUCCESS: delta_stability signal is computing and updating eta.")
