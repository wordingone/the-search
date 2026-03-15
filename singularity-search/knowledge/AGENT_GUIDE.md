# Agent Quick Reference

## For AI Agents Reading This

You are an AI agent starting a session on the_singularity_search project. This guide tells you what to do.

## Step 1: Read state.md (ONCE)

```python
with open('.knowledge/state.md') as f:
    state = f.read()
```

That's it. You now know:
- What doesn't work (19 constraints)
- Why it doesn't work (root causes)
- Where the project is (Session 8, Stage 2→3 transition)
- What decisions are still in effect

## Step 2: Before Running Any Experiment

**MANDATORY: Run the Experiment Gate before any compute-intensive work.**

```python
from experiment_gate import ExperimentGate

gate = ExperimentGate("your_experiment_name")

# Phase 1: Theory (seconds, free)
p1 = gate.phase1_theory(
    approach_tags=["stage3", "eta"],       # tags for constraint matching
    signal_autocorr=0.45,                   # lag-1 autocorrelation of driving signal
    frozen_added=1,                         # new frozen elements introduced
    frozen_removed=2,                       # frozen elements eliminated
    principle_ii=True,                      # is signal self-generated?
    signal_description="windowed delta_rz"  # human-readable
)
print(p1.summary())
if not gate.phase1_passed:
    # STOP. Fix theory issues before spending compute.
    exit()

# Phase 2: Validate (minutes, one candidate, 10 seeds)
from experiment_gate import make_eval_fn
eval_fn = make_eval_fn()
p2 = gate.phase2_validate(your_rule_config, eval_fn)
print(p2.summary())
if not gate.phase2_passed:
    # STOP. Single candidate didn't show genuine signal.
    exit()

# Phase 3: Search (hours, only if Phase 2 passed)
p3 = gate.phase3_approve()
print(p3.summary())
# Now you may run evolutionary/grid search with 10+ seed eval
```

**Why this matters:** Sessions 7-8 wasted ~45% compute running Phase 3 (expensive search) before Phase 1 (theory) would have killed the candidates. Inverting the order saves most of that waste.

## Step 3: When You Complete Work

If you ran an experiment, made a discovery, or hit a constraint:

```bash
# Create an entry (interactive)
python .knowledge/ingest.py

# Create from your experiment output
python .knowledge/ingest.py --from results/experiment_42.json

# After adding entries, recompile
python .knowledge/compile.py
```

## Entry Template for Experiments

```json
{
  "type": "experiment",
  "title": "One-line description",
  "tags": ["stage3", "relevant-tag"],
  "status": "failed",
  "content": {
    "hypothesis": "What you expected",
    "method": "What you did",
    "result": "What happened",
    "metrics": {"key": "value"},
    "root_cause": "Why it failed (if failed)"
  },
  "constraints_implied": [
    "Never try X because Y"
  ],
  "session": 9
}
```

## Reading Constraints

```python
import json
with open('.knowledge/constraints.json') as f:
    constraints = json.load(f)

# Check if your approach violates a constraint
for c in constraints:
    if c['active'] and 'self-referential' in c['tags']:
        print(f"Constraint: {c['rule']}")
```

## When NOT to Use This

Don't use this for:
- Transient debugging info (use logs)
- Raw data (use results/)
- Code (use src/)
- Exploratory notes (use your context)

Use this for:
- Experiments that completed (success or failure)
- Discoveries that change understanding
- Decisions that constrain future work
- Architectural changes

## Evaluation Protocol (as of Session 8)

**Standard protocol:** n_perm=8, n_trials=6 (2x exposure), 10 seeds minimum.
- Seeds: [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
- K values: [4, 6, 8, 10] (do NOT use K=3 or K=12)
- CV target: ~20% (down from 29% with old protocol)
- **Never use 3-seed evaluation for final validation** (c018)

## Key Constraints (as of Session 8)

19 active constraints. Critical ones:
1. **[c001-c007]** Beta/gamma are frozen — no local proxy, no per-cell decomposition
2. **[c008-c009]** Self-referential eta causes bang-bang oscillation
3. **[c011]** Per-cell eta adaptation doesn't improve performance
4. **[c012-c013]** resp_z derivative tower collapses at order 1 — need new signal family
5. **[c014-c017]** Evaluation protocol: use 2x exposure, 10+ seeds, K=[4,6,8,10]
6. **[c018-c019]** Evolutionary search with 3-seed eval selects for noise

Read state.md for full list and context.

## Current State (as of Session 8)

- **Stage:** Stage 2→3 transition (blocked)
- **Blocker:** resp_z signal family exhausted (drives exactly 1 stage)
- **Frozen frame:** 7 elements (eta, 3 multipliers, threshold, 2 clip bounds)
- **Next:** Need new signal family for Stage 3 (candidates: windowed delta_rz, inter-cell correlations, frequency-domain features, information geometry)

## System Health Check

```bash
python .knowledge/test_system.py
```

Should show "SUCCESS: All 9 tests passed"

## Questions?

Read README.md for full documentation.
