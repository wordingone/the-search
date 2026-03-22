# The Search

Can a system improve itself by criteria it generates?

720+ experiments across 12 architecture families testing substrates for recursive self-improvement. No solution found. The constraint map, self-modification hierarchy, and feasible region characterization are the contributions.

## Structure

- `paper/` — Research paper ([PDF](paper/paper.pdf))
- `constraints/` — Constraint map, constitution (R1-R6), experiment log
- `experiments/` — All experiment scripts (Steps 1-718)
- `substrates/` — Substrate implementations
- `viz/` — Interactive search space visualization

## Quick Start

```bash
# 1. Set up environment (one command)
bash setup.sh

# 2. Activate venv
source .venv/Scripts/activate   # Windows (Git Bash / MSYS2)
# or:
source .venv/bin/activate        # Linux / Mac

# 3. Run an experiment
PYTHONPATH=. python experiments/step0720_cifar_dynamic_r3.py

# 4. Run reference substrate validation
PYTHONPATH=. python substrates/step0674.py

# Rebuild the paper
cd paper && python build.py --pdf

# Regenerate visualization
cd viz && python viz.py
```

## Citation

```bibtex
@article{han2026search,
  title={Characterizing the Feasible Region for Self-Modifying Substrates in Interactive Environments},
  author={Han, Hyun Jun},
  year={2026}
}
```

## License

CC-BY-NC 4.0
