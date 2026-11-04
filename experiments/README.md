# Experiments

## Reproducing experiments

### Current era (Steps 1305+, PRISM infrastructure)

```bash
# Setup
bash experiments/setup.sh

# Run any step
PYTHONUTF8=1 python experiments/steps/step1392_dendritic_spatial_stdp.py
```

Step scripts in the current era are self-contained: each imports `prism_masked.py` from its own directory, defines the substrate, runs PRISM evaluation, and writes results to `experiments/results/results_all/results_NNNN/`.

### Earlier eras (Steps <1305)

Earlier experiments used different infrastructure that evolved across phases:

| Step range | Era | Infrastructure | How to run |
|-----------|-----|---------------|-----------|
| 1-416 | Phase 1 (Codebook) | Per-substrate scripts | `python experiments/steps/stepNNNN.py` (most are self-contained) |
| 417-777 | Phase 2 (Graph/LSH) | Per-substrate scripts | Same as above |
| 778-1006 | Post-ban (800b/916) | `run.py` harness | `python experiments/run.py --step NNNN --substrate experiments/steps/stepNNNN.py` |
| 1007-1250 | Debate/Composition | PRISM runner | `python experiments/run.py --step NNNN` |
| 1251-1304 | Composition era | Composition scripts | `python experiments/steps/stepNNNN_*.py` |
| 1305+ | Self-generated criteria | `prism_masked.py` | See "Current era" above |

**Note:** Steps <1305 that import `substrates.*` depend on a package not included in this repository (`.gitignore:56` excludes `substrates/`; 81 `.py` files import it, e.g. `experiments/run.py:29`). They are kept as a record and will not run as-is.

Results for all eras are in `experiments/results/results_all/`. Format varies by era:
- Phase 2: `experiments/results/results_all/phase2_589_1000/` (step directories with raw outputs)
- Debate: `experiments/results/results_all/debate_1104_1250/` (run logs)
- PRISM runs: `experiments/results/results_all/prism_914_1044/` (JSON evaluations)
- Composition+: `experiments/results/results_all/results_NNNN/` (JSONL per game per draw)

## File inventory

- `steps/` — 192 experiment scripts (all eras)
- `experiments/results/results_all/` — 147 result sets (all eras)
- `environments/` — Game environment wrappers (click_target.py, kb_sequence.py, mbpp_game.py)
- `run.py` — Universal PRISM harness (Steps 778-1250)
- `setup.sh` — Environment setup script
