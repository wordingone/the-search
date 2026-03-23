# The Search

Can a system improve itself by criteria it generates?

777+ experiments across 12 architecture families testing substrates for recursive self-improvement on published benchmarks (Split-CIFAR-100, Atari 100K) and interactive games (ARC-AGI-3). No R3-compliant substrate found. Two bans — codebook (Step 416) and graph (Step 777, permanent) — remove both known-working mechanisms, forcing the search into genuinely new territory. The contributions: a formal framework (R1-R6), a constraint map from systematic falsification, an R3 counterfactual metric, and a state decomposition theorem (Proposition 20) characterizing when accumulated state helps vs hurts.

## Results (honest)

| Benchmark | Substrate | Result | Compare |
|-----------|-----------|--------|---------|
| Split-CIFAR-100 (20 tasks, no labels) | 674 | 20.2% avg accuracy | Chance=20%. DER++=29.6% (with labels) |
| Split-CIFAR-100 BWT | 674 | +5.6% | Zero forgetting by construction |
| LS20 navigation (20 seeds, 10K steps) | 674+running-mean | 20/20 L1 | Argmin over visit counts |
| FT09 navigation | 674+running-mean | 20/20 L1 | 7-action space |
| Cross-domain transfer (LS20→CIFAR) | 674 | 0% improvement | Zero transfer in both directions |
| R3 (self-modification) | 674 | FAIL (8 U elements) | 0/777 substrates pass R3 |
| R3 counterfactual | 674 | FAIL (cold > warm, p<0.0001) | Pretraining hurts new environments |
| Atari 100K (no reward) | 674 | 6/26 above random | RoadRunner 11x, most games at/below random |

**Key findings:**
- Navigation is solved by a trivial mechanism (graph + argmin). Not intelligence. **Now banned.**
- Classification = chance without labels. The R1 floor.
- Zero cross-domain transfer. The substrate accumulates location, not knowledge.
- R3 counterfactual FAIL: pretraining hurts (cold > warm, p<0.0001). Visit counts bias exploration.
- Proposition 20: location-dependent state (visit counts) transfers negatively; dynamics-dependent state (forward models) could transfer positively. Post-ban substrates must store dynamics, not location.
- Argmin was load-bearing (14/20 vs random 6/20, n=20). **Now banned.** Post-ban action selection is the open problem.

## Structure

```
paper/           Research paper (PDF + source)
constraints/     Constraint map (CONSTRAINTS.md), constitution (R1-R6), experiment log
experiments/     777+ experiment scripts (chronological, Steps 1-888)
substrates/      Active substrate code:
  base.py          BaseSubstrate interface (5 methods)
  step0674.py      Reference substrate (transition-triggered dual-hash)
  plain_lsh.py     PlainLSH baseline (no refinement)
  chain.py         Chain benchmark runner (pluggable — any gym env)
  judge.py         ConstitutionalJudge (R1-R6 audit + R3 counterfactual)
  calibration_agents.py   R3 calibration baselines
  archive/         Killed families (12 architectures, preserved)
viz/             Search space visualization
```

## Quick Start

```bash
# Setup
bash setup.sh
source .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Run Split-CIFAR-100 benchmark (no API key needed)
PYTHONPATH=. python -c "
from substrates.chain import SplitCIFAR100Wrapper, ChainRunner
from substrates.step0674 import Substrate674
wrapper = SplitCIFAR100Wrapper(500)
result = wrapper.run_seed(Substrate674(), seed=0)
print(f'CIFAR acc={result[\"avg_accuracy\"]:.1%}, BWT={result[\"backward_transfer\"]:+.1%}')
"

# Run R1-R6 audit
PYTHONPATH=. python -c "
from substrates.judge import ConstitutionalJudge
from substrates.step0674 import Substrate674
results = ConstitutionalJudge().audit(Substrate674)
for rule in ['R1','R2','R3','R4','R5','R6']:
    r = results.get(rule, {})
    print(f'{rule}: {\"PASS\" if r.get(\"pass\") else \"FAIL\"} — {r.get(\"detail\",\"\")}')
"

# Run reference substrate on ARC game (requires ARC-AGI-3 API key)
PYTHONPATH=. python substrates/step0674.py
```

## The Framework

Six rules for recursive self-improvement (R1-R6):

| Rule | Requirement | What it means |
|------|------------|---------------|
| R1 | No external objectives | No reward, no labels, no loss function |
| R2 | Adaptation from computation | Parameters ⊆ state. No optimizer. |
| R3 | Minimal frozen frame | Every design choice either self-modified or irreducible |
| R4 | Modifications tested against prior state | Changes must not degrade |
| R5 | One fixed ground truth | Only environmental signals (death, level transitions) |
| R6 | No deletable parts | Every component behaviorally load-bearing |

**R3 is the wall.** 0/777 substrates pass. The constraint map characterizes why. Two bans (codebook + graph) remove both tested mechanisms that satisfy other rules, forcing the search toward genuinely self-modifying architectures.

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
