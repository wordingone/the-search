# FoldCore

A **prototype-vector codebook** for continual learning. Maintains unit vectors on the hypersphere, classifies via nearest-prototype or weighted k-NN, and learns in a single pass with no backpropagation.

The core data structure is a growing list of unit vectors in R^d. The core operations are: spawn a new prototype (when input is novel), update the nearest prototype (additive attraction), merge redundant prototypes (fuse when cosine > threshold), and classify (vote across nearest prototypes). All benchmarks use frozen feature extractors — the system operates on pre-extracted embeddings, not raw pixels.

**This is active research with known structural problems. We are looking for aggressive, constructive review.** Read the source code — there are four files in `src/`, all under 400 lines. If you can identify what this system actually is (or isn't), where the math breaks down, or whether any of this is genuinely new — open an issue. See [Known Issues](#known-issues) and [Open Questions](#open-questions).

## What It Does

FoldCore maintains a growing set of prototype vectors on the unit hypersphere. When input arrives:

- **Spawn**: if no existing prototype is similar enough (cosine < threshold), a new prototype is created
- **Update**: the nearest prototype moves toward the input
- **Merge**: if a new prototype is too similar to an existing one, they fuse
- **Classify**: label of the nearest prototype (or weighted vote across k-nearest)

Every input modifies the system. There is no read-only inference mode.

## Architecture

Two implementations:

### `src/foldcore_manytofew.py` — Canonical kernel (CPU, pure Python)
- Codebook layer: unit vectors in R^d, spawn/update/merge dynamics
- Matrix layer: fixed cells with eigenform dynamics (RK), coupled. Handles generation.
- Many-to-few routing: codebook vectors assign to matrix cells at spawn time
- Dependencies: `src/rk.py` (matrix utilities)

### `src/atomic_fold.py` — Hopfield-equivalent kernel (GPU, PyTorch) [DEPRECATED]
- Single codebook with per-prototype confidence weights (kappa)
- Softmax attention for both training updates and classification
- Energy-gated spawning via Hopfield energy
- **This is mathematically identical to a modern Hopfield network** (Ramsauer et al. 2020). The logsumexp energy, softmax attention, and attractor dynamics are Hopfield's math. Kept for reference, not active development.

### `src/eigenfold.py` — Matrix codebook with eigenform dynamics (CPU, pure Python) [ACTIVE]
- Codebook elements are k×k matrices (not vectors), each seeking eigenform Φ(M) = tanh(αM + βM²/k)
- Classification by **perturbation stability**: input is cross-applied with each element, most stable element (smallest perturbation) wins
- Matrix interactions are **noncommutative** (M_i·M_j ≠ M_j·M_i) — this breaks the symmetry that makes vector-based systems equivalent to Hopfield
- Fold lifecycle: spawn when no element is stable, update winner via cross-application, eigenform recovery after update
- Dependencies: `src/rk.py` (matrix utilities)

## Benchmark Results

All results use frozen feature extractors (no end-to-end training).

### Permuted-MNIST (10 sequential tasks, d=384)

| Method | Avg Accuracy | Forgetting |
|--------|-------------|------------|
| FoldCore (attractive-only, 1-NN) | 56.7% | 0.0pp |
| FoldCore (full gradient, 1-NN) | 84.1% | 11.4pp |
| Fine-tune baseline | ~52.5% | ~47pp |
| EWC | ~95.3% | ~2pp |

### Split-CIFAR-100 (20 tasks, d=512, frozen ResNet-18)

| Method | Avg Accuracy | Forgetting |
|--------|-------------|------------|
| FoldCore (full gradient, k=10 weighted) | 36.6% | 12.9pp |
| FoldCore (full gradient, 1-NN) | 33.5% | 12.6pp |
| FoldCore (attractive-only, 1-NN) | 32.3% | 12.5pp |
| EWC | ~33% | ~16pp |
| DER++ | ~51% | ~8pp |

### Notes on results
- The attractive-only update rule produces structural zero forgetting (0.0pp) because prototypes are never overwritten. The gradient update breaks this by repelling wrong-class prototypes.
- Accuracy gaps vs EWC/DER++ are due to nearest-prototype readout vs learned decision boundaries, not memory failure.
- Published baselines use 60K samples/task (MNIST) and full training pipelines. FoldCore uses 6K/task with random projection.

## Requirements

- Python 3.8+
- `src/foldcore_manytofew.py`: no dependencies beyond stdlib
- `src/atomic_fold.py`: PyTorch with CUDA
- Tests: pytest, numpy

## Running

```bash
# Run tests (skip slow benchmarks)
pytest tests/test_manytofew.py -m "not slow"

# Run all tests including benchmarks
pytest tests/test_manytofew.py
```

## License

CC BY-NC 4.0 — free for non-commercial research and educational use. See [LICENSE](LICENSE).

## Known Issues

### Vector codebook (`foldcore_manytofew.py`, `atomic_fold.py`)
- **Matrix layer is dead for classification.** The `classify()` method reads only the codebook. Removing all matrix cells produces identical results.
- **`atomic_fold.py` is a modern Hopfield network.** The softmax attention, logsumexp energy, and attractor dynamics are mathematically identical to Ramsauer et al. 2020. This was identified by external review. Any results characterize Hopfield behavior, not a new system.
- **Zero forgetting is trivial in attractive-only mode.** Append-only storage preserves old prototypes by construction.
- **Readout is the bottleneck.** 1-NN over prototypes is a weak classifier. The codebook stores well but reads poorly.

### Matrix codebook (`eigenfold.py`) — current direction
- **Early stage.** 100% accuracy on toy data (3 well-separated Gaussian clusters), 22% on P-MNIST 2-task (vs 10% random). The mechanism works but accuracy is low.
- **Threshold calibration is fragile.** Spawn threshold must be tuned per dataset — too low spawns every sample, too high spawns nothing.
- **Update dynamics need work.** Eigenform recovery may wash out learned updates. The balance between perturbation absorption (learning) and eigenform recovery (stability) is unresolved.
- **Frozen feature dependence.** All benchmarks use frozen feature extractors.
- **Merge not implemented.** Self-compression is theorized but not built or tested.

## Open Questions

1. **Is EigenFold's perturbation-stability classification a known mechanism?** Classification by "which element is least perturbed by cross-application" — does this exist in the literature? Closest candidates: ART resonance (but ART uses input-template matching, not self-referential eigenform recovery).
2. **Does noncommutative matrix interaction genuinely escape Hopfield?** Vector-based systems with symmetric pairwise interactions reduce to Hopfield. Matrix cross-application (M_i·M_j ≠ M_j·M_i) breaks this symmetry. Is the resulting system in a fundamentally different complexity class, or does it reduce to something known?
3. **What is the basin structure of Φ(M) = tanh(αM + βM²/k)?** How many distinct eigenforms exist for k=4? How well-separated are the basins? The capacity of the system depends on this landscape geometry.
4. **Can the eigenform landscape provide structural zero-forgetting guarantees?** If perturbations within a basin don't cross basin boundaries, old eigenforms are preserved. Under what conditions does this hold?
5. **What is the right balance between perturbation absorption (learning) and eigenform recovery (stability)?** Current results show the recovery may wash out learned updates, limiting accuracy.

## How to Contribute

The most valuable contribution is honest analysis. Run the code, read the math, and tell us what's wrong.

- **Identify prior art.** If this is a known technique under a different name, say so with a citation.
- **Break the benchmarks.** If the reported numbers don't reproduce, or if a simpler baseline matches them, that's important.
- **Answer the open questions.** Especially #1 (novelty) and #4 (kappa vs online EM).
- **Propose experiments.** What test would distinguish this from existing prototype-based methods?

Open an issue or submit a PR. Blunt feedback is preferred over polite encouragement.

## Status

Active research with significant unresolved problems. The architecture is evolving. Accuracy lags gradient-based continual learning methods.
