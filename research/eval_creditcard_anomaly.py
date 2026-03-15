"""
FluxCore Unsupervised Anomaly Detection — River CreditCard Dataset

Evaluates AUROC for spawn_thresh=0.5 and spawn_thresh=0.9 using two embeddings:
1. Raw unit-sphere normalization (baseline)
2. Running z-score standardization then unit-sphere (better preserves structure)

FluxCore codebook-only (strips matrix dynamics — they dominate runtime ~1.6ms/sample
but add nothing to the novelty score; codebook logic preserved verbatim).
"""

import sys
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_manytofew import _vec_cosine
from river import datasets


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize(v):
    n = math.sqrt(sum(x * x for x in v) + 1e-15)
    return [x / n for x in v]


def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def _vec_scale(v, s):
    return [x * s for x in v]


# ── embeddings ────────────────────────────────────────────────────────────────

def embed_raw(x):
    """Raw unit-sphere normalization."""
    feats = list(x.values())
    n = math.sqrt(sum(v * v for v in feats) + 1e-15)
    return [v / n for v in feats]


class RunningStandardizer:
    """
    Online Welford mean/variance per feature.
    Standardizes each feature to z-score, then unit-sphere.
    """

    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = [0.0] * d
        self.M2 = [0.0] * d

    def update_and_embed(self, feats):
        self.n += 1
        new_mean = list(self.mean)
        new_M2 = list(self.M2)
        for i, x in enumerate(feats):
            delta = x - self.mean[i]
            new_mean[i] = self.mean[i] + delta / self.n
            delta2 = x - new_mean[i]
            new_M2[i] = self.M2[i] + delta * delta2
        self.mean = new_mean
        self.M2 = new_M2

        # z-score (use updated stats for current sample)
        z = []
        for i, x in enumerate(feats):
            if self.n < 2:
                z.append(0.0)
            else:
                std = math.sqrt(self.M2[i] / (self.n - 1) + 1e-15)
                z.append((x - self.mean[i]) / std)

        return _normalize(z)


# ── FluxCore codebook-only kernel ─────────────────────────────────────────────

class FluxCodebook:
    """Minimal FluxCore codebook: spawn/merge/update — no matrix dynamics."""

    def __init__(self, spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015):
        self.spawn_thresh = spawn_thresh
        self.merge_thresh = merge_thresh
        self.lr_codebook = lr_codebook
        self.codebook = []
        self.total_spawned = 0
        self.total_merged = 0

    def _try_merge_new(self, new_idx):
        if new_idx == 0:
            return new_idx
        new_v = self.codebook[new_idx]
        best_i, best_abs = -1, 0.0
        for i in range(new_idx):
            a = abs(_vec_cosine(new_v, self.codebook[i]))
            if a > best_abs:
                best_abs, best_i = a, i
        if best_abs > self.merge_thresh:
            self.codebook[best_i] = _normalize(_vec_add(self.codebook[best_i], new_v))
            del self.codebook[new_idx]
            self.total_merged += 1
            return best_i
        return new_idx

    def novelty(self, vec):
        """1 - max cosine similarity to codebook. 1.0 if empty."""
        if not self.codebook:
            return 1.0
        max_sim = max(_vec_cosine(v, vec) for v in self.codebook)
        return 1.0 - max_sim

    def update(self, vec):
        cb_winner = None
        max_sim = -1.0
        if self.codebook:
            sims = [_vec_cosine(v, vec) for v in self.codebook]
            cb_winner = max(range(len(self.codebook)), key=lambda i: sims[i])
            max_sim = sims[cb_winner]

        if cb_winner is None or max_sim < self.spawn_thresh:
            new_v = _normalize(vec)
            self.codebook.append(new_v)
            self.total_spawned += 1
            cb_winner = len(self.codebook) - 1
            cb_winner = self._try_merge_new(cb_winner)

        self.codebook[cb_winner] = _normalize(
            _vec_add(self.codebook[cb_winner], _vec_scale(vec, self.lr_codebook))
        )


# ── AUROC ─────────────────────────────────────────────────────────────────────

def compute_auroc(scores_labels):
    """Trapezoidal AUROC. Higher score = more anomalous. label=1 is fraud."""
    sorted_sl = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    total_pos = sum(1 for _, y in sorted_sl if y == 1)
    total_neg = len(sorted_sl) - total_pos

    if total_pos == 0 or total_neg == 0:
        return float('nan')

    tp = 0
    fp = 0
    auc = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for _, label in sorted_sl:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr = tpr
        prev_fpr = fpr

    return auc


# ── eval ──────────────────────────────────────────────────────────────────────

def run_eval(spawn_thresh, embedding_mode='zscore'):
    label = f"spawn_thresh={spawn_thresh}  embedding={embedding_mode}"
    print(f"\n{'='*65}")
    print(f"FluxCore Codebook — {label}")
    print(f"{'='*65}")

    kernel = FluxCodebook(spawn_thresh=spawn_thresh)

    d = 30  # CreditCard: Time + V1-V28 + Amount = 30 features
    standardizer = RunningStandardizer(d) if embedding_mode == 'zcore' else None

    scores_labels = []
    total = 0
    fraud_count = 0
    t0 = time.time()

    for x, y in datasets.CreditCard():
        feats = list(x.values())
        is_fraud = int(y)

        if embedding_mode == 'zcore':
            vec = standardizer.update_and_embed(feats)
        else:
            # raw unit-sphere
            n = math.sqrt(sum(v * v for v in feats) + 1e-15)
            vec = [v / n for v in feats]

        score = kernel.novelty(vec)
        scores_labels.append((score, is_fraud))
        kernel.update(vec)

        total += 1
        fraud_count += is_fraud

        if total % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} / ~284,807  {elapsed:.1f}s  codebook={len(kernel.codebook)}")

    elapsed = time.time() - t0
    fraud_rate = fraud_count / total if total > 0 else 0.0
    auroc = compute_auroc(scores_labels)

    print(f"\nResults:")
    print(f"  Total samples    : {total:,}")
    print(f"  Fraud count      : {fraud_count:,}  ({fraud_rate*100:.3f}%)")
    print(f"  Final codebook   : {len(kernel.codebook)} vectors")
    print(f"  Total spawned    : {kernel.total_spawned}")
    print(f"  Total merged     : {kernel.total_merged}")
    print(f"  AUROC            : {auroc:.4f}")
    print(f"  Elapsed          : {elapsed:.1f}s")

    return auroc, len(kernel.codebook), total, fraud_rate


if __name__ == "__main__":
    results = {}

    # Primary: z-score standardization (preserves feature scale structure)
    for thresh in [0.5, 0.9]:
        auroc, cb_size, total, fraud_rate = run_eval(thresh, embedding_mode='zcore')
        results[(thresh, 'zcore')] = {
            "auroc": auroc,
            "codebook_size": cb_size,
            "total": total,
            "fraud_rate": fraud_rate,
        }

    # Baseline: raw unit-sphere (as specified in brief)
    for thresh in [0.5, 0.9]:
        auroc, cb_size, total, fraud_rate = run_eval(thresh, embedding_mode='raw')
        results[(thresh, 'raw')] = {
            "auroc": auroc,
            "codebook_size": cb_size,
            "total": total,
            "fraud_rate": fraud_rate,
        }

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'embedding':<10} {'thresh':<10} {'AUROC':<10} {'Codebook':<12} {'Fraud%'}")
    print(f"{'-'*65}")
    for (thresh, emb), r in results.items():
        print(
            f"{emb:<10} {thresh:<10} {r['auroc']:.4f}     {r['codebook_size']:<12} "
            f"{r['fraud_rate']*100:.3f}%"
        )
