#!/usr/bin/env python3
"""
Diagnostic: FluxCore ManyToFewKernel vs HoeffdingTree on Elec2 (45312 samples).
Uses numpy for fast codebook cosine lookups; kernel internals unchanged.
"""
import sys, math, time
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_manytofew import ManyToFewKernel, _vec_cosine
from river import datasets
from river.tree import HoeffdingTreeClassifier

FEATURE_KEYS = ['date','day','period','nswprice','nswdemand','vicprice','vicdemand','transfer']
D = 64
D_PER = D // len(FEATURE_KEYS)  # 8

def embed(x):
    feats = [float(x[k]) for k in FEATURE_KEYS]
    projected = []
    for v in feats:
        for i in range(1, D_PER + 1):
            projected.append(math.sin(v * math.pi * i))
    n = math.sqrt(sum(v*v for v in projected) + 1e-15)
    return [v/n for v in projected]

# Pre-load all data once
print("Loading Elec2 dataset...", flush=True)
dataset = list(datasets.Elec2())
print(f"Total samples: {len(dataset)}", flush=True)

# Pre-compute all embeddings
print("Pre-computing embeddings...", flush=True)
embeddings = [embed(x) for x, y in dataset]
labels_all = [int(y) for x, y in dataset]
emb_arr = np.array(embeddings, dtype=np.float32)  # (N, D) — all normalized already
print("Done.", flush=True)

results = []

for spawn_thresh in [0.9, 0.95]:
    print(f"\nRunning spawn_thresh={spawn_thresh}...", flush=True)
    t0 = time.time()

    kernel = ManyToFewKernel(
        n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
        tau=0.3, k_couple=5, merge_thresh=0.99,
        lr_codebook=0.015, spawn_thresh=spawn_thresh
    )

    # Codebook as numpy matrix — rebuilt incrementally
    # cb_vecs: (M, D) float32, kept in sync with kernel.codebook
    cb_vecs = None         # numpy array, None until first spawn
    cb_labels = []         # [[count_0, count_1], ...] per codebook vector

    correct = 0
    total = 0
    prev_report = 0

    for idx in range(len(dataset)):
        r = embeddings[idx]
        r_np = emb_arr[idx]   # shape (D,)
        label = labels_all[idx]

        prev_cb_size = len(kernel.codebook)

        # --- Prediction (before update) ---
        pred = None
        if cb_vecs is not None and len(kernel.codebook) >= 1:
            # Fast cosine via numpy dot (vectors already normalized)
            sims = cb_vecs.dot(r_np)  # shape (M,)
            k_nn = min(3, len(kernel.codebook))
            top_k = np.argpartition(sims, -k_nn)[-k_nn:]
            votes = [0, 0]
            for i in top_k:
                lbl_counts = cb_labels[i]
                dominant = 0 if lbl_counts[0] >= lbl_counts[1] else 1
                votes[dominant] += 1
            pred = 0 if votes[0] >= votes[1] else 1

        # --- Kernel step ---
        kernel.step(r)

        # --- Sync cb_vecs and cb_labels ---
        new_size = len(kernel.codebook)
        if new_size != prev_cb_size:
            # Codebook changed (grew or merged) — rebuild numpy array
            cb_vecs = np.array(kernel.codebook, dtype=np.float32)
            while len(cb_labels) < new_size:
                cb_labels.append([0, 0])
            # Trim if merged
            while len(cb_labels) > new_size:
                cb_labels.pop()
        elif new_size > 0 and cb_vecs is None:
            cb_vecs = np.array(kernel.codebook, dtype=np.float32)

        # Update winner label — use fast numpy cosine
        if cb_vecs is not None and new_size > 0:
            sims = cb_vecs.dot(r_np)
            winner = int(np.argmax(sims))
            cb_labels[winner][label] += 1

            # Also update the changed codebook vector in cb_vecs
            # (kernel updated the winner's vector in-place)
            cb_vecs[winner] = kernel.codebook[winner]

        # Score
        if pred is not None:
            if pred == label:
                correct += 1
            total += 1

        if idx > 0 and idx % 5000 == 0:
            print(f"  {idx}/{len(dataset)}  cb_size={new_size}  acc={correct/total:.4f}", flush=True)

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0
    cb_size = len(kernel.codebook)
    spawned = kernel.total_spawned
    merged = kernel.total_merged

    print(f"  Done in {elapsed:.1f}s  final cb_size={cb_size}  spawned={spawned}  merged={merged}", flush=True)
    results.append((spawn_thresh, accuracy, cb_size, spawned, merged))

# --- Hoeffding Tree baseline ---
print("\nRunning Hoeffding Tree...", flush=True)
t0 = time.time()
ht = HoeffdingTreeClassifier()
correct = 0
total = 0
for x, y in dataset:
    pred = ht.predict_one(x)
    ht.learn_one(x, y)
    if pred is not None:
        correct += 1 if pred == y else 0
        total += 1

ht_acc = correct / total if total > 0 else 0.0
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

print("\n--- RESULTS ---")
print(f"{'spawn_thresh':>12}  {'accuracy':>9}  {'cb_size':>8}  {'spawned':>8}  {'merged':>8}")
for (st, acc, cbs, sp, mg) in results:
    print(f"{st:>12.2f}  {acc:>9.4f}  {cbs:>8}  {sp:>8}  {mg:>8}")
print(f"\nHoeffding Tree accuracy: {ht_acc:.4f}")
